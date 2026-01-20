import json
import os
import random
import time

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import wandb
from absl import app, flags

from utils.pretraining import ATCModel
from utils.datasets import ATCDataset
from utils.encoders import encoder_modules
from utils.env_utils import make_env_and_datasets
from utils.flax_utils import TrainState, restore_pretrain_state, save_encoder_params, save_pretrain_state
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'ATC', 'Run group.')
flags.DEFINE_string('wandb_project', 'OGBench-ATC', 'Weights & Biases project.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'visual-antmaze-large-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_step', None, 'Restore step.')
flags.DEFINE_string('dataset_path', None, 'Optional dataset path override.')

flags.DEFINE_integer('train_steps', 200000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('save_interval', 50000, 'Saving interval.')

flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_string('encoder', 'impala_small', 'Encoder module name.')
flags.DEFINE_multi_integer('predictor_hidden_dims', (512, 512), 'Predictor MLP hidden dimensions.')
flags.DEFINE_float('lr', 3e-4, 'Learning rate.')
flags.DEFINE_float('momentum', 0.99, 'EMA momentum for the target encoder.')
flags.DEFINE_integer('k', 16, 'Temporal offset for positive pairs.')
flags.DEFINE_integer('k_min', None, 'Minimum temporal offset for random k sampling.')
flags.DEFINE_integer('k_max', None, 'Maximum temporal offset for random k sampling.')
flags.DEFINE_integer('k_eval', None, 'Temporal offset for validation.')
flags.DEFINE_integer('frame_stack', 3, 'Number of frames to stack.')
flags.DEFINE_float('p_aug', 1.0, 'Probability of applying image augmentation.')
flags.DEFINE_integer('augment_padding', 4, 'Padding size for random shift.')
flags.DEFINE_bool('preprocess_frame_stack', True, 'Whether to precompute frame stacks in datasets.')
flags.DEFINE_bool('concat_obs', False, 'Whether to concatenate observations with themselves for encoder input.')


def atc_loss(model_def, params, target_params, batch, train=True, vq_loss_weight=1.0):
    codes, preds, bilinear, vq_info = model_def.apply({'params': params}, batch['observations'], train=train)
    target_codes = model_def.apply(
        {'params': target_params},
        batch['positive_observations'],
        train=train,
        method=ATCModel.encode,
    )
    logits = jnp.einsum('ik,kl,jl->ij', preds, bilinear, target_codes)
    labels = jnp.arange(logits.shape[0])
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

    pos = jnp.mean(jnp.diag(logits))
    neg = (jnp.sum(logits) - jnp.sum(jnp.diag(logits))) / (logits.size - logits.shape[0])
    accuracy = jnp.mean(jnp.argmax(logits, axis=1) == labels)

    info = {
        'loss': loss,
        'accuracy': accuracy,
        'logits/mean': logits.mean(),
        'logits/pos': pos,
        'logits/neg': neg,
        'code/norm': jnp.linalg.norm(codes, axis=-1).mean(),
        'pred/norm': jnp.linalg.norm(preds, axis=-1).mean(),
    }

    return loss, info


def apply_vq_updates(params, vq_updates):
    if vq_updates is None:
        return params
    is_frozen = isinstance(params, flax.core.FrozenDict)
    params_dict = flax.core.unfreeze(params) if is_frozen else params
    if 'encoder' not in params_dict or 'quantizer' not in params_dict['encoder']:
        raise ValueError('VQ updates require encoder.quantizer params.')
    params_dict['encoder']['quantizer']['codebook'] = vq_updates['codebook']
    params_dict['encoder']['quantizer']['ema_counts'] = vq_updates['ema_counts']
    params_dict['encoder']['quantizer']['ema_embeddings'] = vq_updates['ema_embeddings']
    return flax.core.freeze(params_dict) if is_frozen else params_dict


def make_update_fn(model_def, momentum, vq_loss_weight, vq_enabled):
    @jax.jit
    def update(state, target_params, batch):
        def loss_fn(params):
            return atc_loss(
                model_def,
                params,
                target_params,
                batch,
                train=True,
                vq_loss_weight=vq_loss_weight,
            )

        new_state, info = state.apply_loss_fn(loss_fn)
        if vq_enabled:
            vq_updates = info.get('_vq_updates')
            new_params = apply_vq_updates(new_state.params, vq_updates)
            new_state = new_state.replace(params=new_params)
        new_target_params = jax.tree_util.tree_map(
            lambda tp, p: tp * momentum + p * (1.0 - momentum),
            target_params,
            new_state.params,
        )
        return new_state, new_target_params, info

    return update


def eval_metrics(model_def, params, target_params, batch, vq_loss_weight):
    _, info = atc_loss(model_def, params, target_params, batch, train=False, vq_loss_weight=vq_loss_weight)
    return info


def main(_):
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project=FLAGS.wandb_project, group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Load datasets.
    _, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name,
        frame_stack=None,
        dataset_path=FLAGS.dataset_path,
    )
    dataset_config = dict(
        frame_stack=FLAGS.frame_stack,
        p_aug=FLAGS.p_aug,
        augment_padding=FLAGS.augment_padding,
    )
    train_dataset = ATCDataset(
        train_dataset,
        dataset_config,
        preprocess_frame_stack=FLAGS.preprocess_frame_stack,
    )
    if val_dataset is not None:
        val_dataset = ATCDataset(
            val_dataset,
            dataset_config,
            preprocess_frame_stack=FLAGS.preprocess_frame_stack,
        )

    # Initialize model.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    encoder_module = encoder_modules[FLAGS.encoder]
    encoder_def = encoder_module()
    model_def = ATCModel(encoder=encoder_def, predictor_hidden_dims=tuple(FLAGS.predictor_hidden_dims))

    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, init_rng = jax.random.split(rng)

    example_obs = train_dataset.get_observations(np.array([0]))
    if FLAGS.concat_obs:
        example_obs = np.concatenate([example_obs, example_obs], axis=-1)
    params = model_def.init(init_rng, example_obs, train=True)['params']
    state = TrainState.create(model_def, params, tx=optax.adam(learning_rate=FLAGS.lr))
    target_params = flax.core.unfreeze(params)

    if FLAGS.restore_path is not None:
        if FLAGS.restore_step is None:
            raise ValueError('restore_step must be set when restore_path is provided.')
        state, target_params = restore_pretrain_state(
            state,
            target_params,
            FLAGS.restore_path,
            FLAGS.restore_step,
            prefix='atc',
        )

    update_fn = make_update_fn(model_def, FLAGS.momentum, FLAGS.vq_loss_weight, FLAGS.vq_enabled)

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    def sample_k():
        if FLAGS.k_min is not None and FLAGS.k_max is not None:
            return np.random.randint(FLAGS.k_min, FLAGS.k_max + 1)
        return FLAGS.k

    def eval_k():
        if FLAGS.k_eval is not None:
            return FLAGS.k_eval
        if FLAGS.k_min is not None and FLAGS.k_max is not None:
            return FLAGS.k_min
        return FLAGS.k

    for step in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        k = sample_k()
        batch = train_dataset.sample(FLAGS.batch_size, k=k, evaluation=False)
        batch = jax.tree_util.tree_map(jnp.asarray, batch)
        if FLAGS.concat_obs:
            batch['observations'] = jnp.concatenate([batch['observations'], batch['observations']], axis=-1)
            batch['positive_observations'] = jnp.concatenate(
                [batch['positive_observations'], batch['positive_observations']], axis=-1
            )
        state, target_params, info = update_fn(state, target_params, batch)
        if '_vq_updates' in info:
            info = info.copy()
            info.pop('_vq_updates')

        if step % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in info.items()}
            train_metrics['training/k'] = k
            if val_dataset is not None:
                val_k = eval_k()
                val_batch = val_dataset.sample(FLAGS.batch_size, k=val_k, evaluation=True)
                val_batch = jax.tree_util.tree_map(jnp.asarray, val_batch)
                if FLAGS.concat_obs:
                    val_batch['observations'] = jnp.concatenate(
                        [val_batch['observations'], val_batch['observations']], axis=-1
                    )
                    val_batch['positive_observations'] = jnp.concatenate(
                        [val_batch['positive_observations'], val_batch['positive_observations']], axis=-1
                    )
                val_info = eval_metrics(model_def, state.params, target_params, val_batch, FLAGS.vq_loss_weight)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=step)
            train_logger.log(train_metrics, step=step)

        if step % FLAGS.save_interval == 0:
            save_pretrain_state(state, target_params, FLAGS.save_dir, step, prefix='atc')
            save_encoder_params(state.params['encoder'], FLAGS.save_dir, step, prefix='encoder')

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
