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

from utils.pretraining import VAEModel
from utils.datasets import ATCDataset
from utils.encoders import encoder_modules
from utils.env_utils import make_env_and_datasets
from utils.flax_utils import TrainState, restore_pretrain_state, save_encoder_params, save_pretrain_state
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'VAE', 'Run group.')
flags.DEFINE_string('wandb_project', 'OGBench-VAE', 'Weights & Biases project.')
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
flags.DEFINE_integer('latent_dim', 512, 'VAE latent dimension.')
flags.DEFINE_float('lr', 3e-4, 'Learning rate.')
flags.DEFINE_float('kl_weight', 1.0, 'Weight for KL divergence term (beta in beta-VAE).')
flags.DEFINE_integer('frame_stack', 3, 'Number of frames to stack.')
flags.DEFINE_float('p_aug', 1.0, 'Probability of applying image augmentation.')
flags.DEFINE_integer('augment_padding', 4, 'Padding size for random shift.')
flags.DEFINE_bool('preprocess_frame_stack', True, 'Whether to precompute frame stacks in datasets.')


def vae_loss(model_def, params, batch, rng, train=True, kl_weight=1.0):
    """Compute VAE loss: reconstruction + KL divergence."""
    observations = batch['observations']
    target = observations.astype(jnp.float32) / 255.0  # Normalize to [0, 1]

    reconstruction, mu, log_var = model_def.apply(
        {'params': params},
        observations,
        train=train,
        rngs={'reparameterize': rng},
    )

    # Reconstruction loss (MSE)
    reconstruction = jax.nn.sigmoid(reconstruction)  # Output in [0, 1]
    recon_loss = jnp.mean((reconstruction - target) ** 2)

    # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    kl_loss = -0.5 * jnp.mean(1 + log_var - mu ** 2 - jnp.exp(log_var))

    loss = recon_loss + kl_weight * kl_loss

    info = {
        'loss': loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss,
        'mu/mean': jnp.mean(mu),
        'mu/std': jnp.std(mu),
        'log_var/mean': jnp.mean(log_var),
    }

    return loss, info


def make_update_fn(model_def, kl_weight):
    @jax.jit
    def update(state, batch, rng):
        def loss_fn(params):
            return vae_loss(model_def, params, batch, rng, train=True, kl_weight=kl_weight)

        new_state, info = state.apply_loss_fn(loss_fn)
        return new_state, info

    return update


def eval_metrics(model_def, params, batch, rng, kl_weight):
    _, info = vae_loss(model_def, params, batch, rng, train=False, kl_weight=kl_weight)
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

    # Determine output channels from frame stack
    example_obs = train_dataset.get_observations(np.array([0]))
    output_channels = example_obs.shape[-1]  # frame_stack * 3 (RGB)

    model_def = VAEModel(
        encoder=encoder_def,
        latent_dim=FLAGS.latent_dim,
        output_channels=output_channels,
    )

    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, init_rng, reparam_rng = jax.random.split(rng, 3)

    params = model_def.init(
        {'params': init_rng, 'reparameterize': reparam_rng},
        example_obs,
        train=True,
    )['params']
    state = TrainState.create(model_def, params, tx=optax.adam(learning_rate=FLAGS.lr))

    if FLAGS.restore_path is not None:
        if FLAGS.restore_step is None:
            raise ValueError('restore_step must be set when restore_path is provided.')
        state, _ = restore_pretrain_state(
            state,
            state.params,  # VAE doesn't use target params
            FLAGS.restore_path,
            FLAGS.restore_step,
            prefix='vae',
        )

    update_fn = make_update_fn(model_def, FLAGS.kl_weight)

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    for step in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        rng, batch_rng, update_rng = jax.random.split(rng, 3)

        # Sample batch (use k=1 since VAE doesn't need positive pairs)
        batch = train_dataset.sample(FLAGS.batch_size, k=1, evaluation=False)
        batch = jax.tree_util.tree_map(jnp.asarray, batch)

        state, info = update_fn(state, batch, update_rng)

        if step % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in info.items()}
            if val_dataset is not None:
                rng, val_rng = jax.random.split(rng)
                val_batch = val_dataset.sample(FLAGS.batch_size, k=1, evaluation=True)
                val_batch = jax.tree_util.tree_map(jnp.asarray, val_batch)
                val_info = eval_metrics(model_def, state.params, val_batch, val_rng, FLAGS.kl_weight)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=step)
            train_logger.log(train_metrics, step=step)

        if step % FLAGS.save_interval == 0:
            save_pretrain_state(state, state.params, FLAGS.save_dir, step, prefix='vae')
            save_encoder_params(state.params['encoder'], FLAGS.save_dir, step, prefix='encoder')

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
