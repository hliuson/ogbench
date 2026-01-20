import inspect
import json
import os
import random
import time
from collections import defaultdict

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from agents import agents
from ml_collections import config_flags
from utils.datasets import Dataset, GCDataset, HGCDataset
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate
from utils.flax_utils import inject_encoder_params, load_encoder_params, print_param_tree, restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')
flags.DEFINE_string('dataset_path', None, 'Optional dataset path override.')
flags.DEFINE_string('encoder_restore_path', None, 'Restore path for encoder params.')
flags.DEFINE_integer('encoder_restore_step', None, 'Restore step for encoder params.')
flags.DEFINE_string('encoder_restore_prefix', 'encoder', 'Prefix for encoder params files.')
flags.DEFINE_string('encoder_restore_match_key', None, 'Optional key for matching encoder params subtree.')

flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')
flags.DEFINE_bool('preprocess_frame_stack', True, 'Whether to precompute frame stacks in datasets.')

flags.DEFINE_integer('eval_tasks', None, 'Number of tasks to evaluate (None for all).')
flags.DEFINE_integer('eval_episodes', 20, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_integer('eval_on_cpu', 1, 'Whether to evaluate on CPU.')
flags.DEFINE_bool('initial_eval', False, 'Whether to run evaluation before training.')

config_flags.DEFINE_config_file('agent', 'agents/gciql.py', lock_config=False)


def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project='OGBench', group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Set up environment and dataset.
    config = FLAGS.agent
    env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name,
        frame_stack=config['frame_stack'],
        dataset_path=FLAGS.dataset_path,
    )

    dataset_class = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
    }[config['dataset_class']]
    if val_dataset is not None:
        val_dataset = dataset_class(
            Dataset.create(**val_dataset),
            config,
            preprocess_frame_stack=FLAGS.preprocess_frame_stack,
        )
    train_dataset = dataset_class(
        Dataset.create(**train_dataset),
        config,
        preprocess_frame_stack=FLAGS.preprocess_frame_stack,
    )

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = train_dataset.sample(1)
    if config['discrete']:
        # Fill with the maximum action to let the agent know the action space size.
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

    agent_class = agents[config['agent_name']]
    # Some agents (sharsa) take example_batch directly to support oracle rep mode.
    sig = inspect.signature(agent_class.create)
    if 'example_batch' in sig.parameters:
        agent = agent_class.create(
            FLAGS.seed,
            example_batch,
            config,
        )
    else:
        agent = agent_class.create(
            FLAGS.seed,
            example_batch['observations'],
            example_batch['actions'],
            config,
        )

    # Log network architecture.
    print('\n' + '=' * 60)
    print('Network Architecture (parameter tree):')
    print('=' * 60)
    print_param_tree(agent.network.params, max_depth=4)
    print('=' * 60 + '\n')

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
    if FLAGS.encoder_restore_path is not None:
        if FLAGS.encoder_restore_step is None:
            raise ValueError('encoder_restore_step must be set when encoder_restore_path is provided.')
        encoder_params = load_encoder_params(
            FLAGS.encoder_restore_path,
            FLAGS.encoder_restore_step,
            prefix=FLAGS.encoder_restore_prefix,
        )
        new_params, num_replaced, injected_paths = inject_encoder_params(
            agent.network.params,
            encoder_params,
            match_key=FLAGS.encoder_restore_match_key,
        )
        agent = agent.replace(network=agent.network.replace(params=new_params))
        print('\n' + '=' * 60)
        print(f'Encoder Injection Summary:')
        print(f'  Match key: {FLAGS.encoder_restore_match_key}')
        print(f'  Injected into {num_replaced} subtree(s):')
        for path in injected_paths:
            print(f'    - {".".join(path)}')
        print('=' * 60 + '\n')

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # Update agent.
        batch = train_dataset.sample(config['batch_size'])
        # Pass step to agents that support it (e.g., for warmup schedules)
        if 'high_actor_warmup_steps' in config:
            agent, update_info = agent.update(batch, step=i)
        else:
            agent, update_info = agent.update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                if 'high_actor_warmup_steps' in config:
                    _, val_info = agent.total_loss(val_batch, grad_params=None, step=i)
                else:
                    _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if (FLAGS.initial_eval and i == 1) or i % FLAGS.eval_interval == 0:
            if FLAGS.eval_on_cpu:
                eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
            else:
                eval_agent = agent
            renders = []
            eval_metrics = {}
            overall_metrics = defaultdict(list)
            task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
            num_tasks = FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)
            for task_id in tqdm.trange(1, num_tasks + 1):
                task_name = task_infos[task_id - 1]['task_name']
                eval_info, trajs, cur_renders = evaluate(
                    agent=eval_agent,
                    env=env,
                    task_id=task_id,
                    config=config,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                    eval_temperature=FLAGS.eval_temperature,
                    eval_gaussian=FLAGS.eval_gaussian,
                )
                renders.extend(cur_renders)
                metric_names = ['success']
                eval_metrics.update(
                    {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                )
                for k, v in eval_info.items():
                    if k in metric_names:
                        overall_metrics[k].append(v)
            for k, v in overall_metrics.items():
                eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders, n_cols=num_tasks)
                eval_metrics['video'] = video

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
