#!/usr/bin/env python3
"""Generate and optionally submit the 100M reference suite as SLURM jobs.

Examples:
    uv run python submit_100m_reference_suite_slurm.py
    uv run python submit_100m_reference_suite_slurm.py --submit
    uv run python submit_100m_reference_suite_slurm.py \
        --task puzzle \
        --method cf-trl \
        --seed 0 \
        --sweep agent.z_proposal_awr_beta=1.0,3.0 \
        --sweep agent.cf_expectile=0.7,0.9 \
        --tag z-sweep
"""

from __future__ import annotations

import argparse
import itertools
import os
import re
import shlex
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT_DIR = Path(__file__).resolve().parent
IMPLS_DIR = ROOT_DIR / 'impls'

DEFAULT_DATA_DIR = Path(os.environ.get('DATA_DIR', '/scratch/engin_root/engin1/hliuson/.ogbench/data'))
DEFAULT_SAVE_DIR = Path(os.environ.get('SAVE_DIR', '/scratch/engin_root/engin1/hliuson/ogbench_exp'))
DEFAULT_JOB_SCRIPT_DIR = Path(os.environ.get('JOB_SCRIPT_DIR', str(ROOT_DIR / 'generated_slurm_jobs')))
DEFAULT_LOG_DIR = Path(os.environ.get('LOG_DIR', str(ROOT_DIR / 'impls/slurm_logs')))
DEFAULT_UV_CACHE_DIR = Path(os.environ.get('UV_CACHE_DIR', '/scratch/engin_root/engin1/hliuson/.cache/uv'))
DEFAULT_PYTHON_RUNNER = os.environ.get('PYTHON_RUNNER', 'uv run python')

DEFAULT_MAIN_ARGS = (
    ('train_steps', '1000000'),
    ('log_interval', '5000'),
    ('eval_interval', '100000'),
    ('save_interval', '100000'),
    ('eval_episodes', '20'),
    ('eval_on_cpu', '0'),
    ('video_episodes', '0'),
)

RESERVED_OVERRIDE_KEYS = {
    'agent',
    'dataset_path',
    'dataset_replace_interval',
    'env_name',
    'run_group',
    'save_dir',
    'seed',
}


@dataclass(frozen=True)
class TaskSpec:
    key: str
    aliases: Tuple[str, ...]
    env_name: str
    dataset_name: str
    dataset_replace_interval: int
    job_suffix: str
    run_group_suffix: str


@dataclass(frozen=True)
class MethodSpec:
    key: str
    aliases: Tuple[str, ...]
    agent_path: str
    job_prefix: str
    wandb_name: str
    base_args: Tuple[Tuple[str, str], ...]


TASKS: Mapping[str, TaskSpec] = {
    'puzzle-4x6': TaskSpec(
        key='puzzle-4x6',
        aliases=('puzzle-4x6', 'p46'),
        env_name='puzzle-4x6-play-oraclerep-v0',
        dataset_name='puzzle-4x6-play-100m-v0',
        dataset_replace_interval=1000,
        job_suffix='p46-100m',
        run_group_suffix='p46',
    ),
    'cube-triple': TaskSpec(
        key='cube-triple',
        aliases=('cube-triple', 'ct'),
        env_name='cube-triple-play-oraclerep-v0',
        dataset_name='cube-triple-play-100m-v0',
        dataset_replace_interval=1000,
        job_suffix='ct-100m',
        run_group_suffix='ct',
    ),
    'cube-quadruple': TaskSpec(
        key='cube-quadruple',
        aliases=('cube-quadruple', 'cq'),
        env_name='cube-quadruple-play-oraclerep-v0',
        dataset_name='cube-quadruple-play-100m-v0',
        dataset_replace_interval=1000,
        job_suffix='cq-100m',
        run_group_suffix='cq',
    ),
    'humanoidmaze-giant-navigate': TaskSpec(
        key='humanoidmaze-giant-navigate',
        aliases=('hm-giant-nav', 'hmg-nav'),
        env_name='humanoidmaze-giant-navigate-oraclerep-v0',
        dataset_name='humanoidmaze-giant-navigate-100m-v0',
        dataset_replace_interval=100000,
        job_suffix='hmg-nav-100m',
        run_group_suffix='hmg-nav',
    ),
}

METHODS: Mapping[str, MethodSpec] = {
    'trl': MethodSpec(
        key='trl',
        aliases=(),
        agent_path='agents/trl.py',
        job_prefix='trl',
        wandb_name='trl',
        base_args=(
            ('agent.batch_size', '1024'),
            ('agent.discount', '0.999'),
            ('agent.expectile', '0.7'),
            ('agent.value_hidden_dims', '(1024, 1024, 1024, 1024)'),
            ('agent.actor_hidden_dims', '(1024, 1024, 1024, 1024)'),
            ('agent.value_p_curgoal', '0.0'),
            ('agent.value_p_trajgoal', '1.0'),
            ('agent.value_p_randomgoal', '0.0'),
            ('agent.actor_p_trajgoal', '0.5'),
            ('agent.actor_p_randomgoal', '0.5'),
            ('agent.q_action_n_step', '25'),
        ),
    ),
    'trl_original': MethodSpec(
        key='trl_original',
        aliases=('trl-original', 'trlo', 'trlorig'),
        agent_path='agents/trl_original.py',
        job_prefix='trlo',
        wandb_name='trl-original',
        base_args=(
            ('agent.batch_size', '1024'),
            ('agent.discount', '0.999'),
            ('agent.expectile', '0.7'),
            ('agent.value_hidden_dims', '(1024, 1024, 1024, 1024)'),
            ('agent.actor_hidden_dims', '(1024, 1024, 1024, 1024)'),
            ('agent.value_p_curgoal', '0.0'),
            ('agent.value_p_trajgoal', '1.0'),
            ('agent.value_p_randomgoal', '0.0'),
            ('agent.actor_p_trajgoal', '0.5'),
            ('agent.actor_p_randomgoal', '0.5'),
        ),
    ),
    'mc': MethodSpec(
        key='mc',
        aliases=(),
        agent_path='agents/mc.py',
        job_prefix='mc',
        wandb_name='mc',
        base_args=(
            ('agent.batch_size', '1024'),
            ('agent.discount', '0.999'),
            ('agent.value_hidden_dims', '(1024, 1024, 1024, 1024)'),
            ('agent.actor_hidden_dims', '(1024, 1024, 1024, 1024)'),
            ('agent.value_p_curgoal', '0.2'),
            ('agent.value_p_trajgoal', '0.5'),
            ('agent.value_p_randomgoal', '0.3'),
            ('agent.actor_p_trajgoal', '0.5'),
            ('agent.actor_p_randomgoal', '0.5'),
            ('agent.q_action_n_step', '25'),
        ),
    ),
    'gciql': MethodSpec(
        key='gciql',
        aliases=(),
        agent_path='agents/gciql_nstep.py',
        job_prefix='gciql',
        wandb_name='gciql',
        base_args=(
            ('agent.batch_size', '1024'),
            ('agent.discount', '0.999'),
            ('agent.expectile', '0.9'),
            ('agent.cf_expectile', '0.9'),
            ('agent.critic_n_step', '50'),
            ('agent.value_hidden_dims', '(1024, 1024, 1024, 1024)'),
            ('agent.actor_hidden_dims', '(1024, 1024, 1024, 1024)'),
            ('agent.value_p_curgoal', '0.2'),
            ('agent.value_p_trajgoal', '0.5'),
            ('agent.value_p_randomgoal', '0.3'),
            ('agent.actor_p_trajgoal', '0.5'),
            ('agent.actor_p_randomgoal', '0.5'),
            ('agent.q_action_n_step', '25'),
        ),
    ),
    'cf_trl': MethodSpec(
        key='cf_trl',
        aliases=('cf', 'cf-trl', 'cftrl', 'latent', 'latent-trl', 'ltrl'),
        agent_path='agents/cf_trl.py',
        job_prefix='cftrl',
        wandb_name='cf-trl',
        base_args=(
            ('agent.batch_size', '1024'),
            ('agent.discount', '0.999'),
            ('agent.expectile', '0.7'),
            ('agent.value_hidden_dims', '(1024, 1024, 1024, 1024)'),
            ('agent.actor_hidden_dims', '(1024, 1024, 1024, 1024)'),
            ('agent.cf_num_z_proposals', '1'),
            ('agent.value_p_curgoal', '0.0'),
            ('agent.value_p_trajgoal', '0.8'),
            ('agent.value_p_randomgoal', '0.2'),
            ('agent.actor_p_trajgoal', '0.5'),
            ('agent.actor_p_randomgoal', '0.5'),
            ('agent.cf_expectile', '0.7'),
            ('agent.q_action_n_step', '25'),
            ('agent.z_proposal_awr_beta', '10.0'),
            ('agent.z_proposal_awr_max_weight', '20.0'),
            ('agent.z_proposal_lr', '3e-3'),
        ),
    ),
}


def parse_bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'n', 'off'}:
        return False
    raise ValueError(f'Invalid boolean value for {name}: {raw!r}')


def default_seeds() -> List[int]:
    raw = os.environ.get('SEEDS')
    if raw is None:
        return [0, 1, 2]
    seeds = [part for part in re.split(r'[,\s]+', raw.strip()) if part]
    return [int(seed) for seed in seeds]


def normalize_key(name: str) -> str:
    return name.strip().lower()


def build_alias_map(specs: Mapping[str, object]) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    for key, spec in specs.items():
        alias_map[normalize_key(key)] = key
        for alias in spec.aliases:
            alias_map[normalize_key(alias)] = key
    return alias_map


TASK_ALIASES = build_alias_map(TASKS)
METHOD_ALIASES = build_alias_map(METHODS)


def resolve_names(requested: Sequence[str] | None, alias_map: Mapping[str, str], label: str) -> List[str]:
    if not requested:
        return list(alias_map[key] for key in alias_map if key == alias_map[key])

    resolved: List[str] = []
    seen = set()
    for item in requested:
        canonical = alias_map.get(normalize_key(item))
        if canonical is None:
            raise ValueError(f'Unknown {label}: {item}')
        if canonical not in seen:
            resolved.append(canonical)
            seen.add(canonical)
    return resolved


def parse_assignment(text: str) -> Tuple[str, str]:
    if '=' not in text:
        raise ValueError(f'Expected KEY=VALUE, got {text!r}')
    key, value = text.split('=', 1)
    key = key.strip()
    value = value.strip()
    if not key or not value:
        raise ValueError(f'Expected KEY=VALUE, got {text!r}')
    return key, value


def parse_overrides(items: Sequence[str]) -> OrderedDict[str, str]:
    overrides: OrderedDict[str, str] = OrderedDict()
    for item in items:
        key, value = parse_assignment(item)
        if key in RESERVED_OVERRIDE_KEYS:
            raise ValueError(f'Override key {key!r} is reserved; use a dedicated CLI flag instead.')
        overrides[key] = value
    return overrides


def split_sweep_values(raw_value: str) -> List[str]:
    if any(char in raw_value for char in '()[]{} '):
        return [raw_value]
    return [part.strip() for part in raw_value.split(',') if part.strip()]


def parse_sweeps(items: Sequence[str]) -> List[Tuple[str, List[str]]]:
    grouped: OrderedDict[str, List[str]] = OrderedDict()
    for item in items:
        key, raw_value = parse_assignment(item)
        if key in RESERVED_OVERRIDE_KEYS:
            raise ValueError(f'Sweep key {key!r} is reserved; use a dedicated CLI flag instead.')
        grouped.setdefault(key, [])
        grouped[key].extend(split_sweep_values(raw_value))

    axes: List[Tuple[str, List[str]]] = []
    for key, values in grouped.items():
        deduped = list(OrderedDict((value, None) for value in values).keys())
        axes.append((key, deduped))
    return axes


def iter_sweep_combinations(axes: Sequence[Tuple[str, Sequence[str]]]) -> Iterable[OrderedDict[str, str]]:
    if not axes:
        yield OrderedDict()
        return

    keys = [key for key, _ in axes]
    values_product = itertools.product(*(values for _, values in axes))
    for values in values_product:
        yield OrderedDict(zip(keys, values))


def leaf_name(key: str) -> str:
    return key.rsplit('.', 1)[-1]


def sanitize_filename_token(text: str) -> str:
    sanitized = re.sub(r'[^A-Za-z0-9]+', '-', text.strip()).strip('-').lower()
    if not sanitized:
        raise ValueError(f'Could not derive a filename token from {text!r}')
    return sanitized


def compact_value_token(value: str) -> str:
    token = value.strip()
    token = token.replace('.', 'p')
    token = token.replace('-', 'm')
    token = re.sub(r'[^A-Za-z0-9]+', '', token)
    return token.lower() or 'value'


def compact_name_token(name: str) -> str:
    token = re.sub(r'[^A-Za-z0-9]+', '', name)
    return token.lower() or 'param'


def sweep_name_tokens(assignments: Mapping[str, str]) -> List[str]:
    return [f"{compact_name_token(leaf_name(key))}{compact_value_token(value)}" for key, value in assignments.items()]


def format_command_lines(args: Sequence[str]) -> str:
    quoted = [shlex.quote(arg) for arg in args]
    lines = []
    for index, arg in enumerate(quoted):
        suffix = ' \\' if index < len(quoted) - 1 else ''
        prefix = '' if index == 0 else '  '
        lines.append(f'{prefix}{arg}{suffix}')
    return '\n'.join(lines)


def render_job_script(
    *,
    job_name: str,
    log_dir: Path,
    account: str,
    partition: str,
    gpus: int,
    time_limit: str,
    mem: str,
    python_runner: Sequence[str],
    dataset_path: Path,
    dataset_name: str,
    uv_cache_dir: Path,
    command_args: Sequence[str],
    wandb_name: str,
    wandb_tags: Sequence[str],
) -> str:
    base_url = f'https://rail.eecs.berkeley.edu/datasets/ogbench/{dataset_name}'
    command_block = format_command_lines([*python_runner, 'main.py', *command_args])
    return f"""#!/bin/bash
# Generated by submit_100m_reference_suite_slurm.py
#SBATCH --account={account}
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpus}
#SBATCH --time={time_limit}
#SBATCH --mem={mem}
#SBATCH --output={log_dir}/slurm-%j.out

set -euo pipefail
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WANDB_ENTITY=latent-trl
export WANDB_PROJECT=OGBench
export WANDB_NAME={shlex.quote(wandb_name)}
export WANDB_TAGS={shlex.quote(','.join(wandb_tags))}
module load uv
export UV_CACHE_DIR={shlex.quote(str(uv_cache_dir))}
mkdir -p "$UV_CACHE_DIR"

DATASET_PATH={shlex.quote(str(dataset_path))}
BASE_URL={shlex.quote(base_url)}
mkdir -p "$DATASET_PATH"
if ! ls "$DATASET_PATH"/*.npz >/dev/null 2>&1; then
  (
    cd "$DATASET_PATH"
    wget -nc -r -np -nH --cut-dirs=4 -A "*.npz" "$BASE_URL/"
  )
fi

cd {shlex.quote(str(IMPLS_DIR))}
{command_block}
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Generate and optionally submit the OGBench 100M reference suite as individual SLURM jobs.',
    )
    parser.add_argument('--task', action='append', help='Task key or alias. Repeat to select multiple tasks.')
    parser.add_argument('--method', action='append', help='Method key or alias. Repeat to select multiple methods.')
    parser.add_argument('--seed', type=int, action='append', help='Seed. Repeat to select multiple seeds.')
    parser.add_argument('--set', dest='overrides', action='append', default=[], metavar='KEY=VALUE')
    parser.add_argument(
        '--sweep',
        action='append',
        default=[],
        metavar='KEY=VALUE[,VALUE2...]',
        help='Repeatable sweep axis. Repeat the same KEY to add more values, especially for tuple-valued settings.',
    )
    parser.add_argument('--tag', action='append', default=[], help='Extra label appended to job names and WANDB names.')

    submit_group = parser.add_mutually_exclusive_group()
    submit_group.add_argument('--submit', dest='submit', action='store_true', help='Queue each generated script with sbatch.')
    submit_group.add_argument(
        '--write-only',
        dest='submit',
        action='store_false',
        help='Only write scripts without submitting them.',
    )
    parser.set_defaults(submit=parse_bool_env('SUBMIT', False))

    parser.add_argument('--list-tasks', action='store_true', help='List built-in tasks and exit.')
    parser.add_argument('--list-methods', action='store_true', help='List built-in methods and exit.')

    parser.add_argument('--account', default='spitis0')
    parser.add_argument('--partition', default='spgpu')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--time', dest='time_limit', default='24:00:00')
    parser.add_argument('--mem', default='64G')

    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--save-dir', type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument('--job-script-dir', type=Path, default=DEFAULT_JOB_SCRIPT_DIR)
    parser.add_argument('--log-dir', type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument('--uv-cache-dir', type=Path, default=DEFAULT_UV_CACHE_DIR)
    parser.add_argument('--python-runner', default=DEFAULT_PYTHON_RUNNER)
    return parser


def print_task_table() -> None:
    for key, task in TASKS.items():
        alias_text = ', '.join(task.aliases) if task.aliases else '-'
        print(f'{key:28} aliases={alias_text:10} env={task.env_name}')


def print_method_table() -> None:
    for key, method in METHODS.items():
        alias_text = ', '.join(method.aliases) if method.aliases else '-'
        print(f'{key:28} aliases={alias_text:10} agent={method.agent_path}')


def normalize_path(path: Path, *, base_dir: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_tasks:
        print_task_table()
        return 0
    if args.list_methods:
        print_method_table()
        return 0

    try:
        task_keys = resolve_names(args.task, TASK_ALIASES, 'task')
        method_keys = resolve_names(args.method, METHOD_ALIASES, 'method')
        seeds = args.seed if args.seed else default_seeds()
        overrides = parse_overrides(args.overrides)
        sweep_axes = parse_sweeps(args.sweep)
    except ValueError as exc:
        parser.error(str(exc))

    if not seeds:
        parser.error('At least one seed is required.')

    python_runner = shlex.split(args.python_runner)
    if not python_runner:
        parser.error('--python-runner must not be empty.')

    job_script_dir = normalize_path(args.job_script_dir, base_dir=ROOT_DIR)
    log_dir = normalize_path(args.log_dir, base_dir=ROOT_DIR)
    data_dir = normalize_path(args.data_dir, base_dir=ROOT_DIR)
    save_dir = normalize_path(args.save_dir, base_dir=ROOT_DIR)
    uv_cache_dir = normalize_path(args.uv_cache_dir, base_dir=ROOT_DIR)

    job_script_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    seen_job_names = set()

    for task_key in task_keys:
        task = TASKS[task_key]
        dataset_path = data_dir / task.dataset_name

        for method_key in method_keys:
            method = METHODS[method_key]
            run_group = f'{method.job_prefix}-{task.run_group_suffix}'

            for sweep_values in iter_sweep_combinations(sweep_axes):
                variant_tokens = [sanitize_filename_token(tag) for tag in args.tag]
                variant_tokens.extend(sweep_name_tokens(sweep_values))
                variant_suffix = f"-{'-'.join(variant_tokens)}" if variant_tokens else ''

                wandb_parts = [method.wandb_name]
                wandb_parts.extend(args.tag)
                if sweep_values:
                    wandb_parts.extend(f'{leaf_name(key)}={value}' for key, value in sweep_values.items())
                else:
                    wandb_parts.append('reference')
                wandb_name = ' '.join(wandb_parts)

                for seed in seeds:
                    job_name = f'{method.job_prefix}-{task.job_suffix}{variant_suffix}-s{seed}'
                    if job_name in seen_job_names:
                        raise ValueError(f'Duplicate job name generated: {job_name}')
                    seen_job_names.add(job_name)

                    command_args: OrderedDict[str, str] = OrderedDict()
                    command_args['env_name'] = task.env_name
                    command_args['dataset_path'] = str(dataset_path)
                    command_args['dataset_replace_interval'] = str(task.dataset_replace_interval)
                    command_args['save_dir'] = str(save_dir)
                    command_args['seed'] = str(seed)
                    command_args['agent'] = method.agent_path
                    command_args['run_group'] = run_group

                    for key, value in DEFAULT_MAIN_ARGS:
                        command_args[key] = value
                    for key, value in method.base_args:
                        command_args[key] = value
                    for key, value in overrides.items():
                        command_args[key] = value
                    for key, value in sweep_values.items():
                        command_args[key] = value

                    script_text = render_job_script(
                        job_name=job_name,
                        log_dir=log_dir,
                        account=args.account,
                        partition=args.partition,
                        gpus=args.gpus,
                        time_limit=args.time_limit,
                        mem=args.mem,
                        python_runner=python_runner,
                        dataset_path=dataset_path,
                        dataset_name=task.dataset_name,
                        uv_cache_dir=uv_cache_dir,
                        command_args=[f'--{key}={value}' for key, value in command_args.items()],
                        wandb_name=wandb_name,
                        wandb_tags=args.tag,
                    )

                    script_path = job_script_dir / f'{job_name}.sh'
                    script_path.write_text(script_text)
                    script_path.chmod(0o755)
                    print(f'wrote {script_path}')
                    generated += 1

                    if args.submit:
                        completed = subprocess.run(
                            ['sbatch', str(script_path)],
                            check=True,
                            cwd=str(ROOT_DIR),
                            capture_output=True,
                            text=True,
                        )
                        print(completed.stdout.strip())

    print(f'generated {generated} job script(s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
