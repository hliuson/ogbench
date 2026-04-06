#!/bin/bash
# TRL baseline on humanoidmaze-giant-navigate-oraclerep-v0 with the 100M navigate dataset.

#SBATCH --account=spitis0
#SBATCH --job-name=trl-hmg-nav
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=spgpu
#SBATCH --output=/home/hliuson/research/ogbench/impls/slurm_logs/slurm-%j.out

set -euo pipefail

Q_SHORT_N_STEP="${Q_SHORT_N_STEP:-25}"
EXPECTILE="${EXPECTILE:-0.7}"
VALUE_P_CURGOAL="${VALUE_P_CURGOAL:-0.0}"
VALUE_P_TRAJGOAL="${VALUE_P_TRAJGOAL:-0.8}"
VALUE_P_RANDOMGOAL="${VALUE_P_RANDOMGOAL:-0.2}"
ACTOR_P_TRAJGOAL="${ACTOR_P_TRAJGOAL:-0.5}"
ACTOR_P_RANDOMGOAL="${ACTOR_P_RANDOMGOAL:-0.5}"

NAME_PARTS=()
if [ "${Q_SHORT_N_STEP}" != "25" ]; then
    NAME_PARTS+=("an=${Q_SHORT_N_STEP}")
fi
if [ "${VALUE_P_CURGOAL}" != "0.0" ]; then
    NAME_PARTS+=("cur=${VALUE_P_CURGOAL}")
fi
SHORT_NAME="${NAME_PARTS[*]}"
if [ -z "${SHORT_NAME}" ]; then
    SHORT_NAME="baseline"
fi

export WANDB_NAME="trl ${SHORT_NAME}"
export WANDB_TAGS="trl,humanoidmaze,giant,navigate,oraclerep,100m,policy_extraction"
export WANDB_API_KEY=62aa3ffda175f641d18a968e6d57826a73c207da
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

module load uv
export UV_CACHE_DIR="/scratch/engin_root/engin1/hliuson/.cache/uv"
mkdir -p "${UV_CACHE_DIR}"

cd /home/hliuson/research/ogbench/impls

DATASET_PATH="/scratch/engin_root/engin1/hliuson/.ogbench/data/humanoidmaze-giant-navigate-100m-v0"
SAVE_DIR="/scratch/engin_root/engin1/hliuson/ogbench_exp"

uv run python main.py \
    --env_name=humanoidmaze-giant-navigate-oraclerep-v0 \
    --dataset_path="${DATASET_PATH}" \
    --dataset_replace_interval=100000 \
    --save_dir="${SAVE_DIR}" \
    --seed=0 \
    --agent=agents/trl.py \
    --train_steps=1000000 \
    --log_interval=5000 \
    --eval_interval=100000 \
    --save_interval=100000 \
    --eval_episodes=20 \
    --eval_on_cpu=0 \
    --video_episodes=0 \
    --agent.batch_size=1024 \
    --agent.discount=0.999 \
    --agent.expectile="${EXPECTILE}" \
    --agent.value_hidden_dims="(1024, 1024, 1024, 1024)" \
    --agent.actor_hidden_dims="(1024, 1024, 1024, 1024)" \
    --agent.value_p_curgoal="${VALUE_P_CURGOAL}" \
    --agent.value_p_trajgoal="${VALUE_P_TRAJGOAL}" \
    --agent.value_p_randomgoal="${VALUE_P_RANDOMGOAL}" \
    --agent.actor_p_trajgoal="${ACTOR_P_TRAJGOAL}" \
    --agent.actor_p_randomgoal="${ACTOR_P_RANDOMGOAL}" \
    --agent.q_short_n_step="${Q_SHORT_N_STEP}" \
    --run_group="trl-hmaze-giant-navigate-oraclerep-100m"
