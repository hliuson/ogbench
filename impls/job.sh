#!/bin/bash
# (See https://arc-ts.umich.edu/greatlakes/user-guide/ for command details)

# Set up batch job settings
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=spgpu
export WANDB_API_KEY=62aa3ffda175f641d18a968e6d57826a73c207da
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl


module load python/3.12.1
module load uv


uv run main.py --env_name=cube-quadruple-play-v0 --eval_episodes=5 --agent=agents/htrl.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10
