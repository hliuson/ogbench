#!/bin/bash
set -euo pipefail

# Plain reference launch script for 100M oraclerep experiments.
# Edit DATA_DIR, SAVE_DIR, or PYTHON_RUNNER if needed, then run from the repo root.

DATA_DIR="${DATA_DIR:-/scratch/engin_root/engin1/hliuson/.ogbench/data}"
SAVE_DIR="${SAVE_DIR:-/scratch/engin_root/engin1/hliuson/ogbench_exp}"
PYTHON_RUNNER="${PYTHON_RUNNER:-uv run python}"

# shellcheck disable=SC2206
PYTHON_CMD=(${PYTHON_RUNNER})
SEEDS=(0 1 2)

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

ensure_dataset() {
  local dataset_name="$1"
  local dataset_dir="${DATA_DIR}/${dataset_name}"
  local base_url="https://rail.eecs.berkeley.edu/datasets/ogbench/${dataset_name}"

  if ls "${dataset_dir}"/*.npz >/dev/null 2>&1; then
    return
  fi

  mkdir -p "${dataset_dir}"
  (
    cd "${dataset_dir}"
    wget -nc -r -np -nH --cut-dirs=4 -A "*.npz" "${base_url}/"
  )
}

run_main() {
  local env_name="$1"
  local dataset_path="$2"
  local seed="$3"
  local agent_path="$4"
  local run_group="$5"
  local dataset_replace_interval="$6"
  shift 6

  "${PYTHON_CMD[@]}" main.py \
    --env_name="${env_name}" \
    --dataset_path="${dataset_path}" \
    --dataset_replace_interval="${dataset_replace_interval}" \
    --save_dir="${SAVE_DIR}" \
    --seed="${seed}" \
    --agent="${agent_path}" \
    --train_steps=1000000 \
    --log_interval=5000 \
    --eval_interval=100000 \
    --save_interval=100000 \
    --eval_episodes=20 \
    --eval_on_cpu=0 \
    --video_episodes=0 \
    "$@" \
    --run_group="${run_group}"
}

run_trl() {
  local env_name="$1"
  local dataset_path="$2"
  local seed="$3"
  local run_group="$4"
  local dataset_replace_interval="$5"
  run_main "${env_name}" "${dataset_path}" "${seed}" agents/trl.py "${run_group}" "${dataset_replace_interval}" \
    --agent.batch_size=1024 \
    --agent.discount=0.999 \
    --agent.expectile=0.7 \
    --agent.value_hidden_dims="(1024, 1024, 1024, 1024)" \
    --agent.actor_hidden_dims="(1024, 1024, 1024, 1024)" \
    --agent.value_p_curgoal=0.0 \
    --agent.value_p_trajgoal=0.8 \
    --agent.value_p_randomgoal=0.2 \
    --agent.actor_p_trajgoal=0.5 \
    --agent.actor_p_randomgoal=0.5 \
    --agent.q_short_n_step=25
}

run_mc() {
  local env_name="$1"
  local dataset_path="$2"
  local seed="$3"
  local run_group="$4"
  local dataset_replace_interval="$5"
  run_main "${env_name}" "${dataset_path}" "${seed}" agents/mc.py "${run_group}" "${dataset_replace_interval}" \
    --agent.batch_size=1024 \
    --agent.discount=0.999 \
    --agent.value_hidden_dims="(1024, 1024, 1024, 1024)" \
    --agent.actor_hidden_dims="(1024, 1024, 1024, 1024)" \
    --agent.value_p_curgoal=0.0 \
    --agent.value_p_trajgoal=0.8 \
    --agent.value_p_randomgoal=0.2 \
    --agent.actor_p_trajgoal=0.5 \
    --agent.actor_p_randomgoal=0.5 \
    --agent.q_short_n_step=25
}

run_gciql() {
  local env_name="$1"
  local dataset_path="$2"
  local seed="$3"
  local run_group="$4"
  local dataset_replace_interval="$5"
  run_main "${env_name}" "${dataset_path}" "${seed}" agents/gciql_nstep.py "${run_group}" "${dataset_replace_interval}" \
    --agent.batch_size=1024 \
    --agent.discount=0.999 \
    --agent.expectile=0.9 \
    --agent.cf_expectile=0.9 \
    --agent.critic_n_step=25 \
    --agent.value_hidden_dims="(1024, 1024, 1024, 1024)" \
    --agent.actor_hidden_dims="(1024, 1024, 1024, 1024)" \
    --agent.value_p_curgoal=0.0 \
    --agent.value_p_trajgoal=0.8 \
    --agent.value_p_randomgoal=0.2 \
    --agent.actor_p_trajgoal=0.5 \
    --agent.actor_p_randomgoal=0.5 \
    --agent.q_short_n_step=25
}

run_latent_trl() {
  local env_name="$1"
  local dataset_path="$2"
  local seed="$3"
  local run_group="$4"
  local dataset_replace_interval="$5"
  local q_short_n_step="$6"
  run_main "${env_name}" "${dataset_path}" "${seed}" agents/latent_trl.py "${run_group}" "${dataset_replace_interval}" \
    --agent.batch_size=1024 \
    --agent.discount=0.999 \
    --agent.expectile=0.7 \
    --agent.value_hidden_dims="(1024, 1024, 1024, 1024)" \
    --agent.actor_hidden_dims="(1024, 1024, 1024, 1024)" \
    --agent.z_dim=32 \
    --agent.state_z_dim=32 \
    --agent.vae_encoder_hidden_dims="(256, 256, 256, 256)" \
    --agent.vae_decoder_hidden_dims="(256, 256)" \
    --agent.reg_coef=1.0 \
    --agent.vae_recon_coef=0.25 \
    --agent.vae_beta=0.01 \
    --agent.midpoint_decoder_coef=1.0 \
    --agent.direct_intraj_value_max_offset=32 \
    --agent.z_proposal_coef=1.0 \
    --agent.cf_num_z_proposals=1 \
    --agent.value_p_curgoal=0.0 \
    --agent.value_p_trajgoal=0.8 \
    --agent.value_p_randomgoal=0.2 \
    --agent.actor_p_trajgoal=0.5 \
    --agent.actor_p_randomgoal=0.5 \
    --agent.sigreg_coef=0.001 \
    --agent.cf_expectile=0.7 \
    --agent.q_short_n_step="${q_short_n_step}" \
    --agent.z_proposal_awr_beta=1.0 \
    --agent.z_proposal_awr_max_weight=20.0 \
    --agent.z_proposal_awr_num_random_support=1 \
    --agent.z_proposal_awr_value_eps=0.01 \
    --agent.z_proposal_awr_intraj_prob=0.5
}

ensure_dataset puzzle-4x6-play-100m-v0
ensure_dataset cube-triple-play-100m-v0
ensure_dataset humanoidmaze-giant-navigate-100m-v0

cd "$(dirname "$0")/impls"

for seed in "${SEEDS[@]}"; do
  run_trl   puzzle-4x6-play-oraclerep-v0              "${DATA_DIR}/puzzle-4x6-play-100m-v0"              "${seed}" trl-p46-100m            1000
  run_mc    puzzle-4x6-play-oraclerep-v0              "${DATA_DIR}/puzzle-4x6-play-100m-v0"              "${seed}" mc-p46-100m             1000
  run_gciql puzzle-4x6-play-oraclerep-v0              "${DATA_DIR}/puzzle-4x6-play-100m-v0"              "${seed}" gciql-p46-100m          1000
  run_latent_trl puzzle-4x6-play-oraclerep-v0         "${DATA_DIR}/puzzle-4x6-play-100m-v0"              "${seed}" ltrl-p46-100m           1000 1

done

for seed in "${SEEDS[@]}"; do
  run_trl   cube-triple-play-oraclerep-v0             "${DATA_DIR}/cube-triple-play-100m-v0"             "${seed}" trl-ct-100m             1000
  run_mc    cube-triple-play-oraclerep-v0             "${DATA_DIR}/cube-triple-play-100m-v0"             "${seed}" mc-ct-100m              1000
  run_gciql cube-triple-play-oraclerep-v0             "${DATA_DIR}/cube-triple-play-100m-v0"             "${seed}" gciql-ct-100m           1000
  run_latent_trl cube-triple-play-oraclerep-v0        "${DATA_DIR}/cube-triple-play-100m-v0"             "${seed}" ltrl-ct-100m            1000 25

done

for seed in "${SEEDS[@]}"; do
  run_trl   humanoidmaze-giant-navigate-oraclerep-v0  "${DATA_DIR}/humanoidmaze-giant-navigate-100m-v0"  "${seed}" trl-hmg-nav-100m        100000
  run_mc    humanoidmaze-giant-navigate-oraclerep-v0  "${DATA_DIR}/humanoidmaze-giant-navigate-100m-v0"  "${seed}" mc-hmg-nav-100m         100000
  run_gciql humanoidmaze-giant-navigate-oraclerep-v0  "${DATA_DIR}/humanoidmaze-giant-navigate-100m-v0"  "${seed}" gciql-hmg-nav-100m      100000
  run_latent_trl humanoidmaze-giant-navigate-oraclerep-v0 "${DATA_DIR}/humanoidmaze-giant-navigate-100m-v0" "${seed}" ltrl-hmg-nav-100m     100000 25

done
