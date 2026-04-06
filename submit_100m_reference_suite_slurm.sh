#!/bin/bash
set -euo pipefail

# Generate the 100M reference suite as separate SLURM jobs.
# By default this only writes one small job script per run into JOB_SCRIPT_DIR.
# Set SUBMIT=1 to also call sbatch on each generated script.

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMPLS_DIR="${ROOT_DIR}/impls"
DATA_DIR="${DATA_DIR:-/scratch/engin_root/engin1/hliuson/.ogbench/data}"
SAVE_DIR="${SAVE_DIR:-/scratch/engin_root/engin1/hliuson/ogbench_exp}"
PYTHON_RUNNER="${PYTHON_RUNNER:-uv run python}"
JOB_SCRIPT_DIR="${JOB_SCRIPT_DIR:-${ROOT_DIR}/generated_slurm_jobs}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/impls/slurm_logs}"
SUBMIT="${SUBMIT:-0}"
SEEDS=(0 1 2)

mkdir -p "${JOB_SCRIPT_DIR}" "${LOG_DIR}"

submit_job() {
  local job_name="$1"
  local env_name="$2"
  local dataset_name="$3"
  local dataset_replace_interval="$4"
  local seed="$5"
  local agent_path="$6"
  local run_group="$7"
  shift 7

  local script_path="${JOB_SCRIPT_DIR}/${job_name}.sh"
  local dataset_path="${DATA_DIR}/${dataset_name}"
  local base_url="https://rail.eecs.berkeley.edu/datasets/ogbench/${dataset_name}"

  cat > "${script_path}" <<JOB
#!/bin/bash
#SBATCH --account=spitis0
#SBATCH --job-name=${job_name}
#SBATCH --nodes=1
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=${LOG_DIR}/slurm-%j.out

set -euo pipefail
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
module load uv
export UV_CACHE_DIR=/scratch/engin_root/engin1/hliuson/.cache/uv
mkdir -p "\${UV_CACHE_DIR}"
mkdir -p "${dataset_path}"
if ! ls "${dataset_path}"/*.npz >/dev/null 2>&1; then
  (
    cd "${dataset_path}"
    wget -nc -r -np -nH --cut-dirs=4 -A "*.npz" "${base_url}/"
  )
fi
cd "${IMPLS_DIR}"
${PYTHON_RUNNER} main.py \
  --env_name=${env_name} \
  --dataset_path=${dataset_path} \
  --dataset_replace_interval=${dataset_replace_interval} \
  --save_dir=${SAVE_DIR} \
  --seed=${seed} \
  --agent=${agent_path} \
  --train_steps=1000000 \
  --log_interval=5000 \
  --eval_interval=100000 \
  --save_interval=100000 \
  --eval_episodes=20 \
  --eval_on_cpu=0 \
  --video_episodes=0 \
  --run_group=${run_group} \
$*
JOB

  chmod +x "${script_path}"
  echo "wrote ${script_path}"
  if [ "${SUBMIT}" = "1" ]; then
    sbatch "${script_path}"
  fi
}

submit_trl() {
  local env_name="$1" dataset_name="$2" dataset_replace_interval="$3" seed="$4" run_group="$5" short_name="$6"
  submit_job "${short_name}-s${seed}" "${env_name}" "${dataset_name}" "${dataset_replace_interval}" "${seed}" agents/trl.py "${run_group}" \
    --agent.batch_size=1024 \
    --agent.discount=0.999 \
    --agent.expectile=0.7 \
    --agent.value_hidden_dims='(1024, 1024, 1024, 1024)' \
    --agent.actor_hidden_dims='(1024, 1024, 1024, 1024)' \
    --agent.value_p_curgoal=0.0 \
    --agent.value_p_trajgoal=0.8 \
    --agent.value_p_randomgoal=0.2 \
    --agent.actor_p_trajgoal=0.5 \
    --agent.actor_p_randomgoal=0.5 \
    --agent.q_action_n_step=25
}

submit_mc() {
  local env_name="$1" dataset_name="$2" dataset_replace_interval="$3" seed="$4" run_group="$5" short_name="$6"
  submit_job "${short_name}-s${seed}" "${env_name}" "${dataset_name}" "${dataset_replace_interval}" "${seed}" agents/mc.py "${run_group}" \
    --agent.batch_size=1024 \
    --agent.discount=0.999 \
    --agent.value_hidden_dims='(1024, 1024, 1024, 1024)' \
    --agent.actor_hidden_dims='(1024, 1024, 1024, 1024)' \
    --agent.value_p_curgoal=0.0 \
    --agent.value_p_trajgoal=0.8 \
    --agent.value_p_randomgoal=0.2 \
    --agent.actor_p_trajgoal=0.5 \
    --agent.actor_p_randomgoal=0.5 \
    --agent.q_action_n_step=25
}

submit_gciql() {
  local env_name="$1" dataset_name="$2" dataset_replace_interval="$3" seed="$4" run_group="$5" short_name="$6"
  submit_job "${short_name}-s${seed}" "${env_name}" "${dataset_name}" "${dataset_replace_interval}" "${seed}" agents/gciql_nstep.py "${run_group}" \
    --agent.batch_size=1024 \
    --agent.discount=0.999 \
    --agent.expectile=0.9 \
    --agent.cf_expectile=0.9 \
    --agent.critic_n_step=25 \
    --agent.value_hidden_dims='(1024, 1024, 1024, 1024)' \
    --agent.actor_hidden_dims='(1024, 1024, 1024, 1024)' \
    --agent.value_p_curgoal=0.0 \
    --agent.value_p_trajgoal=0.8 \
    --agent.value_p_randomgoal=0.2 \
    --agent.actor_p_trajgoal=0.5 \
    --agent.actor_p_randomgoal=0.5 \
    --agent.q_action_n_step=25
}

submit_latent_trl() {
  local env_name="$1" dataset_name="$2" dataset_replace_interval="$3" seed="$4" run_group="$5" short_name="$6"
  submit_job "${short_name}-s${seed}" "${env_name}" "${dataset_name}" "${dataset_replace_interval}" "${seed}" agents/latent_trl.py "${run_group}" \
    --agent.batch_size=1024 \
    --agent.discount=0.999 \
    --agent.expectile=0.7 \
    --agent.value_hidden_dims='(1024, 1024, 1024, 1024)' \
    --agent.actor_hidden_dims='(1024, 1024, 1024, 1024)' \
    --agent.z_dim=32 \
    --agent.state_z_dim=32 \
    --agent.vae_encoder_hidden_dims='(256, 256, 256, 256)' \
    --agent.vae_decoder_hidden_dims='(256, 256)' \
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
    --agent.q_action_n_step=25 \
    --agent.z_proposal_awr_beta=1.0 \
    --agent.z_proposal_awr_max_weight=20.0 \
    --agent.z_proposal_awr_num_random_support=1 \
    --agent.z_proposal_awr_value_eps=0.01 \
    --agent.z_proposal_awr_intraj_prob=0.5
}

for seed in "${SEEDS[@]}"; do
  submit_trl   puzzle-4x6-play-oraclerep-v0             puzzle-4x6-play-100m-v0             1000   "${seed}" trl-p46-100m       trl-p46
  submit_mc    puzzle-4x6-play-oraclerep-v0             puzzle-4x6-play-100m-v0             1000   "${seed}" mc-p46-100m        mc-p46
  submit_gciql puzzle-4x6-play-oraclerep-v0             puzzle-4x6-play-100m-v0             1000   "${seed}" gciql-p46-100m     gciql-p46
  submit_latent_trl puzzle-4x6-play-oraclerep-v0        puzzle-4x6-play-100m-v0             1000   "${seed}" ltrl-p46-100m      ltrl-p46

done

for seed in "${SEEDS[@]}"; do
  submit_trl   cube-triple-play-oraclerep-v0            cube-triple-play-100m-v0            1000   "${seed}" trl-ct-100m        trl-ct
  submit_mc    cube-triple-play-oraclerep-v0            cube-triple-play-100m-v0            1000   "${seed}" mc-ct-100m         mc-ct
  submit_gciql cube-triple-play-oraclerep-v0            cube-triple-play-100m-v0            1000   "${seed}" gciql-ct-100m      gciql-ct
  submit_latent_trl cube-triple-play-oraclerep-v0       cube-triple-play-100m-v0            1000   "${seed}" ltrl-ct-100m       ltrl-ct

done

for seed in "${SEEDS[@]}"; do
  submit_trl   humanoidmaze-giant-navigate-oraclerep-v0 humanoidmaze-giant-navigate-100m-v0 100000 "${seed}" trl-hmg-nav-100m   trl-hmg-nav
  submit_mc    humanoidmaze-giant-navigate-oraclerep-v0 humanoidmaze-giant-navigate-100m-v0 100000 "${seed}" mc-hmg-nav-100m    mc-hmg-nav
  submit_gciql humanoidmaze-giant-navigate-oraclerep-v0 humanoidmaze-giant-navigate-100m-v0 100000 "${seed}" gciql-hmg-nav-100m gciql-hmg-nav
  submit_latent_trl humanoidmaze-giant-navigate-oraclerep-v0 humanoidmaze-giant-navigate-100m-v0 100000 "${seed}" ltrl-hmg-nav-100m ltrl-hmg-nav

done
