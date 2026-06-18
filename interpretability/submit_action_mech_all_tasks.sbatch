#!/bin/bash
#SBATCH -J ling_action_mech_submit
#SBATCH --gres=gpu:h200:1
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH -t 10:00:00
#SBATCH -o LingActionMechSubmit-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qdai41@gatech.edu

source ~/.bashrc
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/storage/home/hcoda1/9/qdai41/scratch/cosmos/LingBot-VA-Modification}"
cd "${REPO_ROOT}"

ACCOUNT="${ACCOUNT:-${SLURM_JOB_ACCOUNT:-${SLURM_ACCOUNT:-}}}"
PARTITION_SLURM="${PARTITION_SLURM:-${SLURM_JOB_PARTITION:-${SLURM_PARTITION:-}}}"
QOS="${QOS:-${SLURM_JOB_QOS:-${SLURM_QOS:-}}}"
GPU_GRES="${GPU_GRES:-gpu:h200:1}"

SBATCH_ACCOUNT_ARGS=()
[[ -n "${ACCOUNT}" ]] && SBATCH_ACCOUNT_ARGS+=(--account="${ACCOUNT}")
SBATCH_PARTITION_ARGS=()
[[ -n "${PARTITION_SLURM}" ]] && SBATCH_PARTITION_ARGS+=(--partition="${PARTITION_SLURM}")
SBATCH_QOS_ARGS=()
[[ -n "${QOS}" ]] && SBATCH_QOS_ARGS+=(--qos="${QOS}")

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/storage/scratch1/9/qdai41/cosmos/LingBot-VA-Modification/outputs}"
ROOT="${ROOT:-${OUTPUT_ROOT}/action_mech_all_tasks_${RUN_TAG}}"

ALL_TASKS="${ALL_TASKS:-0 1 2 3 4 5 6 7 8 9}"
NOMINAL_TASKS="${NOMINAL_TASKS:-${ALL_TASKS}}"
CAMERA_TASKS="${CAMERA_TASKS:-${ALL_TASKS}}"
INIT_POS_TASKS="${INIT_POS_TASKS:-${ALL_TASKS}}"
GAUSSIAN_TASKS="${GAUSSIAN_TASKS:-${ALL_TASKS}}"

NUM_EPISODES="${NUM_EPISODES:-20}"
NUM_SHARDS="${NUM_SHARDS:-4}"
BASE_SEED="${BASE_SEED:-0}"
TOPK_INFER_PER_TRAJ="${TOPK_INFER_PER_TRAJ:-10}"
SELECTED_TIMESTEPS="${SELECTED_TIMESTEPS:-0,10,20,30,40}"
MODE="${MODE:-action}"
DISABLE_VIDEO="${DISABLE_VIDEO:-1}"

NOMINAL_RUN_DIR="${NOMINAL_RUN_DIR:-${ROOT}/nominal}"
CAMERA_RUN_DIR="${CAMERA_RUN_DIR:-${ROOT}/camera}"
INIT_POS_RUN_DIR="${INIT_POS_RUN_DIR:-${ROOT}/init_pos}"
GAUSSIAN_RUN_DIR="${GAUSSIAN_RUN_DIR:-${ROOT}/gaussian}"
FIGURE_ROOT="${FIGURE_ROOT:-${ROOT}/figures}"

mkdir -p "${ROOT}" "${FIGURE_ROOT}"

COMMON_ENV=(
  "NUM_EPISODES=${NUM_EPISODES}"
  "NUM_SHARDS=${NUM_SHARDS}"
  "BASE_SEED=${BASE_SEED}"
  "TOPK_INFER_PER_TRAJ=${TOPK_INFER_PER_TRAJ}"
  "SELECTED_TIMESTEPS=${SELECTED_TIMESTEPS}"
  "MODE=${MODE}"
  "DISABLE_VIDEO=${DISABLE_VIDEO}"
  "ACCOUNT=${ACCOUNT}"
  "PARTITION_SLURM=${PARTITION_SLURM}"
  "QOS=${QOS}"
  "GPU_GRES=${GPU_GRES}"
)

parse_merge_id() {
  sed -n 's/.*merge job[[:space:]]*:[[:space:]]*\([0-9]\+\).*/\1/p' | tail -1
}

echo "===== action LingBot mechanistic analysis: all tasks ====="
echo "root              : ${ROOT}"
echo "tasks             : ${ALL_TASKS}"
echo "mode              : ${MODE}"
echo "selected timesteps: ${SELECTED_TIMESTEPS}"
echo "episodes/shards   : ${NUM_EPISODES}/${NUM_SHARDS}"
echo "figure root       : ${FIGURE_ROOT}"
echo

NOMINAL_LOG="${ROOT}/nominal_submit.log"
env "${COMMON_ENV[@]}" \
  TASK_IDS="${NOMINAL_TASKS}" \
  RUN_DIR="${NOMINAL_RUN_DIR}" \
  bash interpretability/submit_collect_nominal.sh | tee "${NOMINAL_LOG}"
NOMINAL_MERGE_JOB_ID="$(parse_merge_id < "${NOMINAL_LOG}")"
if [[ -z "${NOMINAL_MERGE_JOB_ID}" ]]; then
  echo "[error] could not parse nominal merge job id from ${NOMINAL_LOG}" >&2
  exit 2
fi
echo "nominal merge job: ${NOMINAL_MERGE_JOB_ID}"

CAMERA_LOG="${ROOT}/camera_submit.log"
env "${COMMON_ENV[@]}" \
  TASK_IDS="${CAMERA_TASKS}" \
  RUN_DIR="${CAMERA_RUN_DIR}" \
  NOMINAL_RUN_DIR="${NOMINAL_RUN_DIR}" \
  EXTRA_MERGE_DEPENDENCY="${NOMINAL_MERGE_JOB_ID}" \
  PERTURB_SPEC="scripts/lqr/configs/perturb_spec_camera.yaml" \
  OUT_BASE="${ROOT}/camera_out_base_unused" \
  bash interpretability/submit_collect_activations.sh | tee "${CAMERA_LOG}"
CAMERA_MERGE_JOB_ID="$(parse_merge_id < "${CAMERA_LOG}")"

INIT_POS_LOG="${ROOT}/init_pos_submit.log"
env "${COMMON_ENV[@]}" \
  TASK_IDS="${INIT_POS_TASKS}" \
  RUN_DIR="${INIT_POS_RUN_DIR}" \
  NOMINAL_RUN_DIR="${NOMINAL_RUN_DIR}" \
  EXTRA_MERGE_DEPENDENCY="${NOMINAL_MERGE_JOB_ID}" \
  PERTURB_SPEC="scripts/lqr/configs/perturb_spec_init_pos.yaml" \
  OUT_BASE="${ROOT}/init_pos_out_base_unused" \
  bash interpretability/submit_collect_activations.sh | tee "${INIT_POS_LOG}"
INIT_POS_MERGE_JOB_ID="$(parse_merge_id < "${INIT_POS_LOG}")"

GAUSSIAN_LOG="${ROOT}/gaussian_submit.log"
env "${COMMON_ENV[@]}" \
  TASK_IDS="${GAUSSIAN_TASKS}" \
  RUN_DIR="${GAUSSIAN_RUN_DIR}" \
  NOMINAL_RUN_DIR="${NOMINAL_RUN_DIR}" \
  EXTRA_MERGE_DEPENDENCY="${NOMINAL_MERGE_JOB_ID}" \
  PERTURB_SPEC="scripts/lqr/configs/perturb_spec_gaussian.yaml" \
  OUT_BASE="${ROOT}/gaussian_out_base_unused" \
  bash interpretability/submit_collect_activations.sh | tee "${GAUSSIAN_LOG}"
GAUSSIAN_MERGE_JOB_ID="$(parse_merge_id < "${GAUSSIAN_LOG}")"

for pair in \
  "camera:${CAMERA_MERGE_JOB_ID}" \
  "init_pos:${INIT_POS_MERGE_JOB_ID}" \
  "gaussian:${GAUSSIAN_MERGE_JOB_ID}"; do
  label="${pair%%:*}"
  job_id="${pair#*:}"
  if [[ -z "${job_id}" ]]; then
    echo "[error] could not parse ${label} merge job id" >&2
    exit 2
  fi
done

submit_figure_job() {
  local name="$1"
  local tasks="$2"
  local perturb_dir="$3"
  local merge_job_id="$4"
  local tasks_export
  tasks_export="$(echo "${tasks}" | tr ' ' ';' | tr -s ';')"

  sbatch --parsable \
    "${SBATCH_ACCOUNT_ARGS[@]}" \
    "${SBATCH_PARTITION_ARGS[@]}" \
    "${SBATCH_QOS_ARGS[@]}" \
    --dependency="afterany:${merge_job_id}" \
    --job-name="ling_action_${name}_fig" \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",FIGURE_ROOT="${FIGURE_ROOT}",NOMINAL_RUN_DIR="${NOMINAL_RUN_DIR}",PERTURB_RUN_DIR="${perturb_dir}",PERTURB_NAME="${name}",TASKS="${tasks_export}" \
    interpretability/run_action_mech_one_perturb.sbatch
}

CAMERA_FIG_JOB_ID="$(submit_figure_job camera "${CAMERA_TASKS}" "${CAMERA_RUN_DIR}" "${CAMERA_MERGE_JOB_ID}")"
INIT_POS_FIG_JOB_ID="$(submit_figure_job init_pos "${INIT_POS_TASKS}" "${INIT_POS_RUN_DIR}" "${INIT_POS_MERGE_JOB_ID}")"
GAUSSIAN_FIG_JOB_ID="$(submit_figure_job gaussian "${GAUSSIAN_TASKS}" "${GAUSSIAN_RUN_DIR}" "${GAUSSIAN_MERGE_JOB_ID}")"

cat <<EOF
===== submitted action mechanistic analysis workflow =====
root              : ${ROOT}
nominal_run_dir   : ${NOMINAL_RUN_DIR}
camera_run_dir    : ${CAMERA_RUN_DIR}
init_pos_run_dir  : ${INIT_POS_RUN_DIR}
gaussian_run_dir  : ${GAUSSIAN_RUN_DIR}
figure_root       : ${FIGURE_ROOT}
nominal_merge     : ${NOMINAL_MERGE_JOB_ID}
camera_merge      : ${CAMERA_MERGE_JOB_ID}
init_pos_merge    : ${INIT_POS_MERGE_JOB_ID}
gaussian_merge    : ${GAUSSIAN_MERGE_JOB_ID}
camera_fig_job    : ${CAMERA_FIG_JOB_ID} afterany:${CAMERA_MERGE_JOB_ID}
init_pos_fig_job  : ${INIT_POS_FIG_JOB_ID} afterany:${INIT_POS_MERGE_JOB_ID}
gaussian_fig_job  : ${GAUSSIAN_FIG_JOB_ID} afterany:${GAUSSIAN_MERGE_JOB_ID}
EOF
