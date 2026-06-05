#!/bin/bash
#SBATCH -J ling_video_mech_submit
#SBATCH --gres=gpu:h200:1
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH -t 10:00:00
#SBATCH -o LingVideoMechSubmit-%j.out
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
[[ -n "$ACCOUNT" ]] && SBATCH_ACCOUNT_ARGS+=(--account="$ACCOUNT")
SBATCH_PARTITION_ARGS=()
[[ -n "$PARTITION_SLURM" ]] && SBATCH_PARTITION_ARGS+=(--partition="$PARTITION_SLURM")
SBATCH_QOS_ARGS=()
[[ -n "$QOS" ]] && SBATCH_QOS_ARGS+=(--qos="$QOS")

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/storage/scratch1/9/qdai41/cosmos/LingBot-VA-Modification/outputs}"
ROOT="${ROOT:-${OUTPUT_ROOT}/video_mech_${RUN_TAG}}"
NUM_EPISODES="${NUM_EPISODES:-20}"
NUM_SHARDS="${NUM_SHARDS:-4}"
BASE_SEED="${BASE_SEED:-0}"
TOPK_INFER_PER_TRAJ="${TOPK_INFER_PER_TRAJ:-10}"
SELECTED_TIMESTEPS="${SELECTED_TIMESTEPS:-0,4,9,14,19}"
MODE="${MODE:-video}"

NOMINAL_TASKS="${NOMINAL_TASKS:-0 1 2 3 4 5 6 7 9}"
CAMERA_TASKS="${CAMERA_TASKS:-0 2 4 5 9}"
INIT_POS_TASKS="${INIT_POS_TASKS:-1 2 3 7 9}"
GAUSSIAN_TASKS="${GAUSSIAN_TASKS:-6 0 1 4 7}"

NOMINAL_RUN_DIR="${NOMINAL_RUN_DIR:-${ROOT}/nominal}"
CAMERA_RUN_DIR="${CAMERA_RUN_DIR:-${ROOT}/camera}"
INIT_POS_RUN_DIR="${INIT_POS_RUN_DIR:-${ROOT}/init_pos}"
GAUSSIAN_RUN_DIR="${GAUSSIAN_RUN_DIR:-${ROOT}/gaussian}"

mkdir -p "${ROOT}"

submit_and_capture() {
  local label="$1"
  shift
  local log_path="${ROOT}/${label}_submit.log"
  echo "===== submitting ${label} ====="
  "$@" | tee "${log_path}"
  local merge_id
  merge_id="$(sed -n 's/.*merge job[[:space:]]*:[[:space:]]*\([0-9]\+\).*/\1/p' "${log_path}" | tail -1)"
  if [[ -z "${merge_id}" ]]; then
    echo "[error] could not parse merge job id from ${log_path}" >&2
    exit 2
  fi
  echo "${merge_id}"
}

COMMON_ENV=(
  "NUM_EPISODES=${NUM_EPISODES}"
  "NUM_SHARDS=${NUM_SHARDS}"
  "BASE_SEED=${BASE_SEED}"
  "TOPK_INFER_PER_TRAJ=${TOPK_INFER_PER_TRAJ}"
  "SELECTED_TIMESTEPS=${SELECTED_TIMESTEPS}"
  "MODE=${MODE}"
  "DISABLE_VIDEO=1"
  "ACCOUNT=${ACCOUNT}"
  "PARTITION_SLURM=${PARTITION_SLURM}"
  "QOS=${QOS}"
  "GPU_GRES=${GPU_GRES}"
)

NOMINAL_MERGE_JOB_ID="$(
  env "${COMMON_ENV[@]}" \
    TASK_IDS="${NOMINAL_TASKS}" \
    RUN_DIR="${NOMINAL_RUN_DIR}" \
    bash interpretability/submit_collect_nominal.sh
)"
NOMINAL_MERGE_JOB_ID="$(printf '%s\n' "${NOMINAL_MERGE_JOB_ID}" | sed -n 's/.*merge job[[:space:]]*:[[:space:]]*\([0-9]\+\).*/\1/p' | tail -1)"
if [[ -z "${NOMINAL_MERGE_JOB_ID}" ]]; then
  echo "[error] could not parse nominal merge job id" >&2
  exit 2
fi

echo "nominal merge job: ${NOMINAL_MERGE_JOB_ID}"

CAMERA_MERGE_JOB_ID="$(
  env "${COMMON_ENV[@]}" \
    TASK_IDS="${CAMERA_TASKS}" \
    RUN_DIR="${CAMERA_RUN_DIR}" \
    NOMINAL_RUN_DIR="${NOMINAL_RUN_DIR}" \
    EXTRA_MERGE_DEPENDENCY="${NOMINAL_MERGE_JOB_ID}" \
    PERTURB_SPEC="scripts/lqr/configs/perturb_spec_camera.yaml" \
    OUT_BASE="${ROOT}/camera_out_base_unused" \
    bash interpretability/submit_collect_activations.sh
)"
CAMERA_MERGE_JOB_ID="$(printf '%s\n' "${CAMERA_MERGE_JOB_ID}" | sed -n 's/.*merge job[[:space:]]*:[[:space:]]*\([0-9]\+\).*/\1/p' | tail -1)"

INIT_POS_MERGE_JOB_ID="$(
  env "${COMMON_ENV[@]}" \
    TASK_IDS="${INIT_POS_TASKS}" \
    RUN_DIR="${INIT_POS_RUN_DIR}" \
    NOMINAL_RUN_DIR="${NOMINAL_RUN_DIR}" \
    EXTRA_MERGE_DEPENDENCY="${NOMINAL_MERGE_JOB_ID}" \
    PERTURB_SPEC="scripts/lqr/configs/perturb_spec_init_pos.yaml" \
    OUT_BASE="${ROOT}/init_pos_out_base_unused" \
    bash interpretability/submit_collect_activations.sh
)"
INIT_POS_MERGE_JOB_ID="$(printf '%s\n' "${INIT_POS_MERGE_JOB_ID}" | sed -n 's/.*merge job[[:space:]]*:[[:space:]]*\([0-9]\+\).*/\1/p' | tail -1)"

GAUSSIAN_MERGE_JOB_ID="$(
  env "${COMMON_ENV[@]}" \
    TASK_IDS="${GAUSSIAN_TASKS}" \
    RUN_DIR="${GAUSSIAN_RUN_DIR}" \
    NOMINAL_RUN_DIR="${NOMINAL_RUN_DIR}" \
    EXTRA_MERGE_DEPENDENCY="${NOMINAL_MERGE_JOB_ID}" \
    PERTURB_SPEC="scripts/lqr/configs/perturb_spec_gaussian.yaml" \
    OUT_BASE="${ROOT}/gaussian_out_base_unused" \
    bash interpretability/submit_collect_activations.sh
)"
GAUSSIAN_MERGE_JOB_ID="$(printf '%s\n' "${GAUSSIAN_MERGE_JOB_ID}" | sed -n 's/.*merge job[[:space:]]*:[[:space:]]*\([0-9]\+\).*/\1/p' | tail -1)"

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

FIG_DEP="${CAMERA_MERGE_JOB_ID}:${INIT_POS_MERGE_JOB_ID}:${GAUSSIAN_MERGE_JOB_ID}"
FIG_JOB_ID="$(sbatch --parsable \
  "${SBATCH_ACCOUNT_ARGS[@]}" \
  "${SBATCH_PARTITION_ARGS[@]}" \
  "${SBATCH_QOS_ARGS[@]}" \
  --dependency="afterok:${FIG_DEP}" \
  --export=ALL,REPO_ROOT="${REPO_ROOT}",FIGURE_ROOT="${ROOT}/figures",NOMINAL_RUN_DIR="${NOMINAL_RUN_DIR}",CAMERA_RUN_DIR="${CAMERA_RUN_DIR}",INIT_POS_RUN_DIR="${INIT_POS_RUN_DIR}",GAUSSIAN_RUN_DIR="${GAUSSIAN_RUN_DIR}" \
  interpretability/run_video_mech_figures.sbatch)"

cat <<EOF
===== submitted video mechanistic analysis workflow =====
root              : ${ROOT}
nominal_run_dir   : ${NOMINAL_RUN_DIR}
camera_run_dir    : ${CAMERA_RUN_DIR}
init_pos_run_dir  : ${INIT_POS_RUN_DIR}
gaussian_run_dir  : ${GAUSSIAN_RUN_DIR}
nominal_merge     : ${NOMINAL_MERGE_JOB_ID}
camera_merge      : ${CAMERA_MERGE_JOB_ID}
init_pos_merge    : ${INIT_POS_MERGE_JOB_ID}
gaussian_merge    : ${GAUSSIAN_MERGE_JOB_ID}
figure_job        : ${FIG_JOB_ID} afterok:${FIG_DEP}
EOF
