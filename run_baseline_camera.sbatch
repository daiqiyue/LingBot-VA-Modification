#!/bin/bash
#SBATCH -J lingbot_policy_baseline_camera
#SBATCH -A gts-gchou3-ideasci23_dgx
#SBATCH -q inferno
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#SBATCH -t 24:00:00
#SBATCH -o Policy_Baseline_Camera-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qdai41@gatech.edu
#
# Pure policy rollout under ctrlwam cam_random_large camera-view perturbation
# (no LQR). Defaults evaluate libero_10 tasks 0,2,4,5,9.
#
# ctrlwam camera pipeline uses:
#   cam_random_large: pos_sigma=0.10m, rot_sigma=8deg, fov_sigma=5deg
#   visibility constrained workspace: table_z=0.90, visible_fraction=0.55,
#   margin=8px, max_rejection_attempts=2000
#   collect seed=42; eval sweep seed=99
#
# Examples:
#   sbatch run_baseline_camera.sbatch
#   sbatch --export=ALL,TASK_IDS="0",NUM_EPISODES=5 run_baseline_camera.sbatch
#   sbatch --export=ALL,RANDOM_CAMERA_BASE_SEED=42 run_baseline_camera.sbatch

source ~/.bashrc
set -euo pipefail

conda activate "${CONDA_ENV:-lingbot}"
cd /storage/home/hcoda1/9/qdai41/scratch/cosmos/LingBot-VA-Modification
export PYTHONPATH=.

export CONFIG_NAME="${CONFIG_NAME:-libero}"
export LIBERO_BENCHMARK="${LIBERO_BENCHMARK:-libero_10}"

# Space-separated explicit LIBERO task ids.
export TASK_IDS="${TASK_IDS:-0 2 4 5 9}"
export NUM_EPISODES="${NUM_EPISODES:-10}"

export PERTURB_SPEC="${PERTURB_SPEC:-scripts/lqr/configs/perturb_spec_camera.yaml}"

# ctrlwam camera eval rollouts use random seed 99.
export RANDOM_CAMERA_BASE_SEED="${RANDOM_CAMERA_BASE_SEED:-99}"

export PROMPT="${PROMPT:-}"
export RESUME="${RESUME:-0}"
export SAVE_ROOT="${SAVE_ROOT:-}"
export EVAL_STARTUP_WAIT_SEC="${EVAL_STARTUP_WAIT_SEC:-1200}"

source scripts/lqr/slurm_port.sh
DEFAULT_MASTER_PORT="$((12000 + (${SLURM_JOB_ID:-0} % 20000)))"
export MASTER_PORT="${MASTER_PORT:-${DEFAULT_MASTER_PORT}}"

TS="$(date +%Y%m%d_%H%M%S)"
TASK_TAG="$(echo "${TASK_IDS}" | tr ' ' '_')"
PERTURB_TAG="$(basename "${PERTURB_SPEC}" .yaml)"
PERTURB_TAG="${PERTURB_TAG#perturb_spec_}"
PERTURB_TAG="${PERTURB_TAG#eval_}"
export OUT_DIR="${OUT_DIR:-outputs/libero_baseline_${PERTURB_TAG}_seed${RANDOM_CAMERA_BASE_SEED}_tasks${TASK_TAG}_${TS}}"

echo "===== Libero policy baseline (ctrlwam cam_random_large, no LQR) ====="
echo "  REPO=$(pwd)"
echo "  PERTURB_SPEC=${PERTURB_SPEC}"
echo "  cam_random_large: pos=0.10m rot=8deg fov=5deg visibility-constrained"
echo "  RANDOM_CAMERA_BASE_SEED=${RANDOM_CAMERA_BASE_SEED}  (ctrlwam eval=99, collect=42)"
echo "  TASK_IDS=${TASK_IDS}"
echo "  NUM_EPISODES=${NUM_EPISODES}"
echo "  OUT_DIR=${OUT_DIR}"
echo "  PORT=${PORT}"
echo "  MASTER_PORT=${MASTER_PORT}"
echo "======================================================================"

POLICY_EVAL_ARGS=(
  python scripts/lqr/run_libero_policy_eval.py
  --config-name "${CONFIG_NAME}"
  --libero-benchmark "${LIBERO_BENCHMARK}"
  --task-ids ${TASK_IDS}
  --num-episodes "${NUM_EPISODES}"
  --perturb-spec "${PERTURB_SPEC}"
  --random-camera-base-seed "${RANDOM_CAMERA_BASE_SEED}"
  --out-dir "${OUT_DIR}"
  --port "${PORT}"
  --master-port "${MASTER_PORT}"
  --startup-wait-sec "${EVAL_STARTUP_WAIT_SEC}"
  --save-root "${SAVE_ROOT}"
)
if [[ -n "${PROMPT}" ]]; then
  POLICY_EVAL_ARGS+=(--prompt "${PROMPT}")
fi
if [[ "${RESUME}" == "1" ]]; then
  POLICY_EVAL_ARGS+=(--resume)
fi

"${POLICY_EVAL_ARGS[@]}"
echo "[done] summary: ${OUT_DIR}/summary.json"
