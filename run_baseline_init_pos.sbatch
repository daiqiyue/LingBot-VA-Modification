#!/bin/bash
#SBATCH -J lingbot_policy_baseline_init_pos
#SBATCH -A gts-gchou3-paid2
#SBATCH -q inferno
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH -t 24:00:00
#SBATCH -o Policy_Baseline_InitPos-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qdai41@gatech.edu
#
# Pure policy rollout under ctrlwam gripper-XYZ perturbation (no LQR).
#
# ctrlwam run_task07_gripper_xyz_pipeline.sh uses:
#   PRESET=xyz_random_xlarge_3   (sigma=0.10 m isotropic, per-episode Gaussian)
#   BASE_SEED=42                 (collect / SVD / Jacobian)
#   SWEEP_BASE_SEED=99           (LQR eval rollouts in step 5)
#   N_EPISODES=50
#
# LingBot equivalent: PERTURB_SPEC=perturb_spec_init_pos.yaml + GRIPPER_XYZ_BASE_SEED.
#
# Examples:
#   # tasks 1,2,3,7,9 — same perturb as ctrlwam eval (seed 99), 20 eps
#   sbatch run_baseline_init_pos.sbatch
#
#   # single task 7 only
#   sbatch --export=ALL,TASK_IDS="7" run_baseline_init_pos.sbatch
#
#   # match ctrlwam collect seed (42) instead of eval seed (99)
#   sbatch --export=ALL,GRIPPER_XYZ_BASE_SEED=42 run_baseline_init_pos.sbatch
#
#   # fewer episodes for a quick smoke test
#   sbatch --export=ALL,NUM_EPISODES=10 run_baseline_init_pos.sbatch

source ~/.bashrc
set -euo pipefail

conda activate "${CONDA_ENV:-lingbot}"
cd /storage/home/hcoda1/9/qdai41/scratch/cosmos/LingBot-VA-Modification
export PYTHONPATH=.

export CONFIG_NAME="${CONFIG_NAME:-libero}"
export LIBERO_BENCHMARK="${LIBERO_BENCHMARK:-libero_10}"

# libero_10 tasks to evaluate (space-separated). Default: 1, 2, 3, 7, 9.
export TASK_IDS="${TASK_IDS:-1 2 3 7 9}"

# ctrlwam collect uses 50; override with NUM_EPISODES=50 if needed.
export NUM_EPISODES="${NUM_EPISODES:-10}"

# xyz_random_xlarge_3 + base_seed (see GRIPPER_XYZ_BASE_SEED below).
export PERTURB_SPEC="${PERTURB_SPEC:-scripts/lqr/configs/perturb_spec_init_pos.yaml}"

# ctrlwam step-5 eval rollouts use SWEEP_BASE_SEED=99 (SVD built with seed 42).
export GRIPPER_XYZ_BASE_SEED="${GRIPPER_XYZ_BASE_SEED:-99}"

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
export OUT_DIR="${OUT_DIR:-outputs/libero_baseline_${PERTURB_TAG}_seed${GRIPPER_XYZ_BASE_SEED}_tasks${TASK_TAG}_${TS}}"

echo "===== Libero policy baseline (ctrlwam gripper XYZ, no LQR) ====="
echo "  REPO=$(pwd)"
echo "  PERTURB_SPEC=${PERTURB_SPEC}"
echo "  preset=xyz_random_xlarge_3 (from spec)"
echo "  GRIPPER_XYZ_BASE_SEED=${GRIPPER_XYZ_BASE_SEED}  (ctrlwam eval=99, collect=42)"
echo "  TASK_IDS=${TASK_IDS}"
echo "  NUM_EPISODES=${NUM_EPISODES}"
echo "  OUT_DIR=${OUT_DIR}"
echo "  PORT=${PORT}"
echo "  MASTER_PORT=${MASTER_PORT}"
echo "================================================================"

POLICY_EVAL_ARGS=(
  python scripts/lqr/run_libero_policy_eval.py
  --config-name "${CONFIG_NAME}"
  --libero-benchmark "${LIBERO_BENCHMARK}"
  --task-ids ${TASK_IDS}
  --num-episodes "${NUM_EPISODES}"
  --perturb-spec "${PERTURB_SPEC}"
  --gripper-xyz-base-seed "${GRIPPER_XYZ_BASE_SEED}"
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
