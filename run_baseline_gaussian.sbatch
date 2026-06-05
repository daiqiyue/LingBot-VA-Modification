#!/bin/bash
#SBATCH -J lingbot_policy_baseline_gaussian
#SBATCH -A gts-gchou3-ideas_l40s
#SBATCH -q inferno
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40s:1
#SBATCH -t 24:00:00
#SBATCH -o Policy_Baseline_Gaussian-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qdai41@gatech.edu
#
# Pure policy rollout under ctrlwam noise_extreme perturbation (no LQR).
#
# ctrlwam run_noise_extreme_pipeline.sh uses:
#   noise_extreme: sigma=90 (uint8 pixel units) on agentview + wrist images
#   per_episode_seed: rng(seed=noise_seed_base + episode_idx)
#   collect: noise_seed_base=0 (episode_idx only)
#   eval (run_sweep_noise_extreme_seed99.sh): NOISE_SEED_BASE=99
#   N_EPISODES=50 for sweep; task0 collect uses 10 eps for SVD
#
# LingBot equivalent: PERTURB_SPEC=perturb_spec_gaussian.yaml + overrides below.
#
# Examples:
#   # tasks 0,1,4,6,7 — same perturb as ctrlwam eval (seed 99), 20 eps
#   sbatch run_baseline_gaussian.sbatch
#
#   # single task 0 only
#   sbatch --export=ALL,TASK_IDS="0" run_baseline_gaussian.sbatch
#
#   # match ctrlwam collect seed (0) instead of eval seed (99)
#   sbatch --export=ALL,AGENTVIEW_NOISE_SEED_BASE=0 run_baseline_gaussian.sbatch

source ~/.bashrc
set -euo pipefail

conda activate "${CONDA_ENV:-lingbot}"
cd /storage/home/hcoda1/9/qdai41/scratch/cosmos/LingBot-VA-Modification
export PYTHONPATH=.

export CONFIG_NAME="${CONFIG_NAME:-libero}"
export LIBERO_BENCHMARK="${LIBERO_BENCHMARK:-libero_10}"

# libero_10 tasks to evaluate (space-separated).
export TASK_IDS="${TASK_IDS:-0 1 4 6 7}"

export NUM_EPISODES="${NUM_EPISODES:-20}"

# noise_extreme: kind=gaussian, sigma=90
export PERTURB_SPEC="${PERTURB_SPEC:-scripts/lqr/configs/perturb_spec_gaussian_10.yaml}"
export AGENTVIEW_NOISE_SIGMA="${AGENTVIEW_NOISE_SIGMA:-30}"

# ctrlwam eval rollouts use NOISE_SEED_BASE=99 (contrastive collect uses 0).
export AGENTVIEW_NOISE_SEED_BASE="${AGENTVIEW_NOISE_SEED_BASE:-99}"

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
export OUT_DIR="${OUT_DIR:-outputs/libero_baseline_${PERTURB_TAG}_seed${AGENTVIEW_NOISE_SEED_BASE}_tasks${TASK_TAG}_${TS}}"

echo "===== Libero policy baseline (ctrlwam noise_extreme, no LQR) ====="
echo "  REPO=$(pwd)"
echo "  PERTURB_SPEC=${PERTURB_SPEC}"
echo "  noise_extreme: sigma=${AGENTVIEW_NOISE_SIGMA} (agentview + wrist)"
echo "  AGENTVIEW_NOISE_SEED_BASE=${AGENTVIEW_NOISE_SEED_BASE}  (ctrlwam eval=99, collect=0)"
echo "  TASK_IDS=${TASK_IDS}"
echo "  NUM_EPISODES=${NUM_EPISODES}"
echo "  OUT_DIR=${OUT_DIR}"
echo "  PORT=${PORT}"
echo "  MASTER_PORT=${MASTER_PORT}"
echo "====================================================================="

POLICY_EVAL_ARGS=(
  python scripts/lqr/run_libero_policy_eval.py
  --config-name "${CONFIG_NAME}"
  --libero-benchmark "${LIBERO_BENCHMARK}"
  --task-ids ${TASK_IDS}
  --num-episodes "${NUM_EPISODES}"
  --perturb-spec "${PERTURB_SPEC}"
  --agentview-noise-sigma "${AGENTVIEW_NOISE_SIGMA}"
  --agentview-noise-seed-base "${AGENTVIEW_NOISE_SEED_BASE}"
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
