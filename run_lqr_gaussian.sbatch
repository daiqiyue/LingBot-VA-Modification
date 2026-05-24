#!/bin/bash
#SBATCH -J lingbot_lqr_gaussian
#SBATCH -A gts-gchou3
#SBATCH -N 1
#SBATCH --gres=gpu:h200:1
#SBATCH -t 20:00:00
#SBATCH -o Lqr_Gaussian_Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qdai41@gatech.edu

source ~/.bashrc
set -euo pipefail

conda activate lingbot
cd /storage/home/hcoda1/9/qdai41/scratch/cosmos/lingbot-va

export PYTHONPATH=.

# ---- Pipeline defaults (override via sbatch --export or env) ----
export CONFIG_NAME="${CONFIG_NAME:-libero}"
export LIBERO_BENCHMARK="${LIBERO_BENCHMARK:-libero_10}"
export TASK_ID="${TASK_ID:-0}"
export NUM_EPISODES="${NUM_EPISODES:-10}"
export TOPK_INFER_PER_TRAJ="${TOPK_INFER_PER_TRAJ:-10}"
export SELECTED_TIMESTEPS="${SELECTED_TIMESTEPS:-0,10,20,30,40}"
export COLLECT_MODE="${COLLECT_MODE:-action}"
export PERTURB_SPEC="${PERTURB_SPEC:-scripts/lqr/configs/perturb_spec_gaussian.yaml}"
export TARGET_VARIANTS="${TARGET_VARIANTS:-}"
export PAIR_SEED="${PAIR_SEED:-0}"
export NUM_SAMPLES="${NUM_SAMPLES:-200}"
export K_TARGET="${K_TARGET:-32}"
export P_OVER="${P_OVER:-8}"
export RIDGE="${RIDGE:-1e-3}"
export JAC_SUBDIR="${JAC_SUBDIR:-A_tilde_lingbot}"
export LQR_CONFIG="${LQR_CONFIG:-scripts/lqr/configs/lqr_config.yaml}"
export TASK_RANGE_START="${TASK_RANGE_START:-0}"
export TASK_RANGE_END="${TASK_RANGE_END:-2}"
export EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-10}"
export PORT="${PORT:-29056}"
export SKIP_EVAL="${SKIP_EVAL:-0}"

# Optional prompt override for eval
export PROMPT="${PROMPT:-}"

echo "Running GAUSSIAN-LQR pipeline with:"
echo "  PERTURB_SPEC=${PERTURB_SPEC}"
echo "  NUM_EPISODES=${NUM_EPISODES}, TOPK=${TOPK_INFER_PER_TRAJ}"
echo "  SELECTED_TIMESTEPS=${SELECTED_TIMESTEPS}"
echo "  TASK_RANGE=[${TASK_RANGE_START}, ${TASK_RANGE_END})"
echo "  SKIP_EVAL=${SKIP_EVAL}"

bash scripts/lqr/run_lqr_pipeline.sh
