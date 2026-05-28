#!/bin/bash
#SBATCH -J lingbot_lqr_init_pos_time_noPar
#SBATCH -A gts-gchou3-ideasci23_dgx
#SBATCH -N 1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -o Lqr_InitPos_TimeNoPar-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qdai41@gatech.edu
#
# Init-position LQR branch with:
#   - denoise steps 5,10,...,50 in human 1-based indexing
#   - no layer partition pooling: one SVD basis per LingBot layer
#
# LingBot hook indices are 0-based, so this runs
# SELECTED_TIMESTEPS=4,9,14,19,24,29,34,39,44,49.

source ~/.bashrc
set -euo pipefail

conda activate "${CONDA_ENV:-lingbot}"
cd /storage/home/hcoda1/9/qdai41/scratch/cosmos/LingBot-VA-Modification

export PYTHONPATH=.

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
DEFAULT_MASTER_PORT="$((12000 + (${SLURM_JOB_ID:-0} % 20000)))"
export MASTER_PORT="${MASTER_PORT:-${DEFAULT_MASTER_PORT}}"
export RANK="${RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export LOCAL_RANK="${LOCAL_RANK:-0}"

export CONFIG_NAME="${CONFIG_NAME:-libero}"
export LIBERO_BENCHMARK="${LIBERO_BENCHMARK:-libero_10}"
export TASK_ID="${TASK_ID:-7}"
export NUM_EPISODES="${NUM_EPISODES:-50}"
export PERTURB_SPEC="${PERTURB_SPEC:-scripts/lqr/configs/perturb_spec_init_pos.yaml}"
export PAIR_INIT_BY_SIMILARITY="${PAIR_INIT_BY_SIMILARITY:-1}"
export PAIR_FEATURE="${PAIR_FEATURE:-proprio}"
export PAIR_MATCH_MODE="${PAIR_MATCH_MODE:-nn-greedy}"
export PAIR_MAX_ROWS="${PAIR_MAX_ROWS:--1}"
export PAIR_MAX_DISTANCE="${PAIR_MAX_DISTANCE:--1}"

export EXISTING_PAIRS_ALL_DIR="${EXISTING_PAIRS_ALL_DIR:-outputs/lqr/pairs_init_pos_20260528_012010__paired}"
export EXISTING_COLLECT_DIR="${EXISTING_COLLECT_DIR:-}"
export EXISTING_SVD_DIR="${EXISTING_SVD_DIR:-}"
export NUM_SAMPLES="${NUM_SAMPLES:--1}"
export PROMPT="${PROMPT:-put both the alphabet soup and the cream cheese box in the basket}"

export TIME_BRANCH_HUMAN_TIMESTEPS="${TIME_BRANCH_HUMAN_TIMESTEPS:-5,10,15,20,25,30,35,40,45,50}"
export SELECTED_TIMESTEPS="${SELECTED_TIMESTEPS:-4,9,14,19,24,29,34,39,44,49}"
export COLLECT_MODE="${COLLECT_MODE:-action}"
export K_TARGET="${K_TARGET:-64}"
export P_OVER="${P_OVER:-10}"
export PARTITIONS="${PARTITIONS:-0-0,1-1,2-2,3-3,4-4,5-5,6-6,7-7,8-8,9-9,10-10,11-11,12-12,13-13,14-14,15-15,16-16,17-17,18-18,19-19,20-20,21-21,22-22,23-23,24-24,25-25,26-26,27-27,28-28,29-29}"
export JAC_METHOD="${JAC_METHOD:-vjp}"
export JAC_OBS_INDEX="${JAC_OBS_INDEX:-0}"
export JAC_NUM_SHARDS="${JAC_NUM_SHARDS:-1}"
export JAC_SUBDIR="${JAC_SUBDIR:-A_tilde_lingbot_time_noPar}"
export LQR_CONFIG="${LQR_CONFIG:-scripts/lqr/configs/lqr_config.yaml}"

export OUT_BASE="${OUT_BASE:-outputs/lqr_time_noPar}"
export EVAL_OUT_BASE="${EVAL_OUT_BASE:-outputs/lqr_eval_time_noPar_init_pos_${SLURM_JOB_ID:-local}}"

export TASK_RANGE_START="${TASK_RANGE_START:-7}"
export TASK_RANGE_END="${TASK_RANGE_END:-8}"
export EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-20}"
source scripts/lqr/slurm_port.sh
export EVAL_STARTUP_WAIT_SEC="${EVAL_STARTUP_WAIT_SEC:-1200}"
export INJECT_MODE="${INJECT_MODE:-auto}"
export SKIP_EVAL="${SKIP_EVAL:-0}"

echo "Running LingBot LQR init-position time no-partition pipeline"
echo "  REPO=$(pwd)"
echo "  TASK_ID=${TASK_ID}"
echo "  NUM_EPISODES=${NUM_EPISODES}"
echo "  PERTURB_SPEC=${PERTURB_SPEC}"
echo "  EXISTING_COLLECT_DIR=${EXISTING_COLLECT_DIR:-<none>}"
echo "  EXISTING_PAIRS_ALL_DIR=${EXISTING_PAIRS_ALL_DIR:-<none>}"
echo "  EXISTING_SVD_DIR=${EXISTING_SVD_DIR:-<none>}"
echo "  TIME_BRANCH_HUMAN_TIMESTEPS=${TIME_BRANCH_HUMAN_TIMESTEPS}"
echo "  SELECTED_TIMESTEPS=${SELECTED_TIMESTEPS} (LingBot 0-based hook indices)"
echo "  PARTITIONS=${PARTITIONS} (one SVD basis per layer)"
echo "  NUM_SAMPLES=${NUM_SAMPLES}"
echo "  PROMPT=${PROMPT}"
echo "  JAC_SUBDIR=${JAC_SUBDIR}"
echo "  OUT_BASE=${OUT_BASE}"
echo "  EVAL_OUT_BASE=${EVAL_OUT_BASE}"
echo "  TASK_RANGE=[${TASK_RANGE_START}, ${TASK_RANGE_END})"
echo "  SKIP_EVAL=${SKIP_EVAL}"
echo "  PORT=${PORT} (job ${SLURM_JOB_ID:-local})"

export ACTIVATE_CONDA=0
bash scripts/lqr/run_lqr_pipeline.sh
