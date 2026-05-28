#!/bin/bash
#SBATCH -J lingbot_lqr_camera_time_noPar
#SBATCH -A gts-gchou3-ideasci23_dgx
#SBATCH -N 1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -o Lqr_Camera_TimeNoPar-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qdai41@gatech.edu
#
# Camera-view LQR branch with:
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

export TASK_ID="${TASK_ID:-0}"
export PROMPT="${PROMPT:-put both the alphabet soup and the tomato sauce in the basket}"
export PERTURB_SPEC="${PERTURB_SPEC:-scripts/lqr/configs/perturb_spec_camera.yaml}"

export CAM_BASE_SEED="${CAM_BASE_SEED:-42}"
export EVAL_CAM_BASE_SEED="${EVAL_CAM_BASE_SEED:-99}"
export N_POS="${N_POS:-10}"
export N_NEG="${N_NEG:-10}"

export TIME_BRANCH_HUMAN_TIMESTEPS="${TIME_BRANCH_HUMAN_TIMESTEPS:-5,10,15,20,25,30,35,40,45,50}"
export SELECTED_TIMESTEPS="${SELECTED_TIMESTEPS:-4,9,14,19,24,29,34,39,44,49}"
export COLLECT_MODE="${COLLECT_MODE:-action}"
export NUM_SAMPLES="${NUM_SAMPLES:--1}"
export K_TARGET="${K_TARGET:-64}"
export P_OVER="${P_OVER:-10}"
export PARTITIONS="${PARTITIONS:-0-0,1-1,2-2,3-3,4-4,5-5,6-6,7-7,8-8,9-9,10-10,11-11,12-12,13-13,14-14,15-15,16-16,17-17,18-18,19-19,20-20,21-21,22-22,23-23,24-24,25-25,26-26,27-27,28-28,29-29}"

export JAC_METHOD="${JAC_METHOD:-vjp}"
export JAC_OBS_INDEX="${JAC_OBS_INDEX:-0}"
export JAC_NUM_SHARDS="${JAC_NUM_SHARDS:-1}"
export JAC_SUBDIR="${JAC_SUBDIR:-A_tilde_lingbot_time_noPar}"

export OUT_BASE="${OUT_BASE:-outputs/lqr_time_noPar}"
export LQR_CONFIG="${LQR_CONFIG:-scripts/lqr/configs/lqr_config.yaml}"
export TASK_RANGE_START="${TASK_RANGE_START:-0}"
export TASK_RANGE_END="${TASK_RANGE_END:-1}"
export EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-20}"
export EVAL_OUT_BASE="${EVAL_OUT_BASE:-outputs/lqr_eval_time_noPar_camera_${SLURM_JOB_ID:-local}}"
source scripts/lqr/slurm_port.sh
export EVAL_STARTUP_WAIT_SEC="${EVAL_STARTUP_WAIT_SEC:-1200}"
export INJECT_MODE="${INJECT_MODE:-auto}"

export START_AT="${START_AT:-1}"
export SUBMIT_EVAL="${SUBMIT_EVAL:-1}"

echo "Running LingBot LQR camera time no-partition pipeline"
echo "  REPO=$(pwd)"
echo "  TASK_ID=${TASK_ID}"
echo "  PROMPT=${PROMPT}"
echo "  PERTURB_SPEC=${PERTURB_SPEC}"
echo "  CAM_BASE_SEED=${CAM_BASE_SEED} (collect)"
echo "  EVAL_CAM_BASE_SEED=${EVAL_CAM_BASE_SEED} (eval)"
echo "  N_POS=${N_POS}"
echo "  N_NEG=${N_NEG}"
echo "  TIME_BRANCH_HUMAN_TIMESTEPS=${TIME_BRANCH_HUMAN_TIMESTEPS}"
echo "  SELECTED_TIMESTEPS=${SELECTED_TIMESTEPS} (LingBot 0-based hook indices)"
echo "  PARTITIONS=${PARTITIONS} (one SVD basis per layer)"
echo "  NUM_SAMPLES=${NUM_SAMPLES}"
echo "  K_TARGET=${K_TARGET}"
echo "  P_OVER=${P_OVER}"
echo "  JAC_SUBDIR=${JAC_SUBDIR}"
echo "  OUT_BASE=${OUT_BASE}"
echo "  EVAL_OUT_BASE=${EVAL_OUT_BASE}"
echo "  TASK_RANGE=[${TASK_RANGE_START}, ${TASK_RANGE_END})"
echo "  EVAL_NUM_EPISODES=${EVAL_NUM_EPISODES}"
echo "  START_AT=${START_AT}"
echo "  SUBMIT_EVAL=${SUBMIT_EVAL}"
echo "  PORT=${PORT} (job ${SLURM_JOB_ID:-local})"

export ACTIVATE_CONDA=0
bash scripts/lqr/run_cam_random_large_pipeline.sh
