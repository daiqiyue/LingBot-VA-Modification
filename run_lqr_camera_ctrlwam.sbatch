#!/bin/bash
#SBATCH -J lingbot_lqr_camera
#SBATCH -A gts-gchou3-ideas_l40s
#SBATCH -N 1
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -o Lqr_Camera-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qdai41@gatech.edu
#
# ctrlwam run_cam_random_large_pipeline.sh (LingBot variant):
#   collect: cam_random_large, base_seed=42, N_POS=N_NEG=10
#   SVD:     action mode, all paired rows, k=64, p=10
#   jac:     VJP from negative.npz row 0
#   eval:    camera perturbation with base_seed=99
#
# Examples:
#   sbatch run_lqr_camera_ctrlwam.sbatch
#   sbatch --export=ALL,SUBMIT_EVAL=0 run_lqr_camera_ctrlwam.sbatch
#   sbatch --export=ALL,N_POS=1,N_NEG=1,K_TARGET=1,SUBMIT_EVAL=0 run_lqr_camera_ctrlwam.sbatch
#   sbatch --export=ALL,START_AT=2,PAIRS_DIR=outputs/lqr/pairs_cam_random_large_seed42_... run_lqr_camera_ctrlwam.sbatch

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

# Canonical ctrlwam camera-position pipeline task.
export TASK_ID="${TASK_ID:-0}"
export PROMPT="${PROMPT:-put both the alphabet soup and the tomato sauce in the basket}"
export PERTURB_SPEC="${PERTURB_SPEC:-scripts/lqr/configs/perturb_spec_camera.yaml}"

# cam_random_large collection uses seed 42; eval sweep convention uses seed 99.
export CAM_BASE_SEED="${CAM_BASE_SEED:-42}"
export EVAL_CAM_BASE_SEED="${EVAL_CAM_BASE_SEED:-99}"
export N_POS="${N_POS:-10}"
export N_NEG="${N_NEG:-10}"

export SELECTED_TIMESTEPS="${SELECTED_TIMESTEPS:-0,10,20,30,40}"
export COLLECT_MODE="${COLLECT_MODE:-action}"
export NUM_SAMPLES="${NUM_SAMPLES:--1}"
export K_TARGET="${K_TARGET:-64}"
export P_OVER="${P_OVER:-10}"
export PARTITIONS="${PARTITIONS:-}"

export JAC_METHOD="${JAC_METHOD:-vjp}"
export JAC_OBS_INDEX="${JAC_OBS_INDEX:-0}"
export JAC_NUM_SHARDS="${JAC_NUM_SHARDS:-1}"
export JAC_SUBDIR="${JAC_SUBDIR:-A_tilde_lingbot}"

export LQR_CONFIG="${LQR_CONFIG:-scripts/lqr/configs/lqr_config.yaml}"
export TASK_RANGE_START="${TASK_RANGE_START:-0}"
export TASK_RANGE_END="${TASK_RANGE_END:-1}"
export EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-20}"
source scripts/lqr/slurm_port.sh
export EVAL_STARTUP_WAIT_SEC="${EVAL_STARTUP_WAIT_SEC:-1200}"
export INJECT_MODE="${INJECT_MODE:-auto}"

# Wrapper controls.
export START_AT="${START_AT:-1}"
export SUBMIT_EVAL="${SUBMIT_EVAL:-1}"

echo "Running LingBot LQR camera ctrlwam-style pipeline"
echo "  REPO=$(pwd)"
echo "  TASK_ID=${TASK_ID}"
echo "  PROMPT=${PROMPT}"
echo "  PERTURB_SPEC=${PERTURB_SPEC}"
echo "  CAM_BASE_SEED=${CAM_BASE_SEED} (collect)"
echo "  EVAL_CAM_BASE_SEED=${EVAL_CAM_BASE_SEED} (eval)"
echo "  N_POS=${N_POS}"
echo "  N_NEG=${N_NEG}"
echo "  NUM_SAMPLES=${NUM_SAMPLES}"
echo "  K_TARGET=${K_TARGET}"
echo "  P_OVER=${P_OVER}"
echo "  PARTITIONS=${PARTITIONS:-<auto 3-partition>}"
echo "  JAC_METHOD=${JAC_METHOD}"
echo "  TASK_RANGE=[${TASK_RANGE_START}, ${TASK_RANGE_END})"
echo "  EVAL_NUM_EPISODES=${EVAL_NUM_EPISODES}"
echo "  START_AT=${START_AT}"
echo "  SUBMIT_EVAL=${SUBMIT_EVAL}"
echo "  PORT=${PORT} (job ${SLURM_JOB_ID:-local})"

export ACTIVATE_CONDA=0
bash scripts/lqr/run_cam_random_large_pipeline.sh
