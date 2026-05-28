#!/bin/bash
#SBATCH -J lingbot_lqr_camera
#SBATCH -A gts-gchou3
#SBATCH -N 1
#SBATCH --gres=gpu:h200:1
#SBATCH -t 20:00:00
#SBATCH -o Lqr_Camera_Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qdai41@gatech.edu

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
export NUM_EPISODES="${NUM_EPISODES:-10}"
export N_POS="${N_POS:-10}"
export N_NEG="${N_NEG:-10}"
export SELECTED_TIMESTEPS="${SELECTED_TIMESTEPS:-0,10,20,30,40}"
export COLLECT_MODE="${COLLECT_MODE:-action}"
export PERTURB_SPEC="${PERTURB_SPEC:-scripts/lqr/configs/perturb_spec_camera.yaml}"
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
export TASK_RANGE_END="${TASK_RANGE_END:-2}"
export EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-20}"
source scripts/lqr/slurm_port.sh
export EVAL_STARTUP_WAIT_SEC="${EVAL_STARTUP_WAIT_SEC:-1200}"
export INJECT_MODE="${INJECT_MODE:-auto}"
export SKIP_EVAL="${SKIP_EVAL:-0}"
export PROMPT="${PROMPT:-}"

echo "Running CAMERA-LQR pipeline with:"
echo "  REPO=$(pwd)"
echo "  PERTURB_SPEC=${PERTURB_SPEC}"
echo "  N_POS=${N_POS}"
echo "  N_NEG=${N_NEG}"
echo "  PARTITIONS=${PARTITIONS:-<auto 3-partition>}"
echo "  JAC_METHOD=${JAC_METHOD}"
echo "  TASK_RANGE=[${TASK_RANGE_START}, ${TASK_RANGE_END})"
echo "  SKIP_EVAL=${SKIP_EVAL}"

export ACTIVATE_CONDA=0
bash scripts/lqr/run_lqr_pipeline.sh
