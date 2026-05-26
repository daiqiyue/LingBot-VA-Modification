#!/bin/bash
#SBATCH -J lingbot_lqr_eval_camera
#SBATCH -A gts-gchou3-ideas_l40s
#SBATCH -q inferno
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40s:1
#SBATCH -t 20:00:00
#SBATCH -o Lqr_eval_camera_Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qdai41@gatech.edu

source ~/.bashrc
set -euo pipefail

conda activate lingbot
cd /storage/home/hcoda1/9/qdai41/scratch/cosmos/lingbot-va

export PYTHONPATH=.

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
DEFAULT_MASTER_PORT="$((12000 + (${SLURM_JOB_ID:-0} % 20000)))"
export MASTER_PORT="${MASTER_PORT:-${DEFAULT_MASTER_PORT}}"
export RANK="${RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export LOCAL_RANK="${LOCAL_RANK:-0}"

export STARTUP_WAIT_SEC="${STARTUP_WAIT_SEC:-1200}"

echo "LQR camera eval startup wait: ${STARTUP_WAIT_SEC}s"
echo "DDP env: MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} RANK=${RANK} WORLD_SIZE=${WORLD_SIZE} LOCAL_RANK=${LOCAL_RANK}"

# Override SVD_DIR via: sbatch --export=ALL,SVD_DIR=outputs/lqr/your_camera_svd ...
python scripts/lqr/run_libero_lqr_eval.py \
  --config-name libero \
  --libero-benchmark libero_10 \
  --task-range 0 2 \
  --num-episodes 10 \
  --startup-wait-sec "${STARTUP_WAIT_SEC}" \
  --port 29056 \
  --svd-dir "${SVD_DIR:-outputs/lqr/svd_all_perturb_camera_pos_3_20260525_012831}" \
  --jac-dir-act A_tilde_lingbot \
  --lqr-config scripts/lqr/configs/lqr_config.yaml \
  --perturb-spec scripts/lqr/configs/eval_camera.yaml \
  --out-dir "${EVAL_OUT_DIR:-outputs/lqr_eval_camera_pos_3_20260525_012831}"
