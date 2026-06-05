#!/bin/bash
#SBATCH -J lqr_eval_init_tnp_sel
#SBATCH -A gts-gchou3-ideas_l40s
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40s:1
#SBATCH -t 20:00:00
#SBATCH -o Lqr_eval_init_pos_time_noPar_selected-%j.out
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

export STARTUP_WAIT_SEC="${STARTUP_WAIT_SEC:-1200}"
source scripts/lqr/slurm_port.sh

SVD_DIR="${SVD_DIR:-outputs/lqr_time_noPar/svd_all_perturb_init_pos_20260528_113245}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-outputs/lqr_eval_time_noPar_init_pos_9274403}"
TASK_IDS=(${TASK_IDS:-1 2 3 7 9})
NUM_EPISODES="${NUM_EPISODES:-20}"
GRIPPER_XYZ_BASE_SEED="${GRIPPER_XYZ_BASE_SEED:-99}"

echo "Running selected init-position time-noPar LQR eval"
echo "  SVD_DIR=${SVD_DIR}"
echo "  TASK_IDS=${TASK_IDS[*]}"
echo "  NUM_EPISODES=${NUM_EPISODES}"
echo "  GRIPPER_XYZ_BASE_SEED=${GRIPPER_XYZ_BASE_SEED}"
echo "  EVAL_OUT_DIR=${EVAL_OUT_DIR}"
echo "  PORT=${PORT} (job ${SLURM_JOB_ID:-local})"

python scripts/lqr/run_libero_lqr_eval.py \
  --config-name libero \
  --libero-benchmark libero_10 \
  --task-ids "${TASK_IDS[@]}" \
  --num-episodes "${NUM_EPISODES}" \
  --startup-wait-sec "${STARTUP_WAIT_SEC}" \
  --port "${PORT}" \
  --svd-dir "${SVD_DIR}" \
  --jac-dir-act A_tilde_lingbot_time_noPar \
  --lqr-config scripts/lqr/configs/lqr_config.yaml \
  --perturb-spec scripts/lqr/configs/perturb_spec_init_pos.yaml \
  --gripper-xyz-base-seed "${GRIPPER_XYZ_BASE_SEED}" \
  --out-dir "${EVAL_OUT_DIR}" \
  --resume
