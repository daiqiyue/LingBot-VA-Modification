#!/bin/bash
#SBATCH -J lqr_eval_init_sel
#SBATCH -A gts-gchou3-ideas_l40s
#SBATCH -q inferno
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40s:1
#SBATCH -t 20:00:00
#SBATCH -o Lqr_eval_init_pos_selected-%j.out
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

SVD_DIR="${SVD_DIR:-outputs/lqr/svd_all_perturb_init_pos_20260528_104511}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-outputs/lqr_eval_selected_init_pos_20260528_104511_no_title}"

echo "Running selected init-position LQR eval"
echo "  SVD_DIR=${SVD_DIR}"
echo "  TASK_IDS=1 2 3 7 9"
echo "  NUM_EPISODES=20"
echo "  EVAL_OUT_DIR=${EVAL_OUT_DIR}"
echo "  PORT=${PORT} (job ${SLURM_JOB_ID:-local})"

python scripts/lqr/run_libero_lqr_eval.py \
  --config-name libero \
  --libero-benchmark libero_10 \
  --task-ids 1 2 3 7 9 \
  --num-episodes 10 \
  --startup-wait-sec "${STARTUP_WAIT_SEC}" \
  --port "${PORT}" \
  --svd-dir "${SVD_DIR}" \
  --jac-dir-act A_tilde_lingbot \
  --lqr-config scripts/lqr/configs/lqr_config.yaml \
  --perturb-spec scripts/lqr/configs/perturb_spec_init_pos.yaml \
  --out-dir "${EVAL_OUT_DIR}" \
  --resume
