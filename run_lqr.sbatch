#!/bin/bash
#SBATCH -J lingbot_lqr
#SBATCH -A gts-gchou3
#SBATCH -N 1
#SBATCH --gres=gpu:h200:1
#SBATCH -t 20:00:00
#SBATCH -o Lqr_Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qdai41@gatech.edu

source ~/.bashrc
set -euo pipefail

conda activate lingbot
cd /storage/home/hcoda1/9/qdai41/scratch/cosmos/lingbot-va

export PYTHONPATH=.
TS="$(date +%Y%m%d_%H%M%S)"
PAIRS_DIR="outputs/lqr/pairs_${TS}"
PAIRS_ALL_DIR="outputs/lqr/pairs_all_${TS}"
SVD_DIR="outputs/lqr/svd_all_perturb_${TS}"
EVAL_DIR="outputs/lqr_eval_all_perturb_${TS}"

python scripts/lqr/run_collect_inputs.py \
  --libero-benchmark libero_10 \
  --task-id 0 \
  --num-samples 10 \
  --perturb-spec scripts/lqr/configs/perturb_spec_init_pos.yaml \
  --mode both \
  --out-dir "${PAIRS_DIR}"

python scripts/lqr/build_all_pairs.py \
  --collect-dir "${PAIRS_DIR}" \
  --out-dir "${PAIRS_ALL_DIR}"

python scripts/lqr/run_partition_svd.py \
  --pairs-dir "${PAIRS_ALL_DIR}" \
  --out-dir "${SVD_DIR}" \
  --config-name libero \
  --mode action \
  --selected-timesteps 0,10,20,30,40 \
  --num-samples 40 \
  --k-target 32

python scripts/lqr/run_compute_jacobians.py \
  --svd-dir "${SVD_DIR}" \
  --out-subdir A_tilde_lingbot


python scripts/lqr/run_libero_lqr_eval.py \
  --config-name libero \
  --libero-benchmark libero_10 \
  --task-range 0 2 \
  --num-episodes 10 \
  --port 29056 \
  --svd-dir "${SVD_DIR}" \
  --jac-dir-act A_tilde_lingbot \
  --lqr-config scripts/lqr/configs/lqr_config.yaml \
  --perturb-spec scripts/lqr/configs/perturb_spec_init_pos.yaml \
  --out-dir "${EVAL_DIR}"