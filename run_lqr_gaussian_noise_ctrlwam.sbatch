#!/bin/bash
#SBATCH -J lingbot_lqr_noise
#SBATCH -A gts-gchou3-ideas_l40s
#SBATCH -q inferno
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40s:1
#SBATCH -t 24:00:00
#SBATCH -o Lqr_GaussianNoise-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qdai41@gatech.edu
#
# ctrlwam run_noise_extreme_pipeline.sh (LingBot per-task variant):
#   noise_extreme: sigma=90 on agentview + wrist
#   collect: noise_seed_base=0  -> rng(seed=episode_idx)
#   eval:    NOISE_SEED_BASE=99 -> rng(seed=99+episode_idx)
#
# Default: collect pairs + SVD + Jacobian + LQR eval on libero_10 task 6.
#
# Resume job 9240981 (skip collect + SVD; rerun Jacobian + eval):
#   sbatch run_lqr_gaussian_noise_ctrlwam.sbatch
# Fresh collect:
#   sbatch --export=ALL,EXISTING_COLLECT_DIR=,EXISTING_SVD_DIR= run_lqr_gaussian_noise_ctrlwam.sbatch
#
# Examples:
#   sbatch run_lqr_gaussian_noise_ctrlwam.sbatch
#   sbatch --export=ALL,SKIP_EVAL=1 run_lqr_gaussian_noise_ctrlwam.sbatch
#   sbatch --export=ALL,NUM_EPISODES=20,EVAL_NUM_EPISODES=50 run_lqr_gaussian_noise_ctrlwam.sbatch

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

# Collect row-aligned clean/noisy pairs on this task (ctrlwam task0 uses 10 for SVD).
export TASK_ID="${TASK_ID:-6}"
export NUM_EPISODES="${NUM_EPISODES:-10}"
export PERTURB_SPEC="${PERTURB_SPEC:-scripts/lqr/configs/perturb_spec_gaussian.yaml}"
export PAIR_INIT_BY_SIMILARITY="${PAIR_INIT_BY_SIMILARITY:-0}"

# Row-aligned clean/noisy pairs from job 9233555 (gaussian skips extra pairing).
export EXISTING_COLLECT_DIR="${EXISTING_COLLECT_DIR:-outputs/lqr/pairs_gaussian_20260528_022256}"
export EXISTING_SVD_DIR="${EXISTING_SVD_DIR:-outputs/lqr/svd_all_perturb_gaussian_20260528_084735}"
export NUM_SAMPLES="${NUM_SAMPLES:--1}"
export JAC_RESUME="${JAC_RESUME:-0}"

export SELECTED_TIMESTEPS="${SELECTED_TIMESTEPS:-0,10,20,30,40}"
export COLLECT_MODE="${COLLECT_MODE:-action}"
export K_TARGET="${K_TARGET:-64}"
export P_OVER="${P_OVER:-10}"
export PARTITIONS="${PARTITIONS:-}"
export JAC_METHOD="${JAC_METHOD:-vjp}"
export JAC_OBS_INDEX="${JAC_OBS_INDEX:-0}"
export JAC_NUM_SHARDS="${JAC_NUM_SHARDS:-1}"
export JAC_SUBDIR="${JAC_SUBDIR:-A_tilde_lingbot}"
export LQR_CONFIG="${LQR_CONFIG:-scripts/lqr/configs/lqr_config.yaml}"

# LQR eval on the same task as collect.
export TASK_RANGE_START="${TASK_RANGE_START:-6}"
export TASK_RANGE_END="${TASK_RANGE_END:-7}"
export EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-20}"

# ctrlwam run_sweep_noise_extreme_seed99.sh eval rollouts.
export AGENTVIEW_NOISE_SIGMA="${AGENTVIEW_NOISE_SIGMA:-90}"
export AGENTVIEW_NOISE_SEED_BASE="${AGENTVIEW_NOISE_SEED_BASE:-99}"

source scripts/lqr/slurm_port.sh
export EVAL_STARTUP_WAIT_SEC="${EVAL_STARTUP_WAIT_SEC:-1200}"
export INJECT_MODE="${INJECT_MODE:-auto}"
export SKIP_EVAL="${SKIP_EVAL:-0}"
export PROMPT="${PROMPT:-}"

echo "Running LingBot LQR gaussian noise ctrlwam-style pipeline"
echo "  REPO=$(pwd)"
echo "  TASK_ID=${TASK_ID} (collect)"
echo "  NUM_EPISODES=${NUM_EPISODES}"
echo "  PERTURB_SPEC=${PERTURB_SPEC}"
echo "  EXISTING_COLLECT_DIR=${EXISTING_COLLECT_DIR:-<none>}"
echo "  EXISTING_SVD_DIR=${EXISTING_SVD_DIR:-<none>}"
echo "  NUM_SAMPLES=${NUM_SAMPLES}"
echo "  JAC_RESUME=${JAC_RESUME}"
echo "  noise_extreme: sigma=${AGENTVIEW_NOISE_SIGMA}"
echo "  eval AGENTVIEW_NOISE_SEED_BASE=${AGENTVIEW_NOISE_SEED_BASE}"
echo "  PARTITIONS=${PARTITIONS:-<auto 3-partition>}"
echo "  JAC_METHOD=${JAC_METHOD}"
echo "  TASK_RANGE=[${TASK_RANGE_START}, ${TASK_RANGE_END}) (eval)"
echo "  EVAL_NUM_EPISODES=${EVAL_NUM_EPISODES}"
echo "  SKIP_EVAL=${SKIP_EVAL}"
echo "  PORT=${PORT} (job ${SLURM_JOB_ID:-local})"

export ACTIVATE_CONDA=0
bash scripts/lqr/run_lqr_pipeline.sh
