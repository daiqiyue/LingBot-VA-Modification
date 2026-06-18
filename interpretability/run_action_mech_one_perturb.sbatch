#!/bin/bash
#SBATCH -J ling_action_mech_fig
#SBATCH --gres=gpu:h200:1
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH -t 06:00:00
#SBATCH -o LingActionMechFigure-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qdai41@gatech.edu

source ~/.bashrc
set -euo pipefail

CONDA_ENV_PATH="${CONDA_ENV_PATH:-/storage/scratch1/9/qdai41/.conda/envs/lingbot}"
conda activate "${CONDA_ENV_PATH}"

REPO_ROOT="${REPO_ROOT:-/storage/home/hcoda1/9/qdai41/scratch/cosmos/LingBot-VA-Modification}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

: "${FIGURE_ROOT:?Set FIGURE_ROOT to the figure output root}"
: "${NOMINAL_RUN_DIR:?Set NOMINAL_RUN_DIR to the nominal collect run directory}"
: "${PERTURB_RUN_DIR:?Set PERTURB_RUN_DIR to the perturb collect run directory}"
: "${PERTURB_NAME:?Set PERTURB_NAME to camera, init_pos, or gaussian}"

TASKS="${TASKS:-0 1 2 3 4 5 6 7 8 9}"
mkdir -p "${FIGURE_ROOT}"

echo "===== action mechanistic figure worker ====="
echo "perturb      : ${PERTURB_NAME}"
echo "tasks        : ${TASKS}"
echo "nominal      : ${NOMINAL_RUN_DIR}"
echo "perturb dir  : ${PERTURB_RUN_DIR}"
echo "figure root  : ${FIGURE_ROOT}"

echo "===== rebuilding manifests from any recovered trajectory records ====="
python interpretability/rebuild_collect_manifests.py --root "${NOMINAL_RUN_DIR}"
python interpretability/rebuild_collect_manifests.py --root "${PERTURB_RUN_DIR}"

CACHE_PATH="${FIGURE_ROOT}/activations_dict_action_${PERTURB_NAME}.npz"
INTERPRET_OUT="${FIGURE_ROOT}/interpret_output_action_${PERTURB_NAME}"
FINAL_OUT="${FIGURE_ROOT}/final_fig_action_${PERTURB_NAME}"

echo "===== ${PERTURB_NAME}: interpret_ling.py ====="
echo "cache        : ${CACHE_PATH}"
echo "interpret out: ${INTERPRET_OUT}"

NOMINAL_RUN_DIR="${NOMINAL_RUN_DIR}" \
PERTURB_RUN_DIR="${PERTURB_RUN_DIR}" \
TASKS="${TASKS}" \
ACTIVATIONS_PATH="${CACHE_PATH}" \
OUTPUT_DIR="${INTERPRET_OUT}" \
  python interpretability/interpret_ling.py

echo "===== ${PERTURB_NAME}: make_fig_final.py ====="
echo "final out    : ${FINAL_OUT}"

TASKS="${TASKS}" \
CAMERA_TASKS="${TASKS}" \
ACTIVATIONS_PATH="${CACHE_PATH}" \
OUTPUT_DIR="${FINAL_OUT}" \
  python interpretability/make_fig_final.py

echo "===== ${PERTURB_NAME}: done ====="
