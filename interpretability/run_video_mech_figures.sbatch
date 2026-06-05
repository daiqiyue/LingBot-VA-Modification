#!/bin/bash
#SBATCH -J ling_video_mech_figs
#SBATCH --gres=gpu:h200:1
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -o LingVideoMechFigures-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qdai41@gatech.edu

source ~/.bashrc
set -euo pipefail

CONDA_ENV_PATH="${CONDA_ENV_PATH:-/storage/scratch1/9/qdai41/.conda/envs/lingbot}"
conda activate "${CONDA_ENV_PATH:-${CONDA_ENV:-lingbot}}"

REPO_ROOT="${REPO_ROOT:-/storage/home/hcoda1/9/qdai41/scratch/cosmos/LingBot-VA-Modification}"
cd "${REPO_ROOT}"

export PYTHONPATH=.
FIGURE_ROOT="${FIGURE_ROOT:-/storage/scratch1/9/qdai41/cosmos/LingBot-VA-Modification/outputs/video_mech_figures_manual}"
mkdir -p "${FIGURE_ROOT}"

: "${NOMINAL_RUN_DIR:?Set NOMINAL_RUN_DIR to the nominal collect run directory}"
: "${CAMERA_RUN_DIR:?Set CAMERA_RUN_DIR to the camera perturb collect run directory}"
: "${INIT_POS_RUN_DIR:?Set INIT_POS_RUN_DIR to the init-position perturb collect run directory}"
: "${GAUSSIAN_RUN_DIR:?Set GAUSSIAN_RUN_DIR to the gaussian perturb collect run directory}"

run_one() {
  local name="$1"
  local tasks="$2"
  local perturb_dir="$3"
  local cache_path="$4"
  local interpret_out="$5"
  local final_out="$6"

  echo "===== ${name}: interpret_ling.py ====="
  echo "  tasks       : ${tasks}"
  echo "  nominal     : ${NOMINAL_RUN_DIR}"
  echo "  perturb     : ${perturb_dir}"
  echo "  cache       : ${cache_path}"
  echo "  interpret   : ${interpret_out}"
  echo "  final       : ${final_out}"

  NOMINAL_RUN_DIR="${NOMINAL_RUN_DIR}" \
  PERTURB_RUN_DIR="${perturb_dir}" \
  TASKS="${tasks}" \
  ACTIVATIONS_PATH="${cache_path}" \
  OUTPUT_DIR="${interpret_out}" \
    python interpretability/interpret_ling.py

  echo "===== ${name}: make_fig_final.py ====="
  TASKS="${tasks}" \
  CAMERA_TASKS="${tasks}" \
  ACTIVATIONS_PATH="${cache_path}" \
  OUTPUT_DIR="${final_out}" \
    python interpretability/make_fig_final.py
}

run_one \
  "camera" \
  "0 2 4 5 9" \
  "${CAMERA_RUN_DIR}" \
  "${FIGURE_ROOT}/activations_dict_video_camera.npz" \
  "${FIGURE_ROOT}/interpret_output_video_camera" \
  "${FIGURE_ROOT}/final_fig_video_camera"

run_one \
  "init_pos" \
  "1 2 3 7 9" \
  "${INIT_POS_RUN_DIR}" \
  "${FIGURE_ROOT}/activations_dict_video_init_pos.npz" \
  "${FIGURE_ROOT}/interpret_output_video_init_pos" \
  "${FIGURE_ROOT}/final_fig_video_init_pos"

run_one \
  "gaussian" \
  "6 0 1 4 7" \
  "${GAUSSIAN_RUN_DIR}" \
  "${FIGURE_ROOT}/activations_dict_video_gaussian.npz" \
  "${FIGURE_ROOT}/interpret_output_video_gaussian" \
  "${FIGURE_ROOT}/final_fig_video_gaussian"

echo "All mechanistic analysis figures finished."
