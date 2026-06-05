# LingBot LIBERO Runbook

This file is the practical command guide for the current baseline, LQR, and eval scripts.
Run all commands from the repo root:

```bash
cd /storage/home/hcoda1/9/qdai41/scratch/cosmos/LingBot-VA-Modification
```

Most jobs write outputs under `outputs/` and Slurm logs in the repo root. Use `--export=ALL,KEY=value` to override any variable shown in an sbatch file.

## Baseline

Baseline means the LingBot policy is evaluated under a perturbation with no LQR steering.

Gaussian noise baseline:

```bash
sbatch run_baseline_gaussian.sbatch
```

Defaults:

- tasks: `0 1 4 6 7`
- episodes: `20`
- perturbation: `scripts/lqr/configs/perturb_spec_gaussian.yaml`
- eval noise seed base: `99`
- noise sigma: `90`
- output: `outputs/libero_baseline_gaussian_seed99_tasks...`

Camera baseline:

```bash
sbatch run_baseline_camera.sbatch
```

Defaults:

- tasks: `0 2 4 5 9`
- episodes: `20`
- perturbation: `scripts/lqr/configs/perturb_spec_camera.yaml`
- eval camera seed: `99`

Init-position baseline:

```bash
sbatch run_baseline_init_pos.sbatch
```

Defaults:

- tasks: `1 2 7 9`
- episodes: `20`
- perturbation: `scripts/lqr/configs/perturb_spec_init_pos.yaml`
- eval gripper seed: `99`

Useful baseline overrides:

```bash
sbatch --export=ALL,TASK_IDS="6",NUM_EPISODES=5 run_baseline_gaussian.sbatch
sbatch --export=ALL,TASK_IDS="0",NUM_EPISODES=5 run_baseline_camera.sbatch
sbatch --export=ALL,TASK_IDS="7",NUM_EPISODES=5 run_baseline_init_pos.sbatch
```

Resume a baseline output directory:

```bash
sbatch --export=ALL,RESUME=1,OUT_DIR=outputs/your_baseline_dir run_baseline_gaussian.sbatch
```

## LQR Full Runs

LQR full runs do collection, SVD, Jacobian, then LQR eval.

Gaussian LQR:

```bash
sbatch run_lqr_gaussian_noise_ctrlwam.sbatch
```

Camera LQR:

```bash
sbatch run_lqr_camera_ctrlwam.sbatch
```

Init-position LQR:

```bash
sbatch run_lqr_init_pos_ctrlwam.sbatch
```

Time-branch LQR uses denoise steps `5,10,...,50` in human indexing, stored as LingBot hook indices `4,9,...,49`.

Gaussian time branch:

```bash
sbatch run_lqr_gaussian_ctrlwam_time.sbatch
```

Camera time branch:

```bash
sbatch run_lqr_camera_ctrlwam_time.sbatch
```

Init-position time branch:

```bash
sbatch run_lqr_init_pos_ctrlwam_time.sbatch
```

No-partition time branch uses one SVD basis per layer. It is heavier but more fine-grained:

```bash
sbatch run_lqr_gaussian_ctrlwam_time_noPar.sbatch
sbatch run_lqr_camera_ctrlwam_time_noPar.sbatch
sbatch run_lqr_init_pos_ctrlwam_time_noPar.sbatch
```

Common LQR controls:

- `SKIP_EVAL=1`: stop after SVD and Jacobian.
- `SUBMIT_EVAL=0`: camera pipeline only, skip eval.
- `NUM_SAMPLES=128`: reduce rows used for SVD.
- `N_POS=1,N_NEG=1,K_TARGET=1`: quick smoke test for camera collection/SVD.
- `EVAL_NUM_EPISODES=5`: shorter eval.
- `TASK_RANGE_START=6,TASK_RANGE_END=7`: eval one task range.

Examples:

```bash
sbatch --export=ALL,SKIP_EVAL=1 run_lqr_gaussian_ctrlwam_time_noPar.sbatch
sbatch --export=ALL,SUBMIT_EVAL=0 run_lqr_camera_ctrlwam_time.sbatch
sbatch --export=ALL,NUM_SAMPLES=128,EVAL_NUM_EPISODES=5 run_lqr_gaussian_ctrlwam_time.sbatch
```

## Resume LQR After Failure

The pipeline has expensive stages. Reuse artifacts instead of starting over.

Camera pipeline stage controls:

- `START_AT=1`: collect pairs, SVD, Jacobian, eval.
- `START_AT=2`: skip collect, start at SVD.
- `START_AT=3`: skip collect and SVD, start at Jacobian.
- `START_AT=4`: skip collect/SVD/Jacobian, start at eval.

Resume camera from collected pairs:

```bash
sbatch --export=ALL,START_AT=2,PAIRS_DIR=outputs/lqr_time/pairs_cam_random_large_seed42_YYYYMMDD_HHMMSS run_lqr_camera_ctrlwam_time.sbatch
```

Resume camera from existing SVD:

```bash
sbatch --export=ALL,START_AT=3,PAIRS_DIR=outputs/lqr_time/pairs_cam_random_large_seed42_YYYYMMDD_HHMMSS,SVD_DIR=outputs/lqr_time/svd_cam_random_large_seed42_YYYYMMDD_HHMMSS run_lqr_camera_ctrlwam_time.sbatch
```

For Gaussian/init-position `run_lqr_pipeline.sh` based jobs, use:

- `EXISTING_COLLECT_DIR`: reuse raw collected pairs.
- `EXISTING_PAIRS_ALL_DIR`: reuse already row-aligned pairs.
- `EXISTING_SVD_DIR`: reuse SVD and skip SVD.

Resume Gaussian no-partition time eval from existing SVD/Jacobian:

```bash
sbatch --export=ALL,EXISTING_SVD_DIR=outputs/lqr_time_noPar/svd_all_perturb_gaussian_YYYYMMDD_HHMMSS run_lqr_gaussian_ctrlwam_time_noPar.sbatch
```

Resume Gaussian time eval from existing SVD/Jacobian:

```bash
sbatch --export=ALL,EXISTING_SVD_DIR=outputs/lqr_time/svd_all_perturb_gaussian_YYYYMMDD_HHMMSS run_lqr_gaussian_ctrlwam_time.sbatch
```

Resume init-position time branch from existing paired NPZs:

```bash
sbatch --export=ALL,EXISTING_PAIRS_ALL_DIR=outputs/lqr/pairs_init_pos_YYYYMMDD_HHMMSS__paired run_lqr_init_pos_ctrlwam_time.sbatch
```

If Jacobian already exists at `${SVD_DIR}/${JAC_SUBDIR}/A_tilde__full.pt`, the pipeline skips Jacobian automatically unless `FORCE_STEP4=1`.

## LQR Eval Only

Eval only requires:

- `SVD_DIR` containing `svd_summary.pt` and V files.
- `${SVD_DIR}/${JAC_SUBDIR}/A_tilde__full.pt`.
- Matching `JAC_SUBDIR` for the branch.

Recommended: use the same sbatch wrapper with `EXISTING_SVD_DIR` or camera `START_AT=4`, because it sets perturbation seeds and ports correctly.

Gaussian no-partition eval only:

```bash
sbatch --export=ALL,EXISTING_SVD_DIR=outputs/lqr_time_noPar/svd_all_perturb_gaussian_YYYYMMDD_HHMMSS run_lqr_gaussian_ctrlwam_time_noPar.sbatch
```

Camera time eval only:

```bash
sbatch --export=ALL,START_AT=4,SVD_DIR=outputs/lqr_time/svd_cam_random_large_seed42_YYYYMMDD_HHMMSS run_lqr_camera_ctrlwam_time.sbatch
```

Manual eval command, useful inside an interactive allocation:

```bash
python scripts/lqr/run_libero_lqr_eval.py \
  --config-name libero \
  --libero-benchmark libero_10 \
  --task-range 6 7 \
  --num-episodes 20 \
  --svd-dir outputs/lqr_time_noPar/svd_all_perturb_gaussian_YYYYMMDD_HHMMSS \
  --jac-dir-act A_tilde_lingbot_time_noPar \
  --lqr-config scripts/lqr/configs/lqr_config.yaml \
  --inject-mode action \
  --perturb-spec scripts/lqr/configs/perturb_spec_gaussian.yaml \
  --agentview-noise-sigma 90 \
  --agentview-noise-seed-base 99 \
  --out-dir outputs/lqr_eval_manual_gaussian
```

Camera manual eval should use:

```bash
--perturb-spec scripts/lqr/configs/perturb_spec_camera.yaml \
--random-camera-base-seed 99
```

Init-position manual eval should use:

```bash
--perturb-spec scripts/lqr/configs/perturb_spec_init_pos.yaml \
--gripper-xyz-base-seed 99
```

## Artifact Names

Important paths:

- pairs: `positive.npz`, `negative.npz`, `manifest.json`, `prompt.txt`
- SVD: `svd_summary.pt`, `config.json`, `V_part*_k*.pt`
- Jacobian: `${SVD_DIR}/${JAC_SUBDIR}/A_tilde__full.pt`
- eval summary: `${EVAL_OUT_BASE}/summary.json`

Common `JAC_SUBDIR` values:

- default 3-partition: `A_tilde_lingbot`
- time branch: `A_tilde_lingbot_time`
- time no-partition branch: `A_tilde_lingbot_time_noPar`

## Troubleshooting

`run_collect_pairs.py ... Aborted` with low memory usage usually comes from robosuite/EGL cleanup, not Slurm OOM. If pair files exist, continue from them:

```bash
sbatch --export=ALL,START_AT=2,PAIRS_DIR=outputs/.../pairs_cam_random_large_seed42_... run_lqr_camera_ctrlwam_time.sbatch
```

`Detected oom_kill` means Slurm memory was too low. Use an sbatch with `--mem=128G`, reduce `NUM_SAMPLES`, or use existing pairs/SVD to resume.

`FileNotFoundError ... A_tilde_lingbot/A_tilde__full.pt` means `JAC_SUBDIR` does not match the artifact. Use the correct branch-specific value, for example `A_tilde_lingbot_time_noPar`.

`torch.linalg.solve ... matrix is singular` happens during Riccati gain construction. The solver now adds jitter and falls back to pseudo-inverse; rerun eval with the current code.

