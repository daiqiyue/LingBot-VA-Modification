# Lingbot LQR Workflow (Simple and Practical)

This README records the current LQR pipeline in plain language:

1. How to run the full workflow.
2. How to reuse already collected activations (skip collect).
3. How to avoid common failures.

The LQR scripts are under `scripts/lqr/`.  
This file is placed in `scripts/activation_steering/` as a convenient project note.

## 0) Quick mental model

Current LQR flow has 5 stages:

1. Collect trajectory activations.
2. Build positive/negative pairs.
3. Run SVD on activation pairs.
4. Compute projected Jacobians.
5. Run LQR-steered eval.

## 1) Recommended entrypoints

Run from repo root:

```bash
cd /storage/home/hcoda1/9/qdai41/scratch/cosmos/lingbot-va
```

### Main (init_pos)

```bash
sbatch run_lqr.sbatch
```

### Camera perturbation

```bash
sbatch run_lqr_camera.sbatch
```

### Gaussian perturbation

```bash
sbatch run_lqr_gaussian.sbatch
```

## 2) Reuse already collected activations (skip collect)

If you already have a collect output directory with:

- `manifest.json`
- `trajectory_records/`

then you can skip stage 1 and continue from pairing:

```bash
sbatch --export=EXISTING_COLLECT_DIR=/abs/path/to/outputs/lqr/pairs_xxx run_lqr.sbatch
```

For camera / gaussian entrypoints:

```bash
sbatch --export=EXISTING_COLLECT_DIR=/abs/path/to/outputs/lqr/pairs_camera_xxx run_lqr_camera.sbatch
sbatch --export=EXISTING_COLLECT_DIR=/abs/path/to/outputs/lqr/pairs_gaussian_xxx run_lqr_gaussian.sbatch
```

Behavior:

- `EXISTING_COLLECT_DIR` empty: run collect as usual.
- `EXISTING_COLLECT_DIR` set: skip collect, directly use this directory for `build_all_pairs.py`.

## 3) Output directory naming (with perturbation suffix)

Pipeline now auto-extracts a perturbation tag from `PERTURB_SPEC` file name:

- `perturb_spec_init_pos.yaml` -> `init_pos`
- `perturb_spec_camera.yaml` -> `camera`
- `perturb_spec_gaussian.yaml` -> `gaussian`

Default outputs become:

- `outputs/lqr/pairs_<tag>_<timestamp>`
- `outputs/lqr/pairs_all_<tag>_<timestamp>`
- `outputs/lqr/svd_all_perturb_<tag>_<timestamp>`
- `outputs/lqr_eval_all_perturb_<tag>_<timestamp>`

## 4) Important runtime knobs

All of these can be overridden via `sbatch --export=...`.

- `COLLECT_MODE` (default: `action`)
- `INJECT_MODE` (default: `auto`)
- `EVAL_STARTUP_WAIT_SEC` (default: `1200`)
- `TARGET_VARIANTS`
- `NUM_EPISODES`, `EVAL_NUM_EPISODES`
- `TOPK_INFER_PER_TRAJ`
- `PAIR_SEED`
- `K_TARGET`, `NUM_SAMPLES`, `RIDGE`
- `SKIP_EVAL`

### Recommended for action-based artifacts

If your SVD/Jacobian artifacts were built from `COLLECT_MODE=action`, run eval with:

```bash
sbatch --export=INJECT_MODE=action run_lqr.sbatch
```

This avoids cross-branch shape mismatch during LQR injection.

## 5) Run only eval (when SVD/Jac are already ready)

```bash
sbatch run_lqr_eval.sbatch
```

`run_lqr_eval.sbatch` already sets distributed env defaults and a longer startup wait.

## 6) Troubleshooting

### A) `MASTER_ADDR expected, but not set`

Cause: torch distributed `env://` variables missing for eval server startup.

Fix: use updated sbatch scripts, or manually export:

- `MASTER_ADDR`
- `MASTER_PORT`
- `RANK`
- `WORLD_SIZE`
- `LOCAL_RANK`

### B) `LQR server did not start at port ...`

Cause: startup timeout too short on slower machines.

Fix: increase:

```bash
sbatch --export=EVAL_STARTUP_WAIT_SEC=1200 run_lqr.sbatch
```

### C) `mat1 and mat2 shapes cannot be multiplied`

Cause: injection branch does not match offline artifact branch (action vs video).

Fix: explicitly set `INJECT_MODE` to match artifact modality, most commonly:

```bash
INJECT_MODE=action
```

### D) Collect video looks jittery

Current collect video export is aligned to client behavior (imageio-based writer and environment-frame-centric sequence). If you still see jitter, check whether you are replaying an old output directory from a previous run.

## 7) Minimal practical examples

### Full fresh run

```bash
sbatch run_lqr.sbatch
```

### Reuse collected activations and force action injection

```bash
sbatch --export=EXISTING_COLLECT_DIR=/storage/home/.../outputs/lqr/pairs_init_pos_20260524_123456,INJECT_MODE=action run_lqr.sbatch
```

### Skip eval (build artifacts only)

```bash
sbatch --export=SKIP_EVAL=1 run_lqr.sbatch
```
