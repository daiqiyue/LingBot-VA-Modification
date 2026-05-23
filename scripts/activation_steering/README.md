# Activation Steering on LIBERO (One-Pass Primary Workflow)

This directory now supports a **one-pass workflow**:

- You provide `perturb_spec`.
- The script automatically runs nominal + all perturb variants under tracing.
- It directly builds `steering_bank.pt`.

No manual `collect -> pairs -> trace-index` pipeline is required for the main use case.

## 0) Prerequisites

- Run from repository root (contains `scripts/`, `evaluation/`, `wan_va/`).
- LIBERO environment and checkpoints are ready.
- `pyyaml` installed if you use YAML config files.
- Use a single free port (examples use `29056`).

## 1) Prepare perturb spec

Example config:

- `scripts/activation_steering/configs/perturb_spec_init_pos.yaml`

Expected structure:
- one `nominal` variant
- one or more non-nominal perturb variants

The one-pass script will:
- run nominal trace once
- run each non-nominal variant trace once
- aggregate all perturbed traces against nominal traces

## 2) One-pass build steering bank (recommended)

```bash
python scripts/activation_steering/run_libero_build_steering_bank_onepass.py \
  --config-name libero \
  --libero-benchmark libero_10 \
  --task-range 0 1 \
  --num-episodes 1 \
  --port 29056 \
  --trace-out-dir outputs/actadd_onepass_debug/trace_bank \
  --token-policy scripts/activation_steering/configs/token_policy.yaml \
  --layers 15,19,22,25,27 \
  --modality both \
  --startup-wait-sec 240 \
  --perturb-spec scripts/activation_steering/configs/perturb_spec_init_pos.yaml \
  --phase-filter infer \
  --update-cache-filter 0 \
  --agg trimmed_mean \
  --normalize l2 \
  --out-path outputs/actadd_onepass_debug/steering_bank.pt
```

Main outputs:
- `.../steering_bank.pt`
- `.../steering_bank.pt.meta.json`
- `.../trace_bank/nominal_trace/...`
- `.../trace_bank/perturbed_trace_*...`

## 3) Run steered evaluation

```bash
python scripts/activation_steering/run_libero_steered_eval.py \
  --config-name libero \
  --libero-benchmark libero_10 \
  --task-range 0 1 \
  --num-episodes 10 \
  --port 29056 \
  --steering-bank outputs/actadd_onepass_debug/steering_bank.pt \
  --steering-config scripts/activation_steering/configs/steering_config.yaml \
  --perturb-spec scripts/activation_steering/configs/perturb_spec_init_pos.yaml \
  --out-dir outputs/actadd_onepass_debug/steered_eval
```

Outputs:
- `metrics_nominal.json`
- `metrics_perturbed.json`
- `summary.json`

## 4) Slurm batch entrypoint

`run_actadd.sbatch` already uses this one-pass flow:

1. one-pass bank build
2. steered eval

```bash
sbatch run_actadd.sbatch
```

Optional overrides:

```bash
TRACE_NUM_EPISODES=3 TASK_START=0 TASK_END=1 EVAL_NUM_EPISODES=10 sbatch run_actadd.sbatch
```

## 5) Legacy scripts (still available)

The old multi-step scripts remain in this folder for compatibility and experiments:

- `run_libero_collect.py`
- `build_pairs.py`
- `index_traces.py`
- `build_steering_bank.py`

Use them only if you explicitly need pair-index based workflows.

## Troubleshooting

- **No logs from server wrappers**
  - `hook_trace_activations.py` and `patch_infer_with_steering.py` now initialize LingBot logger in wrapper mode.

- **Port never listens**
  - Ensure no stale process occupies the same port and increase `--startup-wait-sec` for slow model initialization.

- **No vectors in bank**
  - Check `phase-filter` / `update-cache-filter`.
  - Ensure traces exist under both nominal and perturbed run-tags.

## Notes

- `hook_trace_activations.py` and `patch_infer_with_steering.py` are runtime monkey patches only.
- No source modifications to `wan_va/` or `evaluation/` are required to run the workflow.
