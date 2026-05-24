import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


def _load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_record(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"record not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pool-based activation pairs from trajectory records.")
    parser.add_argument("--collect-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pair-seed", type=int, default=0)
    parser.add_argument("--pairing-mode", type=str, choices=["pool"], default="pool")
    args = parser.parse_args()

    if args.pairing_mode != "pool":
        raise ValueError("Only pool pairing is supported.")

    manifest = _load_manifest(args.collect_dir / "manifest.json")
    records = manifest.get("records", [])
    if not records:
        raise ValueError("No records in collect manifest.")

    nominal_success_pool: List[Dict[str, Any]] = []
    perturb_failure_pool: List[Dict[str, Any]] = []
    dropped_nominal_fail = 0
    dropped_perturb_success = 0
    total_activation_records = 0

    for row in records:
        rec = _load_record(Path(row["path"]))
        is_nominal = bool(rec.get("is_nominal", False))
        traj_success = bool(rec.get("trajectory_success", False))
        captures = list(rec.get("captures", []))
        total_activation_records += len(captures)
        for cap in captures:
            item = {
                "variant_name": rec["variant_name"],
                "task_id": int(rec["task_id"]),
                "episode_idx": int(rec["episode_idx"]),
                "inference_idx_in_traj": int(cap["inference_idx_in_traj"]),
                "frame_st_id": int(cap["frame_st_id"]),
                "activations": cap["activations"],
            }
            if is_nominal:
                if traj_success:
                    nominal_success_pool.append(item)
                else:
                    dropped_nominal_fail += 1
            else:
                if traj_success:
                    dropped_perturb_success += 1
                else:
                    perturb_failure_pool.append(item)

    if not nominal_success_pool:
        raise RuntimeError("No positive candidates after filtering: nominal success pool is empty.")
    if not perturb_failure_pool:
        raise RuntimeError("No negative candidates after filtering: perturbation failure pool is empty.")

    rng = np.random.default_rng(seed=int(args.pair_seed))
    rng.shuffle(nominal_success_pool)
    rng.shuffle(perturb_failure_pool)
    n_pair = min(len(nominal_success_pool), len(perturb_failure_pool))
    pos = nominal_success_pool[:n_pair]
    neg = perturb_failure_pool[:n_pair]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pos_fp = args.out_dir / "positive.pt"
    neg_fp = args.out_dir / "negative.pt"
    torch.save(
        {
            "records": pos,
            "mode": manifest.get("mode"),
            "selected_timesteps": manifest.get("selected_timesteps"),
            "layers": manifest.get("layers"),
            "prompt": manifest.get("task_language"),
        },
        pos_fp,
    )
    torch.save(
        {
            "records": neg,
            "mode": manifest.get("mode"),
            "selected_timesteps": manifest.get("selected_timesteps"),
            "layers": manifest.get("layers"),
            "prompt": manifest.get("task_language"),
        },
        neg_fp,
    )

    out_manifest = {
        "mode": "pool",
        "collect_dir": str(args.collect_dir.resolve()),
        "pair_seed": int(args.pair_seed),
        "total_activation_records": int(total_activation_records),
        "dropped_nominal_fail": int(dropped_nominal_fail),
        "dropped_perturb_success": int(dropped_perturb_success),
        "nominal_success_pool_size": int(len(nominal_success_pool)),
        "perturb_failure_pool_size": int(len(perturb_failure_pool)),
        "num_pairs_total": int(n_pair),
        "prompt": manifest.get("task_language"),
        "task_language": manifest.get("task_language"),
        "files": {
            "positive": str(pos_fp.resolve()),
            "negative": str(neg_fp.resolve()),
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(out_manifest, indent=2), encoding="utf-8")
    print(f"[pair] wrote {pos_fp}")
    print(f"[pair] wrote {neg_fp}")
    print(f"[pair] wrote {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
