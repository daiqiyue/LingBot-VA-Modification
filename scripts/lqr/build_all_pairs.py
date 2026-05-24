import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

DEBUG_LOG_PATH = "/storage/home/hcoda1/9/qdai41/scratch/cosmos/.cursor/debug-810aa6.log"
DEBUG_SESSION_ID = "810aa6"


def _dbg_log(run_id: str, hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


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
    run_id = f"pre-fix-{int(time.time())}"

    if args.pairing_mode != "pool":
        raise ValueError("Only pool pairing is supported.")

    manifest = _load_manifest(args.collect_dir / "manifest.json")
    records = manifest.get("records", [])
    if not records:
        raise ValueError("No records in collect manifest.")
    #region agent log
    _dbg_log(
        run_id=run_id,
        hypothesis_id="H4",
        location="scripts/lqr/build_all_pairs.py:main:init",
        message="Loaded collect manifest",
        data={
            "collect_dir": str(args.collect_dir),
            "records_count": len(records),
            "num_episodes": manifest.get("num_episodes"),
            "variants": [str(v.get("name", "")) for v in manifest.get("variants", [])],
        },
    )
    #endregion

    nominal_success_pool: List[Dict[str, Any]] = []
    perturb_failure_pool: List[Dict[str, Any]] = []
    dropped_nominal_fail = 0
    dropped_perturb_success = 0
    total_activation_records = 0
    traj_counter = {
        "nominal_success_traj": 0,
        "nominal_fail_traj": 0,
        "perturb_success_traj": 0,
        "perturb_fail_traj": 0,
    }
    cap_counter = {
        "nominal_success_caps": 0,
        "nominal_fail_caps": 0,
        "perturb_success_caps": 0,
        "perturb_fail_caps": 0,
    }
    variant_traj: Dict[str, Dict[str, int]] = {}
    variant_caps: Dict[str, Dict[str, int]] = {}

    for row in records:
        rec = _load_record(Path(row["path"]))
        is_nominal = bool(rec.get("is_nominal", False))
        traj_success = bool(rec.get("trajectory_success", False))
        variant_name = str(rec.get("variant_name", "unknown"))
        captures = list(rec.get("captures", []))
        total_activation_records += len(captures)
        if variant_name not in variant_traj:
            variant_traj[variant_name] = {"success": 0, "fail": 0}
            variant_caps[variant_name] = {"success_caps": 0, "fail_caps": 0}
        if traj_success:
            variant_traj[variant_name]["success"] += 1
            variant_caps[variant_name]["success_caps"] += len(captures)
        else:
            variant_traj[variant_name]["fail"] += 1
            variant_caps[variant_name]["fail_caps"] += len(captures)
        for cap in captures:
            item = {
                "variant_name": variant_name,
                "task_id": int(rec["task_id"]),
                "episode_idx": int(rec["episode_idx"]),
                "inference_idx_in_traj": int(cap["inference_idx_in_traj"]),
                "frame_st_id": int(cap["frame_st_id"]),
                "activations": cap["activations"],
            }
            if is_nominal:
                if traj_success:
                    cap_counter["nominal_success_caps"] += 1
                    nominal_success_pool.append(item)
                else:
                    cap_counter["nominal_fail_caps"] += 1
                    dropped_nominal_fail += 1
            else:
                if traj_success:
                    cap_counter["perturb_success_caps"] += 1
                    dropped_perturb_success += 1
                else:
                    cap_counter["perturb_fail_caps"] += 1
                    perturb_failure_pool.append(item)
        if is_nominal and traj_success:
            traj_counter["nominal_success_traj"] += 1
        elif is_nominal and (not traj_success):
            traj_counter["nominal_fail_traj"] += 1
        elif (not is_nominal) and traj_success:
            traj_counter["perturb_success_traj"] += 1
        else:
            traj_counter["perturb_fail_traj"] += 1

    #region agent log
    _dbg_log(
        run_id=run_id,
        hypothesis_id="H1,H2,H3",
        location="scripts/lqr/build_all_pairs.py:main:post_scan",
        message="Pool precheck stats",
        data={
            "total_activation_records": total_activation_records,
            "traj_counter": traj_counter,
            "cap_counter": cap_counter,
            "variant_traj": variant_traj,
            "variant_caps": variant_caps,
            "nominal_success_pool_size": len(nominal_success_pool),
            "perturb_failure_pool_size": len(perturb_failure_pool),
        },
    )
    #endregion

    if not nominal_success_pool:
        #region agent log
        _dbg_log(
            run_id=run_id,
            hypothesis_id="H1,H2,H3",
            location="scripts/lqr/build_all_pairs.py:main:raise_no_positive",
            message="No positive candidates",
            data={
                "nominal_success_pool_size": len(nominal_success_pool),
                "traj_counter": traj_counter,
                "variant_traj": variant_traj,
            },
        )
        #endregion
        raise RuntimeError("No positive candidates after filtering: nominal success pool is empty.")
    if not perturb_failure_pool:
        #region agent log
        _dbg_log(
            run_id=run_id,
            hypothesis_id="H1",
            location="scripts/lqr/build_all_pairs.py:main:raise_no_negative",
            message="No negative candidates",
            data={
                "perturb_failure_pool_size": len(perturb_failure_pool),
                "traj_counter": traj_counter,
                "variant_traj": variant_traj,
            },
        )
        #endregion
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
    #region agent log
    _dbg_log(
        run_id=run_id,
        hypothesis_id="H4",
        location="scripts/lqr/build_all_pairs.py:main:success",
        message="Pair construction success",
        data={
            "paired_count": int(n_pair),
            "nominal_success_pool_size": len(nominal_success_pool),
            "perturb_failure_pool_size": len(perturb_failure_pool),
        },
    )
    #endregion
    print(f"[pair] wrote {pos_fp}")
    print(f"[pair] wrote {neg_fp}")
    print(f"[pair] wrote {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
