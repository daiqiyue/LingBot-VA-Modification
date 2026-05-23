import argparse
from collections import defaultdict
from typing import Any, Dict, List

from scripts.activation_steering.common import maybe_load_yaml, read_jsonl, write_jsonl


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Build matched pos/neg pairs for activation steering.")
    parser.add_argument("--rollout-index", type=str, required=True)
    parser.add_argument("--pair-policy", type=str, default=None)
    parser.add_argument("--out-path", type=str, required=True)
    args = parser.parse_args()

    policy = {
        "success_threshold_pos": 1.0,
        "progress_threshold_neg": 0.5,
        "bucket_mix": {"near_miss": 0.5, "systematic": 0.3, "diverse": 0.2},
    }
    if args.pair_policy:
        policy.update(maybe_load_yaml(args.pair_policy))

    rows = read_jsonl(args.rollout_index)
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_name"]].append(row)

    out_pairs: List[Dict[str, Any]] = []
    pair_id = 0
    pos_thr = float(policy["success_threshold_pos"])
    neg_thr = float(policy["progress_threshold_neg"])

    for task_name, task_rows in by_task.items():
        positives = [
            r
            for r in task_rows
            if _as_float(r.get("progress", r.get("success", 0.0))) >= pos_thr
            and r.get("init_variant", "nominal") == "nominal"
        ]
        negatives = [
            r
            for r in task_rows
            if _as_float(r.get("progress", r.get("success", 0.0))) <= neg_thr
            and r.get("init_variant", "nominal") != "nominal"
        ]
        if not positives or not negatives:
            continue

        pos_by_episode = {int(r["episode_idx"]): r for r in positives}
        for neg in negatives:
            ep = int(neg["episode_idx"])
            pos = pos_by_episode.get(ep)
            if pos is None:
                pos = positives[ep % len(positives)]

            variant_name = str(neg.get("init_variant", "perturbed"))
            if any(k in variant_name for k in ["small", "mild", "near"]):
                bucket = "near_miss"
            elif any(k in variant_name for k in ["camera", "eef_x", "eef_y", "axis"]):
                bucket = "systematic"
            else:
                bucket = "diverse"

            out_pairs.append(
                {
                    "pair_id": f"pair_{pair_id:07d}",
                    "task_name": task_name,
                    "prompt": None,
                    "pos_run_id": pos["run_id"],
                    "neg_run_id": neg["run_id"],
                    "pos_episode_idx": int(pos["episode_idx"]),
                    "neg_episode_idx": ep,
                    "pos_video_path": pos.get("video_path"),
                    "neg_video_path": neg.get("video_path"),
                    "pair_bucket": bucket,
                    "match_score": 1.0,
                    "divergence_chunk": 0,
                }
            )
            pair_id += 1

    write_jsonl(args.out_path, out_pairs)
    print(f"[pairs] wrote {len(out_pairs)} pairs -> {args.out_path}")


if __name__ == "__main__":
    main()
