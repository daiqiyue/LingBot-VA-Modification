import argparse
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch

from scripts.activation_steering.common import read_jsonl, write_json


def _aggregate(stacked: torch.Tensor, method: str) -> torch.Tensor:
    # stacked shape [N, C]
    if stacked.shape[0] == 0:
        raise ValueError("No samples provided for aggregation")
    if method == "mean":
        return stacked.mean(dim=0)
    if method == "trimmed_mean":
        if stacked.shape[0] < 5:
            return stacked.mean(dim=0)
        lo = int(0.1 * stacked.shape[0])
        hi = max(lo + 1, int(0.9 * stacked.shape[0]))
        vals, _ = torch.sort(stacked, dim=0)
        return vals[lo:hi].mean(dim=0)
    if method == "median_of_means":
        k = min(5, max(1, stacked.shape[0]))
        chunks = torch.chunk(stacked, k, dim=0)
        means = torch.stack([c.mean(dim=0) for c in chunks], dim=0)
        return means.median(dim=0).values
    raise ValueError(f"Unknown agg method: {method}")


def _l2_normalize(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm(p=2) + 1e-8)


def _load_trace(path: str) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "layers" not in payload or "meta" not in payload:
        raise ValueError(f"Invalid trace payload: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build activation steering bank from pairs and trace index.")
    parser.add_argument("--pairs", type=str, required=True)
    parser.add_argument(
        "--trace-index",
        type=str,
        required=True,
        help="JSONL with rows: run_id, episode_idx, phase, mode, path",
    )
    parser.add_argument("--agg", type=str, choices=["mean", "trimmed_mean", "median_of_means"], default="trimmed_mean")
    parser.add_argument("--normalize", type=str, choices=["l2", "none"], default="l2")
    parser.add_argument("--out-path", type=str, required=True)
    args = parser.parse_args()

    pairs = read_jsonl(args.pairs)
    trace_rows = read_jsonl(args.trace_index)

    by_key: Dict[Tuple[str, int, str], List[str]] = defaultdict(list)
    for r in trace_rows:
        key = (str(r["run_id"]), int(r["episode_idx"]), str(r.get("mode", "video")))
        by_key[key].append(r["path"])

    # deltas[(mode, layer)] -> list[tensor(C)]
    deltas: Dict[Tuple[str, str], List[torch.Tensor]] = defaultdict(list)
    counts_by_bucket: Dict[str, int] = defaultdict(int)

    for p in pairs:
        pos_run = str(p["pos_run_id"])
        neg_run = str(p["neg_run_id"])
        pos_ep = int(p["pos_episode_idx"])
        neg_ep = int(p["neg_episode_idx"])
        bucket = str(p.get("pair_bucket", "unknown"))

        for mode in ("video", "action"):
            pos_paths = by_key.get((pos_run, pos_ep, mode), [])
            neg_paths = by_key.get((neg_run, neg_ep, mode), [])
            if not pos_paths or not neg_paths:
                continue

            pos_payload = _load_trace(pos_paths[0])
            neg_payload = _load_trace(neg_paths[0])

            for layer_key, pos_vec in pos_payload["layers"].items():
                if layer_key not in neg_payload["layers"]:
                    continue
                neg_vec = neg_payload["layers"][layer_key]
                if not torch.is_tensor(pos_vec):
                    pos_vec = torch.tensor(pos_vec)
                if not torch.is_tensor(neg_vec):
                    neg_vec = torch.tensor(neg_vec)
                deltas[(mode, str(layer_key))].append(pos_vec.float() - neg_vec.float())
                counts_by_bucket[bucket] += 1

    vectors: Dict[str, Dict[str, torch.Tensor]] = {"video": {}, "action": {}}
    stats: Dict[str, Any] = {"counts_by_bucket": dict(counts_by_bucket), "sample_count": {}}
    for (mode, layer), samples in deltas.items():
        stacked = torch.stack(samples, dim=0)
        v = _aggregate(stacked, args.agg)
        if args.normalize == "l2":
            v = _l2_normalize(v)
        vectors[mode][layer] = v
        stats["sample_count"][f"{mode}:{layer}"] = int(stacked.shape[0])

    torch.save({"vectors": vectors, "stats": stats}, args.out_path)
    write_json(args.out_path + ".meta.json", {"stats": stats, "layers": {k: list(v.keys()) for k, v in vectors.items()}})
    print(f"[bank] saved steering bank -> {args.out_path}")


if __name__ == "__main__":
    main()
