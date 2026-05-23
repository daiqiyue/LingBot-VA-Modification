import argparse
from typing import Dict, List

import torch

from scripts.activation_steering.common import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw trace manifest into trace-index JSONL.")
    parser.add_argument("--manifest", type=str, required=True, help="traces_manifest.jsonl from hook_trace_activations.py")
    parser.add_argument("--out-path", type=str, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--phase-filter", type=str, default="infer")
    parser.add_argument("--update-cache-filter", type=int, default=0)
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    out: List[Dict] = []
    for r in rows:
        if args.phase_filter and r.get("phase") != args.phase_filter:
            continue
        if args.update_cache_filter is not None and int(r.get("update_cache", -1)) != int(args.update_cache_filter):
            continue
        payload = torch.load(r["path"], map_location="cpu", weights_only=False)
        mode = payload["meta"].get("mode", r.get("mode", "video"))
        out.append(
            {
                "run_id": args.run_id,
                "episode_idx": int(args.episode_idx),
                "phase": payload["meta"].get("phase", r.get("phase", "infer")),
                "mode": mode,
                "path": r["path"],
            }
        )
    write_jsonl(args.out_path, out)
    print(f"[index] wrote {len(out)} rows -> {args.out_path}")


if __name__ == "__main__":
    main()
