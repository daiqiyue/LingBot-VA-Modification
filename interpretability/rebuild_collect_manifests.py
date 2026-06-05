import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_record(path: Path) -> Dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _task_id_from_path(path: Path) -> int:
    for part in path.parts:
        if part.startswith("task_"):
            try:
                return int(part.split("_", 1)[1])
            except ValueError:
                continue
    return -1


def _variant_and_episode_from_name(path: Path) -> tuple[str, int]:
    stem = path.stem
    if "__ep" not in stem:
        return stem, -1
    variant, ep_s = stem.rsplit("__ep", 1)
    try:
        return variant, int(ep_s)
    except ValueError:
        return variant, -1


def _row_from_path(path: Path) -> Dict[str, Any]:
    variant_name, episode_idx = _variant_and_episode_from_name(path)
    return {
        "variant_name": variant_name,
        "is_nominal": bool(variant_name == "nominal"),
        "task_id": _task_id_from_path(path),
        "episode_idx": episode_idx,
        "trajectory_success": True,
        "num_infer_calls": 0,
        "num_captured": 0,
        "path": str(path.resolve()),
        "video_path": None,
    }


def _row_from_record(path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "variant_name": str(payload.get("variant_name", "unknown")),
        "is_nominal": bool(payload.get("is_nominal", False)),
        "task_id": int(payload.get("task_id", -1)),
        "episode_idx": int(payload.get("episode_idx", -1)),
        "trajectory_success": bool(payload.get("trajectory_success", False)),
        "num_infer_calls": int(payload.get("num_infer_calls", 0)),
        "num_captured": int(len(payload.get("captures", []))),
        "path": str(path.resolve()),
        "video_path": payload.get("video_path"),
    }


def rebuild_shard(shard_dir: Path, overwrite: bool, load_record_metadata: bool) -> bool:
    manifest_path = shard_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        try:
            json.loads(manifest_path.read_text(encoding="utf-8"))
            return False
        except json.JSONDecodeError:
            pass

    record_paths = sorted((shard_dir / "trajectory_records").glob("*.pt"))
    if not record_paths:
        return False

    rows: List[Dict[str, Any]] = []
    first_payload: Optional[Dict[str, Any]] = None
    for record_path in record_paths:
        if load_record_metadata:
            payload = _load_record(record_path)
            if first_payload is None:
                first_payload = payload
            rows.append(_row_from_record(record_path, payload))
        else:
            rows.append(_row_from_path(record_path))

    if first_payload is None:
        first_payload = {}
    summary = {
        "config_name": "recovered",
        "libero_benchmark": "libero_10",
        "task_id": int(first_payload.get("task_id", rows[0]["task_id"])),
        "task_language": str(first_payload.get("prompt", "")),
        "num_episodes": len(rows),
        "top_k_inference_per_traj": None,
        "selected_timesteps": list(first_payload.get("selected_timesteps", [])),
        "layers": list(first_payload.get("layers", [])),
        "mode": str(first_payload.get("mode", "")),
        "video_fps": None,
        "video_enabled": any(row.get("video_path") for row in rows),
        "variants": [{"name": row["variant_name"]} for row in rows[:1]],
        "records": rows,
        "recovered_from_records": True,
        "metadata_loaded_from_records": bool(load_record_metadata),
    }
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[rebuild] wrote {manifest_path} ({len(rows)} records)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild collect shard manifests from saved trajectory_records/*.pt files.")
    parser.add_argument("--root", type=Path, required=True, help="Workflow root or one collect run directory.")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing manifest.json files too.")
    parser.add_argument(
        "--load-record-metadata",
        action="store_true",
        help="torch.load each record for exact metadata. Slower; default only infers from paths.",
    )
    args = parser.parse_args()

    rebuilt = 0
    for records_dir in sorted(args.root.rglob("trajectory_records")):
        shard_dir = records_dir.parent
        if rebuild_shard(shard_dir, overwrite=bool(args.overwrite), load_record_metadata=bool(args.load_record_metadata)):
            rebuilt += 1
    print(f"[rebuild] done: rebuilt {rebuilt} shard manifest(s)")


if __name__ == "__main__":
    main()
