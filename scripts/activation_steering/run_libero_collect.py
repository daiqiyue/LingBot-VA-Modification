import argparse
import os
import re
import subprocess
from glob import glob
from typing import Any, Dict, List, Optional

from scripts.activation_steering.common import (
    ensure_dir,
    maybe_load_yaml,
    now_ts,
    stable_run_id,
    write_jsonl,
)


EPISODE_FILE_RE = re.compile(r"^(?P<episode>\d+)_(?P<done>True|False)\.mp4$")


def _load_variants(spec_path: Optional[str]) -> List[Dict[str, Any]]:
    if not spec_path:
        return [{"name": "nominal"}]
    spec = maybe_load_yaml(spec_path)
    variants = spec.get("variants", [])
    if not variants:
        raise ValueError("perturb spec must contain non-empty `variants` list")
    out = []
    for i, v in enumerate(variants):
        if "name" not in v:
            v["name"] = f"variant_{i}"
        out.append(v)
    return out


def _build_cmd(args: argparse.Namespace, run_out_dir: str, variant: Dict[str, Any]) -> List[str]:
    cmd = [
        "python",
        "evaluation/libero/client.py",
        "--libero-benchmark",
        args.libero_benchmark,
        "--port",
        str(args.port),
        "--test-num",
        str(args.num_episodes),
        "--task-range",
        str(args.task_range[0]),
        str(args.task_range[1]),
        "--out-dir",
        run_out_dir,
    ]
    if args.prompt:
        cmd += ["--prompt", args.prompt]
    if variant.get("eef_delta") is not None:
        dx, dy, dz = variant["eef_delta"]
        cmd += ["--eef-delta", str(dx), str(dy), str(dz)]
    if variant.get("eef_preposition_steps") is not None:
        cmd += ["--eef-preposition-steps", str(int(variant["eef_preposition_steps"]))]
    if variant.get("eef_step_size") is not None:
        cmd += ["--eef-step-size", str(float(variant["eef_step_size"]))]
    if variant.get("eef_tolerance") is not None:
        cmd += ["--eef-tolerance", str(float(variant["eef_tolerance"]))]
    if variant.get("camera_rotate_deg") is not None:
        cmd += ["--agentview-camera-rotate-deg", str(float(variant["camera_rotate_deg"]))]
        cmd += ["--agentview-camera-rotate-axis", str(variant.get("camera_axis", "z"))]
    return cmd


def _scan_rollouts(run_out_dir: str, benchmark_name: str, run_id: str, variant: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    task_dirs = glob(os.path.join(run_out_dir, benchmark_name, "*"))
    for task_dir in task_dirs:
        if not os.path.isdir(task_dir):
            continue
        task_name = os.path.basename(task_dir)
        for file_name in os.listdir(task_dir):
            m = EPISODE_FILE_RE.match(file_name)
            if not m:
                continue
            episode_idx = int(m.group("episode"))
            done = m.group("done") == "True"
            rows.append(
                {
                    "run_id": run_id,
                    "task_name": task_name,
                    "episode_idx": episode_idx,
                    "init_variant": variant["name"],
                    "success": done,
                    "progress": float(done),
                    "steps": None,
                    "obs_path": None,
                    "action_path": None,
                    "server_log_path": None,
                    "video_path": os.path.join(task_dir, file_name),
                    "perturb_meta": variant,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect LIBERO trajectories for activation steering.")
    parser.add_argument("--libero-benchmark", type=str, default="libero_10")
    parser.add_argument("--port", type=int, default=29056)
    parser.add_argument("--task-range", type=int, nargs=2, default=[0, 10])
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--mode", choices=["nominal", "perturbed", "both"], default="both")
    parser.add_argument("--perturb-spec", type=str, default=None, help="YAML/JSON with `variants` list.")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    variants = _load_variants(args.perturb_spec)
    if args.mode == "nominal":
        variants = [v for v in variants if v["name"] == "nominal"] or [{"name": "nominal"}]
    elif args.mode == "perturbed":
        variants = [v for v in variants if v["name"] != "nominal"]

    root = ensure_dir(args.out_dir)
    all_rows: List[Dict[str, Any]] = []

    for variant in variants:
        run_id = stable_run_id(
            [
                args.libero_benchmark,
                str(args.task_range),
                str(args.num_episodes),
                variant["name"],
                str(args.seed),
                str(now_ts()),
            ]
        )
        run_out_dir = ensure_dir(os.path.join(root, "runs", run_id))
        cmd = _build_cmd(args, run_out_dir, variant)
        print(f"[collect] running variant={variant['name']} run_id={run_id}")
        print("[collect] cmd:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        rows = _scan_rollouts(run_out_dir, args.libero_benchmark, run_id, variant)
        all_rows.extend(rows)
        print(f"[collect] {variant['name']} episodes indexed: {len(rows)}")

    out_index = os.path.join(root, "rollouts.jsonl")
    write_jsonl(out_index, all_rows)
    print(f"[collect] wrote rollout index: {out_index} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
