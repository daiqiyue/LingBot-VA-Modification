"""Shared helpers for LIBERO policy and LQR eval launchers."""

import argparse
import json
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.lqr.common import maybe_load_yaml

_INIT_POS_KINDS = frozenset(
    {"init_position", "gripper_init", "init_pos", "gripper_xyz"}
)
_GAUSSIAN_KINDS = frozenset({"gaussian", "image_gaussian_noise", "noise"})
_CAMERA_KINDS = frozenset({"camera", "camera_view", "random_camera"})


def wait_for_port(host: str, port: int, timeout_sec: int) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(1.0)
    return False


def stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()


def apply_variant_overrides(
    variants: List[Dict[str, Any]],
    gripper_xyz_base_seed: Optional[int] = None,
    agentview_noise_sigma: Optional[float] = None,
    agentview_noise_seed_base: Optional[int] = None,
    random_camera_base_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if (
        gripper_xyz_base_seed is None
        and agentview_noise_sigma is None
        and agentview_noise_seed_base is None
        and random_camera_base_seed is None
    ):
        return variants
    out: List[Dict[str, Any]] = []
    for variant in variants:
        patched = dict(variant)
        kind = str(patched.get("kind", patched.get("type", ""))).lower()
        if kind in _INIT_POS_KINDS and gripper_xyz_base_seed is not None:
            patched["base_seed"] = int(gripper_xyz_base_seed)
        if kind in _CAMERA_KINDS and random_camera_base_seed is not None:
            patched["base_seed"] = int(random_camera_base_seed)
        if kind in _GAUSSIAN_KINDS:
            if agentview_noise_sigma is not None:
                patched["sigma"] = float(agentview_noise_sigma)
            if agentview_noise_seed_base is not None:
                patched["noise_seed_base"] = int(agentview_noise_seed_base)
                patched["seed_base"] = int(agentview_noise_seed_base)
        out.append(patched)
    return out


def resolve_task_ranges(
    task_range: List[int],
    task_ids: Optional[List[int]],
) -> List[List[int]]:
    if task_ids:
        return [[int(t), int(t) + 1] for t in task_ids]
    return [task_range]


def load_eval_variants(perturb_spec: Optional[str]) -> List[Dict[str, Any]]:
    if not perturb_spec:
        return []
    spec = maybe_load_yaml(perturb_spec)
    if "perturbation" in spec:
        perturb = dict(spec["perturbation"])
        perturb.setdefault("name", perturb.get("kind", "perturbation"))
        return [perturb]
    variants = list(spec.get("variants", []))
    out = [v for v in variants if str(v.get("name", "")) != "nominal"]
    for v in out:
        if "kind" not in v:
            raise ValueError(
                "Eval perturb specs must use ctrlwam-style variant entries with explicit `kind`."
            )
    return out


def build_client_cmd(
    args: argparse.Namespace,
    out_dir: str,
    variant: Optional[Dict[str, Any]],
) -> List[str]:
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
        out_dir,
    ]
    if getattr(args, "resume", False):
        cmd += ["--resume"]
    if args.prompt:
        cmd += ["--prompt", args.prompt]
    if variant:
        kind = str(variant.get("kind", variant.get("type", ""))).lower()
        if kind in _GAUSSIAN_KINDS:
            cmd += ["--agentview-noise-sigma", str(float(variant.get("sigma", 90.0)))]
            cmd += [
                "--agentview-noise-seed-base",
                str(int(variant.get("noise_seed_base", variant.get("seed_base", 0)))),
            ]
            cmd += ["--noise-apply-wrist"]
        if kind in _CAMERA_KINDS:
            cmd += [
                "--random-camera-pos-sigma",
                str(float(variant.get("pos_sigma_m", variant.get("pos_sigma", 0.10)))),
            ]
            cmd += [
                "--random-camera-rot-sigma-deg",
                str(float(variant.get("rot_sigma_deg", 8.0))),
            ]
            cmd += [
                "--random-camera-fov-sigma",
                str(float(variant.get("fov_sigma_deg", variant.get("fov_sigma", 5.0)))),
            ]
            cmd += ["--random-camera-base-seed", str(int(variant.get("base_seed", 42)))]
            cmd += ["--random-camera-name", str(variant.get("camera_name", "agentview"))]
            cmd += [
                "--random-camera-workspace-table-z",
                str(float(variant.get("workspace_table_z", 0.90))),
            ]
            cmd += [
                "--random-camera-workspace-visible-fraction",
                str(float(variant.get("workspace_visible_fraction", 0.55))),
            ]
            cmd += [
                "--random-camera-visibility-margin-px",
                str(int(variant.get("visibility_margin_px", 8))),
            ]
            cmd += [
                "--random-camera-image-size",
                str(int(variant.get("image_size", 128))),
            ]
            cmd += [
                "--random-camera-max-rejection-attempts",
                str(int(variant.get("max_rejection_attempts", 2000))),
            ]
            if not bool(variant.get("enforce_visibility", True)):
                cmd += ["--disable-random-camera-visibility"]
        if kind in _INIT_POS_KINDS:
            cmd += [
                "--gripper-xyz-preset",
                str(variant.get("preset", variant.get("name", "xyz_random_xlarge_3"))),
            ]
            cmd += ["--gripper-xyz-base-seed", str(int(variant.get("base_seed", 42)))]
    return cmd


def collect_task_metrics(
    out_dir: str,
    benchmark_name: str,
    task_range: List[int],
) -> Dict[str, Any]:
    rows = {}
    succ_rates = []
    for task_id in range(task_range[0], task_range[1]):
        fp = Path(out_dir) / f"{benchmark_name}_{task_id}.json"
        if not fp.exists():
            continue
        data = json.loads(fp.read_text(encoding="utf-8"))
        rows[str(task_id)] = data
        succ_rates.append(float(data.get("succ_rate", 0.0)))
    avg = float(sum(succ_rates) / len(succ_rates)) if succ_rates else 0.0
    return {"tasks": rows, "avg_succ_rate": avg}
