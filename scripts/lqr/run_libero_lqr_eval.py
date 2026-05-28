import argparse
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.lqr.common import ensure_dir, maybe_load_yaml


def _wait_for_port(host: str, port: int, timeout_sec: int) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(1.0)
    return False


def _stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()


def _build_server_cmd(args: argparse.Namespace) -> List[str]:
    cmd = [
        "python",
        "scripts/lqr/patch_infer_with_lqr.py",
        "--config-name",
        args.config_name,
        "--port",
        str(args.port),
        "--svd-dir",
        str(args.svd_dir),
        "--jac-dir-act",
        args.jac_dir_act,
        "--lambda-scale",
        str(args.lambda_scale),
        "--q-scale",
        str(args.q_scale),
        "--r-scale",
        str(args.r_scale),
        "--qf-scale",
        str(args.qf_scale),
        "--inject-mode",
        str(args.inject_mode),
        "--save_root",
        args.save_root,
    ]
    return cmd


def _build_client_cmd(args: argparse.Namespace, out_dir: str, variant: Optional[Dict[str, Any]]) -> List[str]:
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
    if args.resume:
        cmd += ["--resume"]
    if args.prompt:
        cmd += ["--prompt", args.prompt]
    if variant:
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
        if variant.get("image_noise_sigma") is not None:
            cmd += ["--agentview-noise-sigma", str(float(variant["image_noise_sigma"]))]
    return cmd


def _load_eval_variants(perturb_spec: Optional[str]) -> List[Dict[str, Any]]:
    if not perturb_spec:
        return []
    spec = maybe_load_yaml(perturb_spec)
    variants = list(spec.get("variants", []))
    return [v for v in variants if str(v.get("name", "")) != "nominal"]


def _collect_task_metrics(out_dir: str, benchmark_name: str, task_range: List[int]) -> Dict[str, Any]:
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


def _load_lqr_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    return maybe_load_yaml(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LIBERO evaluation with LQR-steered server.")
    parser.add_argument("--config-name", type=str, default="libero")
    parser.add_argument("--libero-benchmark", type=str, default="libero_10")
    parser.add_argument("--task-range", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--port", type=int, default=29056)
    parser.add_argument("--startup-wait-sec", type=int, default=240)
    parser.add_argument("--svd-dir", type=str, required=True)
    parser.add_argument("--jac-dir-act", type=str, default="A_tilde_lingbot")
    parser.add_argument("--lqr-config", type=str, default=None)
    parser.add_argument("--lambda-scale", type=float, default=1.0)
    parser.add_argument("--q-scale", type=float, default=10000.0)
    parser.add_argument("--r-scale", type=float, default=75000.0)
    parser.add_argument("--qf-scale", type=float, default=1.0)
    parser.add_argument("--inject-mode", type=str, choices=["auto", "action", "video", "both"], default="auto")
    parser.add_argument("--perturb-spec", type=str, default=None)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument(
        "--save-root",
        dest="save_root",
        type=str,
        default="",
        help="Server debug tensor/video save root. Empty disables VA_Server debug outputs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip existing episode videos in each variant out-dir and continue until --num-episodes exist.",
    )
    args = parser.parse_args()

    cfg = _load_lqr_config(args.lqr_config)
    args.lambda_scale = float(cfg.get("lambda_scale", args.lambda_scale))
    args.q_scale = float(cfg.get("q_scale", args.q_scale))
    args.r_scale = float(cfg.get("r_scale", args.r_scale))
    args.qf_scale = float(cfg.get("qf_scale", args.qf_scale))
    cfg_inject_mode = cfg.get("inject_mode")
    if cfg_inject_mode is not None:
        args.inject_mode = str(cfg_inject_mode)
    elif args.inject_mode == "auto":
        cfg_modality = str(cfg.get("modality", "")).strip().lower()
        if cfg_modality in {"action", "video", "both"}:
            args.inject_mode = cfg_modality
    if cfg.get("jac_dir_act"):
        args.jac_dir_act = str(cfg["jac_dir_act"])

    if not args.perturb_spec:
        raise ValueError("--perturb-spec is required (nominal eval is skipped; only perturbed variants are run).")

    variants = _load_eval_variants(args.perturb_spec)
    if not variants:
        raise ValueError(
            f"No non-nominal variants found in perturb spec: {args.perturb_spec}"
        )

    out_root = ensure_dir(args.out_dir)
    perturbed_root = ensure_dir(os.path.join(out_root, "perturbed"))

    server_cmd = _build_server_cmd(args)
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", ".")
    print(f"[eval] starting lqr server: {' '.join(server_cmd)}")
    server_proc = subprocess.Popen(server_cmd, env=env)
    try:
        if not _wait_for_port("127.0.0.1", args.port, args.startup_wait_sec):
            raise RuntimeError(f"LQR server did not start at port {args.port}")

        variant_metrics = {}
        for variant in variants:
            name = str(variant.get("name", "variant"))
            out_i = ensure_dir(os.path.join(perturbed_root, name))
            pert_cmd = _build_client_cmd(args, out_dir=out_i, variant=variant)
            print(f"[eval] perturbed({name}) client: {' '.join(pert_cmd)}")
            subprocess.run(pert_cmd, check=True, env=env)
            variant_metrics[name] = _collect_task_metrics(out_i, args.libero_benchmark, args.task_range)
    finally:
        _stop_process(server_proc)

    vals = [v["avg_succ_rate"] for v in variant_metrics.values()]
    avg_variant = float(sum(vals) / len(vals)) if vals else 0.0
    summary = {
        "config_name": args.config_name,
        "svd_dir": args.svd_dir,
        "jac_dir_act": args.jac_dir_act,
        "perturb_spec": args.perturb_spec,
        "libero_benchmark": args.libero_benchmark,
        "task_range": args.task_range,
        "num_episodes": args.num_episodes,
        "resume": args.resume,
        "server_save_root": args.save_root,
        "lqr": {
            "lambda_scale": args.lambda_scale,
            "q_scale": args.q_scale,
            "r_scale": args.r_scale,
            "qf_scale": args.qf_scale,
        },
        "perturbed": variant_metrics,
        "avg_succ_rate_over_variants": avg_variant,
    }
    out_fp = Path(out_root) / "summary.json"
    out_fp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[eval] wrote {out_fp}")


if __name__ == "__main__":
    main()
