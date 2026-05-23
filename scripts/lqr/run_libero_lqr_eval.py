import argparse
import os
import signal
import subprocess
import time
from glob import glob
from typing import Any, Dict, List, Optional

from scripts.lqr.common import ensure_dir, maybe_load_yaml, read_json, write_json


def _build_server_cmd(args: argparse.Namespace) -> List[str]:
    cmd = [
        "python",
        "scripts/lqr/patch_infer_with_lqr.py",
        "--config-name",
        args.config_name,
        "--port",
        str(args.port),
        "--save_root",
        args.save_root,
        "--svd-dir",
        args.svd_dir,
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
        "--modality",
        args.modality,
        "--apply-on",
        args.apply_on,
        "--video-steps",
        str(args.video_steps),
        "--action-steps",
        str(args.action_steps),
    ]
    if args.lqr_config:
        cmd += ["--lqr-config", args.lqr_config]
    return cmd


def _build_client_cmd(
    benchmark: str,
    port: int,
    out_dir: str,
    num_episodes: int,
    task_range: List[int],
    prompt: Optional[str],
    variant: Optional[Dict[str, Any]],
) -> List[str]:
    cmd = [
        "python",
        "evaluation/libero/client.py",
        "--libero-benchmark",
        benchmark,
        "--port",
        str(port),
        "--test-num",
        str(num_episodes),
        "--task-range",
        str(task_range[0]),
        str(task_range[1]),
        "--out-dir",
        out_dir,
    ]
    if prompt:
        cmd += ["--prompt", prompt]
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
    return cmd


def _collect_metrics(run_out_dir: str, benchmark: str) -> Dict[str, Any]:
    metric_files = glob(os.path.join(run_out_dir, f"{benchmark}_*.json"))
    per_task = []
    for f in metric_files:
        obj = read_json(f)
        obj["metric_file"] = f
        per_task.append(obj)
    if not per_task:
        return {"num_tasks": 0, "avg_succ_rate": 0.0, "tasks": []}
    avg = sum(float(t.get("succ_rate", 0.0)) for t in per_task) / len(per_task)
    return {"num_tasks": len(per_task), "avg_succ_rate": avg, "tasks": per_task}


def _load_eval_variants(perturb_spec: Optional[str]) -> List[Dict[str, Any]]:
    if not perturb_spec:
        return []
    spec = maybe_load_yaml(perturb_spec)
    out: List[Dict[str, Any]] = []
    for idx, variant in enumerate(spec.get("variants", [])):
        name = str(variant.get("name", f"variant_{idx}"))
        if name == "nominal":
            continue
        v = dict(variant)
        v["name"] = name
        out.append(v)
    return out


def _stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LQR-steered LIBERO evaluation.")
    parser.add_argument("--config-name", type=str, default="libero")
    parser.add_argument("--libero-benchmark", type=str, default="libero_10")
    parser.add_argument("--task-range", type=int, nargs=2, default=[0, 10])
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--port", type=int, default=29056)
    parser.add_argument("--save_root", type=str, default="outputs/libero_lqr/server")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None)

    parser.add_argument("--svd-dir", type=str, required=True)
    parser.add_argument("--jac-dir-act", type=str, required=True)
    parser.add_argument("--lqr-config", type=str, default=None)
    parser.add_argument("--lambda-scale", type=float, default=1.0)
    parser.add_argument("--q-scale", type=float, default=10000.0)
    parser.add_argument("--r-scale", type=float, default=75000.0)
    parser.add_argument("--qf-scale", type=float, default=1.0)
    parser.add_argument("--modality", type=str, choices=["video", "action", "both"], default="both")
    parser.add_argument("--apply-on", type=str, choices=["transient_only", "include_cache_write"], default="transient_only")
    parser.add_argument("--video-steps", type=int, default=20)
    parser.add_argument("--action-steps", type=int, default=50)

    parser.add_argument(
        "--perturb-spec",
        type=str,
        default=None,
        help="YAML/JSON with `variants` list; evaluates all non-nominal variants.",
    )
    parser.add_argument("--startup-wait-sec", type=int, default=20)
    args = parser.parse_args()

    out_root = ensure_dir(args.out_dir)
    nominal_out = ensure_dir(os.path.join(out_root, "nominal"))
    perturbed_root = ensure_dir(os.path.join(out_root, "perturbed"))
    variants = _load_eval_variants(args.perturb_spec)

    env = dict(os.environ)
    env.setdefault("PYTHONPATH", ".")

    server_cmd = _build_server_cmd(args)
    print("[eval] starting lqr server:", " ".join(server_cmd))
    server_proc = subprocess.Popen(server_cmd, env=env)
    try:
        time.sleep(args.startup_wait_sec)

        nominal_cmd = _build_client_cmd(
            benchmark=args.libero_benchmark,
            port=args.port,
            out_dir=nominal_out,
            num_episodes=args.num_episodes,
            task_range=args.task_range,
            prompt=args.prompt,
            variant=None,
        )
        print("[eval] running nominal:", " ".join(nominal_cmd))
        subprocess.run(nominal_cmd, check=True, env=env)

        for variant in variants:
            variant_name = str(variant["name"])
            perturbed_out = ensure_dir(os.path.join(perturbed_root, variant_name))
            pert_cmd = _build_client_cmd(
                benchmark=args.libero_benchmark,
                port=args.port,
                out_dir=perturbed_out,
                num_episodes=args.num_episodes,
                task_range=args.task_range,
                prompt=args.prompt,
                variant=variant,
            )
            print(f"[eval] running perturbed ({variant_name}):", " ".join(pert_cmd))
            subprocess.run(pert_cmd, check=True, env=env)
    finally:
        _stop_process(server_proc)

    nominal_metrics = _collect_metrics(nominal_out, args.libero_benchmark)
    perturbed_by_variant: Dict[str, Dict[str, Any]] = {}
    for variant in variants:
        variant_name = str(variant["name"])
        perturbed_out = os.path.join(perturbed_root, variant_name)
        perturbed_by_variant[variant_name] = _collect_metrics(perturbed_out, args.libero_benchmark)

    if perturbed_by_variant:
        avg_perturbed = sum(v["avg_succ_rate"] for v in perturbed_by_variant.values()) / len(perturbed_by_variant)
    else:
        avg_perturbed = 0.0
    perturbed_metrics = {
        "num_variants": len(perturbed_by_variant),
        "avg_succ_rate_over_variants": avg_perturbed,
        "variants": perturbed_by_variant,
    }
    summary = {
        "nominal": nominal_metrics,
        "perturbed": perturbed_metrics,
        "nominal_drop_vs_perturbed_avg": nominal_metrics["avg_succ_rate"] - perturbed_metrics["avg_succ_rate_over_variants"],
    }
    write_json(os.path.join(out_root, "metrics_nominal.json"), nominal_metrics)
    write_json(os.path.join(out_root, "metrics_perturbed.json"), perturbed_metrics)
    write_json(os.path.join(out_root, "summary.json"), summary)
    print(f"[eval] done. summary: {summary}")


if __name__ == "__main__":
    main()
