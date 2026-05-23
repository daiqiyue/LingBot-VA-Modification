import argparse
import os

from scripts.lqr.common import maybe_load_yaml
from scripts.lqr.lqr_injector import LQRInjector


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Lingbot server with runtime LQR patch.")
    parser.add_argument("--config-name", type=str, default="libero")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--save_root", type=str, default=None)

    parser.add_argument("--svd-dir", type=str, required=True)
    parser.add_argument("--jac-dir-act", type=str, required=True)
    parser.add_argument("--lambda-scale", type=float, default=1.0)
    parser.add_argument("--q-scale", type=float, default=10000.0)
    parser.add_argument("--r-scale", type=float, default=75000.0)
    parser.add_argument("--qf-scale", type=float, default=1.0)
    parser.add_argument("--modality", type=str, choices=["video", "action", "both"], default="both")
    parser.add_argument(
        "--apply-on",
        type=str,
        choices=["transient_only", "include_cache_write"],
        default="transient_only",
    )
    parser.add_argument("--video-steps", type=int, default=20)
    parser.add_argument("--action-steps", type=int, default=50)
    parser.add_argument("--lqr-config", type=str, default=None, help="Optional YAML/JSON config overrides.")
    return parser.parse_args()


def _build_injector(args: argparse.Namespace) -> LQRInjector:
    merged = {
        "svd_dir": args.svd_dir,
        "jac_dir_act": args.jac_dir_act,
        "lambda_scale": args.lambda_scale,
        "q_scale": args.q_scale,
        "r_scale": args.r_scale,
        "qf_scale": args.qf_scale,
        "modality": args.modality,
        "apply_on": args.apply_on,
        "video_steps": args.video_steps,
        "action_steps": args.action_steps,
    }
    if args.lqr_config:
        merged.update(maybe_load_yaml(args.lqr_config))
    return LQRInjector(
        svd_dir=str(merged["svd_dir"]),
        jac_dir_act=str(merged["jac_dir_act"]),
        lambda_scale=float(merged["lambda_scale"]),
        q_scale=float(merged["q_scale"]),
        r_scale=float(merged["r_scale"]),
        qf_scale=float(merged["qf_scale"]),
        modality=str(merged["modality"]),
        apply_on=str(merged["apply_on"]),
        video_steps=int(merged["video_steps"]),
        action_steps=int(merged["action_steps"]),
    )


def main() -> None:
    args = _parse_args()
    injector = _build_injector(args)
    os.environ.setdefault("PYTHONPATH", ".")

    import wan_va.wan_va_server as base_server

    original_init = base_server.VA_Server.__init__
    original_infer_chunk = base_server.VA_Server._infer

    def patched_init(self, job_config):
        original_init(self, job_config)
        transformer = self.transformer
        injector.register_hooks(transformer)
        original_forward = transformer.forward

        def patched_forward(*f_args, **f_kwargs):
            action_mode = bool(f_kwargs.get("action_mode", False))
            update_cache = int(f_kwargs.get("update_cache", 0))
            injector.begin_call(action_mode=action_mode, update_cache=update_cache)
            try:
                return original_forward(*f_args, **f_kwargs)
            finally:
                injector.end_call()

        transformer.forward = patched_forward

    def patched_infer(self, obs, frame_st_id=0):
        injector.on_chunk_start(chunk_id=int(frame_st_id))
        return original_infer_chunk(self, obs, frame_st_id=frame_st_id)

    base_server.VA_Server.__init__ = patched_init
    base_server.VA_Server._infer = patched_infer

    try:
        base_server.run(args)
    finally:
        injector.close()


if __name__ == "__main__":
    main()
