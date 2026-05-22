import numpy as np
from wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy import WebsocketClientPolicy
import argparse
from libero.libero import benchmark
import time
from libero.libero.envs import OffScreenRenderEnv
from pathlib import Path
from tqdm import tqdm
from lerobot.datasets.utils import write_json
import os
import imageio
import cv2


def save_video(
    real_obs_list,
    save_path,
    fps=15,
    video_names=["observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"],
    phase_labels=None,
):
    if not real_obs_list:
        print("❌ No real observation frames")
        return
    if phase_labels is not None and len(phase_labels) != len(real_obs_list):
        raise ValueError(
            f"phase_labels length ({len(phase_labels)}) must match "
            f"real_obs_list length ({len(real_obs_list)})"
        )

    first_obs = real_obs_list[0]
    base_h, width_base = first_obs[video_names[0]].shape[:2]
    target_size = (width_base, base_h)
    
    print(f"Saving video: {len(real_obs_list)} frames...")

    final_frames = []
    for frame_idx, obs in enumerate(real_obs_list):
        frame = np.hstack([cv2.resize(obs[name], target_size) for name in video_names]).astype(np.uint8)
        if phase_labels is not None:
            label = phase_labels[frame_idx]
            cv2.putText(
                frame,
                label,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                label,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        final_frames.append(frame)

    imageio.mimsave(save_path, final_frames, fps=fps)
    print(f"✅ Video saved to: {save_path}")


def construct_single_env(env_args):
    count = 0
    env = None
    env_creation = False
    while not env_creation and count < 5:
        try:
            env = OffScreenRenderEnv(**env_args)
            env_creation = True
        except Exception as e:
            print(f"Error!!!  construct env failed: {e}")
            time.sleep(5)
            count += 1
    if count >= 5:
        return None
    return env


def _extract_obs(obs):
    """
    Extract agentview and eye_in_hand images from raw env obs dict.

    Avoids torch round-trip: the env already returns uint8 numpy arrays [H, W, C].
    We just flip the vertical axis ([::-1]) and make a contiguous copy once.
    """
    agentview = np.ascontiguousarray(obs["agentview_image"][::-1])
    eye_in_hand = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
    return {"observation.images.agentview_rgb": agentview, "observation.images.eye_in_hand_rgb": eye_in_hand}


def _axis_angle_to_quat(axis, angle_rad):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half_angle = angle_rad / 2.0
    return np.array(
        [
            np.cos(half_angle),
            axis[0] * np.sin(half_angle),
            axis[1] * np.sin(half_angle),
            axis[2] * np.sin(half_angle),
        ],
        dtype=np.float64,
    )


def _quat_multiply(quat_a, quat_b):
    aw, ax, ay, az = quat_a
    bw, bx, by, bz = quat_b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def apply_agentview_camera_rotation(env_in, rotate_deg=None, rotate_axis="z"):
    if rotate_deg is None or rotate_deg == 0:
        return

    axis_by_name = {
        "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }
    if rotate_axis not in axis_by_name:
        raise ValueError(f"agentview_camera_rotate_axis must be one of x/y/z, got {rotate_axis}")

    cam_id = env_in.sim.model.camera_name2id("agentview")
    old_quat = np.asarray(env_in.sim.model.cam_quat[cam_id], dtype=np.float64)
    delta_quat = _axis_angle_to_quat(axis_by_name[rotate_axis], np.deg2rad(rotate_deg))
    new_quat = _quat_multiply(delta_quat, old_quat)
    new_quat = new_quat / np.linalg.norm(new_quat)
    env_in.sim.model.cam_quat[cam_id] = new_quat
    env_in.sim.forward()
    print(
        f"Rotated agentview camera by {rotate_deg} deg around {rotate_axis}-axis: "
        f"{old_quat.tolist()} -> {new_quat.tolist()}"
    )


def init_single_env(env_in, init_state):
    env_in.reset()
    env_in.set_init_state(init_state)
    for _ in range(5):
        obs, _, _, _ = env_in.step([0.] * 7)
    return obs


def apply_eef_delta_preposition(
    env_in,
    raw_obs,
    eef_delta=None,
    max_steps=80,
    step_size=0.01,
    tolerance=0.01,
    video_obs_list=None,
    phase_labels=None,
):
    if eef_delta is None:
        return raw_obs
    if max_steps <= 0:
        raise ValueError(f"eef_preposition_steps must be positive, got {max_steps}")
    if step_size <= 0:
        raise ValueError(f"eef_step_size must be positive, got {step_size}")
    if tolerance <= 0:
        raise ValueError(f"eef_tolerance must be positive, got {tolerance}")

    start_pos = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float64)
    target_pos = start_pos + np.asarray(eef_delta, dtype=np.float64)
    if target_pos.shape != (3,):
        raise ValueError(f"eef_delta must have 3 values, got {eef_delta}")

    print(
        f"EEF preposition start={start_pos.tolist()} "
        f"target={target_pos.tolist()} delta={list(eef_delta)}"
    )
    final_error = float(np.linalg.norm(target_pos - start_pos))
    steps_used = 0
    obs = raw_obs
    for step_idx in range(max_steps):
        current_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        error = target_pos - current_pos
        dist = float(np.linalg.norm(error))
        final_error = dist
        if dist <= tolerance:
            break

        delta = error
        if dist > step_size:
            delta = error / dist * step_size

        action = np.array([delta[0], delta[1], delta[2], 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        obs, _, done, _ = env_in.step(action)
        steps_used = step_idx + 1
        if video_obs_list is not None:
            video_obs_list.append(_extract_obs(obs))
        if phase_labels is not None:
            phase_labels.append("preposition")
        if done:
            print("EEF preposition reached env done; continuing to inference from latest obs.")
            break

    final_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    final_error = float(np.linalg.norm(target_pos - final_pos))
    print(
        f"EEF preposition final={final_pos.tolist()} "
        f"error={final_error:.6f} steps={steps_used}"
    )
    return obs


def env_one_step(env_in, action):
    obs, _, done, _ = env_in.step(action)
    return _extract_obs(obs), done


def run_one(
    model,
    libero_benchmark,
    task_idx,
    out_dir,
    episode_idx,
    prompt_override=None,
    eef_delta=None,
    eef_preposition_steps=80,
    eef_step_size=0.01,
    eef_tolerance=0.01,
    agentview_camera_rotate_deg=None,
    agentview_camera_rotate_axis="z",
):
    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[libero_benchmark]()
    num_tasks = benchmark_instance.get_num_tasks()
    assert task_idx < num_tasks, f"Error: error id must smaller than {num_tasks}"
    prompt = benchmark_instance.get_task(task_idx).language
    if prompt_override is not None:
        prompt = prompt_override
    env_args = {
                "bddl_file_name": benchmark_instance.get_task_bddl_file_path(task_idx),
                "camera_heights": 128,
                "camera_widths": 128,
            }
    init_states = benchmark_instance.get_task_init_states(task_idx)

    cur_env = construct_single_env(env_args)
    raw_obs = init_single_env(cur_env, init_states[episode_idx % init_states.shape[0]])
    apply_agentview_camera_rotation(
        cur_env,
        rotate_deg=agentview_camera_rotate_deg,
        rotate_axis=agentview_camera_rotate_axis,
    )
    if agentview_camera_rotate_deg is not None and agentview_camera_rotate_deg != 0:
        raw_obs, _, _, _ = cur_env.step([0.] * 7)
    full_obs_list = []
    phase_labels = []
    raw_obs = apply_eef_delta_preposition(
        cur_env,
        raw_obs,
        eef_delta=eef_delta,
        max_steps=eef_preposition_steps,
        step_size=eef_step_size,
        tolerance=eef_tolerance,
        video_obs_list=full_obs_list,
        phase_labels=phase_labels,
    )
    first_obs = _extract_obs(raw_obs)
    print(f"Prompt: {prompt}")
    ret = model.infer(dict(reset=True, prompt=prompt))

    done = False
    first = True
    while cur_env.env.timestep < 800:
        ret = model.infer(dict(obs=first_obs, prompt=prompt))
        action = ret['action']

        key_frame_list = []
        assert action.shape[2] % 4 == 0
        action_per_frame = action.shape[2] // 4
        start_idx = 1 if first else 0
        for i in range(start_idx, action.shape[1]):
            for j in range(action.shape[2]):
                ee_action = action[:, i, j]
                observes, done = env_one_step(cur_env, ee_action)
                if done:
                    break
                if (j+1) % action_per_frame == 0:
                    full_obs_list.append(observes)
                    phase_labels.append("inference")
                    key_frame_list.append(observes)

            if done:
                break

        first = False

        if done:
            break
        else:
            model.infer(dict(obs=key_frame_list, compute_kv_cache=True, imagine=False, state=action))

    out_file = Path(out_dir) / libero_benchmark / f"{task_idx}_{prompt.replace(' ', '_')}" / f"{episode_idx}_{done}.mp4"
    out_file.parent.mkdir(exist_ok=True, parents=True)

    save_video(
        real_obs_list=full_obs_list,
        save_path=out_file,
        fps=60,
        video_names=["observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"],
        phase_labels=phase_labels,
    )

    cur_env.close()
    return done


def run(
    libero_benchmark,
    port,
    out_dir,
    test_num,
    task_range=None,
    prompt=None,
    eef_delta=None,
    eef_preposition_steps=80,
    eef_step_size=0.01,
    eef_tolerance=0.01,
    agentview_camera_rotate_deg=None,
    agentview_camera_rotate_axis="z",
):
    '''
        task_range: [start, end) for splitting tasks
    '''
    if task_range is None:
        benchmark_dict = benchmark.get_benchmark_dict()
        benchmark_instance = benchmark_dict[libero_benchmark]()
        num_tasks = benchmark_instance.get_num_tasks()
        progress_bar = tqdm(range(num_tasks), total=num_tasks)
    else:
        assert len(task_range) == 2, f'task_range: [start, end) for splitting tasks, however, task_range: {task_range}'
        num_tasks = task_range[1] - task_range[0]
        progress_bar = tqdm(range(task_range[0], task_range[1]), total=num_tasks)

    print(f"#################### Use benchmark: {libero_benchmark}, num_tasks: {num_tasks} #############")
    model = WebsocketClientPolicy(port=port)

    video_save_root_dict = None

    episode_list = range(test_num)
    for task_idx in progress_bar:
        if video_save_root_dict is not None and task_idx in video_save_root_dict:
            video_save_list = os.listdir(os.path.join(out_dir, libero_benchmark, video_save_root_dict[task_idx]))
            video_states = [1 for file in video_save_list if file.split('_')[1].split('.')[0] == 'True']
            succ_num = float(len(video_states))
            episode_list = range(len(video_save_list), test_num)
        else:
            succ_num = 0.

        for episode_idx in tqdm(episode_list, total=len(episode_list)):
            res_i = run_one(
                model,
                libero_benchmark,
                task_idx,
                out_dir,
                episode_idx,
                prompt,
                eef_delta=eef_delta,
                eef_preposition_steps=eef_preposition_steps,
                eef_step_size=eef_step_size,
                eef_tolerance=eef_tolerance,
                agentview_camera_rotate_deg=agentview_camera_rotate_deg,
                agentview_camera_rotate_axis=agentview_camera_rotate_axis,
            )
            succ_num += res_i
            succ_rate = succ_num / (episode_idx + 1)
            print(f"Success rate: {succ_rate}, success num: {succ_num}, total num: {episode_idx + 1}")
            out_file = Path(out_dir) / f"{libero_benchmark}_{task_idx}.json"
            out_file.parent.mkdir(exist_ok=True, parents=True)
            write_json({
                "succ_num": succ_num,
                "total_num": episode_idx + 1.,
                "succ_rate": succ_rate,
                }, out_file
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libero-benchmark",
        type=str,
        default="libero_10",
        choices=["libero_10", "libero_goal", "libero_spatial", "libero_object"],
        help="Benchmark name",
    )
    parser.add_argument(
        "--task-range",
        type=int,
        nargs="+",
        default=[0, 10],
        help="Task range [start, end) for splitting tasks",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=23908,
        help="WebSocket port",
    )
    parser.add_argument(
        "--test-num",
        type=int,
        default=50,
        help="Number of test episodes",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/libero",
        help="Output directory for results",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt for the task (overrides benchmark prompt)",
    )
    parser.add_argument(
        "--eef-delta",
        type=float,
        nargs=3,
        default=None,
        metavar=("DX", "DY", "DZ"),
        help="Move the end-effector by this xyz delta before the first policy inference.",
    )
    parser.add_argument(
        "--eef-preposition-steps",
        type=int,
        default=80,
        help="Maximum number of environment steps used to move the end-effector before inference.",
    )
    parser.add_argument(
        "--eef-step-size",
        type=float,
        default=1.0,
        help="Maximum end-effector xyz motion per preposition step, in meters.",
    )
    parser.add_argument(
        "--eef-tolerance",
        type=float,
        default=0.01,
        help="Stop prepositioning when the end-effector is within this distance of the target, in meters.",
    )
    parser.add_argument(
        "--agentview-camera-rotate-deg",
        type=float,
        default=None,
        help="Rotate the LIBERO third-person agentview camera by this angle in degrees.",
    )
    parser.add_argument(
        "--agentview-camera-rotate-axis",
        type=str,
        choices=["x", "y", "z"],
        default="z",
        help="World axis used by --agentview-camera-rotate-deg.",
    )
    args = parser.parse_args()
    run(**vars(args))
    print("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    main()
