#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def add_sys_path(path: str | None) -> None:
    if not path:
        return
    resolved = str(Path(path).expanduser().resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    xr0_root = script_dir.parent
    parser = argparse.ArgumentParser(description="Single-arm Piper adapter for XR0 runtime smoke tests.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10086)
    parser.add_argument("--processor", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--task", type=str, default="smoke test")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--sleep-s", type=float, default=0.2)
    parser.add_argument("--active-side", choices=("left", "right"), default="left")
    parser.add_argument("--inactive-state-fill", choices=("zero", "copy-active"), default="zero")
    parser.add_argument("--inactive-wrist-fill", choices=("black", "copy-ego", "copy-wrist"), default="black")
    parser.add_argument("--lerobot-src", type=str, default="/home/whz/teleop_evo/src")
    parser.add_argument("--xr0-src", type=str, default=str(xr0_root))
    parser.add_argument("--can", type=str, default="can1")
    parser.add_argument("--robot-id", type=str, default="xr0_smoke_piper")
    parser.add_argument("--top-camera", type=str, required=True)
    parser.add_argument("--wrist-camera", type=str, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-warmup-s", type=int, default=2)
    parser.add_argument("--startup-sleep-s", type=float, default=0.5)
    parser.add_argument("--speed-ratio", type=int, default=20)
    parser.add_argument("--high-follow", action="store_true", default=False)
    parser.add_argument("--require-calibration", action="store_true", default=False)
    return parser.parse_args()


def make_black_image(reference: Image.Image) -> Image.Image:
    return Image.new("RGB", reference.size, color=(0, 0, 0))


def fill_inactive_wrist(mode: str, ego_image: Image.Image, wrist_image: Image.Image) -> Image.Image:
    if mode == "copy-ego":
        return ego_image.copy()
    if mode == "copy-wrist":
        return wrist_image.copy()
    return make_black_image(wrist_image)


def observation_to_pil(image_like) -> Image.Image:
    image = np.asarray(image_like, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC uint8 RGB image, got shape {image.shape}")
    return Image.fromarray(image, mode="RGB")


def extract_single_arm_state(raw_observation: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    joint = np.array(
        [float(raw_observation[f"joint_{index}.pos"]) for index in range(1, 7)],
        dtype=np.float32,
    )
    gripper = np.array([float(raw_observation["gripper.pos"])], dtype=np.float32)
    return joint, gripper


def build_robot_state(args: argparse.Namespace, joint: np.ndarray, gripper: np.ndarray) -> dict[str, np.ndarray]:
    inactive_side = "right" if args.active_side == "left" else "left"
    if args.inactive_state_fill == "copy-active":
        inactive_joint = joint.copy()
        inactive_gripper = gripper.copy()
    else:
        inactive_joint = np.zeros_like(joint)
        inactive_gripper = np.zeros_like(gripper)

    return {
        f"{args.active_side}_arm_joint": joint,
        f"{args.active_side}_gripper_pos": gripper,
        f"{inactive_side}_arm_joint": inactive_joint,
        f"{inactive_side}_gripper_pos": inactive_gripper,
    }


def build_runtime_inputs(args: argparse.Namespace, raw_observation: dict[str, object]) -> dict[str, object]:
    joint, gripper = extract_single_arm_state(raw_observation)
    ego_image = observation_to_pil(raw_observation["ego"])
    wrist_image = observation_to_pil(raw_observation["wrist"])
    inactive_wrist = fill_inactive_wrist(args.inactive_wrist_fill, ego_image, wrist_image)

    if args.active_side == "left":
        left_wrist = wrist_image
        right_wrist = inactive_wrist
    else:
        left_wrist = inactive_wrist
        right_wrist = wrist_image

    return {
        "robot_state": build_robot_state(args, joint, gripper),
        "ego_obs": ego_image,
        "left_wrist_obs": left_wrist,
        "right_wrist_obs": right_wrist,
        "instruction": args.task,
    }


def format_vector(values: np.ndarray, precision: int = 3) -> str:
    rounded = np.round(np.asarray(values, dtype=np.float32), precision)
    return np.array2string(rounded, precision=precision, suppress_small=False)


def make_robot(args: argparse.Namespace):
    add_sys_path(args.lerobot_src)

    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
    from lerobot.robots.piper_follower import PiperFollower, PiperFollowerConfig

    cameras = {
        "ego": RealSenseCameraConfig(
            serial_number_or_name=args.top_camera,
            width=args.width,
            height=args.height,
            fps=args.fps,
            warmup_s=args.camera_warmup_s,
        ),
        "wrist": RealSenseCameraConfig(
            serial_number_or_name=args.wrist_camera,
            width=args.width,
            height=args.height,
            fps=args.fps,
            warmup_s=args.camera_warmup_s,
        ),
    }
    config = PiperFollowerConfig(
        port=args.can,
        id=args.robot_id,
        startup_sleep_s=args.startup_sleep_s,
        speed_ratio=args.speed_ratio,
        high_follow=args.high_follow,
        require_calibration=args.require_calibration,
        cameras=cameras,
    )
    robot = PiperFollower(config)
    robot.connect(calibrate=False)
    return robot


def make_runtime_client(args: argparse.Namespace):
    add_sys_path(args.xr0_src)
    from mibot.server.runtime.client import Client

    return Client(
        host=args.host,
        port=args.port,
        processor_name_or_path=args.processor,
    )


def main() -> None:
    args = parse_args()
    robot = make_robot(args)
    client = None
    try:
        client = make_runtime_client(args)
        print(
            f"Connected robot={args.robot_id} can={args.can} top={args.top_camera} wrist={args.wrist_camera} "
            f"-> xr0_server={args.host}:{args.port}"
        )
        for index in range(args.iterations):
            started = time.perf_counter()
            raw_observation = robot.get_observation()
            outputs = client(**build_runtime_inputs(args, raw_observation))
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            active_joint_delta = outputs["action_components"][f"{args.active_side}_joint"][0]
            active_gripper_delta = outputs["action_components"][f"{args.active_side}_gripper"][0]
            active_joint_target = outputs["action_targets"][f"{args.active_side}_arm_joint"][0]
            active_gripper_target = outputs["action_targets"][f"{args.active_side}_gripper_pos"][0]

            print(
                f"[{index + 1:02d}/{args.iterations:02d}] round_trip={elapsed_ms:.1f}ms "
                f"raw_action_shape={tuple(outputs['raw_action'].shape)} "
                f"{args.active_side}_joint_delta={format_vector(active_joint_delta)} "
                f"{args.active_side}_gripper_delta={format_vector(active_gripper_delta)} "
                f"{args.active_side}_joint_target={format_vector(active_joint_target)} "
                f"{args.active_side}_gripper_target={format_vector(active_gripper_target)}"
            )

            if index + 1 < args.iterations and args.sleep_s > 0:
                time.sleep(args.sleep_s)
    finally:
        if client is not None:
            client.close()
        robot.disconnect()


if __name__ == "__main__":
    main()
