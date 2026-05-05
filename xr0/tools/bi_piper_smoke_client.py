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
    parser = argparse.ArgumentParser(description="Bimanual Piper adapter for XR0 runtime smoke tests.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10086)
    parser.add_argument("--processor", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--task", type=str, default="bimanual smoke test")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--sleep-s", type=float, default=0.2)
    parser.add_argument("--lerobot-src", type=str, default="/home/whz/teleop_evo/src")
    parser.add_argument("--xr0-src", type=str, default=str(xr0_root))
    parser.add_argument("--left-can", type=str, required=True)
    parser.add_argument("--right-can", type=str, required=True)
    parser.add_argument("--robot-id", type=str, default="xr0_smoke_bi_piper")
    parser.add_argument("--top-camera", type=str, required=True)
    parser.add_argument("--left-wrist-camera", type=str, default=None)
    parser.add_argument("--right-wrist-camera", type=str, required=True)
    parser.add_argument(
        "--missing-left-wrist-fill",
        choices=("black", "copy-ego", "copy-right-wrist"),
        default="black",
    )
    parser.add_argument("--ego-camera-side", choices=("left", "right"), default="left")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-warmup-s", type=int, default=2)
    parser.add_argument("--startup-sleep-s", type=float, default=0.5)
    parser.add_argument("--speed-ratio", type=int, default=20)
    parser.add_argument("--high-follow", action="store_true", default=False)
    parser.add_argument("--require-calibration", action="store_true", default=False)
    return parser.parse_args()


def observation_to_pil(image_like) -> Image.Image:
    image = np.asarray(image_like, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC uint8 RGB image, got shape {image.shape}")
    return Image.fromarray(image, mode="RGB")


def make_black_image(reference: Image.Image) -> Image.Image:
    return Image.new("RGB", reference.size, color=(0, 0, 0))


def fill_missing_left_wrist(mode: str, ego_image: Image.Image, right_wrist_image: Image.Image) -> Image.Image:
    if mode == "copy-ego":
        return ego_image.copy()
    if mode == "copy-right-wrist":
        return right_wrist_image.copy()
    return make_black_image(right_wrist_image)


def extract_arm_state(raw_observation: dict[str, object], side: str) -> tuple[np.ndarray, np.ndarray]:
    joint = np.array(
        [float(raw_observation[f"{side}_joint_{index}.pos"]) for index in range(1, 7)],
        dtype=np.float32,
    )
    gripper = np.array([float(raw_observation[f"{side}_gripper.pos"])], dtype=np.float32)
    return joint, gripper


def build_robot_state(raw_observation: dict[str, object]) -> dict[str, np.ndarray]:
    left_joint, left_gripper = extract_arm_state(raw_observation, "left")
    right_joint, right_gripper = extract_arm_state(raw_observation, "right")
    return {
        "left_arm_joint": left_joint,
        "left_gripper_pos": left_gripper,
        "right_arm_joint": right_joint,
        "right_gripper_pos": right_gripper,
    }


def build_runtime_inputs(args: argparse.Namespace, raw_observation: dict[str, object]) -> dict[str, object]:
    ego_key = f"{args.ego_camera_side}_ego"
    if ego_key not in raw_observation:
        raise KeyError(
            f"Expected ego image under '{ego_key}', but available keys are: {sorted(raw_observation.keys())}"
        )

    ego_image = observation_to_pil(raw_observation[ego_key])
    right_wrist_image = observation_to_pil(raw_observation["right_wrist"])
    if "left_wrist" in raw_observation:
        left_wrist_image = observation_to_pil(raw_observation["left_wrist"])
    else:
        left_wrist_image = fill_missing_left_wrist(args.missing_left_wrist_fill, ego_image, right_wrist_image)

    return {
        "robot_state": build_robot_state(raw_observation),
        "ego_obs": ego_image,
        "left_wrist_obs": left_wrist_image,
        "right_wrist_obs": right_wrist_image,
        "instruction": args.task,
    }


def format_vector(values: np.ndarray, precision: int = 3) -> str:
    rounded = np.round(np.asarray(values, dtype=np.float32), precision)
    return np.array2string(rounded, precision=precision, suppress_small=False)


def make_realsense_config(camera_cls, serial_number: str, args: argparse.Namespace):
    return camera_cls(
        serial_number_or_name=serial_number,
        width=args.width,
        height=args.height,
        fps=args.fps,
        warmup_s=args.camera_warmup_s,
    )


def make_robot(args: argparse.Namespace):
    add_sys_path(args.lerobot_src)

    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
    from lerobot.robots.bi_piper_follower import BiPiperFollower, BiPiperFollowerConfig
    from lerobot.robots.piper_follower import PiperFollowerConfigBase

    left_cameras = {}
    if args.left_wrist_camera:
        left_cameras["wrist"] = make_realsense_config(RealSenseCameraConfig, args.left_wrist_camera, args)
    right_cameras = {
        "wrist": make_realsense_config(RealSenseCameraConfig, args.right_wrist_camera, args),
    }
    ego_camera = make_realsense_config(RealSenseCameraConfig, args.top_camera, args)
    if args.ego_camera_side == "left":
        left_cameras["ego"] = ego_camera
    else:
        right_cameras["ego"] = ego_camera

    config = BiPiperFollowerConfig(
        id=args.robot_id,
        left_arm_config=PiperFollowerConfigBase(
            port=args.left_can,
            startup_sleep_s=args.startup_sleep_s,
            speed_ratio=args.speed_ratio,
            high_follow=args.high_follow,
            require_calibration=args.require_calibration,
            cameras=left_cameras,
        ),
        right_arm_config=PiperFollowerConfigBase(
            port=args.right_can,
            startup_sleep_s=args.startup_sleep_s,
            speed_ratio=args.speed_ratio,
            high_follow=args.high_follow,
            require_calibration=args.require_calibration,
            cameras=right_cameras,
        ),
    )
    robot = BiPiperFollower(config)
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
            f"Connected robot={args.robot_id} left_can={args.left_can} right_can={args.right_can} "
            f"ego_side={args.ego_camera_side} top={args.top_camera} "
            f"left_wrist={args.left_wrist_camera or f'masked:{args.missing_left_wrist_fill}'} "
            f"right_wrist={args.right_wrist_camera} "
            f"-> xr0_server={args.host}:{args.port}"
        )
        for index in range(args.iterations):
            started = time.perf_counter()
            raw_observation = robot.get_observation()
            outputs = client(**build_runtime_inputs(args, raw_observation))
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            left_joint_delta = outputs["action_components"]["left_joint"][0]
            left_gripper_delta = outputs["action_components"]["left_gripper"][0]
            right_joint_delta = outputs["action_components"]["right_joint"][0]
            right_gripper_delta = outputs["action_components"]["right_gripper"][0]
            left_joint_target = outputs["action_targets"]["left_arm_joint"][0]
            left_gripper_target = outputs["action_targets"]["left_gripper_pos"][0]
            right_joint_target = outputs["action_targets"]["right_arm_joint"][0]
            right_gripper_target = outputs["action_targets"]["right_gripper_pos"][0]

            print(
                f"[{index + 1:02d}/{args.iterations:02d}] round_trip={elapsed_ms:.1f}ms "
                f"raw_action_shape={tuple(outputs['raw_action'].shape)} "
                f"left_joint_delta={format_vector(left_joint_delta)} "
                f"left_gripper_delta={format_vector(left_gripper_delta)} "
                f"right_joint_delta={format_vector(right_joint_delta)} "
                f"right_gripper_delta={format_vector(right_gripper_delta)}"
            )
            print(
                f"           left_joint_target={format_vector(left_joint_target)} "
                f"left_gripper_target={format_vector(left_gripper_target)} "
                f"right_joint_target={format_vector(right_joint_target)} "
                f"right_gripper_target={format_vector(right_gripper_target)}"
            )

            if index + 1 < args.iterations and args.sleep_s > 0:
                time.sleep(args.sleep_s)
    finally:
        if client is not None:
            client.close()
        robot.disconnect()


if __name__ == "__main__":
    main()
