#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    xr0_root = script_dir.parent
    parser = argparse.ArgumentParser(
        description="Generate a tiny synthetic XR0 dataset for smoke tests, with optional tactile videos."
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(xr0_root / "data" / "mock_xr0_base"),
        help="Dataset root containing json/ and videos/.",
    )
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=48, help="Must be greater than the training action length.")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument(
        "--task",
        type=str,
        default="Move both arms smoothly according to the observations.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--absolute-video-paths", action="store_true")
    parser.add_argument(
        "--include-tactile-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also create tactile_left / tactile_right videos so the tactile JSON augmentation can point to them.",
    )
    parser.add_argument(
        "--video-ext",
        choices=("mp4",),
        default="mp4",
        help="Currently the smoke-test generator writes MP4 videos.",
    )
    parser.add_argument(
        "--writer",
        choices=("auto", "opencv", "imageio"),
        default="auto",
        help="Prefer OpenCV or imageio-ffmpeg. Auto tries OpenCV first.",
    )
    parser.add_argument(
        "--xr0-root",
        type=str,
        default=str(xr0_root),
        help="XR0 project root used when emitting relative video paths in JSON.",
    )
    return parser.parse_args()


def relative_video_path(video_path: Path, xr0_root: Path, use_absolute: bool) -> str:
    if use_absolute:
        return str(video_path.resolve())
    try:
        return os.path.relpath(video_path.resolve(), xr0_root.resolve())
    except ValueError:
        return str(video_path.resolve())


def instruction_payload(task: str) -> dict[str, object]:
    return {
        "general": [
            {
                "images": [
                    "observations.ego",
                    "observations.wrist_left",
                    "observations.wrist_right",
                ],
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            "The following observations are captured from multiple views.\n"
                            "# Ego View\n<image>\n"
                            "# Left-Wrist View\n<image>\n"
                            "# Right-Wrist View\n<image>\n"
                            "Generate robot actions for the task:\n"
                            f"{task}"
                        ),
                    },
                    {
                        "from": "gpt",
                        "value": "<bot></bot>",
                    },
                ],
            }
        ]
    }


def rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float32,
    )


def rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float32,
    )


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def build_rotm(yaw: np.ndarray, pitch: np.ndarray, roll: np.ndarray) -> np.ndarray:
    rotms = []
    for y, p, r in zip(yaw.tolist(), pitch.tolist(), roll.tolist()):
        rotms.append(rot_z(y) @ rot_y(p) @ rot_x(r))
    return np.stack(rotms, axis=0).astype(np.float32)


def build_arm_series(num_frames: int, side: str) -> dict[str, np.ndarray]:
    sign = -1.0 if side == "left" else 1.0
    t = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    phase = 0.0 if side == "left" else 0.35

    current_joint = np.stack(
        [
            0.40 * np.sin(2.0 * np.pi * (t + phase)),
            0.25 * np.cos(2.0 * np.pi * (t * 0.8 + phase)),
            0.18 * np.sin(2.0 * np.pi * (t * 1.2 + phase + 0.1)),
            0.15 * np.cos(2.0 * np.pi * (t * 0.6 + phase + 0.2)),
            0.10 * np.sin(2.0 * np.pi * (t * 1.4 + phase + 0.3)),
            0.08 * np.cos(2.0 * np.pi * (t * 1.1 + phase + 0.4)),
        ],
        axis=1,
    ).astype(np.float32)
    target_joint = (
        current_joint
        + 0.02
        * np.stack(
            [
                np.cos(2.0 * np.pi * (t + phase)),
                np.sin(2.0 * np.pi * (t * 0.8 + phase)),
                np.cos(2.0 * np.pi * (t * 1.2 + phase)),
                np.sin(2.0 * np.pi * (t * 0.6 + phase)),
                np.cos(2.0 * np.pi * (t * 1.4 + phase)),
                np.sin(2.0 * np.pi * (t * 1.1 + phase)),
            ],
            axis=1,
        )
    ).astype(np.float32)

    current_gripper = (0.45 + 0.10 * np.sin(2.0 * np.pi * (t + phase))).reshape(num_frames, 1).astype(np.float32)
    target_gripper = np.clip(
        current_gripper + 0.03 * np.cos(2.0 * np.pi * (t * 1.3 + phase)).reshape(num_frames, 1),
        0.0,
        1.0,
    ).astype(np.float32)

    current_ee_pos = np.stack(
        [
            0.42 + 0.03 * np.sin(2.0 * np.pi * (t + phase)),
            sign * (0.18 + 0.02 * np.cos(2.0 * np.pi * (t * 0.9 + phase))),
            0.16 + 0.015 * np.sin(2.0 * np.pi * (t * 1.1 + phase + 0.2)),
        ],
        axis=1,
    ).astype(np.float32)
    target_ee_pos = (
        current_ee_pos
        + np.stack(
            [
                0.008 * np.cos(2.0 * np.pi * (t + phase)),
                0.006 * np.sin(2.0 * np.pi * (t * 0.7 + phase)),
                0.005 * np.cos(2.0 * np.pi * (t * 1.3 + phase)),
            ],
            axis=1,
        ).astype(np.float32)
    )

    current_yaw = sign * 0.25 + 0.18 * np.sin(2.0 * np.pi * (t + phase))
    current_pitch = 0.10 * np.cos(2.0 * np.pi * (t * 0.7 + phase))
    current_roll = sign * 0.06 * np.sin(2.0 * np.pi * (t * 1.2 + phase))
    target_yaw = current_yaw + 0.04 * np.cos(2.0 * np.pi * (t * 0.9 + phase))
    target_pitch = current_pitch + 0.03 * np.sin(2.0 * np.pi * (t * 0.8 + phase))
    target_roll = current_roll + 0.02 * np.cos(2.0 * np.pi * (t * 1.1 + phase))

    current_rotm = build_rotm(current_yaw, current_pitch, current_roll)
    target_rotm = build_rotm(target_yaw, target_pitch, target_roll)

    return {
        "current_joint": current_joint,
        "target_joint": target_joint,
        "current_gripper": current_gripper,
        "target_gripper": target_gripper,
        "current_ee_pos": current_ee_pos,
        "target_ee_pos": target_ee_pos,
        "current_ee_rotm": current_rotm,
        "target_ee_rotm": target_rotm,
    }


def to_u8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.round(image), 0, 255).astype(np.uint8)


def gradient_mesh(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    return np.meshgrid(x, y)


def rgb_frame(view: str, frame_index: int, num_frames: int, width: int, height: int) -> np.ndarray:
    xx, yy = gradient_mesh(width, height)
    phase = frame_index / max(1, num_frames - 1)
    shift = {
        "ego": 0.0,
        "wrist_left": 0.15,
        "wrist_right": 0.35,
    }[view]

    base = 0.5 + 0.5 * np.sin(2.0 * np.pi * (xx * 0.7 + yy * 0.4 + phase + shift))
    accent = 0.5 + 0.5 * np.cos(2.0 * np.pi * (xx * 0.2 - yy * 0.9 + phase * 0.8 + shift))

    image = np.stack(
        [
            70 + 150 * base,
            40 + 180 * accent,
            30 + 130 * (1.0 - base * 0.7 + accent * 0.3),
        ],
        axis=2,
    )

    cx = int((0.25 + 0.5 * phase) * (width - 1))
    cy = int((0.30 + 0.35 * math.sin(2.0 * np.pi * (phase + shift))) * (height - 1))
    radius = max(8, min(width, height) // 10)
    y_idx, x_idx = np.ogrid[:height, :width]
    mask = (x_idx - cx) ** 2 + (y_idx - cy) ** 2 <= radius**2

    highlight = {
        "ego": np.array([255, 230, 90], dtype=np.float32),
        "wrist_left": np.array([90, 240, 255], dtype=np.float32),
        "wrist_right": np.array([255, 120, 120], dtype=np.float32),
    }[view]
    image[mask] = highlight

    return to_u8(image)


def tactile_colormap(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    red = np.clip(1.8 * value, 0.0, 1.0)
    green = np.clip(1.6 - 2.8 * np.abs(value - 0.5), 0.0, 1.0)
    blue = np.clip(1.4 * (1.0 - value), 0.0, 1.0)
    return np.stack([red, green, blue], axis=2)


def tactile_frame(side: str, frame_index: int, num_frames: int, width: int, height: int) -> np.ndarray:
    xx, yy = gradient_mesh(width, height)
    phase = frame_index / max(1, num_frames - 1)
    shift = 0.0 if side == "left" else 0.2

    centers = [
        (0.35 + 0.18 * math.sin(2.0 * np.pi * (phase + shift)), 0.38 + 0.15 * math.cos(2.0 * np.pi * (phase + shift))),
        (0.62 + 0.10 * math.cos(2.0 * np.pi * (phase * 0.8 + shift)), 0.58 + 0.12 * math.sin(2.0 * np.pi * (phase * 1.1 + shift))),
    ]
    sigma = 0.10
    field = np.zeros((height, width), dtype=np.float32)
    for cx, cy in centers:
        field += np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2))).astype(np.float32)

    field *= 0.65 + 0.35 * math.sin(2.0 * np.pi * (phase + shift) + math.pi / 5.0)
    field = np.clip(field / (field.max() + 1e-6), 0.0, 1.0)

    color = tactile_colormap(field)

    grid_stride = max(12, width // 12)
    color[::grid_stride, :, :] *= 0.7
    color[:, ::grid_stride, :] *= 0.7
    return to_u8(color * 255.0)


def write_video_opencv(path: Path, frames: Iterable[np.ndarray], fps: int) -> bool:
    try:
        import cv2
    except ModuleNotFoundError:
        return False

    frame_list = list(frames)
    if not frame_list:
        raise ValueError("No frames to write.")
    height, width = frame_list[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        return False
    try:
        for frame in frame_list:
            writer.write(cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return path.is_file() and path.stat().st_size > 0


def write_video_imageio(path: Path, frames: Iterable[np.ndarray], fps: int) -> bool:
    has_ffmpeg_pkg = False
    try:
        import imageio_ffmpeg  # noqa: F401

        has_ffmpeg_pkg = True
    except ModuleNotFoundError:
        has_ffmpeg_pkg = False

    if not has_ffmpeg_pkg and shutil.which("ffmpeg") is None:
        return False

    import imageio.v2 as imageio

    frame_list = list(frames)
    if not frame_list:
        raise ValueError("No frames to write.")
    with imageio.get_writer(path, format="FFMPEG", fps=fps, codec="libx264", macro_block_size=None) as writer:
        for frame in frame_list:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    return path.is_file() and path.stat().st_size > 0


def write_video(path: Path, frames: list[np.ndarray], fps: int, preferred_writer: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)

    if preferred_writer in {"auto", "opencv"} and write_video_opencv(path, frames, fps):
        return "opencv"
    if preferred_writer in {"auto", "imageio"} and write_video_imageio(path, frames, fps):
        return "imageio"

    raise RuntimeError(
        "Unable to write MP4 videos. Install either opencv-python-headless or imageio-ffmpeg in the target environment."
    )


def episode_json(
    episode_id: str,
    task: str,
    fps: int,
    num_frames: int,
    observations: dict[str, list[dict[str, object]]],
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
) -> dict[str, object]:
    return {
        "trajectory_type": "success",
        "time": episode_id,
        "num_frames": int(num_frames),
        "instruction": instruction_payload(task),
        "observations": observations,
        "proprios": {
            "left_ee_pos": left["current_ee_pos"].tolist(),
            "left_ee_rotm": left["current_ee_rotm"].reshape(num_frames, 9).tolist(),
            "left_arm_joint": left["current_joint"].tolist(),
            "left_gripper_pos": left["current_gripper"].tolist(),
            "right_ee_pos": right["current_ee_pos"].tolist(),
            "right_ee_rotm": right["current_ee_rotm"].reshape(num_frames, 9).tolist(),
            "right_arm_joint": right["current_joint"].tolist(),
            "right_gripper_pos": right["current_gripper"].tolist(),
        },
        "actions": {
            "left_ee_pos": left["target_ee_pos"].tolist(),
            "left_ee_rotm": left["target_ee_rotm"].reshape(num_frames, 9).tolist(),
            "left_arm_joint": left["target_joint"].tolist(),
            "left_gripper_pos": left["target_gripper"].tolist(),
            "right_ee_pos": right["target_ee_pos"].tolist(),
            "right_ee_rotm": right["target_ee_rotm"].reshape(num_frames, 9).tolist(),
            "right_arm_joint": right["target_joint"].tolist(),
            "right_gripper_pos": right["target_gripper"].tolist(),
        },
    }


def main() -> None:
    args = parse_args()
    if args.num_frames <= 30:
        raise ValueError("--num-frames must be greater than 30 for the default XR0 action length.")

    np.random.seed(int(args.seed))
    output_root = Path(args.output_root).expanduser().resolve()
    xr0_root = Path(args.xr0_root).expanduser().resolve()
    json_dir = output_root / "json"
    videos_dir = output_root / "videos"
    json_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    writer_name: str | None = None
    created_json: list[Path] = []

    for episode_index in range(1, int(args.episodes) + 1):
        episode_id = f"mock_episode_{episode_index:03d}"
        left = build_arm_series(int(args.num_frames), "left")
        right = build_arm_series(int(args.num_frames), "right")

        frame_bank = {
            "ego": [rgb_frame("ego", i, int(args.num_frames), int(args.width), int(args.height)) for i in range(int(args.num_frames))],
            "wrist_left": [
                rgb_frame("wrist_left", i, int(args.num_frames), int(args.width), int(args.height)) for i in range(int(args.num_frames))
            ],
            "wrist_right": [
                rgb_frame("wrist_right", i, int(args.num_frames), int(args.width), int(args.height)) for i in range(int(args.num_frames))
            ],
        }
        if args.include_tactile_videos:
            frame_bank["tactile_left"] = [
                tactile_frame("left", i, int(args.num_frames), int(args.width), int(args.height)) for i in range(int(args.num_frames))
            ]
            frame_bank["tactile_right"] = [
                tactile_frame("right", i, int(args.num_frames), int(args.width), int(args.height)) for i in range(int(args.num_frames))
            ]

        paths: dict[str, Path] = {}
        for name, frames in frame_bank.items():
            video_path = videos_dir / f"{episode_id}_{name}.{args.video_ext}"
            writer_used = write_video(video_path, frames, int(args.fps), args.writer)
            writer_name = writer_name or writer_used
            paths[name] = video_path

        observations = {
            "ego": [
                {
                    "path": relative_video_path(paths["ego"], xr0_root, args.absolute_video_paths),
                    "start": 0,
                    "end": int(args.num_frames),
                    "fps": int(args.fps),
                    "crop_bbox": None,
                }
            ],
            "wrist_left": [
                {
                    "path": relative_video_path(paths["wrist_left"], xr0_root, args.absolute_video_paths),
                    "start": 0,
                    "end": int(args.num_frames),
                    "fps": int(args.fps),
                    "crop_bbox": None,
                }
            ],
            "wrist_right": [
                {
                    "path": relative_video_path(paths["wrist_right"], xr0_root, args.absolute_video_paths),
                    "start": 0,
                    "end": int(args.num_frames),
                    "fps": int(args.fps),
                    "crop_bbox": None,
                }
            ],
        }

        json_payload = episode_json(episode_id, args.task, int(args.fps), int(args.num_frames), observations, left, right)
        json_path = json_dir / f"{episode_id}.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(json_payload, handle, ensure_ascii=False, indent=2)
        created_json.append(json_path)

    print(f"Created {len(created_json)} mock XR0 episode(s) under: {output_root}")
    print(f"JSON dir: {json_dir}")
    print(f"Videos dir: {videos_dir}")
    if args.include_tactile_videos:
        print("Also created tactile_left / tactile_right videos for later tactile JSON augmentation.")
    if writer_name:
        print(f"Video writer: {writer_name}")


if __name__ == "__main__":
    main()
