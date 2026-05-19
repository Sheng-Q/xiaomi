#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import select
import sys
import termios
import time
import tty
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def add_sys_path(path: str | None) -> None:
    if not path:
        return
    resolved = str(Path(path).expanduser().resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    xr0_root = script_dir.parent
    parser = argparse.ArgumentParser(description="Collect bimanual PiPER teleop episodes in XR0 JSON/video format.")
    parser.add_argument("--task", type=str, required=True, help="Task description embedded into XR0 conversations.")
    parser.add_argument("--output-root", type=str, default=str(xr0_root / "data" / "bi_piper_xr0"))
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to attempt. Use 0 for unlimited.")
    parser.add_argument(
        "--episode-seconds",
        type=float,
        default=30.0,
        help="Maximum duration per episode. Use 0 or a negative value for no timeout.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--default-trajectory-type",
        choices=("success", "ongoing", "invalid"),
        default="success",
        help="Default label when an episode ends by timeout or by pressing Enter.",
    )
    parser.add_argument("--video-codec", type=str, default="libx264")
    parser.add_argument("--absolute-video-paths", action="store_true")
    parser.add_argument("--position-unit", choices=("mm", "m"), default="mm")
    parser.add_argument("--lerobot-src", type=str, default="/home/whz/teleop_evo/src")
    parser.add_argument("--left-follower-can", type=str, required=True)
    parser.add_argument("--right-follower-can", type=str, required=True)
    parser.add_argument("--left-leader-can", type=str, required=True)
    parser.add_argument("--right-leader-can", type=str, required=True)
    parser.add_argument("--robot-id", type=str, default="xr0_collect_bi_piper")
    parser.add_argument("--teleop-id", type=str, default="xr0_collect_bi_piper_leader")
    parser.add_argument("--top-camera", type=str, required=True)
    parser.add_argument("--left-wrist-camera", type=str, default=None)
    parser.add_argument("--right-wrist-camera", type=str, required=True)
    parser.add_argument(
        "--missing-left-wrist-fill",
        choices=("black", "copy-ego", "copy-right-wrist"),
        default="black",
        help="Fallback only when no left wrist camera is available. Real left wrist video is still recommended.",
    )
    parser.add_argument("--ego-camera-side", choices=("left", "right"), default="left")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-warmup-s", type=int, default=2)
    parser.add_argument("--follower-startup-sleep-s", type=float, default=0.5)
    parser.add_argument("--leader-startup-sleep-s", type=float, default=0.1)
    parser.add_argument("--follower-speed-ratio", type=int, default=100)
    parser.add_argument("--leader-command-speed-ratio", type=int, default=100)
    parser.add_argument("--follower-high-follow", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--leader-command-high-follow", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--leader-process-isolation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-calibration", action="store_true", default=False)
    return parser.parse_args()


def observation_to_image(image_like: object) -> np.ndarray:
    image = np.asarray(image_like, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC uint8 RGB image, got shape {image.shape}")
    return image


def make_black_image(reference: np.ndarray) -> np.ndarray:
    return np.zeros_like(reference, dtype=np.uint8)


def fill_missing_left_wrist(mode: str, ego_image: np.ndarray, right_wrist_image: np.ndarray) -> np.ndarray:
    if mode == "copy-ego":
        return ego_image.copy()
    if mode == "copy-right-wrist":
        return right_wrist_image.copy()
    return make_black_image(right_wrist_image)


def extract_arm_state(values: dict[str, object], side: str) -> tuple[np.ndarray, np.ndarray]:
    joint = np.array(
        [float(values[f"{side}_joint_{index}.pos"]) for index in range(1, 7)],
        dtype=np.float32,
    )
    gripper = np.array([float(values[f"{side}_gripper.pos"])], dtype=np.float32)
    return joint, gripper


def extract_images(args: argparse.Namespace, raw_observation: dict[str, object]) -> dict[str, np.ndarray]:
    ego_key = f"{args.ego_camera_side}_ego"
    if ego_key not in raw_observation:
        raise KeyError(
            f"Expected ego image under '{ego_key}', but available keys are: {sorted(raw_observation.keys())}"
        )

    ego_image = observation_to_image(raw_observation[ego_key])
    right_wrist_image = observation_to_image(raw_observation["right_wrist"])
    if "left_wrist" in raw_observation:
        left_wrist_image = observation_to_image(raw_observation["left_wrist"])
    else:
        left_wrist_image = fill_missing_left_wrist(args.missing_left_wrist_fill, ego_image, right_wrist_image)

    return {
        "ego": ego_image,
        "wrist_left": left_wrist_image,
        "wrist_right": right_wrist_image,
    }


def euler_rpy_deg_to_rotm(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )


def fk_pose_from_joint_deg(fk_solver, joint_deg: np.ndarray, position_unit: str) -> tuple[np.ndarray, np.ndarray]:
    joint_rad = np.deg2rad(np.asarray(joint_deg, dtype=np.float64)).tolist()
    end_pose = fk_solver.CalFK(joint_rad)[-1]
    pos = np.asarray(end_pose[:3], dtype=np.float32)
    if position_unit == "m":
        pos *= 1e-3
    rotm = euler_rpy_deg_to_rotm(*end_pose[3:6])
    return pos, rotm


def build_arm_record(fk_solver, joint_deg: np.ndarray, gripper: np.ndarray, position_unit: str) -> dict[str, list[float]]:
    ee_pos, ee_rotm = fk_pose_from_joint_deg(fk_solver, joint_deg, position_unit)
    return {
        "ee_pos": ee_pos.astype(np.float32).tolist(),
        "ee_rotm": ee_rotm.reshape(-1).astype(np.float32).tolist(),
        "arm_joint": np.asarray(joint_deg, dtype=np.float32).tolist(),
        "gripper_pos": np.asarray(gripper, dtype=np.float32).reshape(1).tolist(),
    }


def make_realsense_config(camera_cls, serial_number: str, args: argparse.Namespace):
    return camera_cls(
        serial_number_or_name=serial_number,
        width=args.width,
        height=args.height,
        fps=args.fps,
        warmup_s=args.camera_warmup_s,
    )


def make_robot_and_teleop(args: argparse.Namespace):
    add_sys_path(args.lerobot_src)

    from piper_sdk.kinematics.piper_fk import C_PiperForwardKinematics

    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
    from lerobot.robots.bi_piper_follower import BiPiperFollower, BiPiperFollowerConfig
    from lerobot.robots.piper_follower import PiperFollowerConfigBase
    from lerobot.teleoperators.bi_piper_leader import BiPiperLeader, BiPiperLeaderConfig
    from lerobot.teleoperators.piper_leader import PiperLeaderConfigBase
    from lerobot.utils.control_utils import sanity_check_bimanual_piper_pair

    left_cameras: dict[str, object] = {}
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

    robot_cfg = BiPiperFollowerConfig(
        id=args.robot_id,
        left_arm_config=PiperFollowerConfigBase(
            port=args.left_follower_can,
            startup_sleep_s=args.follower_startup_sleep_s,
            speed_ratio=args.follower_speed_ratio,
            high_follow=args.follower_high_follow,
            require_calibration=args.require_calibration,
            cameras=left_cameras,
        ),
        right_arm_config=PiperFollowerConfigBase(
            port=args.right_follower_can,
            startup_sleep_s=args.follower_startup_sleep_s,
            speed_ratio=args.follower_speed_ratio,
            high_follow=args.follower_high_follow,
            require_calibration=args.require_calibration,
            cameras=right_cameras,
        ),
    )
    teleop_cfg = BiPiperLeaderConfig(
        id=args.teleop_id,
        left_arm_config=PiperLeaderConfigBase(
            port=args.left_leader_can,
            startup_sleep_s=args.leader_startup_sleep_s,
            command_speed_ratio=args.leader_command_speed_ratio,
            command_high_follow=args.leader_command_high_follow,
            require_calibration=args.require_calibration,
        ),
        right_arm_config=PiperLeaderConfigBase(
            port=args.right_leader_can,
            startup_sleep_s=args.leader_startup_sleep_s,
            command_speed_ratio=args.leader_command_speed_ratio,
            command_high_follow=args.leader_command_high_follow,
            require_calibration=args.require_calibration,
        ),
        process_isolation=args.leader_process_isolation,
    )
    sanity_check_bimanual_piper_pair(robot_cfg, teleop_cfg)

    robot = BiPiperFollower(robot_cfg)
    teleop = BiPiperLeader(teleop_cfg)
    fk_solver = C_PiperForwardKinematics()

    robot.connect(calibrate=False)
    try:
        teleop.connect(calibrate=False)
    except Exception:
        robot.disconnect()
        raise

    return robot, teleop, fk_solver


@dataclass
class EpisodeBuffer:
    ego_frames: list[np.ndarray] = field(default_factory=list)
    wrist_left_frames: list[np.ndarray] = field(default_factory=list)
    wrist_right_frames: list[np.ndarray] = field(default_factory=list)
    proprios: dict[str, list[list[float]]] = field(
        default_factory=lambda: {
            "left_ee_pos": [],
            "left_ee_rotm": [],
            "left_arm_joint": [],
            "left_gripper_pos": [],
            "right_ee_pos": [],
            "right_ee_rotm": [],
            "right_arm_joint": [],
            "right_gripper_pos": [],
        }
    )
    actions: dict[str, list[list[float]]] = field(
        default_factory=lambda: {
            "left_ee_pos": [],
            "left_ee_rotm": [],
            "left_arm_joint": [],
            "left_gripper_pos": [],
            "right_ee_pos": [],
            "right_ee_rotm": [],
            "right_arm_joint": [],
            "right_gripper_pos": [],
        }
    )

    @property
    def num_frames(self) -> int:
        return len(self.ego_frames)


def append_arm_record(target: dict[str, list[list[float]]], side: str, record: dict[str, list[float]]) -> None:
    target[f"{side}_ee_pos"].append(record["ee_pos"])
    target[f"{side}_ee_rotm"].append(record["ee_rotm"])
    target[f"{side}_arm_joint"].append(record["arm_joint"])
    target[f"{side}_gripper_pos"].append(record["gripper_pos"])


def append_sample(
    episode: EpisodeBuffer,
    args: argparse.Namespace,
    fk_solver,
    raw_observation: dict[str, object],
    sent_action: dict[str, object],
) -> None:
    images = extract_images(args, raw_observation)
    episode.ego_frames.append(images["ego"].copy())
    episode.wrist_left_frames.append(images["wrist_left"].copy())
    episode.wrist_right_frames.append(images["wrist_right"].copy())

    for side in ("left", "right"):
        current_joint, current_gripper = extract_arm_state(raw_observation, side)
        target_joint, target_gripper = extract_arm_state(sent_action, side)
        append_arm_record(
            episode.proprios,
            side,
            build_arm_record(fk_solver, current_joint, current_gripper, args.position_unit),
        )
        append_arm_record(
            episode.actions,
            side,
            build_arm_record(fk_solver, target_joint, target_gripper, args.position_unit),
        )


def format_seconds(value: float) -> str:
    return f"{value:.1f}s"


@contextmanager
def cbreak_stdin(enabled: bool):
    if not enabled:
        yield
        return
    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)


def read_key_nonblocking() -> str | None:
    readable, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not readable:
        return None
    data = os.read(sys.stdin.fileno(), 1)
    if not data:
        return None
    return data.decode("utf-8", errors="ignore")


def prompt_outcome(default_value: str) -> str:
    prompt = (
        f"Save episode as [s]uccess/[f]ailure-invalid/[o]ngoing/[d]iscard/[q]uit "
        f"(ENTER={default_value}): "
    )
    mapping = {
        "": default_value,
        "s": "success",
        "f": "invalid",
        "o": "ongoing",
        "d": "discard",
        "q": "quit",
    }
    while True:
        response = input(prompt).strip().lower()
        if response in mapping:
            return mapping[response]
        print("Please enter one of: s, f, o, d, q, or just press ENTER.")


def make_episode_id(index: int) -> str:
    return f"episode_{time.strftime('%Y-%m-%d_%H_%M_%S')}_{index:03d}"


def video_path_value(video_path: Path, xr0_root: Path, use_absolute: bool) -> str:
    if use_absolute:
        return str(video_path)
    return os.path.relpath(video_path, xr0_root)


def build_instruction(task: str) -> dict[str, object]:
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


def write_video(path: Path, frames: list[np.ndarray], fps: int, codec: str) -> None:
    with imageio.get_writer(path, fps=fps, codec=codec, macro_block_size=None) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))


def save_episode(
    args: argparse.Namespace,
    xr0_root: Path,
    episode_id: str,
    trajectory_type: str,
    episode: EpisodeBuffer,
) -> Path:
    output_root = Path(args.output_root).expanduser().resolve()
    json_dir = output_root / "json"
    videos_dir = output_root / "videos"
    json_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    ego_path = videos_dir / f"{episode_id}_ego.mp4"
    left_path = videos_dir / f"{episode_id}_wrist_left.mp4"
    right_path = videos_dir / f"{episode_id}_wrist_right.mp4"
    write_video(ego_path, episode.ego_frames, args.fps, args.video_codec)
    write_video(left_path, episode.wrist_left_frames, args.fps, args.video_codec)
    write_video(right_path, episode.wrist_right_frames, args.fps, args.video_codec)

    num_frames = episode.num_frames
    json_path = json_dir / f"{episode_id}.json"
    payload = {
        "trajectory_type": trajectory_type,
        "time": episode_id,
        "num_frames": num_frames,
        "instruction": build_instruction(args.task),
        "observations": {
            "ego": [
                {
                    "path": video_path_value(ego_path, xr0_root, args.absolute_video_paths),
                    "start": 0,
                    "end": num_frames,
                    "fps": args.fps,
                    "crop_bbox": None,
                }
            ],
            "wrist_left": [
                {
                    "path": video_path_value(left_path, xr0_root, args.absolute_video_paths),
                    "start": 0,
                    "end": num_frames,
                    "fps": args.fps,
                    "crop_bbox": None,
                }
            ],
            "wrist_right": [
                {
                    "path": video_path_value(right_path, xr0_root, args.absolute_video_paths),
                    "start": 0,
                    "end": num_frames,
                    "fps": args.fps,
                    "crop_bbox": None,
                }
            ],
        },
        "proprios": episode.proprios,
        "actions": episode.actions,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return json_path


def run_episode(
    args: argparse.Namespace,
    fk_solver,
    robot,
    teleop,
    episode_index: int,
) -> tuple[EpisodeBuffer | None, str | None, bool]:
    episode = EpisodeBuffer()
    interactive = sys.stdin.isatty()
    timeout_s = float(args.episode_seconds)
    frame_period_s = 1.0 / float(args.fps)
    outcome: str | None = None
    quit_session = False

    print(
        f"Episode {episode_index:03d}: recording. "
        "Keys: ENTER=end+prompt, s=success, f=invalid, o=ongoing, d=discard, q=quit."
    )
    start_t = time.perf_counter()
    next_frame_t = start_t
    with cbreak_stdin(interactive):
        while True:
            loop_t = time.perf_counter()
            raw_observation = robot.get_observation()
            teleop_action = teleop.get_action()
            sent_action = robot.send_action(teleop_action)
            append_sample(episode, args, fk_solver, raw_observation, sent_action)

            elapsed_s = time.perf_counter() - start_t
            print(
                f"\r  frames={episode.num_frames:05d} elapsed={format_seconds(elapsed_s)}",
                end="",
                flush=True,
            )

            key = read_key_nonblocking() if interactive else None
            if key in ("\r", "\n"):
                break
            if key:
                normalized = key.lower()
                if normalized == "s":
                    outcome = "success"
                    break
                if normalized == "f":
                    outcome = "invalid"
                    break
                if normalized == "o":
                    outcome = "ongoing"
                    break
                if normalized == "d":
                    outcome = "discard"
                    break
                if normalized == "q":
                    outcome = "quit"
                    quit_session = True
                    break

            if timeout_s > 0 and elapsed_s >= timeout_s:
                break

            next_frame_t += frame_period_s
            sleep_s = next_frame_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_frame_t = max(next_frame_t, time.perf_counter())

            if time.perf_counter() - loop_t > frame_period_s * 2:
                next_frame_t = time.perf_counter()

    print()

    if episode.num_frames == 0:
        print("  No frames were recorded for this episode.")
        return None, None, quit_session

    if outcome in {"success", "ongoing", "invalid"}:
        return episode, outcome, quit_session
    if outcome == "discard":
        print("  Episode discarded.")
        return None, None, quit_session
    if outcome == "quit":
        print("  Session aborted. Current episode discarded.")
        return None, None, True

    decided = prompt_outcome(args.default_trajectory_type)
    if decided == "discard":
        print("  Episode discarded.")
        return None, None, quit_session
    if decided == "quit":
        print("  Session aborted. Current episode discarded.")
        return None, None, True
    return episode, decided, quit_session


def main() -> None:
    args = parse_args()
    xr0_root = Path(__file__).resolve().parent.parent
    robot = None
    teleop = None
    saved_count = 0
    attempted = 0
    try:
        robot, teleop, fk_solver = make_robot_and_teleop(args)
        print(
            "Connected "
            f"robot={args.robot_id} follower_can=({args.left_follower_can},{args.right_follower_can}) "
            f"teleop={args.teleop_id} leader_can=({args.left_leader_can},{args.right_leader_can}) "
            f"top={args.top_camera} left_wrist={args.left_wrist_camera or f'masked:{args.missing_left_wrist_fill}'} "
            f"right_wrist={args.right_wrist_camera} output={Path(args.output_root).expanduser().resolve()}"
        )
        print(
            "Start each episode by pressing ENTER. "
            "Use Ctrl-C to stop the whole session between episodes or during recording."
        )

        episode_index = 1
        while args.episodes <= 0 or episode_index <= args.episodes:
            input(f"\nPress ENTER to start episode {episode_index:03d}...")
            attempted += 1
            episode, trajectory_type, should_quit = run_episode(args, fk_solver, robot, teleop, episode_index)
            if episode is not None and trajectory_type is not None:
                episode_id = make_episode_id(episode_index)
                json_path = save_episode(args, xr0_root, episode_id, trajectory_type, episode)
                saved_count += 1
                print(
                    f"  Saved {trajectory_type} episode with {episode.num_frames} frames -> {json_path}"
                )
                if episode.num_frames < 30:
                    print("  Warning: episodes shorter than 30 frames are skipped by the current XR0 JSON dataset loader.")
            if should_quit:
                break
            episode_index += 1
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if teleop is not None:
            teleop.disconnect()
        if robot is not None:
            robot.disconnect()
        print(f"Session finished. attempted={attempted} saved={saved_count}")


if __name__ == "__main__":
    main()
