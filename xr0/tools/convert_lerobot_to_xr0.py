#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def load_fk_class(piper_sdk_root: str | None):
    candidates: list[Path] = []
    if piper_sdk_root:
        root = Path(piper_sdk_root).expanduser().resolve()
        candidates.extend(
            [
                root / "piper_sdk" / "kinematics" / "piper_fk.py",
                root / "kinematics" / "piper_fk.py",
                root / "piper_fk.py",
            ]
        )

    for candidate in candidates:
        if not candidate.is_file():
            continue
        spec = importlib.util.spec_from_file_location("_xr0_piper_fk", candidate)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load FK module spec from {candidate}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fk_class = getattr(module, "C_PiperForwardKinematics", None)
        if fk_class is None:
            raise ImportError(f"{candidate} does not define C_PiperForwardKinematics")
        return fk_class

    try:
        from piper_sdk.kinematics.piper_fk import C_PiperForwardKinematics
    except ModuleNotFoundError as exc:
        search_note = ""
        if piper_sdk_root:
            search_note = (
                " Looked for piper_fk.py under: "
                + ", ".join(str(path) for path in candidates)
                + "."
            )
        raise ModuleNotFoundError(
            "Unable to import piper_sdk forward kinematics. "
            "Install piper_sdk in the current environment, or pass --piper-sdk-root "
            "pointing to a directory that contains piper_sdk/kinematics/piper_fk.py."
            + search_note
        ) from exc
    return C_PiperForwardKinematics


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    xr0_root = script_dir.parent
    parser = argparse.ArgumentParser(
        description=(
            "Convert a local LeRobot dataset into XR0 JSON/video training format. "
            "The converter splits chunk videos into per-episode ego / wrist_left / wrist_right MP4s."
        )
    )
    parser.add_argument("--lerobot-root", type=str, required=True, help="Local LeRobot dataset root.")
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(xr0_root / "data" / "lerobot_xr0"),
        help="Output dataset root that will contain json/ and videos/.",
    )
    parser.add_argument(
        "--xr0-root",
        type=str,
        default=str(xr0_root),
        help="XR0 project root used when emitting relative video paths in JSON.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Override the task text for every episode. If omitted, the converter uses meta/tasks.parquet.",
    )
    parser.add_argument("--fps", type=int, default=None, help="Override FPS. Defaults to meta/info.json fps.")
    parser.add_argument(
        "--trajectory-type",
        choices=("success", "ongoing", "invalid"),
        default="success",
        help="XR0 trajectory label written into every output JSON.",
    )
    parser.add_argument("--ego-camera-key", type=str, default="left_top", help="LeRobot camera key mapped to XR0 ego.")
    parser.add_argument(
        "--left-wrist-camera-key",
        type=str,
        default="left_wrist",
        help="LeRobot camera key mapped to XR0 wrist_left.",
    )
    parser.add_argument(
        "--right-wrist-camera-key",
        type=str,
        default="right_wrist",
        help="LeRobot camera key mapped to XR0 wrist_right.",
    )
    parser.add_argument("--absolute-video-paths", action="store_true", help="Write absolute video paths into JSON.")
    parser.add_argument(
        "--position-unit",
        choices=("mm", "m"),
        default="mm",
        help="XR0 EE position unit. PiPER FK returns mm; use --position-unit m to convert to meters.",
    )
    parser.add_argument(
        "--joint-unit",
        choices=("deg", "rad"),
        default="deg",
        help="Joint unit stored in LeRobot observation.state and action arrays.",
    )
    parser.add_argument("--video-ext", choices=("mp4",), default="mp4")
    parser.add_argument(
        "--writer",
        choices=("auto", "opencv", "imageio"),
        default="auto",
        help="Video writer backend for episode MP4s.",
    )
    parser.add_argument("--episode-start", type=int, default=None, help="Keep only episodes >= this index.")
    parser.add_argument("--episode-end", type=int, default=None, help="Keep only episodes <= this index.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Keep at most this many episodes after filtering.")
    parser.add_argument(
        "--piper-sdk-root",
        type=str,
        default=None,
        help=(
            "Optional path used to locate piper_sdk/kinematics/piper_fk.py directly. "
            "This avoids mixing another environment's full site-packages into sys.path."
        ),
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


def fk_pose_from_joint(fk_solver, joint_values: np.ndarray, joint_unit: str, position_unit: str) -> tuple[np.ndarray, np.ndarray]:
    joint_values = np.asarray(joint_values, dtype=np.float64)
    if joint_unit == "deg":
        joint_rad = np.deg2rad(joint_values)
    else:
        joint_rad = joint_values
    end_pose = fk_solver.CalFK(joint_rad.tolist())[-1]
    pos = np.asarray(end_pose[:3], dtype=np.float32)
    if position_unit == "m":
        pos *= 1e-3
    rotm = euler_rpy_deg_to_rotm(*end_pose[3:6])
    return pos, rotm


@dataclass(frozen=True)
class ArmLayout:
    joint_indices: tuple[int, ...]
    gripper_index: int


@dataclass(frozen=True)
class FeatureLayout:
    left: ArmLayout
    right: ArmLayout


@dataclass
class SourceChunkPlan:
    parquet_path: Path
    assignments: list[int | None]


@dataclass
class EpisodeRecord:
    episode_index: int
    task_index: int | None = None
    frame_indices: list[int] = field(default_factory=list)
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
    video_counts: dict[str, int] = field(
        default_factory=lambda: {
            "ego": 0,
            "wrist_left": 0,
            "wrist_right": 0,
        }
    )

    @property
    def num_frames(self) -> int:
        return len(self.frame_indices)

    def set_task_index(self, task_index: Any) -> None:
        if task_index is None:
            return
        try:
            value = int(task_index)
        except (TypeError, ValueError):
            return
        if self.task_index is None:
            self.task_index = value
        elif self.task_index != value:
            raise ValueError(
                f"Episode {self.episode_index} has multiple task_index values: {self.task_index} vs {value}"
            )


class StreamingVideoWriter:
    def __init__(self, path: Path, fps: int, backend: str):
        self.path = path
        self.fps = int(fps)
        self.backend = backend
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = None
        self._shape: tuple[int, int] | None = None
        self.count = 0

    def _open(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        self._shape = (height, width)
        if self.backend == "opencv":
            import cv2

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(self.path), fourcc, float(self.fps), (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open OpenCV writer for {self.path}")
            self._writer = writer
            return

        import imageio.v2 as imageio

        self._writer = imageio.get_writer(
            str(self.path),
            format="FFMPEG",
            fps=self.fps,
            codec="libx264",
            macro_block_size=None,
        )

    def write(self, frame: np.ndarray) -> None:
        frame = np.asarray(frame, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected HWC RGB frame, got {frame.shape}")
        if self._writer is None:
            self._open(frame)
        if self._shape != frame.shape[:2]:
            raise ValueError(
                f"Frame size changed for {self.path}: expected {self._shape}, got {frame.shape[:2]}"
            )
        if self.backend == "opencv":
            import cv2

            self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        else:
            self._writer.append_data(frame)
        self.count += 1

    def close(self) -> None:
        if self._writer is None:
            return
        self._writer.close() if self.backend == "imageio" else self._writer.release()
        self._writer = None


class EpisodeVideoSinks:
    def __init__(self, paths: dict[str, Path], fps: int, backend: str):
        self.sinks = {
            view: StreamingVideoWriter(path, fps=fps, backend=backend)
            for view, path in paths.items()
        }

    def write(self, view: str, frame: np.ndarray) -> None:
        self.sinks[view].write(frame)

    def close(self) -> None:
        for sink in self.sinks.values():
            sink.close()


def choose_writer_backend(preferred_writer: str) -> str:
    if preferred_writer in {"auto", "opencv"}:
        try:
            import cv2  # noqa: F401

            return "opencv"
        except ModuleNotFoundError:
            pass

    if preferred_writer in {"auto", "imageio"}:
        try:
            import imageio_ffmpeg  # noqa: F401

            return "imageio"
        except ModuleNotFoundError:
            if shutil.which("ffmpeg") is not None:
                return "imageio"

    raise RuntimeError(
        "Unable to write MP4 videos. Install opencv-python-headless, or install imageio-ffmpeg / ffmpeg."
    )


def iter_video_frames(video_path: Path):
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()
        return
    except ModuleNotFoundError:
        pass

    import imageio.v2 as imageio

    reader = imageio.get_reader(str(video_path))
    try:
        for frame in reader:
            yield np.asarray(frame, dtype=np.uint8)
    finally:
        reader.close()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_task_map(meta_dir: Path) -> dict[int, str]:
    task_path = meta_dir / "tasks.parquet"
    if not task_path.is_file():
        return {}

    import pyarrow.parquet as pq

    rows = pq.read_table(task_path).to_pylist()
    task_map: dict[int, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue

        task_index = row.get("task_index")
        if task_index is None:
            for key, value in row.items():
                if key.endswith("index") and isinstance(value, (int, np.integer)):
                    task_index = value
                    break

        task_text = None
        for key in ("task", "text", "instruction", "description", "task_description", "name"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                task_text = value.strip()
                break

        if task_index is None or task_text is None:
            continue
        task_map[int(task_index)] = task_text
    return task_map


def feature_names(info: dict[str, Any], key: str) -> list[str] | None:
    feature = info.get("features", {}).get(key)
    if not isinstance(feature, dict):
        return None
    for candidate in ("names", "feature_names", "columns", "keys"):
        names = feature.get(candidate)
        if isinstance(names, list) and all(isinstance(name, str) for name in names):
            return [str(name) for name in names]
    return None


def infer_feature_layout(names: list[str] | None, fallback_dim: int | None = None) -> FeatureLayout:
    if names:
        joint_index_map: dict[str, list[tuple[int, int]]] = {"left": [], "right": []}
        gripper_index_map: dict[str, list[int]] = {"left": [], "right": []}
        joint_pattern = re.compile(r"^(left|right).*joint[_-]?(\d+)", re.IGNORECASE)
        gripper_pattern = re.compile(r"^(left|right).*gripper", re.IGNORECASE)

        for index, raw_name in enumerate(names):
            name = raw_name.strip().lower().replace(" ", "")
            joint_match = joint_pattern.match(name)
            if joint_match:
                side = joint_match.group(1)
                order = int(joint_match.group(2))
                joint_index_map[side].append((order, index))
                continue

            gripper_match = gripper_pattern.match(name)
            if gripper_match:
                side = gripper_match.group(1)
                gripper_index_map[side].append(index)

        if all(len(joint_index_map[side]) == 6 for side in ("left", "right")) and all(
            len(gripper_index_map[side]) == 1 for side in ("left", "right")
        ):
            return FeatureLayout(
                left=ArmLayout(
                    joint_indices=tuple(index for _, index in sorted(joint_index_map["left"])),
                    gripper_index=gripper_index_map["left"][0],
                ),
                right=ArmLayout(
                    joint_indices=tuple(index for _, index in sorted(joint_index_map["right"])),
                    gripper_index=gripper_index_map["right"][0],
                ),
            )

    if fallback_dim == 14:
        return FeatureLayout(
            left=ArmLayout(joint_indices=(0, 1, 2, 3, 4, 5), gripper_index=6),
            right=ArmLayout(joint_indices=(7, 8, 9, 10, 11, 12), gripper_index=13),
        )

    raise ValueError(
        "Unable to infer left/right joint-gripper layout from LeRobot feature names. "
        f"names={names!r}, fallback_dim={fallback_dim!r}"
    )


def slice_arm(values: np.ndarray, arm_layout: ArmLayout) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    joint = values[list(arm_layout.joint_indices)].astype(np.float32)
    gripper = values[[arm_layout.gripper_index]].astype(np.float32)
    return joint, gripper


def append_arm_series(
    dest: dict[str, list[list[float]]],
    side: str,
    joint: np.ndarray,
    gripper: np.ndarray,
    fk_solver,
    joint_unit: str,
    position_unit: str,
) -> None:
    ee_pos, ee_rotm = fk_pose_from_joint(fk_solver, joint, joint_unit=joint_unit, position_unit=position_unit)
    dest[f"{side}_ee_pos"].append(ee_pos.astype(np.float32).tolist())
    dest[f"{side}_ee_rotm"].append(ee_rotm.reshape(-1).astype(np.float32).tolist())
    dest[f"{side}_arm_joint"].append(np.asarray(joint, dtype=np.float32).tolist())
    dest[f"{side}_gripper_pos"].append(np.asarray(gripper, dtype=np.float32).reshape(1).tolist())


def resolve_task_text(task_override: str | None, task_map: dict[int, str], record: EpisodeRecord) -> str:
    if task_override:
        return task_override
    if record.task_index is not None and record.task_index in task_map:
        return task_map[record.task_index]
    return f"Replay the bimanual teleoperation trajectory from episode {record.episode_index}."


def episode_selected(episode_index: int, selected_order: list[int], args: argparse.Namespace) -> bool:
    if args.episode_start is not None and episode_index < int(args.episode_start):
        return False
    if args.episode_end is not None and episode_index > int(args.episode_end):
        return False
    if episode_index in selected_order:
        return True
    if args.max_episodes is not None and len(selected_order) >= int(args.max_episodes):
        return False
    selected_order.append(episode_index)
    return True


def output_episode_id(episode_index: int) -> str:
    return f"episode_{episode_index:06d}"


def episode_video_paths(output_root: Path, episode_index: int, video_ext: str) -> dict[str, Path]:
    episode_id = output_episode_id(episode_index)
    videos_dir = output_root / "videos"
    return {
        "ego": videos_dir / f"{episode_id}_ego.{video_ext}",
        "wrist_left": videos_dir / f"{episode_id}_wrist_left.{video_ext}",
        "wrist_right": videos_dir / f"{episode_id}_wrist_right.{video_ext}",
    }


def source_video_path(lerobot_root: Path, camera_key: str, parquet_path: Path) -> Path:
    chunk_dir = parquet_path.parent.name
    stem = parquet_path.stem
    return lerobot_root / "videos" / f"observation.images.{camera_key}" / chunk_dir / f"{stem}.mp4"


def episode_json(
    record: EpisodeRecord,
    task: str,
    fps: int,
    output_root: Path,
    xr0_root: Path,
    video_ext: str,
    trajectory_type: str,
    absolute_video_paths: bool,
) -> dict[str, object]:
    episode_id = output_episode_id(record.episode_index)
    video_paths = episode_video_paths(output_root, record.episode_index, video_ext)
    observations = {
        view: [
            {
                "path": relative_video_path(path, xr0_root, absolute_video_paths),
                "start": 0,
                "end": record.num_frames,
                "fps": int(fps),
                "crop_bbox": None,
            }
        ]
        for view, path in video_paths.items()
    }
    return {
        "trajectory_type": trajectory_type,
        "time": episode_id,
        "num_frames": int(record.num_frames),
        "instruction": instruction_payload(task),
        "observations": observations,
        "proprios": record.proprios,
        "actions": record.actions,
    }


def verify_episode_order_contiguous(assignments: list[int | None], closed_episodes: set[int], last_episode: int | None) -> int | None:
    for episode_index in assignments:
        if episode_index is None:
            continue
        if episode_index != last_episode:
            if episode_index in closed_episodes:
                raise ValueError(
                    "Encountered a non-contiguous episode while splitting videos. "
                    "This converter assumes each episode occupies one continuous frame range across chunk files."
                )
            if last_episode is not None:
                closed_episodes.add(last_episode)
            last_episode = episode_index
    return last_episode


def main() -> None:
    args = parse_args()
    C_PiperForwardKinematics = load_fk_class(args.piper_sdk_root)

    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Unable to import pyarrow. Please install pyarrow in the current environment.") from exc

    lerobot_root = Path(args.lerobot_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    xr0_root = Path(args.xr0_root).expanduser().resolve()
    meta_dir = lerobot_root / "meta"
    data_dir = lerobot_root / "data"
    json_dir = output_root / "json"
    videos_dir = output_root / "videos"

    info_path = meta_dir / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot metadata not found: {info_path}")
    info = load_json(info_path)
    fps = int(args.fps if args.fps is not None else info.get("fps", 30))

    parquet_files = sorted(data_dir.glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {data_dir}")

    state_names = feature_names(info, "observation.state")
    action_names = feature_names(info, "action")
    state_dim = None
    action_dim = None
    if isinstance(info.get("features", {}).get("observation.state"), dict):
        state_dim = info["features"]["observation.state"].get("shape", [None])[0]
    if isinstance(info.get("features", {}).get("action"), dict):
        action_dim = info["features"]["action"].get("shape", [None])[0]
    state_layout = infer_feature_layout(state_names, fallback_dim=state_dim)
    action_layout = infer_feature_layout(action_names, fallback_dim=action_dim)
    task_map = load_task_map(meta_dir)
    writer_backend = choose_writer_backend(args.writer)
    fk_solver = C_PiperForwardKinematics()

    json_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    selected_order: list[int] = []
    episode_records: dict[int, EpisodeRecord] = {}
    source_plans: list[SourceChunkPlan] = []
    closed_episodes: set[int] = set()
    last_selected_episode: int | None = None

    for parquet_path in parquet_files:
        available_columns = set(pq.read_schema(parquet_path).names)
        required_columns = ["episode_index", "observation.state", "action"]
        missing_columns = [column for column in required_columns if column not in available_columns]
        if missing_columns:
            raise ValueError(f"{parquet_path} is missing required columns: {missing_columns}")
        selected_columns = required_columns + [
            column for column in ("frame_index", "task_index") if column in available_columns
        ]
        rows = pq.read_table(parquet_path, columns=selected_columns).to_pylist()
        assignments: list[int | None] = []

        for row in rows:
            episode_index = int(row["episode_index"])
            keep = episode_selected(episode_index, selected_order, args)
            assignments.append(episode_index if keep else None)
            if not keep:
                continue

            record = episode_records.setdefault(episode_index, EpisodeRecord(episode_index=episode_index))
            record.set_task_index(row.get("task_index"))
            frame_index = int(row.get("frame_index", record.num_frames))
            record.frame_indices.append(frame_index)

            state_row = np.asarray(row["observation.state"], dtype=np.float32)
            action_row = np.asarray(row["action"], dtype=np.float32)
            left_joint, left_gripper = slice_arm(state_row, state_layout.left)
            right_joint, right_gripper = slice_arm(state_row, state_layout.right)
            left_target_joint, left_target_gripper = slice_arm(action_row, action_layout.left)
            right_target_joint, right_target_gripper = slice_arm(action_row, action_layout.right)

            append_arm_series(
                record.proprios,
                "left",
                left_joint,
                left_gripper,
                fk_solver,
                joint_unit=args.joint_unit,
                position_unit=args.position_unit,
            )
            append_arm_series(
                record.proprios,
                "right",
                right_joint,
                right_gripper,
                fk_solver,
                joint_unit=args.joint_unit,
                position_unit=args.position_unit,
            )
            append_arm_series(
                record.actions,
                "left",
                left_target_joint,
                left_target_gripper,
                fk_solver,
                joint_unit=args.joint_unit,
                position_unit=args.position_unit,
            )
            append_arm_series(
                record.actions,
                "right",
                right_target_joint,
                right_target_gripper,
                fk_solver,
                joint_unit=args.joint_unit,
                position_unit=args.position_unit,
            )

        last_selected_episode = verify_episode_order_contiguous(assignments, closed_episodes, last_selected_episode)
        source_plans.append(SourceChunkPlan(parquet_path=parquet_path, assignments=assignments))

    if not selected_order:
        raise ValueError("No episodes matched the requested filters.")

    current_episode: int | None = None
    current_sinks: EpisodeVideoSinks | None = None
    view_to_camera_key = {
        "ego": args.ego_camera_key,
        "wrist_left": args.left_wrist_camera_key,
        "wrist_right": args.right_wrist_camera_key,
    }

    for plan in source_plans:
        source_videos = {
            view: source_video_path(lerobot_root, camera_key, plan.parquet_path)
            for view, camera_key in view_to_camera_key.items()
        }
        for view, video_path in source_videos.items():
            if not video_path.is_file():
                raise FileNotFoundError(f"Expected source video for {view}: {video_path}")

        frame_iters = {view: iter_video_frames(video_path) for view, video_path in source_videos.items()}
        try:
            for row_index, episode_index in enumerate(plan.assignments):
                frames = {}
                for view, frame_iter in frame_iters.items():
                    try:
                        frames[view] = next(frame_iter)
                    except StopIteration as exc:
                        raise RuntimeError(
                            f"Video {source_videos[view]} ended before row {row_index} in {plan.parquet_path}"
                        ) from exc

                if episode_index is None:
                    continue

                if episode_index != current_episode:
                    if current_sinks is not None:
                        current_sinks.close()
                    current_episode = episode_index
                    current_sinks = EpisodeVideoSinks(
                        paths=episode_video_paths(output_root, episode_index, args.video_ext),
                        fps=fps,
                        backend=writer_backend,
                    )

                record = episode_records[episode_index]
                for view, frame in frames.items():
                    current_sinks.write(view, frame)
                    record.video_counts[view] += 1

            for view, frame_iter in frame_iters.items():
                try:
                    next(frame_iter)
                except StopIteration:
                    continue
                raise RuntimeError(
                    f"Video {source_videos[view]} has more frames than parquet rows in {plan.parquet_path}"
                )
        finally:
            for frame_iter in frame_iters.values():
                close_fn = getattr(frame_iter, "close", None)
                if callable(close_fn):
                    close_fn()

    if current_sinks is not None:
        current_sinks.close()

    created_json: list[Path] = []
    for episode_index in selected_order:
        record = episode_records[episode_index]
        for view, count in record.video_counts.items():
            if count != record.num_frames:
                raise ValueError(
                    f"Episode {episode_index} view {view} wrote {count} frames, expected {record.num_frames}"
                )

        task_text = resolve_task_text(args.task, task_map, record)
        payload = episode_json(
            record=record,
            task=task_text,
            fps=fps,
            output_root=output_root,
            xr0_root=xr0_root,
            video_ext=args.video_ext,
            trajectory_type=args.trajectory_type,
            absolute_video_paths=args.absolute_video_paths,
        )
        json_path = json_dir / f"{output_episode_id(episode_index)}.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        created_json.append(json_path)

    print(f"Converted {len(created_json)} episode(s) from {lerobot_root}")
    print(f"Output root: {output_root}")
    print(f"JSON dir: {json_dir}")
    print(f"Videos dir: {videos_dir}")
    print(f"Video writer backend: {writer_backend}")
    print(f"State layout: {state_layout}")
    print(f"Action layout: {action_layout}")


if __name__ == "__main__":
    main()
