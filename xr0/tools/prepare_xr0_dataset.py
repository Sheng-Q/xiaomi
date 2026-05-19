#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


ACTION_DIM = 32


def add_sys_path(path: str | None) -> None:
    if not path:
        return
    resolved = str(Path(path).expanduser().resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    xr0_root = script_dir.parent
    parser = argparse.ArgumentParser(
        description="Compute XR0 action mean/std from JSON episodes and generate a runnable training config."
    )
    parser.add_argument("--dataset-root", type=str, required=True, help="Dataset root that contains json/ and videos/.")
    parser.add_argument("--json-dir", type=str, default=None, help="Override the JSON directory. Defaults to <dataset-root>/json.")
    parser.add_argument("--dataset-name", type=str, default=None, help="Name used in generated file names and trainer metadata.")
    parser.add_argument("--action-length", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--trainer-project", type=str, default=None)
    parser.add_argument("--trainer-exp-name", type=str, default=None)
    parser.add_argument("--trainer-default-root-dir", type=str, default="./runs")
    parser.add_argument("--trainer-max-steps", type=int, default=30000)
    parser.add_argument("--trainer-val-check-interval", type=int, default=2000)
    parser.add_argument("--trainer-save-interval", type=int, default=5000)
    parser.add_argument("--trainer-num-nodes", type=int, default=-1)
    parser.add_argument("--include-invalid", action="store_true", help="Include invalid episodes when computing stats.")
    parser.add_argument(
        "--min-std",
        type=float,
        default=0.0,
        help="Clamp non-reserved action std to at least this value after estimation.",
    )
    parser.add_argument("--absolute-train-path", action="store_true")
    parser.add_argument("--stats-out", type=str, default=None)
    parser.add_argument("--train-config-out", type=str, default=None)
    parser.add_argument("--xr0-root", type=str, default=str(xr0_root))
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", value.strip().lower()).strip("_")
    return normalized or "xr0_dataset"


def default_json_dir(dataset_root: Path) -> Path:
    candidate = dataset_root / "json"
    if candidate.is_dir():
        return candidate
    return dataset_root


def discover_json_files(json_dir: Path) -> list[Path]:
    if json_dir.is_file() and json_dir.suffix == ".json":
        return [json_dir]
    return sorted(path for path in json_dir.rglob("*.json") if path.is_file())


def load_traj(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def train_path_value(path: Path, xr0_root: Path, force_absolute: bool) -> str:
    if force_absolute:
        return str(path.resolve())
    try:
        return str(path.resolve().relative_to(xr0_root.resolve()))
    except ValueError:
        return str(path.resolve())


def validate_frame_count(name: str, values: np.ndarray, num_frames: int, width: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape[0] != num_frames:
        raise ValueError(f"{name} expected first dimension {num_frames}, got {array.shape}")
    if width > 0:
        array = array.reshape(num_frames, width)
    return array


def load_side_arrays(traj: dict[str, Any], side: str, num_frames: int) -> dict[str, np.ndarray]:
    proprios = traj["proprios"]
    actions = traj["actions"]
    return {
        "current_ee_pos": validate_frame_count(f"proprios.{side}_ee_pos", proprios[f"{side}_ee_pos"], num_frames, 3),
        "current_ee_rotm": validate_frame_count(
            f"proprios.{side}_ee_rotm", proprios[f"{side}_ee_rotm"], num_frames, 9
        ).reshape(num_frames, 3, 3),
        "current_gripper": validate_frame_count(
            f"proprios.{side}_gripper_pos", proprios[f"{side}_gripper_pos"], num_frames, 1
        ),
        "current_joint": validate_frame_count(
            f"proprios.{side}_arm_joint", proprios[f"{side}_arm_joint"], num_frames, 6
        ),
        "target_ee_pos": validate_frame_count(f"actions.{side}_ee_pos", actions[f"{side}_ee_pos"], num_frames, 3),
        "target_ee_rotm": validate_frame_count(
            f"actions.{side}_ee_rotm", actions[f"{side}_ee_rotm"], num_frames, 9
        ).reshape(num_frames, 3, 3),
        "target_gripper": validate_frame_count(
            f"actions.{side}_gripper_pos", actions[f"{side}_gripper_pos"], num_frames, 1
        ),
        "target_joint": validate_frame_count(
            f"actions.{side}_arm_joint", actions[f"{side}_arm_joint"], num_frames, 6
        ),
    }


def build_action_mask(action_length: int) -> np.ndarray:
    mask = np.zeros((action_length, ACTION_DIM), dtype=np.float32)
    for start, end in ((0, 3), (3, 6), (6, 7), (7, 13), (14, 17), (17, 20), (20, 21), (21, 27)):
        mask[:, start:end] = 1.0
    return mask


def axis_from_pi(rotm: np.ndarray) -> np.ndarray:
    rot00, rot11, rot22 = rotm[0, 0], rotm[1, 1], rotm[2, 2]
    if rot00 >= rot11 and rot00 >= rot22:
        axis = np.array([math.sqrt(max((rot00 + 1.0) / 2.0, 0.0)), 0.0, 0.0], dtype=np.float32)
        if axis[0] > 1e-8:
            axis[1] = rotm[0, 1] / (2.0 * axis[0])
            axis[2] = rotm[0, 2] / (2.0 * axis[0])
    elif rot11 >= rot22:
        axis = np.array([0.0, math.sqrt(max((rot11 + 1.0) / 2.0, 0.0)), 0.0], dtype=np.float32)
        if axis[1] > 1e-8:
            axis[0] = rotm[0, 1] / (2.0 * axis[1])
            axis[2] = rotm[1, 2] / (2.0 * axis[1])
    else:
        axis = np.array([0.0, 0.0, math.sqrt(max((rot22 + 1.0) / 2.0, 0.0))], dtype=np.float32)
        if axis[2] > 1e-8:
            axis[0] = rotm[0, 2] / (2.0 * axis[2])
            axis[1] = rotm[1, 2] / (2.0 * axis[2])
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return axis / norm


def rotm2aa_batch(rotms: np.ndarray) -> np.ndarray:
    rotms = np.asarray(rotms, dtype=np.float32)
    theta = np.arccos(np.clip((np.einsum("nii->n", rotms) - 1.0) / 2.0, -1.0, 1.0))
    axis_angle = np.zeros((rotms.shape[0], 3), dtype=np.float32)
    near_zero = theta <= 1e-6
    near_pi = np.abs(theta - np.pi) <= 1e-6
    normal = ~(near_zero | near_pi)

    if np.any(normal):
        axis = np.stack(
            [
                rotms[:, 2, 1] - rotms[:, 1, 2],
                rotms[:, 0, 2] - rotms[:, 2, 0],
                rotms[:, 1, 0] - rotms[:, 0, 1],
            ],
            axis=1,
        )
        axis /= np.linalg.norm(axis, axis=1, keepdims=True) + 1e-12
        axis_angle[normal] = axis[normal] * theta[normal, None]

    if np.any(near_pi):
        for index in np.where(near_pi)[0]:
            axis_angle[index] = axis_from_pi(rotms[index]) * theta[index]

    return axis_angle


def accumulate_side_into_action(
    side_arrays: dict[str, np.ndarray],
    side: str,
    timestep: int,
    sample_count: int,
    action_t: np.ndarray,
) -> None:
    current_rotm = side_arrays["current_ee_rotm"][:sample_count]
    current_rotm_t = np.transpose(current_rotm, (0, 2, 1))
    current_pos = side_arrays["current_ee_pos"][:sample_count]
    current_gripper = side_arrays["current_gripper"][:sample_count]
    current_joint = side_arrays["current_joint"][:sample_count]

    target_pos = side_arrays["target_ee_pos"][timestep : timestep + sample_count]
    target_rotm = side_arrays["target_ee_rotm"][timestep : timestep + sample_count]
    target_gripper = side_arrays["target_gripper"][timestep : timestep + sample_count]
    target_joint = side_arrays["target_joint"][timestep : timestep + sample_count]

    pos_delta = np.einsum("nij,nj->ni", current_rotm_t, target_pos - current_pos)
    rot_delta = np.matmul(current_rotm_t, target_rotm)
    aa_delta = rotm2aa_batch(rot_delta)
    gripper_delta = target_gripper - current_gripper
    joint_delta = target_joint - current_joint

    if side == "left":
        action_t[:, 0:3] = pos_delta
        action_t[:, 3:6] = aa_delta
        action_t[:, 6:7] = gripper_delta
        action_t[:, 7:13] = joint_delta
    else:
        action_t[:, 14:17] = pos_delta
        action_t[:, 17:20] = aa_delta
        action_t[:, 20:21] = gripper_delta
        action_t[:, 21:27] = joint_delta


def accumulate_traj_stats(
    traj: dict[str, Any],
    action_length: int,
    sums: np.ndarray,
    sums_sq: np.ndarray,
) -> int:
    num_frames = int(traj["num_frames"])
    sample_count = num_frames - action_length + 1
    if sample_count <= 0:
        return 0

    left = load_side_arrays(traj, "left", num_frames)
    right = load_side_arrays(traj, "right", num_frames)

    for timestep in range(action_length):
        action_t = np.zeros((sample_count, ACTION_DIM), dtype=np.float32)
        accumulate_side_into_action(left, "left", timestep, sample_count, action_t)
        accumulate_side_into_action(right, "right", timestep, sample_count, action_t)
        sums[timestep] += action_t.sum(axis=0, dtype=np.float64)
        sums_sq[timestep] += np.square(action_t, dtype=np.float64).sum(axis=0, dtype=np.float64)

    return sample_count


def compute_stats(files: list[Path], action_length: int, include_invalid: bool, min_std: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    sums = np.zeros((action_length, ACTION_DIM), dtype=np.float64)
    sums_sq = np.zeros((action_length, ACTION_DIM), dtype=np.float64)
    sample_count = 0
    used_files = 0
    skipped_short = 0
    skipped_invalid = 0
    trajectory_counts: dict[str, int] = {}

    for path in files:
        traj = load_traj(path)
        kind = str(traj.get("trajectory_type", "success"))
        trajectory_counts[kind] = trajectory_counts.get(kind, 0) + 1
        if kind == "invalid" and not include_invalid:
            skipped_invalid += 1
            continue

        added = accumulate_traj_stats(traj, action_length, sums, sums_sq)
        if added <= 0:
            skipped_short += 1
            continue
        used_files += 1
        sample_count += added

    if sample_count <= 0:
        raise RuntimeError("No valid training samples found. Check episode lengths and filters.")

    mean = (sums / float(sample_count)).astype(np.float32)
    variance = (sums_sq / float(sample_count)) - np.square(mean, dtype=np.float64)
    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance).astype(np.float32)

    if min_std > 0:
        mask = build_action_mask(action_length).astype(bool)
        std = np.where(mask, np.maximum(std, np.float32(min_std)), 0.0).astype(np.float32)

    summary = {
        "sample_count": sample_count,
        "used_files": used_files,
        "total_files": len(files),
        "skipped_short": skipped_short,
        "skipped_invalid": skipped_invalid,
        "trajectory_counts": trajectory_counts,
    }
    return mean, std, summary


def dump_yaml_or_json(path: Path, payload: dict[str, Any]) -> None:
    try:
        import yaml  # type: ignore

        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    except Exception:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_train_config(
    dataset_name: str,
    json_dir: Path,
    xr0_root: Path,
    args: argparse.Namespace,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, Any]:
    train_path = train_path_value(json_dir, xr0_root, args.absolute_train_path)
    return {
        "defaults": ["_self_", {"model": "XR0"}, {"trainer": "deepspeed"}],
        "data": {
            "type": "BaseDataModule",
            "params": {
                "type": "json",
                "max_steps": "${trainer.max_steps}",
                "train_datasets": {
                    "batch_size": int(args.batch_size),
                    "action_length": int(args.action_length),
                    "train_path": [train_path],
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                },
            },
        },
        "model": {
            "params": {
                "model": {
                    "action_shape": [int(args.action_length), ACTION_DIM],
                }
            }
        },
        "trainer": {
            "project": args.trainer_project or dataset_name,
            "exp_name": args.trainer_exp_name or dataset_name,
            "default_root_dir": args.trainer_default_root_dir,
            "max_steps": int(args.trainer_max_steps),
            "val_check_interval": int(args.trainer_val_check_interval),
            "save_interval": int(args.trainer_save_interval),
            "num_nodes": int(args.trainer_num_nodes),
        },
    }


def main() -> None:
    args = parse_args()
    xr0_root = Path(args.xr0_root).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    json_dir = Path(args.json_dir).expanduser().resolve() if args.json_dir else default_json_dir(dataset_root).resolve()
    files = discover_json_files(json_dir)
    if not files:
        raise FileNotFoundError(f"No JSON files found under {json_dir}")

    dataset_name = sanitize_name(args.dataset_name or dataset_root.name)
    stats_out = Path(args.stats_out).expanduser().resolve() if args.stats_out else dataset_root / f"stats_action{args.action_length}.json"
    train_config_out = (
        Path(args.train_config_out).expanduser().resolve()
        if args.train_config_out
        else xr0_root / "configs" / f"train_{dataset_name}.yaml"
    )

    mean, std, summary = compute_stats(files, args.action_length, args.include_invalid, args.min_std)
    train_config = build_train_config(dataset_name, json_dir, xr0_root, args, mean, std)

    stats_out.parent.mkdir(parents=True, exist_ok=True)
    train_config_out.parent.mkdir(parents=True, exist_ok=True)

    stats_payload = {
        "dataset_name": dataset_name,
        "dataset_root": str(dataset_root),
        "json_dir": str(json_dir),
        "action_length": int(args.action_length),
        "batch_size": int(args.batch_size),
        "include_invalid": bool(args.include_invalid),
        "min_std": float(args.min_std),
        **summary,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    with stats_out.open("w", encoding="utf-8") as handle:
        json.dump(stats_payload, handle, ensure_ascii=False, indent=2)

    dump_yaml_or_json(train_config_out, train_config)

    print(f"Found {summary['total_files']} json files under {json_dir}")
    print(
        f"Used {summary['used_files']} episodes -> {summary['sample_count']} training windows "
        f"(action_length={args.action_length})"
    )
    print(f"Wrote stats: {stats_out}")
    print(f"Wrote train config: {train_config_out}")
    print("Run training with:")
    print(f"  cd {xr0_root}")
    print(f"  python tools/train.py --config-name {train_config_out.stem}")


if __name__ == "__main__":
    main()
