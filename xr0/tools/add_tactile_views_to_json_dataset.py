#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


DEFAULT_IMAGE_KEYS = [
    "observations.ego",
    "observations.wrist_left",
    "observations.wrist_right",
]
TACTILE_IMAGE_KEYS = [
    "observations.tactile_left",
    "observations.tactile_right",
]
TASK_MARKER = "Generate robot actions for the task:\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment XR0 JSON episodes with left/right tactile heatmap views for VLM smoke tests."
    )
    parser.add_argument("--json-dir", type=str, required=True, help="Source XR0 JSON file or directory.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for augmented JSON files. Defaults to <json-dir>_tactile unless --in-place is used.",
    )
    parser.add_argument("--in-place", action="store_true", help="Modify the source JSON files in place.")
    parser.add_argument(
        "--left-source",
        choices=("reuse_wrist_left", "reuse_wrist_right", "reuse_ego", "replace_wrist_left_suffix"),
        default="reuse_wrist_left",
        help="How to populate observations.tactile_left.",
    )
    parser.add_argument(
        "--right-source",
        choices=("reuse_wrist_left", "reuse_wrist_right", "reuse_ego", "replace_wrist_right_suffix"),
        default="reuse_wrist_right",
        help="How to populate observations.tactile_right.",
    )
    parser.add_argument("--left-replace-from", type=str, default="_wrist_left.mp4")
    parser.add_argument("--left-replace-to", type=str, default="_tactile_left.mp4")
    parser.add_argument("--right-replace-from", type=str, default="_wrist_right.mp4")
    parser.add_argument("--right-replace-to", type=str, default="_tactile_right.mp4")
    parser.add_argument("--left-title", type=str, default="Left-Gripper Tactile Heatmap")
    parser.add_argument("--right-title", type=str, default="Right-Gripper Tactile Heatmap")
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="When using *_replace_* modes, verify the derived tactile video files exist.",
    )
    parser.add_argument(
        "--xr0-root",
        type=str,
        default=str(Path(__file__).resolve().parent.parent),
        help="XR0 project root used to resolve relative video paths during --check-files.",
    )
    return parser.parse_args()


def discover_json_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".json":
            raise ValueError(f"Expected a JSON file, got: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(item for item in path.rglob("*.json") if item.is_file())


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def copy_observation_list(traj: dict[str, Any], key: str) -> list[dict[str, Any]]:
    observations = traj.get("observations", {})
    if key not in observations:
        raise KeyError(f"Missing observations.{key}")
    return copy.deepcopy(observations[key])


def replace_video_suffix(
    entries: list[dict[str, Any]],
    old_suffix: str,
    new_suffix: str,
) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for info in copy.deepcopy(entries):
        path_value = str(info.get("path", ""))
        if not path_value.endswith(old_suffix):
            raise ValueError(f"Video path does not end with {old_suffix!r}: {path_value}")
        info["path"] = path_value[: -len(old_suffix)] + new_suffix
        updated.append(info)
    return updated


def build_tactile_entries(
    traj: dict[str, Any],
    source: str,
    old_suffix: str,
    new_suffix: str,
) -> list[dict[str, Any]]:
    if source == "reuse_wrist_left":
        return copy_observation_list(traj, "wrist_left")
    if source == "reuse_wrist_right":
        return copy_observation_list(traj, "wrist_right")
    if source == "reuse_ego":
        return copy_observation_list(traj, "ego")
    if source == "replace_wrist_left_suffix":
        return replace_video_suffix(copy_observation_list(traj, "wrist_left"), old_suffix, new_suffix)
    if source == "replace_wrist_right_suffix":
        return replace_video_suffix(copy_observation_list(traj, "wrist_right"), old_suffix, new_suffix)
    raise ValueError(f"Unsupported tactile source: {source}")


def resolve_video_path(path_value: str, xr0_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return xr0_root / path


def validate_entries(entries: list[dict[str, Any]], xr0_root: Path) -> None:
    for info in entries:
        resolved = resolve_video_path(str(info["path"]), xr0_root)
        if not resolved.is_file():
            raise FileNotFoundError(f"Missing tactile video file: {resolved}")


def augment_text(text: str, left_title: str, right_title: str) -> str:
    left_marker = f"# {left_title}\n<image>"
    right_marker = f"# {right_title}\n<image>"
    if left_marker in text and right_marker in text:
        return text

    tactile_block = f"# {left_title}\n<image>\n# {right_title}\n<image>\n"
    if TASK_MARKER in text:
        prefix, suffix = text.split(TASK_MARKER, 1)
        prefix = prefix.rstrip("\n")
        return f"{prefix}\n{tactile_block}{TASK_MARKER}{suffix}"
    return f"{text.rstrip()}\n{tactile_block}"


def augment_instruction(traj: dict[str, Any], left_title: str, right_title: str) -> None:
    instruction = traj.get("instruction")
    if not isinstance(instruction, dict):
        raise ValueError("trajectory is missing instruction")
    prompts = instruction.get("general")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("trajectory is missing instruction.general")

    for prompt in prompts:
        existing_images = [item for item in prompt.get("images", []) if item not in TACTILE_IMAGE_KEYS]
        if not existing_images:
            existing_images = list(DEFAULT_IMAGE_KEYS)
        prompt["images"] = existing_images + list(TACTILE_IMAGE_KEYS)

        conversations = prompt.get("conversations", [])
        for turn in conversations:
            if turn.get("from") == "human" and isinstance(turn.get("value"), str):
                turn["value"] = augment_text(turn["value"], left_title, right_title)
                break
        else:
            raise ValueError("instruction.general[*].conversations is missing a human turn")


def build_output_path(source_root: Path, output_root: Path, path: Path) -> Path:
    if source_root.is_file():
        return output_root
    return output_root / path.relative_to(source_root)


def main() -> None:
    args = parse_args()
    source_path = Path(args.json_dir).expanduser().resolve()
    xr0_root = Path(args.xr0_root).expanduser().resolve()
    files = discover_json_files(source_path)
    if not files:
        raise FileNotFoundError(f"No JSON files found under {source_path}")

    if args.in_place and args.output_dir:
        raise ValueError("--output-dir cannot be used together with --in-place")
    if not args.in_place:
        if args.output_dir:
            output_root = Path(args.output_dir).expanduser().resolve()
        elif source_path.is_file():
            output_root = source_path.with_name(source_path.stem + "_tactile.json")
        else:
            output_root = source_path.parent / f"{source_path.name}_tactile"
    else:
        output_root = source_path

    updated_count = 0
    for path in files:
        traj = load_json(path)
        left_entries = build_tactile_entries(
            traj,
            args.left_source,
            args.left_replace_from,
            args.left_replace_to,
        )
        right_entries = build_tactile_entries(
            traj,
            args.right_source,
            args.right_replace_from,
            args.right_replace_to,
        )
        if args.check_files:
            validate_entries(left_entries, xr0_root)
            validate_entries(right_entries, xr0_root)

        observations = traj.setdefault("observations", {})
        observations["tactile_left"] = left_entries
        observations["tactile_right"] = right_entries
        augment_instruction(traj, args.left_title, args.right_title)

        target = path if args.in_place else build_output_path(source_path, output_root, path)
        save_json(target, traj)
        updated_count += 1

    print(f"Updated {updated_count} episode JSON file(s).")
    if args.in_place:
        print("Mode: in-place")
    else:
        print(f"Output: {output_root}")


if __name__ == "__main__":
    main()
