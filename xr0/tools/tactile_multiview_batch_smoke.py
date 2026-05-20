#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
        description="Lightweight smoke test: verify XR0 dataset loading + VLM processor work with 5 image views."
    )
    parser.add_argument("--json-dir", type=str, required=True, help="XR0 JSON directory after tactile augmentation.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--action-length", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=1, help="Only used to satisfy JsonDataset construction.")
    parser.add_argument("--xr0-root", type=str, default=str(xr0_root))
    return parser.parse_args()


def count_images(messages: list[dict]) -> int:
    total = 0
    for turn in messages:
        for part in turn.get("content", []):
            if isinstance(part, dict) and part.get("type") == "image":
                total += 1
    return total


def summarize_shape(name: str, value: object) -> str:
    shape = getattr(value, "shape", None)
    return f"{name}={tuple(shape)}" if shape is not None else f"{name}=<no-shape>"


def main() -> None:
    args = parse_args()
    xr0_root = Path(args.xr0_root).expanduser().resolve()
    add_sys_path(str(xr0_root))

    from mibot.data.collate.custom_collate import CustomCollate
    from mibot.data.datasets.json_dataset import JsonDataset

    mean = np.zeros((args.action_length, ACTION_DIM), dtype=np.float32)
    std = np.ones((args.action_length, ACTION_DIM), dtype=np.float32)
    params = {
        "max_steps": int(args.max_steps),
        "train_datasets": {
            "batch_size": int(args.batch_size),
            "action_length": int(args.action_length),
            "train_path": [str(Path(args.json_dir).expanduser().resolve())],
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
    }

    dataset = JsonDataset(params)
    if len(dataset) == 0:
        raise RuntimeError("Dataset produced zero samples. Check num_frames and action_length.")

    batch_size = max(1, min(int(args.batch_size), len(dataset)))
    batch = [dataset[index] for index in range(batch_size)]
    image_count = count_images(batch[0]["messages"])
    collate = CustomCollate()
    inputs = collate(batch)

    print(f"dataset_files={len(dataset.files)}")
    print(f"dataset_samples={len(dataset)}")
    print(f"batch_size={batch_size}")
    print(f"images_per_sample={image_count}")
    print(summarize_shape("input_ids", inputs.get("input_ids")))
    if "pixel_values" in inputs:
        print(summarize_shape("pixel_values", inputs["pixel_values"]))
    if "image_grid_thw" in inputs:
        print(summarize_shape("image_grid_thw", inputs["image_grid_thw"]))
    print(summarize_shape("state", inputs.get("state")))
    print(summarize_shape("action", inputs.get("action")))
    print(summarize_shape("action_mask", inputs.get("action_mask")))

    if image_count < 5:
        raise RuntimeError(f"Expected at least 5 images per sample after tactile augmentation, got {image_count}")

    print("Smoke test passed: XR0 preprocessing accepted the 5-view prompt.")


if __name__ == "__main__":
    main()
