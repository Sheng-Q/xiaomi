# Copyright (C) 2026 Xiaomi Corporation.
"""
Convert HuggingFace model weights to PyTorch format.

Preferred path:
1. If --model_path points to a local directory containing safetensors files,
   read those shards directly and save a `{"module": ...}` checkpoint with
   the `model.` prefix expected by XR0 training.
2. Otherwise, fall back to transformers AutoModel loading.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch


def log(message: str) -> None:
    print(message, flush=True)


def output_path_for(output_dir: Path, output_filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / output_filename


def summarize_state_dict(state_dict: dict[str, torch.Tensor]) -> None:
    total_params = 0
    dtype_counts: dict[str, int] = {}
    for tensor in state_dict.values():
        total_params += tensor.numel()
        key = str(tensor.dtype)
        dtype_counts[key] = dtype_counts.get(key, 0) + 1

    log(f"Total tensors: {len(state_dict)}")
    log(f"Total parameters: {total_params:,}")
    log(f"Dtype breakdown: {dtype_counts}")


def save_prefixed_state_dict(state_dict: dict[str, torch.Tensor], output_dir: Path, output_filename: str) -> Path:
    prefixed = {f"model.{name}": tensor for name, tensor in state_dict.items()}
    summarize_state_dict(prefixed)

    output_path = output_path_for(output_dir, output_filename)
    log(f"Saving to: {output_path}")
    torch.save({"module": prefixed}, str(output_path))
    file_size = output_path.stat().st_size / (1024**3)
    log(f"Successfully saved!")
    log(f"File size: {file_size:.2f} GB")
    return output_path


def load_index_json(index_path: Path) -> dict[str, object]:
    with index_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def shard_files_from_index(model_dir: Path, index_path: Path) -> list[Path]:
    payload = load_index_json(index_path)
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{index_path} does not contain a valid weight_map")

    seen: set[str] = set()
    shard_names: list[str] = []
    for shard_name in weight_map.values():
        if not isinstance(shard_name, str):
            raise ValueError(f"{index_path} contains a non-string shard entry: {shard_name!r}")
        if shard_name not in seen:
            seen.add(shard_name)
            shard_names.append(shard_name)

    shard_paths = [model_dir / shard_name for shard_name in shard_names]
    missing = [str(path) for path in shard_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing shard files referenced by {index_path}: {missing}")
    return shard_paths


def load_local_safetensors(model_dir: Path) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("safetensors is required for local shard conversion.") from exc

    index_path = model_dir / "model.safetensors.index.json"
    single_path = model_dir / "model.safetensors"

    if index_path.is_file():
        shard_paths = shard_files_from_index(model_dir, index_path)
        log("=" * 60)
        log("Loading local safetensors shards...")
        log("=" * 60)
        log(f"Model dir: {model_dir}")
        log(f"Index file: {index_path}")
        log(f"Found {len(shard_paths)} shard(s)")

        state_dict: dict[str, torch.Tensor] = {}
        for shard_index, shard_path in enumerate(shard_paths, start=1):
            log(f"Loading shard {shard_index}/{len(shard_paths)}: {shard_path.name}")
            shard_state = load_file(str(shard_path), device="cpu")
            overlap = set(state_dict).intersection(shard_state)
            if overlap:
                preview = sorted(overlap)[:5]
                raise ValueError(f"Duplicate tensor keys across shards, e.g. {preview}")
            state_dict.update(shard_state)
        return state_dict

    if single_path.is_file():
        log("=" * 60)
        log("Loading local safetensors file...")
        log("=" * 60)
        log(f"Model file: {single_path}")
        return load_file(str(single_path), device="cpu")

    raise FileNotFoundError(
        f"No local safetensors weights found under {model_dir}. "
        "Expected model.safetensors.index.json or model.safetensors."
    )


def load_via_transformers(model_path: str) -> dict[str, torch.Tensor]:
    from transformers import AutoModel

    log("=" * 60)
    log("Loading model with AutoModel...")
    log("=" * 60)
    log(f"Model path: {model_path}")
    log("Loading model (using AutoModel with flash_attention_2)...")

    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    return model.state_dict()


def load_and_convert_model(model_path: str, output_dir: Path, output_filename: str = "pytorch_model.pt") -> Path:
    model_dir = Path(model_path).expanduser().resolve()

    if model_dir.is_dir():
        state_dict = load_local_safetensors(model_dir)
        return save_prefixed_state_dict(state_dict, output_dir, output_filename)

    state_dict = load_via_transformers(model_path)
    return save_prefixed_state_dict(state_dict, output_dir, output_filename)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HuggingFace model weights to PyTorch format.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="XiaomiRobotics/Xiaomi-Robotics-0-Pretrain",
        help="HuggingFace repo ID or local model directory",
    )
    parser.add_argument("--output_dir", type=str, default="./pretrained_ckpt", help="Output directory for the PyTorch model")
    parser.add_argument("--output_filename", type=str, default="xr0_pretrained.pt", help="Output filename")
    args = parser.parse_args()

    model_path = args.model_path
    output_dir = Path(args.output_dir).expanduser().resolve()

    log("=" * 60)
    log("HuggingFace Model to PyTorch Converter")
    log("=" * 60)
    log(f"Input Model: {model_path}")
    log(f"Output directory: {output_dir}")

    try:
        output_path = load_and_convert_model(model_path, output_dir, args.output_filename)
        log("\n" + "=" * 60)
        log("Conversion completed successfully!")
        log("=" * 60)
        log(f"Saved PyTorch weights to: {output_path}")
    except Exception as exc:
        log(f"\nError during conversion: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
