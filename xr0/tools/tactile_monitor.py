#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2

from mibot.tactile import TactileRuntime, TactileSensorDriver, TactileVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROS-free tactile monitor for XR0 environments.")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Serial port of the tactile controller.")
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--mode", choices=("auto_push", "distributed_poll"), default="auto_push")
    parser.add_argument("--distributed-scale", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional directory to save PNG/NPZ snapshots.")
    parser.add_argument("--save-every", type=int, default=0, help="Save every N received frames. 0 disables saving.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames. 0 means run forever.")
    parser.add_argument("--calibrate", action="store_true", help="Run zero-point calibration after the first frames arrive.")
    parser.add_argument("--calibration-samples", type=int, default=50)
    parser.add_argument("--calibration-interval", type=float, default=0.05)
    parser.add_argument("--poll-interval", type=float, default=0.1, help="Used only in distributed_poll mode.")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def format_force(values) -> str:
    return f"({values[0]:7.3f}, {values[1]:7.3f}, {values[2]:7.3f})"


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    logger = logging.getLogger("tactile-monitor")

    driver = TactileSensorDriver(
        port=args.port,
        baudrate=args.baudrate,
        timeout=args.timeout,
        enable_distributed=args.mode == "auto_push",
        distributed_scale=args.distributed_scale,
        logger=logger,
    )
    runtime = TactileRuntime(
        driver=driver,
        read_mode=args.mode,
        read_timeout=args.timeout,
        poll_interval=args.poll_interval,
        calibration_samples=args.calibration_samples,
        calibration_interval=args.calibration_interval,
        logger=logger,
    )
    visualizer = TactileVisualizer()

    frames_seen = 0
    last_timestamp = None

    try:
        runtime.start()
        if runtime.wait_for_frame(timeout=3.0) is None:
            raise RuntimeError("Timed out waiting for tactile frames")

        if args.calibrate:
            runtime.calibrate()

        while args.max_frames <= 0 or frames_seen < args.max_frames:
            snapshot = runtime.get_snapshot(copy_snapshot=True)
            if snapshot is None:
                time.sleep(0.02)
                continue

            if snapshot.frame.timestamp == last_timestamp:
                time.sleep(0.02)
                continue

            last_timestamp = snapshot.frame.timestamp
            frames_seen += 1

            index_force = snapshot.calibrated_force["index_middle"]
            middle_force = snapshot.calibrated_force["middle_middle"]
            logger.info(
                "[%05d] index=%s middle=%s raw_payload=%s",
                frames_seen,
                format_force(index_force),
                format_force(middle_force),
                "yes" if snapshot.frame.raw_payload_hex else "no",
            )

            if args.output_dir is not None and args.save_every > 0 and frames_seen % args.save_every == 0:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                image = visualizer.render_snapshot(snapshot, calibrated=True)
                image_path = args.output_dir / f"tactile_{frames_seen:05d}.png"
                npz_path = args.output_dir / f"tactile_{frames_seen:05d}.npz"
                cv2.imwrite(str(image_path), image)
                runtime.save_snapshot_npz(str(npz_path), snapshot=snapshot)
                logger.info("Saved tactile artifacts to %s and %s", image_path, npz_path)
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
