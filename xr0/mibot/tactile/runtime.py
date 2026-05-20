from __future__ import annotations

import logging
import os
import threading
import time
from typing import Dict, Optional

import numpy as np

from .driver import TactileSensorDriver
from .types import SENSOR_NAMES, TactileFrame, TactileSnapshot


class TactileRuntime:
    def __init__(
        self,
        driver: TactileSensorDriver,
        read_mode: str = "auto_push",
        read_timeout: float = 0.1,
        poll_interval: float = 0.1,
        calibration_samples: int = 50,
        calibration_interval: float = 0.05,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if read_mode not in {"auto_push", "distributed_poll"}:
            raise ValueError(f"Unsupported tactile read mode: {read_mode}")

        self.driver = driver
        self.read_mode = read_mode
        self.read_timeout = read_timeout
        self.poll_interval = poll_interval
        self.calibration_samples = calibration_samples
        self.calibration_interval = calibration_interval
        self.logger = logger or logging.getLogger(__name__)

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._latest_frame: Optional[TactileFrame] = None
        self._frame_event = threading.Event()
        self.offsets: Dict[str, np.ndarray] = {
            name: np.zeros(3, dtype=np.float32) for name in SENSOR_NAMES
        }
        self.distributed_offsets: Dict[str, Optional[np.ndarray]] = {
            name: None for name in SENSOR_NAMES
        }

    def start(self) -> None:
        if self._running:
            return

        if self.read_mode == "auto_push":
            self.driver.initialize()
        else:
            self.driver.open()

        self._running = True
        self._thread = threading.Thread(target=self._loop, name="tactile-runtime", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self.driver.close()

    def wait_for_frame(self, timeout: float = 3.0) -> Optional[TactileFrame]:
        if self._latest_frame is not None:
            return self.get_latest_frame(copy_frame=True)
        if not self._frame_event.wait(timeout=timeout):
            return None
        return self.get_latest_frame(copy_frame=True)

    def get_latest_frame(self, copy_frame: bool = True) -> Optional[TactileFrame]:
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy() if copy_frame else self._latest_frame

    def get_snapshot(self, copy_snapshot: bool = True) -> Optional[TactileSnapshot]:
        frame = self.get_latest_frame(copy_frame=True)
        if frame is None:
            return None

        with self._lock:
            offsets = {name: value.copy() for name, value in self.offsets.items()}
            distributed_offsets = {
                name: None if value is None else value.copy()
                for name, value in self.distributed_offsets.items()
            }

        calibrated_force: Dict[str, np.ndarray] = {}
        calibrated_distributed: Dict[str, Optional[np.ndarray]] = {}
        for name in SENSOR_NAMES:
            calibrated_force[name] = frame.get_force(name) - offsets[name]
            sensor = frame.sensors.get(name)
            if sensor is None or sensor.distributed is None:
                calibrated_distributed[name] = None
                continue

            offset = distributed_offsets.get(name)
            if offset is None or offset.shape != sensor.distributed.shape:
                calibrated_distributed[name] = sensor.distributed.copy()
            else:
                calibrated_distributed[name] = sensor.distributed - offset

        snapshot = TactileSnapshot(
            frame=frame,
            calibrated_force=calibrated_force,
            offsets=offsets,
            calibrated_distributed=calibrated_distributed,
            distributed_offsets=distributed_offsets,
        )
        return snapshot if not copy_snapshot else snapshot.copy()

    def calibrate(self, sample_count: Optional[int] = None, sample_interval: Optional[float] = None) -> Dict[str, np.ndarray]:
        if not self._running:
            raise RuntimeError("Tactile runtime must be started before calibration")

        sample_count = sample_count or self.calibration_samples
        sample_interval = sample_interval if sample_interval is not None else self.calibration_interval

        samples = {name: [] for name in SENSOR_NAMES}
        distributed_samples = {name: [] for name in SENSOR_NAMES}
        for _ in range(sample_count):
            frame = self.wait_for_frame(timeout=max(self.read_timeout * 10.0, 1.0))
            if frame is None:
                raise RuntimeError("Timed out while waiting for tactile data during calibration")
            for name in SENSOR_NAMES:
                sensor = frame.sensors.get(name)
                if sensor is None:
                    continue
                samples[name].append(sensor.force.copy())
                if sensor.distributed is not None:
                    distributed_samples[name].append(sensor.distributed.copy())
            time.sleep(sample_interval)

        offsets: Dict[str, np.ndarray] = {}
        distributed_offsets: Dict[str, Optional[np.ndarray]] = {}
        with self._lock:
            for name in SENSOR_NAMES:
                if samples[name]:
                    self.offsets[name] = np.mean(np.stack(samples[name], axis=0), axis=0).astype(np.float32)
                else:
                    self.offsets[name] = np.zeros(3, dtype=np.float32)
                offsets[name] = self.offsets[name].copy()

                if distributed_samples[name]:
                    self.distributed_offsets[name] = np.mean(
                        np.stack(distributed_samples[name], axis=0),
                        axis=0,
                    ).astype(np.float32)
                else:
                    self.distributed_offsets[name] = None
                distributed_offsets[name] = None if self.distributed_offsets[name] is None else self.distributed_offsets[name].copy()

        distributed_summary = {}
        for name, offset in distributed_offsets.items():
            if offset is None:
                distributed_summary[name] = None
            else:
                distributed_summary[name] = {
                    "shape": list(offset.shape),
                    "mean_abs": float(np.mean(np.abs(offset))),
                }

        self.logger.info(
            "Calibrated tactile offsets: force=%s distributed=%s",
            {k: v.tolist() for k, v in offsets.items()},
            distributed_summary,
        )
        return offsets

    def reset_calibration(self) -> None:
        with self._lock:
            for name in SENSOR_NAMES:
                self.offsets[name] = np.zeros(3, dtype=np.float32)
                self.distributed_offsets[name] = None

    def save_snapshot_npz(self, path: str, snapshot: Optional[TactileSnapshot] = None) -> str:
        snapshot = snapshot or self.get_snapshot(copy_snapshot=True)
        if snapshot is None:
            raise RuntimeError("No tactile snapshot available to save")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload: Dict[str, np.ndarray] = {}
        for name in SENSOR_NAMES:
            payload[f"{name}_offset"] = snapshot.offsets[name]
            payload[f"{name}_calibrated_force"] = snapshot.calibrated_force[name]
            distributed_offset = snapshot.distributed_offsets.get(name)
            if distributed_offset is not None:
                payload[f"{name}_distributed_offset"] = distributed_offset

            calibrated_distributed = snapshot.calibrated_distributed.get(name)
            if calibrated_distributed is not None:
                payload[f"{name}_calibrated_distributed"] = calibrated_distributed

            sensor = snapshot.frame.sensors.get(name)
            if sensor is None:
                continue
            payload[f"{name}_force"] = sensor.force
            if sensor.distributed is not None:
                payload[f"{name}_distributed"] = sensor.distributed

        np.savez_compressed(path, timestamp=snapshot.frame.timestamp, **payload)
        return path

    def _loop(self) -> None:
        while self._running:
            try:
                if self.read_mode == "auto_push":
                    frame = self.driver.read_frame(timeout=self.read_timeout)
                else:
                    frame = self.driver.read_distributed_snapshot(timeout=max(self.read_timeout, 1.0))
            except Exception as exc:
                self.logger.warning("Tactile runtime read failed: %s", exc)
                time.sleep(max(self.poll_interval, 0.1))
                continue

            if frame is not None:
                with self._lock:
                    self._latest_frame = frame
                    self._frame_event.set()

            if self.read_mode == "distributed_poll":
                time.sleep(self.poll_interval)
