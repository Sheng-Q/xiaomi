from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

SENSOR_NAMES = ("index_middle", "middle_middle")


def _zero_force() -> np.ndarray:
    return np.zeros(3, dtype=np.float32)


@dataclass
class SensorSnapshot:
    force: np.ndarray = field(default_factory=_zero_force)
    distributed: Optional[np.ndarray] = None
    point_count: int = 0

    def copy(self) -> "SensorSnapshot":
        return SensorSnapshot(
            force=self.force.copy(),
            distributed=None if self.distributed is None else self.distributed.copy(),
            point_count=self.point_count,
        )


@dataclass
class TactileFrame:
    timestamp: float
    error_code: Optional[int]
    raw_payload_hex: Optional[str]
    sensors: Dict[str, SensorSnapshot]

    def copy(self) -> "TactileFrame":
        return TactileFrame(
            timestamp=self.timestamp,
            error_code=self.error_code,
            raw_payload_hex=self.raw_payload_hex,
            sensors={name: snapshot.copy() for name, snapshot in self.sensors.items()},
        )

    def get_force(self, sensor_name: str) -> np.ndarray:
        snapshot = self.sensors.get(sensor_name)
        if snapshot is None:
            return _zero_force()
        return snapshot.force.copy()


@dataclass
class TactileSnapshot:
    frame: TactileFrame
    calibrated_force: Dict[str, np.ndarray]
    offsets: Dict[str, np.ndarray]

    def copy(self) -> "TactileSnapshot":
        return TactileSnapshot(
            frame=self.frame.copy(),
            calibrated_force={name: value.copy() for name, value in self.calibrated_force.items()},
            offsets={name: value.copy() for name, value in self.offsets.items()},
        )
