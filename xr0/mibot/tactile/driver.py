from __future__ import annotations

import logging
import struct
import time
from typing import Dict, List, Optional

import numpy as np

from .protocol import TactileSerialProtocol
from .types import SENSOR_NAMES, SensorSnapshot, TactileFrame


class TactileSensorDriver(TactileSerialProtocol):
    SENSOR_LAYOUT = {
        "index_middle": {
            "display_name": "index_middle",
            "config_byte_index": 0,
            "config_mask": 0x20,
            "point_count_offset": 0x0A,
            "poll_address": 0x1A00,
            "fallback_point_count": 68,
        },
        "middle_middle": {
            "display_name": "middle_middle",
            "config_byte_index": 1,
            "config_mask": 0x02,
            "point_count_offset": 0x12,
            "poll_address": 0x2000,
            "fallback_point_count": 68,
        },
    }

    def __init__(
        self,
        port: str,
        baudrate: int = 921600,
        timeout: float = 1.0,
        enable_distributed: bool = True,
        distributed_scale: float = 0.1,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(port=port, baudrate=baudrate, timeout=timeout, logger=logger)
        self.enable_distributed = enable_distributed
        self.distributed_scale = distributed_scale
        self.connected_sensors: List[Dict[str, int | str]] = []
        self.auto_push_mode = False

    def initialize(self) -> None:
        self.open()
        if not self.read_sensor_configuration():
            raise RuntimeError("No supported tactile sensors detected")
        if self.enable_distributed:
            self.read_point_count_table()
        if not self.enable_auto_push_mode():
            raise RuntimeError("Failed to enable tactile auto-push mode")

    def close(self) -> None:
        if self.auto_push_mode:
            self.disable_auto_push_mode()
        super().close()

    def read_sensor_configuration(self) -> bool:
        frame = self.build_request_frame(self.FUNC_READ, 0x0010, 4)
        if not self.send_command(frame):
            return False
        time.sleep(0.05)
        response = self.read_response(0.5, self.RESP_HEAD_GENERAL)
        if response is None or len(response) < 13:
            self.logger.warning("Failed to read tactile sensor configuration")
            return False

        payload = response[8:12]
        self.connected_sensors = []
        for name, layout in self.SENSOR_LAYOUT.items():
            byte_value = payload[layout["config_byte_index"]]
            if byte_value & layout["config_mask"]:
                self.connected_sensors.append(
                    {
                        "name": name,
                        "display_name": layout["display_name"],
                        "point_count": int(layout["fallback_point_count"]),
                    }
                )

        if not self.connected_sensors:
            return False

        self.logger.info(
            "Detected tactile sensors: %s",
            ", ".join(sensor["display_name"] for sensor in self.connected_sensors),
        )
        return True

    def read_point_count_table(self) -> None:
        frame = self.build_request_frame(self.FUNC_READ, 0x0030, 0x38)
        if not self.send_command(frame):
            return
        time.sleep(0.05)
        response = self.read_response(0.8, self.RESP_HEAD_GENERAL)
        if response is None or len(response) < 8 + 0x38 + 1:
            self.logger.warning("Failed to read tactile point-count table")
            return

        table = response[8 : 8 + 0x38]
        for sensor in self.connected_sensors:
            layout = self.SENSOR_LAYOUT[sensor["name"]]
            offset = int(layout["point_count_offset"])
            if offset + 2 > len(table):
                continue
            point_count = int.from_bytes(table[offset : offset + 2], "little")
            if point_count > 0:
                sensor["point_count"] = point_count

        point_counts = {sensor["name"]: sensor["point_count"] for sensor in self.connected_sensors}
        self.logger.info("Tactile point counts: %s", point_counts)

    def enable_auto_push_mode(self) -> bool:
        try:
            data_type = 0x03 if self.enable_distributed else 0x01
            frame = self.build_request_frame(self.FUNC_WRITE, self.DATA_TYPE_REG, 1, bytes([data_type]))
            if not self.send_command(frame):
                return False
            time.sleep(0.05)

            frame = self.build_request_frame(self.FUNC_WRITE, self.AUTO_PUSH_REG, 1, b"\x01")
            if not self.send_command(frame):
                return False
            time.sleep(0.05)
            self.auto_push_mode = True
            self.logger.info("Enabled tactile auto-push mode")
            return True
        except Exception as exc:
            self.logger.error("Failed to enable tactile auto-push mode: %s", exc)
            return False

    def disable_auto_push_mode(self) -> None:
        try:
            frame = self.build_request_frame(self.FUNC_WRITE, self.AUTO_PUSH_REG, 1, b"\x00")
            self.send_command(frame)
        finally:
            self.auto_push_mode = False

    def parse_auto_push_frame(self, data: bytes) -> Optional[Dict[str, object]]:
        if not data or len(data) < 7:
            return None
        if data[:2] != self.RESP_HEAD_AUTO_PUSH:
            return None

        valid_frame_len = int.from_bytes(data[3:5], "little")
        expected_len = 6 + valid_frame_len
        if len(data) < expected_len:
            return None

        data = data[:expected_len]
        error_code = data[5]
        payload_len = max(valid_frame_len - 1, 0)
        payload = data[6 : 6 + payload_len]
        return {"error_code": error_code, "payload": payload}

    def read_frame(self, timeout: float = 0.1) -> Optional[TactileFrame]:
        response = self.read_response(timeout, self.RESP_HEAD_AUTO_PUSH)
        if response is None:
            return None

        parsed = self.parse_auto_push_frame(response)
        if parsed is None:
            return None

        payload = parsed["payload"]
        error_code = int(parsed["error_code"])

        frame = self._try_parse_layout_a(payload, error_code)
        if frame is None:
            frame = self._try_parse_layout_b(payload, error_code)
        return frame

    def read_distributed_snapshot(self, timeout: float = 1.0) -> Optional[TactileFrame]:
        self.open()
        sensors: Dict[str, SensorSnapshot] = {}

        point_lookup = {
            sensor["name"]: int(sensor["point_count"])
            for sensor in self.connected_sensors
            if sensor["name"] in self.SENSOR_LAYOUT
        }

        for name, layout in self.SENSOR_LAYOUT.items():
            point_count = point_lookup.get(name, int(layout["fallback_point_count"]))
            byte_count = point_count * 3
            frame = self.build_request_frame(self.FUNC_READ, int(layout["poll_address"]), byte_count)
            if not self.send_command(frame):
                continue
            time.sleep(0.1)
            response = self.read_response(timeout, self.RESP_HEAD_GENERAL)
            if response is None or len(response) < 8 + byte_count:
                self.logger.warning("Failed to poll distributed tactile data for %s", name)
                continue

            distributed = self._decode_distributed(response[8 : 8 + byte_count], point_count)
            sensors[name] = SensorSnapshot(
                force=np.zeros(3, dtype=np.float32),
                distributed=distributed,
                point_count=point_count,
            )

        if not sensors:
            return None

        return TactileFrame(
            timestamp=time.time(),
            error_code=None,
            raw_payload_hex=None,
            sensors=sensors,
        )

    def _decode_force(self, raw_bytes: bytes) -> np.ndarray:
        fx = struct.unpack("<h", raw_bytes[0:2])[0] * 0.1
        fy = struct.unpack("<h", raw_bytes[2:4])[0] * 0.1
        fz = struct.unpack("<h", raw_bytes[4:6])[0] * 0.1
        return np.asarray([fx, fy, fz], dtype=np.float32)

    def _decode_distributed(self, raw_bytes: bytes, point_count: int) -> np.ndarray:
        distributed = np.zeros((point_count, 3), dtype=np.float32)
        for index in range(point_count):
            offset = index * 3
            fx = struct.unpack("b", raw_bytes[offset : offset + 1])[0] * self.distributed_scale
            fy = struct.unpack("b", raw_bytes[offset + 1 : offset + 2])[0] * self.distributed_scale
            fz = struct.unpack("B", raw_bytes[offset + 2 : offset + 3])[0] * self.distributed_scale
            distributed[index] = (fx, fy, fz)
        return distributed

    def _try_parse_layout_a(self, payload: bytes, error_code: int) -> Optional[TactileFrame]:
        cursor = 0
        sensors: Dict[str, SensorSnapshot] = {}
        for sensor in self.connected_sensors:
            if cursor + 6 > len(payload):
                return None

            point_count = int(sensor["point_count"])
            force = self._decode_force(payload[cursor : cursor + 6])
            cursor += 6

            distributed = None
            if self.enable_distributed and point_count > 0:
                byte_count = point_count * 3
                if cursor + byte_count > len(payload):
                    return None
                distributed = self._decode_distributed(payload[cursor : cursor + byte_count], point_count)
                cursor += byte_count

            sensors[str(sensor["name"])] = SensorSnapshot(force=force, distributed=distributed, point_count=point_count)

        return TactileFrame(
            timestamp=time.time(),
            error_code=error_code,
            raw_payload_hex=payload.hex(),
            sensors=sensors,
        )

    def _try_parse_layout_b(self, payload: bytes, error_code: int) -> Optional[TactileFrame]:
        cursor = 0
        sensor_count = len(self.connected_sensors)
        if cursor + sensor_count * 6 > len(payload):
            return None

        sensors: Dict[str, SensorSnapshot] = {}
        for sensor in self.connected_sensors:
            point_count = int(sensor["point_count"])
            force = self._decode_force(payload[cursor : cursor + 6])
            cursor += 6
            sensors[str(sensor["name"])] = SensorSnapshot(force=force, distributed=None, point_count=point_count)

        if self.enable_distributed:
            for sensor in self.connected_sensors:
                point_count = int(sensor["point_count"])
                if point_count <= 0:
                    continue
                byte_count = point_count * 3
                if cursor + byte_count > len(payload):
                    return None
                sensors[str(sensor["name"])].distributed = self._decode_distributed(
                    payload[cursor : cursor + byte_count],
                    point_count,
                )
                cursor += byte_count

        return TactileFrame(
            timestamp=time.time(),
            error_code=error_code,
            raw_payload_hex=payload.hex(),
            sensors=sensors,
        )
