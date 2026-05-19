from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np
from scipy.interpolate import griddata

from .types import TactileFrame, TactileSnapshot


class TactileVisualizer:
    def __init__(self, output_size: int = 256, vmax_fz: float = 25.5, vmax_shear: float = 12.8) -> None:
        self.output_size = output_size
        self.vmax_fz = vmax_fz
        self.vmax_shear = vmax_shear

    def gen_coords(self, point_count: int) -> np.ndarray:
        if point_count == 68:
            coords = []
            rows, cols = 9, 8
            for row in range(rows):
                for col in range(cols):
                    if (row == 0 and col == 0) or (row == 0 and col == cols - 1):
                        continue
                    if (row == rows - 1 and col == 0) or (row == rows - 1 and col == cols - 1):
                        continue
                    x = col / (cols - 1)
                    y = 1.0 - (row / (rows - 1))
                    coords.append([x, y])
            return np.asarray(coords, dtype=np.float32)

        if point_count == 25:
            coords = []
            rows = cols = 5
            for row in range(rows):
                for col in range(cols):
                    coords.append([col / (cols - 1), row / (rows - 1)])
            return np.asarray(coords, dtype=np.float32)

        cols = int(np.ceil(np.sqrt(point_count)))
        rows = int(np.ceil(point_count / max(cols, 1)))
        coords = []
        index = 0
        for row in range(rows):
            for col in range(cols):
                if index >= point_count:
                    break
                x = col / max(cols - 1, 1)
                y = row / max(rows - 1, 1)
                coords.append([x, y])
                index += 1
        return np.asarray(coords, dtype=np.float32)

    def interpolate_channel(self, coords: Optional[np.ndarray], data: Optional[np.ndarray], channel: int) -> np.ndarray:
        if data is None or coords is None:
            return np.zeros((self.output_size, self.output_size), dtype=np.float32)

        values = data[:, channel].astype(np.float32)
        grid_x, grid_y = np.mgrid[0:1:complex(self.output_size), 0:1:complex(self.output_size)]
        try:
            grid = griddata(coords, values, (grid_x, grid_y), method="cubic", fill_value=0)
        except Exception:
            grid = griddata(coords, values, (grid_x, grid_y), method="linear", fill_value=0)
        return np.nan_to_num(grid, nan=0.0).astype(np.float32)

    def make_fz_heatmap(self, coords: Optional[np.ndarray], data: Optional[np.ndarray]) -> np.ndarray:
        if data is None:
            image = np.zeros((self.output_size, self.output_size), dtype=np.uint8)
            return cv2.applyColorMap(image, cv2.COLORMAP_JET)

        fz = self.interpolate_channel(coords, data, 2)
        fz_norm = np.clip(fz / max(self.vmax_fz, 1e-6) * 255.0, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(fz_norm, cv2.COLORMAP_JET)

    def make_rgb_map(self, coords: Optional[np.ndarray], data: Optional[np.ndarray]) -> np.ndarray:
        if data is None:
            return np.zeros((self.output_size, self.output_size, 3), dtype=np.uint8)

        fx = self.interpolate_channel(coords, data, 0)
        fy = self.interpolate_channel(coords, data, 1)
        fz = self.interpolate_channel(coords, data, 2)

        red = np.clip((fx / max(self.vmax_shear, 1e-6) * 127.0) + 128.0, 0, 255).astype(np.uint8)
        green = np.clip((fy / max(self.vmax_shear, 1e-6) * 127.0) + 128.0, 0, 255).astype(np.uint8)
        blue = np.clip((fz / max(self.vmax_fz, 1e-6) * 255.0), 0, 255).astype(np.uint8)
        return np.stack([blue, green, red], axis=-1)

    def render_snapshot(self, snapshot: TactileSnapshot, calibrated: bool = True) -> np.ndarray:
        force_override = snapshot.calibrated_force if calibrated else None
        return self.render_frame(snapshot.frame, force_override=force_override)

    def render_frame(
        self,
        frame: TactileFrame,
        force_override: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        index_snapshot = frame.sensors.get("index_middle")
        middle_snapshot = frame.sensors.get("middle_middle")

        index_dist = None if index_snapshot is None else index_snapshot.distributed
        middle_dist = None if middle_snapshot is None else middle_snapshot.distributed

        index_coords = self.gen_coords(index_dist.shape[0]) if index_dist is not None else None
        middle_coords = self.gen_coords(middle_dist.shape[0]) if middle_dist is not None else None

        index_hm = self.make_fz_heatmap(index_coords, index_dist)
        middle_hm = self.make_fz_heatmap(middle_coords, middle_dist)
        index_rgb = self.make_rgb_map(index_coords, index_dist)
        middle_rgb = self.make_rgb_map(middle_coords, middle_dist)

        row1 = np.hstack([index_hm, middle_hm])
        row2 = np.hstack([index_rgb, middle_rgb])
        canvas = np.vstack([row1, row2])

        panel_size = self.output_size
        index_force = self._resolve_force("index_middle", frame, force_override)
        middle_force = self._resolve_force("middle_middle", frame, force_override)

        self._overlay_text(canvas, "Index - Fz Heatmap", index_force, 10, 25)
        self._overlay_text(canvas, "Middle - Fz Heatmap", middle_force, panel_size + 10, 25)
        cv2.putText(
            canvas,
            "Index - RGB (Fx,Fy,Fz)",
            (10, panel_size + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            canvas,
            "Middle - RGB (Fx,Fy,Fz)",
            (panel_size + 10, panel_size + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        if frame.error_code is not None:
            cv2.putText(
                canvas,
                f"Err:0x{frame.error_code:02X}",
                (panel_size * 2 - 140, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )
        return canvas

    def _resolve_force(
        self,
        sensor_name: str,
        frame: TactileFrame,
        force_override: Optional[Dict[str, np.ndarray]],
    ) -> Optional[np.ndarray]:
        if force_override is not None and sensor_name in force_override:
            return force_override[sensor_name]
        snapshot = frame.sensors.get(sensor_name)
        return None if snapshot is None else snapshot.force

    def _overlay_text(self, image: np.ndarray, title: str, force: Optional[np.ndarray], x: int, y: int) -> None:
        cv2.putText(image, title, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if force is None:
            return
        fx, fy, fz = force.tolist()
        text = f"F=({fx:.2f},{fy:.2f},{fz:.2f})"
        cv2.putText(image, text, (x, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
