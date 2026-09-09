"""Array and local-window definitions for scalable long-domain TT inversion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from midplane_xy_tt_study.midplane_tt import sensor_coordinates


@dataclass(frozen=True)
class MultiRingDesign:
    z_min_m: float = 1.05
    z_max_m: float = 7.05
    ring_spacing_m: float = 0.15
    sensors_per_ring: int = 48
    length_xy_m: float = 1.5

    @property
    def ring_z_m(self) -> np.ndarray:
        count = int(round((self.z_max_m - self.z_min_m) / self.ring_spacing_m)) + 1
        values = self.z_min_m + self.ring_spacing_m * np.arange(count)
        if not np.isclose(values[-1], self.z_max_m, atol=1e-12):
            raise ValueError("ring range must be an integer multiple of spacing")
        return values

    @property
    def target_z_m(self) -> np.ndarray:
        # Every other ring is reconstructed; adjacent targets share three of
        # five guard/observation rings. This is a computational choice only.
        return self.ring_z_m[2:-2:2]


def global_acquisition(
    design: MultiRingDesign,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    xy, wall_one = sensor_coordinates(
        design.sensors_per_ring, "uniform", design.length_xy_m,
    )
    coordinates: list[tuple[float, float, float]] = []
    walls: list[str] = []
    rings: list[int] = []
    for ring, z in enumerate(design.ring_z_m):
        for point, wall in zip(xy, wall_one):
            coordinates.append((float(point[0]), float(point[1]), float(z)))
            walls.append(wall)
            rings.append(ring)
    return np.asarray(coordinates), tuple(walls), np.asarray(rings, dtype=np.int64)


def local_ring_z(center_z_m: float, spacing_m: float = 0.15) -> tuple[float, ...]:
    return tuple(float(center_z_m + spacing_m * offset) for offset in (-2, -1, 0, 1, 2))
