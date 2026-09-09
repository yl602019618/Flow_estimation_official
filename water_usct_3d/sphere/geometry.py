"""Deterministic spherical acquisition and coverage diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .config import SphereConfig, coerce_config


@dataclass(frozen=True)
class SphereAcquisition:
    points: np.ndarray
    pairs: np.ndarray
    directions: np.ndarray
    lengths: np.ndarray

    @property
    def n_transducers(self) -> int:
        return len(self.points)

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)


def fibonacci_sphere(n: int) -> np.ndarray:
    """Return a pole-free, deterministic Fibonacci point set on S^2."""
    i = np.arange(n, dtype=np.float64)
    z = 1.0 - 2.0 * (i + 0.5) / n
    phi = (np.pi * (3.0 - np.sqrt(5.0))) * i
    radius = np.sqrt(1.0 - z * z)
    return np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))


def build_fibonacci_sphere_acquisition(
    config: SphereConfig | str | dict, n_transducers: int | None = None
) -> SphereAcquisition:
    cfg = coerce_config(config)
    n = int(n_transducers or cfg.section("acquisition")["n_transducers"])
    points = fibonacci_sphere(n)
    pairs = np.asarray(list(combinations(range(n), 2)), dtype=np.int32)
    chords = points[pairs[:, 1]] - points[pairs[:, 0]]
    lengths = np.linalg.norm(chords, axis=1)
    return SphereAcquisition(points, pairs, chords / lengths[:, None], lengths)


def coverage_tensor(acquisition: SphereAcquisition, points: np.ndarray) -> np.ndarray:
    """Local direction tensor from every chord crossing a small probe ball."""
    starts = acquisition.points[acquisition.pairs[:, 0]]
    directions = acquisition.directions
    lengths = acquisition.lengths
    result = np.empty((len(points), 3, 3), dtype=np.float64)
    # A radius scaled with array density gives a stable local angular diagnostic.
    probe = 0.28
    for index, point in enumerate(np.asarray(points)):
        along = np.einsum("ij,ij->i", point - starts, directions)
        along_clip = np.clip(along, 0.0, lengths)
        closest = starts + along_clip[:, None] * directions
        selected = np.linalg.norm(closest - point, axis=1) <= probe
        d = directions[selected]
        result[index] = d.T @ d / max(len(d), 1)
    return result


def geometry_metrics(acquisition: SphereAcquisition) -> dict[str, float | int | bool]:
    axis = np.linspace(-0.75, 0.75, 7)
    probes = np.asarray([(x, y, z) for x in axis for y in axis for z in axis
                         if x*x + y*y + z*z <= 0.75**2])
    tensors = coverage_tensor(acquisition, probes)
    eig = np.linalg.eigvalsh(tensors)
    normalized = eig[:, 0] / np.mean(eig, axis=1)
    cond = eig[:, -1] / eig[:, 0]
    return {
        "n_transducers": acquisition.n_transducers,
        "n_pairs": acquisition.n_pairs,
        "max_radius_error": float(np.max(np.abs(np.linalg.norm(acquisition.points, axis=1) - 1))),
        "minimum_point_separation": float(np.min(np.linalg.norm(
            acquisition.points[:, None] - acquisition.points[None, :] +
            np.eye(acquisition.n_transducers)[:, :, None] * 10, axis=2))),
        "minimum_normalized_eigenvalue": float(normalized.min()),
        "maximum_condition_number": float(cond.max()),
        "minimum_rank": int(np.min(np.linalg.matrix_rank(tensors, tol=1e-12))),
    }

