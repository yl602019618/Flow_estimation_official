"""Noise-aware truncated-SVD inversion."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .basis import RayOperator
from .pulse import DelayData3D


@dataclass(frozen=True)
class SphereInversionResult:
    coefficients: np.ndarray
    truncation_rank: int
    residual_rms_s: float
    target_residual_rms_s: float
    predicted_delays_s: np.ndarray

    def velocity(self, points: np.ndarray, operator: RayOperator) -> np.ndarray:
        return operator.basis.evaluate(points) @ self.coefficients


def invert_sphere_tsvd(
    delays: DelayData3D | np.ndarray,
    operator: RayOperator,
    noise_std: float,
    *,
    clean_relative_cutoff: float = 1.0e-8,
) -> SphereInversionResult:
    data = delays.reciprocal_s if isinstance(delays, DelayData3D) else np.asarray(delays)
    u, singular, vt = operator.left_vectors, operator.singular_values, operator.right_vectors_t
    projected = u.T @ data
    if noise_std <= 0.0:
        rank = int(np.count_nonzero(singular / singular[0] >= clean_relative_cutoff))
    else:
        total_energy = float(data @ data)
        captured = np.concatenate(([0.0], np.cumsum(projected**2)))
        residual_norm = np.sqrt(np.maximum(0.0, total_energy - captured))
        target = np.sqrt(len(data)) * noise_std
        rank = int(np.argmin(np.abs(residual_norm - target)))
    coefficients = np.zeros(operator.matrix.shape[1], dtype=np.float64)
    if rank:
        coefficients = vt[:rank].T @ (projected[:rank] / singular[:rank])
    predicted = operator.matrix @ coefficients
    residual_rms = float(np.sqrt(np.mean((data - predicted)**2)))
    return SphereInversionResult(coefficients, rank, residual_rms, float(noise_std), predicted)

