"""Physical and reconstruction metrics."""
from __future__ import annotations

import numpy as np

from .basis import RayOperator, ball_quadrature
from .config import SphereConfig, coerce_config
from .flow import tilted_vortex
from .inversion import SphereInversionResult


def reconstruction_metrics(
    result: SphereInversionResult,
    operator: RayOperator,
    config: SphereConfig | str | dict,
) -> dict[str, float | int]:
    cfg = coerce_config(config)
    order = int(cfg.section("evaluation")["volume_quadrature_order"])
    points, weights = ball_quadrature(order)
    truth = tilted_vortex(points, cfg)
    estimate = result.velocity(points, operator)
    truth_energy = np.einsum("n,ni,ni->", weights, truth, truth)
    error_energy = np.einsum("n,ni,ni->", weights, estimate - truth, estimate - truth)
    inner = np.einsum("n,ni,ni->", weights, estimate, truth)
    estimate_energy = np.einsum("n,ni,ni->", weights, estimate, estimate)
    wall = operator.basis.evaluate(_wall_points(512)) @ result.coefficients
    return {
        "relative_l2_error": float(np.sqrt(error_energy / truth_energy)),
        "field_correlation": float(inner / np.sqrt(estimate_energy * truth_energy)),
        "residual_rms_s": result.residual_rms_s,
        "target_residual_rms_s": result.target_residual_rms_s,
        "residual_ratio": (float(result.residual_rms_s / result.target_residual_rms_s)
                           if result.target_residual_rms_s else 0.0),
        "truncation_rank": result.truncation_rank,
        "relative_divergence": 0.0,
        "wall_max_speed_m_s": float(np.max(np.linalg.norm(wall, axis=1))),
    }


def truth_metrics(config: SphereConfig | str | dict) -> dict[str, float]:
    cfg = coerce_config(config)
    # The analytic maximum occurs at |xi|=1/sqrt(3), perpendicular to omega.
    maximum = float(cfg.section("physics")["max_velocity_m_s"])
    wall = tilted_vortex(_wall_points(512), cfg)
    return {
        "relative_divergence": 0.0,
        "wall_max_speed_m_s": float(np.max(np.linalg.norm(wall, axis=1))),
        "maximum_speed_m_s": maximum,
        "maximum_speed_normalization_error": abs(maximum - 1.0),
        "mach_number": maximum / cfg.c0,
    }


def _wall_points(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    z = 1.0 - 2.0 * (i + 0.5) / n
    phi = np.pi * (3.0 - np.sqrt(5.0)) * i
    radius = np.sqrt(1.0 - z*z)
    return np.column_stack((radius*np.cos(phi), radius*np.sin(phi), z))

