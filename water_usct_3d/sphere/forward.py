"""Prescribed-chord travel-time forward model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.polynomial.legendre import leggauss

from .config import SphereConfig, coerce_config
from .geometry import SphereAcquisition


@dataclass(frozen=True)
class TravelTimeDataset:
    forward_s: np.ndarray
    reverse_s: np.ndarray
    reciprocal_s: np.ndarray
    linearized_s: np.ndarray
    static_s: np.ndarray
    noise_std_s: float = 0.0


def _flow_values(flow: Callable[[np.ndarray], np.ndarray] | np.ndarray, points: np.ndarray) -> np.ndarray:
    values = flow(points) if callable(flow) else flow
    values = np.asarray(values, dtype=np.float64)
    if values.shape != points.shape:
        raise ValueError("flow must return one 3-vector per quadrature point")
    return values


def simulate_chord_travel_times(
    flow: Callable[[np.ndarray], np.ndarray] | np.ndarray,
    acquisition: SphereAcquisition,
    config: SphereConfig | str | dict,
    *,
    quadrature_order: int | None = None,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> TravelTimeDataset:
    cfg = coerce_config(config)
    order = int(quadrature_order or cfg.section("forward")["truth_quadrature_order"])
    nodes, weights = leggauss(order)
    starts = acquisition.points[acquisition.pairs[:, 0]]
    length = acquisition.lengths
    direction = acquisition.directions
    points = starts[:, None, :] + (0.5 * (nodes + 1.0) * length[:, None])[:, :, None] * direction[:, None, :]
    flat_points = points.reshape(-1, 3)
    velocity = _flow_values(flow, flat_points).reshape(len(length), order, 3)
    parallel = np.einsum("mqi,mi->mq", velocity, direction)
    speed2 = np.einsum("mqi,mqi->mq", velocity, velocity)
    perpendicular2 = np.maximum(0.0, speed2 - parallel**2)
    transverse_speed = np.sqrt(cfg.c0**2 - perpendicular2)
    jacobian = 0.5 * length[:, None]
    forward = cfg.radius * np.sum(weights[None, :] * jacobian /
                                  (parallel + transverse_speed), axis=1)
    reverse = cfg.radius * np.sum(weights[None, :] * jacobian /
                                  (-parallel + transverse_speed), axis=1)
    integral_parallel = np.sum(weights[None, :] * jacobian * parallel, axis=1)
    linearized = -2.0 * cfg.radius * integral_parallel / cfg.c0**2
    static = cfg.radius * length / cfg.c0
    if noise_std:
        generator = rng or np.random.default_rng(cfg.seed)
        error_forward = generator.normal(size=len(length))
        error_reverse = generator.normal(size=len(length))
        difference = error_forward - error_reverse
        difference -= difference.mean()
        scale = noise_std / np.sqrt(np.mean(difference**2))
        error_forward *= scale
        error_reverse *= scale
        forward = forward + error_forward
        reverse = reverse + error_reverse
    return TravelTimeDataset(forward, reverse, forward - reverse, linearized, static,
                             float(noise_std))

