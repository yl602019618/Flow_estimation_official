"""Analytic truth and solenoidal radial-basis modes."""
from __future__ import annotations

import numpy as np

from .config import SphereConfig, coerce_config


OMEGA = np.asarray([1.0, 2.0, 3.0]) / np.sqrt(14.0)
VORTEX_SCALE = 3.0 * np.sqrt(3.0) / 2.0


def tilted_vortex(points: np.ndarray, config: SphereConfig | str | dict) -> np.ndarray:
    cfg = coerce_config(config)
    x = np.asarray(points, dtype=np.float64)
    q = np.maximum(0.0, 1.0 - np.einsum("ij,ij->i", x, x))
    velocity = VORTEX_SCALE * q[:, None] * np.cross(OMEGA, x)
    return float(cfg.section("physics")["max_velocity_m_s"]) * velocity


def basis_centers(config: SphereConfig | str | dict) -> np.ndarray:
    cfg = coerce_config(config)
    section = cfg.section("basis")
    axis = np.linspace(-float(section["grid_extent"]), float(section["grid_extent"]),
                       int(section["grid_n"]))
    centers = np.asarray([(x, y, z) for x in axis for y in axis for z in axis])
    return centers[np.linalg.norm(centers, axis=1) <= float(section["center_radius"]) + 1e-14]


def raw_solenoidal_modes(points: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    """Evaluate curl(f e_k)=grad(f) cross e_k; shape is (N,3,3*C)."""
    x = np.asarray(points, dtype=np.float64)
    delta = x[:, None, :] - centers[None, :, :]
    r2 = np.einsum("nij,nij->ni", delta, delta)
    q = 1.0 - np.einsum("ni,ni->n", x, x)
    gaussian = np.exp(-r2 / (2.0 * sigma**2))
    grad = gaussian[:, :, None] * (
        -4.0 * q[:, None, None] * x[:, None, :]
        - q[:, None, None]**2 * delta / sigma**2
    )
    modes = np.empty((len(x), 3, 3 * len(centers)), dtype=np.float64)
    axes = np.eye(3)
    for k in range(3):
        modes[:, :, k::3] = np.cross(grad, axes[k]).transpose(0, 2, 1)
    return modes


def raw_mode_divergence(points: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    """Analytic divergence of every curl mode (identically zero)."""
    return np.zeros((len(points), 3 * len(centers)), dtype=np.float64)
