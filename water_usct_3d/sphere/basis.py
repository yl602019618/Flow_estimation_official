"""L2-orthogonalized solenoidal Gaussian curl basis and ray operator."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss

from .config import SphereConfig, coerce_config
from .flow import basis_centers, raw_solenoidal_modes
from .geometry import SphereAcquisition


@dataclass(frozen=True)
class SolenoidalBasis:
    centers: np.ndarray
    sigma: float
    transform: np.ndarray
    gram_eigenvalues: np.ndarray

    @property
    def n_raw(self) -> int:
        return 3 * len(self.centers)

    @property
    def n_modes(self) -> int:
        return self.transform.shape[1]

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        return raw_solenoidal_modes(points, self.centers, self.sigma) @ self.transform


@dataclass(frozen=True)
class RayOperator:
    matrix: np.ndarray
    basis: SolenoidalBasis
    singular_values: np.ndarray
    left_vectors: np.ndarray
    right_vectors_t: np.ndarray


def ball_quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    radial_nodes, radial_weights = leggauss(order)
    mu, mu_weights = leggauss(order)
    n_phi = 2 * order
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    r = 0.5 * (radial_nodes + 1.0)
    rw = 0.5 * radial_weights * r**2
    rr, mm, pp = np.meshgrid(r, mu, phi, indexing="ij")
    sin_theta = np.sqrt(1.0 - mm**2)
    points = np.column_stack(((rr * sin_theta * np.cos(pp)).ravel(),
                              (rr * sin_theta * np.sin(pp)).ravel(),
                              (rr * mm).ravel()))
    weights = (rw[:, None, None] * mu_weights[None, :, None] *
               np.full((1, 1, n_phi), 2.0 * np.pi / n_phi)).ravel()
    return points, weights


def build_solenoidal_basis(config: SphereConfig | str | dict) -> SolenoidalBasis:
    cfg = coerce_config(config)
    section = cfg.section("basis")
    centers = basis_centers(cfg)
    sigma = float(section["sigma"])
    points, weights = ball_quadrature(int(section["volume_quadrature_order"]))
    raw = raw_solenoidal_modes(points, centers, sigma)
    gram = np.einsum("n,nic,nid->cd", weights, raw, raw, optimize=True)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    keep = eigenvalues > float(section["gram_relative_cutoff"]) * eigenvalues[-1]
    transform = eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])[None, :]
    return SolenoidalBasis(centers, sigma, transform, eigenvalues[keep])


def build_solenoidal_ray_matrix(
    acquisition: SphereAcquisition,
    basis: SolenoidalBasis,
    config: SphereConfig | str | dict,
) -> RayOperator:
    cfg = coerce_config(config)
    order = int(cfg.section("forward")["operator_quadrature_order"])
    nodes, weights = leggauss(order)
    matrix_raw = np.empty((acquisition.n_pairs, basis.n_raw), dtype=np.float64)
    batch = 128
    for begin in range(0, acquisition.n_pairs, batch):
        end = min(begin + batch, acquisition.n_pairs)
        starts = acquisition.points[acquisition.pairs[begin:end, 0]]
        directions = acquisition.directions[begin:end]
        lengths = acquisition.lengths[begin:end]
        distance = 0.5 * (nodes[None, :] + 1.0) * lengths[:, None]
        points = starts[:, None, :] + distance[:, :, None] * directions[:, None, :]
        modes = raw_solenoidal_modes(points.reshape(-1, 3), basis.centers,
                                     basis.sigma).reshape(end - begin, order, 3, -1)
        longitudinal = np.einsum("mqic,mi->mqc", modes, directions, optimize=True)
        matrix_raw[begin:end] = np.einsum(
            "q,mq,mqc->mc", weights, 0.5 * lengths[:, None], longitudinal,
            optimize=True)
    matrix = (-2.0 * cfg.radius / cfg.c0**2) * (matrix_raw @ basis.transform)
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    return RayOperator(matrix, basis, singular, u, vt)

