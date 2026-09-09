from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import RegularGridInterpolator

from midplane_xy_tt_study.midplane_tt import sensor_coordinates


@dataclass(frozen=True)
class MultiSliceConfig:
    snapshot: str = (
        "examples/06_frequency_ablation/data/frozen_production_snapshot_104.h5"
    )
    baseline: str = (
        "examples/06_frequency_ablation/data/flow_pilot_baseline_re500_10s_clean.h5"
    )
    output: str = "multislice_xyz_tt_study/outputs/main"
    length_xy_m: float = 1.5
    flow_length_z_m: float = 4.0
    slab_z_min_m: float = 2.10
    slab_z_max_m: float = 2.90
    sensor_z_m: tuple[float, ...] = (2.20, 2.35, 2.50, 2.65, 2.80)
    primary_z_min_m: float | None = None
    primary_z_max_m: float | None = None
    sensors_per_layer: int = 32
    frequency_hz: float = 150_000.0
    sound_speed_m_s: float = 1480.0
    seed: int = 20260807
    center_shape: tuple[int, int, int] = (9, 9, 7)
    sigma_xy_m: float = 0.16
    sigma_z_m: float = 0.13
    evaluation_shape: tuple[int, int, int] = (41, 41, 9)
    norm_shape: tuple[int, int, int] = (21, 21, 11)
    quadrature_order: int = 20
    noise_repeats: int = 8
    reference_noise_frequency_hz: float = 20_000.0
    reference_delay_sigma_s: float = 2.23713169990703e-9
    run_pair_ablations: bool = True
    axial_fisher_weight_factor: float = 1.0
    axial_sine_order: int = 0
    operator_dtype: str = "float64"
    operator_batch_size: int = 16


@dataclass(frozen=True)
class CurlBasis3D:
    centers: np.ndarray
    mode_norms: np.ndarray
    sigma_xy_m: float
    sigma_z_m: float
    axial_sine_order: int = 0

    @property
    def curl_mode_count(self) -> int:
        return int(3 * len(self.centers))

    @property
    def axial_mode_count(self) -> int:
        return int(self.axial_sine_order**2)

    @property
    def mode_count(self) -> int:
        return self.curl_mode_count + self.axial_mode_count


def _axes_for_cell_data(
    shape: tuple[int, int, int], lengths: tuple[float, float, float],
) -> tuple[np.ndarray, ...]:
    return tuple(
        (np.arange(count) + 0.5) * length / count
        for count, length in zip(shape, lengths)
    )


def _load_interpolators(config: MultiSliceConfig) -> tuple[RegularGridInterpolator, RegularGridInterpolator, dict[str, object]]:
    with h5py.File(config.snapshot, "r") as handle:
        snapshot = np.asarray(handle["velocity"], dtype=np.float64)
        metadata = {}
        for key, value in handle.attrs.items():
            array = np.asarray(value)
            metadata[key] = array.item() if array.ndim == 0 else array.tolist()
    with h5py.File(config.baseline, "r") as handle:
        # Mature no-sphere mean is independent of the selected wake snapshot.
        baseline_values = np.asarray(handle["velocity"], dtype=np.float64)
        if baseline_values.ndim == 4:
            baseline = baseline_values
        elif baseline_values.ndim == 5:
            baseline = baseline_values[-5:].mean(axis=0)
        else:
            raise ValueError("baseline velocity must have shape (3,nx,ny,nz) or (t,3,nx,ny,nz)")
        baseline_times = (
            np.asarray(handle["times"][-5:], dtype=np.float64)
            if "times" in handle else np.asarray([], dtype=np.float64)
        )
    domain_lengths = (config.length_xy_m, config.length_xy_m, config.flow_length_z_m)
    snapshot_interpolator = RegularGridInterpolator(
        _axes_for_cell_data(snapshot.shape[1:], domain_lengths), np.moveaxis(snapshot, 0, -1),
        bounds_error=False, fill_value=0.0,
    )
    baseline_interpolator = RegularGridInterpolator(
        _axes_for_cell_data(baseline.shape[1:], domain_lengths), np.moveaxis(baseline, 0, -1),
        bounds_error=False, fill_value=None,
    )
    return snapshot_interpolator, baseline_interpolator, {
        "snapshot": str(Path(config.snapshot).resolve()),
        "baseline": str(Path(config.baseline).resolve()),
        "snapshot_attributes": metadata,
        "baseline_average_times_s": baseline_times.tolist(),
        "truth_definition": "production sphere wake minus independent mature no-sphere flow",
    }


def truth_on_grid(
    snapshot: RegularGridInterpolator,
    baseline: RegularGridInterpolator,
    config: MultiSliceConfig,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    axes = (
        np.linspace(0.0, config.length_xy_m, config.evaluation_shape[0]),
        np.linspace(0.0, config.length_xy_m, config.evaluation_shape[1]),
        np.linspace(config.slab_z_min_m, config.slab_z_max_m, config.evaluation_shape[2]),
    )
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    disturbance = snapshot(points) - baseline(points)
    return np.moveaxis(disturbance.reshape(*config.evaluation_shape, 3), -1, 0), axes


def acquisition(config: MultiSliceConfig) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    xy, walls_one = sensor_coordinates(
        config.sensors_per_layer, "uniform", config.length_xy_m,
    )
    coordinates, walls, layers = [], [], []
    for layer, z in enumerate(config.sensor_z_m):
        for point, wall in zip(xy, walls_one):
            coordinates.append((float(point[0]), float(point[1]), float(z)))
            walls.append(wall)
            layers.append(layer)
    return np.asarray(coordinates), tuple(walls), np.asarray(layers, dtype=np.int64)


def select_pairs(walls: tuple[str, ...], layers: np.ndarray, policy: str) -> np.ndarray:
    pairs = []
    for first in range(len(walls)):
        for second in range(first + 1, len(walls)):
            if walls[first] == walls[second]:
                continue
            separation = abs(int(layers[first]) - int(layers[second]))
            if policy == "full" or (policy == "same_layer" and separation == 0):
                pairs.append((first, second))
            elif policy == "adjacent" and separation <= 1:
                pairs.append((first, second))
    if policy not in {"full", "same_layer", "adjacent"}:
        raise ValueError(f"unknown pair policy {policy!r}")
    return np.asarray(pairs, dtype=np.int64)


def _envelope_gradient(
    points: np.ndarray, basis: CurlBasis3D, length_xy_m: float,
) -> np.ndarray:
    """Gradient of bx(x) by(y) anisotropic Gaussian, shape (point,center,3)."""
    points = np.asarray(points, dtype=np.float64)
    delta = points[:, None, :] - basis.centers[None, :, :]
    exponent = (
        (delta[:, :, 0] ** 2 + delta[:, :, 1] ** 2) / basis.sigma_xy_m**2
        + delta[:, :, 2] ** 2 / basis.sigma_z_m**2
    )
    gaussian = np.exp(-0.5 * exponent)
    xn = points[:, 0] / length_xy_m
    yn = points[:, 1] / length_xy_m
    bx = xn**2 * (1.0 - xn)**2
    by = yn**2 * (1.0 - yn)**2
    dbx = 2.0 * xn * (1.0 - xn) * (1.0 - 2.0 * xn) / length_xy_m
    dby = 2.0 * yn * (1.0 - yn) * (1.0 - 2.0 * yn) / length_xy_m
    phi = bx[:, None] * by[:, None] * gaussian
    return np.stack((
        by[:, None] * gaussian * dbx[:, None]
        - phi * delta[:, :, 0] / basis.sigma_xy_m**2,
        bx[:, None] * gaussian * dby[:, None]
        - phi * delta[:, :, 1] / basis.sigma_xy_m**2,
        -phi * delta[:, :, 2] / basis.sigma_z_m**2,
    ), axis=-1)


def build_basis(config: MultiSliceConfig) -> CurlBasis3D:
    cx = np.linspace(0.10, config.length_xy_m - 0.10, config.center_shape[0])
    cy = np.linspace(0.10, config.length_xy_m - 0.10, config.center_shape[1])
    cz = np.linspace(config.slab_z_min_m + 0.05, config.slab_z_max_m - 0.05,
                     config.center_shape[2])
    centers = np.asarray([(x, y, z) for x in cx for y in cy for z in cz])
    placeholder = CurlBasis3D(
        centers, np.ones(3 * len(centers)), config.sigma_xy_m, config.sigma_z_m,
        config.axial_sine_order,
    )
    axes = (
        np.linspace(0.0, config.length_xy_m, config.norm_shape[0]),
        np.linspace(0.0, config.length_xy_m, config.norm_shape[1]),
        np.linspace(config.slab_z_min_m, config.slab_z_max_m, config.norm_shape[2]),
    )
    spacing = tuple(axis[1] - axis[0] for axis in axes)
    volume = float(np.prod(spacing))
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    squared = np.zeros((len(centers), 3), dtype=np.float64)
    for begin in range(0, len(points), 256):
        gradient = _envelope_gradient(points[begin:begin + 256], placeholder,
                                      config.length_xy_m)
        total = np.sum(gradient**2, axis=-1)
        squared += np.sum(total[:, :, None] - gradient**2, axis=0) * volume
    norms = np.sqrt(np.maximum(squared.reshape(-1), 1e-30))
    return CurlBasis3D(
        centers, norms, config.sigma_xy_m, config.sigma_z_m,
        config.axial_sine_order,
    )


def _axial_sine_values(
    points: np.ndarray, order: int, length_xy_m: float, slab_length_m: float,
) -> np.ndarray:
    if order <= 0:
        return np.empty((len(points), 0), dtype=np.float64)
    indices = np.asarray([(m, n) for m in range(1, order + 1)
                          for n in range(1, order + 1)])
    normalization = length_xy_m / 2.0 * np.sqrt(slab_length_m)
    return (
        np.sin(points[:, 0, None] * np.pi * indices[:, 0] / length_xy_m)
        * np.sin(points[:, 1, None] * np.pi * indices[:, 1] / length_xy_m)
        / normalization
    )


def _ray_geometry(coordinates: np.ndarray, pairs: np.ndarray) -> tuple[np.ndarray, ...]:
    starts = coordinates[pairs[:, 0]]
    vectors = coordinates[pairs[:, 1]] - starts
    lengths = np.linalg.norm(vectors, axis=1)
    tangents = vectors / lengths[:, None]
    reference = np.repeat(np.asarray([[0.0, 0.0, 1.0]]), len(pairs), axis=0)
    nearly_axial = np.abs(tangents[:, 2]) > 0.90
    reference[nearly_axial] = (1.0, 0.0, 0.0)
    normal_one = np.cross(tangents, reference)
    normal_one /= np.linalg.norm(normal_one, axis=1, keepdims=True)
    normal_two = np.cross(tangents, normal_one)
    return starts, tangents, lengths, normal_one, normal_two


def build_operator_and_data(
    snapshot: RegularGridInterpolator,
    baseline: RegularGridInterpolator,
    coordinates: np.ndarray,
    pairs: np.ndarray,
    basis: CurlBasis3D,
    config: MultiSliceConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    nodes, axial_weights = leggauss(config.quadrature_order)
    starts, tangents, lengths, normal_one, normal_two = _ray_geometry(coordinates, pairs)
    wavelength = config.sound_speed_m_s / config.frequency_hz
    if config.operator_dtype not in {"float32", "float64"}:
        raise ValueError("operator_dtype must be 'float32' or 'float64'")
    if config.operator_batch_size <= 0:
        raise ValueError("operator_batch_size must be positive")
    matrix_dtype = np.dtype(config.operator_dtype)
    matrix = np.empty((len(pairs), basis.mode_count), dtype=matrix_dtype)
    target = np.empty(len(pairs), dtype=np.float64)
    # Five-point unscented rule has exact transverse Gaussian covariance.
    transverse_weights_base = np.asarray((1/3, 1/6, 1/6, 1/6, 1/6), dtype=np.float64)
    batch_size = int(config.operator_batch_size)
    for begin in range(0, len(pairs), batch_size):
        end = min(begin + batch_size, len(pairs))
        batch_lengths = lengths[begin:end]
        distance = 0.5 * (nodes[None] + 1.0) * batch_lengths[:, None]
        center = starts[begin:end, None] + distance[:, :, None] * tangents[begin:end, None]
        fresnel = np.sqrt(np.maximum(
            wavelength * distance * (batch_lengths[:, None] - distance)
            / batch_lengths[:, None], 1e-18,
        ))
        sigma = np.maximum(fresnel / 2.355, wavelength / 8.0)
        radius = np.sqrt(3.0) * sigma
        offsets = np.stack((
            np.zeros_like(center),
            radius[:, :, None] * normal_one[begin:end, None],
            -radius[:, :, None] * normal_one[begin:end, None],
            radius[:, :, None] * normal_two[begin:end, None],
            -radius[:, :, None] * normal_two[begin:end, None],
        ), axis=2)
        points = center[:, :, None] + offsets
        inside = (
            (points[..., 0] >= 0.0) & (points[..., 0] <= config.length_xy_m)
            & (points[..., 1] >= 0.0) & (points[..., 1] <= config.length_xy_m)
            & (points[..., 2] >= config.slab_z_min_m)
            & (points[..., 2] <= config.slab_z_max_m)
        )
        transverse_weights = transverse_weights_base[None, None, :] * inside
        transverse_weights /= np.maximum(
            transverse_weights.sum(axis=2, keepdims=True), 1e-30,
        )
        flat = points.reshape(-1, 3)
        truth_values = (snapshot(flat) - baseline(flat)).reshape(
            end - begin, config.quadrature_order, 5, 3,
        )
        longitudinal_truth = np.einsum(
            "bqoj,bj->bqo", truth_values, tangents[begin:end], optimize=True,
        )
        axial_truth = np.einsum(
            "bqo,bqo->bq", longitudinal_truth, transverse_weights, optimize=True,
        )
        target[begin:end] = np.einsum(
            "q,bq,b->b", axial_weights, axial_truth,
            0.5 * batch_lengths, optimize=True,
        )
        gradient = _envelope_gradient(flat, basis, config.length_xy_m).reshape(
            end - begin, config.quadrature_order, 5, len(basis.centers), 3,
        )
        longitudinal_modes = np.cross(
            tangents[begin:end, None, None, None], gradient,
        )
        transverse_modes = np.einsum(
            "bqock,bqo->bqck", longitudinal_modes, transverse_weights, optimize=True,
        )
        integrated = np.einsum(
            "q,bqck,b->bck", axial_weights, transverse_modes,
            0.5 * batch_lengths, optimize=True,
        ).reshape(end - begin, -1)
        normalized_curl = integrated / basis.mode_norms[None]
        if basis.axial_mode_count:
            axial_values = _axial_sine_values(
                flat, basis.axial_sine_order, config.length_xy_m,
                config.slab_z_max_m - config.slab_z_min_m,
            ).reshape(
                end - begin, config.quadrature_order, 5, basis.axial_mode_count,
            )
            axial_longitudinal = axial_values * tangents[begin:end, None, None, 2, None]
            axial_transverse = np.einsum(
                "bqok,bqo->bqk", axial_longitudinal, transverse_weights,
                optimize=True,
            )
            axial_integrated = np.einsum(
                "q,bqk,b->bk", axial_weights, axial_transverse,
                0.5 * batch_lengths, optimize=True,
            )
            matrix[begin:end] = np.concatenate((normalized_curl, axial_integrated), axis=1)
        else:
            matrix[begin:end] = normalized_curl
    layer_delta = np.abs(
        coordinates[pairs[:, 0], 2] - coordinates[pairs[:, 1], 2]
    )
    return matrix, target, {
        "wavelength_m": float(wavelength),
        "pair_count": int(len(pairs)),
        "same_layer_pair_fraction": float(np.mean(layer_delta < 1e-12)),
        "maximum_layer_separation_m": float(np.max(layer_delta, initial=0.0)),
        "median_abs_tangent_z": float(np.median(np.abs(tangents[:, 2]))),
        "maximum_abs_tangent_z": float(np.max(np.abs(tangents[:, 2]), initial=0.0)),
        "median_ray_length_m": float(np.median(lengths)),
        "midpoint_fresnel_fwhm_proxy_m": float(
            np.sqrt(wavelength * np.median(lengths) / 4.0)
        ),
    }


def build_target_data(
    snapshot: RegularGridInterpolator,
    baseline: RegularGridInterpolator,
    coordinates: np.ndarray,
    pairs: np.ndarray,
    config: MultiSliceConfig,
) -> np.ndarray:
    """Build only the finite-frequency line-integral data for a shifted window."""
    nodes, axial_weights = leggauss(config.quadrature_order)
    starts, tangents, lengths, normal_one, normal_two = _ray_geometry(coordinates, pairs)
    wavelength = config.sound_speed_m_s / config.frequency_hz
    target = np.empty(len(pairs), dtype=np.float64)
    transverse_weights_base = np.asarray((1/3, 1/6, 1/6, 1/6, 1/6), dtype=np.float64)
    batch_size = 64
    for begin in range(0, len(pairs), batch_size):
        end = min(begin + batch_size, len(pairs))
        batch_lengths = lengths[begin:end]
        distance = 0.5 * (nodes[None] + 1.0) * batch_lengths[:, None]
        center = starts[begin:end, None] + distance[:, :, None] * tangents[begin:end, None]
        fresnel = np.sqrt(np.maximum(
            wavelength * distance * (batch_lengths[:, None] - distance)
            / batch_lengths[:, None], 1e-18,
        ))
        sigma = np.maximum(fresnel / 2.355, wavelength / 8.0)
        radius = np.sqrt(3.0) * sigma
        offsets = np.stack((
            np.zeros_like(center),
            radius[:, :, None] * normal_one[begin:end, None],
            -radius[:, :, None] * normal_one[begin:end, None],
            radius[:, :, None] * normal_two[begin:end, None],
            -radius[:, :, None] * normal_two[begin:end, None],
        ), axis=2)
        points = center[:, :, None] + offsets
        inside = (
            (points[..., 0] >= 0.0) & (points[..., 0] <= config.length_xy_m)
            & (points[..., 1] >= 0.0) & (points[..., 1] <= config.length_xy_m)
            & (points[..., 2] >= config.slab_z_min_m)
            & (points[..., 2] <= config.slab_z_max_m)
        )
        transverse_weights = transverse_weights_base[None, None, :] * inside
        transverse_weights /= np.maximum(
            transverse_weights.sum(axis=2, keepdims=True), 1e-30,
        )
        truth_values = (snapshot(points.reshape(-1, 3)) - baseline(
            points.reshape(-1, 3)
        )).reshape(end - begin, config.quadrature_order, 5, 3)
        longitudinal = np.einsum(
            "bqoj,bj->bqo", truth_values, tangents[begin:end], optimize=True,
        )
        transverse = np.einsum(
            "bqo,bqo->bq", longitudinal, transverse_weights, optimize=True,
        )
        target[begin:end] = np.einsum(
            "q,bq,b->b", axial_weights, transverse, 0.5 * batch_lengths,
            optimize=True,
        )
    return target


def factor_gram(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gram = matrix.T @ matrix
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    order = np.argsort(eigenvalues)[::-1]
    return gram, eigenvalues[order], eigenvectors[:, order]


def solve_gcv(
    gram: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    matrix: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    atb = matrix.T @ target
    transformed = eigenvectors.T @ atb
    scale = np.sqrt(max(float(eigenvalues[0]), 1e-30))
    candidates = []
    # A wide operator cannot contain more observable singular directions than
    # rows. Gram roundoff otherwise turns exact null-space eigenvalues into
    # tiny positives and lets GCV select physically nonexistent directions.
    algebraic_rank_ceiling = min(matrix.shape)
    requested_ranks = (
        128, 256, 384, 512, 768, 1024, 1280, 1536, algebraic_rank_ceiling,
    )
    ranks = sorted({
        min(int(rank), algebraic_rank_ceiling) for rank in requested_ranks
    })
    target_energy = float(np.dot(target, target))
    for rank in ranks:
        values = eigenvalues[:rank]
        projected = transformed[:rank]
        vectors = eigenvectors[:, :rank]
        for relative in np.logspace(-5, 1, 13):
            ridge = float(relative * scale)
            denominator = values + ridge**2
            spectral_coefficients = projected / denominator
            # Exact residual norm from A^T A and A^T b, including the part of b
            # orthogonal to the retained geometry-defined spectral subspace.
            residual_energy = max(
                target_energy
                - 2.0 * float(np.dot(spectral_coefficients, projected))
                + float(np.dot(values * spectral_coefficients, spectral_coefficients)),
                0.0,
            )
            degrees = float(np.sum(values / denominator))
            gcv = float(residual_energy / max(len(target) - degrees, 1e-6)**2)
            candidates.append((gcv, rank, relative, ridge, degrees,
                               residual_energy))
    selected = min(candidates, key=lambda value: value[0])
    selected_rank = int(selected[1])
    selected_values = eigenvalues[:selected_rank]
    selected_projected = transformed[:selected_rank]
    coefficients = eigenvectors[:, :selected_rank] @ (
        selected_projected / (selected_values + float(selected[3])**2)
    )
    residual_relative = float(np.sqrt(selected[5]) / max(np.linalg.norm(target), 1e-30))
    return coefficients, {
        "gcv": selected[0],
        "selected_spectral_rank": int(selected[1]),
        "selected_relative_ridge": float(selected[2]),
        "ridge": float(selected[3]),
        "effective_degrees_of_freedom": float(selected[4]),
        "delay_residual_relative": residual_relative,
        "candidate_summary": [
            {"spectral_rank": int(item[1]), "relative_ridge": float(item[2]),
             "gcv": float(item[0]), "effective_degrees_of_freedom": float(item[4])}
            for item in candidates
        ],
    }


def evaluate_coefficients(
    coefficients: np.ndarray,
    basis: CurlBasis3D,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    config: MultiSliceConfig,
) -> np.ndarray:
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    curl_coefficients = coefficients[:basis.curl_mode_count]
    normalized = curl_coefficients.reshape(len(basis.centers), 3) / basis.mode_norms.reshape(
        len(basis.centers), 3,
    )
    result = np.empty((len(points), 3), dtype=np.float64)
    for begin in range(0, len(points), 512):
        gradient = _envelope_gradient(
            points[begin:begin + 512], basis, config.length_xy_m,
        )
        gx, gy, gz = (gradient[:, :, axis] for axis in range(3))
        a0, a1, a2 = (normalized[:, axis] for axis in range(3))
        result[begin:begin + 512, 0] = -gz @ a1 + gy @ a2
        result[begin:begin + 512, 1] = gz @ a0 - gx @ a2
        result[begin:begin + 512, 2] = -gy @ a0 + gx @ a1
        if basis.axial_mode_count:
            axial = _axial_sine_values(
                points[begin:begin + 512], basis.axial_sine_order,
                config.length_xy_m,
                config.slab_z_max_m - config.slab_z_min_m,
            )
            result[begin:begin + 512, 2] += (
                axial @ coefficients[basis.curl_mode_count:]
            )
    return np.moveaxis(result.reshape(*config.evaluation_shape, 3), -1, 0)


def field_metrics(estimate: np.ndarray, truth: np.ndarray) -> dict[str, object]:
    error = estimate - truth
    component_l2, correlation, amplitude = [], [], []
    for component in range(3):
        a, b = estimate[component].ravel(), truth[component].ravel()
        component_l2.append(float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)))
        correlation.append(float(np.corrcoef(a, b)[0, 1]))
        amplitude.append(float(np.linalg.norm(a) / max(np.linalg.norm(b), 1e-30) - 1.0))
    return {
        "joint_relative_l2": float(np.linalg.norm(error) / max(np.linalg.norm(truth), 1e-30)),
        "component_relative_l2": component_l2,
        "component_correlation": correlation,
        "component_amplitude_bias": amplitude,
        "truth_component_rms_m_s": np.sqrt(np.mean(truth**2, axis=(1, 2, 3))).tolist(),
        "estimate_component_rms_m_s": np.sqrt(np.mean(estimate**2, axis=(1, 2, 3))).tolist(),
    }


def divergence_metric(field: np.ndarray, axes: tuple[np.ndarray, ...]) -> float:
    divergence = sum(
        np.gradient(field[component], axes[component], axis=component, edge_order=2)
        for component in range(3)
    )
    speed_scale = np.linalg.norm(field) / np.sqrt(field[0].size)
    length_scale = min(float(np.mean(np.diff(axis))) for axis in axes)
    return float(
        np.linalg.norm(divergence) / np.sqrt(divergence.size)
        / max(speed_scale / length_scale, 1e-30)
    )


def _aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "joint_relative_l2_mean": float(np.mean([value["joint_relative_l2"] for value in records])),
        "joint_relative_l2_std": float(np.std([value["joint_relative_l2"] for value in records])),
        "component_relative_l2_mean": np.mean(
            [value["component_relative_l2"] for value in records], axis=0,
        ).tolist(),
        "component_relative_l2_std": np.std(
            [value["component_relative_l2"] for value in records], axis=0,
        ).tolist(),
        "component_correlation_mean": np.mean(
            [value["component_correlation"] for value in records], axis=0,
        ).tolist(),
        "component_correlation_std": np.std(
            [value["component_correlation"] for value in records], axis=0,
        ).tolist(),
        "component_amplitude_bias_mean": np.mean(
            [value["component_amplitude_bias"] for value in records], axis=0,
        ).tolist(),
    }


def evaluate_policy(
    matrix: np.ndarray,
    target: np.ndarray,
    basis: CurlBasis3D,
    truth: np.ndarray,
    axes: tuple[np.ndarray, ...],
    config: MultiSliceConfig,
    *,
    seed_offset: int,
    row_weights: np.ndarray | None = None,
    factorization: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[dict[str, object], np.ndarray]:
    if row_weights is None:
        row_weights = np.ones(len(target), dtype=np.float64)
    row_weights = np.asarray(row_weights, dtype=np.float64)
    if row_weights.shape != target.shape or np.any(row_weights <= 0.0):
        raise ValueError("row_weights must be positive and match target")
    fitted_matrix = matrix * row_weights[:, None]
    fitted_target = target * row_weights
    if factorization is None:
        gram, eigenvalues, eigenvectors = factor_gram(fitted_matrix)
    else:
        if not np.allclose(row_weights, 1.0):
            raise ValueError("shared factorization currently requires unit row weights")
        gram, eigenvalues, eigenvectors = factorization
        if gram.shape != (matrix.shape[1], matrix.shape[1]):
            raise ValueError("factorization shape does not match the operator")
    clean_coefficients, clean_fit = solve_gcv(
        gram, eigenvalues, eigenvectors, fitted_matrix, fitted_target,
    )
    clean_field = evaluate_coefficients(clean_coefficients, basis, axes, config)
    delay_sigma = (
        config.reference_delay_sigma_s * config.reference_noise_frequency_hz
        / config.frequency_hz
    )
    line_sigma = 0.5 * config.sound_speed_m_s**2 * delay_sigma
    primary_z_min = (
        config.sensor_z_m[1] if config.primary_z_min_m is None
        else config.primary_z_min_m
    )
    primary_z_max = (
        config.sensor_z_m[-2] if config.primary_z_max_m is None
        else config.primary_z_max_m
    )
    primary_mask = (
        (axes[2] >= primary_z_min - 1e-9)
        & (axes[2] <= primary_z_max + 1e-9)
    )
    if not np.any(primary_mask):
        raise ValueError("primary z ROI does not intersect the evaluation grid")
    clean_primary = field_metrics(
        clean_field[:, :, :, primary_mask], truth[:, :, :, primary_mask],
    )
    noisy_metrics, noisy_full_slab_metrics, noisy_fields, noisy_fits = [], [], [], []
    for repeat in range(config.noise_repeats):
        rng = np.random.default_rng(config.seed + seed_offset + 1009 * repeat)
        noisy_target = (target + line_sigma * rng.standard_normal(len(target))) * row_weights
        coefficients, fit = solve_gcv(
            gram, eigenvalues, eigenvectors, fitted_matrix, noisy_target,
        )
        field = evaluate_coefficients(coefficients, basis, axes, config)
        noisy_fields.append(field)
        noisy_full_slab_metrics.append(field_metrics(field, truth))
        noisy_metrics.append(field_metrics(
            field[:, :, :, primary_mask], truth[:, :, :, primary_mask],
        ))
        noisy_fits.append(fit)
    singular = np.sqrt(np.maximum(eigenvalues, 0.0))
    nonzero = singular[singular >= singular[0] * 1e-12]
    algebraic_rank_ceiling = min(matrix.shape)
    reported_nonzero = nonzero[:algebraic_rank_ceiling]
    mean_field = np.mean(noisy_fields, axis=0)
    return {
        "operator": {
            "row_count": int(matrix.shape[0]),
            "parameter_count": int(matrix.shape[1]),
            "effective_rank_1e3": int(np.sum(singular >= singular[0] * 1e-3)),
            "numerical_rank_1e12": int(min(len(nonzero), algebraic_rank_ceiling)),
            "condition_nonzero": float(reported_nonzero[0] / reported_nonzero[-1]),
        },
        "timing_noise": {
            "delay_sigma_s": float(delay_sigma),
            "line_integral_sigma_m2_s": float(line_sigma),
            "repeat_count": config.noise_repeats,
        },
        "row_weighting": {
            "minimum": float(np.min(row_weights)),
            "maximum": float(np.max(row_weights)),
            "rms": float(np.sqrt(np.mean(row_weights**2))),
            "definition": "1 + (factor-1)*abs(t_z); geometry-only Fisher balancing",
        },
        "clean_fit": clean_fit,
        "primary_roi_z_m": [float(axes[2][primary_mask][0]),
                            float(axes[2][primary_mask][-1])],
        "clean_metrics": clean_primary,
        "clean_metrics_full_slab_diagnostic": field_metrics(clean_field, truth),
        "noisy_metrics": _aggregate(noisy_metrics),
        "noisy_metrics_full_slab_diagnostic": _aggregate(noisy_full_slab_metrics),
        "noisy_selected_relative_ridges": [
            fit["selected_relative_ridge"] for fit in noisy_fits
        ],
        "mean_field_sampled_fd_relative_divergence": divergence_metric(mean_field, axes),
        "continuum_divergence_exact_by_curl_identity": True,
    }, mean_field


def layer_metrics(
    estimate: np.ndarray,
    truth: np.ndarray,
    z_axis: np.ndarray,
    requested_layers: tuple[float, ...],
) -> list[dict[str, object]]:
    rows = []
    for requested in requested_layers:
        index = int(np.argmin(np.abs(z_axis - requested)))
        rows.append({
            "requested_z_m": float(requested),
            "evaluated_z_m": float(z_axis[index]),
            **field_metrics(estimate[:, :, :, index:index + 1],
                            truth[:, :, :, index:index + 1]),
        })
    return rows


def _plot_slices(
    truth: np.ndarray,
    estimate: np.ndarray,
    axes_grid: tuple[np.ndarray, ...],
    config: MultiSliceConfig,
    path: Path,
) -> None:
    target_z = 0.5 * (
        (config.sensor_z_m[1] if config.primary_z_min_m is None else config.primary_z_min_m)
        + (config.sensor_z_m[-2] if config.primary_z_max_m is None else config.primary_z_max_m)
    )
    iz = int(np.argmin(np.abs(axes_grid[2] - target_z)))
    figure, axes = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
    labels = ("Ux'", "Uy'", "Uz'")
    for row in range(3):
        limit = max(float(np.max(np.abs(truth[row, :, :, iz]))), 1e-8)
        for column, (title, value) in enumerate((
            ("Truth", truth[row, :, :, iz]),
            ("Reconstruction", estimate[row, :, :, iz]),
            ("Error", estimate[row, :, :, iz] - truth[row, :, :, iz]),
        )):
            image = axes[row, column].imshow(
                value.T, origin="lower", extent=(0, 1.5, 0, 1.5),
                cmap="RdBu_r", vmin=-limit, vmax=limit,
            )
            axes[row, column].set_title(f"{title}: {labels[row]}")
            axes[row, column].set_xlabel("x (m)")
            axes[row, column].set_ylabel("y (m)")
            figure.colorbar(image, ax=axes[row, column], label="m/s")
    figure.suptitle(
        f"Thin-slab three-component TT reconstruction at z={axes_grid[2][iz]:.2f} m\n"
        f"{len(config.sensor_z_m)} layers, {config.sensors_per_layer} sensors/layer, "
        f"{config.frequency_hz/1000:g} kHz"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_layer_metrics(rows: list[dict[str, object]], path: Path) -> None:
    z = [row["evaluated_z_m"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for component, label in enumerate(("Ux'", "Uy'", "Uz'")):
        axes[0].plot(z, [100 * row["component_relative_l2"][component] for row in rows],
                     "o-", label=label)
        axes[1].plot(z, [row["component_correlation"][component] for row in rows],
                     "o-", label=label)
    axes[0].axhline(45.0, color="black", linestyle=":", label="45% gate")
    axes[1].axhline(0.80, color="black", linestyle=":", label="0.80 gate")
    axes[0].set_ylabel("Component relative L2 (%)")
    axes[1].set_ylabel("Component correlation")
    for axis in axes:
        axis.set_xlabel("z (m)")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_acquisition(coordinates: np.ndarray, layers: np.ndarray, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    scatter = axes[0].scatter(coordinates[:, 0], coordinates[:, 1], c=coordinates[:, 2],
                              cmap="viridis", s=18)
    axes[0].set(xlabel="x (m)", ylabel="y (m)", title="Four-wall coordinates")
    axes[0].set_aspect("equal")
    figure.colorbar(scatter, ax=axes[0], label="z (m)")
    unique, counts = np.unique(layers, return_counts=True)
    axes[1].bar(unique, counts)
    axes[1].set(xlabel="Layer index", ylabel="Sensor count", title="Sensors per z layer")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_experiment(config: MultiSliceConfig) -> dict[str, object]:
    started = time.perf_counter()
    output = Path(config.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot, baseline, truth_metadata = _load_interpolators(config)
    truth, axes = truth_on_grid(snapshot, baseline, config)
    coordinates, walls, layers = acquisition(config)
    full_pairs = select_pairs(walls, layers, "full")
    basis = build_basis(config)
    matrix, data, geometry = build_operator_and_data(
        snapshot, baseline, coordinates, full_pairs, basis, config,
    )
    vectors = coordinates[full_pairs[:, 1]] - coordinates[full_pairs[:, 0]]
    tangents = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    full_weights = (
        1.0 + (config.axial_fisher_weight_factor - 1.0) * np.abs(tangents[:, 2])
    )
    full_result, full_field = evaluate_policy(
        matrix, data, basis, truth, axes, config, seed_offset=0,
        row_weights=full_weights,
    )
    separation = np.abs(layers[full_pairs[:, 0]] - layers[full_pairs[:, 1]])
    ablations = {}
    ablation_fields = {}
    if config.run_pair_ablations:
        for policy, mask, seed_offset in (
            ("same_layer", separation == 0, 100_000),
            ("adjacent", separation <= 1, 200_000),
            ("same_plus_far", (separation == 0) | (separation >= 2), 300_000),
            ("far_only", separation >= 2, 400_000),
        ):
            result, field = evaluate_policy(
                matrix[mask], data[mask], basis, truth, axes, config,
                seed_offset=seed_offset,
            )
            ablations[policy] = result
            ablation_fields[policy] = field
    layers_full = layer_metrics(full_field, truth, axes[2], config.sensor_z_m)
    full_noisy = full_result["noisy_metrics"]
    gates = {
        "joint_l2_le_30pct": bool(full_noisy["joint_relative_l2_mean"] <= 0.30),
        "each_component_l2_le_45pct": bool(
            max(full_noisy["component_relative_l2_mean"]) <= 0.45
        ),
        "each_component_correlation_ge_080": bool(
            min(full_noisy["component_correlation_mean"]) >= 0.80
        ),
        "continuum_divergence_exact": bool(
            full_result["continuum_divergence_exact_by_curl_identity"]
        ),
    }
    report = {
        "contract": {
            **asdict(config),
            "full_wave_used": False,
            "truth_used_for_hyperparameter_selection": False,
            "truth_definition": truth_metadata["truth_definition"],
            "inverse_parameterization": (
                "anisotropic localized curl basis with x/y no-slip envelope; "
                "exactly continuum divergence-free"
                + (
                    f"; augmented by {basis.axial_sine_order}x{basis.axial_sine_order} "
                    "z-invariant axial sine modes"
                    if basis.axial_mode_count else ""
                )
            ),
            "ridge_selection": "GCV on observed line-integral data",
            "primary_metrics": "wake-disturbance Ux', Uy', Uz', never total Uz dominated by 0.5 m/s base",
        },
        "truth_provenance": truth_metadata,
        "truth_metrics": {
            "component_rms_m_s": np.sqrt(np.mean(truth**2, axis=(1, 2, 3))).tolist(),
            "component_max_abs_m_s": np.max(np.abs(truth), axis=(1, 2, 3)).tolist(),
        },
        "acquisition": {
            "sensor_count": int(len(coordinates)),
            "sensors_per_layer": config.sensors_per_layer,
            "layer_count": len(config.sensor_z_m),
            "full_pair_count": int(len(full_pairs)),
            "coordinates": coordinates.tolist(),
            "layers": layers.tolist(),
            **geometry,
        },
        "basis": {
            "center_shape": list(config.center_shape),
            "center_count": int(len(basis.centers)),
            "mode_count": basis.mode_count,
            "curl_mode_count": basis.curl_mode_count,
            "axial_sine_order": basis.axial_sine_order,
            "axial_mode_count": basis.axial_mode_count,
            "sigma_xy_m": basis.sigma_xy_m,
            "sigma_z_m": basis.sigma_z_m,
            "mode_norm_min": float(np.min(basis.mode_norms)),
            "mode_norm_max": float(np.max(basis.mode_norms)),
        },
        "full_cross_layer_result": full_result,
        "pair_ablations": ablations,
        "observed_layer_metrics_for_mean_noisy_field": layers_full,
        "gates": gates,
        "all_gates_passed": bool(all(gates.values())),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    reconstruction_arrays = {
        "truth": truth, "reconstruction": full_field,
        "x": axes[0], "y": axes[1], "z": axes[2],
        "coordinates": coordinates, "pairs": full_pairs,
    }
    for policy, field in ablation_fields.items():
        reconstruction_arrays[f"{policy}_reconstruction"] = field
    np.savez_compressed(output / "reconstruction.npz", **reconstruction_arrays)
    np.savez_compressed(
        output / "operator_diagnostics.npz",
        gram=matrix.T @ matrix,
        data=data,
        mode_norms=basis.mode_norms,
        centers=basis.centers,
    )
    _plot_slices(truth, full_field, axes, config, output / "central_slice_reconstruction.png")
    _plot_layer_metrics(layers_full, output / "layer_metrics.png")
    _plot_acquisition(coordinates, layers, output / "acquisition.png")
    ablation_summary = ""
    if ablations:
        ablation_summary = (
            f"- Same-layer-only component correlations: "
            f"{[round(x,3) for x in ablations['same_layer']['noisy_metrics']['component_correlation_mean']]}.\n"
            f"- Adjacent-or-same component correlations: "
            f"{[round(x,3) for x in ablations['adjacent']['noisy_metrics']['component_correlation_mean']]}.\n"
        )
    summary = f"""# Thin-slab multi-slice Travel-Time result

- Sensors: {len(coordinates)} total, {config.sensors_per_layer} per layer on five z layers.
- Frequency: {config.frequency_hz/1000:g} kHz; full-wave propagation was not used.
- Wake disturbance RMS Ux'/Uy'/Uz': {report['truth_metrics']['component_rms_m_s']} m/s.
- Full cross-layer noisy joint L2: {100*full_noisy['joint_relative_l2_mean']:.2f}%.
- Component L2 Ux'/Uy'/Uz': {[round(100*x,2) for x in full_noisy['component_relative_l2_mean']]}%.
- Component correlation Ux'/Uy'/Uz': {[round(x,3) for x in full_noisy['component_correlation_mean']]}.
{ablation_summary}
- All registered gates passed: {report['all_gates_passed']}.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    return report
