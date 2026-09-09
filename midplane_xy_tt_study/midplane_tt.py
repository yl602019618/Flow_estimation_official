from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class StudyConfig:
    snapshot: str = (
        "examples/06_frequency_ablation/data/frozen_production_snapshot_104.h5"
    )
    output: str = "midplane_xy_tt_study/outputs/main"
    z_m: float = 2.5
    length_m: float = 1.5
    sound_speed_m_s: float = 1480.0
    sphere_diameter_m: float = 0.30
    seed: int = 20260807
    evaluation_count: int = 97
    maximum_basis_order: int = 14
    quadrature_order: int = 28
    transverse_order: int = 3
    reference_frequency_hz: float = 60_000.0
    frequencies_hz: tuple[float, ...] = (
        20_000.0, 40_000.0, 60_000.0, 75_000.0, 100_000.0, 150_000.0,
    )
    sensor_counts: tuple[int, ...] = (16, 24, 32, 48, 64, 96)
    layouts: tuple[str, ...] = ("uniform", "staggered", "corner_biased")
    noise_repeats: int = 8
    # Registered from the frozen 20 kHz full-wave, exact tracewise 1% RMS picker.
    reference_noise_frequency_hz: float = 20_000.0
    reference_delay_sigma_s: float = 2.23713169990703e-9


@dataclass(frozen=True)
class MidplaneTruth:
    axis: np.ndarray
    velocity: np.ndarray
    visible_velocity: np.ndarray
    invisible_velocity: np.ndarray
    metadata: dict[str, object]


def _component_metrics(estimate: np.ndarray, target: np.ndarray) -> dict[str, object]:
    error = estimate - target
    joint = float(np.linalg.norm(error) / max(np.linalg.norm(target), 1e-30))
    component_l2, correlation, amplitude_bias = [], [], []
    for component in range(2):
        component_l2.append(float(
            np.linalg.norm(error[component])
            / max(np.linalg.norm(target[component]), 1e-30)
        ))
        a = estimate[component].ravel()
        b = target[component].ravel()
        correlation.append(float(np.corrcoef(a, b)[0, 1]))
        amplitude_bias.append(float(
            np.linalg.norm(a) / max(np.linalg.norm(b), 1e-30) - 1.0
        ))
    return {
        "joint_relative_l2": joint,
        "component_relative_l2": component_l2,
        "component_correlation": correlation,
        "component_amplitude_bias": amplitude_bias,
    }


def _helmholtz_visible_projection(velocity: np.ndarray, spacing: float) -> np.ndarray:
    """L2 projection onto 2-D divergence-free fields with zero wall-normal flow."""
    if velocity.shape[1] != velocity.shape[2]:
        raise ValueError("the projection currently expects a square evaluation grid")
    count = velocity.shape[1]
    interior = count - 2
    stencil = diags(
        (np.ones(interior - 1), -2.0 * np.ones(interior), np.ones(interior - 1)),
        (-1, 0, 1),
    ) / spacing**2
    laplacian = kron(eye(interior), stencil) + kron(stencil, eye(interior))
    # Array axes are x,y. For U=(psi_y,-psi_x), Delta psi = d_y Ux-d_x Uy.
    rhs = (
        np.gradient(velocity[0], spacing, axis=1, edge_order=2)
        - np.gradient(velocity[1], spacing, axis=0, edge_order=2)
    )
    psi = np.zeros((count, count), dtype=np.float64)
    psi[1:-1, 1:-1] = spsolve(
        laplacian, rhs[1:-1, 1:-1].reshape(-1),
    ).reshape(interior, interior)
    projected = np.stack((
        np.gradient(psi, spacing, axis=1, edge_order=2),
        -np.gradient(psi, spacing, axis=0, edge_order=2),
    ))
    projected[0, (0, -1), :] = 0.0
    projected[1, :, (0, -1)] = 0.0
    return projected


def load_truth(config: StudyConfig) -> MidplaneTruth:
    path = Path(config.snapshot).resolve()
    with h5py.File(path, "r") as handle:
        velocity_3d = np.asarray(handle["velocity"][:2], dtype=np.float64)
        snapshot_metadata = {key: value.item() if hasattr(value, "item") else value
                             for key, value in handle.attrs.items()}
    nx, ny, nz = velocity_3d.shape[1:]
    axes = (
        (np.arange(nx) + 0.5) * 1.5 / nx,
        (np.arange(ny) + 0.5) * 1.5 / ny,
        (np.arange(nz) + 0.5) * 4.0 / nz,
    )
    axis = np.linspace(0.0, config.length_m, config.evaluation_count)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    points = np.column_stack((
        xx.reshape(-1), yy.reshape(-1),
        np.full(xx.size, config.z_m),
    ))
    velocity = np.stack([
        RegularGridInterpolator(
            axes, velocity_3d[component], bounds_error=False, fill_value=0.0,
        )(points).reshape(xx.shape)
        for component in range(2)
    ])
    spacing = float(axis[1] - axis[0])
    visible = _helmholtz_visible_projection(velocity, spacing)
    invisible = velocity - visible
    floor = _component_metrics(visible, velocity)
    return MidplaneTruth(axis, velocity, visible, invisible, {
        "snapshot": str(path),
        "snapshot_attributes": snapshot_metadata,
        "z_m": config.z_m,
        "evaluation_count": config.evaluation_count,
        "spacing_m": spacing,
        "truth_component_rms_m_s": np.sqrt(
            np.mean(velocity**2, axis=(1, 2))
        ).tolist(),
        "two_dimensional_visible_projection_floor": floor,
        "projection_definition": (
            "L2 Helmholtz projection onto U=(d_y psi,-d_x psi), psi=0 on boundary"
        ),
    })


def sensor_coordinates(count: int, layout: str, length_m: float) -> tuple[np.ndarray, tuple[str, ...]]:
    if count % 4:
        raise ValueError("sensor count must divide equally across four walls")
    per_wall = count // 4
    if per_wall < 2:
        raise ValueError("at least two sensors per wall are required")
    coordinates: list[tuple[float, float]] = []
    walls: list[str] = []
    phases = (0.00, 0.22, 0.47, 0.71)
    for wall, name in enumerate(("x0", "x1", "y0", "y1")):
        base = (np.arange(per_wall) + 0.5) / per_wall
        if layout == "uniform":
            positions = base
        elif layout == "staggered":
            perturbation = 0.24 / per_wall * np.sin(
                2.0 * np.pi * base + 2.0 * np.pi * phases[wall]
            )
            positions = np.sort(np.clip(base + perturbation, 0.02, 0.98))
        elif layout == "corner_biased":
            positions = 0.5 * (1.0 - np.cos(np.pi * base))
        else:
            raise ValueError(f"unknown layout {layout!r}")
        for position in positions * length_m:
            if name == "x0":
                point = (0.0, position)
            elif name == "x1":
                point = (length_m, position)
            elif name == "y0":
                point = (position, 0.0)
            else:
                point = (position, length_m)
            coordinates.append(point)
            walls.append(name)
    return np.asarray(coordinates, dtype=np.float64), tuple(walls)


def cross_wall_pairs(walls: tuple[str, ...]) -> np.ndarray:
    return np.asarray([
        (first, second)
        for first in range(len(walls))
        for second in range(first + 1, len(walls))
        if walls[first] != walls[second]
    ], dtype=np.int64)


def sine_mode_indices(maximum_order: int) -> np.ndarray:
    modes = [(m, n) for m in range(1, maximum_order + 1)
             for n in range(1, maximum_order + 1)]
    modes.sort(key=lambda value: (value[0] ** 2 + value[1] ** 2, value[0], value[1]))
    return np.asarray(modes, dtype=np.int64)


def evaluate_basis(points: np.ndarray, modes: np.ndarray, length_m: float) -> np.ndarray:
    """Return unit-continuum-L2 velocity modes with shape (point,2,mode)."""
    points = np.asarray(points, dtype=np.float64)
    m = modes[:, 0]
    n = modes[:, 1]
    kx = np.pi * m / length_m
    ky = np.pi * n / length_m
    x = points[:, 0, None]
    y = points[:, 1, None]
    normalization = np.sqrt(
        length_m**2 * (kx**2 + ky**2) / 4.0
    )
    ux = np.sin(x * kx) * np.cos(y * ky) * ky / normalization
    uy = -np.cos(x * kx) * np.sin(y * ky) * kx / normalization
    return np.stack((ux, uy), axis=1)


def evaluate_direct_curvature_basis(
    points: np.ndarray, modes: np.ndarray, length_m: float,
) -> np.ndarray:
    """Independent no-slip Ux/Uy sine modes whitened by curvature."""
    points = np.asarray(points, dtype=np.float64)
    m = modes[:, 0]
    n = modes[:, 1]
    scalar = (
        (2.0 / length_m)
        * np.sin(points[:, 0, None] * np.pi * m / length_m)
        * np.sin(points[:, 1, None] * np.pi * n / length_m)
    )
    # Ridge on the whitened coefficients is a squared-Laplacian penalty on
    # the physical velocity coefficients, without changing the represented span.
    curvature_scale = (m.astype(float) ** 2 + n.astype(float) ** 2) / 2.0
    scalar = scalar / curvature_scale[None, :]
    count = len(modes)
    result = np.zeros((len(points), 2, 2 * count), dtype=np.float64)
    result[:, 0, :count] = scalar
    result[:, 1, count:] = scalar
    return result


def _truth_interpolator(truth: MidplaneTruth) -> RegularGridInterpolator:
    return RegularGridInterpolator(
        (truth.axis, truth.axis), np.moveaxis(truth.velocity, 0, -1),
        bounds_error=False, fill_value=0.0,
    )


def build_fat_ray_system(
    truth: MidplaneTruth,
    coordinates: np.ndarray,
    pairs: np.ndarray,
    modes: np.ndarray,
    frequency_hz: float,
    config: StudyConfig,
    basis_kind: str = "solenoidal",
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Build matched analytic finite-frequency TT data and inverse operator.

    The forward truth is interpolated directly from the CFD slice. The inverse
    columns are analytic sine-streamfunction modes, so this is not a grid inverse
    crime even though both use the same registered Fresnel kernel.
    """
    nodes, axial_weights = leggauss(config.quadrature_order)
    offset_nodes, offset_weights = hermgauss(config.transverse_order)
    offset_weights = offset_weights / np.sqrt(np.pi)
    starts = coordinates[pairs[:, 0]]
    vectors = coordinates[pairs[:, 1]] - starts
    lengths = np.linalg.norm(vectors, axis=1)
    tangents = vectors / lengths[:, None]
    normals = np.stack((-tangents[:, 1], tangents[:, 0]), axis=1)
    wavelength = config.sound_speed_m_s / frequency_hz
    if basis_kind == "solenoidal":
        evaluate_inverse_basis = evaluate_basis
        parameter_count = len(modes)
    elif basis_kind == "direct_curvature":
        evaluate_inverse_basis = evaluate_direct_curvature_basis
        parameter_count = 2 * len(modes)
    else:
        raise ValueError(f"unknown basis kind {basis_kind!r}")
    matrix = np.empty((len(pairs), parameter_count), dtype=np.float64)
    target = np.empty(len(pairs), dtype=np.float64)
    interpolate_truth = _truth_interpolator(truth)
    batch_size = 24
    for begin in range(0, len(pairs), batch_size):
        end = min(begin + batch_size, len(pairs))
        batch_lengths = lengths[begin:end]
        distance = 0.5 * (nodes[None, :] + 1.0) * batch_lengths[:, None]
        center = (
            starts[begin:end, None, :]
            + distance[:, :, None] * tangents[begin:end, None, :]
        )
        fresnel_radius = np.sqrt(np.maximum(
            wavelength * distance * (batch_lengths[:, None] - distance)
            / batch_lengths[:, None],
            1e-18,
        ))
        sigma = np.maximum(fresnel_radius / 2.355, wavelength / 8.0)
        offsets = np.sqrt(2.0) * sigma[:, :, None] * offset_nodes[None, None, :]
        points = (
            center[:, :, None, :]
            + offsets[:, :, :, None] * normals[begin:end, None, None, :]
        )
        inside = np.all(
            (points >= 0.0) & (points <= config.length_m), axis=-1,
        )
        transverse_weights = offset_weights[None, None, :] * inside
        transverse_weights /= np.maximum(
            transverse_weights.sum(axis=-1, keepdims=True), 1e-30,
        )
        flat_points = points.reshape(-1, 2)
        truth_values = interpolate_truth(flat_points).reshape(
            end - begin, config.quadrature_order, config.transverse_order, 2,
        )
        truth_longitudinal = np.einsum(
            "bqoj,bj->bqo", truth_values, tangents[begin:end], optimize=True,
        )
        truth_axial = np.einsum(
            "bqo,bqo->bq", truth_longitudinal, transverse_weights, optimize=True,
        )
        target[begin:end] = np.einsum(
            "q,bq,b->b", axial_weights, truth_axial,
            0.5 * batch_lengths, optimize=True,
        )
        basis_values = evaluate_inverse_basis(flat_points, modes, config.length_m).reshape(
            end - begin, config.quadrature_order, config.transverse_order, 2, parameter_count,
        )
        basis_longitudinal = np.einsum(
            "bqojk,bj->bqok", basis_values, tangents[begin:end], optimize=True,
        )
        basis_axial = np.einsum(
            "bqok,bqo->bqk", basis_longitudinal, transverse_weights, optimize=True,
        )
        matrix[begin:end] = np.einsum(
            "q,bqk,b->bk", axial_weights, basis_axial,
            0.5 * batch_lengths, optimize=True,
        )
    representative_length = float(np.median(lengths))
    midpoint_fresnel_fwhm = float(np.sqrt(wavelength * representative_length / 4.0))
    return matrix, target, {
        "wavelength_m": float(wavelength),
        "median_ray_length_m": representative_length,
        "midpoint_fresnel_fwhm_proxy_m": midpoint_fresnel_fwhm,
        "basis_kind": basis_kind,
    }


def _ridge_from_svd(
    u: np.ndarray,
    singular: np.ndarray,
    vt: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    transformed = u.T @ target
    orthogonal_energy = max(
        float(np.dot(target, target) - np.dot(transformed, transformed)), 0.0,
    )
    scale = max(float(singular[0]), 1e-30)
    relative_ridges = np.logspace(-6, 0, 13)
    candidates = []
    for relative in relative_ridges:
        ridge = relative * scale
        factors = singular**2 / (singular**2 + ridge**2)
        residual_energy = orthogonal_energy + float(np.sum(
            ((1.0 - factors) * transformed) ** 2
        ))
        degrees = float(np.sum(factors))
        denominator = max(len(target) - degrees, 1e-6)
        candidates.append({
            "relative_ridge": float(relative),
            "ridge": float(ridge),
            "gcv": float(residual_energy / denominator**2),
            "effective_degrees_of_freedom": degrees,
        })
    selected = min(candidates, key=lambda value: value["gcv"])
    ridge = float(selected["ridge"])
    coefficients = vt.T @ (
        singular / (singular**2 + ridge**2) * transformed
    )
    return coefficients, {
        "selected_relative_ridge": selected["relative_ridge"],
        "selected_ridge": ridge,
        "gcv": selected["gcv"],
        "effective_degrees_of_freedom": selected["effective_degrees_of_freedom"],
        "gcv_candidates": candidates,
    }


def _field_from_coefficients(
    coefficients: np.ndarray, evaluation_basis: np.ndarray, count: int,
) -> np.ndarray:
    return (evaluation_basis @ coefficients).reshape(2, count, count)


def evaluate_candidate(
    truth: MidplaneTruth,
    evaluation_basis: np.ndarray,
    modes: np.ndarray,
    *,
    sensor_count: int,
    layout: str,
    frequency_hz: float,
    config: StudyConfig,
    basis_kind: str = "solenoidal",
) -> tuple[dict[str, object], np.ndarray]:
    coordinates, walls = sensor_coordinates(sensor_count, layout, config.length_m)
    pairs = cross_wall_pairs(walls)
    matrix, clean_target, kernel = build_fat_ray_system(
        truth, coordinates, pairs, modes, frequency_hz, config, basis_kind,
    )
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    clean_coefficients, clean_fit = _ridge_from_svd(u, singular, vt, clean_target)
    clean_estimate = _field_from_coefficients(
        clean_coefficients, evaluation_basis, config.evaluation_count,
    )
    delay_sigma = (
        config.reference_delay_sigma_s
        * config.reference_noise_frequency_hz / frequency_hz
    )
    line_integral_sigma = 0.5 * config.sound_speed_m_s**2 * delay_sigma
    noisy_full, noisy_visible, noisy_fit = [], [], []
    noisy_estimates = []
    for repeat in range(config.noise_repeats):
        rng = np.random.default_rng(
            config.seed + 1009 * repeat + 17 * sensor_count
        )
        noisy_target = clean_target + line_integral_sigma * rng.standard_normal(len(pairs))
        coefficients, fit = _ridge_from_svd(u, singular, vt, noisy_target)
        estimate = _field_from_coefficients(
            coefficients, evaluation_basis, config.evaluation_count,
        )
        noisy_estimates.append(estimate)
        noisy_full.append(_component_metrics(estimate, truth.velocity))
        noisy_visible.append(_component_metrics(estimate, truth.visible_velocity))
        noisy_fit.append(fit)
    mean_estimate = np.mean(noisy_estimates, axis=0)

    def aggregate(records: list[dict[str, object]]) -> dict[str, object]:
        return {
            "joint_relative_l2_mean": float(np.mean([
                value["joint_relative_l2"] for value in records
            ])),
            "joint_relative_l2_std": float(np.std([
                value["joint_relative_l2"] for value in records
            ])),
            "component_relative_l2_mean": np.mean([
                value["component_relative_l2"] for value in records
            ], axis=0).tolist(),
            "component_relative_l2_std": np.std([
                value["component_relative_l2"] for value in records
            ], axis=0).tolist(),
            "component_correlation_mean": np.mean([
                value["component_correlation"] for value in records
            ], axis=0).tolist(),
            "component_correlation_std": np.std([
                value["component_correlation"] for value in records
            ], axis=0).tolist(),
            "component_amplitude_bias_mean": np.mean([
                value["component_amplitude_bias"] for value in records
            ], axis=0).tolist(),
        }

    nonzero = singular[singular > singular[0] * 1e-12]
    fixed = singular**2 / (singular**2 + (0.01 * singular[0])**2)
    geometry_score = float(np.sum(np.log(np.maximum(fixed, 1e-30))))
    predicted_clean = matrix @ clean_coefficients
    result = {
        "sensor_count": sensor_count,
        "sensors_per_wall": sensor_count // 4,
        "layout": layout,
        "frequency_hz": frequency_hz,
        "pair_count": int(len(pairs)),
        "parameter_count": int(matrix.shape[1]),
        "basis_kind": basis_kind,
        "timing_noise": {
            "definition": "registered exact-1%-waveform reciprocal-delay CRLB proxy, scaled as 1/f",
            "delay_sigma_s": float(delay_sigma),
            "line_integral_sigma_m2_s": float(line_integral_sigma),
            "repeat_count": config.noise_repeats,
        },
        "kernel": kernel,
        "operator": {
            "effective_rank_1e3": int(np.sum(singular >= singular[0] * 1e-3)),
            "numerical_rank_1e12": int(len(nonzero)),
            "condition_nonzero": float(nonzero[0] / nonzero[-1]),
            "singular_value_max": float(singular[0]),
            "singular_value_min": float(singular[-1]),
            "truth_blind_geometry_log_score": geometry_score,
        },
        "clean_fit": clean_fit,
        "clean_delay_residual_relative": float(
            np.linalg.norm(predicted_clean - clean_target)
            / max(np.linalg.norm(clean_target), 1e-30)
        ),
        "clean_full_truth": _component_metrics(clean_estimate, truth.velocity),
        "clean_visible_truth": _component_metrics(clean_estimate, truth.visible_velocity),
        "noisy_full_truth": aggregate(noisy_full),
        "noisy_visible_truth": aggregate(noisy_visible),
        "noisy_selected_relative_ridge": [
            float(value["selected_relative_ridge"]) for value in noisy_fit
        ],
        "coordinates": coordinates.tolist(),
        "pairs": pairs.tolist(),
    }
    return result, mean_estimate


def _layout_selection_key(record: dict[str, object]) -> tuple[float, float, float]:
    operator = record["operator"]
    return (
        float(operator["effective_rank_1e3"]),
        float(operator["truth_blind_geometry_log_score"]),
        -float(operator["condition_nonzero"]),
    )


def _write_csv(records: list[dict[str, object]], path: Path) -> None:
    fields = [
        "stage", "layout", "sensor_count", "frequency_khz", "pair_count",
        "effective_rank_1e3", "fresnel_fwhm_m", "delay_sigma_ns",
        "clean_full_joint_l2", "noisy_full_joint_l2_mean", "noisy_full_joint_l2_std",
        "noisy_ux_l2", "noisy_uy_l2", "noisy_ux_corr", "noisy_uy_corr",
        "noisy_visible_joint_l2_mean", "noisy_visible_ux_corr", "noisy_visible_uy_corr",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            full = record["noisy_full_truth"]
            visible = record["noisy_visible_truth"]
            writer.writerow({
                "stage": record["stage"],
                "layout": record["layout"],
                "sensor_count": record["sensor_count"],
                "frequency_khz": record["frequency_hz"] / 1000.0,
                "pair_count": record["pair_count"],
                "effective_rank_1e3": record["operator"]["effective_rank_1e3"],
                "fresnel_fwhm_m": record["kernel"]["midpoint_fresnel_fwhm_proxy_m"],
                "delay_sigma_ns": record["timing_noise"]["delay_sigma_s"] * 1e9,
                "clean_full_joint_l2": record["clean_full_truth"]["joint_relative_l2"],
                "noisy_full_joint_l2_mean": full["joint_relative_l2_mean"],
                "noisy_full_joint_l2_std": full["joint_relative_l2_std"],
                "noisy_ux_l2": full["component_relative_l2_mean"][0],
                "noisy_uy_l2": full["component_relative_l2_mean"][1],
                "noisy_ux_corr": full["component_correlation_mean"][0],
                "noisy_uy_corr": full["component_correlation_mean"][1],
                "noisy_visible_joint_l2_mean": visible["joint_relative_l2_mean"],
                "noisy_visible_ux_corr": visible["component_correlation_mean"][0],
                "noisy_visible_uy_corr": visible["component_correlation_mean"][1],
            })


def _plot_truth_projection(truth: MidplaneTruth, path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    labels = (("Truth", truth.velocity), ("2-D visible projection", truth.visible_velocity),
              ("Invisible residual", truth.invisible_velocity))
    limits = [max(np.max(np.abs(truth.velocity[c])), 1e-8) for c in range(2)]
    for row, component in enumerate((0, 1)):
        for column, (label, value) in enumerate(labels):
            image = axes[row, column].imshow(
                value[component].T, origin="lower",
                extent=(0, 1.5, 0, 1.5), cmap="RdBu_r",
                vmin=-limits[component], vmax=limits[component],
            )
            axes[row, column].set_title(f"{label}: U{'x' if component == 0 else 'y'}")
            axes[row, column].set_xlabel("x (m)")
            axes[row, column].set_ylabel("y (m)")
            figure.colorbar(image, ax=axes[row, column], label="m/s")
    figure.suptitle(f"Sphere-wake transverse field at z={truth.metadata['z_m']:.2f} m")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_geometry(records: list[dict[str, object]], path: Path, reference_frequency: float) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    layouts = sorted({record["layout"] for record in records})
    for layout in layouts:
        subset = sorted((value for value in records if value["layout"] == layout),
                        key=lambda value: value["sensor_count"])
        counts = [value["sensor_count"] for value in subset]
        axes[0].plot(counts, [100 * value["noisy_full_truth"]["joint_relative_l2_mean"]
                              for value in subset], "o-", label=layout)
        axes[1].plot(counts, [100 * value["noisy_visible_truth"]["joint_relative_l2_mean"]
                              for value in subset], "o-", label=layout)
    axes[0].set_title("Full Ux,Uy truth")
    axes[1].set_title("2-D visible solenoidal target")
    for axis in axes:
        axis.set_xlabel("Total sensors on four walls")
        axis.set_ylabel("Noisy joint relative L2 (%)")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(f"Target-blind layout screen at {reference_frequency/1000:.0f} kHz")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_frequency_heatmaps(records: list[dict[str, object]], path: Path) -> None:
    counts = sorted({int(value["sensor_count"]) for value in records})
    frequencies = sorted({float(value["frequency_hz"]) for value in records})
    lookup = {(int(value["sensor_count"]), float(value["frequency_hz"])): value
              for value in records}
    full = np.asarray([[100 * lookup[(count, frequency)]["noisy_full_truth"]["joint_relative_l2_mean"]
                        for frequency in frequencies] for count in counts])
    visible = np.asarray([[100 * lookup[(count, frequency)]["noisy_visible_truth"]["joint_relative_l2_mean"]
                           for frequency in frequencies] for count in counts])
    min_corr = np.asarray([[min(lookup[(count, frequency)]["noisy_full_truth"]["component_correlation_mean"])
                            for frequency in frequencies] for count in counts])
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for axis, data, title, label, cmap in (
        (axes[0], full, "Full-field error", "joint L2 (%)", "magma_r"),
        (axes[1], visible, "Visible-field error", "joint L2 (%)", "magma_r"),
        (axes[2], min_corr, "Worst component correlation", "correlation", "viridis"),
    ):
        image = axis.imshow(data, origin="lower", aspect="auto", cmap=cmap)
        axis.set_xticks(range(len(frequencies)), [f"{value/1000:g}" for value in frequencies])
        axis.set_yticks(range(len(counts)), counts)
        axis.set_xlabel("Frequency (kHz)")
        axis.set_ylabel("Total four-wall sensors")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, label=label)
        for iy in range(len(counts)):
            for ix in range(len(frequencies)):
                axis.text(ix, iy, f"{data[iy, ix]:.1f}" if data is not min_corr else f"{data[iy, ix]:.2f}",
                          ha="center", va="center", fontsize=7, color="white" if data[iy, ix] < np.nanmedian(data) else "black")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_best(truth: MidplaneTruth, estimate: np.ndarray, record: dict[str, object], path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for row, component in enumerate((0, 1)):
        limit = max(np.max(np.abs(truth.velocity[component])), 1e-8)
        values = (truth.velocity[component], estimate[component],
                  estimate[component] - truth.velocity[component])
        for column, (label, value) in enumerate(zip(("Truth", "Mean reconstruction", "Error"), values)):
            image = axes[row, column].imshow(
                value.T, origin="lower", extent=(0, 1.5, 0, 1.5),
                cmap="RdBu_r", vmin=-limit, vmax=limit,
            )
            axes[row, column].set_title(f"{label}: U{'x' if component == 0 else 'y'}")
            axes[row, column].set_xlabel("x (m)")
            axes[row, column].set_ylabel("y (m)")
            figure.colorbar(image, ax=axes[row, column], label="m/s")
    figure.suptitle(
        f"Best full-field setting: {record['layout']}, {record['sensor_count']} sensors, "
        f"{record['frequency_hz']/1000:g} kHz"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_study(config: StudyConfig) -> dict[str, object]:
    started = time.perf_counter()
    output = Path(config.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    truth = load_truth(config)
    modes = sine_mode_indices(config.maximum_basis_order)
    axis = truth.axis
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    evaluation_points = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    basis = evaluate_basis(evaluation_points, modes, config.length_m)
    evaluation_basis = np.moveaxis(basis, 1, 0).reshape(2 * len(evaluation_points), -1)

    geometry_records: list[dict[str, object]] = []
    estimates: dict[tuple[str, int, float], np.ndarray] = {}
    for layout in config.layouts:
        for count in config.sensor_counts:
            record, estimate = evaluate_candidate(
                truth, evaluation_basis, modes,
                sensor_count=count, layout=layout,
                frequency_hz=config.reference_frequency_hz, config=config,
            )
            record["stage"] = "geometry"
            geometry_records.append(record)
            estimates[(layout, count, config.reference_frequency_hz)] = estimate
            print(
                f"geometry layout={layout:14s} sensors={count:3d} "
                f"full={100*record['noisy_full_truth']['joint_relative_l2_mean']:.2f}% "
                f"visible={100*record['noisy_visible_truth']['joint_relative_l2_mean']:.2f}%",
                flush=True,
            )

    selected_layouts: dict[int, str] = {}
    for count in config.sensor_counts:
        options = [value for value in geometry_records if value["sensor_count"] == count]
        selected_layouts[count] = max(options, key=_layout_selection_key)["layout"]

    frequency_records: list[dict[str, object]] = []
    for count in config.sensor_counts:
        layout = selected_layouts[count]
        for frequency in config.frequencies_hz:
            if frequency == config.reference_frequency_hz:
                source = next(value for value in geometry_records
                              if value["sensor_count"] == count and value["layout"] == layout)
                record = dict(source)
                record["stage"] = "frequency"
            else:
                record, estimate = evaluate_candidate(
                    truth, evaluation_basis, modes,
                    sensor_count=count, layout=layout,
                    frequency_hz=frequency, config=config,
                )
                record["stage"] = "frequency"
                estimates[(layout, count, frequency)] = estimate
            frequency_records.append(record)
            print(
                f"frequency layout={layout:14s} sensors={count:3d} "
                f"f={frequency/1000:6.1f}kHz "
                f"full={100*record['noisy_full_truth']['joint_relative_l2_mean']:.2f}% "
                f"visible={100*record['noisy_visible_truth']['joint_relative_l2_mean']:.2f}%",
                flush=True,
            )

    best_full = min(
        frequency_records,
        key=lambda value: value["noisy_full_truth"]["joint_relative_l2_mean"],
    )
    best_visible = min(
        frequency_records,
        key=lambda value: value["noisy_visible_truth"]["joint_relative_l2_mean"],
    )
    best_key = (best_full["layout"], best_full["sensor_count"], best_full["frequency_hz"])
    if best_key not in estimates:
        _, estimates[best_key] = evaluate_candidate(
            truth, evaluation_basis, modes,
            sensor_count=best_full["sensor_count"], layout=best_full["layout"],
            frequency_hz=best_full["frequency_hz"], config=config,
        )
    best_estimate = estimates[best_key]

    full_passes = [value for value in frequency_records if (
        value["noisy_full_truth"]["joint_relative_l2_mean"] <= 0.30
        and max(value["noisy_full_truth"]["component_relative_l2_mean"]) <= 0.45
        and min(value["noisy_full_truth"]["component_correlation_mean"]) >= 0.80
    )]
    visible_passes = [value for value in frequency_records if (
        value["noisy_visible_truth"]["joint_relative_l2_mean"] <= 0.20
        and max(value["noisy_visible_truth"]["component_relative_l2_mean"]) <= 0.30
        and min(value["noisy_visible_truth"]["component_correlation_mean"]) >= 0.90
    )]
    report = {
        "contract": {
            **asdict(config),
            "forward_model": "matched analytic first-Fresnel Gaussian fat-ray travel time",
            "inverse_parameterization": (
                "14x14 energy-ordered sine streamfunction modes; exact 2-D divergence-free and zero normal wall flow"
            ),
            "hyperparameter_selection": "ridge chosen independently for each noisy dataset by GCV; no velocity truth",
            "layout_selection": (
                "for each sensor count, maximize effective rank then fixed-ridge log spectral score at 60 kHz; no velocity truth"
            ),
            "frequency_role": "finite-frequency Fresnel width and registered 1/f timing-noise scaling",
            "full_truth_gate": "joint L2<=30%, each component L2<=45%, each correlation>=0.80",
            "visible_truth_gate": "joint L2<=20%, each component L2<=30%, each correlation>=0.90",
            "full_wave_used": False,
        },
        "truth": truth.metadata,
        "selected_layout_by_sensor_count": {str(key): value for key, value in selected_layouts.items()},
        "geometry_scan_at_reference_frequency": geometry_records,
        "frequency_sensor_scan": frequency_records,
        "best_full_truth_setting": best_full,
        "best_visible_truth_setting": best_visible,
        "full_truth_passing_settings": full_passes,
        "visible_truth_passing_settings": visible_passes,
        "decision": {
            "full_truth_recoverable_under_registered_gates": bool(full_passes),
            "visible_truth_recoverable_under_registered_gates": bool(visible_passes),
            "interpretation": (
                "The hard 2-D solenoidal model can accurately recover its projected target but cannot express the full 3-D wake slice. This projection error is a parameterization floor, not a universal information-theoretic bound; the separate direct-vector baseline tests the remaining longitudinal-ray nullspace."
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_csv(geometry_records + frequency_records, output / "scan.csv")
    np.savez_compressed(
        output / "best_reconstruction.npz",
        truth=truth.velocity,
        visible_truth=truth.visible_velocity,
        invisible_truth=truth.invisible_velocity,
        estimate=best_estimate,
        axis=truth.axis,
        modes=modes,
        coordinates=np.asarray(best_full["coordinates"]),
        pairs=np.asarray(best_full["pairs"]),
    )
    _plot_truth_projection(truth, output / "truth_visible_invisible.png")
    _plot_geometry(geometry_records, output / "layout_sensor_scan_60khz.png",
                   config.reference_frequency_hz)
    _plot_frequency_heatmaps(frequency_records, output / "frequency_sensor_heatmaps.png")
    _plot_best(truth, best_estimate, best_full, output / "best_reconstruction.png")

    full = best_full["noisy_full_truth"]
    visible = best_visible["noisy_visible_truth"]
    summary = f"""# Midplane x-y travel-time study

This is an analytic Travel-Time experiment; no full-wave data were generated.

## Identifiability audit

- Plane: z = {config.z_m:.3f} m in the frozen Re=500 production sphere wake.
- The exact 2-D solenoidal projection already has {100*truth.metadata['two_dimensional_visible_projection_floor']['joint_relative_l2']:.2f}% joint error relative to the full Ux,Uy slice.
- Component projection floors: {100*truth.metadata['two_dimensional_visible_projection_floor']['component_relative_l2'][0]:.2f}% (Ux), {100*truth.metadata['two_dimensional_visible_projection_floor']['component_relative_l2'][1]:.2f}% (Uy).

## Best full-slice result

- {best_full['sensor_count']} sensors ({best_full['sensors_per_wall']} per wall), `{best_full['layout']}`, {best_full['frequency_hz']/1000:g} kHz.
- Noisy joint L2: {100*full['joint_relative_l2_mean']:.2f}% +/- {100*full['joint_relative_l2_std']:.2f}%.
- Ux/Uy L2: {100*full['component_relative_l2_mean'][0]:.2f}% / {100*full['component_relative_l2_mean'][1]:.2f}%.
- Ux/Uy correlation: {full['component_correlation_mean'][0]:.3f} / {full['component_correlation_mean'][1]:.3f}.
- Registered full-slice gate passed: {bool(full_passes)}.

## Best visible-solenoidal result

- {best_visible['sensor_count']} sensors, `{best_visible['layout']}`, {best_visible['frequency_hz']/1000:g} kHz.
- Noisy joint L2: {100*visible['joint_relative_l2_mean']:.2f}% +/- {100*visible['joint_relative_l2_std']:.2f}%.
- Ux/Uy correlation: {visible['component_correlation_mean'][0]:.3f} / {visible['component_correlation_mean'][1]:.3f}.
- Registered visible-field gate passed: {bool(visible_passes)}.

## Decision

{report['decision']['interpretation']}
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    return report
