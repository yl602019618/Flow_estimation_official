from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from multislice_xyz_tt_study.multislice_tt import (
    CurlBasis3D,
    MultiSliceConfig,
    _aggregate,
    _load_interpolators,
    _plot_acquisition,
    _plot_layer_metrics,
    _plot_slices,
    acquisition,
    build_basis,
    build_operator_and_data,
    build_target_data,
    divergence_metric,
    evaluate_coefficients,
    field_metrics,
    layer_metrics,
    select_pairs,
    truth_on_grid,
)


@dataclass(frozen=True)
class DenseSolveConfig:
    pair_budget: int = 36_000
    validation_fraction: float = 0.10
    ridge_relatives: tuple[float, ...] = (
        1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3,
        3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1, 3.0e-1,
    )
    power_iterations: int = 16
    cg_max_iterations: int = 400
    cg_relative_tolerance: float = 1.0e-4
    device: str = "cuda"
    pair_seed_offset: int = 71_003
    validation_seed_offset: int = 91_009
    operator_cache: str | None = None


_WALL_ORDER = {"x0": 0, "x1": 1, "y0": 2, "y1": 3}
_SEPARATION_NAMES = ("same", "near", "medium", "far")


def operator_fingerprint(
    coordinates: np.ndarray,
    pairs: np.ndarray,
    basis: CurlBasis3D,
    config: MultiSliceConfig,
) -> str:
    """Fingerprint every quantity that changes the analytic ray operator."""
    digest = hashlib.sha256()
    payload = {
        "length_xy_m": config.length_xy_m,
        "slab_z_min_m": config.slab_z_min_m,
        "slab_z_max_m": config.slab_z_max_m,
        "sound_speed_m_s": config.sound_speed_m_s,
        "frequency_hz": config.frequency_hz,
        "quadrature_order": config.quadrature_order,
        "operator_dtype": config.operator_dtype,
        "axial_sine_order": basis.axial_sine_order,
    }
    digest.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    for values in (coordinates, pairs, basis.centers, basis.mode_norms):
        contiguous = np.ascontiguousarray(values)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _operator_cache_metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def load_operator_cache(
    path: Path, expected_fingerprint: str,
) -> tuple[np.ndarray, dict[str, object]]:
    metadata_path = _operator_cache_metadata_path(path)
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("fingerprint") != expected_fingerprint:
        raise ValueError(f"operator cache fingerprint mismatch: {path}")
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(matrix.shape) != metadata.get("shape"):
        raise ValueError(f"operator cache shape metadata mismatch: {path}")
    if str(matrix.dtype) != metadata.get("dtype"):
        raise ValueError(f"operator cache dtype metadata mismatch: {path}")
    return matrix, metadata


def save_operator_cache(
    path: Path,
    matrix: np.ndarray,
    fingerprint: str,
    geometry: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_matrix = path.with_suffix(path.suffix + ".tmp")
    temporary_metadata = _operator_cache_metadata_path(path).with_suffix(
        _operator_cache_metadata_path(path).suffix + ".tmp"
    )
    with temporary_matrix.open("wb") as handle:
        np.save(handle, matrix, allow_pickle=False)
    metadata = {
        "fingerprint": fingerprint,
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "storage_bytes": int(matrix.nbytes),
        "geometry": geometry,
    }
    temporary_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary_matrix.replace(path)
    temporary_metadata.replace(_operator_cache_metadata_path(path))


def pair_strata(
    walls: tuple[str, ...], layers: np.ndarray, pairs: np.ndarray,
) -> tuple[np.ndarray, dict[int, str]]:
    """Encode wall-pair and axial-separation groups without using the truth."""
    wall_pairs = sorted({
        tuple(sorted((_WALL_ORDER[walls[first]], _WALL_ORDER[walls[second]])))
        for first, second in pairs
    })
    wall_code = {pair: index for index, pair in enumerate(wall_pairs)}
    labels = np.empty(len(pairs), dtype=np.int64)
    names: dict[int, str] = {}
    for index, (first, second) in enumerate(pairs):
        separation = abs(int(layers[first]) - int(layers[second]))
        if separation == 0:
            separation_code = 0
        elif separation <= 2:
            separation_code = 1
        elif separation <= 5:
            separation_code = 2
        else:
            separation_code = 3
        pair = tuple(sorted((_WALL_ORDER[walls[first]], _WALL_ORDER[walls[second]])))
        code = 4 * wall_code[pair] + separation_code
        labels[index] = code
        wall_name = "-".join((walls[first], walls[second]))
        names[code] = f"{wall_name}:{_SEPARATION_NAMES[separation_code]}"
    return labels, names


def stratified_pairs(
    walls: tuple[str, ...],
    layers: np.ndarray,
    budget: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    all_pairs = select_pairs(walls, layers, "full")
    all_labels, label_names = pair_strata(walls, layers, all_pairs)
    if budget <= 0:
        raise ValueError("pair budget must be positive")
    if budget >= len(all_pairs):
        selected = np.arange(len(all_pairs), dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        groups = [np.flatnonzero(all_labels == label) for label in np.unique(all_labels)]
        quota = budget // len(groups)
        chosen: list[np.ndarray] = []
        used = np.zeros(len(all_pairs), dtype=bool)
        for group in groups:
            take = min(quota, len(group))
            subset = rng.choice(group, size=take, replace=False)
            chosen.append(subset)
            used[subset] = True
        retained = int(sum(len(value) for value in chosen))
        if retained < budget:
            remaining = np.flatnonzero(~used)
            extra = rng.choice(remaining, size=budget - retained, replace=False)
            chosen.append(extra)
        selected = np.sort(np.concatenate(chosen))
    pairs = all_pairs[selected]
    labels = all_labels[selected]
    counts = {
        label_names[int(label)]: int(np.sum(labels == label))
        for label in np.unique(labels)
    }
    return pairs, labels, {
        "all_cross_wall_pair_count": int(len(all_pairs)),
        "selected_pair_count": int(len(pairs)),
        "selection_fraction": float(len(pairs) / len(all_pairs)),
        "selection_definition": (
            "target-blind equal allocation over wall-pair x "
            "same/near/medium/far layer-separation strata, then frozen-seed fill"
        ),
        "stratum_counts": counts,
    }


def stratified_train_validation(
    labels: np.ndarray,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < fraction < 0.5:
        raise ValueError("validation fraction must lie in (0, 0.5)")
    rng = np.random.default_rng(seed)
    validation: list[np.ndarray] = []
    for label in np.unique(labels):
        group = np.flatnonzero(labels == label)
        count = min(len(group) - 1, max(1, int(round(fraction * len(group)))))
        validation.append(rng.choice(group, size=count, replace=False))
    validation_index = np.sort(np.concatenate(validation))
    train_mask = np.ones(len(labels), dtype=bool)
    train_mask[validation_index] = False
    return np.flatnonzero(train_mask), validation_index


class _TorchRows:
    def __init__(self, blocks: list[object]):
        self.blocks = blocks
        self.parameter_count = int(blocks[0].shape[1])
        if any(int(block.shape[1]) != self.parameter_count for block in blocks):
            raise ValueError("operator blocks have inconsistent parameter counts")

    def normal(self, vector: object) -> object:
        result = None
        for block in self.blocks:
            contribution = block.T @ (block @ vector)
            result = contribution if result is None else result + contribution
        return result

    def transpose_targets(self, targets: list[object]) -> object:
        result = None
        for block, target in zip(self.blocks, targets):
            contribution = block.T @ target
            result = contribution if result is None else result + contribution
        return result

    def prediction_norm_squared(self, vector: object) -> object:
        result = None
        for block in self.blocks:
            contribution = (block @ vector).square().sum()
            result = contribution if result is None else result + contribution
        return result

    def frobenius_squared(self) -> object:
        result = None
        for block in self.blocks:
            contribution = block.square().sum()
            result = contribution if result is None else result + contribution
        return result


def _spectral_scale(rows: _TorchRows, iterations: int, torch: object) -> float:
    generator = torch.Generator(device=rows.blocks[0].device)
    generator.manual_seed(20260807)
    vector = torch.randn(
        rows.parameter_count, generator=generator,
        device=rows.blocks[0].device, dtype=rows.blocks[0].dtype,
    )
    vector /= torch.linalg.vector_norm(vector).clamp_min(1.0e-30)
    for _ in range(iterations):
        vector = rows.normal(vector)
        vector /= torch.linalg.vector_norm(vector).clamp_min(1.0e-30)
    sigma_squared = torch.dot(vector, rows.normal(vector)).clamp_min(0.0)
    return float(torch.sqrt(sigma_squared).item())


def _ridge_pcg(
    rows: _TorchRows,
    targets: list[object],
    ridge: float,
    max_iterations: int,
    relative_tolerance: float,
    torch: object,
) -> tuple[object, dict[str, object]]:
    right = rows.transpose_targets(targets)
    diagonal = None
    for block in rows.blocks:
        contribution = block.square().sum(dim=0)
        diagonal = contribution if diagonal is None else diagonal + contribution
    diagonal = diagonal + ridge * ridge
    inverse_diagonal = diagonal.clamp_min(torch.finfo(diagonal.dtype).tiny).reciprocal()
    solution = torch.zeros_like(right)
    residual = right.clone()
    preconditioned = inverse_diagonal * residual
    direction = preconditioned.clone()
    rz = torch.dot(residual, preconditioned)
    right_norm = torch.linalg.vector_norm(right).clamp_min(1.0e-30)
    relative = float((torch.linalg.vector_norm(residual) / right_norm).item())
    completed = 0
    for iteration in range(1, max_iterations + 1):
        applied = rows.normal(direction) + ridge * ridge * direction
        denominator = torch.dot(direction, applied)
        if float(torch.abs(denominator).item()) <= torch.finfo(denominator.dtype).tiny:
            break
        alpha = rz / denominator
        solution += alpha * direction
        residual -= alpha * applied
        relative = float((torch.linalg.vector_norm(residual) / right_norm).item())
        completed = iteration
        if relative <= relative_tolerance:
            break
        new_preconditioned = inverse_diagonal * residual
        new_rz = torch.dot(residual, new_preconditioned)
        direction = new_preconditioned + (new_rz / rz) * direction
        preconditioned = new_preconditioned
        rz = new_rz
    return solution, {
        "iterations": int(completed),
        "normal_equation_relative_residual": float(relative),
        "converged": bool(relative <= relative_tolerance),
    }


def scalable_ridge_solve(
    matrix: np.ndarray,
    target: np.ndarray,
    labels: np.ndarray,
    flow_config: MultiSliceConfig,
    solve_config: DenseSolveConfig,
    noise_standard_normals: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, list[np.ndarray], dict[str, object]]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("scalable dense-ring inversion requires PyTorch") from exc
    requested_device = solve_config.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
    if noise_standard_normals is not None:
        if len(noise_standard_normals) != flow_config.noise_repeats:
            raise ValueError("noise realization count does not match noise_repeats")
        if any(np.asarray(values).shape != target.shape for values in noise_standard_normals):
            raise ValueError("each supplied noise realization must match target")
    train_index, validation_index = stratified_train_validation(
        labels, solve_config.validation_fraction,
        flow_config.seed + solve_config.validation_seed_offset,
    )
    dtype = torch.float32 if matrix.dtype == np.float32 else torch.float64
    train_matrix = torch.as_tensor(
        np.ascontiguousarray(matrix[train_index]), dtype=dtype, device=device,
    )
    validation_matrix = torch.as_tensor(
        np.ascontiguousarray(matrix[validation_index]), dtype=dtype, device=device,
    )
    train_target = torch.as_tensor(target[train_index], dtype=dtype, device=device)
    validation_target = torch.as_tensor(target[validation_index], dtype=dtype, device=device)
    train_rows = _TorchRows([train_matrix])
    validation_rows = _TorchRows([validation_matrix])
    full_rows = _TorchRows([train_matrix, validation_matrix])
    train_scale = _spectral_scale(train_rows, solve_config.power_iterations, torch)
    candidates = []
    best = None
    for relative in solve_config.ridge_relatives:
        ridge = float(relative * train_scale)
        coefficients, cg = _ridge_pcg(
            train_rows, [train_target], ridge,
            solve_config.cg_max_iterations,
            solve_config.cg_relative_tolerance, torch,
        )
        validation_residual = validation_matrix @ coefficients - validation_target
        relative_validation = float(
            (torch.linalg.vector_norm(validation_residual)
             / torch.linalg.vector_norm(validation_target).clamp_min(1.0e-30)).item()
        )
        record = {
            "relative_ridge": float(relative),
            "ridge": ridge,
            "validation_delay_residual_relative": relative_validation,
            **cg,
        }
        candidates.append(record)
        selection_key = (not bool(cg["converged"]), relative_validation)
        if best is None or selection_key < best[0]:
            best = (selection_key, float(relative))
    assert best is not None
    full_scale = _spectral_scale(full_rows, solve_config.power_iterations, torch)
    selected_ridge = float(best[1] * full_scale)
    clean_coefficients, clean_cg = _ridge_pcg(
        full_rows, [train_target, validation_target], selected_ridge,
        solve_config.cg_max_iterations, solve_config.cg_relative_tolerance, torch,
    )
    full_target_norm = torch.sqrt(
        train_target.square().sum() + validation_target.square().sum()
    ).clamp_min(1.0e-30)
    clean_residual_squared = (
        (train_matrix @ clean_coefficients - train_target).square().sum()
        + (validation_matrix @ clean_coefficients - validation_target).square().sum()
    )
    clean_delay_residual = float(torch.sqrt(clean_residual_squared).div(full_target_norm).item())
    line_sigma = (
        0.5 * flow_config.sound_speed_m_s**2
        * flow_config.reference_delay_sigma_s
        * flow_config.reference_noise_frequency_hz / flow_config.frequency_hz
    )
    noisy_coefficients: list[np.ndarray] = []
    noisy_solves: list[dict[str, object]] = []
    for repeat in range(flow_config.noise_repeats):
        if noise_standard_normals is None:
            rng = np.random.default_rng(flow_config.seed + 1009 * repeat)
            standard_normal = rng.standard_normal(len(target))
        else:
            standard_normal = np.asarray(noise_standard_normals[repeat], dtype=np.float64)
        noisy = target + line_sigma * standard_normal
        noisy_train = torch.as_tensor(noisy[train_index], dtype=dtype, device=device)
        noisy_validation = torch.as_tensor(noisy[validation_index], dtype=dtype, device=device)
        coefficients, cg = _ridge_pcg(
            full_rows, [noisy_train, noisy_validation], selected_ridge,
            solve_config.cg_max_iterations, solve_config.cg_relative_tolerance, torch,
        )
        residual_squared = (
            (train_matrix @ coefficients - noisy_train).square().sum()
            + (validation_matrix @ coefficients - noisy_validation).square().sum()
        )
        noisy_norm = torch.sqrt(
            noisy_train.square().sum() + noisy_validation.square().sum()
        ).clamp_min(1.0e-30)
        noisy_solves.append({
            **cg,
            "delay_residual_relative": float(torch.sqrt(residual_squared).div(noisy_norm).item()),
        })
        noisy_coefficients.append(coefficients.detach().cpu().numpy().astype(np.float64))
    stable_rank = float(
        (full_rows.frobenius_squared() / max(full_scale * full_scale, 1.0e-30)).item()
    )
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    direct_reference = None
    if matrix.shape[1] <= 1_000:
        matrix64 = np.asarray(matrix, dtype=np.float64)
        direct = np.linalg.solve(
            matrix64.T @ matrix64 + selected_ridge**2 * np.eye(matrix.shape[1]),
            matrix64.T @ target,
        )
        iterative = clean_coefficients.detach().cpu().numpy().astype(np.float64)
        direct_reference = {
            "coefficient_relative_difference": float(
                np.linalg.norm(iterative - direct) / max(np.linalg.norm(direct), 1.0e-30)
            ),
            "direct_delay_residual_relative": float(
                np.linalg.norm(matrix64 @ direct - target) / max(np.linalg.norm(target), 1.0e-30)
            ),
        }
    report = {
        "device": str(device),
        "operator_storage_dtype": str(matrix.dtype),
        "train_row_count": int(len(train_index)),
        "validation_row_count": int(len(validation_index)),
        "validation_fraction_actual": float(len(validation_index) / len(matrix)),
        "ridge_selection": "frozen stratified ray holdout; velocity truth unused",
        "candidate_summary": candidates,
        "selected_relative_ridge": float(best[1]),
        "selected_ridge": selected_ridge,
        "train_spectral_scale": train_scale,
        "full_spectral_scale": full_scale,
        "stable_rank_proxy": stable_rank,
        "clean_delay_residual_relative": clean_delay_residual,
        "clean_cg": clean_cg,
        "noisy_solves": noisy_solves,
        "line_integral_sigma_m2_s": float(line_sigma),
        "peak_torch_cuda_memory_bytes": peak_memory,
        "small_operator_direct_reference": direct_reference,
    }
    return (
        clean_coefficients.detach().cpu().numpy().astype(np.float64),
        noisy_coefficients,
        report,
    )


def roughness_metrics(
    estimate: np.ndarray,
    truth: np.ndarray,
    axes: tuple[np.ndarray, ...],
    primary_mask: np.ndarray,
) -> dict[str, object]:
    def laplacian(field: np.ndarray) -> np.ndarray:
        result = np.zeros_like(field)
        for axis in range(3):
            first = np.gradient(field, axes[axis], axis=axis + 1, edge_order=2)
            result += np.gradient(first, axes[axis], axis=axis + 1, edge_order=2)
        return result

    truth_lap = laplacian(truth)[:, :, :, primary_mask]
    estimate_lap = laplacian(estimate)[:, :, :, primary_mask]
    truth_norm = max(float(np.linalg.norm(truth_lap)), 1.0e-30)
    ratio = float(np.linalg.norm(estimate_lap) / truth_norm)
    return {
        "second_derivative_norm_ratio_estimate_to_truth": ratio,
        "second_derivative_relative_error": float(
            np.linalg.norm(estimate_lap - truth_lap) / truth_norm
        ),
        "log_roughness_mismatch_abs": float(abs(np.log(max(ratio, 1.0e-30)))),
    }


def _plot_five_slices(
    truth: np.ndarray,
    estimate: np.ndarray,
    axes_grid: tuple[np.ndarray, ...],
    path: Path,
) -> None:
    requested = np.linspace(3.5, 4.5, 5)
    indices = [int(np.argmin(np.abs(axes_grid[2] - value))) for value in requested]
    labels = ("Ux'", "Uy'", "Uz'")
    figure, panels = plt.subplots(9, 5, figsize=(16, 24), constrained_layout=True)
    for component in range(3):
        limit = max(float(np.max(np.abs(truth[component, :, :, indices]))), 1.0e-8)
        for column, index in enumerate(indices):
            fields = (
                truth[component, :, :, index],
                estimate[component, :, :, index],
                estimate[component, :, :, index] - truth[component, :, :, index],
            )
            for kind, value in enumerate(fields):
                row = 3 * component + kind
                image = panels[row, column].imshow(
                    value.T, origin="lower", extent=(0.0, 1.5, 0.0, 1.5),
                    cmap="RdBu_r", vmin=-limit, vmax=limit,
                )
                if row == 0:
                    panels[row, column].set_title(f"z={axes_grid[2][index]:.2f} m")
                if column == 0:
                    panels[row, column].set_ylabel(
                        f"{labels[component]} " + ("truth", "recon", "error")[kind]
                    )
                panels[row, column].set_xticks((0.0, 0.75, 1.5))
                panels[row, column].set_yticks((0.0, 0.75, 1.5))
            figure.colorbar(image, ax=panels[3 * component:3 * component + 3, column],
                            label="m/s", shrink=0.55)
    figure.suptitle("Central five-slice Travel-Time reconstruction")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_dense_experiment(
    flow_config: MultiSliceConfig,
    solve_config: DenseSolveConfig,
) -> dict[str, object]:
    started = time.perf_counter()
    output = Path(flow_config.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot, baseline, truth_metadata = _load_interpolators(flow_config)
    truth, axes = truth_on_grid(snapshot, baseline, flow_config)
    coordinates, walls, layers = acquisition(flow_config)
    pairs, strata, pair_report = stratified_pairs(
        walls, layers, solve_config.pair_budget,
        flow_config.seed + solve_config.pair_seed_offset,
    )
    basis: CurlBasis3D = build_basis(flow_config)
    operator_started = time.perf_counter()
    cache_report: dict[str, object] = {
        "enabled": solve_config.operator_cache is not None,
        "hit": False,
    }
    if solve_config.operator_cache is None:
        matrix, data, geometry = build_operator_and_data(
            snapshot, baseline, coordinates, pairs, basis, flow_config,
        )
    else:
        cache_path = Path(solve_config.operator_cache).resolve()
        fingerprint = operator_fingerprint(
            coordinates, pairs, basis, flow_config,
        )
        cache_report.update({
            "path": str(cache_path),
            "fingerprint": fingerprint,
        })
        try:
            matrix, cache_metadata = load_operator_cache(cache_path, fingerprint)
        except FileNotFoundError:
            matrix, data, geometry = build_operator_and_data(
                snapshot, baseline, coordinates, pairs, basis, flow_config,
            )
            save_operator_cache(
                cache_path, matrix, fingerprint, geometry,
            )
            cache_report["created"] = True
        else:
            if tuple(matrix.shape) != (len(pairs), basis.mode_count):
                raise ValueError(
                    f"operator cache has shape {matrix.shape}, expected "
                    f"{(len(pairs), basis.mode_count)}"
                )
            data = build_target_data(
                snapshot, baseline, coordinates, pairs, flow_config,
            )
            geometry = cache_metadata["geometry"]
            cache_report["hit"] = True
    operator_seconds = time.perf_counter() - operator_started
    solve_started = time.perf_counter()
    clean_coefficients, noisy_coefficients, solver_report = scalable_ridge_solve(
        matrix, data, strata, flow_config, solve_config,
    )
    solve_seconds = time.perf_counter() - solve_started
    clean_field = evaluate_coefficients(clean_coefficients, basis, axes, flow_config)
    noisy_fields = [
        evaluate_coefficients(coefficients, basis, axes, flow_config)
        for coefficients in noisy_coefficients
    ]
    mean_field = np.mean(noisy_fields, axis=0)
    primary_mask = (
        (axes[2] >= float(flow_config.primary_z_min_m) - 1.0e-9)
        & (axes[2] <= float(flow_config.primary_z_max_m) + 1.0e-9)
    )
    clean_metrics = field_metrics(
        clean_field[:, :, :, primary_mask], truth[:, :, :, primary_mask],
    )
    noisy_metrics = _aggregate([
        field_metrics(field[:, :, :, primary_mask], truth[:, :, :, primary_mask])
        for field in noisy_fields
    ])
    slice_rows = layer_metrics(
        mean_field, truth, axes[2], tuple(np.linspace(3.5, 4.5, 5)),
    )
    roughness = roughness_metrics(mean_field, truth, axes, primary_mask)
    gates = {
        "joint_l2_le_30pct": bool(noisy_metrics["joint_relative_l2_mean"] <= 0.30),
        "each_component_l2_le_45pct": bool(
            max(noisy_metrics["component_relative_l2_mean"]) <= 0.45
        ),
        "each_component_correlation_ge_080": bool(
            min(noisy_metrics["component_correlation_mean"]) >= 0.80
        ),
        "continuum_divergence_exact": True,
    }
    report = {
        "contract": {
            "flow": asdict(flow_config),
            "solver": asdict(solve_config),
            "full_wave_used": False,
            "truth_used_for_ridge_selection": False,
            "frequency_selection_frozen_before_this_snapshot": True,
            "truth_definition": truth_metadata["truth_definition"],
        },
        "truth_provenance": truth_metadata,
        "truth_metrics": {
            "component_rms_m_s": np.sqrt(np.mean(truth**2, axis=(1, 2, 3))).tolist(),
            "component_max_abs_m_s": np.max(np.abs(truth), axis=(1, 2, 3)).tolist(),
        },
        "acquisition": {
            "sensor_count": int(len(coordinates)),
            "layer_count": int(len(flow_config.sensor_z_m)),
            "sensors_per_layer": int(flow_config.sensors_per_layer),
            **pair_report,
            **geometry,
        },
        "basis": {
            "center_shape": list(flow_config.center_shape),
            "center_count": int(len(basis.centers)),
            "mode_count": int(basis.mode_count),
            "sigma_xy_m": float(basis.sigma_xy_m),
            "sigma_z_m": float(basis.sigma_z_m),
        },
        "operator": {
            "shape": list(matrix.shape),
            "storage_bytes": int(matrix.nbytes),
            "row_to_parameter_ratio": float(matrix.shape[0] / matrix.shape[1]),
            "build_seconds": operator_seconds,
            "cache": cache_report,
        },
        "solver": solver_report,
        "clean_metrics": clean_metrics,
        "noisy_metrics": noisy_metrics,
        "mean_field_sampled_fd_relative_divergence": divergence_metric(mean_field, axes),
        "roughness": roughness,
        "central_five_slice_metrics": slice_rows,
        "gates": gates,
        "all_original_gates_passed": bool(all(gates.values())),
        "timing": {
            "operator_seconds": operator_seconds,
            "solve_seconds": solve_seconds,
            "total_seconds": time.perf_counter() - started,
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        output / "reconstruction.npz",
        truth=truth, reconstruction=mean_field, clean_reconstruction=clean_field,
        x=axes[0], y=axes[1], z=axes[2], coordinates=coordinates, pairs=pairs,
        centers=basis.centers,
    )
    _plot_slices(truth, mean_field, axes, flow_config,
                 output / "central_slice_reconstruction.png")
    _plot_five_slices(truth, mean_field, axes, output / "five_slice_reconstruction.png")
    _plot_layer_metrics(slice_rows, output / "five_slice_metrics.png")
    _plot_acquisition(coordinates, layers, output / "acquisition.png")
    summary = f"""# Dense-ring long-domain Travel-Time result

- Sensors/layers: {len(coordinates)} / {len(flow_config.sensor_z_m)}.
- Selected/full cross-wall pairs: {len(pairs)} / {pair_report['all_cross_wall_pair_count']}.
- Curl modes: {basis.mode_count}; operator shape: {matrix.shape}.
- Mean noisy central-ROI joint L2: {100*noisy_metrics['joint_relative_l2_mean']:.2f}%.
- Component L2 Ux/Uy/Uz: {[round(100*x, 2) for x in noisy_metrics['component_relative_l2_mean']]}%.
- Component correlation Ux/Uy/Uz: {[round(x, 3) for x in noisy_metrics['component_correlation_mean']]}.
- Clean delay residual: {100*solver_report['clean_delay_residual_relative']:.2f}%.
- All original gates passed: {report['all_original_gates_passed']}.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    return report
