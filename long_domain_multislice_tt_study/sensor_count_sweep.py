from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
from itertools import combinations
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

from .dense_ring_tt import (
    DenseSolveConfig,
    load_operator_cache,
    operator_fingerprint,
    pair_strata,
    roughness_metrics,
    save_operator_cache,
    scalable_ridge_solve,
)


@dataclass(frozen=True)
class SensorLevel:
    name: str
    sensors_per_wall_per_ring: int

    @property
    def sensors_per_ring(self) -> int:
        return 4 * self.sensors_per_wall_per_ring


DEFAULT_LEVELS = (
    SensorLevel("L1", 3),
    SensorLevel("L2", 6),
    SensorLevel("L3", 9),
    SensorLevel("L4", 12),
)


def farthest_point_order(values: np.ndarray) -> np.ndarray:
    """Return registered 3/6/9/full maximin prefixes with endpoint coverage."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("values must be a non-empty 1-D array")
    if len(np.unique(values)) != len(values):
        raise ValueError("farthest-point candidates must be distinct")
    coordinate_order = np.argsort(values, kind="stable")
    if len(values) == 1:
        return coordinate_order.astype(np.int64)
    selected = [int(coordinate_order[0]), int(coordinate_order[-1])]
    milestones = sorted({count for count in (3, 6, 9, len(values)) if count <= len(values)})
    for target_count in milestones:
        if target_count <= len(selected):
            continue
        remaining = [int(index) for index in coordinate_order if int(index) not in selected]
        addition_count = target_count - len(selected)
        best: tuple[tuple[object, ...], tuple[int, ...]] | None = None
        for candidate_tuple in combinations(remaining, addition_count):
            augmented = np.sort(values[np.asarray(selected + list(candidate_tuple))])
            gaps = np.diff(augmented)
            key = (
                float(np.max(gaps)), float(np.var(gaps)),
                tuple(float(values[index]) for index in candidate_tuple),
            )
            if best is None or key < best[0]:
                best = (key, candidate_tuple)
        assert best is not None
        additions = list(best[1])
        # Preserve a literal farthest-point ordering inside each registered batch.
        while additions:
            distances = np.asarray([
                np.min(np.abs(values[index] - values[np.asarray(selected)]))
                for index in additions
            ])
            maximum = float(np.max(distances))
            candidates = [
                index for index, distance in zip(additions, distances)
                if np.isclose(distance, maximum, rtol=0.0, atol=1.0e-14)
            ]
            chosen = min(candidates, key=lambda index: float(values[index]))
            selected.append(chosen)
            additions.remove(chosen)
    return np.asarray(selected, dtype=np.int64)


def nested_sensor_masks(
    coordinates: np.ndarray,
    walls: tuple[str, ...],
    layers: np.ndarray,
    levels: tuple[SensorLevel, ...] = DEFAULT_LEVELS,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Select coordinate-only nested subsets from a 12-per-wall acquisition."""
    unique_layers = np.unique(layers)
    wall_names = ("x0", "x1", "y0", "y1")
    maximum = max(level.sensors_per_wall_per_ring for level in levels)
    masks = {level.name: np.zeros(len(coordinates), dtype=bool) for level in levels}
    local_orders: dict[str, list[int]] = {}
    reference_order: np.ndarray | None = None
    for layer in unique_layers:
        for wall in wall_names:
            indices = np.flatnonzero((layers == layer) & (np.asarray(walls) == wall))
            if len(indices) != maximum:
                raise ValueError(
                    f"expected {maximum} candidates for {wall} layer {layer}, got {len(indices)}"
                )
            transverse_axis = 1 if wall.startswith("x") else 0
            order = farthest_point_order(coordinates[indices, transverse_axis])
            if reference_order is None:
                reference_order = order
            elif not np.array_equal(reference_order, order):
                raise ValueError("candidate transverse ordering differs across walls/layers")
            for level in levels:
                masks[level.name][indices[order[:level.sensors_per_wall_per_ring]]] = True
    assert reference_order is not None
    local_orders["zero_based_candidate_indices"] = reference_order.tolist()
    for left, right in zip(levels[:-1], levels[1:]):
        if np.any(masks[left.name] & ~masks[right.name]):
            raise AssertionError(f"{left.name} is not nested in {right.name}")
    return masks, {
        "definition": "coordinate-only deterministic 1-D farthest-point prefix",
        "local_order": local_orders,
        "layer_count": int(len(unique_layers)),
    }


def level_pair_rows(
    full_pairs: np.ndarray,
    sensor_masks: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        name: np.flatnonzero(mask[full_pairs[:, 0]] & mask[full_pairs[:, 1]])
        for name, mask in sensor_masks.items()
    }


def _plot_quality_curve(level_reports: list[dict[str, object]], path: Path) -> None:
    sensors = np.asarray([item["sensors_per_ring"] for item in level_reports])
    joint = 100 * np.asarray([
        item["noisy_metrics"]["joint_relative_l2_mean"] for item in level_reports
    ])
    component_l2 = 100 * np.asarray([
        item["noisy_metrics"]["component_relative_l2_mean"] for item in level_reports
    ])
    component_corr = np.asarray([
        item["noisy_metrics"]["component_correlation_mean"] for item in level_reports
    ])
    residual = 100 * np.asarray([
        item["solver"]["clean_delay_residual_relative"] for item in level_reports
    ])
    stable_rank = np.asarray([
        item["solver"]["stable_rank_proxy"] for item in level_reports
    ])
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes[0, 0].plot(sensors, joint, "ko-", linewidth=2, label="Joint")
    for component, label in enumerate(("Ux'", "Uy'", "Uz'")):
        axes[0, 0].plot(sensors, component_l2[:, component], "o--", label=label)
        axes[0, 1].plot(sensors, component_corr[:, component], "o-", label=label)
    axes[0, 0].set_ylabel("Relative L2 (%)")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Error (log scale; L1 is unstable)")
    axes[0, 1].set_ylabel("Correlation")
    axes[1, 0].plot(sensors, residual, "o-")
    axes[1, 0].set_ylabel("Clean TT residual (%)")
    axes[1, 1].plot(sensors, stable_rank, "o-")
    axes[1, 1].set_ylabel("Stable-rank proxy")
    for axis in axes.flat:
        axis.set_xlabel("Sensors per ring (15 rings fixed)")
        axis.set_xticks(sensors)
        axis.grid(alpha=0.3)
    axes[0, 0].legend()
    axes[0, 1].legend()
    figure.suptitle("Sensor-count versus reconstruction quality (100 kHz, four walls)")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_central_comparison(
    truth: np.ndarray,
    fields: list[np.ndarray],
    labels: list[str],
    axes_grid: tuple[np.ndarray, ...],
    path: Path,
) -> None:
    iz = int(np.argmin(np.abs(axes_grid[2] - 4.0)))
    figure, panels = plt.subplots(3, 1 + len(fields), figsize=(18, 10), constrained_layout=True)
    for component, component_name in enumerate(("Ux'", "Uy'", "Uz'")):
        limit = max(float(np.max(np.abs(truth[component, :, :, iz]))), 1.0e-8)
        values = [truth[component, :, :, iz]] + [field[component, :, :, iz] for field in fields]
        for column, (label, value) in enumerate(zip(["Truth"] + labels, values)):
            image = panels[component, column].imshow(
                value.T, origin="lower", extent=(0, 1.5, 0, 1.5),
                cmap="RdBu_r", vmin=-limit, vmax=limit,
            )
            panels[component, column].set_title(f"{label}: {component_name}")
            panels[component, column].set_xlabel("x (m)")
            if column == 0:
                panels[component, column].set_ylabel("y (m)")
        figure.colorbar(image, ax=panels[component, :], shrink=0.72, label="m/s")
    figure.suptitle(f"Central slice reconstruction at z={axes_grid[2][iz]:.2f} m")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_central_errors(
    truth: np.ndarray,
    fields: list[np.ndarray],
    labels: list[str],
    axes_grid: tuple[np.ndarray, ...],
    path: Path,
) -> None:
    iz = int(np.argmin(np.abs(axes_grid[2] - 4.0)))
    figure, panels = plt.subplots(3, len(fields), figsize=(15, 10), constrained_layout=True)
    for component, component_name in enumerate(("Ux'", "Uy'", "Uz'")):
        errors = [field[component, :, :, iz] - truth[component, :, :, iz] for field in fields]
        limit = max(max(float(np.max(np.abs(value))) for value in errors), 1.0e-8)
        for column, (label, value) in enumerate(zip(labels, errors)):
            image = panels[component, column].imshow(
                value.T, origin="lower", extent=(0, 1.5, 0, 1.5),
                cmap="RdBu_r", vmin=-limit, vmax=limit,
            )
            panels[component, column].set_title(f"{label} error: {component_name}")
            panels[component, column].set_xlabel("x (m)")
            if column == 0:
                panels[component, column].set_ylabel("y (m)")
        figure.colorbar(image, ax=panels[component, :], shrink=0.72, label="m/s")
    figure.suptitle(f"Central slice reconstruction errors at z={axes_grid[2][iz]:.2f} m")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_slice_curves(level_reports: list[dict[str, object]], path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for level in level_reports:
        rows = level["central_five_slice_metrics"]
        z = [row["evaluated_z_m"] for row in rows]
        for component in range(3):
            axes[component].plot(
                z, [100 * row["component_relative_l2"][component] for row in rows],
                "o-", label=level["name"],
            )
    for component, label in enumerate(("Ux'", "Uy'", "Uz'")):
        axes[component].set_title(label)
        axes[component].set_xlabel("z (m)")
        axes[component].set_ylabel("Relative L2 (%)")
        axes[component].grid(alpha=0.3)
        axes[component].legend()
    figure.suptitle("Five-slice component error versus sensor count")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_sensor_count_sweep(
    flow_config: MultiSliceConfig,
    solve_config: DenseSolveConfig,
    levels: tuple[SensorLevel, ...] = DEFAULT_LEVELS,
) -> dict[str, object]:
    started = time.perf_counter()
    output = Path(flow_config.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if flow_config.sensors_per_layer != max(level.sensors_per_ring for level in levels):
        raise ValueError("flow_config must describe the maximum-density acquisition")

    snapshot, baseline, truth_metadata = _load_interpolators(flow_config)
    truth, axes = truth_on_grid(snapshot, baseline, flow_config)
    coordinates, walls, layers = acquisition(flow_config)
    sensor_masks, nesting_report = nested_sensor_masks(coordinates, walls, layers, levels)
    full_pairs = select_pairs(walls, layers, "full")
    pair_rows = level_pair_rows(full_pairs, sensor_masks)
    full_labels, stratum_names = pair_strata(walls, layers, full_pairs)
    basis: CurlBasis3D = build_basis(flow_config)

    fingerprint = operator_fingerprint(coordinates, full_pairs, basis, flow_config)
    cache_report: dict[str, object] = {"enabled": solve_config.operator_cache is not None}
    operator_started = time.perf_counter()
    if solve_config.operator_cache is None:
        matrix, target, geometry = build_operator_and_data(
            snapshot, baseline, coordinates, full_pairs, basis, flow_config,
        )
        cache_report["hit"] = False
    else:
        cache_path = Path(solve_config.operator_cache).resolve()
        cache_report.update({"path": str(cache_path), "fingerprint": fingerprint})
        try:
            matrix, metadata = load_operator_cache(cache_path, fingerprint)
        except FileNotFoundError:
            matrix, target, geometry = build_operator_and_data(
                snapshot, baseline, coordinates, full_pairs, basis, flow_config,
            )
            save_operator_cache(cache_path, matrix, fingerprint, geometry)
            cache_report.update({"hit": False, "created": True})
        else:
            target = build_target_data(snapshot, baseline, coordinates, full_pairs, flow_config)
            geometry = metadata["geometry"]
            cache_report["hit"] = True
    operator_seconds = time.perf_counter() - operator_started
    if tuple(matrix.shape) != (len(full_pairs), basis.mode_count):
        raise ValueError("full operator shape does not match acquisition and basis")

    full_noise = [
        np.random.default_rng(flow_config.seed + 1009 * repeat).standard_normal(len(full_pairs))
        for repeat in range(flow_config.noise_repeats)
    ]
    primary_mask = (
        (axes[2] >= float(flow_config.primary_z_min_m) - 1.0e-9)
        & (axes[2] <= float(flow_config.primary_z_max_m) + 1.0e-9)
    )
    requested_slices = tuple(np.linspace(3.5, 4.5, 5))
    level_reports: list[dict[str, object]] = []
    mean_fields: list[np.ndarray] = []

    for level in levels:
        level_started = time.perf_counter()
        rows = pair_rows[level.name]
        expected_sensors = len(flow_config.sensor_z_m) * level.sensors_per_ring
        expected_pairs = 6 * (
            len(flow_config.sensor_z_m) * level.sensors_per_wall_per_ring
        ) ** 2
        if int(sensor_masks[level.name].sum()) != expected_sensors:
            raise AssertionError(f"{level.name} sensor count mismatch")
        if len(rows) != expected_pairs:
            raise AssertionError(f"{level.name} pair count mismatch")
        level_matrix = matrix if len(rows) == len(full_pairs) else np.asarray(matrix[rows])
        level_target = target[rows]
        level_labels = full_labels[rows]
        level_noise = [values[rows] for values in full_noise]
        current_solve = replace(solve_config, pair_budget=len(rows), operator_cache=None)
        solve_started = time.perf_counter()
        clean_coefficients, noisy_coefficients, solver_report = scalable_ridge_solve(
            level_matrix, level_target, level_labels, flow_config, current_solve,
            noise_standard_normals=level_noise,
        )
        solve_seconds = time.perf_counter() - solve_started
        clean_field = evaluate_coefficients(clean_coefficients, basis, axes, flow_config)
        noisy_fields = [
            evaluate_coefficients(coefficients, basis, axes, flow_config)
            for coefficients in noisy_coefficients
        ]
        mean_field = np.mean(noisy_fields, axis=0)
        mean_fields.append(mean_field)
        clean_metrics = field_metrics(
            clean_field[:, :, :, primary_mask], truth[:, :, :, primary_mask],
        )
        noisy_metrics = _aggregate([
            field_metrics(field[:, :, :, primary_mask], truth[:, :, :, primary_mask])
            for field in noisy_fields
        ])
        slice_rows = layer_metrics(mean_field, truth, axes[2], requested_slices)
        stratum_counts = {
            stratum_names[int(label)]: int(np.sum(level_labels == label))
            for label in np.unique(level_labels)
        }
        level_report = {
            "name": level.name,
            "sensors_per_wall_per_ring": level.sensors_per_wall_per_ring,
            "sensors_per_ring": level.sensors_per_ring,
            "total_sensor_count": expected_sensors,
            "pair_count": int(len(rows)),
            "row_to_parameter_ratio": float(len(rows) / basis.mode_count),
            "stratum_counts": stratum_counts,
            "solver": solver_report,
            "clean_metrics": clean_metrics,
            "noisy_metrics": noisy_metrics,
            "mean_field_sampled_fd_relative_divergence": divergence_metric(mean_field, axes),
            "roughness": roughness_metrics(mean_field, truth, axes, primary_mask),
            "central_five_slice_metrics": slice_rows,
            "timing": {
                "solve_seconds": solve_seconds,
                "level_total_seconds": time.perf_counter() - level_started,
            },
        }
        level_reports.append(level_report)
        level_output = output / level.name.lower()
        level_output.mkdir(exist_ok=True)
        np.savez_compressed(
            level_output / "reconstruction.npz",
            truth=truth, reconstruction=mean_field, clean_reconstruction=clean_field,
            clean_coefficients=clean_coefficients,
            noisy_coefficients=np.asarray(noisy_coefficients),
            x=axes[0], y=axes[1], z=axes[2],
            coordinates=coordinates[sensor_masks[level.name]],
            pairs=full_pairs[rows], centers=basis.centers,
        )
        (level_output / "report.json").write_text(
            json.dumps(level_report, indent=2), encoding="utf-8",
        )
        del level_matrix, noisy_fields, clean_field
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    joint = [item["noisy_metrics"]["joint_relative_l2_mean"] for item in level_reports]
    marginal = [None] + [
        float((joint[index - 1] - joint[index]) / joint[index - 1])
        for index in range(1, len(joint))
    ]
    for item, gain in zip(level_reports, marginal):
        item["marginal_joint_error_reduction_from_previous"] = gain
    report = {
        "contract": {
            "flow": asdict(flow_config),
            "solver": asdict(solve_config),
            "levels": [asdict(level) for level in levels],
            "rings_fixed": len(flow_config.sensor_z_m),
            "frequency_hz_fixed": flow_config.frequency_hz,
            "all_available_cross_wall_pairs_used": True,
            "truth_used_for_ridge_selection": False,
            "common_ray_noise_realizations_identical": True,
            "full_wave_used": False,
        },
        "truth_provenance": truth_metadata,
        "nesting": nesting_report,
        "basis": {
            "center_shape": list(flow_config.center_shape),
            "center_count": int(len(basis.centers)),
            "mode_count": int(basis.mode_count),
            "sigma_xy_m": basis.sigma_xy_m,
            "sigma_z_m": basis.sigma_z_m,
        },
        "operator": {
            "shape": list(matrix.shape),
            "storage_bytes": int(matrix.nbytes),
            "build_seconds": operator_seconds,
            "geometry": geometry,
            "cache": cache_report,
        },
        "levels": level_reports,
        "diagnostics": {
            "joint_error_monotone_nonincreasing": bool(np.all(np.diff(joint) <= 0.0)),
            "best_level_by_mean_noisy_joint_l2": level_reports[int(np.argmin(joint))]["name"],
            "best_mean_noisy_joint_l2": float(np.min(joint)),
        },
        "timing": {
            "operator_seconds": operator_seconds,
            "total_seconds": time.perf_counter() - started,
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "level", "sensors_per_wall_per_ring", "sensors_per_ring", "total_sensors",
            "pair_count", "joint_l2_mean", "ux_l2_mean", "uy_l2_mean", "uz_l2_mean",
            "ux_corr_mean", "uy_corr_mean", "uz_corr_mean", "clean_delay_residual",
            "stable_rank_proxy", "selected_relative_ridge", "solve_seconds",
            "peak_cuda_memory_bytes", "marginal_joint_error_reduction",
        ))
        for item in level_reports:
            noisy = item["noisy_metrics"]
            solver = item["solver"]
            writer.writerow((
                item["name"], item["sensors_per_wall_per_ring"], item["sensors_per_ring"],
                item["total_sensor_count"], item["pair_count"],
                noisy["joint_relative_l2_mean"], *noisy["component_relative_l2_mean"],
                *noisy["component_correlation_mean"], solver["clean_delay_residual_relative"],
                solver["stable_rank_proxy"], solver["selected_relative_ridge"],
                item["timing"]["solve_seconds"], solver["peak_torch_cuda_memory_bytes"],
                item["marginal_joint_error_reduction_from_previous"],
            ))
    labels = [f"{level.name}\n{level.sensors_per_ring}/ring" for level in levels]
    _plot_quality_curve(level_reports, output / "quality_vs_sensor_count.png")
    _plot_central_comparison(
        truth, mean_fields, labels, axes, output / "central_truth_and_reconstructions.png",
    )
    _plot_central_errors(
        truth, mean_fields, labels, axes, output / "central_reconstruction_errors.png",
    )
    _plot_slice_curves(level_reports, output / "five_slice_error_curves.png")
    lines = [
        "# 固定 15 圈的阵元数量扫描结果", "",
        "本实验固定 100 kHz、四侧壁、15 圈和 8,619 个局部无散模态；主曲线使用每档全部跨壁观测对。", "",
        "| Level | 每圈阵元 | 总阵元 | Pair 数 | Joint L2 | Ux/Uy/Uz L2 | Ux/Uy/Uz corr |", 
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for item in level_reports:
        noisy = item["noisy_metrics"]
        lines.append(
            f"| {item['name']} | {item['sensors_per_ring']} | {item['total_sensor_count']} | "
            f"{item['pair_count']} | {100*noisy['joint_relative_l2_mean']:.2f}% | "
            f"{'/'.join(f'{100*x:.2f}%' for x in noisy['component_relative_l2_mean'])} | "
            f"{'/'.join(f'{x:.3f}' for x in noisy['component_correlation_mean'])} |"
        )
    lines.extend((
        "", f"最低 joint error: {report['diagnostics']['best_level_by_mean_noisy_joint_l2']} "
        f"({100*report['diagnostics']['best_mean_noisy_joint_l2']:.2f}%).",
        "", "边际 joint error 降幅（相对前一档）: "
        + ", ".join(
            f"{item['name']}={100*item['marginal_joint_error_reduction_from_previous']:.1f}%"
            for item in level_reports[1:]
        ) + ".",
        "", "所有 clean/noisy PCG 解均收敛: " + str(all(
            item["solver"]["clean_cg"]["converged"]
            and all(solve["converged"] for solve in item["solver"]["noisy_solves"])
            for item in level_reports
        )) + ".",
        "", "注意：该曲线同时包含角向采样位置和观测 pair 数增加的收益，不是固定 ray budget 的纯几何消融。",
    ))
    (output / "SUMMARY_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
