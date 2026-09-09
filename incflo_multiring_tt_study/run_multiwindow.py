from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from multislice_xyz_tt_study.multislice_tt import (
    CurlBasis3D,
    MultiSliceConfig,
    _load_interpolators,
    acquisition,
    build_basis,
    build_operator_and_data,
    build_target_data,
    evaluate_policy,
    factor_gram,
    select_pairs,
    truth_on_grid,
)

from .multiring import local_ring_z
from .run_window import BASIS


def _config(args: argparse.Namespace, center_z: float, output: Path) -> MultiSliceConfig:
    shape, sigma_xy, sigma_z = BASIS[args.basis]
    axial_scale = args.ring_spacing / 0.15
    return MultiSliceConfig(
        snapshot=args.snapshot,
        baseline=args.baseline,
        output=str(output),
        flow_length_z_m=8.0,
        slab_z_min_m=center_z - 0.40 * axial_scale,
        slab_z_max_m=center_z + 0.40 * axial_scale,
        sensor_z_m=local_ring_z(center_z, args.ring_spacing),
        primary_z_min_m=center_z,
        primary_z_max_m=center_z,
        sensors_per_layer=args.sensors_per_ring,
        frequency_hz=100_000.0,
        center_shape=shape,
        sigma_xy_m=sigma_xy,
        sigma_z_m=sigma_z * axial_scale,
        evaluation_shape=(49, 49, 9),
        norm_shape=(25, 25, 15),
        quadrature_order=20,
        noise_repeats=args.noise_repeats,
        run_pair_ablations=False,
    )


def _plot_components(
    centers: list[float], truths: list[np.ndarray], estimates: list[np.ndarray], output: Path,
) -> None:
    for component, label in enumerate(("Ux'", "Uy'", "Uz'")):
        figure, axes = plt.subplots(
            len(centers), 3, figsize=(10.5, 3.2 * len(centers)), constrained_layout=True,
            squeeze=False,
        )
        limit = max(
            float(np.quantile(np.abs(truth[component, :, :, 4]), 0.998))
            for truth in truths
        )
        limit = max(limit, 1e-8)
        for row, (center, truth, estimate) in enumerate(zip(centers, truths, estimates)):
            panels = (
                ("Truth", truth[component, :, :, 4]),
                ("Reconstruction", estimate[component, :, :, 4]),
                ("Error", estimate[component, :, :, 4] - truth[component, :, :, 4]),
            )
            for column, (title, values) in enumerate(panels):
                image = axes[row, column].imshow(
                    values.T, origin="lower", extent=(0, 1.5, 0, 1.5),
                    cmap="RdBu_r", vmin=-limit, vmax=limit,
                )
                axes[row, column].set_title(f"z={center:.2f} m: {title}")
                axes[row, column].set_xlabel("x (m)")
                axes[row, column].set_ylabel("y (m)")
                figure.colorbar(image, ax=axes[row, column], label="m/s", pad=0.01)
        figure.suptitle(f"incflo multi-ring 100 kHz TT: {label}")
        figure.savefig(output / f"{label[:2].lower()}_multiwindow_truth_reconstruction_error.png", dpi=180)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--center-z", type=float, nargs="+", default=(1.65, 3.45, 5.55))
    parser.add_argument("--basis", choices=sorted(BASIS), default="dense")
    parser.add_argument("--sensors-per-ring", type=int, default=48)
    parser.add_argument("--global-ring-count", type=int, default=41)
    parser.add_argument("--ring-spacing", type=float, default=0.15)
    parser.add_argument("--global-z-min", type=float, default=1.05)
    parser.add_argument("--global-z-max", type=float, default=7.05)
    parser.add_argument("--baseline-status", required=True,
                        choices=("provisional_cross_solver", "matched_incflo"))
    parser.add_argument("--noise-repeats", type=int, default=4)
    args = parser.parse_args()
    started = time.perf_counter()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    centers = [float(value) for value in args.center_z]
    canonical = _config(args, centers[0], output)
    snapshot, baseline, provenance = _load_interpolators(canonical)
    coordinates, walls, layers = acquisition(canonical)
    pairs = select_pairs(walls, layers, "full")
    canonical_basis = build_basis(canonical)
    matrix, first_target, geometry = build_operator_and_data(
        snapshot, baseline, coordinates, pairs, canonical_basis, canonical,
    )
    factorization = factor_gram(matrix)
    records = []
    truths: list[np.ndarray] = []
    estimates: list[np.ndarray] = []
    for index, center in enumerate(centers):
        config = _config(args, center, output)
        truth, axes = truth_on_grid(snapshot, baseline, config)
        shifted_basis = CurlBasis3D(
            centers=canonical_basis.centers
            + np.asarray((0.0, 0.0, center - centers[0])),
            mode_norms=canonical_basis.mode_norms,
            sigma_xy_m=canonical_basis.sigma_xy_m,
            sigma_z_m=canonical_basis.sigma_z_m,
            axial_sine_order=0,
        )
        if index == 0:
            target = first_target
        else:
            shifted_coordinates, shifted_walls, shifted_layers = acquisition(config)
            shifted_pairs = select_pairs(shifted_walls, shifted_layers, "full")
            if not np.array_equal(shifted_pairs, pairs):
                raise RuntimeError("local-window pair ordering changed under translation")
            target = build_target_data(
                snapshot, baseline, shifted_coordinates, shifted_pairs, config,
            )
        result, estimate = evaluate_policy(
            matrix, target, shifted_basis, truth, axes, config,
            seed_offset=int(round(center * 100_000)), factorization=factorization,
        )
        truths.append(truth)
        estimates.append(estimate)
        records.append({
            "center_z_m": center,
            "truth_component_rms_m_s": result["clean_metrics"]["truth_component_rms_m_s"],
            "clean_fit": result["clean_fit"],
            "clean_metrics": result["clean_metrics"],
            "noisy_metrics": result["noisy_metrics"],
            "operator": result["operator"],
            "sampled_fd_relative_divergence": result["mean_field_sampled_fd_relative_divergence"],
        })
    report = {
        "contract": {
            "frequency_hz": 100_000.0,
            "basis": args.basis,
            "center_shape": list(canonical.center_shape),
            "mode_count": canonical_basis.mode_count,
            "sensors_per_ring": args.sensors_per_ring,
            "ring_spacing_m": args.ring_spacing,
            "local_rings": 5,
            "local_sensor_count": len(coordinates),
            "local_ray_count": len(pairs),
            "global_ring_count": args.global_ring_count,
            "global_sensor_count": args.global_ring_count * args.sensors_per_ring,
            "global_z_min_m": args.global_z_min,
            "global_z_max_m": args.global_z_max,
            "four_side_walls_only": True,
            "baseline_status": args.baseline_status,
            "formal_wake_error_claim_allowed": args.baseline_status == "matched_incflo",
            "shared_operator_and_factorization": True,
            "truth_used_for_hyperparameter_selection": False,
        },
        "truth_provenance": provenance,
        "geometry": geometry,
        "windows": records,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        output / "reconstruction.npz", centers_z_m=np.asarray(centers),
        truth=np.asarray(truths), reconstruction=np.asarray(estimates),
    )
    np.savez_compressed(
        output / "operator_spectrum.npz", eigenvalues=factorization[1],
        centers=canonical_basis.centers, mode_norms=canonical_basis.mode_norms,
    )
    _plot_components(centers, truths, estimates, output)
    print(json.dumps({
        "report": str(output / "report.json"),
        "elapsed_seconds": report["elapsed_seconds"],
        "windows": [
            {
                "z_m": row["center_z_m"],
                "truth_rms": row["truth_component_rms_m_s"],
                "joint_l2": row["noisy_metrics"]["joint_relative_l2_mean"],
                "component_l2": row["noisy_metrics"]["component_relative_l2_mean"],
                "correlation": row["noisy_metrics"]["component_correlation_mean"],
            }
            for row in records
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
