from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from multislice_xyz_tt_study.multislice_tt import MultiSliceConfig

from .dense_ring_tt import DenseSolveConfig, run_dense_experiment


def profile_config(
    profile: str,
    snapshot: str,
    baseline: str,
    output: str,
    pair_budget: int | None,
    device: str,
) -> tuple[MultiSliceConfig, DenseSolveConfig]:
    common = dict(
        snapshot=snapshot,
        baseline=baseline,
        output=output,
        flow_length_z_m=8.0,
        slab_z_min_m=3.0,
        slab_z_max_m=5.0,
        primary_z_min_m=3.5,
        primary_z_max_m=4.5,
        frequency_hz=100_000.0,
        run_pair_ablations=False,
        axial_sine_order=0,
        axial_fisher_weight_factor=1.0,
    )
    if profile == "smoke":
        flow = MultiSliceConfig(
            **common,
            sensor_z_m=tuple(np.linspace(3.0, 5.0, 5)),
            sensors_per_layer=16,
            center_shape=(5, 5, 5),
            sigma_xy_m=0.20,
            sigma_z_m=0.28,
            evaluation_shape=(25, 25, 11),
            norm_shape=(13, 13, 11),
            quadrature_order=8,
            noise_repeats=2,
            operator_dtype="float32",
            operator_batch_size=8,
        )
        solve = DenseSolveConfig(
            pair_budget=pair_budget or 4_000,
            ridge_relatives=(1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1),
            power_iterations=8,
            cg_max_iterations=300,
            cg_relative_tolerance=1.0e-4,
            device=device,
        )
    elif profile == "density_matched":
        flow = MultiSliceConfig(
            **common,
            sensor_z_m=tuple(np.linspace(3.0, 5.0, 15)),
            sensors_per_layer=48,
            center_shape=(13, 13, 17),
            sigma_xy_m=0.11,
            sigma_z_m=0.13,
            evaluation_shape=(51, 51, 25),
            norm_shape=(25, 25, 25),
            quadrature_order=12,
            noise_repeats=4,
            operator_dtype="float32",
            operator_batch_size=8,
        )
        solve = DenseSolveConfig(pair_budget=pair_budget or 36_000, device=device)
    elif profile == "fine":
        flow = MultiSliceConfig(
            **common,
            sensor_z_m=tuple(np.linspace(3.0, 5.0, 15)),
            sensors_per_layer=48,
            center_shape=(15, 15, 19),
            sigma_xy_m=0.095,
            sigma_z_m=0.11,
            evaluation_shape=(51, 51, 25),
            norm_shape=(25, 25, 25),
            quadrature_order=12,
            noise_repeats=4,
            operator_dtype="float32",
            operator_batch_size=6,
        )
        solve = DenseSolveConfig(pair_budget=pair_budget or 36_000, device=device)
    else:
        raise ValueError(f"unknown profile {profile!r}")
    return flow, solve


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scalable dense-ring four-wall Travel-Time inversion",
    )
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--profile", choices=("smoke", "density_matched", "fine"),
        default="smoke",
    )
    parser.add_argument("--pair-budget", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--cg-max-iterations", type=int,
        help="Override the frozen profile only for a documented solver-convergence audit.",
    )
    parser.add_argument(
        "--operator-cache",
        help=(
            "Optional .npy cache for the snapshot-independent analytic operator; "
            "a geometry/basis fingerprint is verified before reuse."
        ),
    )
    args = parser.parse_args()
    flow, solve = profile_config(
        args.profile, args.snapshot, args.baseline, args.output,
        args.pair_budget, args.device,
    )
    if args.cg_max_iterations is not None:
        if args.cg_max_iterations <= 0:
            parser.error("--cg-max-iterations must be positive")
        solve = replace(solve, cg_max_iterations=args.cg_max_iterations)
    if args.operator_cache is not None:
        solve = replace(solve, operator_cache=args.operator_cache)
    report = run_dense_experiment(flow, solve)
    print(json.dumps({
        "report": str(Path(args.output).resolve() / "report.json"),
        "profile": args.profile,
        "metrics": report["noisy_metrics"],
        "gates": report["gates"],
        "timing": report["timing"],
    }, indent=2))


if __name__ == "__main__":
    main()
