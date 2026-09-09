from __future__ import annotations

import argparse
import json
from pathlib import Path

from multislice_xyz_tt_study.multislice_tt import MultiSliceConfig, run_experiment

from .multiring import local_ring_z


BASIS = {
    "reference": ((13, 13, 7), 0.11, 0.13),
    "dense": ((15, 15, 9), 0.095, 0.10),
}


def make_config(args: argparse.Namespace) -> MultiSliceConfig:
    shape, sigma_xy, sigma_z = BASIS[args.basis]
    half_width = 0.40
    evaluation_shape = (25, 25, 7) if args.smoke else (49, 49, 9)
    norm_shape = (13, 13, 7) if args.smoke else (25, 25, 15)
    if args.smoke:
        shape, sigma_xy, sigma_z = (5, 5, 3), 0.20, 0.20
    return MultiSliceConfig(
        snapshot=args.snapshot,
        baseline=args.baseline,
        output=args.output,
        flow_length_z_m=8.0,
        slab_z_min_m=args.center_z - half_width,
        slab_z_max_m=args.center_z + half_width,
        sensor_z_m=local_ring_z(args.center_z),
        primary_z_min_m=args.center_z,
        primary_z_max_m=args.center_z,
        sensors_per_layer=16 if args.smoke else args.sensors_per_ring,
        frequency_hz=100_000.0,
        center_shape=shape,
        sigma_xy_m=sigma_xy,
        sigma_z_m=sigma_z,
        evaluation_shape=evaluation_shape,
        norm_shape=norm_shape,
        quadrature_order=10 if args.smoke else 20,
        noise_repeats=2 if args.smoke else args.noise_repeats,
        run_pair_ablations=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--center-z", type=float, default=1.65)
    parser.add_argument("--basis", choices=sorted(BASIS), default="dense")
    parser.add_argument("--sensors-per-ring", type=int, default=48)
    parser.add_argument("--baseline-status", required=True,
                        choices=("provisional_cross_solver", "matched_incflo"))
    parser.add_argument("--noise-repeats", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = make_config(args)
    report = run_experiment(config)
    report_path = Path(args.output) / "report.json"
    stored = json.loads(report_path.read_text(encoding="utf-8"))
    stored["incflo_multiring_contract"] = {
        "baseline_status": args.baseline_status,
        "formal_wake_error_claim_allowed": args.baseline_status == "matched_incflo",
        "global_ring_spacing_m": 0.15,
        "global_sensors_per_ring": args.sensors_per_ring,
        "local_window_ring_count": 5,
        "target_slice_z_m": args.center_z,
        "basis_candidate": args.basis,
    }
    report_path.write_text(json.dumps(stored, indent=2) + "\n", encoding="utf-8")
    summary = {
        "report": str(report_path.resolve()),
        "baseline_status": args.baseline_status,
        "basis": args.basis,
        "mode_count": stored["basis"]["mode_count"],
        "ray_count": stored["acquisition"]["full_pair_count"],
        "clean_metrics": stored["full_cross_layer_result"]["clean_metrics"],
        "noisy_metrics": stored["full_cross_layer_result"]["noisy_metrics"],
        "elapsed_seconds": stored["elapsed_seconds"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
