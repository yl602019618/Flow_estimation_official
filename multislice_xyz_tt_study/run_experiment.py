from __future__ import annotations

import argparse
import json

from .multislice_tt import MultiSliceConfig, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Thin-slab multi-slice sphere-wake TT")
    parser.add_argument("--snapshot", default=MultiSliceConfig.snapshot)
    parser.add_argument("--baseline", default=MultiSliceConfig.baseline)
    parser.add_argument("--noise-repeats", type=int, default=8)
    parser.add_argument("--output", default=MultiSliceConfig.output)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sensors-per-layer", type=int, default=MultiSliceConfig.sensors_per_layer)
    parser.add_argument("--frequency-khz", type=float, default=MultiSliceConfig.frequency_hz / 1000)
    parser.add_argument("--center-shape", type=int, nargs=3,
                        default=MultiSliceConfig.center_shape, metavar=("NX", "NY", "NZ"))
    parser.add_argument("--sigma-xy-m", type=float, default=MultiSliceConfig.sigma_xy_m)
    parser.add_argument("--sigma-z-m", type=float, default=MultiSliceConfig.sigma_z_m)
    parser.add_argument("--axial-sine-order", type=int,
                        default=MultiSliceConfig.axial_sine_order)
    parser.add_argument("--no-pair-ablations", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        config = MultiSliceConfig(
            snapshot=args.snapshot, baseline=args.baseline,
            output=args.output, sensor_z_m=(2.35, 2.50, 2.65),
            sensors_per_layer=16, frequency_hz=args.frequency_khz * 1000,
            center_shape=(5, 5, 3), evaluation_shape=(25, 25, 5),
            norm_shape=(13, 13, 7), quadrature_order=10, noise_repeats=2,
        )
    else:
        config = MultiSliceConfig(
            snapshot=args.snapshot, baseline=args.baseline,
            output=args.output, sensors_per_layer=args.sensors_per_layer,
            frequency_hz=args.frequency_khz * 1000,
            center_shape=tuple(args.center_shape), sigma_xy_m=args.sigma_xy_m,
            sigma_z_m=args.sigma_z_m, axial_sine_order=args.axial_sine_order,
            run_pair_ablations=not args.no_pair_ablations, noise_repeats=args.noise_repeats,
        )
    report = run_experiment(config)
    print(json.dumps({
        "report": f"{args.output}/report.json",
        "metrics": report["full_cross_layer_result"]["noisy_metrics"],
        "gates": report["gates"],
        "all_gates_passed": report["all_gates_passed"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
