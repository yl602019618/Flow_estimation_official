"""Command-line interface for the unit-ball travel-time benchmark."""
from __future__ import annotations

import argparse
import json

from .config import load_sphere_config
from .fullwave_pipeline import run_fullwave_benchmark
from .pipeline import run_benchmark
from .storage import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("geometry", "generate", "invert", "evaluate", "benchmark", "fullwave"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
        if name in ("fullwave", "benchmark"):
            command.add_argument("--reuse-waveforms", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "fullwave":
        result = run_fullwave_benchmark(load_sphere_config(args.config),
                                        reuse_waveforms=args.reuse_waveforms)
        print(json.dumps({"status": result["status"], "stage": args.command}, indent=2))
        return 0
    if args.command == "benchmark":
        config = load_sphere_config(args.config)
        prescribed = run_benchmark(config, through="evaluate")
        fullwave = run_fullwave_benchmark(config, reuse_waveforms=args.reuse_waveforms)
        accepted = bool(fullwave["inversion_gates"]["all_passed"])
        summary = {
            "status": "passed" if accepted else "failed",
            "official_observation": "97^3 split-PML full-wave pressure traces",
            "prescribed_ray_reference_status": prescribed["status"],
            "fullwave_acquisition_status": fullwave["status"],
            "inversion_gates": fullwave["inversion_gates"],
            "datasets": fullwave["datasets"],
        }
        write_json(config.output_root / "benchmark_summary.json", summary)
        print(json.dumps({"status": summary["status"], "stage": args.command}, indent=2))
        return 0 if accepted else 2
    through = "evaluate" if args.command in ("evaluate", "benchmark") else args.command
    result = run_benchmark(load_sphere_config(args.config), through=through)
    print(json.dumps({"status": result.get("status", "passed"), "stage": args.command}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
