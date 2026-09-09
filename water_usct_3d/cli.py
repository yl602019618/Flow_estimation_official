"""Command-line interface for the square-duct 50 kHz benchmark only."""

from __future__ import annotations

import argparse
import sys
import time

from ._workflow import (
    _append_run_log,
    _json,
    axial_fwi_command,
    benchmark_command,
    evaluate_command,
    geometry_command,
    invert_command,
)
from .config import load_config
from .data import generate_case
from .profile_validation import validate_frozen_axial_profile
from .verification import verify_acoustic, verify_flow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="water_usct_3d",
        description="Reproducible 3-D square-duct Water-USCT benchmark",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("verify-flow", "geometry", "evaluate", "benchmark"):
        item = sub.add_parser(name)
        item.add_argument("--config", required=True)
    acoustic = sub.add_parser("verify-acoustic")
    acoustic.add_argument("--config", required=True)
    acoustic.add_argument("--smoke", action="store_true")
    generate = sub.add_parser("generate")
    generate.add_argument("--config", required=True)
    generate.add_argument("--case", choices=("matched_clean", "fine_clean", "fine_noisy"),
                          required=True)
    invert = sub.add_parser("invert")
    invert.add_argument("--config", required=True)
    invert.add_argument("--stage", choices=("amplitude", "profile", "full"), required=True)
    invert.add_argument("--initial-beta", type=float, default=0.0)
    invert.add_argument("--production", action="store_true")
    validate = sub.add_parser("validate-profile")
    validate.add_argument("--config", required=True)
    validate.add_argument("--case", choices=("fine_clean", "fine_noisy"), required=True)
    validate.add_argument("--model", choices=("baseline", "fwi"), default="baseline")
    fwi = sub.add_parser("fwi-profile")
    fwi.add_argument("--config", required=True)
    fwi.add_argument("--production", action="store_true")
    fwi.add_argument("--iterations-per-band", type=int)
    fwi.add_argument("--initialization", choices=(
        "good_prior", "travel_time", "blurred_travel_time", "zero_flow"),
        default="good_prior")
    fwi.add_argument("--prepare-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    command_line = "python -m water_usct_3d.cli " + " ".join(
        argv if argv is not None else sys.argv[1:])
    started = time.perf_counter()
    config = load_config(args.config)
    if args.command == "verify-flow":
        result = verify_flow(config)
    elif args.command == "verify-acoustic":
        result = verify_acoustic(config, smoke=args.smoke)
    elif args.command == "geometry":
        result = geometry_command(config)
    elif args.command == "generate":
        result = generate_case(config, args.case)
    elif args.command == "invert":
        result = invert_command(config, args.stage, args.initial_beta, args.production)
    elif args.command == "validate-profile":
        result = validate_frozen_axial_profile(config, args.case, args.model)
    elif args.command == "fwi-profile":
        result = axial_fwi_command(
            config, args.production, args.iterations_per_band,
            args.initialization, args.prepare_only)
    elif args.command == "evaluate":
        result = evaluate_command(config)
    else:
        result = benchmark_command(config)
    result["runtime_seconds"] = time.perf_counter() - started
    status = "PASS" if result.get("passed", result.get("pass", True)) else "FAIL"
    _append_run_log(config.output_root, args.command, status, command_line, result)
    _json(result)


if __name__ == "__main__":
    main()
