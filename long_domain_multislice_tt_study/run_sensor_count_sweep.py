from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from .run_dense_tt import profile_config
from .sensor_count_sweep import run_sensor_count_sweep


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed-ring four-level sensors-per-ring Travel-Time sweep",
    )
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cg-max-iterations", type=int, default=1600)
    parser.add_argument("--operator-cache")
    args = parser.parse_args()
    flow, solve = profile_config(
        "density_matched", args.snapshot, args.baseline, args.output, None, args.device,
    )
    solve = replace(
        solve, cg_max_iterations=args.cg_max_iterations,
        operator_cache=args.operator_cache,
    )
    report = run_sensor_count_sweep(flow, solve)
    print(json.dumps({
        "report": str(Path(args.output).resolve() / "report.json"),
        "best_level": report["diagnostics"]["best_level_by_mean_noisy_joint_l2"],
        "best_joint_l2": report["diagnostics"]["best_mean_noisy_joint_l2"],
        "total_seconds": report["timing"]["total_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
