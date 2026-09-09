#!/usr/bin/env python3
"""Parse incflo timing and failure diagnostics into a stable JSON record."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

from .input_parser import ints, parse_inputs, scalar


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"


def _last_float(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return float(matches[-1]) if matches else None


def _parse_elapsed(value: str) -> float:
    parts = [float(item) for item in value.strip().split(":")]
    if len(parts) == 2:
        return 60.0 * parts[0] + parts[1]
    if len(parts) == 3:
        return 3600.0 * parts[0] + 60.0 * parts[1] + parts[2]
    raise ValueError(f"Unsupported elapsed time: {value}")


def parse_time_file(path: Path | None) -> dict[str, float | None]:
    result: dict[str, float | None] = {
        "wall_seconds": None,
        "user_seconds": None,
        "system_seconds": None,
        "max_rss_kib": None,
    }
    if path is None or not path.exists():
        return result
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"^\s*Elapsed \(wall clock\) time.*\):\s*(\S+)\s*$",
        text,
        flags=re.MULTILINE,
    )
    if match:
        result["wall_seconds"] = _parse_elapsed(match.group(1))
    for label, key in (
        ("User time", "user_seconds"),
        ("System time", "system_seconds"),
        ("Maximum resident set size", "max_rss_kib"),
    ):
        match = re.search(rf"{re.escape(label)}.*?:\s*({FLOAT})", text)
        if match:
            result[key] = float(match.group(1))
    return result


def parse_gpu_csv(path: Path | None, gpu_index: int | None) -> dict[str, float | None]:
    memory: list[float] = []
    utilization: list[float] = []
    if path is None or not path.exists():
        return {"peak_memory_mib": None, "mean_utilization_percent": None}
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.reader(handle):
            if len(row) < 4:
                continue
            try:
                index = int(row[1].strip())
                if gpu_index is not None and index != gpu_index:
                    continue
                memory.append(float(row[2].strip()))
                utilization.append(float(row[3].strip()))
            except ValueError:
                continue
    return {
        "peak_memory_mib": max(memory) if memory else None,
        "mean_utilization_percent": (
            sum(utilization) / len(utilization) if utilization else None
        ),
    }


def parse_log(
    log_path: Path,
    input_path: Path,
    time_path: Path | None,
    gpu_csv_path: Path | None,
    gpu_index: int | None,
    reference_wall_seconds: float,
) -> dict[str, object]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    inputs = parse_inputs(input_path)
    expected_steps = int(scalar(inputs, "max_step"))
    step_numbers = [int(item) for item in re.findall(rf"^Step\s+(\d+)\s+starts", text, re.MULTILINE)]
    completed_steps = max(step_numbers) if step_numbers else text.count("NEW TIME STEP")
    init_seconds = _last_float(rf"Time spent in InitData\(\):\s+({FLOAT})", text)
    evolve_seconds = _last_float(rf"Time spent in Evolve\(\):\s+({FLOAT})", text)
    dt_values = [float(item) for item in re.findall(rf"with dt =\s*({FLOAT})", text)]
    final_times = [float(item) for item in re.findall(rf"to new time\s+({FLOAT})", text)]
    cfl_dt_limits = [
        float(item) for item in re.findall(rf"max dt by CFL\s*:\s*({FLOAT})", text)
    ]
    fixed_dt = float(scalar(inputs, "incflo.fixed_dt"))
    finite_dt = all(math.isfinite(item) for item in dt_values)
    failure_patterns = {
        "nan_or_inf": bool(re.search(r"\b(?:nan|inf)\b", text, re.IGNORECASE)),
        "amrex_abort": "AMReX::Abort" in text or "SIGABRT" in text,
        "solver_failure": bool(
            re.search(r"failed to converge|No convergence|MLMG failed", text, re.IGNORECASE)
        ),
        "cuda_error": bool(re.search(r"CUDA error|cudaError", text, re.IGNORECASE)),
    }
    failure_patterns["nan_or_inf"] = failure_patterns["nan_or_inf"] and not bool(
        re.search(r"no (?:nan|inf)", text, re.IGNORECASE)
    )
    successful = (
        completed_steps == expected_steps
        and evolve_seconds is not None
        and finite_dt
        and not any(failure_patterns.values())
    )
    evolve_seconds_per_step = (
        evolve_seconds / completed_steps
        if evolve_seconds is not None and completed_steps > 0 else None
    )
    reference_seconds_per_step = reference_wall_seconds / 100.0
    raw_speedup = (
        reference_seconds_per_step / evolve_seconds_per_step
        if evolve_seconds_per_step else None
    )
    grid = ints(inputs, "amr.n_cell")
    cell_count = math.prod(grid)
    reference_cell_count = 128 * 128 * 684
    cell_count_ratio = cell_count / reference_cell_count
    normalized_speedup = (
        raw_speedup * cell_count_ratio if raw_speedup is not None else None
    )
    primary_performance_case = (
        expected_steps == 100
        and scalar(inputs, "incflo.geometry") == "sphere"
        and int(scalar(inputs, "amr.plot_int")) == -1
    )
    return {
        "input": str(input_path.resolve()),
        "log": str(log_path.resolve()),
        "expected_steps": expected_steps,
        "completed_steps": completed_steps,
        "fixed_dt_s": fixed_dt,
        "observed_dt_min_s": min(dt_values) if dt_values else None,
        "observed_dt_max_s": max(dt_values) if dt_values else None,
        "final_simulated_time_s": final_times[-1] if final_times else None,
        "cfl_warning": bool(cfl_dt_limits),
        "minimum_reported_cfl_dt_limit_s": min(cfl_dt_limits) if cfl_dt_limits else None,
        "maximum_fixed_dt_over_cfl_limit": (
            fixed_dt / min(cfl_dt_limits) if cfl_dt_limits and fixed_dt > 0 else None
        ),
        "initialization_seconds": init_seconds,
        "evolve_seconds": evolve_seconds,
        "evolve_seconds_per_step": evolve_seconds_per_step,
        "grid": grid,
        "cell_count": cell_count,
        "reference_cell_count": reference_cell_count,
        "cell_count_ratio_vs_reference": cell_count_ratio,
        "pytorch_reference_wall_seconds_100_steps": reference_wall_seconds,
        "raw_speedup_vs_pytorch_reference": raw_speedup,
        "cell_normalized_speedup_vs_pytorch_reference": normalized_speedup,
        "primary_performance_case": primary_performance_case,
        "performance_gate_3x": bool(
            primary_performance_case
            and raw_speedup is not None
            and raw_speedup >= 3.0
        ),
        "failure_flags": failure_patterns,
        "successful": successful,
        "gnu_time": parse_time_file(time_path),
        "gpu": parse_gpu_csv(gpu_csv_path, gpu_index),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--time-file", type=Path)
    parser.add_argument("--gpu-csv", type=Path)
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--reference-wall-seconds", type=float, default=252.93)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = parse_log(
        args.log, args.input, args.time_file, args.gpu_csv, args.gpu_index,
        args.reference_wall_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["successful"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
