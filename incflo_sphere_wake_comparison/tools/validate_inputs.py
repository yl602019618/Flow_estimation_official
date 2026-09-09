#!/usr/bin/env python3
"""Fail-closed validation of all frozen incflo input decks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .input_parser import floats, ints, parse_inputs, scalar


PHYSICAL_EXPECTED = {
    "geometry.prob_lo": (0.0, 0.0, 0.0),
    "geometry.prob_hi": (1.5, 1.5, 8.0),
    "geometry.is_periodic": (0, 0, 0),
    "zlo.velocity": (0.0, 0.0, 0.5),
    "xlo.velocity": (0.0, 0.0, 0.0),
    "xhi.velocity": (0.0, 0.0, 0.0),
    "ylo.velocity": (0.0, 0.0, 0.0),
    "yhi.velocity": (0.0, 0.0, 0.0),
}

SCALAR_EXPECTED = {
    "incflo.fixed_dt": 0.003,
    "incflo.dt_change_max": 1.05,
    "incflo.ro_0": 998.0,
    "incflo.mu": 0.2994,
    "incflo.ic_u": 0.0,
    "incflo.ic_v": 0.0,
    "incflo.ic_w": 0.5,
    "mac_proj.mg_rtol": 4.0e-4,
    "nodal_proj.mg_rtol": 2.0e-4,
}

STRING_EXPECTED = {
    "xlo.type": "no_slip_wall",
    "xhi.type": "no_slip_wall",
    "ylo.type": "no_slip_wall",
    "yhi.type": "no_slip_wall",
    "zlo.type": "mass_inflow",
    "zhi.type": "pressure_outflow",
    "incflo.constant_density": "true",
    "incflo.fluid_model": "newtonian",
    "incflo.advection_type": "MOL",
    "incflo.advect_momentum": "false",
    "incflo.diffusion_type": "0",
    "incflo.use_tensor_solve": "false",
    "incflo.redistribution_type": "StateRedist",
    "mac_proj.bottom_solver": "cg",
}


def _close_tuple(actual: tuple[float, ...], expected: tuple[float, ...]) -> bool:
    return len(actual) == len(expected) and all(
        math.isclose(a, b, rel_tol=0.0, abs_tol=1.0e-12)
        for a, b in zip(actual, expected)
    )


def validate_one(path: Path) -> dict[str, object]:
    values = parse_inputs(path)
    errors: list[str] = []

    for key, expected in PHYSICAL_EXPECTED.items():
        actual = floats(values, key)
        if not _close_tuple(actual, expected):
            errors.append(f"{key}: expected {expected}, got {actual}")

    scalar_expected = dict(SCALAR_EXPECTED)
    if "double_strict" in path.name:
        scalar_expected.update({
            "mac_proj.mg_rtol": 1.0e-8,
            "nodal_proj.mg_rtol": 1.0e-8,
        })
        for key, expected in (
            ("mac_proj.mg_atol", 1.0e-12),
            ("nodal_proj.mg_atol", 1.0e-12),
        ):
            actual = float(scalar(values, key))
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-18):
                errors.append(f"{key}: expected {expected}, got {actual}")

    for key, expected in scalar_expected.items():
        actual = float(scalar(values, key))
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12):
            errors.append(f"{key}: expected {expected}, got {actual}")

    string_expected = dict(STRING_EXPECTED)
    if "double_strict" in path.name:
        string_expected["mac_proj.bottom_solver"] = "bicg"
    for key, expected in string_expected.items():
        actual = scalar(values, key)
        if actual != expected:
            errors.append(f"{key}: expected {expected}, got {actual}")

    grid = ints(values, "amr.n_cell")
    domain_lo = floats(values, "geometry.prob_lo")
    domain_hi = floats(values, "geometry.prob_hi")
    spacing = tuple((hi - lo) / n for lo, hi, n in zip(domain_lo, domain_hi, grid))
    dt = float(scalar(values, "incflo.fixed_dt"))
    speed = floats(values, "zlo.velocity")[2]
    nominal_cfl = speed * dt / min(spacing)
    rho = float(scalar(values, "incflo.ro_0"))
    mu = float(scalar(values, "incflo.mu"))
    reynolds = rho * speed * 0.30 / mu

    geometry = scalar(values, "incflo.geometry")
    if geometry == "sphere":
        if not _close_tuple(floats(values, "sphere.center"), (0.75, 0.75, 0.75)):
            errors.append("sphere.center does not match the flow contract")
        if not math.isclose(float(scalar(values, "sphere.radius")), 0.15):
            errors.append("sphere.radius does not match the flow contract")
        if scalar(values, "sphere.internal_flow") != "false":
            errors.append("sphere must be a solid obstacle")
    elif geometry != "all_regular":
        errors.append(f"unexpected geometry: {geometry}")

    if not math.isclose(reynolds, 500.0, rel_tol=1.0e-12):
        errors.append(f"Re_D must be 500, got {reynolds}")
    if nominal_cfl >= 0.25:
        errors.append(f"nominal inlet CFL must be <0.25, got {nominal_cfl}")
    if not _close_tuple(spacing, (spacing[0],) * 3):
        errors.append(
            "incflo EB redistribution requires exactly uniform physical spacing; "
            f"got {spacing}"
        )
    dt_change_max = float(scalar(values, "incflo.dt_change_max"))
    if not 1.0 < dt_change_max < 1.1:
        errors.append(
            "float32-safe incflo.dt_change_max must be strictly between 1 and 1.1"
        )
    if int(scalar(values, "amr.max_level")) != 0:
        errors.append("uniform comparison requires amr.max_level=0")

    return {
        "path": str(path),
        "valid": not errors,
        "errors": errors,
        "grid": grid,
        "spacing_m": spacing,
        "nominal_inlet_cfl": nominal_cfl,
        "reynolds": reynolds,
        "geometry": geometry,
        "max_step": int(scalar(values, "max_step")),
        "plot_interval": int(scalar(values, "amr.plot_int")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths", nargs="*", type=Path,
        help="Input decks; defaults to every file under ../inputs",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    default_dir = Path(__file__).resolve().parents[1] / "inputs"
    paths = args.paths or sorted(default_dir.glob("inputs.*"))
    reports = [validate_one(path.resolve()) for path in paths]
    payload = {"valid": all(item["valid"] for item in reports), "cases": reports}
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if payload["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
