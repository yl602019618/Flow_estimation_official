#!/usr/bin/env python3
"""Compute lightweight physical diagnostics from a single-level AMReX plotfile."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


BOX_RE = re.compile(
    r"\(\(([-\d]+),([-\d]+),([-\d]+)\) "
    r"\(([-\d]+),([-\d]+),([-\d]+)\) \([^)]+\)\)"
)


def _load_native(plotfile: Path):
    """Read a single-level, native float32 AMReX plotfile without yt."""
    import numpy as np

    header_lines = (plotfile / "Header").read_text(encoding="utf-8").splitlines()
    field_count = int(header_lines[1])
    names = header_lines[2:2 + field_count]
    dim_index = 2 + field_count
    ndim = int(header_lines[dim_index])
    if ndim != 3:
        raise ValueError(f"Only 3-D plotfiles are supported, got ndim={ndim}")
    left = np.fromstring(header_lines[dim_index + 3], sep=" ", dtype=float)
    right = np.fromstring(header_lines[dim_index + 4], sep=" ", dtype=float)

    cell_header = (plotfile / "Level_0" / "Cell_H").read_text(
        encoding="utf-8"
    )
    box_matches = list(BOX_RE.finditer(cell_header))
    disk_entries = re.findall(r"FabOnDisk:\s+(\S+)\s+(\d+)", cell_header)
    if not disk_entries or len(box_matches) < len(disk_entries):
        raise ValueError("Could not parse FAB boxes/offsets from Level_0/Cell_H")
    boxes = []
    for match in box_matches[:len(disk_entries)]:
        values = tuple(int(item) for item in match.groups())
        boxes.append((values[:3], values[3:]))
    dims = np.max(np.asarray([hi for _, hi in boxes], dtype=int), axis=0) + 1
    first_filename, first_offset = disk_entries[0]
    with (plotfile / "Level_0" / first_filename).open("rb") as handle:
        handle.seek(int(first_offset))
        first_fab_header = handle.readline().decode("ascii")
    precision_match = re.search(r"^FAB \(\(8, \((32|64)\s", first_fab_header)
    if precision_match is None:
        raise ValueError(f"Unsupported FAB real descriptor: {first_fab_header!r}")
    real_bits = int(precision_match.group(1))
    dtype = np.dtype("<f4" if real_bits == 32 else "<f8")
    data = np.empty((*dims, field_count), dtype=dtype)

    handles = {}
    try:
        for (lo, hi), (filename, offset_text) in zip(boxes, disk_entries):
            handle = handles.get(filename)
            if handle is None:
                handle = (plotfile / "Level_0" / filename).open("rb")
                handles[filename] = handle
            handle.seek(int(offset_text))
            fab_header = handle.readline().decode("ascii")
            fab_match = re.search(r"\)\)\s+(\d+)\s*$", fab_header)
            if fab_match is None or int(fab_match.group(1)) != field_count:
                raise ValueError(f"Unexpected FAB header: {fab_header!r}")
            shape = tuple(hi[d] - lo[d] + 1 for d in range(3))
            count = math.prod(shape) * field_count
            descriptor = re.search(r"^FAB \(\(8, \((32|64)\s", fab_header)
            if descriptor is None or int(descriptor.group(1)) != real_bits:
                raise ValueError(f"Mixed/unsupported FAB precision: {fab_header!r}")
            values = np.fromfile(handle, dtype=dtype, count=count)
            if values.size != count:
                raise EOFError(f"Short FAB read at {filename}:{offset_text}")
            block = values.reshape(field_count, shape[2], shape[1], shape[0])
            block = block.transpose(3, 2, 1, 0)
            data[
                lo[0]:hi[0] + 1,
                lo[1]:hi[1] + 1,
                lo[2]:hi[2] + 1,
                :,
            ] = block
    finally:
        for handle in handles.values():
            handle.close()
    return names, dims, left, right, data, real_bits


def analyze(plotfile: Path) -> dict[str, object]:
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("analyze_plotfile requires numpy") from exc

    names, dims, left, right, data, real_bits = _load_native(plotfile)
    name_to_index = {name: index for index, name in enumerate(names)}
    missing = {"velx", "vely", "velz", "vfrac"} - set(name_to_index)
    if missing:
        raise KeyError(f"Missing plotfile fields: {sorted(missing)}; available={names}")
    spacing = (right - left) / dims
    ux = data[..., name_to_index["velx"]]
    uy = data[..., name_to_index["vely"]]
    uz = data[..., name_to_index["velz"]]
    vfrac = data[..., name_to_index["vfrac"]]
    finite = bool(
        np.isfinite(ux).all() and np.isfinite(uy).all()
        and np.isfinite(uz).all() and np.isfinite(vfrac).all()
    )
    cell_volume = float(np.prod(spacing))
    domain_volume = float(np.prod(right - left))
    fluid_volume = float(vfrac.sum() * cell_volume)
    solid_volume = domain_volume - fluid_volume
    analytic_sphere_volume = 4.0 * math.pi * 0.15**3 / 3.0
    speed = np.sqrt(ux * ux + uy * uy + uz * uz)
    weights = np.clip(vfrac, 0.0, 1.0)

    # The sphere is away from z ends, so these are uncut layers. This is an
    # approximate cell-centred flux diagnostic, not an exact face flux.
    inlet_flow = float(np.sum(uz[:, :, 0] * weights[:, :, 0]) * spacing[0] * spacing[1])
    outlet_flow = float(np.sum(uz[:, :, -1] * weights[:, :, -1]) * spacing[0] * spacing[1])
    flow_mismatch = abs(outlet_flow - inlet_flow) / max(abs(inlet_flow), 1.0e-30)

    # Interior central-difference diagnostic, excluding walls, domain ends and
    # cut/covered cells plus their immediate neighbours.
    div = (
        (ux[2:, 1:-1, 1:-1] - ux[:-2, 1:-1, 1:-1]) / (2.0 * spacing[0])
        + (uy[1:-1, 2:, 1:-1] - uy[1:-1, :-2, 1:-1]) / (2.0 * spacing[1])
        + (uz[1:-1, 1:-1, 2:] - uz[1:-1, 1:-1, :-2]) / (2.0 * spacing[2])
    )
    interior = (
        weights[1:-1, 1:-1, 1:-1] > 0.999999
    )
    for shifted in (
        weights[2:, 1:-1, 1:-1], weights[:-2, 1:-1, 1:-1],
        weights[1:-1, 2:, 1:-1], weights[1:-1, :-2, 1:-1],
        weights[1:-1, 1:-1, 2:], weights[1:-1, 1:-1, :-2],
    ):
        interior &= shifted > 0.999999
    div_rms = float(np.sqrt(np.mean(div[interior] ** 2))) if interior.any() else None
    velocity_gradient_scale = 0.5 / min(spacing)
    relative_divergence = div_rms / velocity_gradient_scale if div_rms is not None else None
    max_speed = float(np.max(speed[weights > 0.0]))
    fixed_dt = 0.003
    max_cfl = max_speed * fixed_dt / min(spacing)

    return {
        "plotfile": str(plotfile.resolve()),
        "grid": dims.tolist(),
        "spacing_m": spacing.tolist(),
        "storage_precision_bits": real_bits,
        "finite": finite,
        "fluid_volume_m3": fluid_volume,
        "solid_volume_m3": solid_volume,
        "analytic_sphere_volume_m3": analytic_sphere_volume,
        "sphere_volume_relative_error": abs(solid_volume - analytic_sphere_volume) / analytic_sphere_volume,
        "inlet_flow_m3_s": inlet_flow,
        "outlet_flow_m3_s": outlet_flow,
        "approximate_flow_mismatch": flow_mismatch,
        "max_speed_m_s": max_speed,
        "max_cfl": max_cfl,
        "interior_divergence_rms_per_s": div_rms,
        "relative_interior_divergence": relative_divergence,
        "native_eb_divergence_gate_status": "not_measured",
        "divergence_proxy_note": (
            "Cell-centred Cartesian central differences do not match incflo's "
            "nodal embedded-boundary projection operator."
        ),
        "diagnostics": {
            "central_difference_relative_divergence_lt_1e-5": bool(
                relative_divergence is not None and relative_divergence < 1.0e-5
            ),
        },
        "gates": {
            "finite": bool(finite),
            "sphere_volume_error_lt_3pct": bool(
                abs(solid_volume - analytic_sphere_volume) / analytic_sphere_volume < 0.03
            ),
            "flow_mismatch_lt_0p2pct": bool(flow_mismatch < 0.002),
            "max_cfl_lt_0p25": bool(max_cfl < 0.25),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plotfile", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.plotfile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(result["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
