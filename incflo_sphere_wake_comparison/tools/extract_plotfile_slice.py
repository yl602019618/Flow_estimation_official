#!/usr/bin/env python3
"""Extract one axis-aligned plane from a single-level AMReX plotfile.

This deliberately uses only the Python standard library so it can run on the
remote CFD host without installing numpy or yt. The binary output is little-
endian float64, field-major, with each field stored as ``(z, horizontal)``.
"""

from __future__ import annotations

import argparse
import array
import json
import math
import re
import sys
from pathlib import Path


BOX_RE = re.compile(
    r"\(\(([-\d]+),([-\d]+),([-\d]+)\) "
    r"\(([-\d]+),([-\d]+),([-\d]+)\) \([^)]+\)\)"
)


def extract_plane(plotfile: Path, axis: str, coordinate: float, output: Path) -> dict:
    lines = (plotfile / "Header").read_text(encoding="utf-8").splitlines()
    field_count = int(lines[1])
    names = lines[2 : 2 + field_count]
    dim_index = 2 + field_count
    if int(lines[dim_index]) != 3:
        raise ValueError("Only 3-D plotfiles are supported")
    simulation_time = float(lines[dim_index + 1])
    left = [float(value) for value in lines[dim_index + 3].split()]
    right = [float(value) for value in lines[dim_index + 4].split()]

    cell_header = (plotfile / "Level_0" / "Cell_H").read_text(encoding="utf-8")
    disk_entries = re.findall(r"FabOnDisk:\s+(\S+)\s+(\d+)", cell_header)
    matches = list(BOX_RE.finditer(cell_header))
    boxes = []
    for match in matches[: len(disk_entries)]:
        values = tuple(int(value) for value in match.groups())
        boxes.append((values[:3], values[3:]))
    dims = [max(high[d] for _, high in boxes) + 1 for d in range(3)]
    spacing = [(right[d] - left[d]) / dims[d] for d in range(3)]
    fixed_dimension = {"x": 0, "y": 1}[axis]
    horizontal_dimension = 1 if axis == "x" else 0
    fixed_index = min(
        dims[fixed_dimension] - 1,
        max(
            0,
            int(
                round(
                    (coordinate - left[fixed_dimension])
                    / spacing[fixed_dimension]
                    - 0.5
                )
            ),
        ),
    )
    actual_coordinate = (
        left[fixed_dimension] + (fixed_index + 0.5) * spacing[fixed_dimension]
    )

    first_name, first_offset = disk_entries[0]
    with (plotfile / "Level_0" / first_name).open("rb") as handle:
        handle.seek(int(first_offset))
        descriptor = handle.readline().decode("ascii")
    precision = re.search(r"^FAB \(\(8, \((32|64)\s", descriptor)
    if precision is None:
        raise ValueError(f"Unsupported FAB descriptor: {descriptor!r}")
    bits = int(precision.group(1))
    typecode = "f" if bits == 32 else "d"
    itemsize = bits // 8
    plane_size = dims[2] * dims[horizontal_dimension]
    planes = [array.array("d", [math.nan]) * plane_size for _ in names]

    handles = {}
    try:
        for (lo, high), (filename, offset_text) in zip(boxes, disk_entries):
            if not lo[fixed_dimension] <= fixed_index <= high[fixed_dimension]:
                continue
            handle = handles.get(filename)
            if handle is None:
                handle = (plotfile / "Level_0" / filename).open("rb")
                handles[filename] = handle
            handle.seek(int(offset_text))
            fab_header = handle.readline()
            data_start = handle.tell()
            nx = high[0] - lo[0] + 1
            ny = high[1] - lo[1] + 1
            nz = high[2] - lo[2] + 1
            cells_per_field = nx * ny * nz
            if axis == "y":
                local_y = fixed_index - lo[1]
                for field in range(field_count):
                    field_base = data_start + field * cells_per_field * itemsize
                    for local_z in range(nz):
                        cell_offset = (local_z * ny + local_y) * nx
                        handle.seek(field_base + cell_offset * itemsize)
                        values = array.array(typecode)
                        values.fromfile(handle, nx)
                        if sys.byteorder != "little":
                            values.byteswap()
                        target = (lo[2] + local_z) * dims[0] + lo[0]
                        planes[field][target : target + nx] = array.array(
                            "d", (float(value) for value in values)
                        )
            else:
                local_x = fixed_index - lo[0]
                for field in range(field_count):
                    field_base = data_start + field * cells_per_field * itemsize
                    handle.seek(field_base)
                    block = array.array(typecode)
                    block.fromfile(handle, cells_per_field)
                    if sys.byteorder != "little":
                        block.byteswap()
                    for local_z in range(nz):
                        for local_y in range(ny):
                            source = (local_z * ny + local_y) * nx + local_x
                            target = (
                                (lo[2] + local_z) * dims[1] + lo[1] + local_y
                            )
                            planes[field][target] = float(block[source])
    finally:
        for handle in handles.values():
            handle.close()

    if any(math.isnan(value) for plane in planes for value in plane):
        raise RuntimeError("The requested plane was not completely populated")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        for plane in planes:
            if sys.byteorder != "little":
                plane.byteswap()
            plane.tofile(handle)
    metadata = {
        "plotfile": str(plotfile),
        "simulation_time_s": simulation_time,
        "binary": output.name,
        "fields": names,
        "shape": [field_count, dims[2], dims[horizontal_dimension]],
        "storage": (
            "little-endian float64, field-major "
            f"(field,z,{['x', 'y'][horizontal_dimension]})"
        ),
        "domain_left_m": left,
        "domain_right_m": right,
        "grid": dims,
        "spacing_m": spacing,
        "fixed_axis": axis,
        "horizontal_axis": ["x", "y"][horizontal_dimension],
        "requested_coordinate_m": coordinate,
        "cell_index": fixed_index,
        "actual_cell_center_m": actual_coordinate,
    }
    metadata_path = output.with_suffix(output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plotfile", type=Path)
    fixed = parser.add_mutually_exclusive_group()
    fixed.add_argument("--x", type=float)
    fixed.add_argument("--y", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    axis = "x" if args.x is not None else "y"
    coordinate = args.x if args.x is not None else (args.y if args.y is not None else 0.75)
    result = extract_plane(args.plotfile, axis, coordinate, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
