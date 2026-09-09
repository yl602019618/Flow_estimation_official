"""Convert one native AMReX/incflo plotfile to a compact velocity HDF5 file."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from incflo_sphere_wake_comparison.tools.analyze_plotfile import _load_native


def export_snapshot(plotfile: Path, output: Path) -> None:
    names, dims, left, right, data, real_bits = _load_native(plotfile)
    index = {name: position for position, name in enumerate(names)}
    required = ("velx", "vely", "velz", "vfrac")
    missing = set(required) - set(index)
    if missing:
        raise KeyError(f"missing plotfile variables: {sorted(missing)}")
    velocity = np.moveaxis(
        data[..., [index["velx"], index["vely"], index["velz"]]], -1, 0,
    ).astype(np.float32)
    vfrac = np.asarray(data[..., index["vfrac"]], dtype=np.float32)
    if not np.isfinite(velocity).all() or not np.isfinite(vfrac).all():
        raise ValueError("plotfile contains NaN or Inf")
    header = (plotfile / "Header").read_text(encoding="utf-8").splitlines()
    time_s = float(header[2 + len(names) + 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        handle.create_dataset(
            "velocity", data=velocity, compression="gzip", compression_opts=1,
            chunks=(1, min(36, dims[0]), min(36, dims[1]), min(96, dims[2])),
        )
        handle.create_dataset(
            "vfrac", data=vfrac, compression="gzip", compression_opts=1,
            chunks=(min(36, dims[0]), min(36, dims[1]), min(96, dims[2])),
        )
        handle.attrs["time_s"] = time_s
        handle.attrs["domain_left_m"] = left
        handle.attrs["domain_right_m"] = right
        handle.attrs["grid"] = dims
        handle.attrs["source_plotfile"] = str(plotfile.resolve())
        handle.attrs["source_storage_precision_bits"] = real_bits
        handle.attrs["velocity_storage_precision_bits"] = 32
        handle.attrs["solver"] = "incflo strict-double EB; compact float32 export"


def write_uniform_baseline(snapshot: Path, output: Path) -> None:
    with h5py.File(snapshot, "r") as source:
        shape = source["velocity"].shape
        metadata = dict(source.attrs)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        baseline = handle.create_dataset(
            "velocity", shape=shape, dtype=np.float32, compression="gzip",
            compression_opts=1,
            chunks=(1, min(36, shape[1]), min(36, shape[2]), min(96, shape[3])),
            fillvalue=0.0,
        )
        baseline[2] = 0.5
        for key, value in metadata.items():
            handle.attrs[key] = value
        handle.attrs["truth_status"] = "provisional_uniform_baseline"
        handle.attrs["baseline_velocity_m_s"] = (0.0, 0.0, 0.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plotfile", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--uniform-baseline", type=Path)
    args = parser.parse_args()
    export_snapshot(args.plotfile, args.output)
    if args.uniform_baseline:
        write_uniform_baseline(args.output, args.uniform_baseline)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
