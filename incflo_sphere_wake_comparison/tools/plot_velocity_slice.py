#!/usr/bin/env python3
"""Plot the incflo centre-plane velocity fields extracted on the remote host."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(metadata_path: Path):
    import numpy as np

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    binary = metadata_path.parent / metadata["binary"]
    values = np.fromfile(binary, dtype="<f8").reshape(tuple(metadata["shape"]))
    fields = {name: values[index] for index, name in enumerate(metadata["fields"])}
    return metadata, fields


def plot(
    metadata_path: Path,
    output: Path,
    z_max: float | None = None,
    x_metadata_path: Path | None = None,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    metadata, fields = _load(metadata_path)
    x_metadata, x_fields = (
        _load(x_metadata_path) if x_metadata_path is not None else (metadata, fields)
    )
    x = np.linspace(
        metadata["domain_left_m"][0] + metadata["spacing_m"][0] / 2,
        metadata["domain_right_m"][0] - metadata["spacing_m"][0] / 2,
        metadata["grid"][0],
    )
    z = np.linspace(
        metadata["domain_left_m"][2] + metadata["spacing_m"][2] / 2,
        metadata["domain_right_m"][2] - metadata["spacing_m"][2] / 2,
        metadata["grid"][2],
    )
    keep = z <= (z_max if z_max is not None else z[-1] + metadata["spacing_m"][2])
    z = z[keep]
    vfrac = fields["vfrac"][keep, :].T
    ux = np.ma.masked_where(vfrac <= 0.0, fields["velx"][keep, :].T)
    x_vfrac = x_fields["vfrac"][keep, :].T
    uy = np.ma.masked_where(x_vfrac <= 0.0, x_fields["vely"][keep, :].T)
    uzp = np.ma.masked_where(vfrac <= 0.0, fields["velz"][keep, :].T - 0.5)
    speed = np.ma.masked_where(
        vfrac <= 0.0,
        np.sqrt(
            fields["velx"][keep, :].T ** 2
            + fields["vely"][keep, :].T ** 2
            + fields["velz"][keep, :].T ** 2
        ),
    )

    transverse_limit = max(
        0.01,
        float(np.nanpercentile(np.abs(np.concatenate([ux.compressed(), uy.compressed()])), 99.8)),
    )
    axial_limit = max(0.05, float(np.nanpercentile(np.abs(uzp.compressed()), 99.8)))
    panels = [
        (ux, r"$U_x$", "coolwarm", -transverse_limit, transverse_limit, "y", metadata["actual_cell_center_m"]),
        (uy, r"$U_y$", "coolwarm", -transverse_limit, transverse_limit, "x", x_metadata["actual_cell_center_m"]),
        (uzp, r"$U_z-U_\infty$", "coolwarm", -axial_limit, axial_limit, "y", metadata["actual_cell_center_m"]),
        (speed, r"$|U|$", "viridis", 0.0, max(0.55, float(np.nanpercentile(speed.compressed(), 99.8))), "y", metadata["actual_cell_center_m"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 6.8), constrained_layout=True)
    suffix = "full 8 m domain" if z_max is None else f"near-sphere view, z≤{z_max:g} m"
    grid_text = "×".join(str(value) for value in metadata["grid"])
    time_value = metadata.get("simulation_time_s")
    time_text = (
        f"t={time_value:.3f} s" if time_value is not None else "snapshot time unavailable"
    )
    fig.suptitle(
        "incflo strict-double sphere-wake velocity\n"
        f"{grid_text}, {time_text}, {suffix}"
    )
    extent = [z[0], z[-1], x[0], x[-1]]
    for axis, (field, title, cmap, low, high, fixed_axis, fixed_value) in zip(axes.flat, panels):
        image = axis.imshow(
            field,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=cmap,
            vmin=low,
            vmax=high,
            interpolation="bilinear",
        )
        axis.set_title(f"{title} on {fixed_axis}={fixed_value:.4f} m")
        axis.set_xlabel("z (m), downstream")
        axis.set_ylabel(f"{'y' if fixed_axis == 'x' else 'x'} (m)")
        fig.colorbar(image, ax=axis, label="m/s", pad=0.01)
        sphere = plt.Circle((0.75, 0.75), 0.15, fill=False, color="black", linewidth=1.0)
        axis.add_patch(sphere)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--z-max", type=float)
    parser.add_argument("--x-metadata", type=Path)
    args = parser.parse_args()
    plot(args.metadata, args.output, args.z_max, args.x_metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
