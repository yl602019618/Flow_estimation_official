from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare reconstructed slices across sensor levels.")
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    parser.add_argument("--z", nargs="+", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    entries = []
    for path in args.runs:
        report = json.loads((path / "report.json").read_text(encoding="utf-8"))
        arrays = np.load(path / "reconstruction.npz")
        entries.append((int(report["contract"]["sensors_per_ring"]), report, arrays))
    entries.sort(key=lambda item: item[0])
    centers = np.asarray(entries[0][2]["centers_z_m"])
    if not all(np.allclose(centers, entry[2]["centers_z_m"]) for entry in entries[1:]):
        raise ValueError("sensor levels do not use identical target slices")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    labels = ("Ux'", "Uy'", "Uz'")
    for requested_z in args.z:
        window = int(np.argmin(np.abs(centers - requested_z)))
        if not np.isclose(centers[window], requested_z, atol=1e-6):
            raise ValueError(f"requested z={requested_z} is not a target slice")
        truth = entries[0][2]["truth"][window]
        local_z = truth.shape[-1] // 2
        figure, axes = plt.subplots(3, 1 + len(entries), figsize=(16, 9.2), constrained_layout=True)
        for component, label in enumerate(labels):
            target = truth[component, :, :, local_z]
            limit = max(float(np.quantile(np.abs(target), 0.998)), 1e-8)
            panels = [("Truth", target)]
            for sensors, _report, arrays in entries:
                panels.append((f"{sensors}/ring", arrays["reconstruction"][window, component, :, :, local_z]))
            image = None
            for column, (title, values) in enumerate(panels):
                image = axes[component, column].imshow(
                    values.T, origin="lower", extent=(0, 1.5, 0, 1.5),
                    cmap="RdBu_r", vmin=-limit, vmax=limit,
                )
                axes[component, column].set_aspect("equal")
                axes[component, column].set_xlabel("x (m)")
                if column == 0:
                    axes[component, column].set_ylabel(f"{label}\ny (m)")
                else:
                    axes[component, column].set_yticklabels([])
                if component == 0:
                    axes[component, column].set_title(title)
            figure.colorbar(image, ax=axes[component, :], label="m/s", shrink=0.82)
        figure.suptitle(
            f"Provisional sensor-density comparison at z={requested_z:.2f} m\n"
            "100 kHz, fixed 41 rings, five-ring local window, 6,075 divergence-free modes",
            fontsize=15,
        )
        figure.savefig(output / f"sensor_density_reconstruction_z{requested_z:.2f}.png", dpi=200)
        plt.close(figure)


if __name__ == "__main__":
    main()
