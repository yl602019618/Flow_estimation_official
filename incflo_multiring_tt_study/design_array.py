from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .multiring import MultiRingDesign, global_acquisition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="incflo_multiring_tt_study/outputs/acquisition")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    design = MultiRingDesign()
    coordinates, walls, rings = global_acquisition(design)
    payload = {
        "design": {
            "z_min_m": design.z_min_m,
            "z_max_m": design.z_max_m,
            "ring_spacing_m": design.ring_spacing_m,
            "ring_z_m": design.ring_z_m.tolist(),
            "target_z_m": design.target_z_m.tolist(),
            "sensors_per_ring": design.sensors_per_ring,
            "ring_count": len(design.ring_z_m),
            "sensor_count": len(coordinates),
            "walls": sorted(set(walls)),
            "z_end_face_sensors": 0,
            "local_window_ring_count": 5,
        },
        "coordinates": coordinates.tolist(),
        "ring_indices": rings.tolist(),
        "wall_labels": list(walls),
    }
    (output / "acquisition.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8",
    )
    figure, axes = plt.subplots(2, 1, figsize=(13, 5.5), constrained_layout=True)
    for wall in sorted(set(walls)):
        mask = np.asarray(walls) == wall
        transverse = (
            coordinates[mask, 1] if wall.startswith("x") else coordinates[mask, 0]
        )
        axes[0].scatter(coordinates[mask, 2], transverse, s=6, label=wall)
    axes[0].set(
        xlabel="z (m)", ylabel="transverse wall coordinate (m)",
        title="41 rings on four side walls; 48 sensors per ring",
    )
    axes[0].legend(ncol=4)
    axes[0].grid(alpha=0.2)
    axes[1].vlines(design.ring_z_m, 0, 1, color="tab:blue", linewidth=0.8)
    axes[1].scatter(design.target_z_m, np.full_like(design.target_z_m, 0.55),
                    marker="v", color="tab:red", s=20, label="target slices")
    axes[1].set(xlabel="z (m)", yticks=[], ylim=(0, 1), title="Ring and target-slice locations")
    axes[1].legend()
    axes[1].grid(axis="x", alpha=0.2)
    figure.savefig(output / "acquisition.png", dpi=180)
    plt.close(figure)
    print(json.dumps(payload["design"], indent=2))


if __name__ == "__main__":
    main()
