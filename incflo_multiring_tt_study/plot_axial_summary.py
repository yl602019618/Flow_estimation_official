from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot metrics across all TT target slices.")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    windows = report["windows"]
    z = np.asarray([row["center_z_m"] for row in windows], dtype=float)
    truth_rms = np.asarray([row["truth_component_rms_m_s"] for row in windows])
    component_l2 = 100.0 * np.asarray(
        [row["noisy_metrics"]["component_relative_l2_mean"] for row in windows]
    )
    correlation = np.asarray(
        [row["noisy_metrics"]["component_correlation_mean"] for row in windows]
    )
    joint_l2 = 100.0 * np.asarray(
        [row["noisy_metrics"]["joint_relative_l2_mean"] for row in windows]
    )
    labels = ("Ux'", "Uy'", "Uz'")
    colors = ("tab:blue", "tab:orange", "tab:green")

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
    for component, (label, color) in enumerate(zip(labels, colors)):
        axes[0, 0].plot(z, truth_rms[:, component], "o-", color=color, label=label)
        axes[0, 1].plot(z, component_l2[:, component], "o-", color=color, label=label)
        axes[1, 0].plot(z, correlation[:, component], "o-", color=color, label=label)
    axes[1, 1].plot(z, joint_l2, "o-", color="black", label="joint")
    axes[0, 0].set_ylabel("Truth RMS (m/s)")
    axes[0, 1].set_ylabel("Noisy relative L2 (%)")
    axes[1, 0].set_ylabel("Noisy correlation")
    axes[1, 1].set_ylabel("Noisy joint relative L2 (%)")
    axes[0, 1].axhline(45.0, color="black", linestyle=":", label="45% gate")
    axes[1, 0].axhline(0.80, color="black", linestyle=":", label="0.80 gate")
    axes[1, 1].axhline(30.0, color="black", linestyle=":", label="30% gate")
    for axis in axes.flat:
        axis.set_xlabel("target slice z (m)")
        axis.grid(alpha=0.25)
        axis.legend(ncol=2)
    figure.suptitle(
        "Provisional incflo long-domain multi-ring TT inversion\n"
        "100 kHz, four side walls, five local rings per target, dense divergence-free basis"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    plt.close(figure)

    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "z_m", "truth_rms_ux_m_s", "truth_rms_uy_m_s", "truth_rms_uz_m_s",
            "joint_l2_percent", "ux_l2_percent", "uy_l2_percent", "uz_l2_percent",
            "ux_correlation", "uy_correlation", "uz_correlation",
        ])
        for index, z_value in enumerate(z):
            writer.writerow([
                z_value, *truth_rms[index], joint_l2[index], *component_l2[index],
                *correlation[index],
            ])
    print(json.dumps({"figure": str(args.output.resolve()), "csv": str(csv_path.resolve())}))


if __name__ == "__main__":
    main()
