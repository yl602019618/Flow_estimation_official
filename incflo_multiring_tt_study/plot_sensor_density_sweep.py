from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _summarize(path: Path) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    windows = report["windows"]
    z = np.asarray([row["center_z_m"] for row in windows], dtype=float)
    joint = np.asarray(
        [row["noisy_metrics"]["joint_relative_l2_mean"] for row in windows]
    )
    clean_joint = np.asarray(
        [row["clean_metrics"]["joint_relative_l2"] for row in windows]
    )
    component_l2 = np.asarray(
        [row["noisy_metrics"]["component_relative_l2_mean"] for row in windows]
    )
    correlation = np.asarray(
        [row["noisy_metrics"]["component_correlation_mean"] for row in windows]
    )
    contract = report["contract"]
    all_gates = (
        (joint <= 0.30)
        & (np.max(component_l2, axis=1) <= 0.45)
        & (np.min(correlation, axis=1) >= 0.80)
    )
    row = {
        "sensors_per_ring": int(contract["sensors_per_ring"]),
        "global_sensor_count": int(contract["global_sensor_count"]),
        "local_sensor_count": int(contract["local_sensor_count"]),
        "local_ray_count": int(contract["local_ray_count"]),
        "joint_l2_mean": float(np.mean(joint)),
        "joint_l2_median": float(np.median(joint)),
        "joint_l2_max": float(np.max(joint)),
        "clean_joint_l2_mean": float(np.mean(clean_joint)),
        "clean_joint_l2_median": float(np.median(clean_joint)),
        "clean_joint_l2_max": float(np.max(clean_joint)),
        "component_l2_mean": np.mean(component_l2, axis=0).tolist(),
        "component_correlation_mean": np.mean(correlation, axis=0).tolist(),
        "all_gate_pass_count": int(np.sum(all_gates)),
        "slice_count": int(len(windows)),
        "elapsed_seconds": float(report["elapsed_seconds"]),
        "effective_rank_1e3": int(windows[0]["operator"]["effective_rank_1e3"]),
        "clean_selected_rank_median": float(np.median([
            window["clean_fit"]["selected_spectral_rank"] for window in windows
        ])),
    }
    return row, z, 100.0 * joint


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a four-level sensor sweep.")
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summaries, z_axes, joint_rows = [], [], []
    for run in args.runs:
        summary, z, joint = _summarize(run)
        summaries.append(summary)
        z_axes.append(z)
        joint_rows.append(joint)
    order = np.argsort([row["sensors_per_ring"] for row in summaries])
    summaries = [summaries[index] for index in order]
    z_axes = [z_axes[index] for index in order]
    joint_rows = [joint_rows[index] for index in order]
    if not all(np.allclose(z_axes[0], z) for z in z_axes[1:]):
        raise ValueError("sensor levels do not contain identical target slices")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps({"levels": summaries}, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "sensors_per_ring", "global_sensor_count", "local_sensor_count",
            "local_ray_count", "joint_l2_mean_percent", "joint_l2_median_percent",
            "joint_l2_max_percent", "ux_l2_mean_percent", "uy_l2_mean_percent",
            "uz_l2_mean_percent", "ux_corr_mean", "uy_corr_mean", "uz_corr_mean",
            "clean_joint_l2_mean_percent", "clean_joint_l2_median_percent",
            "clean_joint_l2_max_percent", "all_gate_pass_count", "slice_count",
            "effective_rank_1e3", "clean_selected_rank_median", "elapsed_seconds",
        ])
        for row in summaries:
            writer.writerow([
                row["sensors_per_ring"], row["global_sensor_count"],
                row["local_sensor_count"], row["local_ray_count"],
                100 * row["joint_l2_mean"], 100 * row["joint_l2_median"],
                100 * row["joint_l2_max"],
                *[100 * value for value in row["component_l2_mean"]],
                *row["component_correlation_mean"],
                100 * row["clean_joint_l2_mean"], 100 * row["clean_joint_l2_median"],
                100 * row["clean_joint_l2_max"], row["all_gate_pass_count"],
                row["slice_count"], row["effective_rank_1e3"],
                row["clean_selected_rank_median"], row["elapsed_seconds"],
            ])

    sensors = np.asarray([row["sensors_per_ring"] for row in summaries])
    joint_mean = 100 * np.asarray([row["joint_l2_mean"] for row in summaries])
    joint_median = 100 * np.asarray([row["joint_l2_median"] for row in summaries])
    joint_max = 100 * np.asarray([row["joint_l2_max"] for row in summaries])
    component_l2 = 100 * np.asarray([row["component_l2_mean"] for row in summaries])
    correlation = np.asarray([row["component_correlation_mean"] for row in summaries])
    passed = np.asarray([row["all_gate_pass_count"] for row in summaries])
    labels = ("Ux'", "Uy'", "Uz'")

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(sensors, joint_mean, "o-", label="mean")
    axes[0, 0].plot(sensors, joint_median, "s--", label="median")
    axes[0, 0].plot(sensors, joint_max, "^:", label="worst slice")
    axes[0, 0].axhline(30, color="black", linestyle=":", label="30% gate")
    for component, label in enumerate(labels):
        axes[0, 1].plot(sensors, component_l2[:, component], "o-", label=label)
        axes[1, 0].plot(sensors, correlation[:, component], "o-", label=label)
    axes[1, 0].axhline(0.80, color="black", linestyle=":", label="0.80 gate")
    axes[1, 1].plot(sensors, passed, "o-", color="black", label="all gates")
    axes[0, 0].set_ylabel("Joint relative L2 (%)")
    axes[0, 1].set_ylabel("Mean component relative L2 (%)")
    axes[1, 0].set_ylabel("Mean component correlation")
    axes[1, 1].set_ylabel("Slices passing all gates / 19")
    axes[1, 1].set_ylim(0, 19.5)
    for axis in axes.flat:
        axis.set_xlabel("sensors per ring")
        axis.set_xticks(sensors)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(
        "Provisional sensor-density sweep: fixed 41 rings, 100 kHz, 6,075 modes"
    )
    figure.savefig(output / "sensor_density_tradeoff.png", dpi=200)
    plt.close(figure)

    heatmap = np.asarray(joint_rows)
    figure, axis = plt.subplots(figsize=(12, 3.8), constrained_layout=True)
    image = axis.imshow(heatmap, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(45, heatmap.max()))
    axis.set_yticks(np.arange(len(sensors)), [f"{value}/ring" for value in sensors])
    axis.set_xticks(np.arange(len(z_axes[0])), [f"{value:.2f}" for value in z_axes[0]], rotation=60)
    axis.set_xlabel("target slice z (m)")
    axis.set_ylabel("sensor level")
    for row in range(heatmap.shape[0]):
        for column in range(heatmap.shape[1]):
            axis.text(column, row, f"{heatmap[row, column]:.0f}", ha="center", va="center",
                      color="white" if heatmap[row, column] > 0.55 * heatmap.max() else "black",
                      fontsize=7)
    figure.colorbar(image, ax=axis, label="noisy joint relative L2 (%)")
    axis.set_title("Axial distribution of reconstruction error")
    figure.savefig(output / "sensor_density_axial_heatmap.png", dpi=200)
    plt.close(figure)

    clean_mean = 100 * np.asarray([row["clean_joint_l2_mean"] for row in summaries])
    effective_rank = np.asarray([row["effective_rank_1e3"] for row in summaries])
    selected_rank = np.asarray([row["clean_selected_rank_median"] for row in summaries])
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    axes[0].plot(sensors, clean_mean, "o-", label="clean mean")
    axes[0].plot(sensors, joint_mean, "s--", label="noisy mean")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Mean joint relative L2 (%)")
    axes[1].plot(sensors, effective_rank, "o-", label="effective rank at 1e-3")
    axes[1].plot(sensors, selected_rank, "s--", label="median GCV clean rank")
    axes[1].set_ylabel("Spectral rank")
    for axis in axes:
        axis.set_xlabel("sensors per ring")
        axis.set_xticks(sensors)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("GCV/rank diagnostic for the sensor-density sweep")
    figure.savefig(output / "sensor_density_gcv_diagnostic.png", dpi=200)
    plt.close(figure)
    print(json.dumps({"output": str(output), "levels": summaries}, indent=2))


if __name__ == "__main__":
    main()
