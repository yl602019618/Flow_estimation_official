from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read(path: Path, common_z: np.ndarray) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    windows = report["windows"]
    z = np.asarray([row["center_z_m"] for row in windows])
    joint = np.asarray([row["noisy_metrics"]["joint_relative_l2_mean"] for row in windows])
    clean_joint = np.asarray([row["clean_metrics"]["joint_relative_l2"] for row in windows])
    component_l2 = np.asarray([
        row["noisy_metrics"]["component_relative_l2_mean"] for row in windows
    ])
    correlation = np.asarray([
        row["noisy_metrics"]["component_correlation_mean"] for row in windows
    ])
    if common_z[0] < z[0] or common_z[-1] > z[-1]:
        raise ValueError("common comparison grid is outside an experiment target range")
    interpolated = {
        "joint": np.interp(common_z, z, joint),
        "clean_joint": np.interp(common_z, z, clean_joint),
        "component_l2": np.column_stack([
            np.interp(common_z, z, component_l2[:, component]) for component in range(3)
        ]),
        "correlation": np.column_stack([
            np.interp(common_z, z, correlation[:, component]) for component in range(3)
        ]),
    }
    contract = report["contract"]
    gates = (
        (interpolated["joint"] <= 0.30)
        & (np.max(interpolated["component_l2"], axis=1) <= 0.45)
        & (np.min(interpolated["correlation"], axis=1) >= 0.80)
    )
    summary = {
        "global_ring_count": int(contract["global_ring_count"]),
        "ring_spacing_m": float(contract["ring_spacing_m"]),
        "global_sensor_count": int(contract["global_sensor_count"]),
        "local_sensor_count": int(contract["local_sensor_count"]),
        "local_ray_count": int(contract["local_ray_count"]),
        "native_target_count": int(len(z)),
        "common_target_count": int(len(common_z)),
        "joint_l2_mean": float(np.mean(interpolated["joint"])),
        "joint_l2_median": float(np.median(interpolated["joint"])),
        "joint_l2_max": float(np.max(interpolated["joint"])),
        "clean_joint_l2_mean": float(np.mean(interpolated["clean_joint"])),
        "clean_joint_l2_median": float(np.median(interpolated["clean_joint"])),
        "clean_joint_l2_max": float(np.max(interpolated["clean_joint"])),
        "component_l2_mean": np.mean(interpolated["component_l2"], axis=0).tolist(),
        "component_correlation_mean": np.mean(interpolated["correlation"], axis=0).tolist(),
        "all_gate_pass_count": int(np.sum(gates)),
        "effective_rank_1e3": int(windows[0]["operator"]["effective_rank_1e3"]),
        "elapsed_seconds": float(report["elapsed_seconds"]),
    }
    native = {"z": z, "joint": joint}
    return summary, native


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize fixed-coverage ring-count levels.")
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    common_z = np.arange(2.1, 6.0 + 1e-9, 0.3)
    records = [_read(path, common_z) for path in args.runs]
    records.sort(key=lambda value: value[0]["global_ring_count"])
    summaries = [value[0] for value in records]
    native = [value[1] for value in records]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = {"common_z_m": common_z.tolist(), "levels": summaries}
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "ring_count", "ring_spacing_m", "global_sensor_count", "local_ray_count",
            "joint_l2_mean_percent", "joint_l2_median_percent", "joint_l2_max_percent",
            "clean_joint_l2_mean_percent", "clean_joint_l2_median_percent",
            "clean_joint_l2_max_percent",
            "ux_l2_mean_percent", "uy_l2_mean_percent", "uz_l2_mean_percent",
            "ux_corr_mean", "uy_corr_mean", "uz_corr_mean", "all_gate_pass_count",
            "common_target_count", "effective_rank_1e3", "elapsed_seconds",
        ])
        for row in summaries:
            writer.writerow([
                row["global_ring_count"], row["ring_spacing_m"], row["global_sensor_count"],
                row["local_ray_count"], 100 * row["joint_l2_mean"],
                100 * row["joint_l2_median"], 100 * row["joint_l2_max"],
                100 * row["clean_joint_l2_mean"], 100 * row["clean_joint_l2_median"],
                100 * row["clean_joint_l2_max"],
                *[100 * value for value in row["component_l2_mean"]],
                *row["component_correlation_mean"], row["all_gate_pass_count"],
                row["common_target_count"], row["effective_rank_1e3"], row["elapsed_seconds"],
            ])

    counts = np.asarray([row["global_ring_count"] for row in summaries])
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(counts, [100 * row["joint_l2_mean"] for row in summaries], "o-", label="mean")
    axes[0, 0].plot(counts, [100 * row["joint_l2_median"] for row in summaries], "s--", label="median")
    axes[0, 0].plot(counts, [100 * row["joint_l2_max"] for row in summaries], "^:", label="worst")
    for component, label in enumerate(("Ux'", "Uy'", "Uz'")):
        axes[0, 1].plot(counts, [100 * row["component_l2_mean"][component] for row in summaries], "o-", label=label)
        axes[1, 0].plot(counts, [row["component_correlation_mean"][component] for row in summaries], "o-", label=label)
    axes[1, 1].plot(counts, [row["all_gate_pass_count"] for row in summaries], "o-", color="black", label="all gates")
    axes[0, 0].axhline(30, color="black", linestyle=":", label="30% gate")
    axes[1, 0].axhline(0.8, color="black", linestyle=":", label="0.80 gate")
    axes[0, 0].set_ylabel("Joint relative L2 (%)")
    axes[0, 1].set_ylabel("Mean component L2 (%)")
    axes[1, 0].set_ylabel("Mean component correlation")
    axes[1, 1].set_ylabel(f"Common points passing all gates / {len(common_z)}")
    for axis in axes.flat:
        axis.set_xlabel("global ring count over z=1.05-7.05 m")
        axis.set_xticks(counts)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Provisional ring-count sweep: 36 sensors/ring, fixed 6 m coverage")
    figure.savefig(output / "ring_count_tradeoff.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11.5, 4.5), constrained_layout=True)
    for row, curve in zip(summaries, native):
        axis.plot(curve["z"], 100 * curve["joint"], "o-", label=f"{row['global_ring_count']} rings")
    axis.axhline(30, color="black", linestyle=":", label="30% gate")
    axis.set(xlabel="native target z (m)", ylabel="Noisy joint relative L2 (%)",
             title="Axial error on each array's native interior-ring targets")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output / "ring_count_axial_error.png", dpi=200)
    plt.close(figure)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
