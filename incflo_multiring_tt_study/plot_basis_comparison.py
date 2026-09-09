from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load(directory: Path) -> tuple[dict[str, object], np.lib.npyio.NpzFile]:
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    arrays = np.load(directory / "reconstruction.npz")
    return report, arrays


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare reference and dense localized curl bases at one slice."
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--dense", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    reference_report, reference = _load(args.reference)
    dense_report, dense = _load(args.dense)
    if not np.allclose(reference["truth"], dense["truth"]):
        raise ValueError("reference and dense runs do not use identical truth arrays")
    for key in ("x", "y", "z"):
        if not np.allclose(reference[key], dense[key]):
            raise ValueError(f"reference and dense {key} axes differ")

    truth = reference["truth"]
    reference_estimate = reference["reconstruction"]
    dense_estimate = dense["reconstruction"]
    z_index = truth.shape[-1] // 2
    z_value = float(reference["z"][z_index])
    ref_metrics = reference_report["full_cross_layer_result"]["noisy_metrics"]
    dense_metrics = dense_report["full_cross_layer_result"]["noisy_metrics"]
    ref_modes = int(reference_report["basis"]["mode_count"])
    dense_modes = int(dense_report["basis"]["mode_count"])

    figure, axes = plt.subplots(3, 5, figsize=(16.5, 9.4), constrained_layout=True)
    labels = ("Ux'", "Uy'", "Uz'")
    columns = ("Truth", f"Reference\n{ref_modes:,} modes", f"Dense\n{dense_modes:,} modes",
               "Reference error", "Dense error")
    for component, label in enumerate(labels):
        target = truth[component, :, :, z_index]
        ref = reference_estimate[component, :, :, z_index]
        den = dense_estimate[component, :, :, z_index]
        field_limit = max(float(np.quantile(np.abs(target), 0.998)), 1e-8)
        error_limit = max(
            float(np.quantile(np.abs(ref - target), 0.998)),
            float(np.quantile(np.abs(den - target), 0.998)),
            1e-8,
        )
        panels = (target, ref, den, ref - target, den - target)
        field_image = error_image = None
        for column, values in enumerate(panels):
            limit = field_limit if column < 3 else error_limit
            image = axes[component, column].imshow(
                values.T,
                origin="lower",
                extent=(0.0, 1.5, 0.0, 1.5),
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
            )
            if column < 3:
                field_image = image
            else:
                error_image = image
            axes[component, column].set_aspect("equal")
            axes[component, column].set_xlabel("x (m)")
            if column == 0:
                ref_l2 = 100.0 * ref_metrics["component_relative_l2_mean"][component]
                dense_l2 = 100.0 * dense_metrics["component_relative_l2_mean"][component]
                axes[component, column].set_ylabel(
                    f"y (m)\n{label}: L2 {ref_l2:.1f}% -> {dense_l2:.1f}%"
                )
            else:
                axes[component, column].set_yticklabels([])
            if component == 0:
                axes[component, column].set_title(columns[column])
        figure.colorbar(field_image, ax=axes[component, :3], label="m/s", shrink=0.83)
        figure.colorbar(error_image, ax=axes[component, 3:], label="error (m/s)", shrink=0.83)

    ref_joint = 100.0 * ref_metrics["joint_relative_l2_mean"]
    dense_joint = 100.0 * dense_metrics["joint_relative_l2_mean"]
    figure.suptitle(
        "Provisional incflo wake TT inversion: localized divergence-free basis density\n"
        f"z={z_value:.2f} m, 100 kHz, 240 sensors / 21,600 rays, noisy joint L2 "
        f"{ref_joint:.2f}% -> {dense_joint:.2f}%",
        fontsize=15,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    plt.close(figure)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
