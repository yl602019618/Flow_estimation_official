from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COMPONENTS = ("Ux'", "Uy'", "Uz'")


def _slice_metrics(estimate: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    a = estimate.ravel()
    b = truth.ravel()
    error = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))
    correlation = float(np.corrcoef(a, b)[0, 1])
    return error, correlation


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare 100 and 300 kHz TT slices")
    parser.add_argument(
        "--root", default="multislice_xyz_tt_study/outputs",
        help="Directory containing refined_240_100khz and refined_240_300khz",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    low = np.load(root / "refined_240_100khz" / "reconstruction.npz")
    high = np.load(root / "refined_240_300khz" / "reconstruction.npz")
    output = root / "frequency_comparison_100_vs_300khz"
    output.mkdir(parents=True, exist_ok=True)

    truth = np.asarray(high["truth"])
    rec100 = np.asarray(low["reconstruction"])
    rec300 = np.asarray(high["reconstruction"])
    x, y, z = (np.asarray(high[key]) for key in ("x", "y", "z"))
    if not (
        np.allclose(truth, low["truth"])
        and np.allclose(x, low["x"])
        and np.allclose(y, low["y"])
        and np.allclose(z, low["z"])
    ):
        raise ValueError("The 100 and 300 kHz results do not share one truth/evaluation grid")

    requested_z = (2.20, 2.35, 2.50, 2.65, 2.80)
    indices = [int(np.argmin(np.abs(z - value))) for value in requested_z]
    extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
    limits = [max(float(np.max(np.abs(truth[c, :, :, indices]))), 1e-8)
              for c in range(3)]

    figure, axes = plt.subplots(
        len(indices), 9, figsize=(30, 16.5), sharex=True, sharey=True,
        constrained_layout=True,
    )
    images = [None, None, None]
    for row, iz in enumerate(indices):
        for component in range(3):
            for variant, field in enumerate((truth, rec100, rec300)):
                column = 3 * component + variant
                axis = axes[row, column]
                images[component] = axis.imshow(
                    field[component, :, :, iz].T,
                    origin="lower", extent=extent, cmap="RdBu_r",
                    vmin=-limits[component], vmax=limits[component],
                    interpolation="nearest", aspect="equal",
                )
                if variant:
                    error, correlation = _slice_metrics(
                        field[component, :, :, iz], truth[component, :, :, iz],
                    )
                    axis.text(
                        0.03, 0.96, f"L2 {100*error:.1f}%  r {correlation:.3f}",
                        transform=axis.transAxes, va="top", ha="left", fontsize=8,
                        bbox={"facecolor": "white", "alpha": 0.78,
                              "edgecolor": "none", "pad": 1.5},
                    )
                if row == 0:
                    axis.set_title(
                        f"{COMPONENTS[component]} — "
                        f"{('Truth', '100 kHz', '300 kHz')[variant]}"
                    )
                if column == 0:
                    axis.set_ylabel(f"z={z[iz]:.2f} m\ny (m)")
                if row == len(indices) - 1:
                    axis.set_xlabel("x (m)")
    for component in range(3):
        colorbar = figure.colorbar(
            images[component], ax=axes[:, 3*component:3*component+3],
            location="bottom", shrink=0.72, pad=0.025, extend="both",
        )
        colorbar.set_label(f"{COMPONENTS[component]} disturbance (m/s); fixed scale")
    figure.suptitle(
        "Sphere-wake thin-slab Travel-Time reconstruction: 100 vs 300 kHz\n"
        "240 four-wall sensors, five z layers, identical 13×13×7 divergence-free basis",
        fontsize=17,
    )
    figure.savefig(output / "five_xy_slices_truth_100_vs_300khz.png", dpi=180)
    figure.savefig(output / "five_xy_slices_truth_100_vs_300khz.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(
        len(indices), 6, figsize=(21, 16), sharex=True, sharey=True,
        constrained_layout=True,
    )
    error_limits = []
    for component in range(3):
        values = np.concatenate([
            np.abs(rec100[component, :, :, indices] - truth[component, :, :, indices]).ravel(),
            np.abs(rec300[component, :, :, indices] - truth[component, :, :, indices]).ravel(),
        ])
        error_limits.append(max(float(np.quantile(values, 0.995)), 1e-8))
    error_images = [None, None, None]
    for row, iz in enumerate(indices):
        for component in range(3):
            for variant, field in enumerate((rec100, rec300)):
                column = 2 * component + variant
                error = field[component, :, :, iz] - truth[component, :, :, iz]
                error_images[component] = axes[row, column].imshow(
                    error.T, origin="lower", extent=extent, cmap="RdBu_r",
                    vmin=-error_limits[component], vmax=error_limits[component],
                    interpolation="nearest", aspect="equal",
                )
                if row == 0:
                    axes[row, column].set_title(
                        f"{COMPONENTS[component]} error — {('100', '300')[variant]} kHz"
                    )
                if column == 0:
                    axes[row, column].set_ylabel(f"z={z[iz]:.2f} m\ny (m)")
                if row == len(indices) - 1:
                    axes[row, column].set_xlabel("x (m)")
    for component in range(3):
        colorbar = figure.colorbar(
            error_images[component], ax=axes[:, 2*component:2*component+2],
            location="bottom", shrink=0.72, pad=0.025, extend="both",
        )
        colorbar.set_label(f"{COMPONENTS[component]} error (m/s); shared 99.5% scale")
    figure.suptitle("Reconstruction error on the same five x-y slices", fontsize=17)
    figure.savefig(output / "five_xy_slices_error_100_vs_300khz.png", dpi=180)
    figure.savefig(output / "five_xy_slices_error_100_vs_300khz.pdf")
    plt.close(figure)

    for component, file_label in enumerate(("ux", "uy", "uz")):
        figure, axes = plt.subplots(
            len(indices), 5, figsize=(17.5, 16), sharex=True, sharey=True,
        )
        figure.subplots_adjust(
            left=0.055, right=0.985, top=0.945, bottom=0.105,
            wspace=0.12, hspace=0.08,
        )
        physical_image = None
        error_image = None
        for row, iz in enumerate(indices):
            panels = (
                ("GT", truth[component, :, :, iz], False, None),
                ("100 kHz reconstruction", rec100[component, :, :, iz], False, rec100),
                ("100 kHz error", rec100[component, :, :, iz] - truth[component, :, :, iz], True, None),
                ("300 kHz reconstruction", rec300[component, :, :, iz], False, rec300),
                ("300 kHz error", rec300[component, :, :, iz] - truth[component, :, :, iz], True, None),
            )
            for column, (title, values, is_error, metric_field) in enumerate(panels):
                limit = error_limits[component] if is_error else limits[component]
                image = axes[row, column].imshow(
                    values.T, origin="lower", extent=extent, cmap="RdBu_r",
                    vmin=-limit, vmax=limit, interpolation="nearest", aspect="equal",
                )
                if is_error:
                    error_image = image
                else:
                    physical_image = image
                if metric_field is not None:
                    error, correlation = _slice_metrics(
                        metric_field[component, :, :, iz], truth[component, :, :, iz],
                    )
                    axes[row, column].text(
                        0.03, 0.96, f"L2 {100*error:.1f}%  r {correlation:.3f}",
                        transform=axes[row, column].transAxes,
                        va="top", ha="left", fontsize=9,
                        bbox={"facecolor": "white", "alpha": 0.80,
                              "edgecolor": "none", "pad": 1.5},
                    )
                if row == 0:
                    axes[row, column].set_title(title)
                if column == 0:
                    axes[row, column].set_ylabel(f"z={z[iz]:.2f} m\ny (m)")
                if row == len(indices) - 1:
                    axes[row, column].set_xlabel("x (m)")
        physical_cax = figure.add_axes((0.08, 0.045, 0.38, 0.018))
        error_cax = figure.add_axes((0.55, 0.045, 0.38, 0.018))
        physical_colorbar = figure.colorbar(
            physical_image, cax=physical_cax, orientation="horizontal", extend="both",
        )
        physical_colorbar.set_label(
            f"{COMPONENTS[component]} disturbance (m/s); common GT/reconstruction scale"
        )
        error_colorbar = figure.colorbar(
            error_image, cax=error_cax, orientation="horizontal", extend="both",
        )
        error_colorbar.set_label(
            f"{COMPONENTS[component]} reconstruction error (m/s); common frequency scale"
        )
        figure.suptitle(
            f"{COMPONENTS[component]} on five x-y slices: GT, reconstruction, and error",
            fontsize=17,
        )
        figure.savefig(
            output / f"{file_label}_five_slices_gt_reconstruction_error.png", dpi=180,
        )
        figure.savefig(output / f"{file_label}_five_slices_gt_reconstruction_error.pdf")
        plt.close(figure)

    rows = []
    for iz in indices:
        for frequency, field in ((100, rec100), (300, rec300)):
            row = {"z_m": float(z[iz]), "frequency_khz": frequency}
            for component, label in enumerate(("ux", "uy", "uz")):
                error, correlation = _slice_metrics(
                    field[component, :, :, iz], truth[component, :, :, iz],
                )
                row[f"{label}_relative_l2"] = error
                row[f"{label}_correlation"] = correlation
            rows.append(row)
    import json
    (output / "five_slice_metrics.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8",
    )


if __name__ == "__main__":
    main()
