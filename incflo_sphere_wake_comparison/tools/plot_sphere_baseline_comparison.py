#!/usr/bin/env python3
"""Plot time-matched incflo sphere/no-sphere centre-plane comparisons."""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from incflo_sphere_wake_comparison.tools.plot_velocity_slice import _load


def _time(metadata: dict[str, object]) -> float:
    return float(metadata.get("simulation_time_s", 0.0))


def _plane(metadata_path: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    metadata, fields = _load(metadata_path)
    return metadata, {name: np.asarray(value).T for name, value in fields.items()}


def _data_uri(path: Path) -> str:
    with Image.open(path) as source:
        image = source.convert("RGB")
        if image.width > 1200:
            image = image.resize(
                (1200, round(image.height * 1200 / image.width)),
                Image.Resampling.LANCZOS,
            )
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=66, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _write_inline_html(path: Path, figure_path: Path) -> None:
    fragment = f'''<section id="incflo-sphere-baseline-comparison-20260824">
  <h2>incflo 含球与无球同时间对照</h2>
  <figure>
    <img src="{_data_uri(figure_path)}" alt="t=0 与 t=30.969 秒的含球、无球、轴向速度差和速度扰动模对照。" style="display:block;width:100%;height:auto">
    <figcaption class="text-small text-muted">每行是一个共同保存时刻；各列依次为含球 U_z、无球 U_z、差值 ΔU_z 与 |ΔU|。</figcaption>
  </figure>
</section>
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fragment, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sphere", nargs=2, required=True, type=Path)
    parser.add_argument("--baseline", nargs=2, required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--inline-html", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    for sphere_path, baseline_path in zip(args.sphere, args.baseline):
        sphere_meta, sphere = _plane(sphere_path)
        baseline_meta, baseline = _plane(baseline_path)
        if sphere_meta["grid"] != baseline_meta["grid"]:
            raise ValueError("sphere and baseline slice grids differ")
        if abs(_time(sphere_meta) - _time(baseline_meta)) > 1.0e-9:
            raise ValueError("sphere and baseline slice times differ")
        fluid = sphere["vfrac"] > 0.5
        delta = np.stack([
            sphere["velx"] - baseline["velx"],
            sphere["vely"] - baseline["vely"],
            sphere["velz"] - baseline["velz"],
        ])
        delta[:, ~fluid] = np.nan
        rows.append((sphere_meta, sphere, baseline, delta, fluid))

    total_values = np.concatenate([
        value[1]["velz"][value[4]] for value in rows
    ] + [
        value[2]["velz"].ravel() for value in rows
    ])
    total_low, total_high = np.percentile(total_values, [0.3, 99.7])
    delta_values = np.concatenate([value[3][2][np.isfinite(value[3][2])] for value in rows])
    delta_limit = max(0.04, float(np.percentile(np.abs(delta_values), 99.4)))
    magnitude_values = [np.sqrt(np.nansum(value[3] ** 2, axis=0)) for value in rows]
    magnitude_limit = max(
        0.04,
        float(np.percentile(np.concatenate([value[np.isfinite(value)] for value in magnitude_values]), 99.4)),
    )

    figure, axes = plt.subplots(2, 4, figsize=(18, 7.4), constrained_layout=True)
    figure.suptitle(
        "incflo strict-double: sphere and no-sphere flows at every common saved time\n"
        "centre plane y=0.75 m, grid 144×144×768"
    )
    report_rows = []
    images_by_column = [None, None, None, None]
    for row_index, ((metadata, sphere, baseline, delta, fluid), magnitude) in enumerate(zip(rows, magnitude_values)):
        left = metadata["domain_left_m"]
        right = metadata["domain_right_m"]
        extent = [left[2], right[2], left[0], right[0]]
        time_s = _time(metadata)
        panels = [
            (np.where(fluid, sphere["velz"], np.nan), r"Sphere $U_z$", "viridis", total_low, total_high),
            (baseline["velz"], r"No-sphere $U_z$", "viridis", total_low, total_high),
            (delta[2], r"Matched $\Delta U_z$", "RdBu_r", -delta_limit, delta_limit),
            (np.where(fluid, magnitude, np.nan), r"Matched $|\Delta U|$", "magma", 0.0, magnitude_limit),
        ]
        for column, (axis, (field, title, cmap, low, high)) in enumerate(zip(axes[row_index], panels)):
            images_by_column[column] = axis.imshow(
                field,
                origin="lower",
                extent=extent,
                aspect="auto",
                interpolation="bilinear",
                cmap=cmap,
                vmin=low,
                vmax=high,
            )
            axis.set_title(title)
            axis.set_xlabel("z (m)")
            axis.set_ylabel(f"t={time_s:.3f} s\nx (m)")
            axis.axvline(1.0, color="black", linestyle=":", linewidth=0.8)
            axis.axvline(7.05, color="black", linestyle=":", linewidth=0.8)
            circle = plt.Circle(
                (0.75, 0.75),
                0.15,
                fill=False,
                color="black",
                linewidth=1.0,
                linestyle=":" if column == 1 else "-",
            )
            axis.add_patch(circle)
        finite = np.isfinite(delta)
        component_rms = [
            float(np.sqrt(np.mean(delta[index][finite[index]] ** 2))) for index in range(3)
        ]
        report_rows.append({
            "time_s": time_s,
            "component_delta_rms_m_s": component_rms,
            "delta_magnitude_max_m_s": float(np.nanmax(magnitude)),
        })
    for column, image in enumerate(images_by_column):
        assert image is not None
        figure.colorbar(
            image,
            ax=axes[:, column],
            label="m/s",
            shrink=0.78,
            pad=0.01,
        )
    figure_path = output / "incflo_sphere_vs_no_sphere_common_times.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    report = {
        "comparison": "sphere versus no-sphere at every common saved time",
        "common_time_count": len(rows),
        "rows": report_rows,
        "figure": str(figure_path),
        "source_sphere_metadata": [str(path.resolve()) for path in args.sphere],
        "source_baseline_metadata": [str(path.resolve()) for path in args.baseline],
    }
    (output / "comparison_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.inline_html is not None:
        _write_inline_html(args.inline_html.resolve(), figure_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
