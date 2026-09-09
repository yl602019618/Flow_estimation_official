#!/usr/bin/env python3
"""Visualize a time-matched incflo sphere wake and no-sphere baseline."""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from incflo_sphere_wake_comparison.tools.plot_velocity_slice import _load


def _centers(left: float, right: float, count: int) -> np.ndarray:
    spacing = (right - left) / count
    return np.linspace(left + 0.5 * spacing, right - 0.5 * spacing, count)


def _finite_limit(values: np.ndarray, percentile: float, floor: float) -> float:
    finite = np.asarray(values)[np.isfinite(values)]
    return max(floor, float(np.percentile(np.abs(finite), percentile)))


def _slice_time(metadata: dict[str, object]) -> float:
    if "simulation_time_s" in metadata:
        return float(metadata["simulation_time_s"])
    match = re.search(r"([0-9]{5})$", str(metadata["plotfile"]))
    if match is None:
        raise KeyError("slice metadata has neither simulation_time_s nor a step suffix")
    return 0.003 * int(match.group(1))


def _sphere_and_roi(axis: plt.Axes, transverse_coordinate: str) -> None:
    axis.add_patch(plt.Circle((0.75, 0.75), 0.15, fill=False, color="black", linewidth=1.1))
    axis.axvline(1.0, color="black", linestyle=":", linewidth=0.9, alpha=0.8)
    axis.axvline(7.05, color="black", linestyle=":", linewidth=0.9, alpha=0.8)
    axis.set_xlabel("z (m), downstream")
    axis.set_ylabel(f"{transverse_coordinate} (m)")


def _save_overview(
    output: Path,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    sphere_velocity: np.ndarray,
    delta: np.ndarray,
    fluid: np.ndarray,
    time_s: float,
) -> None:
    ix = int(np.argmin(np.abs(x - 0.75)))
    iy = int(np.argmin(np.abs(y - 0.75)))
    center_y_fluid = fluid[:, iy, :]
    center_x_fluid = fluid[ix, :, :]
    total_uz = np.where(center_y_fluid, sphere_velocity[2, :, iy, :], np.nan)
    dux = np.where(center_y_fluid, delta[0, :, iy, :], np.nan)
    duy = np.where(center_x_fluid, delta[1, ix, :, :], np.nan)
    duz = np.where(center_y_fluid, delta[2, :, iy, :], np.nan)
    magnitude = np.where(
        center_y_fluid,
        np.sqrt(np.sum(delta[:, :, iy, :] ** 2, axis=0)),
        np.nan,
    )
    dx = float(x[1] - x[0])
    dz = float(z[1] - z[0])
    omega_y = np.gradient(dux, dz, axis=1) - np.gradient(duz, dx, axis=0)
    omega_y = np.where(center_y_fluid, omega_y, np.nan)

    transverse_limit = _finite_limit(np.concatenate([dux.ravel(), duy.ravel()]), 99.5, 0.03)
    axial_limit = _finite_limit(duz, 99.5, 0.06)
    vorticity_limit = _finite_limit(omega_y, 99.2, 0.5)
    magnitude_limit = _finite_limit(magnitude, 99.5, 0.06)
    total_low = float(np.nanpercentile(total_uz, 0.5))
    total_high = float(np.nanpercentile(total_uz, 99.5))
    panels = [
        (total_uz, r"Total $U_z$ on $y=0.75$ m", "viridis", total_low, total_high, "x"),
        (dux, r"Wake $\Delta U_x$ on $y=0.75$ m", "RdBu_r", -transverse_limit, transverse_limit, "x"),
        (duy, r"Wake $\Delta U_y$ on $x=0.75$ m", "RdBu_r", -transverse_limit, transverse_limit, "y"),
        (duz, r"Wake $\Delta U_z$ on $y=0.75$ m", "RdBu_r", -axial_limit, axial_limit, "x"),
        (magnitude, r"Wake $|\Delta U|$ on $y=0.75$ m", "magma", 0.0, magnitude_limit, "x"),
        (omega_y, r"Wake vorticity $\omega_y$ on $y=0.75$ m", "RdBu_r", -vorticity_limit, vorticity_limit, "x"),
    ]
    figure, axes = plt.subplots(3, 2, figsize=(15.5, 9.5), constrained_layout=True)
    figure.suptitle(
        "incflo strict-double sphere wake relative to the matched no-sphere baseline\n"
        f"grid 144×144×768, t={time_s:.3f} s; dotted lines mark acoustic ROI z=1.00–7.05 m"
    )
    extent = [z[0], z[-1], x[0], x[-1]]
    for axis, (field, title, cmap, low, high, coordinate) in zip(axes.flat, panels):
        image = axis.imshow(
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
        _sphere_and_roi(axis, coordinate)
        figure.colorbar(image, ax=axis, label="s⁻¹" if "vorticity" in title else "m/s", pad=0.01)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _save_cross_sections(
    output: Path,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    delta_uz: np.ndarray,
    fluid: np.ndarray,
    time_s: float,
) -> None:
    requested = np.asarray([1.05, 1.80, 2.55, 3.30, 4.80, 6.30])
    indices = [int(np.argmin(np.abs(z - value))) for value in requested]
    planes = [np.where(fluid[:, :, index], delta_uz[:, :, index], np.nan) for index in indices]
    limit = _finite_limit(np.stack(planes), 99.0, 0.035)
    figure, axes = plt.subplots(2, 3, figsize=(12.5, 8.2), constrained_layout=True)
    figure.suptitle(
        f"Matched incflo axial wake deficit across downstream x–y sections, t={time_s:.3f} s"
    )
    image = None
    for axis, plane, index in zip(axes.flat, planes, indices):
        image = axis.imshow(
            plane.T,
            origin="lower",
            extent=[x[0], x[-1], y[0], y[-1]],
            aspect="equal",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="bilinear",
        )
        axis.set_title(f"z={z[index]:.3f} m")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
    assert image is not None
    figure.colorbar(image, ax=axes, label=r"$\Delta U_z$ (m/s)", shrink=0.78, pad=0.01)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _save_profiles(
    output: Path,
    csv_output: Path,
    z: np.ndarray,
    delta: np.ndarray,
    fluid: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    time_s: float,
) -> dict[str, object]:
    profiles = []
    for component in range(3):
        values = np.where(fluid, delta[component], np.nan)
        profiles.append(np.sqrt(np.nanmean(values**2, axis=(0, 1))))
    profiles_array = np.stack(profiles)
    ix = int(np.argmin(np.abs(x - 0.75)))
    iy = int(np.argmin(np.abs(y - 0.75)))
    centerline = np.nanmean(delta[2, ix - 1 : ix + 2, iy - 1 : iy + 2, :], axis=(0, 1))
    keep = (z >= 1.0) & (z <= 7.05)

    figure, axes = plt.subplots(2, 1, figsize=(11.5, 7.0), sharex=True, constrained_layout=True)
    for component, label in enumerate((r"RMS($\Delta U_x$)", r"RMS($\Delta U_y$)", r"RMS($\Delta U_z$)")):
        axes[0].plot(z[keep], profiles_array[component, keep], linewidth=1.8, label=label)
    axes[0].set_ylabel("Cross-sectional RMS (m/s)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=3)
    axes[1].plot(z[keep], centerline[keep], color="tab:blue", linewidth=1.8)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("z (m), downstream")
    axes[1].set_ylabel(r"Centerline $\Delta U_z$ (m/s)")
    axes[1].grid(alpha=0.25)
    figure.suptitle(f"Axial persistence of the matched incflo sphere wake, t={time_s:.3f} s")
    figure.savefig(output, dpi=180)
    plt.close(figure)

    with csv_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["z_m", "ux_rms_m_s", "uy_rms_m_s", "uz_rms_m_s", "centerline_duz_m_s"])
        for index in np.flatnonzero(keep):
            writer.writerow([z[index], *profiles_array[:, index], centerline[index]])
    return {
        "roi_z_m": [1.0, 7.05],
        "component_rms_mean_m_s": np.nanmean(profiles_array[:, keep], axis=1).tolist(),
        "component_rms_max_m_s": np.nanmax(profiles_array[:, keep], axis=1).tolist(),
        "centerline_duz_min_m_s": float(np.nanmin(centerline[keep])),
        "centerline_duz_max_m_s": float(np.nanmax(centerline[keep])),
    }


def _save_evolution(slice_metadata: list[Path], montage: Path, gif: Path) -> None:
    records = [_load(path) for path in slice_metadata]
    all_fields = []
    for metadata, fields in records:
        field = fields["velz"].T - 0.5
        field = np.ma.masked_where(fields["vfrac"].T <= 0.0, field)
        all_fields.append(field)
    limit = max(0.08, float(np.percentile(np.abs(np.concatenate([field.compressed() for field in all_fields])), 99.3)))
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 6.8), constrained_layout=True)
    figure.suptitle(r"incflo wake development on $y=0.75$ m: raw $U_z-U_\infty$")
    image = None
    for axis, (metadata, _), field in zip(axes.flat, records, all_fields):
        left, right = metadata["domain_left_m"], metadata["domain_right_m"]
        image = axis.imshow(
            field,
            origin="lower",
            extent=[left[2], right[2], left[0], right[0]],
            aspect="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="bilinear",
        )
        axis.set_title(f"t={_slice_time(metadata):.3f} s")
        axis.set_xlabel("z (m)")
        axis.set_ylabel("x (m)")
        axis.add_patch(plt.Circle((0.75, 0.75), 0.15, fill=False, color="black", linewidth=1.0))
    assert image is not None
    figure.colorbar(image, ax=axes, label="m/s", shrink=0.8, pad=0.01)
    figure.savefig(montage, dpi=180)
    plt.close(figure)

    frames = []
    for (metadata, _), field in zip(records, all_fields):
        figure, axis = plt.subplots(figsize=(11.5, 3.1), constrained_layout=True)
        left, right = metadata["domain_left_m"], metadata["domain_right_m"]
        image = axis.imshow(
            field,
            origin="lower",
            extent=[left[2], right[2], left[0], right[0]],
            aspect="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="bilinear",
        )
        axis.set(title=f"incflo raw axial wake, t={_slice_time(metadata):.3f} s", xlabel="z (m)", ylabel="x (m)")
        axis.add_patch(plt.Circle((0.75, 0.75), 0.15, fill=False, color="black", linewidth=1.0))
        figure.colorbar(image, ax=axis, label=r"$U_z-U_\infty$ (m/s)", pad=0.01)
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=115)
        plt.close(figure)
        buffer.seek(0)
        frames.append(Image.open(buffer).convert("RGB"))
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=900, loop=0, optimize=True)


def _jpeg_data_uri(path: Path, width: int = 1000, quality: int = 58) -> str:
    with Image.open(path) as source:
        image = source.convert("RGB")
        if image.width > width:
            height = round(image.height * width / image.width)
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _write_inline_html(output: Path, figures: list[tuple[str, str, Path]]) -> None:
    items = [
        {"label": label, "alt": alt, "src": _jpeg_data_uri(path)}
        for label, alt, path in figures
    ]
    payload = json.dumps(items, ensure_ascii=False)
    fragment = f'''<section id="incflo-flow-viz-20260824">
  <h2>incflo 球体尾流结果</h2>
  <div class="viz-controls" aria-label="选择流场视图">
    {''.join(f'<button type="button" class="btn btn-ghost" aria-pressed="{str(index == 0).lower()}" data-view="{index}">{item["label"]}</button>' for index, item in enumerate(items))}
  </div>
  <figure>
    <img data-main-image src="{items[0]['src']}" alt="{items[0]['alt']}" style="display:block;width:100%;height:auto">
    <figcaption class="text-small text-muted" data-caption>{items[0]['alt']}</figcaption>
  </figure>
  <script>
  (() => {{
    const root = document.getElementById('incflo-flow-viz-20260824');
    const views = {payload};
    const image = root.querySelector('[data-main-image]');
    const caption = root.querySelector('[data-caption]');
    root.querySelectorAll('button[data-view]').forEach(button => {{
      button.addEventListener('click', () => {{
        const index = Number(button.dataset.view);
        image.src = views[index].src;
        image.alt = views[index].alt;
        caption.textContent = views[index].alt;
        root.querySelectorAll('button[data-view]').forEach(peer => peer.setAttribute('aria-pressed', String(peer === button)));
      }});
    }});
  }})();
  </script>
</section>
'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(fragment, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sphere", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--slice-metadata", nargs="*", type=Path, default=[])
    parser.add_argument("--inline-html", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.sphere, "r") as sphere, h5py.File(args.baseline, "r") as baseline:
        sphere_velocity = sphere["velocity"][:]
        baseline_velocity = baseline["velocity"][:]
        sphere_time = float(sphere.attrs.get("time_s", sphere.attrs.get("simulation_time_s")))
        baseline_time = float(baseline.attrs.get("simulation_time_s", baseline.attrs.get("time_s")))
        if sphere_velocity.shape != baseline_velocity.shape or abs(sphere_time - baseline_time) > 1.0e-9:
            raise ValueError("sphere and baseline are not time/grid matched")
        left = np.asarray(sphere.attrs["domain_left_m"], dtype=float)
        right = np.asarray(sphere.attrs["domain_right_m"], dtype=float)
        grid = np.asarray(sphere.attrs["grid"], dtype=int)
        volume_key = "vfrac" if "vfrac" in sphere else "volume_fraction"
        fluid = sphere[volume_key][:] > 0.5
    delta = sphere_velocity - baseline_velocity
    x, y, z = (_centers(left[index], right[index], int(grid[index])) for index in range(3))

    overview = output / "incflo_matched_wake_overview_t30p969.png"
    sections = output / "incflo_matched_wake_cross_sections_t30p969.png"
    profiles = output / "incflo_matched_wake_axial_profiles_t30p969.png"
    profile_csv = output / "incflo_matched_wake_axial_profiles_t30p969.csv"
    _save_overview(overview, x, y, z, sphere_velocity, delta, fluid, sphere_time)
    _save_cross_sections(sections, x, y, z, delta[2], fluid, sphere_time)
    metrics = _save_profiles(profiles, profile_csv, z, delta, fluid, x, y, sphere_time)
    figures = [
        ("总览", "匹配无球基线后的三分量尾流、速度模和涡量总览。", overview),
        ("横截面", "六个下游 x-y 截面上的轴向尾流扰动。", sections),
        ("轴向曲线", "三分量横截面 RMS 与中心线轴向速度亏损。", profiles),
    ]
    if args.slice_metadata:
        montage = output / "incflo_wake_time_evolution.png"
        gif = output / "incflo_wake_time_evolution.gif"
        _save_evolution(args.slice_metadata, montage, gif)
        figures.append(("时间演化", "四个保存时刻的原始轴向尾流发展。", montage))
    else:
        montage = None
        gif = None
    report = {
        "sphere": str(args.sphere.resolve()),
        "baseline": str(args.baseline.resolve()),
        "time_s": sphere_time,
        "grid": grid.tolist(),
        "domain_left_m": left.tolist(),
        "domain_right_m": right.tolist(),
        "matched": True,
        "metrics": metrics,
        "figures": [str(value[2]) for value in figures],
        "gif": str(gif) if gif is not None else None,
    }
    (output / "visualization_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.inline_html is not None:
        _write_inline_html(args.inline_html.resolve(), figures)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
