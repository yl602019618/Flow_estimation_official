from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional

from .acquisition import build_acquisition_3d
from .config import DemoConfig3D
from .operators import relative_norm
from .travel_time import build_ray_matrix, extract_reciprocal_delays_3d


def _straight_ray_delays(velocity: torch.Tensor, coordinates: torch.Tensor,
                         pairs: torch.Tensor, config: DemoConfig3D) -> torch.Tensor:
    values = []
    uz = velocity[2][None, None]
    for a, b in pairs.tolist():
        start, end = coordinates[a], coordinates[b]
        alpha = torch.linspace(0.0, 1.0, 256, dtype=velocity.dtype, device=velocity.device)
        points = start[None] + alpha[:, None] * (end - start)[None]
        # grid_sample coordinates address W,H,D; velocity is stored as x,y,z.
        sample_grid = torch.stack((
            2 * points[:, 2] / config.grid.length_z - 1,
            2 * points[:, 1] / config.grid.length_y - 1,
            2 * points[:, 0] / config.grid.length_x - 1,
        ), dim=-1).reshape(1, 1, 1, -1, 3)
        mean_uz = functional.grid_sample(uz, sample_grid, mode="bilinear",
            padding_mode="border", align_corners=True).mean()
        values.append(-2.0 * (end[2] - start[2]) * mean_uz / config.physics.c0**2)
    return torch.stack(values)


def validate_matched_clean(config: DemoConfig3D, data_path: str | Path) -> dict[str, object]:
    data_path = Path(data_path)
    with h5py.File(data_path, "r") as handle:
        moving = torch.from_numpy(np.asarray(handle["pressure_traces"])).to(torch.float64)
        static = torch.from_numpy(np.asarray(handle["static_pressure_traces"])).to(torch.float64)
        velocity = torch.from_numpy(np.asarray(handle["truth_velocity"])).to(torch.float64)
        noise = torch.from_numpy(np.asarray(handle["noise"])).to(torch.float64)
        truth_origin = str(handle.attrs["truth_origin"])
    acquisition = build_acquisition_3d(config.grid, config.acquisition, dtype=torch.float64)
    delay = extract_reciprocal_delays_3d(moving, static, acquisition)
    theory = _straight_ray_delays(velocity, acquisition.coordinates, delay.pairs, config)
    axial = theory.abs() > torch.finfo(theory.dtype).eps
    transverse = ~axial
    reciprocal_error = torch.stack([
        static[a, b] - static[b, a] for a, b in delay.pairs.tolist()
    ])
    reciprocal_reference = torch.stack([
        static[a, b] for a, b in delay.pairs.tolist()
    ])
    correlation = float(torch.corrcoef(torch.stack((delay.delays[axial], theory[axial])))[0, 1])
    relative_theory_error = relative_norm(delay.delays[axial] - theory[axial], theory[axial])
    sign_fraction = float((torch.sign(delay.delays[axial]) == torch.sign(theory[axial])).to(torch.float64).mean())

    ray_matrix = build_ray_matrix(acquisition, config.inversion.vector_control,
        (config.grid.length_x, config.grid.length_y, config.grid.length_z), config.physics.c0)
    singular = torch.linalg.svdvals(ray_matrix)
    threshold = singular[0] * max(ray_matrix.shape) * torch.finfo(singular.dtype).eps
    rank = int((singular > threshold).sum())
    condition = float(singular[0] / singular[rank - 1]) if rank else float("inf")

    output = data_path.parent
    with h5py.File(output / "reciprocal_delays.h5", "w") as handle:
        handle.create_dataset("measured_delay_s", data=delay.delays.numpy())
        handle.create_dataset("straight_ray_delay_s", data=theory.numpy())
        handle.create_dataset("quality", data=delay.quality.numpy())
        handle.create_dataset("pairs", data=delay.pairs.numpy())
        handle.create_dataset("ray_singular_values", data=singular.numpy())

    representative = int(torch.argmax(delay.delays.abs()))
    a, b = delay.pairs[representative].tolist()
    time_us = np.arange(config.grid.nt) * config.grid.dt * 1e6
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    image = axes[0, 0].imshow(velocity[2, :, :, config.grid.nz // 2].T.numpy(), origin="lower",
        extent=(0, config.grid.length_x, 0, config.grid.length_y), cmap="turbo", vmin=0, vmax=1)
    axes[0, 0].set(title="MAC/RK2 solver axial velocity $U_z$", xlabel="x [m]", ylabel="y [m]")
    figure.colorbar(image, ax=axes[0, 0], label="m/s")
    axes[0, 1].plot(time_us, moving[a, b].numpy(), label=f"{a} to {b}")
    axes[0, 1].plot(time_us, moving[b, a].numpy(), label=f"{b} to {a}", alpha=.8)
    axes[0, 1].set(xlim=(60, 180), title=f"Representative reciprocal waveforms, delay={float(delay.delays[representative]*1e9):.2f} ns",
                   xlabel="time [us]", ylabel="pressure")
    axes[0, 1].legend()
    axes[1, 0].scatter(theory[axial].numpy() * 1e9, delay.delays[axial].numpy() * 1e9, s=18, alpha=.75)
    limit = float(max(theory[axial].abs().max(), delay.delays[axial].abs().max()) * 1e9) * 1.05
    axes[1, 0].plot((-limit, limit), (-limit, limit), "k--", label="straight-ray equality")
    axes[1, 0].set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal",
                   title=f"Measured vs straight-ray delay, r={correlation:.3f}",
                   xlabel="straight-ray delay [ns]", ylabel="measured delay [ns]")
    axes[1, 0].legend()
    colors = torch.sign(acquisition.coordinates[delay.pairs[:, 1], 2] - acquisition.coordinates[delay.pairs[:, 0], 2]).numpy()
    axes[1, 1].scatter(np.arange(delay.delays.numel()), delay.delays.numpy() * 1e9,
                       c=colors, cmap="coolwarm", s=12)
    axes[1, 1].axhline(0, color="k", linewidth=.8)
    axes[1, 1].set(title="All reciprocal direct-wave delays", xlabel="pair index", ylabel="delay [ns]")
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"matched_clean_flow_delay_diagnostics.{suffix}", dpi=200)
    plt.close(figure)

    metrics: dict[str, object] = {
        "passed": bool(torch.isfinite(moving).all() and torch.isfinite(static).all()
                       and sign_fraction == 1.0 and correlation >= 0.9),
        "truth_origin": truth_origin,
        "trace_shape": list(moving.shape),
        "ordered_traces": int(acquisition.ordered_mask.sum()),
        "finite": bool(torch.isfinite(moving).all() and torch.isfinite(static).all() and torch.isfinite(velocity).all()),
        "noise_max_abs": float(noise.abs().max()),
        "ux_max_abs_m_s": float(velocity[0].abs().max()),
        "uy_max_abs_m_s": float(velocity[1].abs().max()),
        "uz_min_m_s": float(velocity[2].min()),
        "uz_max_m_s": float(velocity[2].max()),
        "static_reciprocity_error": relative_norm(reciprocal_error, reciprocal_reference),
        "axial_delay_sign_fraction": sign_fraction,
        "axial_delay_correlation_with_straight_ray": correlation,
        "axial_delay_relative_straight_ray_error": relative_theory_error,
        "axial_delay_median_abs_ns": float(delay.delays[axial].abs().median() * 1e9),
        "same_ring_delay_median_abs_ns": float(delay.delays[transverse].abs().median() * 1e9),
        "delay_quality_min": float(delay.quality.min()),
        "delay_quality_median": float(delay.quality.median()),
        "ray_matrix_shape": list(ray_matrix.shape),
        "ray_effective_rank": rank,
        "ray_condition_number_nonzero": condition,
        "artifacts": [str(output / "reciprocal_delays.h5"),
                      str(output / "matched_clean_flow_delay_diagnostics.png"),
                      str(output / "matched_clean_flow_delay_diagnostics.pdf")],
    }
    with (output / "validation_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    return metrics
