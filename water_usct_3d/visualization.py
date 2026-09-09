from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import h5py
import torch

from .config import Grid3D
from .flow import FlowResult
from .geometry import CoverageResult
from .operators import divergence


def save_geometry(coordinates: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(7, 6)); axis = figure.add_subplot(111, projection="3d")
    axis.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], c=coordinates[:, 2], cmap="viridis")
    for index, point in enumerate(coordinates): axis.text(*point, str(index), fontsize=6)
    axis.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
    figure.tight_layout(); figure.savefig(destination, dpi=180); plt.close(figure)


def save_coverage_slices(result: CoverageResult, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    value = result.normalized_min_eigenvalue.detach().cpu().numpy()
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    slices = (value[value.shape[0]//2], value[:, value.shape[1]//2], value[..., value.shape[2]//2])
    for axis, image, title in zip(axes, slices, ("x mid", "y mid", "z mid")):
        plot = axis.imshow(image.T, origin="lower", cmap="magma"); axis.set_title(title); figure.colorbar(plot, ax=axis)
    figure.tight_layout(); figure.savefig(destination, dpi=180); plt.close(figure)


def _save_both(figure: plt.Figure, base: Path) -> None:
    figure.tight_layout()
    figure.savefig(base.with_suffix(".png"), dpi=200)
    figure.savefig(base.with_suffix(".pdf"))
    plt.close(figure)


def save_profile_diagnostics(truth: np.ndarray, recovered: np.ndarray,
                             control: np.ndarray, candidates: list[dict[str, object]],
                             grid: Grid3D, output: Path) -> list[str]:
    """Plot the strictly axial, z-invariant profile inversion result."""
    output.mkdir(parents=True, exist_ok=True)
    iz = grid.nz // 2
    truth_uz = np.asarray(truth)[2, :, :, iz]
    recovered_uz = np.asarray(recovered)[2, :, :, iz]
    error = recovered_uz - truth_uz
    extent = (0.0, grid.length_x, 0.0, grid.length_y)
    common_max = max(float(np.max(np.abs(truth_uz))), float(np.max(np.abs(recovered_uz))), 1e-12)
    error_max = max(float(np.max(np.abs(error))), 1e-12)

    figure, axes = plt.subplots(2, 2, figsize=(11, 9))
    for axis, value, title, cmap, lower, upper in (
        (axes[0, 0], truth_uz, r"MAC/RK2 truth $U_z$", "viridis", 0.0, common_max),
        (axes[0, 1], recovered_uz, r"Recovered $U_z(x,y)$", "viridis", 0.0, common_max),
        (axes[1, 0], error, r"Recovered - truth $U_z$", "coolwarm", -error_max, error_max),
    ):
        image = axis.imshow(value.T, origin="lower", extent=extent, aspect="equal",
                            cmap=cmap, vmin=lower, vmax=upper)
        axis.set(xlabel="x [m]", ylabel="y [m]", title=title)
        figure.colorbar(image, ax=axis, label="m/s")
    x = np.linspace(0.0, grid.length_x, grid.nx)
    iy = grid.ny // 2
    axes[1, 1].plot(x, truth_uz[:, iy], "k-", lw=2, label="MAC/RK2 truth")
    axes[1, 1].plot(x, recovered_uz[:, iy], "o-", ms=3, fillstyle="none", label="recovered")
    axes[1, 1].scatter(np.linspace(0.0, grid.length_x, control.shape[0]), control[:, control.shape[1] // 2],
                       marker="x", color="tab:red", label="6x6 controls")
    axes[1, 1].set(xlabel="x [m]", ylabel=r"$U_z$ [m/s]", title="Centreline profile")
    axes[1, 1].grid(alpha=.3); axes[1, 1].legend()
    base = output / "profile_inversion_diagnostics"
    _save_both(figure, base)

    names = [str(item["name"]) for item in candidates]
    wave = np.asarray([float(item["waveform_loss"]) for item in candidates])
    delay = np.asarray([float(item["delay_relative_residual"]) for item in candidates])
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, wave); axes[0].set_yscale("log")
    axes[0].set(ylabel="four-band waveform loss", title="Nonlinear selection criterion")
    axes[1].bar(names, delay)
    axes[1].set(ylabel="relative delay residual", title="Travel-time diagnostic")
    for axis in axes: axis.tick_params(axis="x", rotation=12); axis.grid(axis="y", alpha=.3)
    candidate_base = output / "profile_candidate_comparison"
    _save_both(figure, candidate_base)
    return [str(base.with_suffix(".png")), str(base.with_suffix(".pdf")),
            str(candidate_base.with_suffix(".png")), str(candidate_base.with_suffix(".pdf"))]


def save_axial_fwi_diagnostics(truth: np.ndarray, initial: np.ndarray,
                               final: np.ndarray, history: list[dict[str, float]],
                               grid: Grid3D, output: Path) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    iz = grid.nz // 2
    fields = (truth[2, :, :, iz], initial[2, :, :, iz], final[2, :, :, iz])
    common_max = max(float(np.max(np.abs(value))) for value in fields)
    error = fields[2] - fields[0]
    error_max = max(float(np.max(np.abs(error))), 1e-12)
    extent = (0.0, grid.length_x, 0.0, grid.length_y)
    figure, axes = plt.subplots(2, 2, figsize=(11, 9))
    for axis, value, title in zip(axes.ravel()[:3], fields,
                                  ("MAC/RK2 truth $U_z$", "FWI initial $U_z$", "FWI final $U_z$")):
        image = axis.imshow(value.T, origin="lower", extent=extent, aspect="equal",
                            cmap="viridis", vmin=0.0, vmax=common_max)
        axis.set(title=title, xlabel="x [m]", ylabel="y [m]")
        figure.colorbar(image, ax=axis, label="m/s")
    image = axes[1, 1].imshow(error.T, origin="lower", extent=extent, aspect="equal",
                              cmap="coolwarm", vmin=-error_max, vmax=error_max)
    axes[1, 1].set(title="FWI final - truth", xlabel="x [m]", ylabel="y [m]")
    figure.colorbar(image, ax=axes[1, 1], label="m/s")
    base = output / "axial_fwi_models"
    _save_both(figure, base)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    iteration = np.arange(len(history))
    labels = [f"{item['frequency_hz']/1000:.0f}" for item in history]
    axes[0].bar(iteration, [item["normalized_loss"] for item in history])
    axes[0].axhline(100.0, color="k", ls="--", label="band entry")
    axes[0].set_xticks(iteration, labels)
    axes[0].set(xlabel="low-pass cutoff [kHz]", ylabel="accepted objective (entry=100)",
                title="Per-band true FWI improvement")
    axes[0].legend()
    axes[0].grid(alpha=.3)
    axes[1].plot(iteration, [item["max_velocity_update"] for item in history], "o-")
    axes[1].axhline(.05, color="r", ls="--", label="trust limit")
    axes[1].set_xticks(iteration, labels)
    axes[1].set(xlabel="low-pass cutoff [kHz]", ylabel="max update [m/s]",
                title="Accepted velocity updates")
    axes[1].grid(alpha=.3); axes[1].legend()
    history_base = output / "axial_fwi_history"
    _save_both(figure, history_base)
    return [str(base.with_suffix(".png")), str(base.with_suffix(".pdf")),
            str(history_base.with_suffix(".png")), str(history_base.with_suffix(".pdf"))]


def save_axial_initial_model_diagnostics(truth: np.ndarray, initial: np.ndarray,
                                         control: np.ndarray, grid: Grid3D,
                                         output: Path) -> list[str]:
    """Visualize the z-invariant axial model used to initialize waveform FWI."""
    output.mkdir(parents=True, exist_ok=True)
    truth = np.asarray(truth)
    initial = np.asarray(initial)
    control = np.asarray(control).copy()
    # The profile parameterization fixes the transverse-wall controls to zero.
    control[[0, -1], :] = 0.0
    control[:, [0, -1]] = 0.0

    iz = grid.nz // 2
    iy = grid.ny // 2
    truth_uz = truth[2, :, :, iz]
    initial_uz = initial[2, :, :, iz]
    error = initial_uz - truth_uz
    common_max = max(float(np.max(np.abs(truth_uz))),
                     float(np.max(np.abs(initial_uz))), 1e-12)
    error_max = max(float(np.max(np.abs(error))), 1e-12)
    extent_xy = (0.0, grid.length_x, 0.0, grid.length_y)
    extent_xz = (0.0, grid.length_x, 0.0, grid.length_z)
    x = np.linspace(0.0, grid.length_x, grid.nx)
    control_x = np.linspace(0.0, grid.length_x, control.shape[0])

    figure, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    image = axes[0, 0].imshow(initial_uz.T, origin="lower", extent=extent_xy,
                              aspect="equal", cmap="viridis", vmin=0.0,
                              vmax=common_max)
    axes[0, 0].set(title=r"FWI initial $U_z(x,y)$", xlabel="x [m]", ylabel="y [m]")
    figure.colorbar(image, ax=axes[0, 0], label="m/s")

    image = axes[0, 1].imshow(control.T, origin="lower", extent=extent_xy,
                              aspect="equal", cmap="viridis", vmin=0.0,
                              vmax=common_max)
    axes[0, 1].set(title=r"Constrained $6\times6$ controls", xlabel="x [m]", ylabel="y [m]")
    control_y = np.linspace(0.0, grid.length_y, control.shape[1])
    for ix, xpos in enumerate(control_x[1:-1], start=1):
        for jy, ypos in enumerate(control_y[1:-1], start=1):
            value = float(control[ix, jy])
            axes[0, 1].text(xpos, ypos, f"{value:.2f}", ha="center", va="center",
                            fontsize=7, color="white" if value > .55 * common_max else "black")
    figure.colorbar(image, ax=axes[0, 1], label="m/s")

    image = axes[0, 2].imshow(error.T, origin="lower", extent=extent_xy,
                              aspect="equal", cmap="coolwarm", vmin=-error_max,
                              vmax=error_max)
    axes[0, 2].set(title=r"Initial - MAC/RK2 truth $U_z$", xlabel="x [m]", ylabel="y [m]")
    figure.colorbar(image, ax=axes[0, 2], label="m/s")

    axes[1, 0].plot(x, truth_uz[:, iy], "k-", lw=2, label="MAC/RK2 truth")
    axes[1, 0].plot(x, initial_uz[:, iy], color="tab:blue", lw=2,
                    label="FWI initial")
    axes[1, 0].scatter(control_x, control[:, control.shape[1] // 2], marker="x",
                       s=42, color="tab:red", label="control values", zorder=3)
    axes[1, 0].set(title="Centreline axial profile", xlabel="x [m]", ylabel=r"$U_z$ [m/s]")
    axes[1, 0].grid(alpha=.3); axes[1, 0].legend()

    xz = initial[2, :, iy, :].T
    image = axes[1, 1].imshow(xz, origin="lower", extent=extent_xz, aspect="auto",
                              cmap="viridis", vmin=0.0, vmax=common_max)
    axes[1, 1].set(title=r"Centre-plane $U_z(x,z)$: z-invariant",
                   xlabel="x [m]", ylabel="z [m]")
    figure.colorbar(image, ax=axes[1, 1], label="m/s")

    relative_error = float(np.linalg.norm(initial - truth) /
                           max(np.linalg.norm(truth), np.finfo(float).tiny))
    correlation = float(np.dot(initial.ravel(), truth.ravel()) /
                        max(np.linalg.norm(initial) * np.linalg.norm(truth),
                            np.finfo(float).tiny))
    z_variation = float(np.max(np.abs(np.diff(initial, axis=-1))))
    transverse_max = float(np.max(np.abs(initial[:2])))
    wall_max = max(float(np.max(np.abs(initial[:, 0]))),
                   float(np.max(np.abs(initial[:, -1]))),
                   float(np.max(np.abs(initial[:, :, 0]))),
                   float(np.max(np.abs(initial[:, :, -1]))))
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.02, 0.96,
        "Initial-model definition\n\n"
        r"$\mathbf{U}_0=(0,0,U_z(x,y))$" "\n"
        r"$U_z$ is constant along z" "\n"
        r"transverse walls are fixed to zero" "\n\n"
        f"relative L2 error: {relative_error:.3%}\n"
        f"vector correlation: {correlation:.6f}\n"
        f"max |Ux, Uy|: {transverse_max:.2e} m/s\n"
        f"max z variation: {z_variation:.2e} m/s\n"
        f"max wall speed: {wall_max:.2e} m/s",
        transform=axes[1, 2].transAxes, va="top", fontsize=11,
        bbox={"boxstyle": "round,pad=.6", "facecolor": "#f3f5f7", "edgecolor": "#9aa0a6"},
    )
    figure.suptitle("Axial waveform-FWI initial model", fontsize=15)
    base = output / "axial_fwi_initial_model"
    _save_both(figure, base)
    return [str(base.with_suffix(".png")), str(base.with_suffix(".pdf"))]


def save_axial_initialization_ablation(controls: dict[str, np.ndarray],
                                       truth: np.ndarray, grid: Grid3D,
                                       output: Path) -> list[str]:
    """Compare the four registered axial-FWI initializations without selecting one."""
    output.mkdir(parents=True, exist_ok=True)
    ordered = ("good_prior", "travel_time", "blurred_travel_time", "zero_flow")
    titles = {
        "good_prior": "Good prior (upper bound)",
        "travel_time": "Direct reciprocal-delay",
        "blurred_travel_time": r"Delay + fixed Gaussian blur ($\sigma=0.75$)",
        "zero_flow": "Zero flow",
    }
    values = {name: np.asarray(controls[name]) for name in ordered}
    fields = {
        name: np.asarray(torch.nn.functional.interpolate(
            torch.as_tensor(value)[None, None], size=(grid.nx, grid.ny),
            mode="bilinear", align_corners=True)[0, 0])
        for name, value in values.items()
    }
    common_abs = max(max(float(np.max(np.abs(value))) for value in fields.values()), 1e-12)
    extent = (0.0, grid.length_x, 0.0, grid.length_y)
    truth_uz = np.asarray(truth)[2, :, :, grid.nz // 2]

    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for axis, name in zip(axes.ravel()[:4], ordered):
        image = axis.imshow(fields[name].T, origin="lower", extent=extent, aspect="equal",
                            cmap="coolwarm", vmin=-common_abs, vmax=common_abs)
        axis.set(title=titles[name], xlabel="x [m]", ylabel="y [m]")
        figure.colorbar(image, ax=axis, label="m/s")

    x = np.linspace(0.0, grid.length_x, grid.nx)
    iy = grid.ny // 2
    axes[1, 1].plot(x, truth_uz[:, iy], "k-", lw=2, label="MAC/RK2 truth (score only)")
    for name in ordered:
        axes[1, 1].plot(x, fields[name][:, iy], lw=1.5, label=name)
    axes[1, 1].set(title="Centreline profiles", xlabel="x [m]", ylabel=r"$U_z$ [m/s]")
    axes[1, 1].grid(alpha=.3); axes[1, 1].legend(fontsize=8)

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.02, 0.96,
        "Frozen ablation contract\n\n"
        "Observed data: matched_clean\n"
        "Bands: 20 / 35 / 50 / 65 kHz\n"
        "Same regularization and update limit\n"
        "Truth is used only for final scoring\n\n"
        "Primary test:\n"
        "final model error < initial model error\n\n"
        "No branch-specific tuning",
        transform=axes[1, 2].transAxes, va="top", fontsize=11,
        bbox={"boxstyle": "round,pad=.6", "facecolor": "#f3f5f7", "edgecolor": "#9aa0a6"},
    )
    figure.suptitle("Registered axial-FWI initialization ablation", fontsize=15)
    base = output / "initialization_ablation"
    _save_both(figure, base)
    return [str(base.with_suffix(".png")), str(base.with_suffix(".pdf"))]


def save_axial_fwi_ablation_summary(records: list[dict[str, object]],
                                     output: Path) -> list[str]:
    """Plot comparable initial/final model errors and waveform reductions."""
    output.mkdir(parents=True, exist_ok=True)
    names = [str(item["initialization"]) for item in records]
    initial = np.asarray([float(item["initial_profile_relative_l2"]) for item in records]) * 100.0
    final = np.asarray([float(item["final_profile_relative_l2"]) for item in records]) * 100.0
    reduction = np.asarray([float(item["waveform_loss_reduction"]) for item in records]) * 100.0
    x = np.arange(len(names))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    width = .36
    axes[0].bar(x - width / 2, initial, width, label="initial")
    axes[0].bar(x + width / 2, final, width, label="final")
    axes[0].axhline(10.0, color="r", ls="--", label="10% gate")
    axes[0].set_xticks(x, names, rotation=15)
    axes[0].set(ylabel="velocity relative L2 [%]", title="Same-budget model recovery")
    axes[0].grid(axis="y", alpha=.3); axes[0].legend()
    colors = ["tab:green" if bool(item["passed"]) else "tab:orange" for item in records]
    axes[1].bar(x, reduction, color=colors)
    axes[1].set_xticks(x, names, rotation=15)
    axes[1].set(ylabel="65 kHz waveform-loss reduction [%]",
                title="Waveform improvement (green = model gate pass)")
    axes[1].grid(axis="y", alpha=.3)
    base = output / "ablation_summary"
    _save_both(figure, base)
    return [str(base.with_suffix(".png")), str(base.with_suffix(".pdf"))]


def save_acoustic_grid_diagnostics(coarse: np.ndarray, fine: np.ndarray,
                                   reciprocal_difference: np.ndarray,
                                   reciprocal_reference: np.ndarray,
                                   dt: float, output: Path) -> dict[str, object]:
    """Save frozen coarse/fine and reciprocity diagnostics for one point-source shot."""
    output.mkdir(parents=True, exist_ok=True)
    coarse = np.asarray(coarse)[0]
    fine = np.asarray(fine)[0]
    channel_norm = np.linalg.norm(fine, axis=1)
    valid = channel_norm > np.finfo(fine.dtype).tiny
    # The formal comparison is a shot from transducer 0; exclude its collocated
    # self-channel because production acquisition records only the other 23.
    if valid.size:
        valid[0] = False
    mismatch = np.zeros_like(channel_norm)
    mismatch[valid] = np.linalg.norm(coarse[valid] - fine[valid], axis=1) / channel_norm[valid]
    coarse_energy = np.sum(coarse * coarse, axis=1)
    gain = np.ones_like(channel_norm)
    gain[coarse_energy > 0] = (np.sum(coarse * fine, axis=1)[coarse_energy > 0] /
                               coarse_energy[coarse_energy > 0])
    corrected = np.zeros_like(channel_norm)
    corrected[valid] = (np.linalg.norm(gain[valid, None] * coarse[valid] - fine[valid], axis=1) /
                        channel_norm[valid])
    candidates = np.flatnonzero(valid)
    receiver = int(candidates[np.argmax(channel_norm[candidates])])
    correlation = np.correlate(coarse[receiver], fine[receiver], mode="full")
    lag_samples = int(np.argmax(correlation) - (fine.shape[1] - 1))
    time_us = np.arange(fine.shape[1]) * dt * 1e6
    frequency_khz = np.fft.rfftfreq(fine.shape[1], dt) / 1e3
    coarse_spectrum = np.abs(np.fft.rfft(coarse[receiver]))
    fine_spectrum = np.abs(np.fft.rfft(fine[receiver]))

    with h5py.File(output / "point_source_grid_diagnostics.h5", "w") as handle:
        for name, value in {
            "coarse_lowpass_traces": coarse, "fine_lowpass_traces": fine,
            "per_receiver_relative_mismatch": mismatch,
            "per_receiver_amplitude_gain": gain,
            "per_receiver_gain_corrected_mismatch": corrected,
            "reciprocal_difference": reciprocal_difference,
            "reciprocal_reference": reciprocal_reference,
        }.items():
            handle.create_dataset(name, data=value, compression="gzip" if np.ndim(value) >= 2 else None)
        handle.attrs.update(dt=dt, representative_receiver=receiver, lag_samples=lag_samples,
                            transducer_model="trilinear point")

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(time_us, fine[receiver], label="fine 41x41x81", lw=1.5)
    axes[0, 0].plot(time_us, coarse[receiver], label="coarse 33x33x65", lw=1.0, alpha=.8)
    axes[0, 0].set(title=f"Point source, receiver {receiver}", xlabel="time [us]", ylabel="pressure")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=.3)
    axes[0, 1].plot(time_us, coarse[receiver] - fine[receiver])
    axes[0, 1].set(title=f"Residual (cross-correlation lag {lag_samples} samples)", xlabel="time [us]", ylabel="coarse - fine")
    axes[0, 1].grid(alpha=.3)
    axes[1, 0].semilogy(frequency_khz, fine_spectrum + 1e-30, label="fine")
    axes[1, 0].semilogy(frequency_khz, coarse_spectrum + 1e-30, label="coarse")
    axes[1, 0].axvline(65.0, color="k", ls="--", lw=1, label="65 kHz gate")
    axes[1, 0].set(xlim=(0, 100), xlabel="frequency [kHz]", ylabel="magnitude", title="Representative spectrum")
    axes[1, 0].legend(); axes[1, 0].grid(alpha=.3)
    indices = np.arange(mismatch.size)
    axes[1, 1].plot(indices[valid], mismatch[valid], "o-", label="raw")
    axes[1, 1].plot(indices[valid], corrected[valid], "s-", label="gain corrected")
    axes[1, 1].axhline(.05, color="r", ls="--", label="5% gate")
    axes[1, 1].set(xlabel="receiver index", ylabel="relative mismatch", title="Per-receiver mismatch")
    axes[1, 1].legend(); axes[1, 1].grid(alpha=.3)
    _save_both(figure, output / "point_source_coarse_fine_diagnostics")

    pair_norm = np.linalg.norm(reciprocal_reference, axis=1)
    pair_valid = pair_norm > np.finfo(reciprocal_reference.dtype).tiny
    pair_error = np.zeros_like(pair_norm)
    pair_error[pair_valid] = np.linalg.norm(reciprocal_difference[pair_valid], axis=1) / pair_norm[pair_valid]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.semilogy(np.flatnonzero(pair_valid), pair_error[pair_valid], ".")
    axis.axhline(1e-4, color="r", ls="--", label="registered 1e-4 gate")
    axis.set(xlabel="reciprocal-pair index", ylabel="relative waveform error", title="Point-source static reciprocity by pair")
    axis.grid(alpha=.3); axis.legend()
    _save_both(figure, output / "point_source_reciprocity_diagnostics")

    return {
        "representative_receiver": receiver,
        "representative_lag_samples": lag_samples,
        "median_receiver_mismatch": float(np.median(mismatch[valid])),
        "median_gain_corrected_mismatch": float(np.median(corrected[valid])),
        "median_amplitude_gain": float(np.median(gain[valid])),
        "diagnostic_h5": str(output / "point_source_grid_diagnostics.h5"),
        "diagnostic_figures": [str(output / "point_source_coarse_fine_diagnostics.png"),
                               str(output / "point_source_reciprocity_diagnostics.png")],
    }


def save_flow_artifacts(duct: FlowResult, manufactured_solver: FlowResult,
                        analytic_duct: np.ndarray, manufactured_reference: np.ndarray,
                        grid: Grid3D,
                        output: Path, manufactured_records: list[dict[str, float]] | None = None) -> list[str]:
    """Save inspectable flow fields, constraints, profiles, and histories."""
    output.mkdir(parents=True, exist_ok=True)
    velocity = duct.cell_velocity.detach().cpu().numpy()
    manufactured_velocity = manufactured_solver.cell_velocity.detach().cpu().numpy()
    manufactured_error = manufactured_velocity - manufactured_reference
    pressure = duct.pressure.detach().cpu().numpy()
    forcing = manufactured_solver.forcing.detach().cpu().numpy()
    div = divergence(duct.cell_velocity, (grid.dx, grid.dy, grid.dz), order=2).detach().cpu().numpy()
    flow_rate = velocity[2].sum(axis=(0, 1)) * grid.dx * grid.dy
    x = np.linspace(0, grid.length_x, grid.nx)
    y = np.linspace(0, grid.length_y, grid.ny)
    z = np.linspace(0, grid.length_z, grid.nz)
    artifacts: list[str] = []

    h5_path = output / "flow_fields.h5"
    with h5py.File(h5_path, "w") as handle:
        for name, value in {
            "node_velocity": velocity, "analytic_duct_velocity": analytic_duct,
            "manufactured_reference_velocity": manufactured_reference,
            "manufactured_solver_velocity": manufactured_velocity,
            "manufactured_solver_error": manufactured_error,
            "pressure_cells": pressure,
            "manufactured_solver_pressure_cells": manufactured_solver.pressure.detach().cpu().numpy(),
            "manufactured_solver_forcing": forcing,
            "node_divergence": div, "axial_flow_rate": flow_rate,
            "convergence": np.asarray(duct.convergence),
            "pcg_convergence": np.asarray(duct.pcg_convergence),
            "mac_u_faces": duct.face_velocity["u"].detach().cpu().numpy(),
            "mac_v_faces": duct.face_velocity["v"].detach().cpu().numpy(),
            "mac_w_faces": duct.face_velocity["w"].detach().cpu().numpy(),
            "manufactured_solver_mac_u_faces": manufactured_solver.face_velocity["u"].detach().cpu().numpy(),
            "manufactured_solver_mac_v_faces": manufactured_solver.face_velocity["v"].detach().cpu().numpy(),
            "manufactured_solver_mac_w_faces": manufactured_solver.face_velocity["w"].detach().cpu().numpy(),
        }.items():
            handle.create_dataset(name, data=value, compression="gzip" if np.ndim(value) >= 2 else None)
        handle.attrs.update(dx=grid.dx, dy=grid.dy, dz=grid.dz,
                            boundary="x/y no-slip, z periodic", solver="MAC RK2 projection PCG")
    artifacts.append(str(h5_path))

    iz = grid.nz // 2
    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    labels = (r"$U_x$ [m/s]", r"$U_y$ [m/s]", r"$U_z$ [m/s]", r"$|U|$ [m/s]")
    values = (velocity[0, :, :, iz], velocity[1, :, :, iz], velocity[2, :, :, iz],
              np.linalg.norm(velocity[:, :, :, iz], axis=0))
    for axis, value, label in zip(axes.ravel(), values, labels):
        limit = max(float(np.max(np.abs(value))), 1e-12)
        image = axis.imshow(value.T, origin="lower", extent=(0, grid.length_x, 0, grid.length_y), aspect="equal", cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set(xlabel="x [m]", ylabel="y [m]", title=label); figure.colorbar(image, ax=axis)
    _save_both(figure, output / "duct_velocity_components")

    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    iy = grid.ny // 2
    axis.plot(x, analytic_duct[2, :, iy, iz], "k-", lw=2, label="Fourier reference")
    axis.plot(x, velocity[2, :, iy, iz], "o", ms=3, fillstyle="none", label="MAC RK2")
    axis.set(xlabel="x [m]", ylabel=r"$U_z$ [m/s]", title="Square-duct centreline profile")
    axis.grid(alpha=.3); axis.legend()
    _save_both(figure, output / "duct_profile_comparison")

    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    div_limit = max(float(np.max(np.abs(div[:, :, iz]))), 1e-12)
    image = axes[0].imshow(div[:, :, iz].T, origin="lower", extent=(0, grid.length_x, 0, grid.length_y), cmap="RdBu_r", vmin=-div_limit, vmax=div_limit)
    axes[0].set(title=f"Nodal divergence (max={np.max(np.abs(div[:, :, iz])):.1e})", xlabel="x [m]", ylabel="y [m]"); figure.colorbar(image, ax=axes[0])
    pz = pressure.shape[2] // 2
    pressure_limit = max(float(np.max(np.abs(pressure[:, :, pz]))), 1e-12)
    image = axes[1].imshow(pressure[:, :, pz].T, origin="lower", extent=(0, grid.length_x, 0, grid.length_y), cmap="RdBu_r", vmin=-pressure_limit, vmax=pressure_limit)
    axes[1].set(title=f"Cell pressure (max={np.max(np.abs(pressure[:, :, pz])):.1e})", xlabel="x [m]", ylabel="y [m]"); figure.colorbar(image, ax=axes[1])
    wall_errors = [np.max(np.abs(velocity[:, 0])), np.max(np.abs(velocity[:, -1])),
                   np.max(np.abs(velocity[:, :, 0])), np.max(np.abs(velocity[:, :, -1]))]
    axes[2].bar(("x=0", "x=L", "y=0", "y=L"), wall_errors)
    axes[2].set(ylim=(0, max(max(wall_errors) * 1.2, 1e-12)), ylabel="max wall speed [m/s]", title="No-slip wall error")
    for index, value in enumerate(wall_errors):
        axes[2].text(index, max(value, 2e-14), f"{value:.1e}", ha="center")
    _save_both(figure, output / "duct_constraints")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(z, flow_rate); axes[0].set(xlabel="z [m]", ylabel=r"$Q$ [m$^3$/s]", title="Axial flow rate"); axes[0].grid(alpha=.3)
    axes[1].semilogy(np.arange(1, len(duct.convergence) + 1), duct.convergence)
    axes[1].axhline(1e-8, color="r", ls="--", label="registered threshold")
    axes[1].set(xlabel="RK2 step", ylabel="relative velocity change", title="Steady convergence"); axes[1].grid(alpha=.3); axes[1].legend()
    _save_both(figure, output / "duct_flow_rate_convergence")

    if manufactured_records:
        spacing = np.asarray([record["spacing"] for record in manufactured_records])
        error = np.asarray([record["relative_l2"] for record in manufactured_records])
        reference_second = error[-1] * (spacing / spacing[-1]) ** 2
        figure, axis = plt.subplots(figsize=(6, 4.5))
        axis.loglog(spacing, error, "o-", label="manufactured velocity error")
        axis.loglog(spacing, reference_second, "k--", label=r"$O(h^2)$ reference")
        axis.invert_xaxis(); axis.grid(True, which="both", alpha=.3)
        axis.set(xlabel="grid spacing h [m]", ylabel="relative L2 error", title="Manufactured-solution convergence")
        axis.legend(); _save_both(figure, output / "manufactured_grid_convergence")

    miz = max(1, grid.nz // 8)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    stride = max(1, grid.nx // 16)
    axes[0].quiver(x[::stride], y[::stride], manufactured_velocity[0, ::stride, ::stride, miz].T,
                   manufactured_velocity[1, ::stride, ::stride, miz].T, scale=1.0)
    axes[0].set(xlabel="x [m]", ylabel="y [m]", title="MAC solver transverse vectors", aspect="equal")
    ux = manufactured_velocity[0, :, :, miz].T; uy = manufactured_velocity[1, :, :, miz].T
    speed = np.hypot(ux, uy)
    if np.max(speed) > 0:
        axes[1].streamplot(x, y, ux, uy, color=speed, cmap="plasma", density=1.2)
    axes[1].set(xlabel="x [m]", ylabel="y [m]", title="MAC solver transverse streamlines", aspect="equal")
    _save_both(figure, output / "manufactured_vectors_streamlines")

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    component_names = ("Ux", "Uy", "Uz")
    for component, axis in enumerate(axes):
        image = axis.imshow(manufactured_velocity[component, :, :, miz].T, origin="lower",
                            extent=(0, grid.length_x, 0, grid.length_y), cmap="coolwarm")
        axis.set(title=f"MAC solver {component_names[component]}", xlabel="x [m]", ylabel="y [m]"); figure.colorbar(image, ax=axis)
    _save_both(figure, output / "manufactured_components")

    figure, axes = plt.subplots(3, 3, figsize=(12, 11))
    for component in range(3):
        reference_slice = manufactured_reference[component, :, :, miz]
        solver_slice = manufactured_velocity[component, :, :, miz]
        error_slice = manufactured_error[component, :, :, miz]
        common_limit = max(float(np.max(np.abs(reference_slice))), float(np.max(np.abs(solver_slice))), 1e-12)
        error_limit = max(float(np.max(np.abs(error_slice))), 1e-12)
        for row, value, title, limit in (
            (0, reference_slice, f"Reference {component_names[component]}", common_limit),
            (1, solver_slice, f"MAC solver {component_names[component]}", common_limit),
            (2, error_slice, f"Error {component_names[component]}", error_limit),
        ):
            image = axes[row, component].imshow(value.T, origin="lower",
                extent=(0, grid.length_x, 0, grid.length_y), cmap="coolwarm", vmin=-limit, vmax=limit)
            axes[row, component].set(title=title, xlabel="x [m]", ylabel="y [m]")
            figure.colorbar(image, ax=axes[row, component])
    figure.suptitle("Manufactured field: reference vs actual MAC output", fontsize=14)
    _save_both(figure, output / "manufactured_reference_solver_error")

    artifacts.extend(str(path) for path in sorted(output.glob("*.png")))
    artifacts.extend(str(path) for path in sorted(output.glob("*.pdf")))
    return artifacts
