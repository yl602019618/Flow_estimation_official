"""Evaluation and artifact pipeline for open-water full-wave observations."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .basis import build_solenoidal_basis, build_solenoidal_ray_matrix
from .config import SphereConfig, coerce_config
from .evaluation import reconstruction_metrics
from .flow import tilted_vortex
from .forward import simulate_chord_travel_times
from .fullwave import (FullWaveDelays, FullWaveResult, NoisyFullWaveObservation,
                       calibrate_waveform_noise, extract_fullwave_delays,
                       simulate_open_water_fullwave)
from .geometry import SphereAcquisition, build_fibonacci_sphere_acquisition
from .inversion import invert_sphere_tsvd
from .pipeline import GateFailure
from .storage import write_json, write_snapshots


def _delay_metrics(measured: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    error = measured - reference
    return {
        "ray_rmse_ns": float(np.sqrt(np.mean(error**2))/1e-9),
        "ray_bias_ns": float(np.mean(error)/1e-9),
        "ray_correlation": float(np.corrcoef(measured, reference)[0, 1]),
    }


def _richardson(coarse: np.ndarray, fine: np.ndarray, coarse_n: int, fine_n: int) -> np.ndarray:
    ratio = (fine_n - 1) / (coarse_n - 1)
    return (ratio**2 * fine - coarse) / (ratio**2 - 1.0)


def _reciprocity_mismatch(static: FullWaveResult, acquisition: SphereAcquisition) -> float:
    a, b = acquisition.pairs.T
    forward = static.traces[a, b]
    reverse = static.traces[b, a]
    return float(np.linalg.norm(forward-reverse) / np.linalg.norm(forward))


def _save_fullwave(path: Path, acquisition: SphereAcquisition,
                   runs: dict[str, FullWaveResult], delays: dict[str, FullWaveDelays],
                   extrapolated: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("points", data=acquisition.points)
        handle.create_dataset("pairs", data=acquisition.pairs)
        for name, run in runs.items():
            group = handle.create_group(name)
            group.create_dataset("traces", data=run.traces, compression="gzip",
                                 compression_opts=4, shuffle=True,
                                 chunks=(1, run.traces.shape[1], run.traces.shape[2]))
            group.attrs.update(dt_s=run.dt_s, grid_n=run.grid_n, runtime_s=run.runtime_s)
        for name, value in delays.items():
            group = handle.create_group(f"delays_{name}")
            group.create_dataset("directed_shift_s", data=value.directed_shift_s)
            group.create_dataset("reciprocal_s", data=value.reciprocal_s)
            group.create_dataset("fit_residual_rms", data=value.fit_residual_rms)
        handle.create_dataset("reciprocal_observation_s", data=delays["fine"].reciprocal_s)
        handle.create_dataset("reciprocal_richardson_diagnostic_s", data=extrapolated)
        handle.attrs["observation_model"] = "first-order-Mach convected scalar full wave"
        handle.attrs["boundary"] = "open-water box with split-field PML"
        handle.attrs["solver_revision"] = "split_pml_v2_physical_width"


def _save_noisy(path: Path, acquisition: SphereAcquisition,
                observations: dict[str, NoisyFullWaveObservation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("pairs", data=acquisition.pairs)
        for name, observation in observations.items():
            group = handle.create_group(name)
            group.create_dataset("static_traces", data=observation.static.traces,
                                 compression="gzip", compression_opts=4, shuffle=True,
                                 chunks=(1, observation.static.traces.shape[1],
                                         observation.static.traces.shape[2]))
            group.create_dataset("flow_traces", data=observation.flow.traces,
                                 compression="gzip", compression_opts=4, shuffle=True,
                                 chunks=(1, observation.flow.traces.shape[1],
                                         observation.flow.traces.shape[2]))
            group.create_dataset("directed_shift_s", data=observation.delays.directed_shift_s)
            group.create_dataset("reciprocal_s", data=observation.delays.reciprocal_s)
            group.create_dataset("fit_residual_rms", data=observation.delays.fit_residual_rms)
            group.attrs.update(
                amplitude_std=observation.amplitude_std,
                target_delay_std_s=observation.target_delay_std_s,
                realized_delay_std_s=observation.realized_delay_std_s,
                delay_bias_s=observation.delay_bias_s,
                peak_snr_db=observation.peak_snr_db,
                calibration_iterations=observation.calibration_iterations,
            )
        handle.attrs["noise_model"] = "independent additive white Gaussian pressure noise"
        handle.attrs["calibration"] = "global amplitude fitted using delay extraction only"


def _plot_diagnostics(path: Path, reference: np.ndarray, coarse: np.ndarray,
                      fine: np.ndarray, extrapolated: np.ndarray,
                      static: FullWaveResult, flow: FullWaveResult,
                      acquisition: SphereAcquisition) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    scale = 1e9
    axes[0].scatter(reference*scale, coarse*scale, s=5, alpha=.3, label="73^3")
    axes[0].scatter(reference*scale, fine*scale, s=5, alpha=.3, label="97^3")
    axes[0].scatter(reference*scale, extrapolated*scale, s=5, alpha=.3, label="Richardson")
    limit = np.max(np.abs(reference))*scale
    axes[0].plot([-limit, limit], [-limit, limit], "k--", lw=1)
    axes[0].set(xlabel="exact-ray reciprocal delay (ns)", ylabel="full-wave extracted (ns)")
    axes[0].legend(markerscale=2); axes[0].grid(alpha=.2)
    pair = int(np.argmax(np.abs(reference)))
    a, b = acquisition.pairs[pair]
    time = np.arange(static.traces.shape[-1])*static.dt_s*1e6
    axes[1].plot(time, static.traces[a, b], label="zero flow", lw=1)
    axes[1].plot(time, flow.traces[a, b], label="flow", lw=1)
    axes[1].set(xlabel="time (us)", ylabel="pressure (arbitrary)",
                title=f"representative trace {a}->{b}")
    axes[1].legend(); axes[1].grid(alpha=.2)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _plot_noise_summary(path: Path, metrics: dict[str, dict]) -> None:
    names = ["clean", "noise_5ns", "noise_10ns"]
    labels = ["clean", "5 ns", "10 ns"]
    error = [100*metrics[name]["inversion"]["relative_l2_error"] for name in names]
    correlation = [metrics[name]["inversion"]["field_correlation"] for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4))
    axes[0].bar(labels, error, color=["#4c78a8", "#f58518", "#e45756"])
    axes[0].axhline(2, color="#4c78a8", ls="--", lw=1)
    axes[0].axhline(10, color="#f58518", ls="--", lw=1)
    axes[0].axhline(15, color="#e45756", ls="--", lw=1)
    axes[0].set(ylabel="relative L2 error (%)", title="Full-wave observation inversion")
    axes[1].bar(labels, correlation, color=["#4c78a8", "#f58518", "#e45756"])
    axes[1].set_ylim(.9, 1.005)
    axes[1].set(ylabel="field correlation", title="Directional agreement")
    for ax in axes: ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _run_pair(acquisition: SphereAcquisition, config: SphereConfig,
              grid_n: int) -> tuple[FullWaveResult, FullWaveResult, FullWaveDelays]:
    static = simulate_open_water_fullwave(acquisition, config, grid_n=grid_n, with_flow=False)
    flow = simulate_open_water_fullwave(acquisition, config, grid_n=grid_n, with_flow=True)
    return static, flow, extract_fullwave_delays(flow, static, acquisition, config)


def _load_runs(path: Path, config: SphereConfig) -> dict[str, FullWaveResult]:
    section = config.section("full_wave")
    expected = {"coarse_static": int(section["production_coarse_grid_n"]),
                "coarse_flow": int(section["production_coarse_grid_n"]),
                "fine_static": int(section["production_grid_n"]),
                "fine_flow": int(section["production_grid_n"])}
    runs = {}
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("boundary") != "open-water box with split-field PML":
            raise ValueError("cached waveforms do not use the registered split-field PML")
        if handle.attrs.get("solver_revision") != "split_pml_v2_physical_width":
            raise ValueError("cached waveforms predate the physical-width PML contract")
        for name, grid_n in expected.items():
            group = handle[name]
            if int(group.attrs["grid_n"]) != grid_n or float(group.attrs["dt_s"]) != float(section["dt_s"]):
                raise ValueError(f"cached {name} does not match the frozen grid")
            runs[name] = FullWaveResult(group["traces"][...], float(group.attrs["dt_s"]),
                                        grid_n, float(group.attrs["runtime_s"]))
    return runs


def run_fullwave_benchmark(config: SphereConfig | str | dict, *,
                           reuse_waveforms: bool = False) -> dict:
    cfg = coerce_config(config)
    section = cfg.section("full_wave")
    root = cfg.output_root / "fullwave"
    write_snapshots(cfg, root)
    flow_function = lambda points: tilted_vortex(points, cfg)

    cache_path = root / "observations.h5"
    if reuse_waveforms:
        if not cache_path.exists() or not (root / "smoke_metrics.json").exists():
            raise FileNotFoundError("--reuse-waveforms requires validated full-wave artifacts")
        with (root / "smoke_metrics.json").open("r", encoding="utf-8") as stream:
            smoke = json.load(stream)
    else:
        smoke_acquisition = build_fibonacci_sphere_acquisition(cfg, int(section["smoke_transducers"]))
        smoke_reference = simulate_chord_travel_times(flow_function, smoke_acquisition, cfg).reciprocal_s
        smoke_coarse_n = int(section["smoke_grid_n"])
        smoke_fine_n = int(section["smoke_fine_grid_n"])
        sc_static, sc_flow, sc_delay = _run_pair(smoke_acquisition, cfg, smoke_coarse_n)
        sf_static, sf_flow, sf_delay = _run_pair(smoke_acquisition, cfg, smoke_fine_n)
        smoke_extrapolated = _richardson(sc_delay.reciprocal_s, sf_delay.reciprocal_s,
                                         smoke_coarse_n, smoke_fine_n)
        smoke = {
            "coarse": _delay_metrics(sc_delay.reciprocal_s, smoke_reference),
            "fine": _delay_metrics(sf_delay.reciprocal_s, smoke_reference),
            "richardson": _delay_metrics(smoke_extrapolated, smoke_reference),
            "coarse_fine_rms_ns": float(np.sqrt(np.mean(
                (sc_delay.reciprocal_s-sf_delay.reciprocal_s)**2))/1e-9),
            "zero_flow_reciprocity_relative": _reciprocity_mismatch(sf_static, smoke_acquisition),
            "runtime_s": sc_static.runtime_s+sc_flow.runtime_s+sf_static.runtime_s+sf_flow.runtime_s,
        }
        write_json(root / "smoke_metrics.json", smoke)
    if (smoke["fine"]["ray_rmse_ns"] > float(section["smoke_ray_rmse_max_ns"]) or
            smoke["fine"]["ray_correlation"] < float(section["minimum_delay_correlation"])):
        raise GateFailure("full-wave smoke delay gate")

    acquisition = build_fibonacci_sphere_acquisition(cfg)
    reference = simulate_chord_travel_times(flow_function, acquisition, cfg).reciprocal_s
    coarse_n = int(section["production_coarse_grid_n"])
    fine_n = int(section["production_grid_n"])
    if reuse_waveforms:
        cached = _load_runs(cache_path, cfg)
        coarse_static, coarse_flow = cached["coarse_static"], cached["coarse_flow"]
        fine_static, fine_flow = cached["fine_static"], cached["fine_flow"]
        coarse_delay = extract_fullwave_delays(coarse_flow, coarse_static, acquisition, cfg)
        fine_delay = extract_fullwave_delays(fine_flow, fine_static, acquisition, cfg)
    else:
        coarse_static, coarse_flow, coarse_delay = _run_pair(acquisition, cfg, coarse_n)
        fine_static, fine_flow, fine_delay = _run_pair(acquisition, cfg, fine_n)
    extrapolated = _richardson(coarse_delay.reciprocal_s, fine_delay.reciprocal_s,
                               coarse_n, fine_n)
    numerical_sigma = float(np.sqrt(np.mean(
        (coarse_delay.reciprocal_s-fine_delay.reciprocal_s)**2)))
    basis = build_solenoidal_basis(cfg)
    operator = build_solenoidal_ray_matrix(acquisition, basis, cfg)
    inversion = invert_sphere_tsvd(fine_delay.reciprocal_s, operator, 0.0)
    inversion_info = reconstruction_metrics(inversion, operator, cfg)
    dataset_metrics: dict[str, dict] = {
        "clean": {
            "measurement_noise_std_ns": 0.0,
            "ray_diagnostic": _delay_metrics(fine_delay.reciprocal_s, reference),
            "inversion": inversion_info,
        }
    }
    noisy_observations: dict[str, NoisyFullWaveObservation] = {}
    for target_ns in section["waveform_noise_targets_ns"]:
        target_ns = float(target_ns)
        name = f"noise_{int(target_ns)}ns"
        observation = calibrate_waveform_noise(
            fine_static, fine_flow, fine_delay, acquisition, cfg, target_ns*1e-9,
            seed=cfg.seed+int(target_ns*1000),
        )
        noisy_observations[name] = observation
        noisy_inversion = invert_sphere_tsvd(
            observation.delays.reciprocal_s, operator, target_ns*1e-9)
        dataset_metrics[name] = {
            "measurement_noise_std_ns": observation.realized_delay_std_s/1e-9,
            "measurement_noise_bias_ns": observation.delay_bias_s/1e-9,
            "waveform_noise_amplitude_std": observation.amplitude_std,
            "median_peak_snr_db": observation.peak_snr_db,
            "calibration_iterations": observation.calibration_iterations,
            "ray_diagnostic": _delay_metrics(observation.delays.reciprocal_s, reference),
            "inversion": reconstruction_metrics(noisy_inversion, operator, cfg),
        }
    gates = cfg.section("evaluation")
    inversion_gates = {
        "clean_relative_l2": bool(dataset_metrics["clean"]["inversion"]["relative_l2_error"] <=
                                  float(gates["clean_relative_l2_max"])),
        "noise_5ns": bool(
            dataset_metrics["noise_5ns"]["inversion"]["relative_l2_error"] <=
            float(gates["noise_5ns_relative_l2_max"]) and
            dataset_metrics["noise_5ns"]["inversion"]["field_correlation"] >=
            float(gates["noise_5ns_correlation_min"]) and
            float(gates["residual_ratio_min"]) <=
            dataset_metrics["noise_5ns"]["inversion"]["residual_ratio"] <=
            float(gates["residual_ratio_max"])),
        "noise_10ns": bool(
            dataset_metrics["noise_10ns"]["inversion"]["relative_l2_error"] <=
            float(gates["noise_10ns_relative_l2_max"]) and
            dataset_metrics["noise_10ns"]["inversion"]["field_correlation"] >=
            float(gates["noise_10ns_correlation_min"]) and
            float(gates["residual_ratio_min"]) <=
            dataset_metrics["noise_10ns"]["inversion"]["residual_ratio"] <=
            float(gates["residual_ratio_max"])),
    }
    inversion_gates["all_passed"] = all(inversion_gates.values())
    metrics = {
        "status": "passed",
        "coarse": _delay_metrics(coarse_delay.reciprocal_s, reference),
        "fine": _delay_metrics(fine_delay.reciprocal_s, reference),
        "richardson": _delay_metrics(extrapolated, reference),
        "coarse_fine_rms_ns": float(np.sqrt(np.mean(
            (coarse_delay.reciprocal_s-fine_delay.reciprocal_s)**2))/1e-9),
        "estimated_numerical_sigma_ns": numerical_sigma/1e-9,
        "zero_flow_reciprocity_relative": _reciprocity_mismatch(fine_static, acquisition),
        "median_fit_residual": float(np.nanmedian(fine_delay.fit_residual_rms)),
        "maximum_fit_residual": float(np.nanmax(fine_delay.fit_residual_rms)),
        "runtime_s": coarse_static.runtime_s+coarse_flow.runtime_s+
                     fine_static.runtime_s+fine_flow.runtime_s,
        "inversion": inversion_info,
        "datasets": dataset_metrics,
        "inversion_gates": inversion_gates,
    }
    if (metrics["fine"]["ray_rmse_ns"] > float(section["production_ray_rmse_max_ns"]) or
            metrics["fine"]["ray_correlation"] < float(section["minimum_delay_correlation"])):
        metrics["status"] = "failed"
    _save_fullwave(root / "observations.h5", acquisition,
                   {"coarse_static": coarse_static, "coarse_flow": coarse_flow,
                    "fine_static": fine_static, "fine_flow": fine_flow},
                   {"coarse": coarse_delay, "fine": fine_delay}, extrapolated)
    _save_noisy(root / "noisy_observations.h5", acquisition, noisy_observations)
    np.savez_compressed(root / "delays.npz", reference_s=reference,
                        coarse_s=coarse_delay.reciprocal_s, fine_s=fine_delay.reciprocal_s,
                        richardson_s=extrapolated,
                        observation_s=fine_delay.reciprocal_s,
                        inversion_prediction_s=inversion.predicted_delays_s)
    _plot_diagnostics(root / "diagnostics.png", reference, coarse_delay.reciprocal_s,
                      fine_delay.reciprocal_s, extrapolated, fine_static, fine_flow, acquisition)
    _plot_noise_summary(root / "noise_inversion_summary.png", dataset_metrics)
    write_json(root / "metrics.json", metrics)
    if metrics["status"] != "passed":
        raise GateFailure("production full-wave delay gate")
    return metrics
