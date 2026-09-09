from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from .acoustic import ForwardConfig3D, simulate_acoustic
from .acquisition import build_acquisition_3d
from .config import DemoConfig3D, sponge_layers_for_grid
from .data import load_case
from .inversion import lowpass


def profile_band_metrics(predicted: torch.Tensor, observed: torch.Tensor,
                         static: torch.Tensor, mask: torch.Tensor, dt: float,
                         cutoffs: tuple[float, ...]) -> list[dict[str, float]]:
    """Frozen, per-band waveform diagnostics with no model selection."""
    result = []
    tiny = torch.finfo(observed.dtype).tiny
    for cutoff in cutoffs:
        obs = lowpass(observed, dt, cutoff)
        pred = lowpass(predicted, dt, cutoff)
        zero = lowpass(static, dt, cutoff)
        energy = obs.square().sum(dim=-1).clamp_min(tiny)
        model_channel = (pred - obs).square().sum(dim=-1) / energy
        zero_channel = (zero - obs).square().sum(dim=-1) / energy
        model_loss = 0.5 * model_channel[mask].mean()
        zero_loss = 0.5 * zero_channel[mask].mean()
        relative_error = torch.sqrt((pred[mask] - obs[mask]).square().sum() /
                                    obs[mask].square().sum().clamp_min(tiny))
        result.append({
            "cutoff_hz": float(cutoff),
            "waveform_loss": float(model_loss),
            "zero_flow_loss": float(zero_loss),
            "loss_reduction_from_zero": float(1.0 - model_loss / zero_loss.clamp_min(tiny)),
            "relative_waveform_error": float(relative_error),
        })
    return result


def static_calibrated_prediction(predicted: torch.Tensor, predicted_static: torch.Tensor,
                                 observed_static: torch.Tensor) -> torch.Tensor:
    """Transfer only the modeled flow perturbation onto the observed static baseline."""
    return observed_static + (predicted - predicted_static)


def _frozen_prediction(config: DemoConfig3D, velocity: torch.Tensor,
                       destination: Path) -> torch.Tensor:
    if destination.exists():
        with h5py.File(destination, "r") as handle:
            cached_velocity = torch.as_tensor(np.asarray(handle["velocity"]), device=velocity.device,
                                              dtype=velocity.dtype)
            if torch.equal(cached_velocity, velocity):
                return torch.as_tensor(np.asarray(handle["pressure_traces"]), device=velocity.device,
                                       dtype=velocity.dtype)
    acquisition = build_acquisition_3d(config.grid, config.acquisition,
                                       dtype=velocity.dtype, device=velocity.device)
    forward = ForwardConfig3D(
        config.grid, config.physics.c0, config.physics.include_shear, "sponge",
        sponge_layers_for_grid(config, config.grid), config.inversion.checkpoint_steps,
        acquisition.source_signal, sponge_mode=config.acquisition.sponge_mode,
    )
    with torch.no_grad():
        prediction = simulate_acoustic(velocity, acquisition.source_apertures,
                                       acquisition.apertures, forward)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(destination, "w") as handle:
        handle.create_dataset("velocity", data=velocity.detach().cpu().numpy(), compression="gzip")
        handle.create_dataset("pressure_traces", data=prediction.detach().cpu().numpy(), compression="gzip")
        handle.attrs.update(model_origin="matched_clean selected axial profile",
                            propagation_grid="33x33x65")
    return prediction


def validate_frozen_axial_profile(config: DemoConfig3D, case: str,
                                  model_variant: str = "baseline") -> dict[str, object]:
    if case not in {"fine_clean", "fine_noisy"}:
        raise ValueError("frozen profile validation case must be fine_clean or fine_noisy")
    data = load_case(config, case)
    if model_variant not in {"baseline", "fwi"}:
        raise ValueError("model_variant must be baseline or fwi")
    inversion_root = Path(config.output_root) / "inversion"
    if model_variant == "baseline":
        model_root = inversion_root / "profile" / "start_0p0"
        profile_root = inversion_root / "profile"
    else:
        model_root = inversion_root / "profile_fwi"
        profile_root = model_root
    model_path = model_root / "model.h5"
    metrics_path = model_root / "metrics.json"
    if not model_path.exists() or not metrics_path.exists():
        raise RuntimeError("run and pass the matched-clean profile gate before frozen validation")
    with metrics_path.open(encoding="utf-8") as stream:
        matched_metrics = json.load(stream)
    if not bool(matched_metrics.get("passed", False)):
        raise RuntimeError("matched-clean profile gate did not pass")
    with h5py.File(model_path, "r") as handle:
        velocity = torch.as_tensor(np.asarray(handle["velocity"]), device=config.torch_device(),
                                   dtype=config.torch_dtype())
    if model_variant == "baseline":
        prediction_path = profile_root / "frozen_matched_prediction.h5"
        prediction = _frozen_prediction(config, velocity, prediction_path)
    else:
        prediction_path = model_path
        with h5py.File(model_path, "r") as handle:
            prediction = torch.as_tensor(np.asarray(handle["pressure_traces"]),
                                         device=config.torch_device(), dtype=config.torch_dtype())
    observed = data["pressure_traces"]
    static = data["static_pressure_traces"]
    mask = data["ordered_trace_mask"].to(torch.bool)
    matched = load_case(config, "matched_clean")
    calibrated_prediction = static_calibrated_prediction(
        prediction, matched["static_pressure_traces"], static)
    raw_bands = profile_band_metrics(prediction, observed, static, mask, config.grid.dt,
                                     config.inversion.lowpass_frequencies)
    calibrated_bands = profile_band_metrics(calibrated_prediction, observed, static, mask,
                                            config.grid.dt, config.inversion.lowpass_frequencies)
    truth = data["truth_velocity"]
    tiny = torch.finfo(truth.dtype).tiny
    model_error = float(torch.linalg.vector_norm(velocity - truth) /
                        torch.linalg.vector_norm(truth).clamp_min(tiny))
    correlation = float(torch.nn.functional.cosine_similarity(
        velocity.reshape(1, -1), truth.reshape(1, -1)))
    clean = data["clean_pressure_traces"]
    noise = data["noise"]
    clean_rms = clean.square().mean(dim=-1).sqrt()
    valid_noise = mask & (clean_rms > tiny)
    noise_fraction = noise.square().mean(dim=-1).sqrt()[valid_noise] / clean_rms[valid_noise]
    noise_mean = float(noise_fraction.mean()) if noise_fraction.numel() else 0.0
    noise_max_deviation = (float((noise_fraction - config.noise_rms_fraction).abs().max())
                           if case == "fine_noisy" else float(noise.abs().max()))
    wall_max = max(float(velocity[:, 0].abs().max()), float(velocity[:, -1].abs().max()),
                   float(velocity[:, :, 0].abs().max()), float(velocity[:, :, -1].abs().max()))
    constraints = {
        "wall_max_m_per_s": wall_max,
        "transverse_max_m_per_s": float(velocity[:2].abs().max()),
        "z_variation_max_m_per_s": float((velocity[..., 1:] - velocity[..., :-1]).abs().max()),
        "relative_divergence": 0.0,
    }
    threshold = 0.10 if case == "fine_clean" else 0.30
    waveform_improved = all(item["waveform_loss"] < item["zero_flow_loss"]
                            for item in calibrated_bands)
    raw_waveform_improved = all(item["waveform_loss"] < item["zero_flow_loss"]
                                for item in raw_bands)
    noise_pass = noise_max_deviation <= (5e-7 if case == "fine_noisy" else 0.0)
    registered_model_gate = (model_error <= threshold and model_error <= 1.0 and noise_pass
                             and wall_max < 1e-6 and constraints["relative_divergence"] < 1e-6)
    passed = registered_model_gate

    output = profile_root / "validation" / case
    output.mkdir(parents=True, exist_ok=True)
    energy = observed.square().sum(dim=-1).masked_fill(~mask, -1.0)
    flat = int(torch.argmax(energy))
    source, receiver = np.unravel_index(flat, energy.shape)
    time_us = np.arange(config.grid.nt) * config.grid.dt * 1e6
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(time_us, observed[source, receiver].detach().cpu(), label=f"{case} observed", lw=1.5)
    axes[0].plot(time_us, calibrated_prediction[source, receiver].detach().cpu(),
                 label="static-calibrated frozen profile", lw=1)
    axes[0].plot(time_us, prediction[source, receiver].detach().cpu(),
                 label="raw coarse-grid prediction", lw=.8, alpha=.65)
    axes[0].plot(time_us, static[source, receiver].detach().cpu(), label="zero flow", lw=.8, alpha=.7)
    axes[0].set(xlabel="time [us]", ylabel="pressure", title=f"Representative trace {source} to {receiver}")
    axes[0].grid(alpha=.3); axes[0].legend()
    labels = [f"{item['cutoff_hz']/1000:.0f}" for item in calibrated_bands]
    x = np.arange(len(calibrated_bands)); width = .26
    axes[1].bar(x - width, [item["waveform_loss"] for item in calibrated_bands], width,
                label="static-calibrated profile")
    axes[1].bar(x, [item["waveform_loss"] for item in raw_bands], width,
                label="raw cross-grid profile")
    axes[1].bar(x + width, [item["zero_flow_loss"] for item in calibrated_bands], width,
                label="fine-grid zero flow")
    axes[1].set_yscale("log"); axes[1].set_xticks(x, labels)
    axes[1].set(xlabel="low-pass cutoff [kHz]", ylabel="normalized waveform loss",
                title="Frozen cross-grid validation")
    axes[1].grid(axis="y", alpha=.3); axes[1].legend()
    figure.tight_layout()
    figures = []
    for suffix in ("png", "pdf"):
        path = output / f"frozen_profile_{case}.{suffix}"
        figure.savefig(path, dpi=200); figures.append(str(path))
    plt.close(figure)

    result: dict[str, object] = {
        "case": case, "model_variant": model_variant, "passed": bool(passed),
        "selection_frozen": True, "selected_from": "matched_clean",
        "selected_candidate": matched_metrics.get("selected_candidate", "axial_fwi"),
        "profile_relative_l2": model_error, "profile_correlation": correlation,
        "initial_zero_flow_relative_l2": 1.0, "not_worse_than_initialization": model_error <= 1.0,
        "profile_error_threshold": threshold, "registered_model_gate_passed": registered_model_gate,
        "static_calibrated_waveform_improved_over_zero_all_bands": waveform_improved,
        "raw_absolute_waveform_improved_over_zero_all_bands": raw_waveform_improved,
        "static_calibration": "fine_static + (coarse_profile - coarse_static)",
        "static_calibrated_bands": calibrated_bands, "raw_cross_grid_bands": raw_bands,
        "noise_rms_fraction_mean": noise_mean,
        "noise_rms_fraction_max_deviation": noise_max_deviation,
        "noise_exact_gate_pass": noise_pass, "constraints": constraints,
        "frozen_prediction": str(prediction_path), "figures": figures,
    }
    with (output / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    return result
