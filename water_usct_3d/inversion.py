from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .acoustic import ForwardConfig3D, simulate_acoustic
from .acquisition import Acquisition3D
from .config import DemoConfig3D
from .parameterization import gauge_divergence, vector_potential_to_velocity


@dataclass
class Model3D:
    beta: torch.Tensor
    vector_potential: torch.Tensor


@dataclass(frozen=True)
class ObjectiveConfig3D:
    demo: DemoConfig3D
    acquisition: Acquisition3D
    forward: ForwardConfig3D
    lowpass_hz: float | None = None


def lowpass(traces: torch.Tensor, dt: float, cutoff: float | None) -> torch.Tensor:
    if cutoff is None:
        return traces
    spectrum = torch.fft.rfft(traces, dim=-1)
    frequencies = torch.fft.rfftfreq(traces.shape[-1], dt, device=traces.device)
    start = 0.85 * cutoff
    taper = torch.where(frequencies <= start, 1.0, torch.where(
        frequencies >= cutoff, 0.0,
        0.5 * (1.0 + torch.cos(torch.pi * (frequencies - start) / (cutoff - start))),
    ))
    return torch.fft.irfft(spectrum * taper, n=traces.shape[-1], dim=-1)


def _curvature(value: torch.Tensor) -> torch.Tensor:
    terms = []
    for axis in range(value.ndim - 3, value.ndim):
        center = value.narrow(axis, 1, value.shape[axis] - 2)
        left = value.narrow(axis, 0, value.shape[axis] - 2)
        right = value.narrow(axis, 2, value.shape[axis] - 2)
        terms.append(right - 2 * center + left)
    return sum(term.square().mean() for term in terms)


def waveform_objective_3d(model: Model3D | dict[str, torch.Tensor], observed: torch.Tensor, config: ObjectiveConfig3D) -> tuple[torch.Tensor, dict[str, Any]]:
    beta = model.beta if isinstance(model, Model3D) else model["beta"]
    potential = model.vector_potential if isinstance(model, Model3D) else model["vector_potential"]
    velocity = vector_potential_to_velocity(beta, potential, config.demo.grid)
    predicted = simulate_acoustic(velocity, config.acquisition.source_apertures, config.acquisition.apertures, config.forward)
    pred = lowpass(predicted, config.demo.grid.dt, config.lowpass_hz)
    obs = lowpass(observed, config.demo.grid.dt, config.lowpass_hz)
    mask = config.acquisition.ordered_mask[..., None]
    residual = (pred - obs) * mask
    energy = obs.square().sum(dim=-1).clamp_min(torch.finfo(obs.dtype).tiny)
    data_loss = 0.5 * (residual.square().sum(dim=-1) / energy)[config.acquisition.ordered_mask].mean()
    curvature = config.demo.inversion.curvature_weight * _curvature(potential)
    gauge = config.demo.inversion.gauge_weight * gauge_divergence(potential, config.demo.grid).square().mean()
    loss = data_loss + curvature + gauge
    diagnostics = {
        "data_loss": data_loss.detach(), "curvature": curvature.detach(), "gauge": gauge.detach(),
        "regularization_fraction": ((curvature + gauge) / loss.clamp_min(torch.finfo(loss.dtype).tiny)).detach(),
        "predicted": predicted.detach(), "velocity": velocity.detach(),
    }
    return loss, diagnostics


def amplitude_unit_prediction(objective: ObjectiveConfig3D, dtype: torch.dtype) -> torch.Tensor:
    """Full nonlinear prediction for the registered unit duct profile."""
    shape = objective.demo.inversion.vector_control
    potential = torch.zeros((3, *shape), dtype=dtype, device=objective.acquisition.apertures.device)
    with torch.no_grad():
        velocity = vector_potential_to_velocity(1.0, potential, objective.demo.grid)
        return simulate_acoustic(velocity, objective.acquisition.source_apertures,
                                 objective.acquisition.apertures, objective.forward)


def _amplitude_band_target(observed: torch.Tensor, static: torch.Tensor,
                           unit_prediction: torch.Tensor, objective: ObjectiveConfig3D,
                           cutoff: float) -> torch.Tensor:
    obs = lowpass(observed, objective.demo.grid.dt, cutoff)
    baseline = lowpass(static, objective.demo.grid.dt, cutoff)
    unit = lowpass(unit_prediction, objective.demo.grid.dt, cutoff)
    data = obs - baseline
    template = unit - baseline
    energy = obs.square().sum(dim=-1).clamp_min(torch.finfo(obs.dtype).tiny)
    mask = objective.acquisition.ordered_mask
    numerator = ((template * data).sum(dim=-1) / energy)[mask].sum()
    denominator = (template.square().sum(dim=-1) / energy)[mask].sum().clamp_min(torch.finfo(obs.dtype).tiny)
    return numerator / denominator


def invert_amplitude(observed: torch.Tensor, objective: ObjectiveConfig3D, initial_beta: float,
                     iterations: int | None = None, *, static_observed: torch.Tensor | None = None,
                     unit_prediction: torch.Tensor | None = None) -> tuple[Model3D, list[dict[str, float]]]:
    """Low-Mach scalar waveform inversion with frozen trust-region updates.

    A single full nonlinear unit-flow prediction defines the first-order waveform
    template.  Each registered frequency band has its own weighted least-squares
    target, and the scalar iterate approaches it by at most 0.05 m/s per accepted
    update.  A separate nonlinear forward validation remains required by the gate.
    """
    shape = objective.demo.inversion.vector_control
    potential = torch.zeros((3, *shape), dtype=observed.dtype, device=observed.device)
    if static_observed is None:
        raise ValueError("amplitude inversion requires the frozen static calibration data")
    if unit_prediction is None:
        unit_prediction = amplitude_unit_prediction(objective, observed.dtype)
    beta = torch.tensor(float(initial_beta), dtype=observed.dtype, device=observed.device)
    history: list[dict[str, float]] = []
    count = iterations or objective.demo.inversion.development_iterations
    global_iteration = 0
    for cutoff in objective.demo.inversion.lowpass_frequencies:
        target = _amplitude_band_target(observed, static_observed, unit_prediction, objective, float(cutoff))
        baseline = lowpass(static_observed, objective.demo.grid.dt, float(cutoff))
        obs = lowpass(observed, objective.demo.grid.dt, float(cutoff))
        template = lowpass(unit_prediction, objective.demo.grid.dt, float(cutoff)) - baseline
        mask = objective.acquisition.ordered_mask
        energy = obs.square().sum(dim=-1).clamp_min(torch.finfo(obs.dtype).tiny)
        def physical_loss(value: torch.Tensor) -> torch.Tensor:
            residual = baseline + value * template - obs
            return 0.5 * (residual.square().sum(dim=-1) / energy)[mask].mean()
        entry = physical_loss(beta).clamp_min(torch.finfo(beta.dtype).tiny)
        for band_iteration in range(count):
            delta_tensor = (target - beta).clamp(
                -objective.demo.inversion.max_update_velocity,
                objective.demo.inversion.max_update_velocity,
            )
            beta = beta + delta_tensor
            loss = physical_loss(beta)
            history.append({
                "frequency_hz": float(cutoff), "iteration": global_iteration,
                "band_iteration": band_iteration, "beta": float(beta),
                "band_target_beta": float(target), "physical_loss": float(loss),
                "normalized_loss": float(loss / entry * objective.demo.inversion.normalized_initial_objective),
                "data_loss": float(loss), "max_velocity_update": float(delta_tensor.abs()),
            })
            global_iteration += 1
    return Model3D(beta, potential), history


def _max_velocity_update(previous: Model3D, current: Model3D, config: DemoConfig3D) -> torch.Tensor:
    before = vector_potential_to_velocity(previous.beta, previous.vector_potential, config.grid)
    after = vector_potential_to_velocity(current.beta, current.vector_potential, config.grid)
    return torch.linalg.vector_norm(after - before, dim=0).max()


def run_vector_fwi(observed: torch.Tensor, objective: ObjectiveConfig3D, initial: Model3D,
                   *, profile_only: bool = False, production: bool = False) -> tuple[Model3D, list[dict[str, float]]]:
    """Frequency-continuation L-BFGS with physical velocity trust projection."""
    beta = initial.beta.detach().clone().requires_grad_(True)
    base_a = initial.vector_potential.detach().clone()
    if profile_only:
        base_a = base_a.mean(dim=-1, keepdim=True)
    potential = base_a.requires_grad_(True)
    history: list[dict[str, float]] = []
    per_band = objective.demo.inversion.production_iterations if production else objective.demo.inversion.development_iterations
    for cutoff in objective.demo.inversion.lowpass_frequencies:
        band = ObjectiveConfig3D(objective.demo, objective.acquisition, objective.forward, float(cutoff))
        expanded = potential.expand(-1, -1, -1, objective.demo.inversion.vector_control[2]) if profile_only else potential
        with torch.no_grad():
            entry, _ = waveform_objective_3d(Model3D(beta, expanded), observed, band)
        objective_scale = objective.demo.inversion.normalized_initial_objective / max(float(entry), torch.finfo(entry.dtype).tiny)
        optimizer = torch.optim.LBFGS([beta, potential], lr=1.0, max_iter=1,
                                      history_size=objective.demo.inversion.lbfgs_history,
                                      line_search_fn="strong_wolfe")
        stagnant = 0
        previous_loss = float(entry)
        for iteration in range(per_band):
            previous = Model3D(beta.detach().clone(), expanded.detach().clone())
            latest: dict[str, torch.Tensor] = {}
            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                current_a = potential.expand(-1, -1, -1, objective.demo.inversion.vector_control[2]) if profile_only else potential
                loss, diagnostics = waveform_objective_3d(Model3D(beta, current_a), observed, band)
                (loss * objective_scale).backward()
                latest.update(loss=loss.detach(), data=diagnostics["data_loss"], regularization=(diagnostics["curvature"] + diagnostics["gauge"]))
                return loss * objective_scale
            optimizer.step(closure)
            expanded = potential.expand(-1, -1, -1, objective.demo.inversion.vector_control[2]) if profile_only else potential
            candidate = Model3D(beta.detach(), expanded.detach())
            update = float(_max_velocity_update(previous, candidate, objective.demo))
            projected = update > objective.demo.inversion.max_update_velocity
            if projected:
                scale = objective.demo.inversion.max_update_velocity / update
                with torch.no_grad():
                    beta.copy_(previous.beta + scale * (beta - previous.beta))
                    target = previous.vector_potential + scale * (expanded - previous.vector_potential)
                    potential.copy_(target.mean(dim=-1, keepdim=True) if profile_only else target)
                optimizer.state.clear()
                update = objective.demo.inversion.max_update_velocity
            physical = float(latest["loss"])
            improvement = (previous_loss - physical) / max(previous_loss, torch.finfo(entry.dtype).tiny)
            stagnant = stagnant + 1 if improvement < objective.demo.inversion.early_stop_relative else 0
            previous_loss = physical
            history.append({"frequency_hz": float(cutoff), "iteration": iteration, "physical_loss": physical,
                            "normalized_loss": physical * objective_scale, "data_loss": float(latest["data"]),
                            "regularization": float(latest["regularization"]), "max_velocity_update": update,
                            "projected": float(projected), "beta": float(beta.detach())})
            if stagnant >= objective.demo.inversion.early_stop_patience:
                break
    final_a = potential.expand(-1, -1, -1, objective.demo.inversion.vector_control[2]).detach().clone() if profile_only else potential.detach()
    return Model3D(beta.detach(), final_a), history
