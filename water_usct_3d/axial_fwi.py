from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

import torch

from .acoustic import simulate_acoustic
from .inversion import ObjectiveConfig3D, lowpass
from .profile_inversion import axial_profile_to_velocity


@dataclass
class AxialFWIResult3D:
    control: torch.Tensor
    velocity: torch.Tensor
    history: list[dict[str, float]]
    initial_loss: float
    final_loss: float
    closure_evaluations: int


@dataclass(frozen=True)
class _DeviceBatch:
    objective: ObjectiveConfig3D
    observed: torch.Tensor
    static: torch.Tensor
    weight: float


def _control_from_interior(interior: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    control = torch.zeros(shape, dtype=interior.dtype, device=interior.device)
    control[1:-1, 1:-1] = interior.reshape(shape[0] - 2, shape[1] - 2)
    return control


def _profile_curvature(control: torch.Tensor) -> torch.Tensor:
    dxx = control[2:, 1:-1] - 2.0 * control[1:-1, 1:-1] + control[:-2, 1:-1]
    dyy = control[1:-1, 2:] - 2.0 * control[1:-1, 1:-1] + control[1:-1, :-2]
    return 0.5 * (dxx.square().mean() + dyy.square().mean())


def axial_fwi_objective(interior: torch.Tensor, observed: torch.Tensor,
                        static: torch.Tensor, objective: ObjectiveConfig3D,
                        cutoff: float, source_indices: torch.Tensor | None = None,
                        curvature_weight: float | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    """Differentiable flow-perturbation waveform objective for scalar Uz(x,y)."""
    shape = objective.demo.inversion.profile_control
    control = _control_from_interior(interior, shape)
    velocity = axial_profile_to_velocity(control, objective.demo.grid)
    if source_indices is None:
        source_indices = torch.arange(objective.acquisition.source_apertures.shape[0],
                                      device=interior.device)
    sources = objective.acquisition.source_apertures.index_select(0, source_indices)
    predicted = simulate_acoustic(velocity, sources, objective.acquisition.apertures,
                                  objective.forward)
    observed_local = observed.index_select(0, source_indices)
    static_local = static.index_select(0, source_indices)
    predicted_effect = lowpass(predicted - static_local, objective.demo.grid.dt, cutoff)
    observed_effect = lowpass(observed_local - static_local, objective.demo.grid.dt, cutoff)
    mask = objective.acquisition.ordered_mask.index_select(0, source_indices)
    energy = observed_effect.square().sum(dim=-1).clamp_min(torch.finfo(interior.dtype).tiny)
    channel_misfit = (predicted_effect - observed_effect).square().sum(dim=-1) / energy
    data_loss = 0.5 * channel_misfit[mask].mean()
    weight = (objective.demo.inversion.curvature_weight if curvature_weight is None
              else curvature_weight)
    curvature = weight * _profile_curvature(control)
    loss = data_loss + curvature
    return loss, {
        "data_loss": data_loss.detach(), "curvature": curvature.detach(),
        "regularization_fraction": (curvature / loss.clamp_min(torch.finfo(loss.dtype).tiny)).detach(),
        "control": control.detach(), "velocity": velocity.detach(), "predicted": predicted.detach(),
    }


def axial_flow_waveform_loss(predicted: torch.Tensor, observed: torch.Tensor,
                             static: torch.Tensor, mask: torch.Tensor,
                             dt: float, cutoff: float) -> torch.Tensor:
    predicted_effect = lowpass(predicted - static, dt, cutoff)
    observed_effect = lowpass(observed - static, dt, cutoff)
    energy = observed_effect.square().sum(dim=-1).clamp_min(torch.finfo(observed.dtype).tiny)
    return 0.5 * (((predicted_effect - observed_effect).square().sum(dim=-1) / energy)[mask]).mean()


def _prepare_dual_gpu_batches(observed: torch.Tensor, static: torch.Tensor,
                              objective: ObjectiveConfig3D) -> list[_DeviceBatch]:
    if torch.cuda.device_count() < 2:
        raise RuntimeError("two CUDA devices are required for production axial FWI")
    count = objective.acquisition.source_apertures.shape[0]
    index_batches = torch.arange(count).split((count + 1) // 2)
    batches = []
    total_channels = float(objective.acquisition.ordered_mask.sum())
    for device_index, indices_cpu in enumerate(index_batches[:2]):
        device = torch.device(f"cuda:{device_index}")
        indices = indices_cpu.to(objective.acquisition.source_apertures.device)
        acquisition = replace(
            objective.acquisition,
            coordinates=objective.acquisition.coordinates.to(device),
            apertures=objective.acquisition.apertures.to(device),
            source_apertures=objective.acquisition.source_apertures.index_select(0, indices).to(device),
            ordered_mask=objective.acquisition.ordered_mask.index_select(0, indices).to(device),
            reciprocal_pairs=objective.acquisition.reciprocal_pairs.to(device),
            source_signal=objective.acquisition.source_signal.to(device),
        )
        forward = replace(
            objective.forward,
            source_signal=(None if objective.forward.source_signal is None
                           else objective.forward.source_signal.to(device)),
        )
        local_objective = ObjectiveConfig3D(objective.demo, acquisition, forward)
        local_channels = float(acquisition.ordered_mask.sum())
        batches.append(_DeviceBatch(
            local_objective, observed.index_select(0, indices).to(device),
            static.index_select(0, indices).to(device), local_channels / total_channels,
        ))
    return batches


def _batch_loss_gradient(interior: torch.Tensor, batch: _DeviceBatch,
                         cutoff: float) -> tuple[torch.Tensor, torch.Tensor]:
    device = batch.observed.device
    local = interior.to(device).detach().requires_grad_(True)
    loss, _ = axial_fwi_objective(
        local, batch.observed, batch.static, batch.objective, cutoff,
        curvature_weight=0.0,
    )
    gradient = torch.autograd.grad(loss, local)[0]
    return loss.detach(), gradient.detach()


def _batch_loss_only(interior: torch.Tensor, batch: _DeviceBatch,
                     cutoff: float) -> torch.Tensor:
    with torch.no_grad():
        loss, _ = axial_fwi_objective(
            interior.to(batch.observed.device), batch.observed, batch.static,
            batch.objective, cutoff, curvature_weight=0.0,
        )
    return loss.detach()


def _dual_gpu_loss_gradient(interior: torch.Tensor, batches: list[_DeviceBatch],
                            cutoff: float, executor: ThreadPoolExecutor,
                            curvature_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    futures = [executor.submit(_batch_loss_gradient, interior.detach(), batch, cutoff)
               for batch in batches]
    results = [future.result() for future in futures]
    target = interior.device
    data_loss = sum(batch.weight * value[0].to(target)
                    for batch, value in zip(batches, results))
    data_gradient = sum(batch.weight * value[1].to(target)
                        for batch, value in zip(batches, results))
    regularization_variable = interior.detach().clone().requires_grad_(True)
    control = _control_from_interior(
        regularization_variable, batches[0].objective.demo.inversion.profile_control)
    curvature = curvature_weight * _profile_curvature(control)
    regularization_gradient = torch.autograd.grad(curvature, regularization_variable)[0]
    return (data_loss + curvature.detach(), data_gradient + regularization_gradient.detach(),
            data_loss, curvature.detach())


def _dual_gpu_loss_only(interior: torch.Tensor, batches: list[_DeviceBatch],
                        cutoff: float, executor: ThreadPoolExecutor,
                        curvature_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    futures = [executor.submit(_batch_loss_only, interior.detach(), batch, cutoff)
               for batch in batches]
    values = [future.result() for future in futures]
    target = interior.device
    data_loss = sum(batch.weight * value.to(target)
                    for batch, value in zip(batches, values))
    control = _control_from_interior(
        interior, batches[0].objective.demo.inversion.profile_control)
    curvature = curvature_weight * _profile_curvature(control)
    return data_loss + curvature, data_loss, curvature


def run_axial_profile_fwi(observed: torch.Tensor, static: torch.Tensor,
                          objective: ObjectiveConfig3D, initial_control: torch.Tensor,
                          *, production: bool = False,
                          iterations_per_band: int | None = None,
                          max_backtracks: int = 3) -> AxialFWIResult3D:
    """Four-band exact-gradient axial FWI with bounded Armijo backtracking."""
    shape = objective.demo.inversion.profile_control
    if tuple(initial_control.shape) != tuple(shape):
        raise ValueError("initial axial control has the wrong shape")
    current = initial_control[1:-1, 1:-1].detach().reshape(-1).clone()
    per_band = (iterations_per_band if iterations_per_band is not None else
                (objective.demo.inversion.production_iterations if production
                 else objective.demo.inversion.development_iterations))
    max_update = objective.demo.inversion.max_update_velocity
    history: list[dict[str, float]] = []
    best = current.clone()
    best_loss = float("inf")
    initial_loss = None
    closure_evaluations = 0
    global_iteration = 0

    dual_batches = _prepare_dual_gpu_batches(observed, static, objective)
    with ThreadPoolExecutor(max_workers=2) as executor:
        for cutoff_value in objective.demo.inversion.lowpass_frequencies:
            cutoff = float(cutoff_value)
            current = current.to(torch.device("cuda:0"))
            entry, _, _ = _dual_gpu_loss_only(
                current, dual_batches, cutoff, executor,
                objective.demo.inversion.curvature_weight)
            entry_value = float(entry)
            if initial_loss is None:
                initial_loss = entry_value
            objective_scale = objective.demo.inversion.normalized_initial_objective / max(
                entry_value, torch.finfo(entry.dtype).tiny)
            stagnant = 0
            for band_iteration in range(per_band):
                loss, gradient, data_loss, curvature = _dual_gpu_loss_gradient(
                    current, dual_batches, cutoff, executor,
                    objective.demo.inversion.curvature_weight)
                closure_evaluations += 1
                gradient_max = gradient.abs().max().clamp_min(torch.finfo(gradient.dtype).tiny)
                direction = -max_update * gradient / gradient_max
                directional_derivative = (gradient * direction).sum()
                accepted = False
                accepted_loss, accepted_data, accepted_curvature = loss, data_loss, curvature
                candidate = current
                step_fraction = 0.0
                line_search_evaluations = 0
                for backtrack in range(max_backtracks + 1):
                    alpha = 0.5 ** backtrack
                    trial = current + alpha * direction
                    trial_loss, trial_data, trial_curvature = _dual_gpu_loss_only(
                        trial, dual_batches, cutoff, executor,
                        objective.demo.inversion.curvature_weight)
                    line_search_evaluations += 1
                    armijo_bound = loss + 1e-4 * alpha * directional_derivative
                    if float(trial_loss) <= float(armijo_bound):
                        accepted = True
                        candidate, accepted_loss = trial, trial_loss
                        accepted_data, accepted_curvature = trial_data, trial_curvature
                        step_fraction = alpha
                        break
                physical = float(accepted_loss)
                previous_loss = float(loss)
                improvement = ((previous_loss - physical) /
                               max(previous_loss, torch.finfo(loss.dtype).tiny))
                update = float((candidate - current).abs().max())
                if accepted:
                    current = candidate
                stagnant = stagnant + 1 if improvement < objective.demo.inversion.early_stop_relative else 0
                history.append({
                    "frequency_hz": cutoff, "global_iteration": global_iteration,
                    "band_iteration": band_iteration, "physical_loss": physical,
                    "normalized_loss": physical * objective_scale,
                    "data_loss": float(accepted_data),
                    "curvature": float(accepted_curvature),
                    "regularization_fraction": float(accepted_curvature /
                        accepted_loss.clamp_min(torch.finfo(accepted_loss.dtype).tiny)),
                    "max_velocity_update": update, "projected": 0.0,
                    "relative_improvement": improvement,
                    "closure_evaluations": float(closure_evaluations),
                    "line_search_evaluations": float(line_search_evaluations),
                    "step_fraction": step_fraction, "accepted": float(accepted),
                    "gradient_norm": float(gradient.norm()),
                    "gradient_max": float(gradient.abs().max()),
                })
                global_iteration += 1
                if not accepted or stagnant >= objective.demo.inversion.early_stop_patience:
                    break
            best = current.clone()
            best_loss = physical if history else entry_value

    if not history:
        best = current
        best_loss = float(initial_loss if initial_loss is not None else 0.0)
    control = _control_from_interior(best, shape)
    velocity = axial_profile_to_velocity(control, objective.demo.grid)
    return AxialFWIResult3D(control, velocity, history, float(initial_loss),
                            float(best_loss), closure_evaluations)
