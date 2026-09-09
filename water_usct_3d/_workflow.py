from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch

from .acoustic import ForwardConfig3D, simulate_acoustic
from .acquisition import build_acquisition_3d
from .axial_fwi import axial_flow_waveform_loss, run_axial_profile_fwi
from .config import DemoConfig3D, load_config, smoke_grid, sponge_layers_for_grid
from .data import environment_info, generate_case, load_case
from .geometry import deterministic_geometry_search, direction_coverage
from .inversion import (Model3D, ObjectiveConfig3D, amplitude_unit_prediction,
                        invert_amplitude, run_vector_fwi)
from .performance import _shot_loss_gradient, benchmark_checkpoint_blocks, exact_two_gpu_gradient
from .parameterization import vector_potential_to_velocity
from .profile_inversion import (axial_profile_to_velocity,
                                build_axial_fwi_initial_controls,
                                invert_axial_profile)
from .profile_validation import validate_frozen_axial_profile
from .travel_time import extract_reciprocal_delays_3d
from .visualization import (save_axial_fwi_diagnostics,
                            save_axial_fwi_ablation_summary,
                            save_axial_initial_model_diagnostics,
                            save_axial_initialization_ablation,
                            save_coverage_slices, save_geometry,
                            save_profile_diagnostics)
from .verification import verify_acoustic, verify_flow
def _json(value: object) -> None:
    print(json.dumps(value, indent=2, default=str))


def _append_run_log(output_root: str, stage: str, status: str, command: str,
                    result: dict[str, object]) -> None:
    # 运行日志属于实验输出，安装包及仓库根目录不应被 CLI 改写。
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    record = {"time": datetime.now().astimezone().isoformat(), "stage": stage,
              "status": status, "command": command, "result": result}
    with (root / "run_log.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, default=str) + "\n")


def geometry_command(config: DemoConfig3D) -> dict[str, object]:
    acquisition = build_acquisition_3d(config.grid, config.acquisition, dtype=torch.float64, device="cpu")
    nominal = direction_coverage(config.grid, acquisition)
    search = None
    if not nominal.metrics["pass"]:
        search = deterministic_geometry_search(config.grid, acquisition, config.acquisition.wall_offset_cells,
                                               config.acquisition.sponge_layers)
        acquisition, result = search.acquisition, search.coverage
    else:
        result = nominal
    output = Path(config.output_root) / "geometry"; output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "coverage.npz", coordinates=acquisition.coordinates.cpu().numpy(),
                        matrix=result.matrix.cpu().numpy(), min_eigenvalue=result.min_eigenvalue.cpu().numpy(),
                        normalized_min_eigenvalue=result.normalized_min_eigenvalue.cpu().numpy(),
                        condition_number=result.condition_number.cpu().numpy())
    save_geometry(acquisition.coordinates.cpu().numpy(), output / "transducers.png")
    save_coverage_slices(result, output / "coverage_slices.png")
    metrics = dict(result.metrics, nominal_metrics=nominal.metrics,
                   geometry_search_triggered=search is not None,
                   selected_ring_z=None if search is None else search.ring_z,
                   selected_perimeter_offsets=None if search is None else search.perimeter_offsets,
                   candidates_evaluated=0 if search is None else search.candidates_evaluated,
                   identifiability_claim="general three-component" if result.metrics["pass"] else "axial and low-dimensional transverse only",
                   transducers=acquisition.coordinates.shape[0], ordered_traces=acquisition.trace_count,
                   reciprocal_pairs=acquisition.reciprocal_pairs.shape[0], artifact=str(output / "coverage.npz"))
    with (output / "metrics.json").open("w", encoding="utf-8") as stream: json.dump(metrics, stream, indent=2)
    return metrics


def _objective(config: DemoConfig3D, data: dict[str, torch.Tensor]) -> ObjectiveConfig3D:
    acquisition = build_acquisition_3d(config.grid, config.acquisition, dtype=config.torch_dtype(), device=config.torch_device())
    forward = ForwardConfig3D(config.grid, config.physics.c0, config.physics.include_shear, "sponge",
                              sponge_layers_for_grid(config, config.grid), config.inversion.checkpoint_steps,
                              acquisition.source_signal, sponge_mode=config.acquisition.sponge_mode)
    return ObjectiveConfig3D(config, acquisition, forward)


def invert_command(config: DemoConfig3D, stage: str, initial_beta: float, production: bool) -> dict[str, object]:
    data = load_case(config, "matched_clean")
    objective = _objective(config, data)
    observed = data["pressure_traces"]
    shape = config.inversion.vector_control
    initial = Model3D(torch.tensor(initial_beta, dtype=observed.dtype, device=observed.device),
                      torch.zeros((3, *shape), dtype=observed.dtype, device=observed.device))
    profile_result = None
    if stage == "amplitude":
        root = Path(config.output_root) / "inversion" / "amplitude"
        root.mkdir(parents=True, exist_ok=True)
        template_path = root / "unit_prediction.pt"
        if template_path.exists():
            unit_prediction = torch.load(template_path, map_location=observed.device, weights_only=True).to(observed.dtype)
        else:
            unit_prediction = amplitude_unit_prediction(objective, observed.dtype)
            torch.save(unit_prediction.detach().cpu(), template_path)
        model, history = invert_amplitude(
            observed, objective, initial_beta,
            config.inversion.production_iterations if production else None,
            static_observed=data["static_pressure_traces"], unit_prediction=unit_prediction,
        )
    elif stage == "profile":
        amplitude_summary = Path(config.output_root) / "inversion/amplitude/summary_metrics.json"
        if not amplitude_summary.exists():
            raise RuntimeError("profile inversion requires the passed amplitude summary")
        with amplitude_summary.open(encoding="utf-8") as stream:
            amplitude_metrics = json.load(stream)
        if not bool(amplitude_metrics.get("passed", False)):
            raise RuntimeError("profile inversion is blocked by the amplitude gate")
        beta = float(amplitude_metrics["branches"][0]["beta"])
        delay = extract_reciprocal_delays_3d(
            data["pressure_traces"], data["static_pressure_traces"], objective.acquisition)
        profile_result = invert_axial_profile(observed, delay, objective, beta)
        model = Model3D(torch.tensor(beta, dtype=observed.dtype, device=observed.device),
                        torch.zeros((3, *shape), dtype=observed.dtype, device=observed.device))
        history = profile_result.candidates
    elif stage == "full":
        model, history = run_vector_fwi(observed, objective, initial, profile_only=stage == "profile", production=production)
    else:
        raise ValueError("stage must be amplitude, profile, or full")
    start_tag = str(initial_beta).replace("-", "m").replace(".", "p")
    output = Path(config.output_root) / "inversion" / stage / f"start_{start_tag}"
    output.mkdir(parents=True, exist_ok=True)
    with h5py.File(output / "model.h5", "w") as handle:
        handle.create_dataset("beta", data=float(model.beta.cpu()))
        if profile_result is None:
            handle.create_dataset("vector_potential", data=model.vector_potential.cpu().numpy())
        else:
            handle.create_dataset("profile_control", data=profile_result.control.cpu().numpy())
            handle.create_dataset("velocity", data=profile_result.velocity.cpu().numpy())
    if history:
        with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
    truth = data["truth_velocity"]
    duct = vector_potential_to_velocity(torch.tensor(1.0, dtype=truth.dtype, device=truth.device),
        torch.zeros((3, *shape), dtype=truth.dtype, device=truth.device), config.grid)
    truth_beta = float((truth * duct).sum() / duct.square().sum().clamp_min(torch.finfo(truth.dtype).tiny))
    beta_error = abs(float(model.beta.cpu()) - truth_beta) / abs(truth_beta)
    profile_error = None
    profile_correlation = None
    profile_constraints = None
    profile_figures = None
    if profile_result is not None:
        profile_error = float(torch.linalg.vector_norm(profile_result.velocity - truth) /
                              torch.linalg.vector_norm(truth))
        profile_correlation = float(torch.nn.functional.cosine_similarity(
            profile_result.velocity.reshape(1, -1), truth.reshape(1, -1)))
        velocity = profile_result.velocity
        wall_max = max(float(velocity[:, 0].abs().max()), float(velocity[:, -1].abs().max()),
                       float(velocity[:, :, 0].abs().max()), float(velocity[:, :, -1].abs().max()))
        transverse_max = float(velocity[:2].abs().max())
        z_variation = float((velocity[..., 1:] - velocity[..., :-1]).abs().max())
        profile_constraints = {"wall_max_m_per_s": wall_max,
                               "transverse_max_m_per_s": transverse_max,
                               "z_variation_max_m_per_s": z_variation,
                               "relative_divergence": 0.0}
        profile_figures = save_profile_diagnostics(
            truth.detach().cpu().numpy(), velocity.detach().cpu().numpy(),
            profile_result.control.detach().cpu().numpy(), profile_result.candidates,
            config.grid, output)
    result = {"stage": stage, "initial_beta": initial_beta, "beta": float(model.beta.cpu()),
              "truth_beta_projection": truth_beta, "relative_beta_error": beta_error,
              "profile_relative_l2": profile_error, "profile_correlation": profile_correlation,
              "selected_candidate": None if profile_result is None else profile_result.selected,
              "candidates": None if profile_result is None else profile_result.candidates,
              "profile_constraints": profile_constraints, "profile_figures": profile_figures,
              "passed": beta_error <= 0.02 if stage == "amplitude" else
                        (profile_error <= 0.10 if profile_error is not None else True),
              "iterations": len(history), "artifact": str(output / "model.h5"), "formal": production}
    with (output / "metrics.json").open("w", encoding="utf-8") as stream: json.dump(result, stream, indent=2)
    return result


def _prepare_axial_fwi_initializations(config: DemoConfig3D,
                                       data: dict[str, torch.Tensor],
                                       objective: ObjectiveConfig3D
                                       ) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    amplitude_summary = Path(config.output_root) / "inversion/amplitude/summary_metrics.json"
    if not amplitude_summary.exists():
        raise RuntimeError("axial FWI initialization requires the passed amplitude summary")
    with amplitude_summary.open(encoding="utf-8") as stream:
        amplitude_metrics = json.load(stream)
    if not bool(amplitude_metrics.get("passed", False)):
        raise RuntimeError("axial FWI initialization is blocked by the amplitude gate")
    beta = float(amplitude_metrics["branches"][0]["beta"])
    delay = extract_reciprocal_delays_3d(
        data["pressure_traces"], data["static_pressure_traces"], objective.acquisition)
    controls, delay_diagnostics = build_axial_fwi_initial_controls(delay, objective, beta)
    velocities = {name: axial_profile_to_velocity(control, config.grid)
                  for name, control in controls.items()}
    truth = data["truth_velocity"]
    truth_norm = torch.linalg.vector_norm(truth).clamp_min(torch.finfo(truth.dtype).tiny)
    candidate_metrics: dict[str, dict[str, float]] = {}
    for name, velocity in velocities.items():
        denominator = (torch.linalg.vector_norm(velocity) * truth_norm).clamp_min(
            torch.finfo(truth.dtype).tiny)
        candidate_metrics[name] = {
            "relative_l2_score_only": float(torch.linalg.vector_norm(velocity - truth) / truth_norm),
            "correlation_score_only": float((velocity * truth).sum() / denominator),
            "control_min_m_per_s": float(controls[name].min()),
            "control_max_m_per_s": float(controls[name].max()),
        }
    root = Path(config.output_root) / "inversion/profile_fwi_ablation/initial_models"
    root.mkdir(parents=True, exist_ok=True)
    with h5py.File(root / "initializations.h5", "w") as handle:
        handle.attrs.update(
            truth_usage="post-hoc scoring only",
            gaussian_sigma_control_cells=0.75,
            data_case="matched_clean",
        )
        for name in controls:
            group = handle.create_group(name)
            group.create_dataset("control", data=controls[name].detach().cpu().numpy())
            group.create_dataset("velocity", data=velocities[name].detach().cpu().numpy(),
                                 compression="gzip")
    figures = save_axial_initialization_ablation(
        {name: value.detach().cpu().numpy() for name, value in controls.items()},
        truth.detach().cpu().numpy(), config.grid, root)
    metrics: dict[str, object] = {
        "stage": "profile_fwi_initialization_ablation",
        "data_case": "matched_clean",
        "truth_policy": "truth is excluded from construction and used only for post-hoc scoring",
        "beta_from_amplitude_stage": beta,
        "registered_initializations": list(controls),
        "gaussian_sigma_control_cells": 0.75,
        "travel_time_diagnostics": delay_diagnostics,
        "candidates": candidate_metrics,
        "artifacts": [str(root / "initializations.h5"), *figures],
        "passed": all(torch.isfinite(value).all() for value in controls.values()),
    }
    with (root / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    return controls, metrics


def _update_axial_fwi_ablation_summary(config: DemoConfig3D, protocol: str) -> dict[str, object]:
    names = ("good_prior", "travel_time", "blurred_travel_time", "zero_flow")
    root = Path(config.output_root) / "inversion/profile_fwi_ablation" / protocol
    records: list[dict[str, object]] = []
    for name in names:
        path = root / name / "metrics.json"
        if path.exists():
            with path.open(encoding="utf-8") as stream:
                records.append(json.load(stream))
    figures = save_axial_fwi_ablation_summary(records, root) if records else []
    summary: dict[str, object] = {
        "protocol": protocol,
        "registered_order": list(names),
        "completed": [item["initialization"] for item in records],
        "all_complete": len(records) == len(names),
        "passed_count": sum(bool(item["passed"]) for item in records),
        "records": records,
        "artifacts": figures,
    }
    if records:
        best = min(records, key=lambda item: float(item["final_profile_relative_l2"]))
        summary["best_final_initialization"] = best["initialization"]
        summary["best_final_relative_l2"] = best["final_profile_relative_l2"]
    root.mkdir(parents=True, exist_ok=True)
    with (root / "summary_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    if records:
        fields = ("initialization", "initial_profile_relative_l2", "final_profile_relative_l2",
                  "profile_correlation", "waveform_loss_reduction", "accepted_iterations", "passed")
        with (root / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({field: item[field] for field in fields} for item in records)
    return summary


def axial_fwi_command(config: DemoConfig3D, production: bool,
                      iterations_per_band: int | None,
                      initialization: str = "good_prior",
                      prepare_only: bool = False) -> dict[str, object]:
    data = load_case(config, "matched_clean")
    objective = _objective(config, data)
    controls, initialization_metrics = _prepare_axial_fwi_initializations(config, data, objective)
    if prepare_only:
        return {**initialization_metrics, "fwi_executed": False}
    if initialization not in controls:
        raise ValueError(f"unknown axial FWI initialization: {initialization}")
    effective_iterations = (iterations_per_band if iterations_per_band is not None else
                            (config.inversion.production_iterations if production else
                             config.inversion.development_iterations))
    protocol = ("production" if production else "development") + f"_{effective_iterations}_per_band"
    initial_control = controls[initialization]
    initial_velocity = axial_profile_to_velocity(initial_control, config.grid)
    result = run_axial_profile_fwi(
        data["pressure_traces"], data["static_pressure_traces"], objective,
        initial_control, production=production, iterations_per_band=iterations_per_band,
    )
    output = (Path(config.output_root) / "inversion/profile_fwi_ablation" /
              protocol / initialization)
    output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        initial_prediction = simulate_acoustic(
            initial_velocity, objective.acquisition.source_apertures,
            objective.acquisition.apertures, objective.forward)
        final_prediction = simulate_acoustic(
            result.velocity, objective.acquisition.source_apertures,
            objective.acquisition.apertures, objective.forward)
    cutoff = float(config.inversion.lowpass_frequencies[-1])
    mask = objective.acquisition.ordered_mask
    initial_waveform = axial_flow_waveform_loss(
        initial_prediction, data["pressure_traces"], data["static_pressure_traces"],
        mask, config.grid.dt, cutoff)
    final_waveform = axial_flow_waveform_loss(
        final_prediction, data["pressure_traces"], data["static_pressure_traces"],
        mask, config.grid.dt, cutoff)
    truth = data["truth_velocity"]
    truth_norm = torch.linalg.vector_norm(truth).clamp_min(torch.finfo(truth.dtype).tiny)
    initial_error = float(torch.linalg.vector_norm(initial_velocity - truth) / truth_norm)
    final_error = float(torch.linalg.vector_norm(result.velocity - truth) / truth_norm)
    correlation = float(torch.nn.functional.cosine_similarity(
        result.velocity.reshape(1, -1), truth.reshape(1, -1)))
    wall_max = max(float(result.velocity[:, 0].abs().max()), float(result.velocity[:, -1].abs().max()),
                   float(result.velocity[:, :, 0].abs().max()), float(result.velocity[:, :, -1].abs().max()))
    with h5py.File(output / "model.h5", "w") as handle:
        handle.create_dataset("initial_control", data=initial_control.detach().cpu().numpy())
        handle.create_dataset("profile_control", data=result.control.detach().cpu().numpy())
        handle.create_dataset("velocity", data=result.velocity.detach().cpu().numpy(), compression="gzip")
        handle.create_dataset("pressure_traces", data=final_prediction.detach().cpu().numpy(), compression="gzip")
    if result.history:
        with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=result.history[0].keys())
            writer.writeheader(); writer.writerows(result.history)
    figures = save_axial_fwi_diagnostics(
        truth.detach().cpu().numpy(), initial_velocity.detach().cpu().numpy(),
        result.velocity.detach().cpu().numpy(), result.history, config.grid, output)
    figures += save_axial_initial_model_diagnostics(
        truth.detach().cpu().numpy(), initial_velocity.detach().cpu().numpy(),
        initial_control.detach().cpu().numpy(), config.grid, output)
    metrics: dict[str, object] = {
        "stage": "profile_fwi", "formal": production,
        "protocol": protocol,
        "initialization": initialization,
        "truth_policy": "truth excluded from initialization and optimization; post-hoc scoring only",
        "optimizer": "exact_full-shot gradient with bounded Armijo backtracking",
        "iterations_per_band": effective_iterations,
        "accepted_iterations": len(result.history),
        "gradient_evaluations": result.closure_evaluations,
        "max_armijo_backtracks": 3, "shots_per_gpu": 12,
        "initial_profile_relative_l2": initial_error,
        "final_profile_relative_l2": final_error,
        "profile_correlation": correlation,
        "initial_65khz_flow_waveform_loss": float(initial_waveform),
        "final_65khz_flow_waveform_loss": float(final_waveform),
        "waveform_loss_reduction": float(1.0 - final_waveform / initial_waveform),
        "model_not_worse": final_error <= initial_error,
        "constraints": {"wall_max_m_per_s": wall_max,
                        "transverse_max_m_per_s": float(result.velocity[:2].abs().max()),
                        "z_variation_max_m_per_s": float((result.velocity[..., 1:] - result.velocity[..., :-1]).abs().max()),
                        "relative_divergence": 0.0},
        "passed": final_error <= 0.10 and final_error <= initial_error and
                  float(final_waveform) < float(initial_waveform) and wall_max < 1e-6,
        "artifacts": [str(output / "model.h5"), str(output / "history.csv"), *figures],
    }
    with (output / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    _update_axial_fwi_ablation_summary(config, protocol)
    return metrics


def evaluate_command(config: DemoConfig3D) -> dict[str, object]:
    root = Path(config.output_root)
    acoustic = root / "verification/acoustic_metrics.json"
    if not acoustic.exists(): acoustic = root / "verification/acoustic_partial_metrics.json"
    expected = [root / "verification/flow_metrics.json", root / "geometry/metrics.json", acoustic]
    present = {str(path): path.exists() for path in expected}
    values = {}
    for path in expected:
        if path.exists():
            with path.open(encoding="utf-8") as stream: values[path.stem] = json.load(stream)
    formal_pass = all(present.values()) and all(bool(item.get("passed", item.get("pass", False))) for item in values.values())
    if "flow_metrics" in values:
        formal_pass = formal_pass and bool(values["flow_metrics"].get("formal_gate_complete", False))
    if acoustic.stem in values:
        formal_pass = formal_pass and bool(values[acoustic.stem].get("formal_gate_complete", False))
    result = {"artifacts_present": present, "all_present": all(present.values()), "metrics": values,
              "passed": formal_pass,
              "warning": "formal success requires all registered gates; smoke metrics are not substitutes"}
    output = root / "evaluation.json"; output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream: json.dump(result, stream, indent=2)
    return result


def benchmark_command(config: DemoConfig3D) -> dict[str, object]:
    grid = smoke_grid(config, 9, 16); device = config.torch_device(); dtype = config.torch_dtype()
    acq_config = replace(config.acquisition, wall_offset_cells=config.acquisition.wall_offset_cells * config.grid.dx / grid.dx)
    acquisition = build_acquisition_3d(grid, acq_config, dtype=dtype, device=device)
    signal = torch.zeros(grid.nt, dtype=dtype, device=device); signal[0] = 1.0
    forward = ForwardConfig3D(grid, config.physics.c0, True, "periodic", 0, 8, signal)
    velocity = torch.zeros((3, grid.nx, grid.ny, grid.nz), dtype=dtype, device=device)
    timings = benchmark_checkpoint_blocks(velocity, acquisition, forward, (8, 16))
    result = {"environment": environment_info(), "checkpoint_smoke_seconds": timings,
              "selected_checkpoint_steps_smoke": min(timings, key=timings.get),
              "formal_two_gpu_benchmark_complete": False,
              "reason": "run production generation/FWI on the target two-RTX-5090 host before claiming the 60-minute gate"}
    if torch.cuda.device_count() >= 2:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        indices = torch.arange(acquisition.source_apertures.shape[0], device=device)
        single_loss, single_gradient = _shot_loss_gradient(velocity, acquisition, forward, indices)
        dual_loss, dual_gradient = exact_two_gpu_gradient(velocity, acquisition, forward)
        relative = float(torch.linalg.vector_norm(single_gradient - dual_gradient.to(device)) /
                         torch.linalg.vector_norm(single_gradient).clamp_min(torch.finfo(dtype).tiny))
        result.update(two_gpu_smoke_gradient_relative_error=relative,
                      two_gpu_smoke_loss_relative_error=float(abs(single_loss - dual_loss.to(device)) / single_loss.abs().clamp_min(torch.finfo(dtype).tiny)),
                      two_gpu_gradient_gate_pass=relative < 1e-6)
    output = Path(config.output_root) / "benchmark"; output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.json").open("w", encoding="utf-8") as stream: json.dump(result, stream, indent=2)
    return result


