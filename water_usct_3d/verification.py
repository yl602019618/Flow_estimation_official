from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as functional

from .acoustic import ForwardConfig3D, _rk4, background_gradient, simulate_acoustic
from .acquisition import build_acquisition_3d
from .config import DemoConfig3D, Grid3D, fine_grid, smoke_grid, sponge_layers_for_grid
from .flow import FlowConfig, manufactured_flow, simulate_flow
from .flow import FlowResult, square_duct_poiseuille
from .mac import empty_mac_state, enforce_no_slip, mac_divergence, nodes_to_mac, project_mac
from .operators import derivative8, divergence, relative_norm, staggered_gradient8
from .visualization import save_acoustic_grid_diagnostics, save_flow_artifacts


def _write_metrics(config: DemoConfig3D, stage: str, metrics: dict[str, object]) -> Path:
    path = Path(config.output_root) / "verification"
    path.mkdir(parents=True, exist_ok=True)
    destination = path / f"{stage}_metrics.json"
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    return destination


def _valid_receiver_grid_mismatch(coarse: torch.Tensor, fine: torch.Tensor,
                                  ordered_mask: torch.Tensor) -> tuple[float, float]:
    """Return the registered non-self mismatch and all-channel diagnostic."""
    valid = ordered_mask[0]
    registered = relative_norm(coarse[:, valid] - fine[:, valid], fine[:, valid])
    including_self = relative_norm(coarse - fine, fine)
    return registered, including_self


def _manufactured_order(lengths: tuple[float, float, float]) -> tuple[float, list[dict[str, float]]]:
    errors, spacings, records = [], [], []
    for n in (17, 25, 33):
        grid = Grid3D(n, n, n + 2, *lengths, 1e-7, 1e-6, 8)
        result = simulate_flow(FlowConfig(grid, dtype=torch.float64, device=torch.device("cpu"), mode="manufactured"))
        errors.append(result.metrics["relative_l2"])
        spacings.append(grid.dx)
        records.append({"n": float(n), "spacing": grid.dx, "relative_l2": result.metrics["relative_l2"],
                        "mac_relative_divergence": result.metrics["mac_relative_divergence"],
                        "pcg_relative_residual": result.metrics["pcg_relative_residual"]})
    slope = float(torch.linalg.lstsq(
        torch.stack((torch.log(torch.tensor(spacings)), torch.ones(3)), dim=1),
        torch.log(torch.tensor(errors)),
    ).solution[0])
    return slope, records


def _float32_projection_metric(grid: Grid3D, seed: int) -> tuple[float, float]:
    small = Grid3D(17, 17, 19, grid.length_x, grid.length_y, grid.length_z,
                   grid.dt, min(grid.record_time, 10 * grid.dt), grid.spatial_order)
    torch.manual_seed(seed)
    state = empty_mac_state(small, dtype=torch.float32, device=torch.device("cpu"))
    state.u.normal_(); state.v.normal_(); state.w.normal_(); enforce_no_slip(state)
    before = torch.linalg.vector_norm(mac_divergence(state, small))
    projected, history = project_mac(state, small, 1e-4, 998.0, rtol=5e-6, maxiter=5000)
    relative = float(torch.linalg.vector_norm(mac_divergence(projected, small)) / before)
    return relative, history[-1]


def verify_flow(config: DemoConfig3D) -> dict[str, object]:
    result = simulate_flow(FlowConfig(config.grid, config.physics.max_velocity,
                                      config.physics.effective_reynolds, config.physics.rho0,
                                      torch.float64, torch.device("cpu"), "duct"))
    metrics: dict[str, object] = dict(result.metrics)
    projection32, pcg32 = _float32_projection_metric(config.grid, config.seed)
    metrics["relative_divergence_float32"] = projection32
    metrics["pcg_relative_residual_float32"] = pcg32
    manufactured_order, manufactured_records = _manufactured_order(
        (config.grid.length_x, config.grid.length_y, config.grid.length_z))
    metrics["manufactured_order"] = manufactured_order
    metrics["manufactured_convergence"] = manufactured_records
    limits = config.verification
    checks = {
        "duct_l2": metrics["relative_l2"] <= limits["flow_l2_max"],
        "manufactured_order": metrics["manufactured_order"] >= limits["manufactured_order_min"],
        "divergence": metrics["mac_relative_divergence"] <= limits["divergence_float64_max"],
        "divergence_float32": metrics["relative_divergence_float32"] <= limits["divergence_float32_max"],
        "wall": metrics["wall_velocity_relative"] <= limits["wall_velocity_max"],
        "flow_rate": metrics["flow_rate_variation"] <= limits["flow_rate_variation_max"],
        "momentum": metrics["momentum_residual"] <= limits["momentum_residual_max"],
    }
    reference = square_duct_poiseuille(config.grid, config.physics.max_velocity,
        dtype=torch.float64, device=torch.device("cpu"))
    manufactured_velocity, _ = manufactured_flow(config.grid,
        config.physics.max_velocity, 0.1, dtype=torch.float64, device=torch.device("cpu"))
    manufactured_solver = simulate_flow(FlowConfig(config.grid, config.physics.max_velocity,
        config.physics.effective_reynolds, config.physics.rho0, torch.float64,
        torch.device("cpu"), "manufactured"))
    metrics["production_manufactured_relative_l2"] = manufactured_solver.metrics["relative_l2"]
    metrics["production_manufactured_mac_divergence"] = manufactured_solver.metrics["mac_relative_divergence"]
    metrics["production_manufactured_step_change"] = manufactured_solver.metrics["steady_relative_change"]
    metrics["production_manufactured_pcg_residual"] = manufactured_solver.metrics["pcg_relative_residual"]
    output = Path(config.output_root) / "flow"
    artifacts = save_flow_artifacts(result, manufactured_solver,
        reference.detach().cpu().numpy(), manufactured_velocity.detach().cpu().numpy(),
        config.grid, output, manufactured_records)
    metrics.update(checks=checks, passed=all(checks.values()), formal_gate_complete=all(checks.values()),
                   gate_scope="production-grid MAC RK2 projection plus manufactured convergence",
                   artifacts=artifacts)
    metrics["metrics_path"] = str(_write_metrics(config, "flow", metrics))
    return metrics


def _derivative_order() -> float:
    errors, spacings = [], []
    for n in (17, 25, 33, 49):
        x = torch.arange(n, dtype=torch.float64) / n
        value = torch.sin(2 * torch.pi * x)
        faces = (torch.arange(n, dtype=torch.float64) + 0.5) / n
        exact = 2 * torch.pi * torch.cos(2 * torch.pi * faces)
        numerical = staggered_gradient8(value, 0, 1.0 / n, periodic=True)
        errors.append(float(torch.linalg.vector_norm(numerical - exact) / torch.linalg.vector_norm(exact)))
        spacings.append(1.0 / n)
    design = torch.stack((torch.log(torch.tensor(spacings)), torch.ones(4)), dim=1)
    return float(torch.linalg.lstsq(design, torch.log(torch.tensor(errors))).solution[0])


def _dispersion_error(config: DemoConfig3D, velocity: float = 0.0) -> float:
    k = 2 * math.pi * config.acquisition.center_frequency / config.physics.c0
    h = config.grid.dz
    staggered_symbol = 2.0 * sum(
        coefficient * math.sin((radius - 0.5) * k * h)
        for radius, coefficient in enumerate((1225/1024, -245/3072, 49/5120, -5/7168), start=1)
    ) / h
    advective_symbol = 2.0 * sum(
        coefficient * math.sin(radius * k * h)
        for radius, coefficient in {1: 4/5, 2: -1/5, 3: 4/105, 4: -1/280}.items()
    ) / h
    omega = config.physics.c0 * staggered_symbol + velocity * advective_symbol
    q = omega * config.grid.dt
    amplification = 1 + 1j*q - q*q/2 - 1j*q**3/6 + q**4/24
    omega_num = abs(math.atan2(amplification.imag, amplification.real)) / config.grid.dt
    return abs(omega_num / ((config.physics.c0 + velocity) * k) - 1.0)


def _taylor_test(grid: Grid3D, acquisition, forward: ForwardConfig3D, seed: int) -> tuple[float, float]:
    torch.manual_seed(seed)
    base = (0.01 * torch.randn((3, grid.nx, grid.ny, grid.nz), dtype=acquisition.apertures.dtype,
                               device=acquisition.apertures.device)).requires_grad_(True)
    direction = torch.randn_like(base); direction = direction / torch.linalg.vector_norm(direction)
    def objective(value: torch.Tensor) -> torch.Tensor:
        return simulate_acoustic(value, acquisition.source_apertures[:1], acquisition.apertures[:3], forward).square().mean()
    value = objective(base); gradient = torch.autograd.grad(value, base)[0]
    directional = (gradient * direction).sum()
    epsilons = (1.0, 0.5, 0.25, 0.125)
    remainders = []
    for epsilon in epsilons:
        perturbed = objective(base.detach() + epsilon * direction)
        remainders.append(abs(perturbed - value.detach() - epsilon * directional))
    slopes = [float(torch.log(remainders[i] / remainders[i + 1]) / math.log(2.0)) for i in range(len(remainders) - 1)]
    epsilon = 1e-2
    finite_difference = (objective(base.detach() + epsilon * direction) - objective(base.detach() - epsilon * direction)) / (2 * epsilon)
    relative_error = float(abs(finite_difference - directional) / finite_difference.abs().clamp_min(torch.finfo(base.dtype).tiny))
    return min(slopes), relative_error


def verify_acoustic(config: DemoConfig3D, *, smoke: bool = False) -> dict[str, object]:
    grid = smoke_grid(config, 9, 16) if smoke else config.grid
    dtype, device = torch.float64, torch.device("cpu") if smoke else config.torch_device()
    acq_config = config.acquisition
    if smoke:
        acq_config = replace(acq_config, wall_offset_cells=config.acquisition.wall_offset_cells * config.grid.dx / grid.dx)
    acquisition = build_acquisition_3d(grid, acq_config, dtype=dtype, device=device)
    if smoke:
        # An impulse provides observable smoke traces within a short record.
        signal = torch.zeros(grid.nt, dtype=dtype, device=device); signal[0] = 1.0
    else:
        signal = acquisition.source_signal
    forward = ForwardConfig3D(grid, config.physics.c0, True, "rigid" if smoke else "sponge",
                              0 if smoke else sponge_layers_for_grid(config, grid),
                              min(config.inversion.checkpoint_steps, grid.nt), signal,
                              sponge_mode=config.acquisition.sponge_mode)
    zero = torch.zeros((3, grid.nx, grid.ny, grid.nz), dtype=dtype, device=device)
    with torch.no_grad():
        traces = simulate_acoustic(zero, acquisition.source_apertures, acquisition.apertures, forward)
    reciprocal = torch.stack([traces[a, b] - traces[b, a] for a, b in acquisition.reciprocal_pairs.tolist()])
    reference = torch.stack([traces[a, b] for a, b in acquisition.reciprocal_pairs.tolist()])
    reciprocity = relative_norm(reciprocal, reference)
    # Shear toggle on a deterministic manufactured background.
    velocity, _ = manufactured_flow(grid, 0.2, 0.05, dtype=dtype, device=device)
    with torch.no_grad():
        shear = simulate_acoustic(velocity, acquisition.source_apertures[:1], acquisition.apertures[:2], forward)
        no_shear = simulate_acoustic(velocity, acquisition.source_apertures[:1], acquisition.apertures[:2], replace(forward, include_shear=False))
    shear_difference = relative_norm(shear - no_shear, shear)
    # Periodic no-source energy check on a smooth pressure mode.
    x, y, z = grid.coordinates(dtype=dtype, device=device)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
    state = torch.zeros((1, 4, grid.nx, grid.ny, grid.nz), dtype=dtype, device=device)
    state[:, 0] = torch.sin(2*torch.pi*xx/grid.length_x) * torch.sin(2*torch.pi*yy/grid.length_y) * torch.sin(2*torch.pi*zz/grid.length_z)
    initial_energy = state.square().sum()
    energy_forward = replace(forward, boundary="periodic", sponge_layers=0)
    grad = background_gradient(zero, energy_forward)
    for _ in range(grid.nt):
        state = _rk4(state, zero, grad, energy_forward, torch.zeros_like(state[:, 0]))
    energy_drift = float(abs(state.square().sum() / initial_energy - 1.0))
    taylor_grid = grid
    taylor_acquisition = acquisition
    taylor_forward = forward
    if not smoke:
        taylor_grid = replace(grid, record_time=min(grid.nt, 96) * grid.dt)
        taylor_acquisition = replace(acquisition, source_signal=signal[:taylor_grid.nt])
        taylor_forward = replace(forward, grid=taylor_grid, source_signal=signal[:taylor_grid.nt], checkpoint_steps=min(32, taylor_grid.nt))
    taylor_slope, taylor_error = _taylor_test(taylor_grid, taylor_acquisition, taylor_forward, config.seed)
    grid_mismatch = None
    grid_mismatch_including_self = None
    grid_diagnostics: dict[str, object] = {}
    if not smoke:
        fine = fine_grid(config)
        fine_acq_config = replace(config.acquisition,
            wall_offset_cells=config.acquisition.wall_offset_cells * config.grid.dx / fine.dx)
        fine_acquisition = build_acquisition_3d(fine, fine_acq_config, dtype=dtype, device=device)
        coarse_forward = replace(forward, boundary="sponge", sponge_layers=sponge_layers_for_grid(config, grid))
        fine_forward = ForwardConfig3D(fine, config.physics.c0, True, "sponge", sponge_layers_for_grid(config, fine),
                                       config.inversion.checkpoint_steps, fine_acquisition.source_signal,
                                       sponge_mode=config.acquisition.sponge_mode)
        with torch.no_grad():
            coarse_trace = simulate_acoustic(zero, acquisition.source_apertures[:1], acquisition.apertures, coarse_forward)
            fine_zero = torch.zeros((3, fine.nx, fine.ny, fine.nz), dtype=dtype, device=device)
            fine_trace = simulate_acoustic(fine_zero, fine_acquisition.source_apertures[:1], fine_acquisition.apertures, fine_forward)
        fine_trace = functional.interpolate(fine_trace.reshape(-1, 1, fine.nt), size=grid.nt,
                                            mode="linear", align_corners=True).reshape_as(coarse_trace)
        from .inversion import lowpass
        coarse_band = lowpass(coarse_trace, grid.dt, 65_000.0)
        fine_band = lowpass(fine_trace, grid.dt, 65_000.0)
        grid_mismatch, grid_mismatch_including_self = _valid_receiver_grid_mismatch(
            coarse_band, fine_band, acquisition.ordered_mask)
        grid_diagnostics = save_acoustic_grid_diagnostics(
            coarse_band.detach().cpu().numpy(), fine_band.detach().cpu().numpy(),
            reciprocal.detach().cpu().numpy(), reference.detach().cpu().numpy(), grid.dt,
            Path(config.output_root) / "acoustic",
        )
    metrics: dict[str, object] = {
        "derivative_order": _derivative_order(), "phase_error": _dispersion_error(config),
        "doppler_plus_error": _dispersion_error(config, 1.0), "doppler_minus_error": _dispersion_error(config, -1.0),
        "reciprocity_error": reciprocity, "shear_relative_difference": shear_difference,
        "energy_drift": energy_drift, "finite": bool(torch.isfinite(traces).all()),
        "gate_scope": "smoke" if smoke else "production grid selected gates",
        "acoustic_layout": "pressure nodes; normal acoustic velocities on half-grid faces; eighth-order mass-adjoint coupling",
        "boundary_contract": {
            "x_y": "rigid: pressure/tangential even, normal velocity odd and zero",
            "z": "pressure-only damping sponge terminated by rigid parity plane",
            "sponge_mode": config.acquisition.sponge_mode,
            "physical_width_m": config.acquisition.sponge_layers * config.grid.dz,
            "production_layers": sponge_layers_for_grid(config, config.grid),
            "fine_layers": sponge_layers_for_grid(config, fine_grid(config)),
        },
        "grid_mismatch": grid_mismatch,
        "grid_mismatch_including_self_diagnostic": grid_mismatch_including_self,
        "grid_mismatch_receiver_count": 23 if not smoke else None,
        "taylor_slope": taylor_slope, "taylor_gradient_error": taylor_error,
    }
    limits = config.verification
    checks = {
        "derivative_order": metrics["derivative_order"] >= limits["derivative_order_min"],
        "phase": metrics["phase_error"] <= limits["phase_error_max"],
        "doppler": max(metrics["doppler_plus_error"], metrics["doppler_minus_error"]) <= limits["phase_error_max"],
        "shear": metrics["shear_relative_difference"] > 0.0,
        "reciprocity": metrics["reciprocity_error"] <= limits["reciprocity_max"],
        "energy": metrics["energy_drift"] <= limits["energy_drift_max"],
        "taylor": metrics["taylor_slope"] >= limits["taylor_slope_min"] and metrics["taylor_gradient_error"] <= limits["taylor_gradient_error_max"],
        "finite": metrics["finite"],
    }
    if not smoke:
        checks["grid_mismatch"] = metrics["grid_mismatch"] <= limits["grid_mismatch_max"]
    metrics.update(checks=checks, passed=all(checks.values()), formal_gate_complete=not smoke,
                   taylor_time_steps=taylor_grid.nt)
    metrics.update(grid_diagnostics)
    metrics["metrics_path"] = str(_write_metrics(config, "acoustic_smoke" if smoke else "acoustic", metrics))
    return metrics
