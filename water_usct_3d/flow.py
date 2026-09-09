from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import cg

from .config import Grid3D
from .mac import (MACState, cell_gradient, cell_laplacian, empty_mac_state,
                  faces_to_cells, mac_divergence, mac_to_nodes, nodes_to_mac,
                  project_mac, rk2_projection_step)
from .operators import curl, divergence, laplacian2, relative_norm


@dataclass(frozen=True)
class FlowConfig:
    grid: Grid3D
    max_velocity: float = 1.0
    effective_reynolds: float = 100.0
    rho0: float = 998.0
    dtype: torch.dtype = torch.float64
    device: torch.device = torch.device("cpu")
    mode: str = "duct"
    transverse_peak: float = 0.1
    pcg_rtol: float = 1e-10
    pcg_maxiter: int = 10_000
    steady_relative_tolerance: float = 1e-9
    steady_consecutive_steps: int = 20
    max_steps: int = 20_000
    time_step: float | None = None


@dataclass
class FlowResult:
    face_velocity: dict[str, torch.Tensor]
    cell_velocity: torch.Tensor
    pressure: torch.Tensor
    forcing: torch.Tensor
    metrics: dict[str, float]
    convergence: list[float]
    pcg_convergence: list[float]


def square_duct_poiseuille(grid: Grid3D, max_velocity: float, *, dtype: torch.dtype, device: torch.device, terms: int = 31) -> torch.Tensor:
    """Fourier-series square-duct profile, normalized to the requested maximum."""
    x, y, _ = grid.coordinates(dtype=dtype, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    profile = torch.zeros_like(xx)
    for m in range(1, terms + 1, 2):
        for n in range(1, terms + 1, 2):
            coefficient = 16.0 / (torch.pi**2 * m * n * ((m / grid.length_x) ** 2 + (n / grid.length_y) ** 2))
            profile = profile + coefficient * torch.sin(m * torch.pi * xx / grid.length_x) * torch.sin(n * torch.pi * yy / grid.length_y)
    profile = max_velocity * profile / profile.max()
    velocity = torch.zeros((3, grid.nx, grid.ny, grid.nz), dtype=dtype, device=device)
    velocity[2] = profile[..., None]
    return velocity


def manufactured_flow(grid: Grid3D, max_velocity: float, transverse_peak: float, *, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    duct = square_duct_poiseuille(grid, max_velocity, dtype=dtype, device=device)
    x, y, z = grid.coordinates(dtype=dtype, device=device)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
    potential = torch.zeros_like(duct)
    potential[2] = torch.sin(torch.pi * xx / grid.length_x) ** 2 * torch.sin(torch.pi * yy / grid.length_y) ** 2 * torch.sin(2 * torch.pi * zz / grid.length_z)
    perturbation = curl(potential, (grid.dx, grid.dy, grid.dz), order=2)
    transverse = torch.linalg.vector_norm(perturbation[:2], dim=0).max()
    perturbation = perturbation * (transverse_peak / transverse.clamp_min(torch.finfo(dtype).tiny))
    velocity = duct + perturbation
    # Forcing balances steady convection and effective viscosity at Re=100.
    nu_eff = max_velocity * grid.length_x / 100.0
    grad = torch.stack([
        torch.stack(torch.gradient(velocity[c], spacing=(grid.dx, grid.dy, grid.dz), edge_order=2))
        for c in range(3)
    ])
    convection = sum(velocity[j][None] * grad[:, j] for j in range(3))
    forcing = convection - nu_eff * torch.stack([laplacian2(velocity[c], (grid.dx, grid.dy, grid.dz)) for c in range(3)])
    return velocity, forcing


def _cell_centered_duct_unit_solution(config: FlowConfig) -> np.ndarray:
    """Reference solve for choosing the constant RK2 body force, not the result."""
    nx, ny = config.grid.nx - 1, config.grid.ny - 1
    diagonal_x = np.full(nx, 2.0); diagonal_x[[0, -1]] = 3.0
    diagonal_y = np.full(ny, 2.0); diagonal_y[[0, -1]] = 3.0
    tx = diags([-np.ones(nx - 1), diagonal_x, -np.ones(nx - 1)], [-1, 0, 1]) / config.grid.dx**2
    ty = diags([-np.ones(ny - 1), diagonal_y, -np.ones(ny - 1)], [-1, 0, 1]) / config.grid.dy**2
    matrix = kron(eye(ny), tx) + kron(ty, eye(nx))
    rhs = np.ones(nx * ny, dtype=np.float64)
    residuals: list[float] = []
    def callback(x: np.ndarray) -> None:
        residuals.append(float(np.linalg.norm(matrix @ x - rhs) / np.linalg.norm(rhs)))
    solution, info = cg(matrix, rhs, rtol=config.pcg_rtol, atol=0.0, maxiter=config.pcg_maxiter, callback=callback)
    if info != 0:
        raise RuntimeError(f"duct PCG did not converge: info={info}")
    return solution.reshape(ny, nx).T


def _duct_mac_rk2(config: FlowConfig) -> tuple[MACState, list[float], list[float], float, int, float]:
    """RK2 integration of the z-invariant MAC axial momentum equation."""
    grid = config.grid
    viscosity = config.max_velocity * grid.length_x / config.effective_reynolds
    unit = _cell_centered_duct_unit_solution(config)
    force_value = viscosity * config.max_velocity / float(unit.max())
    w = torch.zeros((grid.nx - 1, grid.ny - 1), dtype=config.dtype, device=config.device)
    forcing = torch.full_like(w, force_value)
    dt_stable = 0.40 / (viscosity * (1 / grid.dx**2 + 1 / grid.dy**2))
    dt = min(config.time_step or dt_stable, dt_stable)

    def rhs(value: torch.Tensor) -> torch.Tensor:
        # A singleton periodic z dimension makes this exactly the MAC cell operator.
        return viscosity * cell_laplacian(value[..., None], grid)[..., 0] + forcing

    convergence: list[float] = []
    consecutive = 0
    for step in range(1, config.max_steps + 1):
        k1 = rhs(w)
        candidate = w + 0.5 * dt * (k1 + rhs(w + dt * k1))
        relative_change = float(torch.linalg.vector_norm(candidate - w) /
                                torch.linalg.vector_norm(candidate).clamp_min(torch.finfo(config.dtype).tiny))
        convergence.append(relative_change)
        w = candidate
        consecutive = consecutive + 1 if relative_change < config.steady_relative_tolerance else 0
        if consecutive >= config.steady_consecutive_steps:
            break
    else:
        raise RuntimeError(f"duct RK2 did not reach steady state in {config.max_steps} steps")
    state = empty_mac_state(grid, dtype=config.dtype, device=config.device)
    state.w[:] = w[..., None]
    residual = rhs(w)
    momentum_residual = float(torch.linalg.vector_norm(residual) /
                              torch.linalg.vector_norm(forcing).clamp_min(torch.finfo(config.dtype).tiny))
    # Exercise the matrix-free pressure path on the converged state. Its RHS is zero.
    state, pcg = project_mac(state, grid, dt, config.rho0, rtol=config.pcg_rtol, maxiter=config.pcg_maxiter)
    return state, convergence, pcg, momentum_residual, step, force_value


def _metrics(velocity: torch.Tensor, reference: torch.Tensor, grid: Grid3D,
             forcing: torch.Tensor, residuals: list[float], face_state: MACState,
             momentum_residual: float) -> dict[str, float]:
    div = divergence(velocity, (grid.dx, grid.dy, grid.dz), order=2)
    speed_scale = torch.linalg.vector_norm(velocity, dim=0).max().clamp_min(torch.finfo(velocity.dtype).tiny)
    relative_div = float((torch.linalg.vector_norm(div) / (torch.linalg.vector_norm(velocity) / min(grid.dx, grid.dy, grid.dz))).cpu())
    walls = torch.cat((velocity[:, 0].reshape(-1), velocity[:, -1].reshape(-1), velocity[:, :, 0].reshape(-1), velocity[:, :, -1].reshape(-1)))
    flow_rate = velocity[2].sum(dim=(0, 1)) * grid.dx * grid.dy
    variation = float(((flow_rate.max() - flow_rate.min()) / flow_rate.abs().mean().clamp_min(torch.finfo(velocity.dtype).tiny)).cpu())
    return {
        "relative_l2": relative_norm(velocity - reference, reference),
        "relative_divergence": relative_div,
        "wall_velocity_relative": float((walls.abs().max() / speed_scale).cpu()),
        "flow_rate_variation": variation,
        "momentum_residual": momentum_residual,
        "steady_relative_change": residuals[-1] if residuals else 0.0,
        "mac_relative_divergence": float(torch.linalg.vector_norm(mac_divergence(face_state, grid)) /
            (torch.linalg.vector_norm(faces_to_cells(face_state)) / min(grid.dx, grid.dy, grid.dz)).clamp_min(torch.finfo(velocity.dtype).tiny)),
        "forcing_rms": float(forcing.square().mean().sqrt().cpu()),
    }


def simulate_flow(flow_config: FlowConfig) -> FlowResult:
    grid = flow_config.grid
    reference = square_duct_poiseuille(grid, flow_config.max_velocity, dtype=flow_config.dtype, device=flow_config.device)
    if flow_config.mode == "duct":
        state, convergence, pcg, momentum_residual, steps, force_value = _duct_mac_rk2(flow_config)
        velocity = mac_to_nodes(state, grid)
        forcing = torch.zeros_like(velocity)
        forcing[2] = force_value
    elif flow_config.mode == "manufactured":
        velocity, forcing = manufactured_flow(grid, flow_config.max_velocity, flow_config.transverse_peak, dtype=flow_config.dtype, device=flow_config.device)
        reference = velocity.clone()
        state = nodes_to_mac(velocity, grid)
        # First project the sampled exact field, then construct its discrete steady force.
        state, pcg = project_mac(state, grid, 1e-3, flow_config.rho0,
                                 rtol=flow_config.pcg_rtol, maxiter=flow_config.pcg_maxiter)
        cells = faces_to_cells(state)
        viscosity = flow_config.max_velocity * grid.length_x / flow_config.effective_reynolds
        discrete_forcing = torch.stack([
            sum(cells[axis] * cell_gradient(cells[component], axis,
                (grid.dx, grid.dy, grid.dz)[axis]) for axis in range(3)) -
            viscosity * cell_laplacian(cells[component], grid)
            for component in range(3)
        ])
        forcing = functional.interpolate(
            discrete_forcing[None], size=(grid.nx, grid.ny, grid.nz),
            mode="trilinear", align_corners=False,
        )[0]
        updated, step_pcg = rk2_projection_step(state, grid, 1e-5, flow_config.rho0,
            viscosity, discrete_forcing, pcg_rtol=flow_config.pcg_rtol, pcg_maxiter=flow_config.pcg_maxiter)
        convergence = [float(torch.linalg.vector_norm(faces_to_cells(updated) - cells) /
                             torch.linalg.vector_norm(cells).clamp_min(torch.finfo(flow_config.dtype).tiny))]
        state = updated; pcg = pcg + step_pcg
        velocity = mac_to_nodes(state, grid)
        momentum_residual = convergence[-1]
        steps = 1
    else:
        raise ValueError("mode must be 'duct' or 'manufactured'")
    pressure = state.pressure
    metrics = _metrics(velocity, reference, grid, forcing, convergence, state, momentum_residual)
    metrics.update(time_steps=float(steps), pcg_iterations=float(max(0, len(pcg) - 1)),
                   pcg_relative_residual=float(pcg[-1]))
    return FlowResult({"u": state.u, "v": state.v, "w": state.w}, velocity,
                      pressure, forcing, metrics, convergence, pcg)
