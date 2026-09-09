from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import Grid3D


@dataclass
class MACState:
    """Face velocities and cell-centred pressure.

    ``u`` has shape ``(mx+1,my,mz)``, ``v`` ``(mx,my+1,mz)``, and the
    periodic z-face field ``w`` ``(mx,my,mz)``. Pressure uses
    ``(mx,my,mz)``, where ``m* = n* - 1`` relative to the acoustic nodes.
    """

    u: torch.Tensor
    v: torch.Tensor
    w: torch.Tensor
    pressure: torch.Tensor

    def clone(self) -> "MACState":
        return MACState(self.u.clone(), self.v.clone(), self.w.clone(), self.pressure.clone())


def empty_mac_state(grid: Grid3D, *, dtype: torch.dtype, device: torch.device) -> MACState:
    mx, my, mz = grid.nx - 1, grid.ny - 1, grid.nz - 1
    return MACState(
        torch.zeros((mx + 1, my, mz), dtype=dtype, device=device),
        torch.zeros((mx, my + 1, mz), dtype=dtype, device=device),
        torch.zeros((mx, my, mz), dtype=dtype, device=device),
        torch.zeros((mx, my, mz), dtype=dtype, device=device),
    )


def enforce_no_slip(state: MACState) -> MACState:
    state.u[0] = 0.0; state.u[-1] = 0.0
    state.v[:, 0] = 0.0; state.v[:, -1] = 0.0
    return state


def mac_divergence(state: MACState, grid: Grid3D) -> torch.Tensor:
    return ((state.u[1:] - state.u[:-1]) / grid.dx +
            (state.v[:, 1:] - state.v[:, :-1]) / grid.dy +
            (torch.roll(state.w, -1, dims=2) - state.w) / grid.dz)


def pressure_gradient(pressure: torch.Tensor, grid: Grid3D) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mx, my, mz = pressure.shape
    gx = torch.zeros((mx + 1, my, mz), dtype=pressure.dtype, device=pressure.device)
    gy = torch.zeros((mx, my + 1, mz), dtype=pressure.dtype, device=pressure.device)
    gx[1:mx] = (pressure[1:] - pressure[:-1]) / grid.dx
    gy[:, 1:my] = (pressure[:, 1:] - pressure[:, :-1]) / grid.dy
    gz = (pressure - torch.roll(pressure, 1, dims=2)) / grid.dz
    return gx, gy, gz


def negative_pressure_laplacian(pressure: torch.Tensor, grid: Grid3D) -> torch.Tensor:
    gx, gy, gz = pressure_gradient(pressure, grid)
    gradient_state = MACState(gx, gy, gz, torch.zeros_like(pressure))
    return -mac_divergence(gradient_state, grid)


def _pressure_diagonal(pressure: torch.Tensor, grid: Grid3D) -> torch.Tensor:
    diagonal = torch.full_like(pressure, 2 / grid.dx**2 + 2 / grid.dy**2 + 2 / grid.dz**2)
    diagonal[0] -= 1 / grid.dx**2; diagonal[-1] -= 1 / grid.dx**2
    diagonal[:, 0] -= 1 / grid.dy**2; diagonal[:, -1] -= 1 / grid.dy**2
    return diagonal


def pcg_pressure(rhs: torch.Tensor, grid: Grid3D, *, rtol: float = 1e-10,
                 maxiter: int = 10_000) -> tuple[torch.Tensor, list[float]]:
    """Matrix-free mean-zero PCG for Neumann-x/y, periodic-z pressure."""
    rhs = rhs - rhs.mean()
    rhs_norm = torch.linalg.vector_norm(rhs)
    if float(rhs_norm) == 0.0:
        return torch.zeros_like(rhs), [0.0]
    pressure = torch.zeros_like(rhs)
    residual = rhs.clone()
    inverse_diagonal = _pressure_diagonal(rhs, grid).reciprocal()
    preconditioned = inverse_diagonal * residual
    direction = preconditioned.clone()
    rz = (residual * preconditioned).sum()
    history = [1.0]
    for _ in range(maxiter):
        applied = negative_pressure_laplacian(direction, grid)
        denominator = (direction * applied).sum()
        if abs(float(denominator)) <= torch.finfo(rhs.dtype).tiny:
            break
        alpha = rz / denominator
        pressure = pressure + alpha * direction
        pressure = pressure - pressure.mean()
        residual = residual - alpha * applied
        relative = float(torch.linalg.vector_norm(residual) / rhs_norm)
        history.append(relative)
        if relative <= rtol:
            return pressure, history
        new_preconditioned = inverse_diagonal * residual
        new_rz = (residual * new_preconditioned).sum()
        direction = new_preconditioned + (new_rz / rz) * direction
        preconditioned, rz = new_preconditioned, new_rz
    raise RuntimeError(f"pressure PCG failed: residual={history[-1]:.3e}, iterations={len(history)-1}")


def project_mac(state: MACState, grid: Grid3D, dt: float, rho0: float,
                *, rtol: float = 1e-10, maxiter: int = 10_000) -> tuple[MACState, list[float]]:
    divergence = mac_divergence(state, grid)
    rhs = -(rho0 / dt) * divergence
    pressure, history = pcg_pressure(rhs, grid, rtol=rtol, maxiter=maxiter)
    gx, gy, gz = pressure_gradient(pressure, grid)
    projected = MACState(
        state.u - (dt / rho0) * gx,
        state.v - (dt / rho0) * gy,
        state.w - (dt / rho0) * gz,
        pressure,
    )
    return enforce_no_slip(projected), history


def faces_to_cells(state: MACState) -> torch.Tensor:
    return torch.stack((
        0.5 * (state.u[:-1] + state.u[1:]),
        0.5 * (state.v[:, :-1] + state.v[:, 1:]),
        0.5 * (state.w + torch.roll(state.w, -1, dims=2)),
    ))


def cells_to_face_rhs(rhs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, mx, my, mz = rhs.shape
    u = torch.zeros((mx + 1, my, mz), dtype=rhs.dtype, device=rhs.device)
    v = torch.zeros((mx, my + 1, mz), dtype=rhs.dtype, device=rhs.device)
    u[1:mx] = 0.5 * (rhs[0, :-1] + rhs[0, 1:])
    v[:, 1:my] = 0.5 * (rhs[1, :, :-1] + rhs[1, :, 1:])
    # w[k] lies between cells k-1 and k.
    w = 0.5 * (rhs[2] + torch.roll(rhs[2], 1, dims=2))
    return u, v, w


def _shift_cell(field: torch.Tensor, axis: int, offset: int, *, odd_wall: bool) -> torch.Tensor:
    if axis == 2:
        return torch.roll(field, -offset, dims=axis)
    n = field.shape[axis]
    index = torch.arange(n, device=field.device) + offset
    low, high = index < 0, index >= n
    index = torch.where(low, torch.zeros_like(index), torch.where(high, torch.full_like(index, n - 1), index))
    value = field.index_select(axis, index)
    if odd_wall:
        shape = [1] * field.ndim; shape[axis] = n
        sign = torch.where(low | high, -torch.ones(n, device=field.device), torch.ones(n, device=field.device))
        value = value * sign.reshape(shape)
    return value


def cell_gradient(field: torch.Tensor, axis: int, spacing: float, *, odd_wall: bool = True) -> torch.Tensor:
    return (_shift_cell(field, axis, 1, odd_wall=odd_wall) -
            _shift_cell(field, axis, -1, odd_wall=odd_wall)) / (2 * spacing)


def cell_laplacian(field: torch.Tensor, grid: Grid3D, *, odd_wall: bool = True) -> torch.Tensor:
    result = torch.zeros_like(field)
    for axis, spacing in enumerate((grid.dx, grid.dy, grid.dz)):
        result += (_shift_cell(field, axis, 1, odd_wall=odd_wall) - 2 * field +
                   _shift_cell(field, axis, -1, odd_wall=odd_wall)) / spacing**2
    return result


def momentum_rhs(state: MACState, grid: Grid3D, viscosity: float,
                 forcing: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    velocity = faces_to_cells(state)
    rhs = []
    for component in range(3):
        convection = sum(velocity[axis] * cell_gradient(
            velocity[component], axis, (grid.dx, grid.dy, grid.dz)[axis], odd_wall=True
        ) for axis in range(3))
        rhs.append(-convection + viscosity * cell_laplacian(velocity[component], grid) + forcing[component])
    return cells_to_face_rhs(torch.stack(rhs))


def rk2_projection_step(state: MACState, grid: Grid3D, dt: float, rho0: float,
                        viscosity: float, forcing: torch.Tensor, *, pcg_rtol: float = 1e-10,
                        pcg_maxiter: int = 10_000) -> tuple[MACState, list[float]]:
    """Explicit Heun/RK2 momentum predictor followed by a MAC projection."""
    k1 = momentum_rhs(state, grid, viscosity, forcing)
    euler = MACState(state.u + dt * k1[0], state.v + dt * k1[1],
                     state.w + dt * k1[2], state.pressure)
    enforce_no_slip(euler)
    k2 = momentum_rhs(euler, grid, viscosity, forcing)
    predicted = MACState(
        state.u + 0.5 * dt * (k1[0] + k2[0]),
        state.v + 0.5 * dt * (k1[1] + k2[1]),
        state.w + 0.5 * dt * (k1[2] + k2[2]),
        state.pressure,
    )
    enforce_no_slip(predicted)
    return project_mac(predicted, grid, dt, rho0, rtol=pcg_rtol, maxiter=pcg_maxiter)


def mac_to_nodes(state: MACState, grid: Grid3D) -> torch.Tensor:
    """Second-order interpolation of MAC velocities to acoustic-grid nodes."""
    cells = faces_to_cells(state)
    result = torch.zeros((3, grid.nx, grid.ny, grid.nz), dtype=cells.dtype, device=cells.device)
    # Average adjacent x/y cells; periodic z cell centres surround every z node.
    for component in range(3):
        value = cells[component]
        xy_nodes = torch.zeros((grid.nx, grid.ny, value.shape[2]), dtype=value.dtype, device=value.device)
        xy_nodes[1:-1, 1:-1] = 0.25 * (value[:-1, :-1] + value[1:, :-1] + value[:-1, 1:] + value[1:, 1:])
        result[component, :, :, :-1] = 0.5 * (xy_nodes + torch.roll(xy_nodes, 1, dims=2))
        result[component, :, :, -1] = result[component, :, :, 0]
    # Exact no-slip values on all x/y wall nodes.
    result[:, 0] = 0; result[:, -1] = 0
    result[:, :, 0] = 0; result[:, :, -1] = 0
    return result


def nodes_to_mac(velocity: torch.Tensor, grid: Grid3D) -> MACState:
    """Interpolate periodic-z nodal velocities to their MAC face locations."""
    u = 0.25 * (velocity[0, :, :-1, :-1] + velocity[0, :, 1:, :-1] +
                velocity[0, :, :-1, 1:] + velocity[0, :, 1:, 1:])
    v = 0.25 * (velocity[1, :-1, :, :-1] + velocity[1, 1:, :, :-1] +
                velocity[1, :-1, :, 1:] + velocity[1, 1:, :, 1:])
    w = 0.25 * (velocity[2, :-1, :-1, :-1] + velocity[2, 1:, :-1, :-1] +
                velocity[2, :-1, 1:, :-1] + velocity[2, 1:, 1:, :-1])
    pressure = torch.zeros((grid.nx - 1, grid.ny - 1, grid.nz - 1),
                           dtype=velocity.dtype, device=velocity.device)
    return enforce_no_slip(MACState(u, v, w, pressure))
