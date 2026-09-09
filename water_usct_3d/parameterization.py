from __future__ import annotations

import torch
import torch.nn.functional as functional

from .config import Grid3D
from .flow import square_duct_poiseuille
from .operators import curl, divergence


def constrain_vector_potential(potential: torch.Tensor) -> torch.Tensor:
    """Apply tangential A=0 constraints that imply zero perturbation wall-normal flow."""
    ax, ay, az = potential.unbind(0)
    mask_x = torch.ones_like(ax); mask_x[:, 0, :] = 0; mask_x[:, -1, :] = 0
    mask_y = torch.ones_like(ay); mask_y[0, :, :] = 0; mask_y[-1, :, :] = 0
    mask_z = mask_x * mask_y
    return torch.stack((ax * mask_x, ay * mask_y, az * mask_z))


def interpolate_vector_potential(control: torch.Tensor, grid: Grid3D) -> torch.Tensor:
    if control.ndim != 4 or control.shape[0] != 3:
        raise ValueError("A must have shape (3, nx_control, ny_control, nz_control)")
    full = functional.interpolate(control[None], size=(grid.nx, grid.ny, grid.nz), mode="trilinear", align_corners=True)[0]
    return constrain_vector_potential(full)


def vector_potential_to_velocity(beta: torch.Tensor | float, A: torch.Tensor, grid: Grid3D) -> torch.Tensor:
    potential = interpolate_vector_potential(A, grid) if A.shape[1:] != (grid.nx, grid.ny, grid.nz) else constrain_vector_potential(A)
    perturbation = curl(potential, (grid.dx, grid.dy, grid.dz), order=2)
    duct = square_duct_poiseuille(grid, 1.0, dtype=A.dtype, device=A.device)
    return torch.as_tensor(beta, dtype=A.dtype, device=A.device) * duct + perturbation


def gauge_divergence(A: torch.Tensor, grid: Grid3D) -> torch.Tensor:
    full = interpolate_vector_potential(A, grid) if A.shape[1:] != (grid.nx, grid.ny, grid.nz) else constrain_vector_potential(A)
    return divergence(full, (grid.dx, grid.dy, grid.dz), order=2)
