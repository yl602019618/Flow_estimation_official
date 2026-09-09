from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import AcquisitionConfig3D, Grid3D


@dataclass
class Acquisition3D:
    coordinates: torch.Tensor
    wall_ids: tuple[str, ...]
    apertures: torch.Tensor
    source_apertures: torch.Tensor
    ordered_mask: torch.Tensor
    reciprocal_pairs: torch.Tensor
    source_signal: torch.Tensor
    dt: float

    @property
    def trace_count(self) -> int:
        return int(self.ordered_mask.sum().item())


def ricker(time: torch.Tensor, frequency: float, delay_cycles: float = 2.0) -> torch.Tensor:
    delay = delay_cycles / frequency
    argument = torch.pi * frequency * (time - delay)
    return (1.0 - 2.0 * argument.square()) * torch.exp(-argument.square())


def transducer_coordinates(grid: Grid3D, config: AcquisitionConfig3D, *, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, tuple[str, ...]]:
    coords: list[tuple[float, float, float]] = []
    walls: list[str] = []
    ox, oy = config.wall_offset_cells * grid.dx, config.wall_offset_cells * grid.dy
    for ring_index, (z, positions) in enumerate(zip(config.ring_z, config.ring_positions)):
        start = len(coords)
        for fraction in positions:
            coords.append((fraction * grid.length_x, oy, z)); walls.append("y0")
        for fraction in positions:
            coords.append((grid.length_x - ox, fraction * grid.length_y, z)); walls.append("x1")
        for fraction in reversed(positions):
            coords.append((fraction * grid.length_x, grid.length_y - oy, z)); walls.append("y1")
        for fraction in reversed(positions):
            coords.append((ox, fraction * grid.length_y, z)); walls.append("x0")
        shift = config.ring_perimeter_offsets[ring_index]
        if shift:
            shifted: list[tuple[float, float, float]] = []
            shifted_walls: list[str] = []
            for x_value, y_value, z_value in coords[start:start + 8]:
                if abs(y_value - oy) < 1e-12: perimeter = x_value / grid.length_x
                elif abs(x_value - (grid.length_x - ox)) < 1e-12: perimeter = 1 + y_value / grid.length_y
                elif abs(y_value - (grid.length_y - oy)) < 1e-12: perimeter = 3 - x_value / grid.length_x
                else: perimeter = 4 - y_value / grid.length_y
                perimeter = (perimeter + shift) % 4
                if perimeter < 1:
                    tangent = min(max(perimeter * grid.length_x, ox), grid.length_x - ox)
                    shifted.append((tangent, oy, z_value)); shifted_walls.append("y0")
                elif perimeter < 2:
                    tangent = min(max((perimeter - 1) * grid.length_y, oy), grid.length_y - oy)
                    shifted.append((grid.length_x - ox, tangent, z_value)); shifted_walls.append("x1")
                elif perimeter < 3:
                    tangent = min(max((3 - perimeter) * grid.length_x, ox), grid.length_x - ox)
                    shifted.append((tangent, grid.length_y - oy, z_value)); shifted_walls.append("y1")
                else:
                    tangent = min(max((4 - perimeter) * grid.length_y, oy), grid.length_y - oy)
                    shifted.append((ox, tangent, z_value)); shifted_walls.append("x0")
            coords[start:start + 8] = shifted
            walls[start:start + 8] = shifted_walls
    return torch.tensor(coords, dtype=dtype, device=device), tuple(walls)


def point_apertures(grid: Grid3D, coordinates: torch.Tensor) -> torch.Tensor:
    """Trilinear samples at physical points, with at most eight nonzero nodes."""
    sizes = (grid.nx, grid.ny, grid.nz)
    spacings = torch.tensor((grid.dx, grid.dy, grid.dz), dtype=coordinates.dtype,
                            device=coordinates.device)
    normalized = coordinates / spacings
    lower = torch.floor(normalized).to(torch.long)
    upper_bounds = torch.tensor(sizes, dtype=torch.long, device=coordinates.device) - 2
    lower = torch.minimum(torch.maximum(lower, torch.zeros_like(lower)), upper_bounds)
    fraction = (normalized - lower.to(coordinates.dtype)).clamp(0.0, 1.0)
    apertures = torch.zeros((coordinates.shape[0], *sizes), dtype=coordinates.dtype,
                            device=coordinates.device)
    shots = torch.arange(coordinates.shape[0], device=coordinates.device)
    for bx in (0, 1):
        for by in (0, 1):
            for bz in (0, 1):
                bits = torch.tensor((bx, by, bz), dtype=torch.long, device=coordinates.device)
                indices = lower + bits
                factors = torch.where(bits.bool()[None], fraction, 1.0 - fraction)
                # Avoid a three-value CUDA reduction/JIT for this fixed tensor.
                weights = factors[:, 0] * factors[:, 1] * factors[:, 2]
                apertures[shots, indices[:, 0], indices[:, 1], indices[:, 2]] += weights
    return apertures


def quadrature_mass(grid: Grid3D, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    wx = torch.ones(grid.nx, dtype=dtype, device=device); wx[[0, -1]] = 0.5
    wy = torch.ones(grid.ny, dtype=dtype, device=device); wy[[0, -1]] = 0.5
    wz = torch.ones(grid.nz, dtype=dtype, device=device); wz[[0, -1]] = 0.5
    return grid.cell_volume * wx[:, None, None] * wy[None, :, None] * wz[None, None, :]


def build_acquisition_3d(grid: Grid3D, acquisition_config: AcquisitionConfig3D, *, dtype: torch.dtype = torch.float32, device: torch.device | str = "cpu") -> Acquisition3D:
    device = torch.device(device)
    coordinates, walls = transducer_coordinates(grid, acquisition_config, dtype=dtype, device=device)
    if torch.pdist(coordinates).min() <= torch.finfo(dtype).eps:
        raise ValueError("transducer geometry contains duplicate coordinates; preserve the physical wall offset when changing grids")
    if acquisition_config.transducer_model != "point":
        raise ValueError("only the registered trilinear point transducer model is supported")
    apertures = point_apertures(grid, coordinates)
    count = coordinates.shape[0]
    mask = ~torch.eye(count, dtype=torch.bool, device=device)
    pairs = torch.triu_indices(count, count, offset=1, device=device).T
    time = torch.arange(grid.nt, dtype=dtype, device=device) * grid.dt
    signal = acquisition_config.source_amplitude * ricker(time, acquisition_config.center_frequency, acquisition_config.source_delay_cycles)
    # C is trilinear point sampling and B=M^-1 C^T is its exact mass-adjoint
    # pressure-rate point injection under the nodal trapezoidal mass.
    sources = apertures / quadrature_mass(grid, dtype, device)
    return Acquisition3D(coordinates, walls, apertures, sources, mask, pairs, signal, grid.dt)
