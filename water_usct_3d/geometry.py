from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from itertools import product

import torch

from .acquisition import Acquisition3D
from .config import Grid3D


@dataclass
class CoverageResult:
    matrix: torch.Tensor
    min_eigenvalue: torch.Tensor
    normalized_min_eigenvalue: torch.Tensor
    condition_number: torch.Tensor
    metrics: dict[str, float | int | bool]


@dataclass
class GeometrySearchResult:
    acquisition: Acquisition3D
    coverage: CoverageResult
    ring_z: tuple[float, float, float]
    perimeter_offsets: tuple[float, float, float]
    candidates_evaluated: int


def direction_coverage(grid: Grid3D, acquisition: Acquisition3D, central_fraction: float = 0.6) -> CoverageResult:
    x, y, z = grid.coordinates(dtype=acquisition.coordinates.dtype, device=acquisition.coordinates.device)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
    points = torch.stack((xx, yy, zz), dim=-1)
    matrix = torch.zeros((*points.shape[:-1], 3, 3), dtype=points.dtype, device=points.device)
    scale = 2.5 * max(grid.dx, grid.dy, grid.dz)
    for a, b in acquisition.reciprocal_pairs.tolist():
        start, end = acquisition.coordinates[a], acquisition.coordinates[b]
        ray = end - start
        length = torch.linalg.vector_norm(ray)
        tangent = ray / length
        relative = points - start
        alpha = (relative * tangent).sum(dim=-1).clamp(0.0, float(length))
        closest = start + alpha[..., None] * tangent
        distance2 = (points - closest).square().sum(dim=-1)
        weight = torch.exp(-0.5 * distance2 / scale**2) / length
        matrix += weight[..., None, None] * torch.outer(tangent, tangent)
    eig = torch.linalg.eigvalsh(matrix)
    minimum, maximum = eig[..., 0], eig[..., -1]
    normalized = minimum / eig.sum(dim=-1).clamp_min(torch.finfo(eig.dtype).tiny)
    condition = maximum / minimum.clamp_min(torch.finfo(eig.dtype).tiny)
    margin_x = int(round((1.0 - central_fraction) * grid.nx / 2.0))
    margin_y = int(round((1.0 - central_fraction) * grid.ny / 2.0))
    margin_z = int(round((1.0 - central_fraction) * grid.nz / 2.0))
    central = eig[margin_x:grid.nx-margin_x, margin_y:grid.ny-margin_y, margin_z:grid.nz-margin_z]
    central_norm = normalized[margin_x:grid.nx-margin_x, margin_y:grid.ny-margin_y, margin_z:grid.nz-margin_z]
    central_condition = condition[margin_x:grid.nx-margin_x, margin_y:grid.ny-margin_y, margin_z:grid.nz-margin_z]
    rank = int((central.mean(dim=(0, 1, 2)) > central.max() * 1e-10).sum().item())
    min_normalized = float(central_norm.min().detach().cpu())
    metrics: dict[str, float | int | bool] = {
        "central_rank": rank,
        "central_min_eigenvalue": float(central[..., 0].min().detach().cpu()),
        "central_normalized_min_eigenvalue": min_normalized,
        "central_condition_max": float(central_condition.max().detach().cpu()),
        "pass": rank == 3 and min_normalized >= 1e-3,
    }
    return CoverageResult(matrix, minimum, normalized, condition, metrics)


def _shift_square_ring(coordinates: torch.Tensor, shift: float, z: float, grid: Grid3D, offset_cells: float) -> torch.Tensor:
    ox, oy = offset_cells * grid.dx, offset_cells * grid.dy
    values = []
    for coordinate in coordinates:
        x, y = float(coordinate[0]), float(coordinate[1])
        if abs(y - oy) < 1e-7:
            perimeter = x / grid.length_x
        elif abs(x - (grid.length_x - ox)) < 1e-7:
            perimeter = 1.0 + y / grid.length_y
        elif abs(y - (grid.length_y - oy)) < 1e-7:
            perimeter = 3.0 - x / grid.length_x
        else:
            perimeter = 4.0 - y / grid.length_y
        perimeter = (perimeter + shift) % 4.0
        if perimeter < 1.0:
            point = (min(max(perimeter * grid.length_x, ox), grid.length_x - ox), oy, z)
        elif perimeter < 2.0:
            point = (grid.length_x - ox, min(max((perimeter - 1.0) * grid.length_y, oy), grid.length_y - oy), z)
        elif perimeter < 3.0:
            point = (min(max((3.0 - perimeter) * grid.length_x, ox), grid.length_x - ox), grid.length_y - oy, z)
        else:
            point = (ox, min(max((4.0 - perimeter) * grid.length_y, oy), grid.length_y - oy), z)
        values.append(point)
    return torch.tensor(values, dtype=coordinates.dtype, device=coordinates.device)


def _sample_score(grid: Grid3D, acquisition: Acquisition3D) -> float:
    axes = (torch.linspace(0.2 * grid.length_x, 0.8 * grid.length_x, 5, dtype=acquisition.coordinates.dtype),
            torch.linspace(0.2 * grid.length_y, 0.8 * grid.length_y, 5, dtype=acquisition.coordinates.dtype),
            torch.linspace(0.2 * grid.length_z, 0.8 * grid.length_z, 7, dtype=acquisition.coordinates.dtype))
    points = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3).to(acquisition.coordinates.device)
    matrix = torch.zeros(points.shape[0], 3, 3, dtype=points.dtype, device=points.device)
    scale = 2.5 * max(grid.dx, grid.dy, grid.dz)
    for a, b in acquisition.reciprocal_pairs.tolist():
        start, end = acquisition.coordinates[a], acquisition.coordinates[b]
        ray = end - start; length = torch.linalg.vector_norm(ray); tangent = ray / length
        alpha = ((points - start) * tangent).sum(dim=-1).clamp(0.0, float(length))
        distance2 = (points - (start + alpha[..., None] * tangent)).square().sum(dim=-1)
        matrix += (torch.exp(-0.5 * distance2 / scale**2) / length)[..., None, None] * torch.outer(tangent, tangent)
    eig = torch.linalg.eigvalsh(matrix)
    return float((eig[:, 0] / eig.sum(dim=-1).clamp_min(torch.finfo(eig.dtype).tiny)).min().cpu())


def deterministic_geometry_search(grid: Grid3D, acquisition: Acquisition3D, offset_cells: float,
                                  sponge_layers: int = 0) -> GeometrySearchResult:
    """Registered search over only ring z positions and circumferential offsets."""
    z_candidates = (
        (0.025 * grid.length_z / 0.20, 0.10 * grid.length_z / 0.20, 0.175 * grid.length_z / 0.20),
        (0.03 * grid.length_z / 0.20, 0.10 * grid.length_z / 0.20, 0.17 * grid.length_z / 0.20),
        (0.04 * grid.length_z / 0.20, 0.10 * grid.length_z / 0.20, 0.16 * grid.length_z / 0.20),
        (0.05 * grid.length_z / 0.20, 0.10 * grid.length_z / 0.20, 0.15 * grid.length_z / 0.20),
    )
    offsets = (-0.25, 0.0, 0.25)
    best_score, best = -1.0, None
    count = 0
    for ring_z in z_candidates:
        sponge_width = sponge_layers * grid.dz
        if ring_z[0] <= sponge_width or ring_z[-1] >= grid.length_z - sponge_width:
            continue
        for shifts in product(offsets, repeat=3):
            coordinates = torch.cat([_shift_square_ring(acquisition.coordinates[8*i:8*(i+1)], shifts[i], ring_z[i], grid, offset_cells) for i in range(3)])
            candidate = replace(acquisition, coordinates=coordinates)
            score = _sample_score(grid, candidate); count += 1
            if score > best_score:
                best_score, best = score, (candidate, ring_z, shifts)
    assert best is not None
    candidate, ring_z, shifts = best
    return GeometrySearchResult(candidate, direction_coverage(grid, candidate), ring_z, shifts, count)
