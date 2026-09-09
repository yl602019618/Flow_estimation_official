from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from .acoustic import simulate_acoustic
from .inversion import ObjectiveConfig3D, lowpass
from .travel_time import DelayData3D


@dataclass
class ProfileResult3D:
    control: torch.Tensor
    velocity: torch.Tensor
    candidates: list[dict[str, float | str]]
    selected: str


def axial_profile_to_velocity(control: torch.Tensor, grid) -> torch.Tensor:
    """Bilinear z-invariant Uz profile with exact zero wall controls."""
    if control.ndim != 2:
        raise ValueError("axial profile control must be two-dimensional")
    constrained = control.clone()
    constrained[0] = 0.0; constrained[-1] = 0.0
    constrained[:, 0] = 0.0; constrained[:, -1] = 0.0
    profile = functional.interpolate(constrained[None, None], size=(grid.nx, grid.ny),
                                     mode="bilinear", align_corners=True)[0, 0]
    velocity = torch.zeros((3, grid.nx, grid.ny, grid.nz), dtype=control.dtype, device=control.device)
    velocity[2] = profile[..., None]
    return velocity


def square_duct_profile_control(beta: float, shape: tuple[int, int], *,
                                dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Sample the known square-duct Fourier family on an axial control grid."""
    nx, ny = shape
    x = torch.linspace(0.0, 1.0, nx, dtype=dtype, device=device)
    y = torch.linspace(0.0, 1.0, ny, dtype=dtype, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    base = torch.zeros_like(xx)
    for m in range(1, 32, 2):
        for n in range(1, 32, 2):
            coefficient = 16.0 / (torch.pi**2 * m * n * (m*m + n*n))
            base = base + coefficient * torch.sin(m * torch.pi * xx) * torch.sin(n * torch.pi * yy)
    base = float(beta) * base / base.max().clamp_min(torch.finfo(dtype).tiny)
    base[0] = 0.0; base[-1] = 0.0
    base[:, 0] = 0.0; base[:, -1] = 0.0
    return base


def direct_axial_travel_time_control(delay: DelayData3D,
                                     objective: ObjectiveConfig3D) -> tuple[torch.Tensor, dict[str, float]]:
    """Recover Uz controls from reciprocal delays without a duct-profile prior."""
    shape = objective.demo.inversion.profile_control
    matrix, interior = build_axial_delay_matrix(
        objective.acquisition, shape,
        (objective.demo.grid.length_x, objective.demo.grid.length_y),
        objective.demo.physics.c0,
    )
    weight = delay.quality.clamp_min(0.0).sqrt()
    weighted_matrix = matrix * weight[:, None]
    weighted_delay = delay.delays * weight
    normal = weighted_matrix.T @ weighted_matrix
    scale = torch.trace(normal).clamp_min(torch.finfo(normal.dtype).tiny) / normal.shape[0]
    regularization = objective.demo.inversion.regularization_candidates[-1] * scale
    interior_value = torch.linalg.solve(
        normal + regularization * torch.eye(normal.shape[0], dtype=normal.dtype, device=normal.device),
        weighted_matrix.T @ weighted_delay,
    )
    control = torch.zeros(shape, dtype=matrix.dtype, device=matrix.device)
    control[interior] = interior_value
    singular = torch.linalg.svdvals(weighted_matrix)
    threshold = singular[0] * max(weighted_matrix.shape) * torch.finfo(singular.dtype).eps
    rank = int((singular > threshold).sum())
    condition = float(singular[0] / singular[rank - 1]) if rank else float("inf")
    residual = float(torch.linalg.vector_norm(matrix @ interior_value - delay.delays) /
                     torch.linalg.vector_norm(delay.delays).clamp_min(torch.finfo(delay.delays.dtype).tiny))
    return control, {
        "effective_rank": float(rank), "condition_number": condition,
        "relative_delay_residual": residual,
        "regularization": float(regularization),
    }


def gaussian_blur_axial_control(control: torch.Tensor, sigma_cells: float = 0.75) -> torch.Tensor:
    """Apply the registered fixed Gaussian blur and restore exact zero walls."""
    radius = 2
    coordinate = torch.arange(-radius, radius + 1, dtype=control.dtype, device=control.device)
    kernel_1d = torch.exp(-0.5 * (coordinate / sigma_cells).square())
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = torch.outer(kernel_1d, kernel_1d)[None, None]
    blurred = functional.conv2d(control[None, None], kernel, padding=radius)[0, 0]
    blurred[0] = 0.0; blurred[-1] = 0.0
    blurred[:, 0] = 0.0; blurred[:, -1] = 0.0
    return blurred


def build_axial_fwi_initial_controls(delay: DelayData3D,
                                     objective: ObjectiveConfig3D,
                                     beta: float) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Build the four frozen initializations used by the axial-FWI ablation."""
    shape = objective.demo.inversion.profile_control
    good = square_duct_profile_control(
        beta, shape, dtype=delay.delays.dtype, device=delay.delays.device)
    travel_time, diagnostics = direct_axial_travel_time_control(delay, objective)
    controls = {
        "good_prior": good,
        "travel_time": travel_time,
        "blurred_travel_time": gaussian_blur_axial_control(travel_time, sigma_cells=0.75),
        "zero_flow": torch.zeros_like(good),
    }
    return controls, diagnostics


def build_axial_delay_matrix(acquisition, shape: tuple[int, int], lengths: tuple[float, float], c0: float = 1480.0,
                             samples: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    nx, ny = shape
    interior = torch.zeros((nx, ny), dtype=torch.bool, device=acquisition.coordinates.device)
    interior[1:-1, 1:-1] = True
    rows = []
    for a, b in acquisition.reciprocal_pairs.tolist():
        start, end = acquisition.coordinates[a], acquisition.coordinates[b]
        row = torch.zeros((nx, ny), dtype=start.dtype, device=start.device)
        for alpha in torch.linspace(0.0, 1.0, samples, dtype=start.dtype, device=start.device):
            point = start + alpha * (end - start)
            index = torch.stack((point[0] / lengths[0] * (nx - 1), point[1] / lengths[1] * (ny - 1)))
            low = torch.floor(index).to(torch.long).clamp_min(0)
            low = torch.minimum(low, torch.tensor((nx - 2, ny - 2), device=low.device))
            fraction = index - low
            for ix in (0, 1):
                for iy in (0, 1):
                    weight = (fraction[0] if ix else 1 - fraction[0]) * (fraction[1] if iy else 1 - fraction[1])
                    row[low[0] + ix, low[1] + iy] += weight / (samples - 1)
        row *= -2.0 * (end[2] - start[2]) / c0**2
        rows.append(row[interior])
    return torch.stack(rows), interior


def _profile_loss(predicted: torch.Tensor, observed: torch.Tensor,
                  objective: ObjectiveConfig3D) -> float:
    values = []
    mask = objective.acquisition.ordered_mask
    for cutoff in objective.demo.inversion.lowpass_frequencies:
        pred = lowpass(predicted, objective.demo.grid.dt, float(cutoff))
        obs = lowpass(observed, objective.demo.grid.dt, float(cutoff))
        energy = obs.square().sum(dim=-1).clamp_min(torch.finfo(obs.dtype).tiny)
        values.append(0.5 * ((pred - obs).square().sum(dim=-1) / energy)[mask].mean())
    return float(torch.stack(values).mean())


def invert_axial_profile(observed: torch.Tensor, delay: DelayData3D,
                         objective: ObjectiveConfig3D, beta: float) -> ProfileResult3D:
    """Select amplitude-prior or delay-updated 6x6 profile by nonlinear waveforms."""
    shape = objective.demo.inversion.profile_control
    base = square_duct_profile_control(
        beta, shape, dtype=observed.dtype, device=observed.device)
    matrix, interior = build_axial_delay_matrix(objective.acquisition, shape,
        (objective.demo.grid.length_x, objective.demo.grid.length_y), objective.demo.physics.c0)
    base_interior = base[interior]
    duct_direction = base_interior / max(beta, torch.finfo(base.dtype).tiny)
    null = torch.linalg.qr(duct_direction[:, None], mode="complete").Q[:, 1:]
    weighted = delay.quality.clamp_min(0.0).sqrt()
    system = matrix @ null
    residual = delay.delays - matrix @ base_interior
    lhs = (system * weighted[:, None]).T @ (system * weighted[:, None])
    rhs = (system * weighted[:, None]).T @ (residual * weighted)
    scale = torch.trace(lhs).clamp_min(torch.finfo(lhs.dtype).tiny) / lhs.shape[0]
    regularization = objective.demo.inversion.regularization_candidates[-1] * scale
    update = torch.linalg.solve(lhs + regularization * torch.eye(lhs.shape[0], dtype=lhs.dtype, device=lhs.device), rhs)
    delay_control = base.clone()
    delay_control[interior] = base_interior + null @ update

    controls = {"amplitude_prior": base, "delay_updated": delay_control}
    candidates: list[dict[str, float | str]] = []
    predictions: dict[str, torch.Tensor] = {}
    for name, control in controls.items():
        velocity = axial_profile_to_velocity(control, objective.demo.grid)
        with torch.no_grad():
            prediction = simulate_acoustic(velocity, objective.acquisition.source_apertures,
                                           objective.acquisition.apertures, objective.forward)
        predictions[name] = prediction
        candidates.append({
            "name": name,
            "waveform_loss": _profile_loss(prediction, observed, objective),
            "delay_relative_residual": float(torch.linalg.vector_norm(matrix @ control[interior] - delay.delays) /
                                             torch.linalg.vector_norm(delay.delays)),
            "control_min": float(control.min()), "control_max": float(control.max()),
        })
    selected = min(candidates, key=lambda item: float(item["waveform_loss"]))["name"]
    selected_control = controls[str(selected)]
    return ProfileResult3D(selected_control, axial_profile_to_velocity(selected_control, objective.demo.grid),
                           candidates, str(selected))
