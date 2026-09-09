from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .acquisition import Acquisition3D


@dataclass
class DelayData3D:
    delays: torch.Tensor
    pairs: torch.Tensor
    dt: float
    quality: torch.Tensor


@dataclass
class InitialModel3D:
    beta: float
    velocity_controls: torch.Tensor
    singular_values: torch.Tensor
    effective_rank: int
    condition_number: float
    weakest_mode: torch.Tensor


def _subsample_lag(first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = first.numel()
    # torch.conv1d already evaluates cross-correlation rather than mathematical
    # convolution.  Flipping ``second`` here would therefore compute a
    # convolution and can move a known three-sample lag by tens of samples.
    corr = torch.nn.functional.conv1d(first[None, None], second[None, None], padding=n - 1)[0, 0]
    index = int(torch.argmax(corr).item())
    delta = 0.0
    if 0 < index < corr.numel() - 1:
        left, center, right = corr[index - 1:index + 2]
        denominator = left - 2 * center + right
        if abs(float(denominator)) > torch.finfo(corr.dtype).eps:
            delta = float(0.5 * (left - right) / denominator)
    lag = index - (n - 1) + delta
    quality = corr[index] / (torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)).clamp_min(torch.finfo(corr.dtype).tiny)
    return torch.as_tensor(lag, dtype=first.dtype, device=first.device), quality


def _linearized_time_shift(moving: torch.Tensor, static: torch.Tensor,
                           weight: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Low-Mach time shift from the static waveform's temporal tangent."""
    derivative = torch.zeros_like(static)
    derivative[1:-1] = (static[2:] - static[:-2]) / (2.0 * dt)
    derivative[0] = (static[1] - static[0]) / dt
    derivative[-1] = (static[-1] - static[-2]) / dt
    denominator = (weight * derivative.square()).sum().clamp_min(torch.finfo(static.dtype).tiny)
    shift = -(weight * (moving - static) * derivative).sum() / denominator
    weighted_moving = moving * weight.sqrt()
    weighted_static = static * weight.sqrt()
    quality = (weighted_moving * weighted_static).sum() / (
        torch.linalg.vector_norm(weighted_moving) * torch.linalg.vector_norm(weighted_static)
    ).clamp_min(torch.finfo(static.dtype).tiny)
    return shift, quality


def extract_reciprocal_delays_3d(flow_data: torch.Tensor, static_data: torch.Tensor, acquisition: Acquisition3D) -> DelayData3D:
    """Extract static-calibrated lags inside the predicted direct-wave window."""
    delays, quality = [], []
    nt = flow_data.shape[-1]
    # This metadata calculation is performed once and is not part of the
    # differentiable forward model.  Keep it on the CPU so extracting delays
    # does not depend on CUDA's optional complex-abs NVRTC kernel.
    source_signal_cpu = acquisition.source_signal.detach().cpu()
    source_peak = int(torch.argmax(source_signal_cpu.abs()).item())
    spectrum = torch.fft.rfft(source_signal_cpu)
    power = spectrum.real.square() + spectrum.imag.square()
    frequencies = torch.fft.rfftfreq(source_signal_cpu.numel(), acquisition.dt)
    frequency = float(frequencies[1:][torch.argmax(power[1:])])
    pre = max(4, int(round(0.75 / (frequency * acquisition.dt))))
    post = max(4, int(round(1.00 / (frequency * acquisition.dt))))
    for a, b in acquisition.reciprocal_pairs.tolist():
        distance = torch.linalg.vector_norm(acquisition.coordinates[b] - acquisition.coordinates[a])
        center = source_peak + int(round(float(distance) / (1480.0 * acquisition.dt)))
        start, stop = max(0, center - pre), min(nt, center + post + 1)
        window = torch.hann_window(stop - start, periodic=False, dtype=flow_data.dtype,
                                   device=flow_data.device).square()
        shift_ab, score_ab = _linearized_time_shift(
            flow_data[a, b, start:stop], static_data[a, b, start:stop], window, acquisition.dt)
        shift_ba, score_ba = _linearized_time_shift(
            flow_data[b, a, start:stop], static_data[b, a, start:stop], window, acquisition.dt)
        delays.append(shift_ab - shift_ba)
        quality.append(torch.minimum(score_ab, score_ba))
    return DelayData3D(torch.stack(delays), acquisition.reciprocal_pairs, acquisition.dt, torch.stack(quality))


def build_ray_matrix(acquisition: Acquisition3D, control_shape: tuple[int, int, int], lengths: tuple[float, float, float], c0: float = 1480.0, samples: int = 96) -> torch.Tensor:
    nx, ny, nz = control_shape
    rows = []
    for a, b in acquisition.reciprocal_pairs.tolist():
        start, end = acquisition.coordinates[a], acquisition.coordinates[b]
        ray = end - start
        length = torch.linalg.vector_norm(ray)
        tangent = ray / length
        row = torch.zeros(3, nx, ny, nz, dtype=start.dtype, device=start.device)
        for alpha in torch.linspace(0.0, 1.0, samples, dtype=start.dtype, device=start.device):
            point = start + alpha * ray
            indices = torch.stack((point[0] / lengths[0] * (nx - 1), point[1] / lengths[1] * (ny - 1), point[2] / lengths[2] * (nz - 1)))
            low = torch.floor(indices).to(torch.long)
            frac = indices - low
            low = torch.minimum(low, torch.tensor((nx - 2, ny - 2, nz - 2), device=low.device))
            for ix in (0, 1):
                for iy in (0, 1):
                    for iz in (0, 1):
                        weight = (frac[0] if ix else 1-frac[0]) * (frac[1] if iy else 1-frac[1]) * (frac[2] if iz else 1-frac[2])
                        row[:, low[0]+ix, low[1]+iy, low[2]+iz] += (-2.0 / c0**2) * tangent * length / (samples - 1) * weight
        rows.append(row.reshape(-1))
    return torch.stack(rows)


def invert_travel_time_3d(delay_data: DelayData3D, parameterization: dict[str, object]) -> InitialModel3D:
    matrix = parameterization.get("matrix")
    if not isinstance(matrix, torch.Tensor):
        matrix = build_ray_matrix(parameterization["acquisition"], parameterization["control_shape"], parameterization["lengths"], float(parameterization.get("c0", 1480.0)))
    weight = delay_data.quality.clamp_min(0.0).sqrt()
    weighted_matrix = matrix * weight[:, None]
    weighted_delays = delay_data.delays * weight
    regularization = float(parameterization.get("regularization", 1e-6))
    lhs = weighted_matrix.T @ weighted_matrix + regularization * torch.eye(matrix.shape[1], dtype=matrix.dtype, device=matrix.device)
    controls = torch.linalg.solve(lhs, weighted_matrix.T @ weighted_delays)
    singular = torch.linalg.svdvals(weighted_matrix)
    threshold = singular[0] * max(weighted_matrix.shape) * torch.finfo(singular.dtype).eps
    rank = int((singular > threshold).sum().item())
    condition = float((singular[0] / singular[rank - 1]).cpu()) if rank else float("inf")
    _, _, vh = torch.linalg.svd(weighted_matrix, full_matrices=False)
    shape = tuple(parameterization["control_shape"])
    velocity = controls.reshape(3, *shape)
    duct = parameterization.get("duct_control")
    beta = 0.0
    if isinstance(duct, torch.Tensor):
        beta = float((velocity * duct).sum().div(duct.square().sum().clamp_min(torch.finfo(velocity.dtype).tiny)).cpu())
    return InitialModel3D(beta, velocity, singular, rank, condition, vh[-1].reshape(3, *shape))
