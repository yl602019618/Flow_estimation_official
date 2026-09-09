"""Open-water finite-difference convected-wave observation generator."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch

from .config import SphereConfig, coerce_config
from .flow import tilted_vortex
from .geometry import SphereAcquisition


@dataclass(frozen=True)
class FullWaveResult:
    traces: np.ndarray
    dt_s: float
    grid_n: int
    runtime_s: float


@dataclass(frozen=True)
class FullWaveDelays:
    directed_shift_s: np.ndarray
    reciprocal_s: np.ndarray
    fit_residual_rms: np.ndarray


@dataclass(frozen=True)
class NoisyFullWaveObservation:
    static: FullWaveResult
    flow: FullWaveResult
    delays: FullWaveDelays
    amplitude_std: float
    target_delay_std_s: float
    realized_delay_std_s: float
    delay_bias_s: float
    peak_snr_db: float
    calibration_iterations: int


def _trilinear(points_m: np.ndarray, half_extent: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    spacing = 2.0 * half_extent / (n - 1)
    coordinate = (points_m + half_extent) / spacing
    lower = np.floor(coordinate).astype(np.int64)
    fraction = coordinate - lower
    if np.any(lower < 0) or np.any(lower + 1 >= n):
        raise ValueError("transducer lies outside the full-wave grid")
    indices, weights = [], []
    for ox in (0, 1):
        for oy in (0, 1):
            for oz in (0, 1):
                ix = lower[:, 0] + ox
                iy = lower[:, 1] + oy
                iz = lower[:, 2] + oz
                indices.append((ix * n + iy) * n + iz)
                weight = ((fraction[:, 0] if ox else 1-fraction[:, 0]) *
                          (fraction[:, 1] if oy else 1-fraction[:, 1]) *
                          (fraction[:, 2] if oz else 1-fraction[:, 2]))
                weights.append(weight)
    return np.stack(indices, axis=1), np.stack(weights, axis=1)


def _pml_profiles(n: int, layers: int, maximum: float, *, device: torch.device,
                  dtype: torch.dtype) -> torch.Tensor:
    index = torch.arange(n, device=device, dtype=dtype)
    distance = torch.minimum(index, n - 1 - index)
    ramp = torch.clamp((layers - distance) / layers, min=0.0).pow(4)
    return maximum * torch.stack((
        ramp[:, None, None].expand(n, n, n),
        ramp[None, :, None].expand(n, n, n),
        ramp[None, None, :].expand(n, n, n),
    ))


def _advection(q: torch.Tensor, velocity: torch.Tensor, spacing: float) -> torch.Tensor:
    advection = torch.zeros_like(q)
    for axis in range(3):
        dimension = axis + 1
        gradient = (torch.roll(q, 2, dimension) - 8*torch.roll(q, 1, dimension)
                    + 8*torch.roll(q, -1, dimension) - torch.roll(q, -2, dimension)) / (12*spacing)
        advection = advection + velocity[axis][None] * gradient
    return advection


def _second_derivative(field: torch.Tensor, dimension: int, spacing: float) -> torch.Tensor:
    return (-torch.roll(field, 2, dimension) + 16*torch.roll(field, 1, dimension) - 30*field
            + 16*torch.roll(field, -1, dimension) - torch.roll(field, -2, dimension)) / (12*spacing**2)


def simulate_open_water_fullwave(
    acquisition: SphereAcquisition,
    config: SphereConfig | str | dict,
    *,
    grid_n: int | None = None,
    with_flow: bool = True,
) -> FullWaveResult:
    cfg = coerce_config(config)
    section = cfg.section("full_wave")
    n = int(grid_n or section["production_grid_n"])
    dt = float(section["dt_s"])
    nt = int(round(float(section["record_time_s"]) / dt))
    half_extent = float(section["domain_half_extent_m"])
    spacing = 2.0 * half_extent / (n - 1)
    # Conservative fourth-order CFL guard.
    if cfg.c0 * dt / spacing > 0.45:
        raise ValueError("full-wave CFL exceeds the frozen 0.45 limit")
    requested = str(section.get("device", "cuda"))
    device = torch.device(requested if requested != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("full-wave configuration requests CUDA, but CUDA is unavailable")
    dtype = torch.float32
    axis = torch.linspace(-half_extent, half_extent, n, device=device, dtype=dtype)
    xx, yy, zz = torch.meshgrid(axis, axis, axis, indexing="ij")
    if with_flow:
        dimensionless = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3).cpu().numpy() / cfg.radius
        velocity_np = tilted_vortex(dimensionless, cfg)
        velocity = torch.as_tensor(velocity_np.T.reshape(3, n, n, n), device=device, dtype=dtype)
    else:
        velocity = torch.zeros((3, n, n, n), device=device, dtype=dtype)
    pml_layers = max(3, int(round(float(section["absorber_width_m"])/spacing)))
    sigma = _pml_profiles(n, pml_layers,
                          float(section["absorber_max_s_inv"]), device=device, dtype=dtype)
    boundary_mask = torch.ones((n, n, n), device=device, dtype=dtype)
    boundary_mask[:2] = 0; boundary_mask[-2:] = 0
    boundary_mask[:, :2] = 0; boundary_mask[:, -2:] = 0
    boundary_mask[:, :, :2] = 0; boundary_mask[:, :, -2:] = 0
    points_m = acquisition.points * cfg.radius
    indices_np, weights_np = _trilinear(points_m, half_extent, n)
    indices = torch.as_tensor(indices_np, device=device)
    weights = torch.as_tensor(weights_np, device=device, dtype=dtype)
    shots = acquisition.n_transducers
    shot_ids = torch.arange(shots, device=device)[:, None]
    # Berenger-style split-field scalar PML: p = p_x + p_y + p_z and
    # p_i,tt + sigma_i p_i,t = c^2 d_ii p - 2/3 U.grad(p_t) + s/3.
    previous = torch.zeros((shots, 3, n, n, n), device=device, dtype=dtype)
    current = torch.zeros_like(previous)
    traces = torch.empty((shots, shots, nt), device="cpu", dtype=dtype)
    frequency = float(cfg.section("signal")["center_frequency_hz"])
    delay = float(cfg.section("signal")["source_delay_cycles"]) / frequency
    time = torch.arange(nt, device=device, dtype=dtype) * dt
    tau = time - delay
    wavelet = (1 - 2*(torch.pi*frequency*tau)**2) * torch.exp(-(torch.pi*frequency*tau)**2)
    started = perf_counter()
    with torch.no_grad():
        for step in range(nt):
            total_current = current.sum(dim=1)
            total_previous = previous.sum(dim=1)
            q = (total_current - total_previous) / dt
            advection = _advection(q, velocity, spacing)
            following = torch.empty_like(current)
            for axis_index in range(3):
                directional = _second_derivative(total_current, axis_index + 1, spacing)
                sigma_dt = sigma[axis_index] * dt
                following[:, axis_index] = (
                    2*current[:, axis_index] - (1 - .5*sigma_dt)*previous[:, axis_index]
                    + dt**2 * (cfg.c0**2*directional - (2.0/3.0)*advection)
                ) / (1 + .5*sigma_dt)
                flat_component = following[:, axis_index].reshape(shots, -1)
                flat_component[shot_ids, indices] += wavelet[step] * weights / 3.0
            following.mul_(boundary_mask[None, None])
            total_following = following.sum(dim=1)
            receiver_values = total_following.reshape(shots, -1)[:, indices]
            traces[:, :, step] = torch.sum(receiver_values * weights[None], dim=-1).cpu()
            previous, current = current, following
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime = perf_counter() - started
    values = traces.numpy()
    if not np.isfinite(values).all():
        raise FloatingPointError("full-wave propagation produced NaN/Inf")
    return FullWaveResult(values, dt, n, runtime)


def extract_fullwave_delays(
    flow: FullWaveResult,
    static: FullWaveResult,
    acquisition: SphereAcquisition,
    config: SphereConfig | str | dict,
) -> FullWaveDelays:
    """Fit flow trace = gain*(static trace shifted by delta) near first arrival."""
    cfg = coerce_config(config)
    if flow.traces.shape != static.traces.shape or flow.dt_s != static.dt_s:
        raise ValueError("flow and static full-wave data are incompatible")
    frequency = float(cfg.section("signal")["center_frequency_hz"])
    source_delay = float(cfg.section("signal")["source_delay_cycles"]) / frequency
    half_window = float(cfg.section("full_wave")["extraction_half_window_cycles"]) / frequency
    directed = np.full((acquisition.n_transducers, acquisition.n_transducers), np.nan)
    residual = np.full_like(directed, np.nan)
    for pair_index, (a, b) in enumerate(acquisition.pairs):
        arrival = cfg.radius * acquisition.lengths[pair_index] / cfg.c0 + source_delay
        center = int(round(arrival / flow.dt_s))
        radius = int(round(half_window / flow.dt_s))
        selection = slice(max(2, center-radius), min(flow.traces.shape[-1]-2, center+radius+1))
        for source, receiver in ((a, b), (b, a)):
            reference_full = static.traces[source, receiver].astype(np.float64)
            reference = reference_full[selection]
            observed = flow.traces[source, receiver, selection].astype(np.float64)
            sample = np.arange(selection.start, selection.stop)
            derivative = (reference_full[sample-2] - 8*reference_full[sample-1]
                          + 8*reference_full[sample+1] - reference_full[sample+2]) / (12*flow.dt_s)
            design = np.column_stack((reference, derivative, np.ones_like(reference)))
            coefficient, *_ = np.linalg.lstsq(design, observed, rcond=None)
            shift = -coefficient[1] / coefficient[0]
            fitted = design @ coefficient
            directed[source, receiver] = shift
            residual[source, receiver] = np.linalg.norm(observed-fitted) / max(np.linalg.norm(observed), 1e-30)
    reciprocal = directed[acquisition.pairs[:, 0], acquisition.pairs[:, 1]] - \
                 directed[acquisition.pairs[:, 1], acquisition.pairs[:, 0]]
    return FullWaveDelays(directed, reciprocal, residual)


def calibrate_waveform_noise(
    static: FullWaveResult,
    flow: FullWaveResult,
    clean_delays: FullWaveDelays,
    acquisition: SphereAcquisition,
    config: SphereConfig | str | dict,
    target_delay_std_s: float,
    *,
    seed: int,
) -> NoisyFullWaveObservation:
    """Scale one global white-noise level to a reciprocal-delay uncertainty.

    Calibration uses only the clean/noisy waveform extraction difference.  It
    never reads a velocity field, ray reference, or reconstruction metric.
    """
    cfg = coerce_config(config)
    section = cfg.section("full_wave")
    generator = np.random.default_rng(seed)
    noise_static = generator.standard_normal(static.traces.shape, dtype=np.float32)
    noise_flow = generator.standard_normal(flow.traces.shape, dtype=np.float32)
    a, b = acquisition.pairs.T
    pair_peak = np.max(np.abs(static.traces[a, b]), axis=1)
    amplitude = float(np.median(pair_peak) *
                      float(section["waveform_noise_initial_peak_fraction"]))
    tolerance = float(section["waveform_noise_calibration_tolerance"])
    maximum_iterations = int(section["waveform_noise_calibration_iterations"])
    noisy_static = noisy_flow = noisy_delays = None
    realized = 0.0
    for iteration in range(1, maximum_iterations + 1):
        noisy_static = FullWaveResult(
            static.traces + amplitude*noise_static, static.dt_s, static.grid_n, static.runtime_s)
        noisy_flow = FullWaveResult(
            flow.traces + amplitude*noise_flow, flow.dt_s, flow.grid_n, flow.runtime_s)
        noisy_delays = extract_fullwave_delays(noisy_flow, noisy_static, acquisition, cfg)
        perturbation = noisy_delays.reciprocal_s-clean_delays.reciprocal_s
        realized = float(np.std(perturbation))
        if not np.isfinite(realized) or realized <= 0:
            raise FloatingPointError("waveform-noise calibration produced an invalid delay spread")
        if abs(realized/target_delay_std_s - 1.0) <= tolerance:
            break
        amplitude *= target_delay_std_s/realized
    assert noisy_static is not None and noisy_flow is not None and noisy_delays is not None
    perturbation = noisy_delays.reciprocal_s-clean_delays.reciprocal_s
    peak_snr_db = float(20*np.log10(np.median(pair_peak)/amplitude))
    return NoisyFullWaveObservation(
        noisy_static, noisy_flow, noisy_delays, amplitude, target_delay_std_s,
        float(np.std(perturbation)), float(np.mean(perturbation)), peak_snr_db, iteration,
    )
