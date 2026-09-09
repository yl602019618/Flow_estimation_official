from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint

from .config import Grid3D
from .operators import derivative8, staggered_divergence8, staggered_gradient8


@dataclass(frozen=True)
class ForwardConfig3D:
    grid: Grid3D
    c0: float = 1480.0
    include_shear: bool = True
    boundary: str = "sponge"  # sponge, rigid, periodic
    sponge_layers: int = 12
    checkpoint_steps: int = 32
    source_signal: torch.Tensor | None = None
    compile_step: bool = False
    sponge_mode: str = "pressure"


def _derivative(field: torch.Tensor, component: int | None, spatial_axis: int, config: ForwardConfig3D) -> torch.Tensor:
    axis = field.ndim - 3 + spatial_axis
    periodic = config.boundary == "periodic" or spatial_axis == 2 and config.boundary == "periodic"
    odd = component == spatial_axis and config.boundary != "periodic"
    return derivative8(field, axis, (config.grid.dx, config.grid.dy, config.grid.dz)[spatial_axis], periodic=periodic, odd=odd)


def _sponge(grid: Grid3D, layers: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    result = torch.zeros(grid.nz, dtype=dtype, device=device)
    layers = min(layers, max(0, (grid.nz - 1) // 2))
    if layers:
        distance = torch.arange(layers, 0, -1, dtype=dtype, device=device) / layers
        ramp = torch.sin(0.5 * torch.pi * distance).square()
        result[:layers] = ramp
        result[-layers:] = torch.flip(ramp, (0,))
    # About exp(-6) attenuation while crossing the layer at c0.
    return result * (6.0 * 1480.0 / max(layers * grid.dz, grid.dz))


def background_gradient(velocity: torch.Tensor, config: ForwardConfig3D) -> torch.Tensor:
    values = []
    for component in range(3):
        values.append(torch.stack([_derivative(velocity[component], component, axis, config) for axis in range(3)]))
    return torch.stack(values)  # component, derivative direction, x, y, z


def _staggered_gradient(field: torch.Tensor, spatial_axis: int, config: ForwardConfig3D) -> torch.Tensor:
    axis = field.ndim - 3 + spatial_axis
    return staggered_gradient8(
        field, axis, (config.grid.dx, config.grid.dy, config.grid.dz)[spatial_axis],
        periodic=config.boundary == "periodic",
    )


def _staggered_divergence(field: torch.Tensor, spatial_axis: int, config: ForwardConfig3D) -> torch.Tensor:
    axis = field.ndim - 3 + spatial_axis
    return staggered_divergence8(
        field, axis, (config.grid.dx, config.grid.dy, config.grid.dz)[spatial_axis],
        periodic=config.boundary == "periodic",
    )


def _zero_padded_face_slots(value: torch.Tensor, config: ForwardConfig3D) -> torch.Tensor:
    """Zero the non-physical terminal slot of every embedded face array."""
    if config.boundary == "periodic":
        return value
    value = value.clone()
    value[:, 0, -1, :, :] = 0.0
    value[:, 1, :, -1, :] = 0.0
    value[:, 2, :, :, -1] = 0.0
    return value


def _rhs(state: torch.Tensor, velocity: torch.Tensor, grad_u: torch.Tensor, config: ForwardConfig3D, source: torch.Tensor | None) -> torch.Tensor:
    pressure, acoustic_velocity = state[:, 0], state[:, 1:]
    # The acoustic pressure/velocity coupling is genuinely staggered.  The
    # material derivative remains node-centred; this is a separate smooth
    # background-flow term and does not reintroduce the collocated acoustic
    # Nyquist null mode.
    grad_p = torch.stack([_staggered_gradient(pressure, axis, config) for axis in range(3)], dim=1)
    div_w = sum(_staggered_divergence(acoustic_velocity[:, axis], axis, config) for axis in range(3))
    adv_p = sum(velocity[axis][None] * _derivative(pressure, None, axis, config) for axis in range(3))
    dp = -adv_p - config.c0 * div_w
    result_w: list[torch.Tensor] = []
    for component in range(3):
        adv = sum(velocity[axis][None] * _derivative(acoustic_velocity[:, component], component, axis, config) for axis in range(3))
        shear = sum(grad_u[component, axis][None] * acoustic_velocity[:, axis] for axis in range(3)) if config.include_shear else 0.0
        result_w.append(-adv - shear - config.c0 * grad_p[:, component])
    rhs_w = _zero_padded_face_slots(torch.stack(result_w, dim=1), config)
    rhs = torch.cat((dp[:, None], rhs_w), dim=1)
    if source is not None:
        rhs = rhs.clone()
        rhs[:, 0] = rhs[:, 0] + source
    if config.boundary == "sponge" and config.sponge_layers > 0:
        sigma = _sponge(config.grid, config.sponge_layers, state.dtype, state.device)
        if config.sponge_mode != "pressure":
            raise ValueError("the registered reciprocal sponge damps pressure only")
        # At zero background flow this yields the reciprocal scalar equation
        # p_tt - c^2 Laplacian(p) + sigma(z) p_t = 0.
        rhs = rhs.clone()
        rhs[:, 0] = rhs[:, 0] - sigma[None, None, None, :] * pressure
    return rhs


def _rk4(state: torch.Tensor, velocity: torch.Tensor, grad_u: torch.Tensor, config: ForwardConfig3D, source: torch.Tensor) -> torch.Tensor:
    dt = config.grid.dt
    k1 = _rhs(state, velocity, grad_u, config, source)
    k2 = _rhs(state + 0.5 * dt * k1, velocity, grad_u, config, source)
    k3 = _rhs(state + 0.5 * dt * k2, velocity, grad_u, config, source)
    k4 = _rhs(state + dt * k3, velocity, grad_u, config, source)
    result = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    # Normal velocity has no wall degree of freedom.  Only the terminal padding
    # slot of each embedded face array is constrained; zeroing both node ends
    # would incorrectly remove the first interior half-grid face.
    if config.boundary != "periodic":
        result = result.clone()
        result[:, 1, -1, :, :] = 0.0
        result[:, 2, :, -1, :] = 0.0
        result[:, 3, :, :, -1] = 0.0
    return result


def simulate_acoustic(U: torch.Tensor, source_batch: torch.Tensor, receiver_array: torch.Tensor, forward_config: ForwardConfig3D) -> torch.Tensor:
    """Return pressure traces with shape ``(source, receiver, time)``."""
    grid = forward_config.grid
    if U.shape != (3, grid.nx, grid.ny, grid.nz):
        raise ValueError("U has incompatible shape")
    if source_batch.ndim == 3:
        source_batch = source_batch[None]
    shots = source_batch.shape[0]
    state = torch.zeros((shots, 4, grid.nx, grid.ny, grid.nz), dtype=U.dtype, device=U.device)
    receivers = receiver_array.to(dtype=U.dtype, device=U.device)
    sources = source_batch.to(dtype=U.dtype, device=U.device)
    signal = forward_config.source_signal
    if signal is None:
        signal = torch.zeros(grid.nt, dtype=U.dtype, device=U.device)
    else:
        signal = signal.to(dtype=U.dtype, device=U.device)
    if signal.numel() != grid.nt:
        raise ValueError("source_signal length must equal grid.nt")
    grad_u = background_gradient(U, forward_config)
    stepper = _rk4
    if forward_config.compile_step and hasattr(torch, "compile"):
        # CUDA graphs are intentionally not requested: output reuse is unsafe for
        # the retained checkpoint states used by discrete autograd.
        stepper = torch.compile(_rk4, fullgraph=False)

    def run_block(initial: torch.Tensor, start: int, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        current = initial
        samples = []
        for step in range(start, start + count):
            samples.append(torch.einsum("sxyz,rxyz->sr", current[:, 0], receivers))
            current = stepper(current, U, grad_u, forward_config, sources * signal[step])
        return current, torch.stack(samples, dim=-1)

    blocks = []
    block_size = max(1, forward_config.checkpoint_steps)
    use_checkpoint = torch.is_grad_enabled() and (U.requires_grad or source_batch.requires_grad)
    for start in range(0, grid.nt, block_size):
        count = min(block_size, grid.nt - start)
        if use_checkpoint:
            # Bind integer loop bounds now; late binding silently changes gradients.
            fn = lambda value, _start=start, _count=count: run_block(value, _start, _count)
            state, trace = checkpoint(fn, state, use_reentrant=False)
        else:
            state, trace = run_block(state, start, count)
        blocks.append(trace)
    traces = torch.cat(blocks, dim=-1)
    if not torch.isfinite(traces).all():
        raise FloatingPointError("acoustic propagation produced NaN/Inf")
    return traces


def acoustic_energy(state: torch.Tensor) -> torch.Tensor:
    return 0.5 * state.square().sum()
