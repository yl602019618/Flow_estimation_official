from __future__ import annotations

import time
from dataclasses import replace

import torch

from .acoustic import ForwardConfig3D, simulate_acoustic
from .acquisition import Acquisition3D


def _shot_loss_gradient(velocity: torch.Tensor, acquisition: Acquisition3D,
                        forward: ForwardConfig3D, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    local = velocity.detach().clone().requires_grad_(True)
    sources = acquisition.source_apertures.index_select(0, indices)
    traces = simulate_acoustic(local, sources, acquisition.apertures, forward)
    loss = traces.square().sum() / acquisition.source_apertures.shape[0]
    gradient = torch.autograd.grad(loss, local)[0]
    return loss.detach(), gradient.detach()


def exact_two_gpu_gradient(velocity: torch.Tensor, acquisition: Acquisition3D,
                           forward: ForwardConfig3D) -> tuple[torch.Tensor, torch.Tensor]:
    """Split shots equally and sum exact gradients; one process, two CUDA devices.

    The production launcher can replace the final peer-to-peer sum with
    ``torch.distributed.all_reduce`` without changing the arithmetic contract.
    """
    if torch.cuda.device_count() < 2:
        raise RuntimeError("two CUDA devices are required")
    splits = torch.arange(acquisition.source_apertures.shape[0]).split(
        (acquisition.source_apertures.shape[0] + 1) // 2
    )
    losses, gradients = [], []
    for device_index, indices in enumerate(splits[:2]):
        device = torch.device(f"cuda:{device_index}")
        local_acquisition = replace(
            acquisition, coordinates=acquisition.coordinates.to(device), apertures=acquisition.apertures.to(device),
            source_apertures=acquisition.source_apertures.to(device), ordered_mask=acquisition.ordered_mask.to(device),
            reciprocal_pairs=acquisition.reciprocal_pairs.to(device), source_signal=acquisition.source_signal.to(device),
        )
        local_forward = replace(forward, source_signal=forward.source_signal.to(device) if forward.source_signal is not None else None)
        loss, gradient = _shot_loss_gradient(velocity.to(device), local_acquisition, local_forward, indices.to(device))
        losses.append(loss.to("cuda:0")); gradients.append(gradient.to("cuda:0"))
    return sum(losses), sum(gradients)


def benchmark_checkpoint_blocks(velocity: torch.Tensor, acquisition: Acquisition3D,
                                forward: ForwardConfig3D, candidates: tuple[int, ...] = (32, 64)) -> dict[int, float]:
    timings = {}
    for block in candidates:
        candidate = replace(forward, checkpoint_steps=block)
        if velocity.is_cuda: torch.cuda.synchronize(velocity.device)
        start = time.perf_counter()
        with torch.no_grad():
            simulate_acoustic(velocity, acquisition.source_apertures[:1], acquisition.apertures, candidate)
        if velocity.is_cuda: torch.cuda.synchronize(velocity.device)
        timings[block] = time.perf_counter() - start
    return timings
