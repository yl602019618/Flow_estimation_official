from pathlib import Path

import torch

from water_usct_3d.acoustic import ForwardConfig3D, _rhs, _sponge, background_gradient, simulate_acoustic
from water_usct_3d.acquisition import build_acquisition_3d
from water_usct_3d.config import AcquisitionConfig3D, Grid3D, fine_grid, load_config, sponge_layers_for_grid
from water_usct_3d.verification import _valid_receiver_grid_mismatch


def _setup():
    grid = Grid3D(9, 9, 11, .1, .1, .2, 3e-7, 8*3e-7)
    ac = AcquisitionConfig3D((.05,.1,.15), ((1/3,2/3),(.25,.75),(1/3,2/3)), wall_offset_cells=.5)
    acquisition = build_acquisition_3d(grid, ac, dtype=torch.float64)
    signal = torch.zeros(grid.nt, dtype=torch.float64); signal[0] = 1.0
    forward = ForwardConfig3D(grid, source_signal=signal, boundary="rigid", sponge_layers=0, checkpoint_steps=4)
    return grid, acquisition, forward


def test_trace_shape_finite_and_static_reciprocity():
    grid, acquisition, forward = _setup()
    velocity = torch.zeros(3, grid.nx, grid.ny, grid.nz, dtype=torch.float64)
    traces = simulate_acoustic(velocity, acquisition.source_apertures, acquisition.apertures, forward)
    assert traces.shape == (24, 24, grid.nt)
    assert torch.isfinite(traces).all()
    errors = torch.stack([traces[a,b] - traces[b,a] for a,b in acquisition.reciprocal_pairs.tolist()])
    refs = torch.stack([traces[a,b] for a,b in acquisition.reciprocal_pairs.tolist()])
    assert float(torch.linalg.vector_norm(errors) / torch.linalg.vector_norm(refs)) < 1e-10


def test_checkpointed_velocity_gradient_matches_full_history():
    grid, acquisition, forward = _setup()
    velocity = torch.zeros(3, grid.nx, grid.ny, grid.nz, dtype=torch.float64, requires_grad=True)
    traces = simulate_acoustic(velocity, acquisition.source_apertures[:1], acquisition.apertures[:2], forward)
    loss = traces.square().sum(); gradient = torch.autograd.grad(loss, velocity)[0]
    velocity2 = torch.zeros_like(velocity, requires_grad=True)
    traces2 = simulate_acoustic(velocity2, acquisition.source_apertures[:1], acquisition.apertures[:2], ForwardConfig3D(grid, source_signal=forward.source_signal, boundary="rigid", sponge_layers=0, checkpoint_steps=grid.nt))
    gradient2 = torch.autograd.grad(traces2.square().sum(), velocity2)[0]
    assert torch.allclose(gradient, gradient2, rtol=1e-9, atol=1e-12)


def test_registered_sponge_damps_pressure_only():
    grid = Grid3D(9, 9, 11, .1, .1, .2, 3e-7, 3e-6)
    torch.manual_seed(4)
    state = torch.randn(2, 4, grid.nx, grid.ny, grid.nz, dtype=torch.float64)
    velocity = torch.zeros(3, grid.nx, grid.ny, grid.nz, dtype=torch.float64)
    damped = ForwardConfig3D(grid, boundary="sponge", sponge_layers=3, sponge_mode="pressure")
    undamped = ForwardConfig3D(grid, boundary="sponge", sponge_layers=0, sponge_mode="pressure")
    grad = background_gradient(velocity, damped)
    difference = _rhs(state, velocity, grad, damped, None) - _rhs(state, velocity, grad, undamped, None)
    sigma = _sponge(grid, 3, state.dtype, state.device)
    assert torch.allclose(difference[:, 0], -sigma[None, None, None, :] * state[:, 0])
    assert torch.count_nonzero(difference[:, 1:]) == 0


def test_sponge_physical_width_is_grid_independent():
    config = load_config(Path(__file__).parents[1] / "examples/01_square_duct/config.yaml")
    fine = fine_grid(config)
    assert sponge_layers_for_grid(config, config.grid) == 12
    assert sponge_layers_for_grid(config, fine) == 15
    assert abs(12 * config.grid.dz - 15 * fine.dz) < 1e-15


def test_pressure_sponge_point_response_is_reciprocal():
    grid = Grid3D(9, 9, 17, .1, .1, .2, 1e-6, 120e-6)
    acquisition_config = AcquisitionConfig3D(
        (.04, .10, .16), ((1/3, 2/3), (.25, .75), (1/3, 2/3)),
        wall_offset_cells=.5, sponge_layers=3, sponge_mode="pressure",
    )
    acquisition = build_acquisition_3d(grid, acquisition_config, dtype=torch.float64)
    signal = torch.zeros(grid.nt, dtype=torch.float64); signal[0] = 1.0
    forward = ForwardConfig3D(grid, boundary="sponge", sponge_layers=3,
                              checkpoint_steps=grid.nt, source_signal=signal,
                              sponge_mode="pressure")
    velocity = torch.zeros(3, grid.nx, grid.ny, grid.nz, dtype=torch.float64)
    indices = torch.tensor((0, 8))
    traces = simulate_acoustic(velocity, acquisition.source_apertures[indices],
                               acquisition.apertures[indices], forward)
    error = torch.linalg.vector_norm(traces[0, 1] - traces[1, 0])
    reference = torch.linalg.vector_norm(traces[0, 1])
    assert float(error / reference) < 1e-10


def test_grid_mismatch_excludes_collocated_self_channel():
    fine = torch.ones((1, 3, 4), dtype=torch.float64)
    coarse = fine.clone()
    coarse[:, 0] = 100.0
    coarse[:, 1] = 2.0
    mask = ~torch.eye(3, dtype=torch.bool)
    registered, including_self = _valid_receiver_grid_mismatch(coarse, fine, mask)
    assert abs(registered - 1 / 2**0.5) < 1e-14
    assert including_self > 50.0
