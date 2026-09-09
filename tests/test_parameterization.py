import torch

from water_usct_3d.config import Grid3D
from water_usct_3d.operators import divergence
from water_usct_3d.parameterization import vector_potential_to_velocity


def test_vector_potential_velocity_is_constrained():
    grid = Grid3D(17, 17, 19, .1, .1, .2, 1e-7, 1e-6)
    torch.manual_seed(3)
    control = torch.randn(3, 6, 6, 8, dtype=torch.float64) * 1e-4
    velocity = vector_potential_to_velocity(0.7, control, grid)
    div = divergence(velocity, (grid.dx, grid.dy, grid.dz), order=2)
    assert float(torch.linalg.vector_norm(div) / (torch.linalg.vector_norm(velocity) / grid.dx)) < 1e-12
    assert velocity[0, 0].abs().max() < 1e-12
    assert velocity[0, -1].abs().max() < 1e-12
    assert velocity[1, :, 0].abs().max() < 1e-12
    assert velocity[1, :, -1].abs().max() < 1e-12
