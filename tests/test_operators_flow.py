import torch

from water_usct_3d.config import Grid3D
from water_usct_3d.flow import FlowConfig, simulate_flow
from water_usct_3d.mac import empty_mac_state, enforce_no_slip, mac_divergence, project_mac
from water_usct_3d.operators import (
    curl, derivative8, divergence, staggered_divergence8, staggered_gradient8,
)


def test_eighth_order_periodic_derivative():
    errors = []
    for n in (17, 33):
        x = torch.arange(n, dtype=torch.float64) / n
        error = torch.linalg.vector_norm(derivative8(torch.sin(2*torch.pi*x), 0, 1/n, periodic=True) - 2*torch.pi*torch.cos(2*torch.pi*x))
        errors.append(float(error))
    assert errors[0] / errors[1] > 100


def test_eighth_order_periodic_staggered_derivative_and_nyquist_response():
    errors = []
    for n in (17, 33):
        x = torch.arange(n, dtype=torch.float64) / n
        faces = (torch.arange(n, dtype=torch.float64) + 0.5) / n
        numerical = staggered_gradient8(torch.sin(2*torch.pi*x), 0, 1/n, periodic=True)
        errors.append(float(torch.linalg.vector_norm(numerical - 2*torch.pi*torch.cos(2*torch.pi*faces))))
    assert errors[0] / errors[1] > 100
    checkerboard = (-1.0) ** torch.arange(16, dtype=torch.float64)
    response = staggered_gradient8(checkerboard, 0, 1.0, periodic=True)
    assert float(torch.linalg.vector_norm(response)) > 1.0


def test_staggered_gradient_divergence_are_negative_adjoints():
    torch.manual_seed(8)
    for periodic in (False, True):
        pressure = torch.randn(17, dtype=torch.float64)
        faces = torch.randn(17, dtype=torch.float64)
        gradient = staggered_gradient8(pressure, 0, 0.1, periodic=periodic)
        divergence_value = staggered_divergence8(faces, 0, 0.1, periodic=periodic)
        node_weights = torch.ones(17, dtype=torch.float64)
        if not periodic:
            node_weights[[0, -1]] = 0.5
            faces = faces.clone(); faces[-1] = 0.0
        defect = torch.dot(gradient, faces) + torch.dot(node_weights * pressure, divergence_value)
        scale = torch.linalg.vector_norm(gradient) * torch.linalg.vector_norm(faces)
        assert float(abs(defect) / scale) < 1e-14
        if not periodic:
            assert gradient[-1] == 0.0


def test_discrete_divergence_of_curl_and_wall_normals():
    grid = Grid3D(17, 17, 19, .1, .1, .2, 1e-7, 1e-6)
    torch.manual_seed(7)
    potential = torch.randn(3, grid.nx, grid.ny, grid.nz, dtype=torch.float64)
    velocity = curl(potential, (grid.dx, grid.dy, grid.dz), order=2)
    div = divergence(velocity, (grid.dx, grid.dy, grid.dz), order=2)
    assert float(torch.linalg.vector_norm(div) / torch.linalg.vector_norm(velocity)) < 1e-12


def test_duct_solver_matches_fourier_reference():
    grid = Grid3D(33, 33, 17, .1, .1, .2, 1e-7, 1e-6)
    result = simulate_flow(FlowConfig(grid))
    assert result.metrics["relative_l2"] < 0.01
    assert result.metrics["relative_divergence"] < 1e-12
    assert result.metrics["wall_velocity_relative"] == 0.0
    assert result.metrics["momentum_residual"] < 1e-6
    assert result.metrics["steady_relative_change"] < 1e-8


def test_matrix_free_mac_projection_removes_divergence():
    grid = Grid3D(9, 9, 11, .1, .1, .2, 1e-4, 1e-3)
    torch.manual_seed(11)
    state = empty_mac_state(grid, dtype=torch.float64, device=torch.device("cpu"))
    state.u.normal_(); state.v.normal_(); state.w.normal_(); enforce_no_slip(state)
    before = torch.linalg.vector_norm(mac_divergence(state, grid))
    projected, history = project_mac(state, grid, 1e-4, 998.0, rtol=1e-10)
    after = torch.linalg.vector_norm(mac_divergence(projected, grid))
    assert float(after / before) < 1e-9
    assert history[-1] < 1e-10


def test_manufactured_mode_returns_actual_projected_solver_field():
    grid = Grid3D(9, 9, 11, .1, .1, .2, 1e-4, 1e-3)
    result = simulate_flow(FlowConfig(grid, mode="manufactured"))
    assert result.metrics["relative_l2"] < 0.1
    assert result.metrics["mac_relative_divergence"] < 1e-10
    assert result.metrics["steady_relative_change"] < 1e-12
