import torch

from water_usct_3d.acquisition import Acquisition3D, build_acquisition_3d, ricker
from water_usct_3d.config import AcquisitionConfig3D, Grid3D
from water_usct_3d.data import _exact_noise
from water_usct_3d.travel_time import _subsample_lag, build_ray_matrix, extract_reciprocal_delays_3d


def test_subsample_lag_recovers_known_integer_and_fractional_shift():
    time = torch.arange(256, dtype=torch.float64)
    first = torch.exp(-0.5 * ((time - 100.0) / 9.0).square())
    for shift in (3.0, -2.25):
        second = torch.exp(-0.5 * ((time - (100.0 + shift)) / 9.0).square())
        lag, quality = _subsample_lag(first, second)
        assert abs(float(lag) + shift) < 2e-3
        assert float(quality) > 0.99


def test_reciprocal_delay_uses_direct_window_and_static_calibration():
    dt, nt = 3e-7, 800
    time = torch.arange(nt, dtype=torch.float64) * dt
    signal = ricker(time, 50_000.0, 2.0)
    coordinates = torch.tensor(((0.02, 0.02, 0.04), (0.08, 0.08, 0.16)), dtype=torch.float64)
    acquisition = Acquisition3D(
        coordinates, ("x0", "x1"), torch.empty(0), torch.empty(0),
        torch.tensor(((False, True), (True, False))), torch.tensor(((0, 1),)), signal, dt,
    )
    distance = torch.linalg.vector_norm(coordinates[1] - coordinates[0])
    center = 2.0 / 50_000.0 + float(distance) / 1480.0
    width = 9e-6
    static = torch.zeros((2, 2, nt), dtype=torch.float64)
    moving = torch.zeros_like(static)
    static_wave = torch.exp(-0.5 * ((time - center) / width).square())
    static[0, 1] = static_wave; static[1, 0] = static_wave
    shift = 0.35 * dt
    moving[0, 1] = torch.exp(-0.5 * ((time - (center - shift / 2)) / width).square())
    moving[1, 0] = torch.exp(-0.5 * ((time - (center + shift / 2)) / width).square())
    delay = extract_reciprocal_delays_3d(moving, static, acquisition)
    assert abs(float(delay.delays[0]) + shift) < 2e-10
    assert float(delay.quality[0]) > 0.99


def test_noise_has_exact_tracewise_rms():
    clean = torch.rand(4, 5, 32, dtype=torch.float64)
    noisy, noise = _exact_noise(clean, .01, 20260807)
    ratio = noise.square().mean(-1).sqrt() / clean.square().mean(-1).sqrt()
    assert torch.allclose(ratio, torch.full_like(ratio, .01), rtol=1e-12, atol=1e-12)
    assert torch.allclose(noisy, clean + noise)


def test_three_dimensional_ray_matrix_contract():
    grid = Grid3D(9, 9, 11, .1, .1, .2, 3e-7, 8*3e-7)
    config = AcquisitionConfig3D((.05,.1,.15), ((1/3,2/3),(.25,.75),(1/3,2/3)), wall_offset_cells=.5)
    acquisition = build_acquisition_3d(grid, config, dtype=torch.float64)
    matrix = build_ray_matrix(acquisition, (3, 3, 4), (.1, .1, .2), samples=8)
    assert matrix.shape == (276, 3*3*3*4)
    assert torch.isfinite(matrix).all()
    assert torch.count_nonzero(matrix) > 0
