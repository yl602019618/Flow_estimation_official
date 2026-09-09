from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from water_usct_3d.sphere.basis import build_solenoidal_basis, build_solenoidal_ray_matrix
from water_usct_3d.sphere.config import load_sphere_config
from water_usct_3d.sphere.evaluation import reconstruction_metrics, truth_metrics
from water_usct_3d.sphere.flow import basis_centers, raw_mode_divergence, tilted_vortex
from water_usct_3d.sphere.forward import simulate_chord_travel_times
from water_usct_3d.sphere.geometry import (
    SphereAcquisition, build_fibonacci_sphere_acquisition, geometry_metrics,
)
from water_usct_3d.sphere.inversion import invert_sphere_tsvd
from water_usct_3d.sphere.pulse import extract_sphere_reciprocal_delays, synthesize_delayed_pulses


CONFIG = Path(__file__).parents[1] / "examples" / "02_ball_10cm" / "config.yaml"


@pytest.fixture(scope="module")
def setup():
    config = load_sphere_config(CONFIG)
    acquisition = build_fibonacci_sphere_acquisition(config)
    basis = build_solenoidal_basis(config)
    operator = build_solenoidal_ray_matrix(acquisition, basis, config)
    return config, acquisition, basis, operator


def test_fibonacci_geometry_and_complete_pairs():
    config = load_sphere_config(CONFIG)
    acquisition = build_fibonacci_sphere_acquisition(config)
    metrics = geometry_metrics(acquisition)
    assert acquisition.points.shape == (96, 3)
    assert acquisition.pairs.shape == (4560, 2)
    assert metrics["max_radius_error"] < 1e-14
    assert metrics["minimum_point_separation"] > 0
    assert metrics["minimum_rank"] == 3
    assert metrics["minimum_normalized_eigenvalue"] >= 0.30
    assert metrics["maximum_condition_number"] <= 1.2


def test_tilted_vortex_physical_constraints():
    config = load_sphere_config(CONFIG)
    metrics = truth_metrics(config)
    assert metrics["relative_divergence"] <= 1e-10
    assert metrics["wall_max_speed_m_s"] <= 1e-10
    assert metrics["maximum_speed_normalization_error"] <= 1e-6
    points = np.asarray([[.2, .1, -.3], [-.4, .25, .15]])
    assert np.all(np.linalg.norm(tilted_vortex(points, config), axis=1) > 0)
    assert np.all(np.any(np.abs(tilted_vortex(points, config)) > 0, axis=0))


def test_forward_sign_zero_and_quadrature_convergence():
    config = load_sphere_config(CONFIG)
    acquisition = build_fibonacci_sphere_acquisition(config, 32)
    flow = lambda points: tilted_vortex(points, config)
    q32 = simulate_chord_travel_times(flow, acquisition, config, quadrature_order=32)
    q64 = simulate_chord_travel_times(flow, acquisition, config, quadrature_order=64)
    zero = simulate_chord_travel_times(lambda p: np.zeros_like(p), acquisition, config)
    assert np.max(np.abs(zero.reciprocal_s)) == 0
    assert np.linalg.norm(q32.reciprocal_s-q64.reciprocal_s) / np.linalg.norm(q64.reciprocal_s) < 1e-8
    assert np.linalg.norm(q64.reciprocal_s-q64.linearized_s) / np.linalg.norm(q64.linearized_s) < 1e-6
    reversed_acq = SphereAcquisition(acquisition.points, acquisition.pairs[:, ::-1],
                                     -acquisition.directions, acquisition.lengths)
    reverse = simulate_chord_travel_times(flow, reversed_acq, config)
    np.testing.assert_allclose(reverse.reciprocal_s, -q64.reciprocal_s, atol=1e-18)


def test_pulse_extraction_is_sub_nanosecond():
    config = load_sphere_config(CONFIG)
    acquisition = build_fibonacci_sphere_acquisition(config, 12)
    data = simulate_chord_travel_times(lambda p: tilted_vortex(p, config), acquisition, config)
    traces, static = synthesize_delayed_pulses(data, config)
    extracted = extract_sphere_reciprocal_delays(traces, static, acquisition)
    error = extracted.reciprocal_s-data.reciprocal_s
    assert np.sqrt(np.mean(error**2)) <= .5e-9
    assert abs(np.mean(error)) <= .1e-9


def test_solenoidal_basis_count_boundary_and_gauge(setup):
    config, _, basis, _ = setup
    assert len(basis_centers(config)) == 123
    assert basis.n_raw == 369
    assert basis.n_modes == 364
    wall = build_fibonacci_sphere_acquisition(config, 32).points
    assert np.max(np.abs(basis.evaluate(wall))) <= 1e-10
    assert np.max(np.abs(raw_mode_divergence(wall, basis.centers, basis.sigma))) == 0


def test_clean_tsvd_end_to_end(setup):
    config, acquisition, _, operator = setup
    data = simulate_chord_travel_times(lambda p: tilted_vortex(p, config), acquisition, config)
    traces, static = synthesize_delayed_pulses(data, config)
    delays = extract_sphere_reciprocal_delays(traces, static, acquisition)
    result = invert_sphere_tsvd(delays, operator, 0.0)
    metrics = reconstruction_metrics(result, operator, config)
    assert metrics["relative_l2_error"] <= .02
    assert metrics["field_correlation"] >= .999
    assert metrics["relative_divergence"] <= 1e-10


@pytest.mark.parametrize("noise_ns,error_max,correlation_min", [(5, .10, .98), (10, .15, .95)])
def test_noisy_morozov_gate(setup, noise_ns, error_max, correlation_min):
    config, acquisition, _, operator = setup
    data = simulate_chord_travel_times(lambda p: tilted_vortex(p, config), acquisition, config,
                                       noise_std=noise_ns*1e-9,
                                       rng=np.random.default_rng(config.seed))
    result = invert_sphere_tsvd(data.reciprocal_s, operator, noise_ns*1e-9)
    metrics = reconstruction_metrics(result, operator, config)
    assert metrics["relative_l2_error"] <= error_max
    assert metrics["field_correlation"] >= correlation_min
    assert .9 <= metrics["residual_ratio"] <= 1.1
