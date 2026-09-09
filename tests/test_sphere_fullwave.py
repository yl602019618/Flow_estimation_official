from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from water_usct_3d.sphere.config import SphereConfig, load_sphere_config
from water_usct_3d.sphere.fullwave import (
    FullWaveResult, _pml_profiles, calibrate_waveform_noise,
    extract_fullwave_delays, simulate_open_water_fullwave,
)
from water_usct_3d.sphere.geometry import build_fibonacci_sphere_acquisition


CONFIG = Path(__file__).parents[1] / "examples" / "02_ball_10cm" / "config.yaml"


def test_split_pml_profiles_are_directional_and_zero_inside():
    sigma = _pml_profiles(17, 3, 10.0, device=torch.device("cpu"), dtype=torch.float64)
    assert sigma.shape == (3, 17, 17, 17)
    assert torch.all(sigma[:, 8, 8, 8] == 0)
    assert sigma[0, 0, 8, 8] > 0 and sigma[1, 0, 8, 8] == 0
    assert sigma[1, 8, 0, 8] > 0 and sigma[2, 8, 8, 0] > 0


def test_fullwave_delay_fit_recovers_subsample_reciprocal_shift():
    config = load_sphere_config(CONFIG)
    acquisition = build_fibonacci_sphere_acquisition(config, 6)
    dt = float(config.section("full_wave")["dt_s"])
    nt = int(round(float(config.section("full_wave")["record_time_s"])/dt))
    frequency = float(config.section("signal")["center_frequency_hz"])
    source_delay = float(config.section("signal")["source_delay_cycles"])/frequency
    time = np.arange(nt)*dt
    static = np.zeros((6, 6, nt), dtype=np.float32)
    flow = np.zeros_like(static)
    expected = np.zeros((6, 6))
    for pair_index, (a, b) in enumerate(acquisition.pairs):
        arrival = config.radius*acquisition.lengths[pair_index]/config.c0 + source_delay
        expected[a, b] = (pair_index % 5 - 2)*8e-9
        expected[b, a] = -(pair_index % 3 - 1)*6e-9
        for source, receiver in ((a, b), (b, a)):
            tau0 = time-arrival
            tau1 = time-arrival-expected[source, receiver]
            x0 = (np.pi*frequency*tau0)**2
            x1 = (np.pi*frequency*tau1)**2
            static[source, receiver] = (1-2*x0)*np.exp(-x0)
            flow[source, receiver] = (1-2*x1)*np.exp(-x1)
    result = extract_fullwave_delays(FullWaveResult(flow, dt, 1, 0.0),
                                     FullWaveResult(static, dt, 1, 0.0),
                                     acquisition, config)
    target = expected[acquisition.pairs[:, 0], acquisition.pairs[:, 1]] - \
             expected[acquisition.pairs[:, 1], acquisition.pairs[:, 0]]
    assert np.sqrt(np.mean((result.reciprocal_s-target)**2)) < .1e-9


def test_fullwave_cfl_guard_fails_before_propagation():
    config = load_sphere_config(CONFIG)
    raw = copy.deepcopy(config.raw)
    raw["full_wave"]["dt_s"] = 1e-5
    unsafe = SphereConfig(raw, "test")
    acquisition = build_fibonacci_sphere_acquisition(unsafe, 4)
    with pytest.raises(ValueError, match="CFL"):
        simulate_open_water_fullwave(acquisition, unsafe, grid_n=17)


def test_waveform_noise_is_calibrated_without_ray_truth():
    config = load_sphere_config(CONFIG)
    acquisition = build_fibonacci_sphere_acquisition(config, 6)
    dt = float(config.section("full_wave")["dt_s"])
    nt = int(round(float(config.section("full_wave")["record_time_s"])/dt))
    frequency = float(config.section("signal")["center_frequency_hz"])
    delay = float(config.section("signal")["source_delay_cycles"])/frequency
    time = np.arange(nt)*dt
    traces = np.zeros((6, 6, nt), dtype=np.float32)
    for pair_index, (a, b) in enumerate(acquisition.pairs):
        arrival = config.radius*acquisition.lengths[pair_index]/config.c0 + delay
        value = (np.pi*frequency*(time-arrival))**2
        pulse = ((1-2*value)*np.exp(-value)).astype(np.float32)
        traces[a, b] = pulse; traces[b, a] = pulse
    static = FullWaveResult(traces, dt, 1, 0.0)
    flow = FullWaveResult(traces.copy(), dt, 1, 0.0)
    clean = extract_fullwave_delays(flow, static, acquisition, config)
    target = 5e-9
    noisy = calibrate_waveform_noise(static, flow, clean, acquisition, config,
                                     target, seed=config.seed)
    assert abs(noisy.realized_delay_std_s/target-1) <= 1e-3
    assert noisy.amplitude_std > 0
    assert np.isfinite(noisy.peak_snr_db)
