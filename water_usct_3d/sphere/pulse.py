"""Ricker pulse synthesis and sub-sample reciprocal-delay extraction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SphereConfig, coerce_config
from .forward import TravelTimeDataset
from .geometry import SphereAcquisition


@dataclass(frozen=True)
class PulseStatic:
    dt_s: float
    center_frequency_hz: float
    source_delay_s: float
    static_travel_s: np.ndarray


@dataclass(frozen=True)
class DelayData3D:
    reciprocal_s: np.ndarray
    forward_s: np.ndarray
    reverse_s: np.ndarray
    noise_std_s: float = 0.0


def _ricker(tau: np.ndarray, frequency: float) -> np.ndarray:
    value = (np.pi * frequency * tau)**2
    return (1.0 - 2.0 * value) * np.exp(-value)


def synthesize_delayed_pulses(
    travel_times: TravelTimeDataset,
    config: SphereConfig | str | dict,
) -> tuple[np.ndarray, PulseStatic]:
    cfg = coerce_config(config)
    signal = cfg.section("signal")
    dt = float(signal["dt_s"])
    nt = int(round(float(signal["record_time_s"]) / dt))
    frequency = float(signal["center_frequency_hz"])
    source_delay = float(signal["source_delay_cycles"]) / frequency
    time = np.arange(nt, dtype=np.float64) * dt
    arrivals = np.stack((travel_times.forward_s, travel_times.reverse_s))
    tau = time[None, None, :] - arrivals[:, :, None] - source_delay
    traces = _ricker(tau, frequency).astype(np.float32)
    static = PulseStatic(dt, frequency, source_delay, travel_times.static_s)
    return traces, static


def _peak_times(traces: np.ndarray, dt: float, delay: float) -> np.ndarray:
    flat = traces.reshape(-1, traces.shape[-1]).astype(np.float64)
    peak = np.argmax(flat, axis=1)
    if np.any((peak == 0) | (peak == flat.shape[1] - 1)):
        raise ValueError("pulse peak lies on the recording boundary")
    rows = np.arange(len(flat))
    ym = flat[rows, peak - 1]
    y0 = flat[rows, peak]
    yp = flat[rows, peak + 1]
    denominator = ym - 2.0 * y0 + yp
    offset = 0.5 * (ym - yp) / denominator
    return (peak + offset) * dt - delay


def extract_sphere_reciprocal_delays(
    traces: np.ndarray,
    static: PulseStatic,
    acquisition: SphereAcquisition,
) -> DelayData3D:
    if traces.shape[:2] != (2, acquisition.n_pairs):
        raise ValueError("traces must have shape (2, n_pairs, nt)")
    arrivals = _peak_times(traces, static.dt_s, static.source_delay_s).reshape(2, -1)
    difference = arrivals[0] - arrivals[1]
    return DelayData3D(difference, arrivals[0], arrivals[1],
                       float(np.std(difference - np.mean(difference))))

