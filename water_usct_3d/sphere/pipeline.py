"""Fail-closed orchestration for the frozen unit-ball benchmark."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .basis import build_solenoidal_basis, build_solenoidal_ray_matrix
from .config import SphereConfig, coerce_config
from .evaluation import reconstruction_metrics, truth_metrics
from .flow import tilted_vortex
from .forward import simulate_chord_travel_times
from .geometry import build_fibonacci_sphere_acquisition, geometry_metrics
from .inversion import invert_sphere_tsvd
from .pulse import extract_sphere_reciprocal_delays, synthesize_delayed_pulses
from .storage import save_dataset, save_operator, write_json, write_snapshots
from .visualization import (plot_ablation, plot_coverage, plot_fit, plot_sensors,
                            plot_slices, plot_spectrum)


class GateFailure(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateFailure(message)


def _name(noise_ns: float) -> str:
    return "clean" if noise_ns == 0 else f"noise_{int(noise_ns)}ns"


def run_benchmark(config: SphereConfig | str | dict, *, through: str = "evaluate") -> dict:
    cfg = coerce_config(config)
    root = cfg.output_root
    write_snapshots(cfg, root)
    acquisition = build_fibonacci_sphere_acquisition(cfg)
    geom = geometry_metrics(acquisition)
    write_json(root / "geometry_metrics.json", geom)
    plot_sensors(acquisition, root / "sensors.png")
    plot_coverage(acquisition, root / "coverage.png")
    _require(geom["n_pairs"] == 4560, "geometry gate: reciprocal pair count")
    _require(geom["max_radius_error"] < 1e-12 and geom["minimum_point_separation"] > 0,
             "geometry gate: invalid sphere points")
    _require(geom["minimum_rank"] == 3 and geom["minimum_normalized_eigenvalue"] >= .30
             and geom["maximum_condition_number"] <= 1.2, "geometry coverage gate")
    if through == "geometry":
        return {"geometry": geom}

    truth_info = truth_metrics(cfg)
    write_json(root / "truth_metrics.json", truth_info)
    _require(truth_info["relative_divergence"] <= 1e-10 and
             truth_info["wall_max_speed_m_s"] <= 1e-10 and
             truth_info["maximum_speed_normalization_error"] <= 1e-6, "truth physics gate")
    flow = lambda points: tilted_vortex(points, cfg)
    q32 = simulate_chord_travel_times(flow, acquisition, cfg, quadrature_order=32)
    clean = simulate_chord_travel_times(flow, acquisition, cfg, quadrature_order=64)
    zero = simulate_chord_travel_times(lambda x: np.zeros_like(x), acquisition, cfg)
    quadrature_relative = float(np.linalg.norm(q32.reciprocal_s-clean.reciprocal_s) /
                                np.linalg.norm(clean.reciprocal_s))
    linear_relative = float(np.linalg.norm(clean.reciprocal_s-clean.linearized_s) /
                            np.linalg.norm(clean.linearized_s))
    forward_metrics = {
        "zero_flow_max_abs_s": float(np.max(np.abs(zero.reciprocal_s))),
        "swap_sign_max_abs_s": float(np.max(np.abs(clean.reciprocal_s +
                                                    (clean.reverse_s-clean.forward_s)))),
        "quadrature_32_64_relative": quadrature_relative,
        "exact_linearized_relative": linear_relative,
    }
    write_json(root / "forward_metrics.json", forward_metrics)
    _require(forward_metrics["zero_flow_max_abs_s"] <= 1e-18 and
             forward_metrics["swap_sign_max_abs_s"] <= 1e-18 and
             quadrature_relative <= 1e-8 and linear_relative <= 1e-6, "forward gate")

    noise_values = [float(x) for x in cfg.section("noise")["reciprocal_std_ns"]]
    datasets = {}
    extracted = {}
    pulse_metrics = {}
    generator = np.random.default_rng(cfg.seed)
    for noise_ns in noise_values:
        name = _name(noise_ns)
        data = clean if noise_ns == 0 else simulate_chord_travel_times(
            flow, acquisition, cfg, noise_std=noise_ns*1e-9, rng=generator)
        traces, static = synthesize_delayed_pulses(data, cfg)
        delays = extract_sphere_reciprocal_delays(traces, static, acquisition)
        error = delays.reciprocal_s - data.reciprocal_s
        pulse_metrics[name] = {"delay_rmse_ns": float(np.sqrt(np.mean(error**2))/1e-9),
                               "delay_bias_ns": float(np.mean(error)/1e-9),
                               "realized_noise_std_ns": float(np.std(
                                   data.reciprocal_s-clean.reciprocal_s)/1e-9)}
        save_dataset(root / "data" / f"{name}.h5", data, traces, static)
        datasets[name], extracted[name] = data, delays
    write_json(root / "pulse_metrics.json", pulse_metrics)
    _require(pulse_metrics["clean"]["delay_rmse_ns"] <= .5 and
             abs(pulse_metrics["clean"]["delay_bias_ns"]) <= .1, "pulse extraction gate")
    if through == "generate":
        return {"geometry": geom, "truth": truth_info, "forward": forward_metrics,
                "pulse": pulse_metrics}

    basis = build_solenoidal_basis(cfg)
    operator = build_solenoidal_ray_matrix(acquisition, basis, cfg)
    save_operator(root / "operator.npz", acquisition, operator)
    plot_spectrum(operator.singular_values, root / "singular_spectrum.png")
    results, metrics = {}, {}
    cutoff = float(cfg.section("inversion")["clean_relative_singular_cutoff"])
    for noise_ns in noise_values:
        name = _name(noise_ns)
        result = invert_sphere_tsvd(extracted[name], operator, noise_ns*1e-9,
                                    clean_relative_cutoff=cutoff)
        results[name] = result
        metrics[name] = reconstruction_metrics(result, operator, cfg)
    write_json(root / "inversion_metrics.json", metrics)
    gates = cfg.section("evaluation")
    _require(metrics["clean"]["relative_l2_error"] <= float(gates["clean_relative_l2_max"]),
             "clean inversion gate")
    for ns in (5, 10):
        name = f"noise_{ns}ns"
        _require(metrics[name]["relative_l2_error"] <= float(gates[f"noise_{ns}ns_relative_l2_max"])
                 and metrics[name]["field_correlation"] >= float(gates[f"noise_{ns}ns_correlation_min"])
                 and float(gates["residual_ratio_min"]) <= metrics[name]["residual_ratio"] <=
                 float(gates["residual_ratio_max"]), f"{name} inversion gate")
    if through == "invert":
        return metrics

    ablation = []
    for n in cfg.section("acquisition")["ablation_counts"]:
        acq_n = build_fibonacci_sphere_acquisition(cfg, int(n))
        gm = geometry_metrics(acq_n)
        op_n = operator if int(n) == acquisition.n_transducers else build_solenoidal_ray_matrix(acq_n, basis, cfg)
        gm["ray_condition_number"] = float(op_n.singular_values[0] / op_n.singular_values[-1])
        gm["ray_rank"] = int(np.count_nonzero(op_n.singular_values/op_n.singular_values[0] >= cutoff))
        ablation.append(gm)
    write_json(root / "ablation_metrics.json", ablation)
    plot_ablation(ablation, root / "ablation.png")

    axis = np.linspace(-1, 1, 101)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    xy = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    xz = np.column_stack((xx.ravel(), np.zeros(xx.size), yy.ravel()))
    yz = np.column_stack((np.zeros(xx.size), xx.ravel(), yy.ravel()))
    plane_points = [plane[np.linalg.norm(plane, axis=1) <= 1] for plane in (xy, xz, yz)]
    points = np.concatenate(plane_points)
    planes = np.concatenate([np.full(len(value), index, dtype=np.int8)
                             for index, value in enumerate(plane_points)])
    truth_slice = tilted_vortex(points, cfg)
    estimates = {name: result.velocity(points, operator) for name, result in results.items()}
    plot_slices(points, planes, truth_slice, estimates, root / "velocity_slices.png")
    plot_fit(extracted["noise_10ns"].reciprocal_s,
             results["noise_10ns"].predicted_delays_s, root / "travel_time_fit.png")
    np.savez_compressed(root / "velocity_fields.npz", points=points, planes=planes,
                        truth=truth_slice,
                        **estimates)
    summary = {"status": "passed", "geometry": geom, "truth": truth_info,
               "forward": forward_metrics, "pulse": pulse_metrics,
               "inversion": metrics, "ablation": ablation}
    write_json(root / "metrics.json", summary)
    return summary
