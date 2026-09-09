"""Compare the 10 cm/100 kHz and 50 cm/20 kHz full-wave benchmarks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from .config import SphereConfig, load_sphere_config
from .storage import write_json


def _load_metrics(config: SphereConfig) -> dict:
    with (config.output_root / "fullwave" / "metrics.json").open("r", encoding="utf-8") as stream:
        return json.load(stream)


def build_scale_metrics(small: SphereConfig, large: SphereConfig) -> dict:
    small_metrics, large_metrics = _load_metrics(small), _load_metrics(large)
    small_delay = np.load(small.output_root / "fullwave" / "delays.npz")["observation_s"]
    large_delay = np.load(large.output_root / "fullwave" / "delays.npz")["observation_s"]
    expected_scale = large.radius/small.radius
    fitted_scale = float(np.dot(small_delay, large_delay)/np.dot(small_delay, small_delay))
    scaling_residual = large_delay-expected_scale*small_delay
    fine_n = int(small.section("full_wave")["production_grid_n"])
    fixed_frequency_n = int(round((fine_n-1)*expected_scale))+1
    fixed_frequency_points = fixed_frequency_n**3
    # Lower bound: previous/current/following for three split pressure fields,
    # float32, all 96 sources. Derivative temporaries are deliberately excluded.
    shots = int(small.section("acquisition")["n_transducers"])
    state_lower_bound_gib = fixed_frequency_points*shots*3*3*4/2**30
    fixed_frequency_work_multiplier = expected_scale**3 * expected_scale
    wavelength_small = small.c0/float(small.section("signal")["center_frequency_hz"])
    wavelength_large = large.c0/float(large.section("signal")["center_frequency_hz"])
    result = {
        "similarity": {
            "radius_scale": expected_scale,
            "frequency_scale": float(large.section("signal")["center_frequency_hz"])/
                               float(small.section("signal")["center_frequency_hz"]),
            "diameter_wavelengths_10cm": 2*small.radius/wavelength_small,
            "diameter_wavelengths_50cm_20khz": 2*large.radius/wavelength_large,
            "reciprocal_delay_fitted_scale": fitted_scale,
            "reciprocal_delay_scaling_rmse_ns": float(np.sqrt(np.mean(scaling_residual**2))/1e-9),
            "reciprocal_delay_scaling_correlation": float(np.corrcoef(
                expected_scale*small_delay, large_delay)[0, 1]),
        },
        "cases": {},
        "fixed_50cm_100khz_feasibility": {
            "diameter_wavelengths": 2*large.radius/wavelength_small,
            "required_fine_grid_n": fixed_frequency_n,
            "required_grid_points": fixed_frequency_points,
            "all_source_split_state_lower_bound_gib": state_lower_bound_gib,
            "work_multiplier_vs_10cm_100khz": fixed_frequency_work_multiplier,
            "note": "lower-bound memory excludes PML, velocity, derivatives, and stencil temporaries",
        },
    }
    for name, cfg, metrics in (("10cm_100khz", small, small_metrics),
                               ("50cm_20khz", large, large_metrics)):
        result["cases"][name] = {
            "radius_m": cfg.radius,
            "frequency_hz": float(cfg.section("signal")["center_frequency_hz"]),
            "fullwave_ray_rmse_ns": metrics["fine"]["ray_rmse_ns"],
            "grid_difference_rms_ns": metrics["coarse_fine_rms_ns"],
            "clean_relative_l2": metrics["datasets"]["clean"]["inversion"]["relative_l2_error"],
            "noise_5ns_relative_l2": metrics["datasets"]["noise_5ns"]["inversion"]["relative_l2_error"],
            "noise_10ns_relative_l2": metrics["datasets"]["noise_10ns"]["inversion"]["relative_l2_error"],
            "noise_5ns_correlation": metrics["datasets"]["noise_5ns"]["inversion"]["field_correlation"],
            "noise_10ns_correlation": metrics["datasets"]["noise_10ns"]["inversion"]["field_correlation"],
            "noise_5ns_peak_snr_db": metrics["datasets"]["noise_5ns"]["median_peak_snr_db"],
            "noise_10ns_peak_snr_db": metrics["datasets"]["noise_10ns"]["median_peak_snr_db"],
        }
    return result


def plot_scale_comparison(metrics: dict, small_delay: np.ndarray, large_delay: np.ndarray,
                          path: Path) -> None:
    cases = metrics["cases"]
    small = cases["10cm_100khz"]
    large = cases["50cm_20khz"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)

    ax = axes[0, 0]
    ax.add_patch(Circle((-.65, 0), .1, facecolor="#4c78a8", alpha=.75))
    ax.add_patch(Circle((.2, 0), .5, facecolor="#f58518", alpha=.55))
    ax.text(-.65, -.14, "R=10 cm\n100 kHz", ha="center", va="top")
    ax.text(.2, -.54, "R=50 cm\n20 kHz", ha="center", va="top")
    ax.set(xlim=(-.9, .75), ylim=(-.65, .65), aspect="equal", title="Physical aperture")
    ax.set_xlabel("metres"); ax.set_ylabel("metres"); ax.grid(alpha=.15)

    ax = axes[0, 1]
    expected = metrics["similarity"]["radius_scale"]*small_delay*1e9
    observed = large_delay*1e9
    limit = max(np.max(np.abs(expected)), np.max(np.abs(observed)))
    ax.scatter(expected, observed, s=5, alpha=.25)
    ax.plot([-limit, limit], [-limit, limit], "k--", lw=1)
    ax.set(xlabel="5 x 10 cm delay (ns)", ylabel="50 cm delay (ns)",
           title="Full-wave reciprocal-delay scaling")
    ax.grid(alpha=.2)

    ax = axes[1, 0]
    labels = ["clean", "5 ns", "10 ns"]
    small_error = 100*np.asarray([small["clean_relative_l2"], small["noise_5ns_relative_l2"],
                                  small["noise_10ns_relative_l2"]])
    large_error = 100*np.asarray([large["clean_relative_l2"], large["noise_5ns_relative_l2"],
                                  large["noise_10ns_relative_l2"]])
    x = np.arange(3); width = .36
    ax.bar(x-width/2, small_error, width, label="10 cm / 100 kHz")
    ax.bar(x+width/2, large_error, width, label="50 cm / 20 kHz")
    ax.set_xticks(x, labels); ax.set_ylabel("relative L2 error (%)")
    ax.set_title("Reconstruction at fixed timing noise"); ax.legend(); ax.grid(axis="y", alpha=.2)

    ax = axes[1, 1]
    snr_small = [small["noise_5ns_peak_snr_db"], small["noise_10ns_peak_snr_db"]]
    snr_large = [large["noise_5ns_peak_snr_db"], large["noise_10ns_peak_snr_db"]]
    x = np.arange(2)
    ax.bar(x-width/2, snr_small, width, label="10 cm / 100 kHz")
    ax.bar(x+width/2, snr_large, width, label="50 cm / 20 kHz")
    ax.set_xticks(x, ["5 ns", "10 ns"]); ax.set_ylabel("median peak SNR (dB)")
    feasibility = metrics["fixed_50cm_100khz_feasibility"]
    ax.set_title(f"Timing precision; fixed 100 kHz needs {feasibility['required_fine_grid_n']}³")
    ax.legend(); ax.grid(axis="y", alpha=.2)
    fig.savefig(path, dpi=180); plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-config", default="examples/02_ball_10cm/reference_config.yaml")
    parser.add_argument("--large-config", default="examples/03_ball_50cm/reference_config.yaml")
    parser.add_argument("--output", default="examples/04_scale_comparison/outputs")
    args = parser.parse_args(argv)
    small, large = load_sphere_config(args.small_config), load_sphere_config(args.large_config)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    metrics = build_scale_metrics(small, large)
    small_delay = np.load(small.output_root / "fullwave" / "delays.npz")["observation_s"]
    large_delay = np.load(large.output_root / "fullwave" / "delays.npz")["observation_s"]
    write_json(output / "metrics.json", metrics)
    plot_scale_comparison(metrics, small_delay, large_delay, output / "scale_comparison.png")
    print(json.dumps({"status": "passed", "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
