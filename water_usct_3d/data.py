from __future__ import annotations

import json
import platform
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as functional
import yaml

from .acoustic import ForwardConfig3D, simulate_acoustic
from .acquisition import build_acquisition_3d
from .config import DemoConfig3D, fine_grid, sponge_layers_for_grid
from .flow import FlowConfig, simulate_flow


def case_directory(config: DemoConfig3D, case: str) -> Path:
    path = Path(config.output_root) / case
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_generation_gates(config: DemoConfig3D) -> None:
    root = Path(config.output_root)
    paths = {
        "flow": root / "verification/flow_metrics.json",
        "geometry": root / "geometry/metrics.json",
        "acoustic": root / "verification/acoustic_metrics.json",
    }
    # Compatibility with the first formal run; it remains a valid failed artifact.
    if not paths["acoustic"].exists() and (root / "verification/acoustic_partial_metrics.json").exists():
        paths["acoustic"] = root / "verification/acoustic_partial_metrics.json"
    failures = []
    for stage, path in paths.items():
        if not path.exists():
            failures.append(f"{stage}: missing {path}"); continue
        with path.open(encoding="utf-8") as stream:
            metrics = json.load(stream)
        passed = bool(metrics.get("passed", metrics.get("pass", False)))
        if stage in {"flow", "acoustic"}:
            passed = passed and bool(metrics.get("formal_gate_complete", False))
        if not passed:
            failures.append(f"{stage}: failed {path}")
    if failures:
        raise RuntimeError("formal generation is blocked by upstream gates: " + "; ".join(failures))


def environment_info() -> dict[str, object]:
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }


def _exact_noise(clean: torch.Tensor, fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=clean.device).manual_seed(seed)
    noise = torch.randn(clean.shape, dtype=clean.dtype, device=clean.device, generator=generator)
    noise = noise / noise.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(torch.finfo(clean.dtype).tiny)
    noise = noise * (fraction * clean.square().mean(dim=-1, keepdim=True).sqrt())
    return clean + noise, noise


def _resample(traces: torch.Tensor, nt: int) -> torch.Tensor:
    shape = traces.shape
    return functional.interpolate(traces.reshape(-1, 1, shape[-1]), size=nt, mode="linear", align_corners=True).reshape(*shape[:-1], nt)


def _load_fine_clean_for_noise(config: DemoConfig3D, device: torch.device,
                               dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Load the immutable clean realization from which fine_noisy is derived."""
    source = Path(config.output_root) / "fine_clean" / "data.h5"
    if not source.exists():
        raise RuntimeError(f"fine_noisy requires the frozen fine_clean dataset: {source}")
    names = ("clean_pressure_traces", "static_pressure_traces", "truth_velocity",
             "generation_truth_velocity", "transducer_coordinates", "ordered_trace_mask")
    result: dict[str, torch.Tensor] = {}
    with h5py.File(source, "r") as handle:
        for name in names:
            tensor = torch.as_tensor(np.asarray(handle[name]), device=device)
            result[name] = tensor.to(dtype) if tensor.is_floating_point() else tensor
    return result


def generate_case(config: DemoConfig3D, case: str) -> dict[str, object]:
    if case not in {"matched_clean", "fine_clean", "fine_noisy"}:
        raise ValueError("case must be matched_clean, fine_clean, or fine_noisy")
    require_generation_gates(config)
    torch.manual_seed(config.seed); np.random.seed(config.seed)
    device, dtype = config.torch_device(), config.torch_dtype()
    generation_grid = config.grid if case == "matched_clean" else fine_grid(config)
    acq_config = config.acquisition
    if case != "matched_clean":
        ratio = config.grid.dx / generation_grid.dx
        acq_config = replace(acq_config, wall_offset_cells=acq_config.wall_offset_cells * ratio)
    acquisition = build_acquisition_3d(generation_grid, acq_config, dtype=dtype, device=device)
    derived_from = None
    if case == "fine_noisy":
        frozen = _load_fine_clean_for_noise(config, device, dtype)
        clean = frozen["clean_pressure_traces"]
        static = frozen["static_pressure_traces"]
        inverse_velocity = frozen["truth_velocity"]
        velocity = frozen["generation_truth_velocity"]
        coordinates = frozen["transducer_coordinates"]
        ordered_mask = frozen["ordered_trace_mask"]
        traces, noise = _exact_noise(clean, config.noise_rms_fraction, config.seed)
        derived_from = str(Path(config.output_root) / "fine_clean" / "data.h5")
    else:
        # Dataset truth is the converged MAC/RK2/projection result, not the
        # Fourier profile used only as the flow verification reference.
        flow = simulate_flow(FlowConfig(generation_grid, config.physics.max_velocity,
            config.physics.effective_reynolds, config.physics.rho0, dtype, device, "duct"))
        velocity = flow.cell_velocity
        if generation_grid == config.grid:
            inverse_velocity = velocity
        else:
            inverse_flow = simulate_flow(FlowConfig(config.grid, config.physics.max_velocity,
                config.physics.effective_reynolds, config.physics.rho0, dtype, device, "duct"))
            inverse_velocity = inverse_flow.cell_velocity
        forward = ForwardConfig3D(generation_grid, config.physics.c0, config.physics.include_shear, "sponge",
                                  sponge_layers_for_grid(config, generation_grid), config.inversion.checkpoint_steps,
                                  acquisition.source_signal, sponge_mode=acq_config.sponge_mode)
        with torch.no_grad():
            clean = simulate_acoustic(velocity, acquisition.source_apertures, acquisition.apertures, forward)
            static = simulate_acoustic(torch.zeros_like(velocity), acquisition.source_apertures, acquisition.apertures, forward)
        if generation_grid.nt != config.grid.nt:
            clean, static = _resample(clean, config.grid.nt), _resample(static, config.grid.nt)
        traces, noise = clean, torch.zeros_like(clean)
        coordinates = acquisition.coordinates
        ordered_mask = acquisition.ordered_mask
    directory = case_directory(config, case)
    with (directory / "config_snapshot.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config.as_dict(), stream, sort_keys=False)
    with (directory / "environment.json").open("w", encoding="utf-8") as stream:
        json.dump(environment_info(), stream, indent=2)
    with h5py.File(directory / "data.h5", "w") as handle:
        for name, value in {"pressure_traces": traces, "clean_pressure_traces": clean, "noise": noise,
                            "static_pressure_traces": static, "truth_velocity": inverse_velocity,
                            "generation_truth_velocity": velocity,
                            "transducer_coordinates": coordinates,
                            "ordered_trace_mask": ordered_mask}.items():
            array = value.detach().cpu().numpy()
            handle.create_dataset(name, data=array, compression="gzip" if array.ndim >= 2 else None)
        handle.attrs.update(case=case, seed=config.seed, dt=config.grid.dt, trace_count=acquisition.trace_count,
                            truth_origin="converged MAC RK2 projection solver",
                            source_normalization="nodal trapezoidal mass adjoint B=M^-1 C^T",
                            sponge_mode=acq_config.sponge_mode,
                            sponge_width_m=config.acquisition.sponge_layers * config.grid.dz,
                            transducer_model="trilinear physical point sample and mass-adjoint point injection",
                            derived_from="" if derived_from is None else derived_from)
    metadata = {"case": case, "data_path": str(directory / "data.h5"), "ordered_traces": acquisition.trace_count,
                "noise_rms_fraction": config.noise_rms_fraction if case == "fine_noisy" else 0.0,
                "generation_grid": [generation_grid.nx, generation_grid.ny, generation_grid.nz, generation_grid.nt],
                "derived_from": derived_from}
    if case == "matched_clean":
        from .dataset_validation import validate_matched_clean
        metadata["validation"] = validate_matched_clean(config, directory / "data.h5")
    with (directory / "generation_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    return metadata


def load_case(config: DemoConfig3D, case: str) -> dict[str, torch.Tensor]:
    path = case_directory(config, case) / "data.h5"
    if not path.exists():
        raise FileNotFoundError(f"generate case first: {path}")
    result = {}
    with h5py.File(path, "r") as handle:
        for name in handle:
            tensor = torch.as_tensor(np.asarray(handle[name]), device=config.torch_device())
            result[name] = tensor.to(config.torch_dtype()) if tensor.is_floating_point() else tensor
    return result
