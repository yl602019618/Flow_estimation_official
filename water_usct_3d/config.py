from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml


@dataclass(frozen=True)
class Grid3D:
    nx: int
    ny: int
    nz: int
    length_x: float
    length_y: float
    length_z: float
    dt: float
    record_time: float
    spatial_order: int = 8

    @property
    def dx(self) -> float:
        return self.length_x / (self.nx - 1)

    @property
    def dy(self) -> float:
        return self.length_y / (self.ny - 1)

    @property
    def dz(self) -> float:
        return self.length_z / (self.nz - 1)

    @property
    def nt(self) -> int:
        return int(round(self.record_time / self.dt))

    @property
    def cell_volume(self) -> float:
        return self.dx * self.dy * self.dz

    def coordinates(self, *, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, ...]:
        return (
            torch.linspace(0.0, self.length_x, self.nx, dtype=dtype, device=device),
            torch.linspace(0.0, self.length_y, self.ny, dtype=dtype, device=device),
            torch.linspace(0.0, self.length_z, self.nz, dtype=dtype, device=device),
        )


@dataclass(frozen=True)
class Physics3D:
    rho0: float = 998.0
    c0: float = 1480.0
    viscosity: float = 1.0e-3
    max_velocity: float = 1.0
    effective_reynolds: float = 100.0
    include_shear: bool = True


@dataclass(frozen=True)
class AcquisitionConfig3D:
    ring_z: tuple[float, ...]
    ring_positions: tuple[tuple[float, float], ...]
    ring_perimeter_offsets: tuple[float, ...] = (0.0, 0.0, 0.0)
    wall_offset_cells: float = 2.0
    transducer_model: str = "point"
    center_frequency: float = 50_000.0
    source_delay_cycles: float = 2.0
    source_amplitude: float = 1.0
    sponge_layers: int = 12
    sponge_mode: str = "pressure"


@dataclass(frozen=True)
class InversionConfig3D:
    profile_control: tuple[int, int] = (6, 6)
    vector_control: tuple[int, int, int] = (6, 6, 8)
    lowpass_frequencies: tuple[float, ...] = (20_000.0, 35_000.0, 50_000.0, 65_000.0)
    regularization_candidates: tuple[float, ...] = (1e-8, 1e-7, 1e-6, 1e-5)
    curvature_weight: float = 1e-7
    gauge_weight: float = 1e-7
    checkpoint_steps: int = 32
    development_iterations: int = 5
    production_iterations: int = 15
    lbfgs_history: int = 10
    normalized_initial_objective: float = 100.0
    initial_gradient_velocity: float = 0.05
    max_update_velocity: float = 0.05
    early_stop_patience: int = 5
    early_stop_relative: float = 1e-4


@dataclass(frozen=True)
class DemoConfig3D:
    seed: int
    output_root: str
    device: str
    dtype: str
    grid: Grid3D
    fine_grid: dict[str, Any]
    physics: Physics3D
    acquisition: AcquisitionConfig3D
    inversion: InversionConfig3D
    noise_rms_fraction: float
    verification: dict[str, float]
    source_path: str

    def torch_dtype(self, verification: bool = False) -> torch.dtype:
        if verification:
            return torch.float64
        return {"float32": torch.float32, "float64": torch.float64}[self.dtype]

    def torch_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("source_path", None)
        return value


def _tuple(data: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    result = dict(data)
    for name in names:
        if name in result:
            result[name] = tuple(result[name])
    return result


def load_config(path: str | Path) -> DemoConfig3D:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    acquisition = _tuple(raw["acquisition"], ("ring_z", "ring_positions", "ring_perimeter_offsets"))
    acquisition["ring_positions"] = tuple(tuple(v) for v in acquisition["ring_positions"])
    inversion = _tuple(
        raw["inversion"],
        ("profile_control", "vector_control", "lowpass_frequencies",
         "regularization_candidates"),
    )
    return DemoConfig3D(
        seed=int(raw["seed"]),
        output_root=str(raw["output_root"]),
        device=str(raw.get("device", "auto")),
        dtype=str(raw.get("dtype", "float32")),
        grid=Grid3D(**raw["grid"]),
        fine_grid=dict(raw["fine_grid"]),
        physics=Physics3D(**raw["physics"]),
        acquisition=AcquisitionConfig3D(**acquisition),
        inversion=InversionConfig3D(**inversion),
        noise_rms_fraction=float(raw.get("noise_rms_fraction", 0.01)),
        verification={str(k): float(v) for k, v in raw["verification"].items()},
        source_path=str(source),
    )


def fine_grid(config: DemoConfig3D) -> Grid3D:
    return Grid3D(
        nx=int(config.fine_grid["nx"]), ny=int(config.fine_grid["ny"]), nz=int(config.fine_grid["nz"]),
        length_x=config.grid.length_x, length_y=config.grid.length_y, length_z=config.grid.length_z,
        dt=float(config.fine_grid["dt"]), record_time=config.grid.record_time,
        spatial_order=config.grid.spatial_order,
    )


def sponge_layers_for_grid(config: DemoConfig3D, grid: Grid3D) -> int:
    """Preserve the production sponge's physical thickness across grids."""
    width = config.acquisition.sponge_layers * config.grid.dz
    return max(1, int(round(width / grid.dz)))


def smoke_grid(config: DemoConfig3D, n: int = 9, nt: int = 12) -> Grid3D:
    """Small grid used only by smoke verification; never by production claims."""
    n = max(n, config.grid.spatial_order + 1)
    return Grid3D(n, n, n + 2, config.grid.length_x, config.grid.length_y,
                  config.grid.length_z, config.grid.dt, nt * config.grid.dt,
                  config.grid.spatial_order)
