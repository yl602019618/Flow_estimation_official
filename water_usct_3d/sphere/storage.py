"""Deterministic benchmark artifact writers."""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from .basis import RayOperator
from .config import SphereConfig
from .forward import TravelTimeDataset
from .geometry import SphereAcquisition
from .pulse import PulseStatic


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def write_snapshots(config: SphereConfig, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "config_snapshot.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config.raw, stream, sort_keys=False)
    write_json(root / "environment.json", {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "seed": config.seed,
    })


def save_dataset(path: Path, dataset: TravelTimeDataset, traces: np.ndarray,
                 static: PulseStatic) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        for name in ("forward_s", "reverse_s", "reciprocal_s", "linearized_s", "static_s"):
            handle.create_dataset(name, data=getattr(dataset, name))
        handle.create_dataset("traces", data=traces, compression="gzip", compression_opts=4,
                              shuffle=True, chunks=(1, min(128, traces.shape[1]), traces.shape[2]))
        handle.attrs["noise_std_s"] = dataset.noise_std_s
        handle.attrs["dt_s"] = static.dt_s
        handle.attrs["center_frequency_hz"] = static.center_frequency_hz
        handle.attrs["source_delay_s"] = static.source_delay_s


def save_operator(path: Path, acquisition: SphereAcquisition, operator: RayOperator) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, points=acquisition.points, pairs=acquisition.pairs,
                        directions=acquisition.directions, lengths=acquisition.lengths,
                        matrix=operator.matrix, centers=operator.basis.centers,
                        transform=operator.basis.transform,
                        gram_eigenvalues=operator.basis.gram_eigenvalues,
                        singular_values=operator.singular_values)

