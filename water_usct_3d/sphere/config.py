"""Configuration types for the unit-ball travel-time benchmark."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SphereConfig:
    raw: dict[str, Any]
    source_path: str

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def output_root(self) -> Path:
        return Path(self.raw["output_root"])

    def section(self, name: str) -> dict[str, Any]:
        return self.raw[name]

    @property
    def radius(self) -> float:
        return float(self.raw["physics"]["radius_m"])

    @property
    def c0(self) -> float:
        return float(self.raw["physics"]["c0_m_s"])


def load_sphere_config(path: str | Path) -> SphereConfig:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    return SphereConfig(raw=raw, source_path=str(source))


def coerce_config(config: SphereConfig | str | Path | dict[str, Any]) -> SphereConfig:
    if isinstance(config, SphereConfig):
        return config
    if isinstance(config, (str, Path)):
        return load_sphere_config(config)
    return SphereConfig(raw=config, source_path="<mapping>")

