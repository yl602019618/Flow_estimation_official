"""Small parser for the scalar/list subset of AMReX ParmParse inputs."""

from __future__ import annotations

from pathlib import Path


def parse_inputs(path: str | Path) -> dict[str, list[str]]:
    """Return last-assignment-wins tokens keyed by ParmParse name."""
    values: dict[str, list[str]] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"Malformed input line in {path}: {raw_line!r}")
        key, raw_value = line.split("=", 1)
        key = key.strip()
        tokens = raw_value.strip().replace('"', "").split()
        if not key or not tokens:
            raise ValueError(f"Empty key/value in {path}: {raw_line!r}")
        values[key] = tokens
    return values


def scalar(values: dict[str, list[str]], key: str) -> str:
    tokens = values[key]
    if len(tokens) != 1:
        raise ValueError(f"{key} should be scalar, got {tokens}")
    return tokens[0]


def floats(values: dict[str, list[str]], key: str) -> tuple[float, ...]:
    return tuple(float(item) for item in values[key])


def ints(values: dict[str, list[str]], key: str) -> tuple[int, ...]:
    return tuple(int(item) for item in values[key])

