from pathlib import Path

import torch

from water_usct_3d.acquisition import build_acquisition_3d, quadrature_mass
from water_usct_3d.config import fine_grid, load_config


CONFIG = Path(__file__).parents[1] / "examples/01_square_duct/config.yaml"


def test_fixed_grid_and_trace_contract():
    config = load_config(CONFIG)
    assert (config.grid.nx, config.grid.ny, config.grid.nz, config.grid.nt) == (33, 33, 65, 800)
    assert abs(config.grid.dx - 0.003125) < 1e-15
    acquisition = build_acquisition_3d(config.grid, config.acquisition, dtype=torch.float64)
    assert acquisition.coordinates.shape == (24, 3)
    xy = acquisition.coordinates[:, :2]
    offset = config.acquisition.wall_offset_cells * config.grid.dx
    assert bool(((xy >= offset - 1e-14) & (xy <= config.grid.length_x - offset + 1e-14)).all())
    assert acquisition.trace_count == 552
    assert acquisition.reciprocal_pairs.shape == (276, 2)
    assert config.acquisition.transducer_model == "point"
    assert torch.allclose(acquisition.apertures.sum((1, 2, 3)), torch.ones(24, dtype=torch.float64))
    assert int((acquisition.apertures > 0).sum(dim=(1, 2, 3)).max()) <= 8
    x, y, z = config.grid.coordinates(dtype=torch.float64, device=torch.device("cpu"))
    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
    reconstructed = torch.stack([
        (acquisition.apertures * xx).sum((1, 2, 3)),
        (acquisition.apertures * yy).sum((1, 2, 3)),
        (acquisition.apertures * zz).sum((1, 2, 3)),
    ], dim=1)
    assert torch.allclose(reconstructed, acquisition.coordinates, atol=1e-14, rtol=0.0)
    fine = fine_grid(config)
    fine_config = config.acquisition.__class__(**{
        **config.acquisition.__dict__,
        "wall_offset_cells": config.acquisition.wall_offset_cells * config.grid.dx / fine.dx,
    })
    fine_acquisition = build_acquisition_3d(fine, fine_config, dtype=torch.float64)
    common_spacing = 0.0125
    assert torch.allclose(acquisition.coordinates / common_spacing,
                          torch.round(acquisition.coordinates / common_spacing), atol=1e-12, rtol=0.0)
    assert torch.allclose(fine_acquisition.coordinates, acquisition.coordinates, atol=1e-14, rtol=0.0)
    assert torch.allclose(acquisition.apertures.amax((1, 2, 3)), torch.ones(24, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(fine_acquisition.apertures.amax((1, 2, 3)), torch.ones(24, dtype=torch.float64), atol=1e-12)
    mass = quadrature_mass(config.grid, torch.float64, torch.device("cpu"))
    assert torch.allclose((acquisition.source_apertures * mass).sum((1, 2, 3)), torch.ones(24, dtype=torch.float64))
