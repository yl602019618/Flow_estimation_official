from pathlib import Path

import torch

from water_usct_3d.acoustic import ForwardConfig3D
from water_usct_3d.acquisition import build_acquisition_3d
from water_usct_3d.config import load_config
from water_usct_3d.inversion import ObjectiveConfig3D, invert_amplitude


def test_amplitude_template_reaches_same_target_from_registered_starts():
    config = load_config(Path(__file__).parents[1] / "examples/01_square_duct/config.yaml")
    acquisition = build_acquisition_3d(config.grid, config.acquisition, dtype=torch.float64)
    forward = ForwardConfig3D(config.grid, source_signal=acquisition.source_signal)
    objective = ObjectiveConfig3D(config, acquisition, forward)
    torch.manual_seed(12)
    static = 0.1 * torch.randn((24, 24, config.grid.nt), dtype=torch.float64)
    template = 0.01 * torch.randn_like(static)
    unit = static + template
    truth = 0.997
    observed = static + truth * template
    for start in (0.0, 0.8, 1.2):
        model, history = invert_amplitude(
            observed, objective, start, static_observed=static, unit_prediction=unit)
        assert abs(float(model.beta) - truth) < 1e-10
        assert len(history) == 4 * config.inversion.development_iterations
        assert max(item["max_velocity_update"] for item in history) <= 0.05 + 1e-12
