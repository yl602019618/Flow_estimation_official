from dataclasses import replace

import torch

from water_usct_3d.acoustic import ForwardConfig3D
from water_usct_3d.acquisition import build_acquisition_3d
from water_usct_3d.config import AcquisitionConfig3D, Grid3D
from water_usct_3d.verification import _taylor_test


def test_discrete_acoustic_taylor_remainder():
    grid = Grid3D(9, 9, 11, .1, .1, .2, 3e-7, 8*3e-7)
    config = AcquisitionConfig3D((.05,.1,.15), ((1/3,2/3),(.25,.75),(1/3,2/3)), wall_offset_cells=.5)
    acquisition = build_acquisition_3d(grid, config, dtype=torch.float64)
    signal = torch.zeros(grid.nt, dtype=torch.float64); signal[0] = 1.0
    forward = ForwardConfig3D(grid, source_signal=signal, boundary="rigid", sponge_layers=0, checkpoint_steps=4)
    slope, error = _taylor_test(grid, acquisition, forward, 20260807)
    assert slope > 1.8
    assert error < 1e-5
