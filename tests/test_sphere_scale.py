from pathlib import Path

import pytest

from water_usct_3d.sphere.config import load_sphere_config


ROOT = Path(__file__).parents[1]


def test_50cm_case_preserves_acoustic_similarity():
    small = load_sphere_config(ROOT / "examples" / "02_ball_10cm" / "config.yaml")
    large = load_sphere_config(ROOT / "examples" / "03_ball_50cm" / "config.yaml")
    scale = large.radius/small.radius
    assert scale == 5
    assert large.section("signal")["center_frequency_hz"] == \
           small.section("signal")["center_frequency_hz"]/scale
    for key in ("domain_half_extent_m", "absorber_width_m", "record_time_s", "dt_s"):
        assert large.section("full_wave")[key] == pytest.approx(
            scale*small.section("full_wave")[key])
    small_size = 2*small.radius*small.section("signal")["center_frequency_hz"]/small.c0
    large_size = 2*large.radius*large.section("signal")["center_frequency_hz"]/large.c0
    assert large_size == pytest.approx(small_size)
    assert round((small.section("full_wave")["production_grid_n"]-1)*scale)+1 == 481
