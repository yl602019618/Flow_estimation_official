from __future__ import annotations

import numpy as np
import h5py

from incflo_multiring_tt_study.multiring import (
    MultiRingDesign,
    global_acquisition,
    local_ring_z,
)
from multislice_xyz_tt_study.multislice_tt import MultiSliceConfig, _load_interpolators
from multislice_xyz_tt_study.multislice_tt import CurlBasis3D, _envelope_gradient


def test_global_array_preserves_prior_density_and_uses_only_four_walls():
    design = MultiRingDesign()
    coordinates, walls, rings = global_acquisition(design)
    assert len(design.ring_z_m) == 41
    assert np.allclose(np.diff(design.ring_z_m), 0.15)
    assert coordinates.shape == (41 * 48, 3)
    assert set(walls) == {"x0", "x1", "y0", "y1"}
    assert len(np.unique(rings)) == 41
    assert np.all((coordinates[:, 2] > 0.0) & (coordinates[:, 2] < 8.0))


def test_sensor_density_levels_keep_ring_count_and_wall_balance():
    for sensors_per_ring in (12, 24, 36, 48):
        design = MultiRingDesign(sensors_per_ring=sensors_per_ring)
        coordinates, walls, rings = global_acquisition(design)
        assert coordinates.shape == (41 * sensors_per_ring, 3)
        assert len(np.unique(rings)) == 41
        for ring in range(41):
            ring_walls = np.asarray(walls)[rings == ring]
            assert all(np.sum(ring_walls == wall) == sensors_per_ring // 4
                       for wall in ("x0", "x1", "y0", "y1"))


def test_local_window_has_symmetric_guards_and_five_rings():
    rings = np.asarray(local_ring_z(1.65))
    assert len(rings) == 5
    assert np.allclose(rings, (1.35, 1.50, 1.65, 1.80, 1.95))
    assert np.isclose(rings[2], 1.65)


def test_fixed_coverage_ring_count_levels_have_registered_spacings():
    for count, spacing in ((15, 6 / 14), (20, 6 / 19), (25, 0.25)):
        design = MultiRingDesign(
            z_min_m=1.05, z_max_m=7.05, ring_spacing_m=spacing,
            sensors_per_ring=36,
        )
        assert len(design.ring_z_m) == count
        assert np.isclose(design.ring_z_m[0], 1.05)
        assert np.isclose(design.ring_z_m[-1], 7.05)


def test_incflo_vector_attributes_are_json_compatible(tmp_path):
    snapshot = tmp_path / "snapshot.h5"
    baseline = tmp_path / "baseline.h5"
    for path, value in ((snapshot, 1.0), (baseline, 0.0)):
        with h5py.File(path, "w") as handle:
            handle.create_dataset("velocity", data=np.full((3, 3, 3, 4), value))
            handle.attrs["domain_right_m"] = (1.5, 1.5, 8.0)
    config = MultiSliceConfig(
        snapshot=str(snapshot), baseline=str(baseline), flow_length_z_m=8.0,
    )
    _, _, metadata = _load_interpolators(config)
    assert metadata["snapshot_attributes"]["domain_right_m"] == [1.5, 1.5, 8.0]


def test_local_curl_basis_is_translation_invariant_in_z():
    centers = np.asarray([[0.5, 0.7, 1.5], [0.9, 0.4, 1.8]])
    basis = CurlBasis3D(centers, np.ones(6), 0.11, 0.13)
    points = np.asarray([[0.3, 0.8, 1.6], [1.1, 0.2, 1.9]])
    shift = 2.1
    shifted = CurlBasis3D(
        centers + np.asarray((0.0, 0.0, shift)), np.ones(6), 0.11, 0.13,
    )
    assert np.allclose(
        _envelope_gradient(points, basis, 1.5),
        _envelope_gradient(points + np.asarray((0.0, 0.0, shift)), shifted, 1.5),
    )
