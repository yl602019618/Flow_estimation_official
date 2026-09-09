import numpy as np

from multislice_xyz_tt_study.multislice_tt import (
    MultiSliceConfig,
    _envelope_gradient,
    acquisition,
    build_basis,
    evaluate_coefficients,
    select_pairs,
    solve_gcv,
)


def test_multilayer_acquisition_uses_only_four_walls_and_has_cross_layer_pairs():
    config = MultiSliceConfig(sensor_z_m=(2.35, 2.50, 2.65), sensors_per_layer=16,
                              center_shape=(3, 3, 3), norm_shape=(9, 9, 5))
    coordinates, walls, layers = acquisition(config)
    pairs = select_pairs(walls, layers, "full")
    assert coordinates.shape == (48, 3)
    assert set(walls) == {"x0", "x1", "y0", "y1"}
    assert np.any(layers[pairs[:, 0]] != layers[pairs[:, 1]])
    assert all(walls[a] != walls[b] for a, b in pairs)


def test_localized_curl_field_is_divergence_free_to_discretization_accuracy():
    config = MultiSliceConfig(
        center_shape=(3, 3, 3), norm_shape=(11, 11, 7),
        evaluation_shape=(31, 31, 15), sensor_z_m=(2.35, 2.5, 2.65),
    )
    basis = build_basis(config)
    axes = (
        np.linspace(0, 1.5, config.evaluation_shape[0]),
        np.linspace(0, 1.5, config.evaluation_shape[1]),
        np.linspace(config.slab_z_min_m, config.slab_z_max_m, config.evaluation_shape[2]),
    )
    rng = np.random.default_rng(7)
    coefficients = rng.normal(size=basis.mode_count)
    field = evaluate_coefficients(coefficients, basis, axes, config)
    divergence = sum(
        np.gradient(field[component], axes[component], axis=component, edge_order=2)
        for component in range(3)
    )
    relative = np.linalg.norm(divergence) / (
        np.linalg.norm(field) / min(np.mean(np.diff(axis)) for axis in axes)
    )
    assert relative < 0.03
    assert np.max(np.abs(field[:, (0, -1), :, :])) < 1e-10
    assert np.max(np.abs(field[:, :, (0, -1), :])) < 1e-10


def test_axial_sine_modes_preserve_divergence_and_side_wall_conditions():
    config = MultiSliceConfig(
        center_shape=(2, 2, 2), norm_shape=(9, 9, 5),
        evaluation_shape=(25, 25, 9), sensor_z_m=(2.35, 2.5, 2.65),
        axial_sine_order=3,
    )
    basis = build_basis(config)
    axes = (
        np.linspace(0, 1.5, config.evaluation_shape[0]),
        np.linspace(0, 1.5, config.evaluation_shape[1]),
        np.linspace(config.slab_z_min_m, config.slab_z_max_m, config.evaluation_shape[2]),
    )
    coefficients = np.zeros(basis.mode_count)
    coefficients[basis.curl_mode_count:] = np.arange(1, basis.axial_mode_count + 1)
    field = evaluate_coefficients(coefficients, basis, axes, config)
    divergence = np.gradient(field[2], axes[2], axis=2, edge_order=2)
    assert np.max(np.abs(field[:2])) == 0.0
    assert np.max(np.abs(divergence)) < 1e-12
    assert np.max(np.abs(field[:, (0, -1), :, :])) < 1e-12
    assert np.max(np.abs(field[:, :, (0, -1), :])) < 1e-12


def test_gcv_returns_the_selected_spectral_solution():
    matrix = np.asarray([[1.0, 0.2], [0.3, 1.1], [0.5, -0.4]])
    target = np.asarray([0.7, -0.2, 0.4])
    gram = matrix.T @ matrix
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    coefficients, report = solve_gcv(
        gram, values[order], vectors[:, order], matrix, target,
    )
    rank = report["selected_spectral_rank"]
    ridge = report["ridge"]
    selected_vectors = vectors[:, order][:, :rank]
    expected = selected_vectors @ (
        selected_vectors.T @ matrix.T @ target / (values[order][:rank] + ridge**2)
    )
    assert np.allclose(coefficients, expected)


def test_gcv_never_selects_more_spectral_directions_than_operator_rows():
    rng = np.random.default_rng(20260807)
    matrix = rng.normal(size=(5, 9))
    target = rng.normal(size=5)
    gram = matrix.T @ matrix
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    coefficients, report = solve_gcv(
        gram, values[order], vectors[:, order], matrix, target,
    )
    assert coefficients.shape == (9,)
    assert report["selected_spectral_rank"] <= 5
    assert max(
        candidate["spectral_rank"] for candidate in report["candidate_summary"]
    ) <= 5
