import torch
from concurrent.futures import ThreadPoolExecutor

from water_usct_3d.acoustic import ForwardConfig3D, simulate_acoustic
from water_usct_3d.acquisition import build_acquisition_3d
from water_usct_3d.axial_fwi import (_dual_gpu_loss_gradient,
                                     _prepare_dual_gpu_batches,
                                     axial_fwi_objective)
from water_usct_3d.config import (AcquisitionConfig3D, DemoConfig3D, Grid3D,
                                  InversionConfig3D, Physics3D)
from water_usct_3d.inversion import ObjectiveConfig3D
from water_usct_3d.profile_inversion import (axial_profile_to_velocity,
                                             build_axial_fwi_initial_controls)
from water_usct_3d.profile_validation import profile_band_metrics, static_calibrated_prediction
from water_usct_3d.travel_time import DelayData3D


def test_axial_profile_is_z_invariant_and_has_exact_zero_walls():
    grid = Grid3D(17, 17, 19, .1, .1, .2, 3e-7, 3e-6)
    control = torch.rand((6, 6), dtype=torch.float64)
    velocity = axial_profile_to_velocity(control, grid)
    assert torch.count_nonzero(velocity[:2]) == 0
    assert torch.allclose(velocity[2, ..., 0], velocity[2, ..., -1])
    assert torch.count_nonzero(velocity[2, 0]) == 0
    assert torch.count_nonzero(velocity[2, -1]) == 0
    assert torch.count_nonzero(velocity[2, :, 0]) == 0
    assert torch.count_nonzero(velocity[2, :, -1]) == 0


def test_registered_axial_fwi_initializations_are_finite_and_truth_independent():
    grid = Grid3D(9, 9, 11, .1, .1, .2, 3e-7, 8 * 3e-7, spatial_order=8)
    acquisition_config = AcquisitionConfig3D(
        (.04, .10, .16), ((.375, .625), (.25, .75), (.375, .625)),
        wall_offset_cells=1.0,
    )
    inversion = InversionConfig3D(profile_control=(6, 6), vector_control=(4, 4, 4))
    demo = DemoConfig3D(20260807, "unused", "cpu", "float64", grid, {}, Physics3D(),
                        acquisition_config, inversion, .01, {}, "unused")
    acquisition = build_acquisition_3d(grid, acquisition_config, dtype=torch.float64)
    forward = ForwardConfig3D(grid, boundary="rigid", sponge_layers=0,
                              checkpoint_steps=4, source_signal=acquisition.source_signal)
    objective = ObjectiveConfig3D(demo, acquisition, forward)
    count = acquisition.reciprocal_pairs.shape[0]
    delay = DelayData3D(torch.linspace(-2e-8, 2e-8, count, dtype=torch.float64),
                        acquisition.reciprocal_pairs, grid.dt,
                        torch.ones(count, dtype=torch.float64))
    controls, diagnostics = build_axial_fwi_initial_controls(delay, objective, beta=.8)
    assert tuple(controls) == ("good_prior", "travel_time", "blurred_travel_time", "zero_flow")
    assert diagnostics["effective_rank"] > 0
    assert torch.count_nonzero(controls["good_prior"]) > 0
    assert torch.count_nonzero(controls["travel_time"]) > 0
    assert torch.count_nonzero(controls["zero_flow"]) == 0
    for control in controls.values():
        assert torch.isfinite(control).all()
        assert torch.count_nonzero(control[0]) == 0
        assert torch.count_nonzero(control[-1]) == 0
        assert torch.count_nonzero(control[:, 0]) == 0
        assert torch.count_nonzero(control[:, -1]) == 0


def test_profile_band_metrics_favor_exact_prediction_over_zero_flow():
    observed = torch.randn((2, 2, 64), dtype=torch.float64)
    predicted = observed.clone()
    static = torch.zeros_like(observed)
    mask = ~torch.eye(2, dtype=torch.bool)
    metrics = profile_band_metrics(predicted, observed, static, mask, 3e-7, (20_000.0, 65_000.0))
    assert all(item["waveform_loss"] == 0.0 for item in metrics)
    assert all(item["zero_flow_loss"] > 0.0 for item in metrics)
    assert all(item["loss_reduction_from_zero"] == 1.0 for item in metrics)


def test_static_calibration_preserves_only_the_modeled_flow_perturbation():
    predicted_static = torch.randn((2, 3, 16), dtype=torch.float64)
    perturbation = torch.randn_like(predicted_static) * 1e-3
    observed_static = torch.randn_like(predicted_static)
    calibrated = static_calibrated_prediction(
        predicted_static + perturbation, predicted_static, observed_static)
    assert torch.allclose(calibrated - observed_static, perturbation)


def test_axial_fwi_control_gradient_passes_directional_taylor_test():
    grid = Grid3D(9, 9, 11, .1, .1, .2, 3e-7, 8 * 3e-7, spatial_order=8)
    acquisition_config = AcquisitionConfig3D(
        (.04, .10, .16), ((.375, .625), (.25, .75), (.375, .625)),
        wall_offset_cells=1.0,
    )
    inversion = InversionConfig3D(profile_control=(4, 4), vector_control=(4, 4, 4),
                                  lowpass_frequencies=(65_000.0,), checkpoint_steps=4)
    demo = DemoConfig3D(20260807, "unused", "cpu", "float64", grid, {}, Physics3D(),
                        acquisition_config, inversion, .01, {}, "unused")
    acquisition = build_acquisition_3d(grid, acquisition_config, dtype=torch.float64)
    forward = ForwardConfig3D(grid, boundary="rigid", sponge_layers=0,
                              checkpoint_steps=4, source_signal=acquisition.source_signal)
    objective = ObjectiveConfig3D(demo, acquisition, forward)
    truth = torch.tensor((.35, .55, .65, .45), dtype=torch.float64)
    indices = torch.tensor((0,), dtype=torch.long)
    with torch.no_grad():
        control = torch.zeros((4, 4), dtype=torch.float64)
        control[1:-1, 1:-1] = truth.reshape(2, 2)
        velocity = axial_profile_to_velocity(control, grid)
        observed = simulate_acoustic(velocity, acquisition.source_apertures,
                                     acquisition.apertures, forward)
        static = simulate_acoustic(torch.zeros_like(velocity), acquisition.source_apertures,
                                   acquisition.apertures, forward)
    base = (truth * .9).requires_grad_(True)
    loss, _ = axial_fwi_objective(base, observed, static, objective, 65_000.0,
                                  source_indices=indices, curvature_weight=0.0)
    gradient = torch.autograd.grad(loss, base)[0]
    direction = torch.tensor((.2, -.3, .4, -.1), dtype=torch.float64)
    directional = (gradient * direction).sum()
    errors = []
    for epsilon in (1e-3, 5e-4, 2.5e-4):
        perturbed, _ = axial_fwi_objective(base.detach() + epsilon * direction,
            observed, static, objective, 65_000.0, source_indices=indices, curvature_weight=0.0)
        errors.append(abs(float(perturbed - loss.detach() - epsilon * directional.detach())))
    slopes = [torch.log2(torch.tensor(errors[i] / errors[i + 1])).item() for i in range(2)]
    assert min(slopes) > 1.8


def test_dual_gpu_axial_gradient_matches_all_shot_gradient():
    if torch.cuda.device_count() < 2:
        return
    grid = Grid3D(9, 9, 11, .1, .1, .2, 3e-7, 6 * 3e-7, spatial_order=8)
    acquisition_config = AcquisitionConfig3D(
        (.04, .10, .16), ((.375, .625), (.25, .75), (.375, .625)),
        wall_offset_cells=1.0,
    )
    inversion = InversionConfig3D(profile_control=(4, 4), vector_control=(4, 4, 4),
                                  lowpass_frequencies=(65_000.0,), checkpoint_steps=3)
    demo = DemoConfig3D(20260807, "unused", "cuda:0", "float64", grid, {}, Physics3D(),
                        acquisition_config, inversion, .01, {}, "unused")
    acquisition = build_acquisition_3d(grid, acquisition_config, dtype=torch.float64,
                                       device="cuda:0")
    forward = ForwardConfig3D(grid, boundary="rigid", sponge_layers=0,
                              checkpoint_steps=3, source_signal=acquisition.source_signal)
    objective = ObjectiveConfig3D(demo, acquisition, forward)
    truth = torch.tensor((.35, .55, .65, .45), dtype=torch.float64, device="cuda:0")
    with torch.no_grad():
        control = torch.zeros((4, 4), dtype=torch.float64, device="cuda:0")
        control[1:-1, 1:-1] = truth.reshape(2, 2)
        velocity = axial_profile_to_velocity(control, grid)
        observed = simulate_acoustic(velocity, acquisition.source_apertures,
                                     acquisition.apertures, forward)
        static = simulate_acoustic(torch.zeros_like(velocity), acquisition.source_apertures,
                                   acquisition.apertures, forward)
    base = (truth * .9).requires_grad_(True)
    full_loss, _ = axial_fwi_objective(base, observed, static, objective, 65_000.0)
    full_gradient = torch.autograd.grad(full_loss, base)[0]
    batches = _prepare_dual_gpu_batches(observed, static, objective)
    with ThreadPoolExecutor(2) as executor:
        dual_loss, dual_gradient, _, _ = _dual_gpu_loss_gradient(
            base.detach(), batches, 65_000.0, executor, inversion.curvature_weight)
    assert torch.allclose(dual_loss, full_loss.detach(), rtol=1e-9, atol=1e-12)
    assert torch.allclose(dual_gradient, full_gradient, rtol=1e-8, atol=1e-11)
