"""Public API for the independent three-dimensional Water-USCT demo."""

from .acquisition import Acquisition3D, build_acquisition_3d
from .acoustic import ForwardConfig3D, simulate_acoustic
from .flow import FlowConfig, FlowResult, simulate_flow
from .inversion import waveform_objective_3d
from .parameterization import vector_potential_to_velocity
from .travel_time import extract_reciprocal_delays_3d, invert_travel_time_3d

__all__ = [
    "Acquisition3D", "FlowConfig", "FlowResult", "ForwardConfig3D",
    "build_acquisition_3d", "extract_reciprocal_delays_3d",
    "invert_travel_time_3d", "simulate_acoustic", "simulate_flow",
    "vector_potential_to_velocity", "waveform_objective_3d",
]
