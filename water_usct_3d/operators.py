from __future__ import annotations

from collections.abc import Sequence

import torch


FIRST_DERIVATIVE_8 = {
    -4: 1.0 / 280.0, -3: -4.0 / 105.0, -2: 1.0 / 5.0, -1: -4.0 / 5.0,
    1: 4.0 / 5.0, 2: -1.0 / 5.0, 3: 4.0 / 105.0, 4: -1.0 / 280.0,
}

# Eighth-order derivative at i+1/2 from samples on integer nodes.  The
# coefficients multiply f[i+r] - f[i+1-r], r=1,...,4.
STAGGERED_DERIVATIVE_8 = (
    1225.0 / 1024.0,
    -245.0 / 3072.0,
    49.0 / 5120.0,
    -5.0 / 7168.0,
)


def _axis_index(n: int, offset: int, device: torch.device, periodic: bool, odd: bool) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.arange(n, device=device) + offset
    if periodic:
        return base.remainder(n), torch.ones(n, device=device)
    low, high = base < 0, base >= n
    index = torch.where(low, -base, torch.where(high, 2 * (n - 1) - base, base))
    sign = torch.ones(n, device=device)
    if odd:
        sign = torch.where(low | high, -sign, sign)
    return index, sign


def derivative8(field: torch.Tensor, axis: int, spacing: float, *, periodic: bool = False, odd: bool = False) -> torch.Tensor:
    """Eighth-order centered derivative with differentiable parity ghosts."""
    axis %= field.ndim
    if field.shape[axis] < 9:
        raise ValueError("eighth-order derivative needs at least 9 points")
    result = torch.zeros_like(field)
    shape = [1] * field.ndim
    shape[axis] = field.shape[axis]
    for offset, coefficient in FIRST_DERIVATIVE_8.items():
        index, sign = _axis_index(field.shape[axis], offset, field.device, periodic, odd)
        result = result + coefficient * field.index_select(axis, index) * sign.reshape(shape)
    return result / spacing


def staggered_gradient8(field: torch.Tensor, axis: int, spacing: float, *, periodic: bool = False) -> torch.Tensor:
    """Map node samples to half-grid faces with an eighth-order stencil.

    For a non-periodic interval, pressure is extended evenly through the two
    rigid walls.  Slots ``0 .. n-2`` represent the physical interior faces and
    slot ``n-1`` is an explicitly zero padding slot.  Keeping the padded shape
    lets the acoustic state remain a single tensor without pretending that a
    normal-velocity degree of freedom exists on a rigid wall.
    """
    axis %= field.ndim
    n = field.shape[axis]
    if n < 9:
        raise ValueError("eighth-order staggered derivative needs at least 9 points")
    result = torch.zeros_like(field)
    for radius, coefficient in enumerate(STAGGERED_DERIVATIVE_8, start=1):
        right, _ = _axis_index(n, radius, field.device, periodic, False)
        left, _ = _axis_index(n, 1 - radius, field.device, periodic, False)
        result = result + coefficient * (
            field.index_select(axis, right) - field.index_select(axis, left)
        )
    result = result / spacing
    if not periodic:
        mask = torch.ones(n, dtype=field.dtype, device=field.device)
        mask[-1] = 0.0
        shape = [1] * field.ndim
        shape[axis] = n
        result = result * mask.reshape(shape)
    return result


def staggered_divergence8(face_field: torch.Tensor, axis: int, spacing: float, *, periodic: bool = False) -> torch.Tensor:
    """Negative quadrature-mass adjoint of :func:`staggered_gradient8`.

    Interior face volumes are uniform while boundary pressure nodes carry the
    trapezoidal half weight.  This identity is enforced algebraically, including
    the reflected rigid-wall closure, so ``<Gp,w>_face + <p,Dw>_node`` is
    roundoff zero and the point injection ``M^-1 C^T`` is reciprocal.
    """
    axis %= face_field.ndim
    n = face_field.shape[axis]
    if n < 9:
        raise ValueError("eighth-order staggered derivative needs at least 9 points")
    active = face_field
    if not periodic:
        mask = torch.ones(n, dtype=face_field.dtype, device=face_field.device)
        mask[-1] = 0.0
        shape = [1] * face_field.ndim
        shape[axis] = n
        active = face_field * mask.reshape(shape)
    result = torch.zeros_like(face_field)
    for radius, coefficient in enumerate(STAGGERED_DERIVATIVE_8, start=1):
        right, _ = _axis_index(n, radius, face_field.device, periodic, False)
        left, _ = _axis_index(n, 1 - radius, face_field.device, periodic, False)
        result = result.index_add(axis, right, -coefficient * active)
        result = result.index_add(axis, left, coefficient * active)
    result = result / spacing
    if not periodic:
        inverse_node_weights = torch.ones(n, dtype=face_field.dtype, device=face_field.device)
        inverse_node_weights[[0, -1]] = 2.0
        shape = [1] * face_field.ndim
        shape[axis] = n
        result = result * inverse_node_weights.reshape(shape)
    return result


def derivative2(field: torch.Tensor, axis: int, spacing: float, *, periodic: bool = False) -> torch.Tensor:
    axis %= field.ndim
    n = field.shape[axis]
    left, _ = _axis_index(n, -1, field.device, periodic, False)
    right, _ = _axis_index(n, 1, field.device, periodic, False)
    return (field.index_select(axis, right) - field.index_select(axis, left)) / (2.0 * spacing)


def gradient8(field: torch.Tensor, spacing: Sequence[float], periodic: Sequence[bool] = (False, False, False)) -> torch.Tensor:
    axes = tuple(range(field.ndim - 3, field.ndim))
    return torch.stack([derivative8(field, a, h, periodic=p) for a, h, p in zip(axes, spacing, periodic)], dim=0)


def divergence(velocity: torch.Tensor, spacing: Sequence[float], *, order: int = 2) -> torch.Tensor:
    if velocity.shape[0] != 3:
        raise ValueError("velocity shape must start with three components")
    derivative = derivative8 if order == 8 else derivative2
    return sum(derivative(velocity[i], i, spacing[i], periodic=(i == 2)) for i in range(3))


def curl(vector_potential: torch.Tensor, spacing: Sequence[float], *, order: int = 2) -> torch.Tensor:
    derivative = derivative8 if order == 8 else derivative2
    ax, ay, az = vector_potential
    dx, dy, dz = spacing
    return torch.stack((
        derivative(az, 1, dy) - derivative(ay, 2, dz, periodic=True),
        derivative(ax, 2, dz, periodic=True) - derivative(az, 0, dx),
        derivative(ay, 0, dx) - derivative(ax, 1, dy),
    ))


def laplacian2(field: torch.Tensor, spacing: Sequence[float], periodic: Sequence[bool] = (False, False, True)) -> torch.Tensor:
    result = torch.zeros_like(field)
    axes = tuple(range(field.ndim - 3, field.ndim))
    for axis, h, is_periodic in zip(axes, spacing, periodic):
        n = field.shape[axis]
        left, _ = _axis_index(n, -1, field.device, is_periodic, False)
        right, _ = _axis_index(n, 1, field.device, is_periodic, False)
        result = result + (field.index_select(axis, right) - 2.0 * field + field.index_select(axis, left)) / h**2
    return result


def relative_norm(value: torch.Tensor, reference: torch.Tensor) -> float:
    tiny = torch.finfo(value.dtype).tiny
    return float(torch.linalg.vector_norm(value).div(torch.linalg.vector_norm(reference).clamp_min(tiny)).detach().cpu())
