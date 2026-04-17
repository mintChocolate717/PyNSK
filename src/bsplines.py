"""
B-spline basis functions for IGA spatial discretization.

Basis matrices are returned as jax.numpy arrays (float64, no gradient).
Control point arrays are then multiplied by these constant matrices
during residual assembly — JAX differentiates through the physics only,
not through basis evaluation.

Convention: open-uniform knot vectors, parametric domain [0, 1].
"""

import jax.numpy as jnp
import numpy as np
from scipy.interpolate import BSpline


def make_knot_vector(n_ctrl: int, degree: int) -> jnp.ndarray:
    """Open-uniform knot vector: (degree+1) repeated knots at each end.

    Args:
        n_ctrl: number of B-spline control points (n)
        degree: polynomial degree p (must satisfy n >= p + 1)

    Returns:
        knots: shape (n_ctrl + degree + 1,)
    """
    if n_ctrl < degree + 1:
        raise ValueError(f"Require n_ctrl >= degree + 1, got n_ctrl={n_ctrl}, degree={degree}")
    n_interior = n_ctrl - degree - 1
    interior = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
    knots = np.concatenate([np.zeros(degree + 1), interior, np.ones(degree + 1)])
    return jnp.array(knots, dtype=jnp.float64)


def _identity_spline(knots: np.ndarray, degree: int) -> BSpline:
    """scipy BSpline with identity coefficient matrix.

    Evaluating at x of shape (n_pts,) returns shape (n_pts, n_ctrl)
    where column A is basis function B_A evaluated at each point.
    """
    n_ctrl = len(knots) - degree - 1
    return BSpline(knots, np.eye(n_ctrl), degree)


def basis_matrix(xi_pts: jnp.ndarray, knots: jnp.ndarray, degree: int) -> jnp.ndarray:
    """B-spline basis matrix evaluated at quadrature points.

    Args:
        xi_pts: shape (n_pts,) — parametric coordinates in [0, 1]
        knots:  shape (n_ctrl + degree + 1,)
        degree: polynomial degree p

    Returns:
        N: shape (n_pts, n_ctrl), N[q, A] = B_A(xi_q)
    """
    t = np.asarray(knots)
    x = np.asarray(xi_pts)
    vals = _identity_spline(t, degree)(x)
    return jnp.array(vals, dtype=jnp.float64)


def basis_deriv_matrix(
    xi_pts: jnp.ndarray, knots: jnp.ndarray, degree: int, order: int
) -> jnp.ndarray:
    """nth-order derivative of the B-spline basis matrix.

    Args:
        xi_pts: shape (n_pts,)
        knots:  shape (n_ctrl + degree + 1,)
        degree: polynomial degree p
        order:  derivative order (1 → dN/dxi, 2 → d²N/dxi²)

    Returns:
        dN: shape (n_pts, n_ctrl), dN[q, A] = d^order B_A / dxi^order at xi_q
    """
    if order == 0:
        return basis_matrix(xi_pts, knots, degree)
    n_ctrl = len(knots) - degree - 1
    n_pts = len(xi_pts)
    if order > degree:
        return jnp.zeros((n_pts, n_ctrl), dtype=jnp.float64)
    t = np.asarray(knots)
    x = np.asarray(xi_pts)
    vals = _identity_spline(t, degree).derivative(nu=order)(x)
    return jnp.array(vals, dtype=jnp.float64)
