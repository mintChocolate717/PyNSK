"""
Initial condition utilities for the NSK bubble problem.

Provides:
- bubble_profile: analytic tanh ρ profile + equilibrium (u=0) V, uniform ϑ.
- from_bspline_projection: L² projection of a physical-space function
  onto the B-spline space (returns control-point array).

Control-point dictionary keys follow the convention:
    'rho', 'u', 'vartheta', 'V'
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np

from src.bsplines import basis_matrix, make_knot_vector  # noqa: F401
from src.constitutive import chemical_potential
from src.quadrature import quadrature_points


def _gauss_on_element(
    knots: np.ndarray, degree: int, n_quad: int, R_max: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xi_pts, r_pts, w_pts = quadrature_points(jnp.asarray(knots), degree, n_quad, R_max)
    return np.asarray(xi_pts), np.asarray(r_pts), np.asarray(w_pts)


def from_bspline_projection(
    field_fn: Callable[[np.ndarray], np.ndarray],
    knots: np.ndarray,
    degree: int,
    n_quad: int,
    R_max: float = 1.0,
) -> jnp.ndarray:
    """L² projection of a physical-space function onto the B-spline space.

    Solves  M c = b  where
        M_{AB} = ∫ N_A(r) N_B(r) dr           (line integral, no r² weight)
        b_A   = ∫ N_A(r) f(r) dr

    The unweighted L² inner product is used so polynomial functions
    representable by the B-spline basis are recovered exactly.

    Args:
        field_fn: callable f(r) → array; evaluated at quadrature r-values.
        knots:    B-spline knot vector on [0, 1], shape (n_ctrl + p + 1,).
        degree:   polynomial degree p.
        n_quad:   Gauss-Legendre points per element (≥ p + 1).
        R_max:    physical outer radius. r = R_max * xi.

    Returns:
        control-point array of shape (n_ctrl,), jax.numpy float64.
    """
    knots_np = np.asarray(knots, dtype=np.float64)
    xi_pts, r_pts, w_pts = _gauss_on_element(knots_np, degree, n_quad, R_max)
    N = np.asarray(basis_matrix(jnp.asarray(xi_pts), jnp.asarray(knots_np), degree))
    f_vals = np.asarray(field_fn(r_pts), dtype=np.float64)

    # Unweighted L² (dr only) — physical-space projection on [0, R_max].
    M = (N.T * w_pts) @ N
    b = (N.T * w_pts) @ f_vals

    c = np.linalg.solve(M, b)
    return jnp.asarray(c, dtype=jnp.float64)


def bubble_profile(
    r: np.ndarray | None = None,
    R_bubble: float = 0.3,
    interface_width: float = 0.05,
    rho_liq: float = 0.6,
    rho_vap: float = 0.05,
    vartheta_0: float = 0.85,
    *,
    knots: np.ndarray | None = None,
    degree: int | None = None,
    n_quad: int | None = None,
    R_max: float = 1.0,
    We: float = 1.0,
    gamma: float = 1.4,
) -> dict[str, jnp.ndarray]:
    """Initial-condition control-point dictionary for a spherical bubble.

    Density profile (physical r):
        ρ(r) = ½(ρ_liq + ρ_vap) + ½(ρ_liq − ρ_vap) * tanh((r − R_bubble)/w)

    Velocity is zero.
    Temperature is uniform at ``vartheta_0``.
    V is taken from the auxiliary-equation equilibrium at u=0:
        V = (ν_loc(ρ, ϑ) − (1/(We ϑ)) ∇ρ · 0) / ϑ
          = ν_loc(ρ, ϑ) / ϑ
    (The ∇ρ term vanishes in the strong-form auxiliary eq. only at
    stationary profiles; here we evaluate V = ν_loc/ϑ as the best
    pointwise equilibrium guess, consistent with the docstring spec.)

    Two calling modes:

    1. **Analytic evaluation on r-grid**:
       Provide ``r`` as an array of physical coordinates; returns dicts
       of arrays at those r-values (NOT control points).

    2. **Control-point projection** (preferred for solver use):
       Provide ``knots``, ``degree``, ``n_quad`` (and optionally ``R_max``);
       each field is L² projected onto the B-spline space and control-point
       arrays are returned.
    """

    def rho_fn(rr):
        rr = np.asarray(rr, dtype=np.float64)
        return 0.5 * (rho_liq + rho_vap) + 0.5 * (rho_liq - rho_vap) * np.tanh(
            (rr - R_bubble) / interface_width
        )

    def u_fn(rr):
        return np.zeros_like(np.asarray(rr, dtype=np.float64))

    def vartheta_fn(rr):
        return vartheta_0 * np.ones_like(np.asarray(rr, dtype=np.float64))

    def V_fn(rr):
        rho_v = rho_fn(rr)
        theta_v = vartheta_fn(rr)
        nu_loc = np.asarray(chemical_potential(jnp.asarray(rho_v), jnp.asarray(theta_v), gamma))
        return nu_loc / theta_v

    if knots is None:
        # Mode 1: pointwise evaluation on provided r-grid
        if r is None:
            raise ValueError("Provide either `r` or (`knots`, `degree`, `n_quad`).")
        return {
            "rho": jnp.asarray(rho_fn(r)),
            "u": jnp.asarray(u_fn(r)),
            "vartheta": jnp.asarray(vartheta_fn(r)),
            "V": jnp.asarray(V_fn(r)),
        }

    # Mode 2: B-spline projection
    if degree is None or n_quad is None:
        raise ValueError("Control-point mode requires `knots`, `degree`, and `n_quad`.")

    return {
        "rho": from_bspline_projection(rho_fn, knots, degree, n_quad, R_max=R_max),
        "u": from_bspline_projection(u_fn, knots, degree, n_quad, R_max=R_max),
        "vartheta": from_bspline_projection(vartheta_fn, knots, degree, n_quad, R_max=R_max),
        "V": from_bspline_projection(V_fn, knots, degree, n_quad, R_max=R_max),
    }
