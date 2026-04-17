"""
Gaussian quadrature for 1D spherically symmetric IGA.

Coordinate mapping (per LaTeX §Spatial Discretization):
    r(xi_ref) = (r_{e+1} - r_e)/2 * xi_ref + (r_{e+1} + r_e)/2
    J_e = dr/d(xi_ref) = (r_{e+1} - r_e) / 2

The spherical r² factor in the volume integral is NOT included here —
it is applied element-by-element inside each residual function.

Usage pattern:
    xi_pts, r_pts, weights = quadrature_points(knots, degree, n_gauss, R_max)
    N   = basis_matrix(xi_pts, knots, degree)          # (n_pts, n_ctrl)
    dN  = basis_deriv_matrix(xi_pts, knots, degree, 1) # (n_pts, n_ctrl)
    d2N = basis_deriv_matrix(xi_pts, knots, degree, 2) # (n_pts, n_ctrl)
    rho = N @ d_ctrl                                   # field at quad pts
"""

import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss


def gauss_legendre(n_pts: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Gauss-Legendre points and weights on [-1, 1].

    Integrates polynomials of degree up to 2*n_pts - 1 exactly.

    Args:
        n_pts: number of quadrature points per element

    Returns:
        xi_ref: shape (n_pts,) — points on [-1, 1]
        weights: shape (n_pts,) — corresponding weights
    """
    pts, wts = leggauss(n_pts)
    return jnp.array(pts), jnp.array(wts)


def recommended_n_gauss(degree: int) -> int:
    """Minimum Gauss points to integrate B-spline mass matrices exactly.

    For degree-p basis, the mass integrand N_A * N_B is degree 2p.
    Gauss-Legendre with n points integrates degree 2n-1 exactly,
    so n = p + 1 suffices. Add extra points for nonlinear physics.
    """
    return degree + 1


def quadrature_points(
    knots: jnp.ndarray,
    degree: int,
    n_gauss: int,
    R_max: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """All quadrature points across the 1D radial mesh.

    Elements are the non-zero knot spans of the B-spline knot vector.
    The geometry mapping is linear: r = R_max * xi (xi in [0, 1]).

    Args:
        knots:   shape (n_ctrl + degree + 1,) — B-spline knot vector in [0, 1]
        degree:  polynomial degree (used to identify elements via knots only)
        n_gauss: number of Gauss points per element
        R_max:   outer radius of the physical domain [0, R_max]

    Returns:
        xi_pts:  shape (n_elem * n_gauss,) — parametric coords in [0, 1]
                 pass directly to basis_matrix / basis_deriv_matrix
        r_pts:   shape (n_elem * n_gauss,) — physical radial coords r = R_max * xi
        weights: shape (n_elem * n_gauss,) — w_q * J_e  (element Jacobian included)
                 does NOT include the r² spherical factor
    """
    t = np.asarray(knots)
    xi_ref, w_ref = leggauss(n_gauss)

    # Non-zero knot spans define the elements
    unique_knots = np.unique(t)
    xi_a_arr = unique_knots[:-1]
    xi_b_arr = unique_knots[1:]

    xi_all, r_all, w_all = [], [], []
    for xi_a, xi_b in zip(xi_a_arr, xi_b_arr, strict=True):
        xi_half = (xi_b - xi_a) / 2.0
        xi_mid = (xi_b + xi_a) / 2.0

        xi_elem = xi_mid + xi_half * xi_ref  # parametric in [xi_a, xi_b]
        r_elem = R_max * xi_elem  # physical radial coord
        J_e = R_max * xi_half  # dr / d(xi_ref)
        w_elem = w_ref * J_e

        xi_all.append(xi_elem)
        r_all.append(r_elem)
        w_all.append(w_elem)

    return (
        jnp.array(np.concatenate(xi_all)),
        jnp.array(np.concatenate(r_all)),
        jnp.array(np.concatenate(w_all)),
    )
