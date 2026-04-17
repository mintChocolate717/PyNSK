"""Tests for src/initial_conditions.py (D1).

Verifies:
- Density tanh profile: interface centered at R_bubble, far-field = rho_liq,
  center ~ rho_vap.
- L2 projection recovers polynomials representable by the B-spline basis
  exactly (within numerical precision).
- Control-point projection produces arrays of the right shape, keyed by
  {'rho', 'u', 'vartheta', 'V'}.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from src.bsplines import make_knot_vector
from src.initial_conditions import bubble_profile, from_bspline_projection

GAMMA = 1.4


def test_bubble_profile_pointwise_interface_center():
    """rho at r = R_bubble should equal average of rho_liq and rho_vap (tanh 0)."""
    R_bubble = 0.3
    rho_liq = 0.6
    rho_vap = 0.05
    r = np.array([R_bubble])
    out = bubble_profile(
        r=r,
        R_bubble=R_bubble,
        interface_width=0.05,
        rho_liq=rho_liq,
        rho_vap=rho_vap,
        vartheta_0=0.85,
    )
    mid = 0.5 * (rho_liq + rho_vap)
    assert jnp.allclose(out["rho"], mid, atol=1e-12)


def test_bubble_profile_far_field_liquid():
    """For r >> R_bubble, rho → rho_liq."""
    rho_liq = 0.6
    out = bubble_profile(
        r=np.array([1.0]),
        R_bubble=0.3,
        interface_width=0.01,
        rho_liq=rho_liq,
        rho_vap=0.05,
        vartheta_0=0.85,
    )
    assert jnp.allclose(out["rho"], rho_liq, atol=1e-6)


def test_bubble_profile_center_vapor():
    """At r=0 (well below R_bubble), rho → rho_vap."""
    rho_vap = 0.05
    out = bubble_profile(
        r=np.array([0.0]),
        R_bubble=0.3,
        interface_width=0.01,
        rho_liq=0.6,
        rho_vap=rho_vap,
        vartheta_0=0.85,
    )
    assert jnp.allclose(out["rho"], rho_vap, atol=1e-6)


def test_bubble_profile_uniform_temperature():
    """vartheta is exactly vartheta_0 everywhere; u is zero everywhere."""
    r = np.linspace(0.0, 1.0, 11)
    vartheta_0 = 0.85
    out = bubble_profile(
        r=r,
        R_bubble=0.3,
        interface_width=0.05,
        rho_liq=0.6,
        rho_vap=0.05,
        vartheta_0=vartheta_0,
    )
    assert jnp.allclose(out["vartheta"], vartheta_0)
    assert jnp.allclose(out["u"], 0.0)


def test_bubble_profile_keys():
    r = np.linspace(0.1, 0.9, 5)
    out = bubble_profile(
        r=r, R_bubble=0.3, interface_width=0.05, rho_liq=0.6, rho_vap=0.05, vartheta_0=0.85
    )
    assert set(out.keys()) == {"rho", "u", "vartheta", "V"}


def test_projection_recovers_constant():
    """A constant function is projected exactly onto the B-spline space."""
    degree = 2
    n_ctrl = 6
    knots = np.asarray(make_knot_vector(n_ctrl, degree))
    n_quad = degree + 2

    c = 3.14159

    ctrl = from_bspline_projection(lambda r: c * np.ones_like(r), knots, degree, n_quad, R_max=1.0)
    assert ctrl.shape == (n_ctrl,)
    # Partition-of-unity: each control coefficient equals the constant.
    assert jnp.allclose(ctrl, c, atol=1e-8)


def test_projection_recovers_linear():
    """A linear function f(r)=2r + 1 is in the B-spline space (p>=1) exactly."""
    degree = 2
    n_ctrl = 6
    knots = np.asarray(make_knot_vector(n_ctrl, degree))
    n_quad = degree + 2

    def f(r):
        return 2.0 * r + 1.0

    ctrl = from_bspline_projection(f, knots, degree, n_quad, R_max=1.0)

    # Evaluate at quadrature r-points and compare.
    from src.bsplines import basis_matrix
    from src.quadrature import quadrature_points

    xi_pts, r_pts, _ = quadrature_points(jnp.asarray(knots), degree, n_quad, 1.0)
    N = basis_matrix(xi_pts, jnp.asarray(knots), degree)
    f_proj = np.asarray(N @ ctrl)
    f_exact = f(np.asarray(r_pts))
    assert np.allclose(f_proj, f_exact, atol=1e-9)


def test_projection_recovers_quadratic_with_p2():
    """p=2 basis represents quadratics exactly — projection recovers them."""
    degree = 2
    n_ctrl = 8
    knots = np.asarray(make_knot_vector(n_ctrl, degree))
    n_quad = degree + 2

    def f(r):
        return 0.5 * r**2 + 3.0 * r - 2.0

    ctrl = from_bspline_projection(f, knots, degree, n_quad, R_max=1.0)

    from src.bsplines import basis_matrix
    from src.quadrature import quadrature_points

    xi_pts, r_pts, _ = quadrature_points(jnp.asarray(knots), degree, n_quad, 1.0)
    N = basis_matrix(xi_pts, jnp.asarray(knots), degree)
    f_proj = np.asarray(N @ ctrl)
    f_exact = f(np.asarray(r_pts))
    assert np.allclose(f_proj, f_exact, atol=1e-8)


def test_bubble_profile_control_points_mode():
    """Control-point projection mode returns arrays of length n_ctrl."""
    degree = 2
    n_ctrl = 10
    knots = np.asarray(make_knot_vector(n_ctrl, degree))
    out = bubble_profile(
        R_bubble=0.3,
        interface_width=0.05,
        rho_liq=0.6,
        rho_vap=0.05,
        vartheta_0=0.85,
        knots=knots,
        degree=degree,
        n_quad=degree + 2,
        R_max=1.0,
        We=1.0,
        gamma=GAMMA,
    )
    assert set(out.keys()) == {"rho", "u", "vartheta", "V"}
    for field in ("rho", "u", "vartheta", "V"):
        assert out[field].shape == (n_ctrl,)
    # u is identically zero → control points all zero
    assert jnp.allclose(out["u"], 0.0, atol=1e-12)
    # vartheta is a positive constant → control points all equal to vartheta_0
    assert jnp.allclose(out["vartheta"], 0.85, atol=1e-8)


def test_bubble_profile_requires_r_or_knots():
    with pytest.raises(ValueError):
        bubble_profile(
            R_bubble=0.3, interface_width=0.05, rho_liq=0.6, rho_vap=0.05, vartheta_0=0.85
        )
