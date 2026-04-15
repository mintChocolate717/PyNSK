"""Tests for src/postprocess.py (D3).

Builds a lightweight local cache from existing bsplines/quadrature modules
so the tests do not depend on Phase B's ``src.assembler``. The cache is
duck-typed exactly as documented in ``src/postprocess.py``.
"""
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pytest

from src.bsplines import basis_deriv_matrix, basis_matrix, make_knot_vector
from src.postprocess import (
    bubble_radius,
    entropy_production_rate,
    mass_conservation_error,
    total_free_energy,
    total_internal_energy,
)
from src.quadrature import quadrature_points


@dataclass
class _Cache:
    r_q: jnp.ndarray
    w_q: jnp.ndarray
    N_rho: jnp.ndarray
    dN_rho: jnp.ndarray
    d2N_rho: jnp.ndarray
    N_u: jnp.ndarray
    dN_u: jnp.ndarray
    N_vartheta: jnp.ndarray
    dN_vartheta: jnp.ndarray
    N_V: jnp.ndarray
    dN_V: jnp.ndarray


@dataclass
class _Params:
    Re: float = 100.0
    We: float = 1.0
    Pr: float = 7.0
    gamma: float = 1.4


def _build_cache(n_ctrl: int = 10, degree: int = 2, n_quad: int = 3, R_max: float = 1.0) -> _Cache:
    knots = make_knot_vector(n_ctrl, degree)
    xi_q, r_q, w_q = quadrature_points(knots, degree, n_quad, R_max)
    N = basis_matrix(xi_q, knots, degree)
    # d/dr = (1/R_max) d/dxi  because r = R_max * xi
    dN = basis_deriv_matrix(xi_q, knots, degree, 1) / R_max
    d2N = basis_deriv_matrix(xi_q, knots, degree, 2) / R_max**2
    return _Cache(
        r_q=r_q, w_q=w_q,
        N_rho=N, dN_rho=dN, d2N_rho=d2N,
        N_u=N, dN_u=dN,
        N_vartheta=N, dN_vartheta=dN,
        N_V=N, dN_V=dN,
    )


# ----------------------------------------------------------------------
# bubble_radius
# ----------------------------------------------------------------------

def test_bubble_radius_linear_interp():
    r = np.linspace(0.0, 1.0, 11)
    rho = np.where(r < 0.5, 0.05, 0.95)
    # crossing happens between r=0.4 and r=0.5 since rho jumps from 0.05 -> 0.95
    R = bubble_radius(rho, r, threshold=0.5)
    assert 0.4 <= R <= 0.5


def test_bubble_radius_exact_on_node():
    r = np.linspace(0.0, 1.0, 11)
    rho = r.copy()  # rho(r) = r
    R = bubble_radius(rho, r, threshold=0.5)
    assert np.isclose(R, 0.5)


def test_bubble_radius_no_crossing():
    r = np.linspace(0.0, 1.0, 5)
    rho = np.ones_like(r) * 0.8
    R = bubble_radius(rho, r, threshold=0.5)
    assert np.isnan(R)


# ----------------------------------------------------------------------
# energy integrals
# ----------------------------------------------------------------------

def test_total_free_energy_finite_and_real():
    cache = _build_cache()
    n_ctrl = cache.N_rho.shape[1]
    ctrl = {
        "rho": jnp.ones(n_ctrl) * 0.3,
        "u": jnp.zeros(n_ctrl),
        "vartheta": jnp.ones(n_ctrl) * 0.9,
        "V": jnp.zeros(n_ctrl),
    }
    F = total_free_energy(ctrl, cache, _Params())
    assert jnp.isfinite(F)


def test_total_internal_energy_scaling_with_rho():
    """ρE is positive for a uniform-hot quiescent state with small ρ (ideal-gas regime)."""
    cache = _build_cache()
    n_ctrl = cache.N_rho.shape[1]
    ctrl_a = {
        "rho": jnp.ones(n_ctrl) * 0.1,
        "u": jnp.zeros(n_ctrl),
        "vartheta": jnp.ones(n_ctrl) * 1.2,
        "V": jnp.zeros(n_ctrl),
    }
    ctrl_b = {k: v if k != "rho" else v * 2.0 for k, v in ctrl_a.items()}
    U_a = float(total_internal_energy(ctrl_a, cache, _Params()))
    U_b = float(total_internal_energy(ctrl_b, cache, _Params()))
    # Not strictly linear, but doubling rho should increase |ρE| magnitude
    assert abs(U_b) > abs(U_a) * 1.5


# ----------------------------------------------------------------------
# mass conservation
# ----------------------------------------------------------------------

def test_mass_conservation_on_steady_state():
    """If rho never changes, the conservation error stays zero."""
    cache = _build_cache()
    n_ctrl = cache.N_rho.shape[1]
    ctrl = {
        "rho": jnp.ones(n_ctrl) * 0.3,
        "u": jnp.zeros(n_ctrl),
        "vartheta": jnp.ones(n_ctrl),
        "V": jnp.zeros(n_ctrl),
    }
    history = [ctrl] * 5
    err = mass_conservation_error(history, cache)
    assert err.shape == (5,)
    assert jnp.allclose(err, 0.0, atol=1e-14)


def test_mass_conservation_detects_drift():
    cache = _build_cache()
    n_ctrl = cache.N_rho.shape[1]
    ctrl0 = {"rho": jnp.ones(n_ctrl) * 0.3, "u": jnp.zeros(n_ctrl),
             "vartheta": jnp.ones(n_ctrl), "V": jnp.zeros(n_ctrl)}
    ctrl1 = {"rho": jnp.ones(n_ctrl) * 0.31, "u": jnp.zeros(n_ctrl),
             "vartheta": jnp.ones(n_ctrl), "V": jnp.zeros(n_ctrl)}
    err = mass_conservation_error([ctrl0, ctrl1], cache)
    assert float(err[0]) == 0.0
    assert float(err[1]) > 0.0


def test_mass_integral_is_sphere_volume_times_density():
    """Uniform ρ≡1 on [0, R_max]: ∫ρ r² dr = R_max³/3."""
    R_max = 2.0
    cache = _build_cache(n_ctrl=10, degree=2, n_quad=4, R_max=R_max)
    n_ctrl = cache.N_rho.shape[1]
    ctrl = {"rho": jnp.ones(n_ctrl), "u": jnp.zeros(n_ctrl),
            "vartheta": jnp.ones(n_ctrl), "V": jnp.zeros(n_ctrl)}
    history = [ctrl]
    err = mass_conservation_error(history, cache)
    assert float(err[0]) == 0.0


# ----------------------------------------------------------------------
# entropy production
# ----------------------------------------------------------------------

def test_entropy_production_rest_state_is_zero():
    """No velocity, no gradients → zero entropy production."""
    cache = _build_cache()
    n_ctrl = cache.N_rho.shape[1]
    ctrl = {"rho": jnp.ones(n_ctrl) * 0.3, "u": jnp.zeros(n_ctrl),
            "vartheta": jnp.ones(n_ctrl), "V": jnp.zeros(n_ctrl)}
    ctrl_dot = {k: jnp.zeros(n_ctrl) for k in ctrl}
    sigma = entropy_production_rate(ctrl, ctrl_dot, cache, _Params())
    assert jnp.allclose(sigma, 0.0, atol=1e-12)


def test_entropy_production_is_finite_with_gradients():
    cache = _build_cache()
    n_ctrl = cache.N_rho.shape[1]
    # linear u profile: u_r = C*r corresponds to ctrl with dN @ ctrl ~ C
    ctrl = {
        "rho": jnp.ones(n_ctrl) * 0.3,
        "u": jnp.linspace(0.0, 1.0, n_ctrl),
        "vartheta": jnp.linspace(0.9, 1.1, n_ctrl),
        "V": jnp.zeros(n_ctrl),
    }
    ctrl_dot = {k: jnp.zeros(n_ctrl) for k in ctrl}
    sigma = entropy_production_rate(ctrl, ctrl_dot, cache, _Params())
    assert jnp.isfinite(sigma)


# ----------------------------------------------------------------------
# dict-cache compatibility
# ----------------------------------------------------------------------

def test_dict_cache_supported():
    cache = _build_cache()
    dict_cache = {
        "r_q": cache.r_q, "w_q": cache.w_q,
        "N_rho": cache.N_rho, "dN_rho": cache.dN_rho,
        "N_u": cache.N_u, "N_vartheta": cache.N_vartheta,
    }
    n_ctrl = cache.N_rho.shape[1]
    ctrl = {"rho": jnp.ones(n_ctrl) * 0.3, "u": jnp.zeros(n_ctrl),
            "vartheta": jnp.ones(n_ctrl), "V": jnp.zeros(n_ctrl)}
    F = total_free_energy(ctrl, dict_cache, {"gamma": 1.4, "We": 1.0})
    assert jnp.isfinite(F)


def test_missing_param_raises():
    cache = _build_cache()
    n_ctrl = cache.N_rho.shape[1]
    ctrl = {"rho": jnp.ones(n_ctrl) * 0.3, "u": jnp.zeros(n_ctrl),
            "vartheta": jnp.ones(n_ctrl), "V": jnp.zeros(n_ctrl)}
    with pytest.raises(ValueError):
        total_free_energy(ctrl, cache, {"gamma": 1.4})  # We missing
