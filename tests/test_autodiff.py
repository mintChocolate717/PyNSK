"""Autodiff-vs-finite-difference cross-check on all four element residuals.

Each element residual is a smooth function of its control-point arrays.
We build a small random test case (degree-3 B-spline basis, a handful
of control points), then verify that ``jax.jacfwd`` over each control
vector matches a central finite-difference estimate to ~1e-6 tolerance.

This is the prerequisite for Phase C, which will assemble the global
tangent stiffness K_tan from these same derivatives.
"""
import jax
import jax.numpy as jnp
import numpy as np

from src.bsplines import basis_deriv_matrix, basis_matrix, make_knot_vector
from src.quadrature import gauss_legendre
from src.residuals import (
    element_residual_auxiliary,
    element_residual_energy,
    element_residual_mass,
    element_residual_momentum,
)

# ── test fixture ──────────────────────────────────────────────────────────────

DEGREE = 3
N_CTRL = 6
N_GAUSS = 4
GAMMA = 1.4
RE = 50.0
WE = 100.0
PR = 7.0
B_R = 0.0
R_S = 0.0


def _mesh():
    """Build a small single-element-ish setup over the full knot vector.

    We reuse one Gauss rule and treat the entire parametric domain [0,1]
    as a single macro-element via J_e, which is fine for the unit test
    — we're not testing integration error, just autodiff vs FD on a
    smooth integrand.
    """
    knots = make_knot_vector(N_CTRL, DEGREE)
    xi_ref, w_ref = gauss_legendre(N_GAUSS)
    # Map Gauss nodes from [-1,1] to [0,1]
    xi_pts = 0.5 * (xi_ref + 1.0)
    J_e = 0.5      # dξ_phys/dξ_ref (parametric Jacobian)
    R_MAX = 2.0
    r_q = R_MAX * xi_pts + 0.5  # shift to keep r away from 0
    w_q = w_ref

    N   = basis_matrix(xi_pts, knots, DEGREE)
    dN  = basis_deriv_matrix(xi_pts, knots, DEGREE, 1) / R_MAX  # chain rule
    d2N = basis_deriv_matrix(xi_pts, knots, DEGREE, 2) / R_MAX**2
    return N, dN, d2N, r_q, w_q, J_e


def _random_controls(rng):
    """Physically-plausible random control points."""
    ctrl_rho      = rng.uniform(0.15, 0.45, size=N_CTRL)
    ctrl_vartheta = rng.uniform(0.80, 1.20, size=N_CTRL)
    ctrl_u        = rng.uniform(-0.20, 0.20, size=N_CTRL)
    ctrl_V        = rng.uniform(-0.20, 0.20, size=N_CTRL)
    ctrl_rho_dot  = rng.uniform(-0.05, 0.05, size=N_CTRL)
    ctrl_u_dot    = rng.uniform(-0.05, 0.05, size=N_CTRL)
    ctrl_vt_dot   = rng.uniform(-0.05, 0.05, size=N_CTRL)
    return tuple(jnp.array(a) for a in (
        ctrl_rho, ctrl_vartheta, ctrl_u, ctrl_V,
        ctrl_rho_dot, ctrl_u_dot, ctrl_vt_dot,
    ))


def _fd_jacobian(f, x, eps=1e-6):
    """Central-difference Jacobian of f: R^n -> R^m."""
    x = np.asarray(x, dtype=np.float64)
    m = np.asarray(f(jnp.array(x))).shape[0]
    n = x.shape[0]
    J = np.zeros((m, n))
    for k in range(n):
        xp = x.copy()
        xp[k] += eps
        xm = x.copy()
        xm[k] -= eps
        J[:, k] = (np.asarray(f(jnp.array(xp))) - np.asarray(f(jnp.array(xm)))) / (2 * eps)
    return J


# ── mass ──────────────────────────────────────────────────────────────────────

def _pack_mass(N, dN, r_q, w_q, J_e, ctrl_u, ctrl_rho_dot):
    def R(ctrl_rho):
        return element_residual_mass(
            N, dN, N, ctrl_rho, ctrl_rho_dot, ctrl_u, r_q, w_q, J_e,
        )
    return R


def test_autodiff_mass_vs_fd():
    rng = np.random.default_rng(1)
    N, dN, d2N, r_q, w_q, J_e = _mesh()
    ctrl_rho, ctrl_vt, ctrl_u, ctrl_V, ctrl_rho_dot, ctrl_u_dot, ctrl_vt_dot = _random_controls(rng)

    R = _pack_mass(N, dN, r_q, w_q, J_e, ctrl_u, ctrl_rho_dot)
    J_ad = np.asarray(jax.jacfwd(R)(ctrl_rho))
    J_fd = _fd_jacobian(R, ctrl_rho)
    assert np.allclose(J_ad, J_fd, atol=1e-6, rtol=1e-6)


# ── momentum ─────────────────────────────────────────────────────────────────

def _pack_momentum(N, dN, d2N, r_q, w_q, J_e,
                   ctrl_rho, ctrl_rho_dot, ctrl_vartheta, ctrl_V,
                   ctrl_u_dot):
    def R(ctrl_u):
        return element_residual_momentum(
            N, dN, N, dN, d2N, N, dN, N,
            ctrl_rho, ctrl_rho_dot, ctrl_u, ctrl_u_dot, ctrl_vartheta, ctrl_V,
            r_q, w_q, J_e, RE, WE, GAMMA, B_R,
        )
    return R


def test_autodiff_momentum_vs_fd_wrt_u():
    rng = np.random.default_rng(2)
    N, dN, d2N, r_q, w_q, J_e = _mesh()
    ctrl_rho, ctrl_vt, ctrl_u, ctrl_V, ctrl_rho_dot, ctrl_u_dot, ctrl_vt_dot = _random_controls(rng)
    R = _pack_momentum(N, dN, d2N, r_q, w_q, J_e,
                       ctrl_rho, ctrl_rho_dot, ctrl_vt, ctrl_V, ctrl_u_dot)
    J_ad = np.asarray(jax.jacfwd(R)(ctrl_u))
    J_fd = _fd_jacobian(R, ctrl_u)
    assert np.allclose(J_ad, J_fd, atol=1e-6, rtol=1e-6)


def test_autodiff_momentum_vs_fd_wrt_rho():
    rng = np.random.default_rng(3)
    N, dN, d2N, r_q, w_q, J_e = _mesh()
    ctrl_rho, ctrl_vt, ctrl_u, ctrl_V, ctrl_rho_dot, ctrl_u_dot, ctrl_vt_dot = _random_controls(rng)

    def R(cr):
        return element_residual_momentum(
            N, dN, N, dN, d2N, N, dN, N,
            cr, ctrl_rho_dot, ctrl_u, ctrl_u_dot, ctrl_vt, ctrl_V,
            r_q, w_q, J_e, RE, WE, GAMMA, B_R,
        )

    J_ad = np.asarray(jax.jacfwd(R)(ctrl_rho))
    J_fd = _fd_jacobian(R, ctrl_rho)
    # Second derivatives of rho amplify FD roundoff; loosen tolerance slightly.
    assert np.allclose(J_ad, J_fd, atol=5e-5, rtol=5e-5)


# ── energy ───────────────────────────────────────────────────────────────────

def test_autodiff_energy_vs_fd_wrt_vartheta():
    rng = np.random.default_rng(4)
    N, dN, d2N, r_q, w_q, J_e = _mesh()
    ctrl_rho, ctrl_vt, ctrl_u, ctrl_V, ctrl_rho_dot, ctrl_u_dot, ctrl_vt_dot = _random_controls(rng)

    def R(cvt):
        return element_residual_energy(
            N, dN, N, dN, d2N, N, dN, N,
            ctrl_rho, ctrl_rho_dot, ctrl_u, ctrl_u_dot,
            cvt, ctrl_vt_dot, ctrl_V,
            r_q, w_q, J_e, RE, WE, GAMMA, PR, B_R, R_S,
        )

    J_ad = np.asarray(jax.jacfwd(R)(ctrl_vt))
    J_fd = _fd_jacobian(R, ctrl_vt)
    assert np.allclose(J_ad, J_fd, atol=1e-6, rtol=1e-6)


def test_autodiff_energy_vs_fd_wrt_u():
    rng = np.random.default_rng(5)
    N, dN, d2N, r_q, w_q, J_e = _mesh()
    ctrl_rho, ctrl_vt, ctrl_u, ctrl_V, ctrl_rho_dot, ctrl_u_dot, ctrl_vt_dot = _random_controls(rng)

    def R(cu):
        return element_residual_energy(
            N, dN, N, dN, d2N, N, dN, N,
            ctrl_rho, ctrl_rho_dot, cu, ctrl_u_dot,
            ctrl_vt, ctrl_vt_dot, ctrl_V,
            r_q, w_q, J_e, RE, WE, GAMMA, PR, B_R, R_S,
        )

    J_ad = np.asarray(jax.jacfwd(R)(ctrl_u))
    J_fd = _fd_jacobian(R, ctrl_u)
    assert np.allclose(J_ad, J_fd, atol=1e-6, rtol=1e-6)


# ── auxiliary ────────────────────────────────────────────────────────────────

def test_autodiff_auxiliary_vs_fd_wrt_V():
    rng = np.random.default_rng(6)
    N, dN, d2N, r_q, w_q, J_e = _mesh()
    ctrl_rho, ctrl_vt, ctrl_u, ctrl_V, ctrl_rho_dot, ctrl_u_dot, ctrl_vt_dot = _random_controls(rng)

    def R(cv):
        return element_residual_auxiliary(
            N, dN, N, dN, N, N,
            ctrl_rho, ctrl_u, ctrl_vt, cv,
            r_q, w_q, J_e, WE, GAMMA,
        )

    J_ad = np.asarray(jax.jacfwd(R)(ctrl_V))
    J_fd = _fd_jacobian(R, ctrl_V)
    assert np.allclose(J_ad, J_fd, atol=1e-6, rtol=1e-6)


def test_autodiff_auxiliary_vs_fd_wrt_rho():
    rng = np.random.default_rng(7)
    N, dN, d2N, r_q, w_q, J_e = _mesh()
    ctrl_rho, ctrl_vt, ctrl_u, ctrl_V, ctrl_rho_dot, ctrl_u_dot, ctrl_vt_dot = _random_controls(rng)

    def R(cr):
        return element_residual_auxiliary(
            N, dN, N, dN, N, N,
            cr, ctrl_u, ctrl_vt, ctrl_V,
            r_q, w_q, J_e, WE, GAMMA,
        )

    J_ad = np.asarray(jax.jacfwd(R)(ctrl_rho))
    J_fd = _fd_jacobian(R, ctrl_rho)
    assert np.allclose(J_ad, J_fd, atol=1e-6, rtol=1e-6)


# ── sanity: reverse-mode agrees with forward-mode ────────────────────────────

def test_reverse_mode_matches_forward_mode_mass():
    """jacrev and jacfwd must return the same Jacobian."""
    rng = np.random.default_rng(8)
    N, dN, d2N, r_q, w_q, J_e = _mesh()
    ctrl_rho, ctrl_vt, ctrl_u, ctrl_V, ctrl_rho_dot, ctrl_u_dot, ctrl_vt_dot = _random_controls(rng)
    R = _pack_mass(N, dN, r_q, w_q, J_e, ctrl_u, ctrl_rho_dot)
    J_fwd = jax.jacfwd(R)(ctrl_rho)
    J_rev = jax.jacrev(R)(ctrl_rho)
    assert jnp.allclose(J_fwd, J_rev, atol=1e-12)
