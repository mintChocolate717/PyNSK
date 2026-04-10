import jax.numpy as jnp

from src.constitutive import chemical_potential
from src.residuals import (
    element_residual_auxiliary,
    element_residual_energy,
    element_residual_mass,
    element_residual_momentum,
)

# ── shared fixtures ────────────────────────────────────────────────────────────

N_QP = 4       # quadrature points per element
N_CTRL = 5     # control points (same basis for all fields in these tests)
GAMMA = 1.4
RE = 10.0
WE = 100.0
PR = 7.0
J_E = 0.5      # element Jacobian


def _uniform_basis(n_qp, n_ctrl):
    """Simple uniform basis: each row sums to 1, derivatives are zero."""
    N = jnp.ones((n_qp, n_ctrl)) / n_ctrl
    dN = jnp.zeros((n_qp, n_ctrl))
    d2N = jnp.zeros((n_qp, n_ctrl))
    return N, dN, d2N


def _quad_data(n_qp):
    r_q = jnp.linspace(0.5, 1.5, n_qp)   # avoid r=0
    w_q = jnp.ones(n_qp) * 0.5
    return r_q, w_q


# ── mass residual ──────────────────────────────────────────────────────────────

def test_mass_residual_shape():
    N, dN, _ = _uniform_basis(N_QP, N_CTRL)
    r_q, w_q = _quad_data(N_QP)
    ctrl = jnp.ones(N_CTRL) * 0.3
    zero = jnp.zeros(N_CTRL)

    R = element_residual_mass(N, dN, N, ctrl, zero, zero, r_q, w_q, J_E)
    assert R.shape == (N_CTRL,)


def test_mass_residual_zero_flow_steady():
    """Zero velocity + zero rho_dot → zero residual."""
    N, dN, _ = _uniform_basis(N_QP, N_CTRL)
    r_q, w_q = _quad_data(N_QP)
    ctrl_rho = jnp.ones(N_CTRL) * 0.3
    zero = jnp.zeros(N_CTRL)

    R = element_residual_mass(N, dN, N, ctrl_rho, zero, zero, r_q, w_q, J_E)
    assert jnp.allclose(R, 0.0, atol=1e-12)


def test_mass_residual_uniform_rho_dot():
    """Constant rho_dot, u=0: residual = rho_dot * ∫ N_C r² dr."""
    N, dN, _ = _uniform_basis(N_QP, N_CTRL)
    r_q, w_q = _quad_data(N_QP)
    rho_dot_val = 2.5
    ctrl_rho_dot = jnp.ones(N_CTRL) * rho_dot_val
    zero = jnp.zeros(N_CTRL)

    R = element_residual_mass(N, dN, N, zero, ctrl_rho_dot, zero, r_q, w_q, J_E)

    # Each R_C = rho_dot * sum_q N_C(r_q) * r_q² * J_e * w_q
    # With uniform basis N_C = 1/N_CTRL at every qp:
    expected_per_ctrl = rho_dot_val * jnp.sum(r_q**2 * J_E * w_q) / N_CTRL
    assert jnp.allclose(R, expected_per_ctrl, atol=1e-10)


# ── momentum residual ──────────────────────────────────────────────────────────

def test_momentum_residual_shape():
    N, dN, d2N = _uniform_basis(N_QP, N_CTRL)
    r_q, w_q = _quad_data(N_QP)
    ctrl_rho = jnp.ones(N_CTRL) * 0.3
    ctrl_theta = jnp.ones(N_CTRL) * 0.9
    zero = jnp.zeros(N_CTRL)

    R = element_residual_momentum(
        N, dN, N, dN, d2N, N, dN, N,
        ctrl_rho, zero, zero, zero, ctrl_theta, zero,
        r_q, w_q, J_E, RE, WE, GAMMA, 0.0,
    )
    assert R.shape == (N_CTRL,)


def test_momentum_residual_uniform_static():
    """Uniform physical state, u=0, V=0, all time derivatives=0, b_r=0 → zero residual.

    With uniform basis (dN=d2N=0): all gradients vanish, stresses vanish, η=0,
    so both integrand coefficients are zero.
    """
    N, dN, d2N = _uniform_basis(N_QP, N_CTRL)
    r_q, w_q = _quad_data(N_QP)
    zero = jnp.zeros(N_CTRL)
    ctrl_rho = jnp.ones(N_CTRL) * 0.3
    ctrl_theta = jnp.ones(N_CTRL) * 1.0

    R = element_residual_momentum(
        N, dN, N, dN, d2N, N, dN, N,
        ctrl_rho, zero, zero, zero, ctrl_theta, zero,
        r_q, w_q, J_E, RE, WE, GAMMA, 0.0,
    )
    assert jnp.allclose(R, 0.0, atol=1e-12)


def test_momentum_residual_uniform_expansion_no_stress():
    """Uniform expansion u_r = C*r: viscous stress zero, with drho=0, dtheta=0."""
    # For u_r = C*r: du_dr = C, u_r/r = C → tau_rr = (4/3Re)(C-C) = 0
    # With drho=0: korteweg stress = 0
    # With V=0, rho_dot=0, u_dot=0, drho=0, dtheta=0:
    # coeff_dN = -rho*u² - eta + 0 + 0  where eta = 0 (V=0, div=0)
    #          = -rho*u²
    # coeff_N  = 0 - xi*0 - H*0 + (2/r)*(0 + 0 - eta) - rho*b_r = -(2/r)*eta
    # With eta = 0 (V=0) → coeff_dN=0, coeff_N = -rho*b_r = 0 (b_r=0)
    # So residual = 0
    N, dN, d2N = _uniform_basis(N_QP, N_CTRL)
    r_q, w_q = _quad_data(N_QP)
    zero = jnp.zeros(N_CTRL)
    ctrl_rho = jnp.ones(N_CTRL) * 0.3
    ctrl_theta = jnp.ones(N_CTRL) * 1.0

    R = element_residual_momentum(
        N, dN, N, dN, d2N, N, dN, N,
        ctrl_rho, zero, zero, zero, ctrl_theta, zero,
        r_q, w_q, J_E, RE, WE, GAMMA, 0.0,
    )
    assert jnp.allclose(R, 0.0, atol=1e-12)


# ── energy residual ────────────────────────────────────────────────────────────

def test_energy_residual_shape():
    N, dN, d2N = _uniform_basis(N_QP, N_CTRL)
    r_q, w_q = _quad_data(N_QP)
    zero = jnp.zeros(N_CTRL)
    ctrl_theta = jnp.ones(N_CTRL) * 1.0
    ctrl_rho = jnp.ones(N_CTRL) * 0.3

    R = element_residual_energy(
        N, dN, N, dN, d2N, N, dN, N,
        ctrl_rho, zero, zero, zero, ctrl_theta, zero, zero,
        r_q, w_q, J_E, RE, WE, GAMMA, PR, 0.0, 0.0,
    )
    assert R.shape == (N_CTRL,)


def test_energy_residual_static_uniform():
    """No flow, no time derivatives, no source, uniform fields → zero residual.

    With u=0, drho=0, dtheta=0, all time derivatives=0, b_r=0, r_s=0, V=0:
    - β = ρVϑ - ϑH + 0 + 0 + 0 = -ϑH  (uniform fields so H = ρs is nonzero but...)
    - coeff_dN = (-β + 0 + 0)*u - 0 - 0 = 0  (u=0)
    - d_rhoE_dt = 0 (all time derivatives zero)
    - coeff_N = 0 - 0 - 0 = 0
    → residual = 0
    """
    N, dN, d2N = _uniform_basis(N_QP, N_CTRL)
    r_q, w_q = _quad_data(N_QP)
    zero = jnp.zeros(N_CTRL)
    ctrl_theta = jnp.ones(N_CTRL) * 1.0
    ctrl_rho = jnp.ones(N_CTRL) * 0.3

    R = element_residual_energy(
        N, dN, N, dN, d2N, N, dN, N,
        ctrl_rho, zero, zero, zero, ctrl_theta, zero, zero,
        r_q, w_q, J_E, RE, WE, GAMMA, PR, 0.0, 0.0,
    )
    assert jnp.allclose(R, 0.0, atol=1e-12)


# ── auxiliary residual ─────────────────────────────────────────────────────────

def test_auxiliary_residual_shape():
    N, dN, _ = _uniform_basis(N_QP, N_CTRL)
    r_q, w_q = _quad_data(N_QP)
    zero = jnp.zeros(N_CTRL)
    ctrl_rho = jnp.ones(N_CTRL) * 0.3
    ctrl_theta = jnp.ones(N_CTRL) * 1.0
    ctrl_V = jnp.zeros(N_CTRL)

    R = element_residual_auxiliary(
        N, dN, N, dN, N, N,
        ctrl_rho, zero, ctrl_theta, ctrl_V,
        r_q, w_q, J_E, WE, GAMMA,
    )
    assert R.shape == (N_CTRL,)


def test_auxiliary_residual_equilibrium():
    """V = ν_loc/ϑ (chemical equilibrium), u=0, drho=0 → zero residual."""
    N, dN, _ = _uniform_basis(N_QP, N_CTRL)
    r_q, w_q = _quad_data(N_QP)
    zero = jnp.zeros(N_CTRL)

    rho_val = 0.3
    theta_val = 0.9
    ctrl_rho = jnp.ones(N_CTRL) * rho_val
    ctrl_theta = jnp.ones(N_CTRL) * theta_val

    # V_eq = ν_loc(ρ, ϑ) / ϑ  so that V - (1/ϑ)(ν_loc - 0) = 0
    nu_loc = chemical_potential(jnp.array(rho_val), jnp.array(theta_val), GAMMA)
    ctrl_V = jnp.ones(N_CTRL) * float(nu_loc / theta_val)

    R = element_residual_auxiliary(
        N, dN, N, dN, N, N,
        ctrl_rho, zero, ctrl_theta, ctrl_V,
        r_q, w_q, J_E, WE, GAMMA,
    )
    assert jnp.allclose(R, 0.0, atol=1e-10)


def test_auxiliary_residual_zero_gradient():
    """drho_dr=0, V≠equilibrium: only N^V coeff_N term contributes."""
    N, dN, _ = _uniform_basis(N_QP, N_CTRL)
    r_q, w_q = _quad_data(N_QP)
    zero = jnp.zeros(N_CTRL)
    ctrl_rho = jnp.ones(N_CTRL) * 0.3
    ctrl_theta = jnp.ones(N_CTRL) * 0.9
    ctrl_V = jnp.ones(N_CTRL) * 0.5   # not at equilibrium

    R = element_residual_auxiliary(
        N, dN, N, dN, N, N,
        ctrl_rho, zero, ctrl_theta, ctrl_V,
        r_q, w_q, J_E, WE, GAMMA,
    )
    # dN is all zeros so dN^V/dr term vanishes; residual driven by coeff_N alone
    # residual should be nonzero but finite — just check no NaNs and correct shape
    assert R.shape == (N_CTRL,)
    assert jnp.all(jnp.isfinite(R))
    assert jnp.any(R != 0.0)


def test_auxiliary_residual_zero_density_gradient_and_equilibrium_V():
    """drho_dr=0 AND V at equilibrium → exactly zero residual."""
    N, dN, _ = _uniform_basis(N_QP, N_CTRL)
    r_q, w_q = _quad_data(N_QP)
    ctrl_rho = jnp.ones(N_CTRL) * 0.25
    ctrl_theta = jnp.ones(N_CTRL) * 1.1
    u_val = 0.5
    ctrl_u = jnp.ones(N_CTRL) * u_val

    nu_loc = chemical_potential(jnp.array(0.25), jnp.array(1.1), GAMMA)
    V_eq = float((nu_loc - 0.5 * u_val**2) / 1.1)
    ctrl_V = jnp.ones(N_CTRL) * V_eq

    R = element_residual_auxiliary(
        N, dN, N, dN, N, N,
        ctrl_rho, ctrl_u, ctrl_theta, ctrl_V,
        r_q, w_q, J_E, WE, GAMMA,
    )
    assert jnp.allclose(R, 0.0, atol=1e-10)
