"""Tests for src/assembler.py — IEN, basis cache, global residual, BC helpers."""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.assembler import (  # noqa: E402
    apply_dirichlet,
    assemble_residual,
    build_basis_cache,
    dirichlet_far_field,
    element_connectivity,
    symmetry_bc_at_origin,
)
from src.bsplines import make_knot_vector  # noqa: E402
from src.constitutive import chemical_potential  # noqa: E402
from src.residuals import (  # noqa: E402
    element_residual_auxiliary,
    element_residual_energy,
    element_residual_mass,
    element_residual_momentum,
)

# ── shared fixtures ────────────────────────────────────────────────────────

R_MAX = 4.0
DEGREE = 2
N_GAUSS = 3
N_ELEM = 4                  # 4 elements
N_CTRL = N_ELEM + DEGREE    # 6 control points

PARAMS = {
    "Re": 10.0,
    "We": 100.0,
    "Pr": 7.0,
    "gamma": 1.4,
    "b_r": 0.0,
    "r_s": 0.0,
}


def _make_cache():
    knots = make_knot_vector(N_CTRL, DEGREE)
    return build_basis_cache(knots, DEGREE, N_GAUSS, R_MAX)


def _zeros_ctrl(n_ctrl=N_CTRL):
    return jnp.zeros(n_ctrl, dtype=jnp.float64)


# ── IEN ────────────────────────────────────────────────────────────────────

def test_ien_shape():
    IEN = element_connectivity(N_CTRL, DEGREE)
    assert IEN.shape == (N_ELEM, DEGREE + 1)


def test_ien_overlap():
    """Consecutive elements share `degree` control points."""
    IEN = np.asarray(element_connectivity(N_CTRL, DEGREE))
    for e in range(N_ELEM - 1):
        shared = set(IEN[e]).intersection(IEN[e + 1])
        assert len(shared) == DEGREE, (
            f"Elements {e} and {e+1} share {len(shared)} DOFs, expected {DEGREE}"
        )


def test_ien_monotone_and_bounds():
    IEN = np.asarray(element_connectivity(N_CTRL, DEGREE))
    assert IEN.min() == 0
    assert IEN.max() == N_CTRL - 1
    # Each row strictly increasing
    assert np.all(np.diff(IEN, axis=1) == 1)


# ── Basis cache ────────────────────────────────────────────────────────────

def test_basis_cache_shapes():
    cache = _make_cache()
    p1 = DEGREE + 1
    assert cache["xi"].shape == (N_ELEM, N_GAUSS)
    assert cache["r"].shape == (N_ELEM, N_GAUSS)
    assert cache["w"].shape == (N_ELEM, N_GAUSS)
    assert cache["J"].shape == (N_ELEM,)
    assert cache["N"].shape == (N_ELEM, N_GAUSS, p1)
    assert cache["dN"].shape == (N_ELEM, N_GAUSS, p1)
    assert cache["d2N"].shape == (N_ELEM, N_GAUSS, p1)
    assert cache["IEN"].shape == (N_ELEM, p1)


def test_basis_cache_partition_of_unity():
    """Sum of the (p+1) non-zero basis functions at each quadrature point is 1."""
    cache = _make_cache()
    assert np.allclose(np.asarray(cache["N"]).sum(axis=-1), 1.0, atol=1e-12)


def test_basis_cache_r_range():
    cache = _make_cache()
    r = np.asarray(cache["r"])
    assert r.min() >= 0.0
    assert r.max() <= R_MAX


# ── Single-element assembly equals element residual ────────────────────────

def test_single_element_mesh_equals_element_residual():
    """With 1 element, assembled global residual equals the element residual
    for each block (up to ordering)."""
    degree = 2
    n_elem = 1
    n_ctrl = n_elem + degree
    knots = make_knot_vector(n_ctrl, degree)
    cache = build_basis_cache(knots, degree, 4, R_MAX)

    rng = np.random.default_rng(0)
    # Build physically-reasonable state: rho in (0, 1), vartheta > 0
    ctrl = {
        "rho": jnp.asarray(0.3 + 0.05 * rng.standard_normal(n_ctrl)),
        "u": jnp.asarray(0.1 * rng.standard_normal(n_ctrl)),
        "vartheta": jnp.asarray(1.0 + 0.05 * rng.standard_normal(n_ctrl)),
        "V": jnp.asarray(0.1 * rng.standard_normal(n_ctrl)),
    }
    ctrl_dot = {
        "rho": jnp.asarray(0.01 * rng.standard_normal(n_ctrl)),
        "u": jnp.asarray(0.01 * rng.standard_normal(n_ctrl)),
        "vartheta": jnp.asarray(0.01 * rng.standard_normal(n_ctrl)),
    }

    R = assemble_residual(ctrl, ctrl_dot, cache, PARAMS)
    assert R.shape == (4 * n_ctrl,)

    # Manual single-element computation using the raw quadrature data.
    N_e = cache["N"][0]
    dN_e = cache["dN"][0]
    d2N_e = cache["d2N"][0]
    r_e = cache["r"][0]
    w_e = cache["w"][0] / cache["J"][0]       # unfold the pre-folded Jacobian
    J_e = cache["J"][0]
    IEN_e = cache["IEN"][0]

    d_rho = ctrl["rho"][IEN_e]
    d_u = ctrl["u"][IEN_e]
    d_vth = ctrl["vartheta"][IEN_e]
    d_V = ctrl["V"][IEN_e]
    d_rho_dot = ctrl_dot["rho"][IEN_e]
    d_u_dot = ctrl_dot["u"][IEN_e]
    d_vth_dot = ctrl_dot["vartheta"][IEN_e]

    R_rho_e = element_residual_mass(
        N_e, dN_e, N_e, d_rho, d_rho_dot, d_u, r_e, w_e, J_e
    )
    R_u_e = element_residual_momentum(
        N_e, dN_e, N_e, dN_e, d2N_e, N_e, dN_e, N_e,
        d_rho, d_rho_dot, d_u, d_u_dot, d_vth, d_V,
        r_e, w_e, J_e,
        PARAMS["Re"], PARAMS["We"], PARAMS["gamma"], PARAMS["b_r"],
    )
    R_vth_e = element_residual_energy(
        N_e, dN_e, N_e, dN_e, d2N_e, N_e, dN_e, N_e,
        d_rho, d_rho_dot, d_u, d_u_dot, d_vth, d_vth_dot, d_V,
        r_e, w_e, J_e,
        PARAMS["Re"], PARAMS["We"], PARAMS["gamma"],
        PARAMS["Pr"], PARAMS["b_r"], PARAMS["r_s"],
    )
    R_V_e = element_residual_auxiliary(
        N_e, dN_e, N_e, dN_e, N_e, N_e,
        d_rho, d_u, d_vth, d_V,
        r_e, w_e, J_e, PARAMS["We"], PARAMS["gamma"],
    )

    assert np.allclose(np.asarray(R[:n_ctrl]), np.asarray(R_rho_e), atol=1e-12)
    assert np.allclose(np.asarray(R[n_ctrl : 2 * n_ctrl]), np.asarray(R_u_e), atol=1e-12)
    assert np.allclose(np.asarray(R[2 * n_ctrl : 3 * n_ctrl]), np.asarray(R_vth_e), atol=1e-12)
    assert np.allclose(np.asarray(R[3 * n_ctrl : 4 * n_ctrl]), np.asarray(R_V_e), atol=1e-12)


# ── Time-derivative linearity ──────────────────────────────────────────────

def test_assembly_linearity():
    """Freeze the state; the residual is affine in ctrl_dot.

        R(x, a*c1 + b*c2) == a*R(x, c1) + b*R(x, c2) + (1 - a - b) * R(x, 0)

    (affine because the steady-state part R(x, 0) is a constant offset.)
    """
    cache = _make_cache()
    rng = np.random.default_rng(1)

    ctrl = {
        "rho": jnp.asarray(0.3 * np.ones(N_CTRL)),
        "u": jnp.asarray(0.05 * rng.standard_normal(N_CTRL)),
        "vartheta": jnp.asarray(1.0 * np.ones(N_CTRL)),
        "V": jnp.asarray(0.1 * rng.standard_normal(N_CTRL)),
    }

    def make_dot(seed):
        r = np.random.default_rng(seed)
        return {
            "rho": jnp.asarray(r.standard_normal(N_CTRL)),
            "u": jnp.asarray(r.standard_normal(N_CTRL)),
            "vartheta": jnp.asarray(r.standard_normal(N_CTRL)),
        }

    zero_dot = {
        "rho": _zeros_ctrl(),
        "u": _zeros_ctrl(),
        "vartheta": _zeros_ctrl(),
    }
    c1 = make_dot(10)
    c2 = make_dot(11)

    a, b = 0.7, -0.4
    combo = {k: a * c1[k] + b * c2[k] for k in c1}

    R0 = assemble_residual(ctrl, zero_dot, cache, PARAMS)
    R1 = assemble_residual(ctrl, c1, cache, PARAMS)
    R2 = assemble_residual(ctrl, c2, cache, PARAMS)
    R_combo = assemble_residual(ctrl, combo, cache, PARAMS)

    # Linearity of the time-derivative contribution:
    expected = a * (R1 - R0) + b * (R2 - R0) + R0
    assert np.allclose(np.asarray(R_combo), np.asarray(expected), atol=1e-10)


# ── Equilibrium: zero flow + uniform fields + no sources → R = 0 ───────────

def test_assembly_zero_on_equilibrium():
    """Zero velocity, uniform ρ and ϑ, V at chemical equilibrium, no body
    force, no heat source, all time derivatives zero.

    In 1D spherical geometry, a uniform state produces a *boundary*
    contribution at r=R_max for the momentum block equal to
    ``-ρ ν_loc R_max²`` — this is the natural pressure force that the
    far-field Dirichlet BC (``u=0``) absorbs.  On every *interior* DOF
    and in every other field block the residual must vanish identically.
    """
    cache = _make_cache()

    rho_val = 0.3
    vth_val = 1.0
    gamma = PARAMS["gamma"]

    nu_loc = float(chemical_potential(jnp.array(rho_val), jnp.array(vth_val), gamma))
    V_eq = nu_loc / vth_val

    ctrl = {
        "rho": rho_val * jnp.ones(N_CTRL),
        "u": _zeros_ctrl(),
        "vartheta": vth_val * jnp.ones(N_CTRL),
        "V": V_eq * jnp.ones(N_CTRL),
    }
    ctrl_dot = {
        "rho": _zeros_ctrl(),
        "u": _zeros_ctrl(),
        "vartheta": _zeros_ctrl(),
    }

    R = assemble_residual(ctrl, ctrl_dot, cache, PARAMS)
    assert R.shape == (4 * N_CTRL,)

    # ρ, ϑ, V blocks are identically zero.
    R_np = np.asarray(R)
    assert np.allclose(R_np[0:N_CTRL], 0.0, atol=1e-10)                  # ρ
    assert np.allclose(R_np[2 * N_CTRL : 3 * N_CTRL], 0.0, atol=1e-10)    # ϑ
    assert np.allclose(R_np[3 * N_CTRL : 4 * N_CTRL], 0.0, atol=1e-10)    # V

    # u block: only the last DOF carries the far-field pressure term.
    R_u = R_np[N_CTRL : 2 * N_CTRL]
    assert np.allclose(R_u[:-1], 0.0, atol=1e-10)
    expected_boundary = -rho_val * nu_loc * R_MAX**2
    assert R_u[-1] == pytest.approx(expected_boundary, rel=1e-10)

    # After applying far-field Dirichlet with u_∞=0, the residual at the
    # last u-DOF is overwritten by (current - value) = 0, and the whole
    # residual is then zero to float64 precision.
    dof_idx, values = dirichlet_far_field(N_CTRL, rho_val, vth_val, u_inf=0.0)
    current = jnp.concatenate([ctrl["rho"], ctrl["u"], ctrl["vartheta"], ctrl["V"]])
    R_bc, _ = apply_dirichlet(R, None, dof_idx, values, current)
    assert np.allclose(np.asarray(R_bc), 0.0, atol=1e-10)


# ── Dirichlet BC application ──────────────────────────────────────────────

def test_dirichlet_application():
    """After `apply_dirichlet`, a Newton step δ = -K^{-1} R drives
    constrained DOFs exactly to the prescribed values."""
    n = 8
    rng = np.random.default_rng(2)
    K = jnp.asarray(rng.standard_normal((n, n)) + n * np.eye(n))
    R = jnp.asarray(rng.standard_normal(n))
    current = jnp.asarray(rng.standard_normal(n))

    dof_idx = jnp.asarray([1, 4, 6], dtype=jnp.int32)
    values = jnp.asarray([3.14, -1.0, 0.5])

    R_mod, K_mod = apply_dirichlet(R, K, dof_idx, values, current)

    # Rows and columns at constrained DOFs zeroed out, diagonal = 1.
    Km = np.asarray(K_mod)
    for i in np.asarray(dof_idx):
        assert np.allclose(Km[i, :], np.eye(n)[i]), f"row {i} not unit"
        assert np.allclose(Km[:, i], np.eye(n)[i]), f"col {i} not unit"

    # Residual entry: R_i = current_i - values_i → Newton δ_i = values_i - current_i
    Rm = np.asarray(R_mod)
    assert np.allclose(Rm[np.asarray(dof_idx)],
                       np.asarray(current)[np.asarray(dof_idx)] - np.asarray(values))

    # Solve the modified system and confirm values are hit.
    delta = np.linalg.solve(Km, -Rm)
    new_vals = np.asarray(current) + delta
    assert np.allclose(new_vals[np.asarray(dof_idx)], np.asarray(values), atol=1e-12)


def test_dirichlet_residual_only():
    """When K is None only the residual is modified."""
    n = 5
    R = jnp.asarray(np.arange(n, dtype=np.float64))
    current = jnp.asarray(np.array([0.0, 2.0, 3.0, 4.0, 5.0]))
    dof_idx = jnp.asarray([2], dtype=jnp.int32)
    values = jnp.asarray([1.0])

    R_mod, K_mod = apply_dirichlet(R, None, dof_idx, values, current)
    assert K_mod is None
    # R[2] should be current[2] - values[0] = 3.0 - 1.0 = 2.0
    assert float(R_mod[2]) == pytest.approx(2.0)
    # Untouched entries preserved.
    assert float(R_mod[0]) == 0.0
    assert float(R_mod[4]) == 4.0


# ── Symmetry & far-field helpers ──────────────────────────────────────────

def test_symmetry_bc_at_origin():
    """`symmetry_bc_at_origin` returns the u-block DOF index at r=0 only.

    Applying it makes a Newton step pin u[0]=0.
    """
    dof_idx, values = symmetry_bc_at_origin(N_CTRL)
    # Exactly one constraint — the first u-DOF.
    assert dof_idx.shape == (1,)
    assert int(dof_idx[0]) == N_CTRL   # start of the u block
    assert float(values[0]) == 0.0

    # Smoke-test that apply_dirichlet plays nicely.
    n_dof = 4 * N_CTRL
    R = jnp.ones(n_dof)
    K = jnp.eye(n_dof) * 2.0 + 0.1
    current = jnp.zeros(n_dof).at[N_CTRL].set(0.7)   # u[0] currently 0.7
    R_mod, K_mod = apply_dirichlet(R, K, dof_idx, values, current)

    delta = np.linalg.solve(np.asarray(K_mod), -np.asarray(R_mod))
    new_state = np.asarray(current) + delta
    assert abs(new_state[N_CTRL]) < 1e-12


def test_dirichlet_far_field():
    rho_inf = 0.05
    vth_inf = 1.2
    dof_idx, values = dirichlet_far_field(N_CTRL, rho_inf, vth_inf, u_inf=0.0)
    assert dof_idx.shape == (3,)
    last = N_CTRL - 1
    idx = np.asarray(dof_idx)
    vals = np.asarray(values)
    # ρ block last DOF
    assert last in idx
    # u block last DOF
    assert (N_CTRL + last) in idx
    # vartheta block last DOF
    assert (2 * N_CTRL + last) in idx
    # Values correspond
    loc_rho = int(np.where(idx == last)[0][0])
    loc_u = int(np.where(idx == N_CTRL + last)[0][0])
    loc_vth = int(np.where(idx == 2 * N_CTRL + last)[0][0])
    assert vals[loc_rho] == pytest.approx(rho_inf)
    assert vals[loc_u] == pytest.approx(0.0)
    assert vals[loc_vth] == pytest.approx(vth_inf)
