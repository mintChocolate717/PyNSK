"""Tests for the Phase-C solver infrastructure (generalized-α + Newton).

All integrator-level tests deliberately use tiny analytic ODEs so Phase C
can be validated independently of the assembler (Phase B). Tests that would
need the full PDE residual are skipped when ``src.assembler`` is not
importable.
"""

from __future__ import annotations

import os
import tempfile

import jax.numpy as jnp
import numpy as np
import pytest

import src  # noqa: F401  — triggers float64 enable
from src.solver import (
    GenAlphaParams,
    TimeState,
    TimeStepper,
    load_state,
    newton_solve,
    save_state,
    spectrum,
)


# ─────────────────────────────────────────────────────────────────────────────
# GenAlphaParams
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("rho_inf", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_genalpha_parameters_consistency(rho_inf):
    """α_m, α_f, γ must match their analytic formulas exactly."""
    ga = GenAlphaParams(rho_inf=rho_inf)
    expected_am = (2.0 - rho_inf) / (1.0 + rho_inf)
    expected_af = 1.0 / (1.0 + rho_inf)
    expected_gamma = 0.5 + expected_am - expected_af

    assert ga.alpha_m == pytest.approx(expected_am, rel=1e-14)
    assert ga.alpha_f == pytest.approx(expected_af, rel=1e-14)
    assert ga.gamma == pytest.approx(expected_gamma, rel=1e-14)


def test_genalpha_parameters_rejects_out_of_range():
    with pytest.raises(ValueError):
        GenAlphaParams(rho_inf=-0.1)
    with pytest.raises(ValueError):
        GenAlphaParams(rho_inf=1.5)


# ─────────────────────────────────────────────────────────────────────────────
# Integrator accuracy
# ─────────────────────────────────────────────────────────────────────────────


def _linear_decay_error(rho_inf: float, n: int, lam: float = -1.0, T: float = 1.0):
    """Integrate ẏ = λy from y(0)=1 with n steps and return |y_N − exp(λT)|."""

    def residual(d, d_dot, t):
        return d_dot - lam * d

    ga = GenAlphaParams(rho_inf=rho_inf)
    dt = T / n
    stepper = TimeStepper(residual, None, None, ga, dt)
    state = TimeState(d=jnp.array([1.0]), d_dot=jnp.array([lam]), t=0.0)
    hist = stepper.run(state, n)
    y_num = float(hist[-1].d[0])
    return abs(y_num - float(jnp.exp(lam * T)))


def test_genalpha_second_order_accuracy():
    """Global error on ẏ = λy should halve by ~4 under dt-refinement."""
    ns = [20, 40, 80, 160, 320]
    errs = [_linear_decay_error(rho_inf=0.5, n=n) for n in ns]

    # Successive error ratios should approach 4 as dt → 0.
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    # Asymptotic ratios: use the last three refinements
    for r in ratios[-3:]:
        assert 3.5 < r < 4.5, f"expected ~4 (2nd-order), got {r}"


def test_genalpha_high_frequency_damping():
    """With ρ_∞ = 0 the fast mode is damped much more than with ρ_∞ = 1."""
    # Two uncoupled modes: slow (λ = -1) and very-fast (λ = -1000).
    # At dt·|λ| = 100 the midpoint rule (ρ_∞=1) has |A| ≈ 0.96 so the fast
    # mode barely decays; ρ_∞=0 achieves asymptotic annihilation (|A|≪1).
    L = jnp.array([[-1.0, 0.0], [0.0, -1000.0]])

    def residual(d, d_dot, t):
        return d_dot - L @ d

    y0 = jnp.array([1.0, 1.0])
    ydot0 = L @ y0
    T = 1.0
    dt = 0.1

    results = {}
    for rho in (1.0, 0.0):
        ga = GenAlphaParams(rho_inf=rho)
        stepper = TimeStepper(residual, None, None, ga, dt)
        state = TimeState(d=y0, d_dot=ydot0, t=0.0)
        hist = stepper.run(state, int(T / dt))
        results[rho] = np.abs(np.asarray(hist[-1].d))

    # Slow-mode accuracy: both should be within 2% of exp(-1).
    exact_slow = float(jnp.exp(-1.0))
    assert abs(results[1.0][0] - exact_slow) / exact_slow < 0.02
    assert abs(results[0.0][0] - exact_slow) / exact_slow < 0.02

    # Fast-mode: ρ_∞=0 residual must be orders of magnitude smaller
    # than the ρ_∞=1 (midpoint) residual, which is barely damped.
    assert results[0.0][1] < 1e-3 * results[1.0][1]


# ─────────────────────────────────────────────────────────────────────────────
# Newton
# ─────────────────────────────────────────────────────────────────────────────


def test_newton_linear_one_iteration():
    """Linear residual converges in exactly one Newton iteration."""
    A = jnp.array([[2.0, 1.0], [1.0, 3.0]])
    b = jnp.array([5.0, 7.0])

    def R(x):
        return A @ x - b

    x0 = jnp.array([0.0, 0.0])
    x, _, iters, hist = newton_solve(R, x0, x0, tol=1e-10)
    assert iters == 1
    assert len(hist) >= 2
    assert hist[-1] < 1e-10
    expected = jnp.linalg.solve(A, b)
    assert jnp.allclose(x, expected, atol=1e-10)


def test_newton_quadratic_convergence():
    """Newton on y² − 2 = 0 shows quadratic convergence."""
    def R(y):
        return y**2 - 2.0

    y0 = jnp.array([1.0])
    y, _, iters, hist = newton_solve(R, y0, y0, tol=1e-12)

    # Solution found.
    assert abs(float(y[0]) - float(jnp.sqrt(2.0))) < 1e-10

    # Quadratic convergence: r_{k+1} / r_k^2 bounded.
    # Only inspect the range where both residuals are well above round-off.
    ratios = []
    for i in range(1, len(hist) - 1):
        if hist[i] > 1e-8 and hist[i + 1] > 1e-14:
            ratios.append(hist[i + 1] / hist[i] ** 2)
    # For y²−2 at y*=√2, the quadratic constant is 1/(2√2) ≈ 0.354.
    assert len(ratios) >= 1
    for r in ratios:
        assert r < 1.0, f"not quadratic: ratio = {r}"


def test_newton_backtracking_activates():
    """Poor initial guess triggers backtracking without divergence."""
    # R(x) = x^3 - 1; far-from-root start.
    def R(x):
        return x**3 - 1.0

    x, _, iters, hist = newton_solve(
        R, jnp.array([10.0]), jnp.array([0.0]), tol=1e-10, max_iter=40
    )
    assert abs(float(x[0]) - 1.0) < 1e-8
    # Residual must be monotone non-increasing (up to numerical noise)
    # thanks to the backtracking line-search.
    for i in range(len(hist) - 1):
        assert hist[i + 1] <= hist[i] + 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint / restart
# ─────────────────────────────────────────────────────────────────────────────


def test_checkpoint_roundtrip():
    state = TimeState(
        d=jnp.arange(10, dtype=jnp.float64),
        d_dot=jnp.linspace(-1.0, 1.0, 10),
        t=0.25,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt.npz")
        save_state(state, path, dt=0.01, seed=42)

        loaded, dt, seed = load_state(path)

    assert dt == pytest.approx(0.01)
    assert seed == 42
    assert loaded.t == pytest.approx(0.25)
    assert jnp.allclose(loaded.d, state.d)
    assert jnp.allclose(loaded.d_dot, state.d_dot)


# ─────────────────────────────────────────────────────────────────────────────
# Eigen-diagnostics
# ─────────────────────────────────────────────────────────────────────────────


def test_spectrum_shape():
    """spectrum(...) returns the requested number of (complex) eigenvalues."""
    # Build a residual whose tangent is a diagonal matrix with known eigs.
    diag = jnp.array([10.0, -3.0, 0.5, 2.0, -7.0, 1.0, 0.1, -0.2])

    def R(x):
        return diag * x

    d = jnp.zeros_like(diag)
    d_dot = jnp.zeros_like(diag)

    k = 4
    eigs = spectrum(R, d, d_dot, k=k)
    assert eigs.shape == (k,)
    # jnp.linalg.eigvals returns complex — explicit dtype check.
    assert jnp.iscomplexobj(eigs)

    # Largest-magnitude eigs of diag(...) are ±10 and ±7.
    mags = jnp.sort(jnp.abs(eigs))[::-1]
    assert float(mags[0]) == pytest.approx(10.0, rel=1e-10)
    assert float(mags[1]) == pytest.approx(7.0, rel=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# TimeStepper bookkeeping
# ─────────────────────────────────────────────────────────────────────────────


def test_step_history_records_newton_iterations():
    """Every step appends a record with iteration count and residual norms."""

    def residual(d, d_dot, t):
        return d_dot + d  # ẏ = -y

    ga = GenAlphaParams(rho_inf=0.5)
    stepper = TimeStepper(residual, None, None, ga, dt=0.1)
    state = TimeState(d=jnp.array([1.0]), d_dot=jnp.array([-1.0]), t=0.0)

    n = 5
    stepper.run(state, n)
    assert len(stepper.step_history) == n
    for rec in stepper.step_history:
        assert "newton_iters" in rec
        assert "residual_norms" in rec
        assert rec["residual_norms"][-1] < 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# Optional: full PDE wiring (skipped if Phase B not merged)
# ─────────────────────────────────────────────────────────────────────────────


def test_stepper_wires_to_assembler():
    """If Phase B's assembler is present, the stepper must accept its contract."""
    try:
        from src import assembler  # noqa: F401
    except ImportError:
        pytest.skip("Phase B assembler not merged in this worktree")

    # Deliberately light: confirm the symbol exists and is callable.
    assert callable(getattr(assembler, "assemble_residual", None))
    assert callable(getattr(assembler, "build_basis_cache", None))
    assert callable(getattr(assembler, "apply_dirichlet", None))
