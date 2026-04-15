"""Tests for B-spline basis functions (src/bsplines.py).

Each test corresponds to a mathematical property that must hold for
any correct B-spline implementation.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from src.bsplines import basis_deriv_matrix, basis_matrix, make_knot_vector

# ── Knot vector ──────────────────────────────────────────────────────────────

def test_knot_vector_length():
    for n, p in [(5, 2), (10, 3), (4, 3), (6, 4)]:
        t = make_knot_vector(n, p)
        assert len(t) == n + p + 1, f"Failed for n={n}, p={p}"


def test_knot_vector_open_ends():
    for p in [2, 3, 4]:
        t = np.asarray(make_knot_vector(p + 3, p))
        assert np.allclose(t[:p+1], 0.0), "Left end not repeated correctly"
        assert np.allclose(t[-(p+1):], 1.0), "Right end not repeated correctly"


def test_knot_vector_interior_uniform():
    t = np.asarray(make_knot_vector(7, 2))  # 4 interior knots
    interior = t[3:-3]
    diffs = np.diff(interior)
    assert np.allclose(diffs, diffs[0]), "Interior knots not uniformly spaced"


def test_knot_vector_invalid_raises():
    with pytest.raises(ValueError):
        make_knot_vector(2, 3)  # n_ctrl < degree + 1


# ── Partition of unity ────────────────────────────────────────────────────────

@pytest.mark.parametrize("degree", [2, 3, 4])
def test_partition_of_unity(degree):
    """Sum of all basis functions equals 1 everywhere in [0, 1]."""
    n_ctrl = degree + 5
    t = make_knot_vector(n_ctrl, degree)
    xi = jnp.linspace(0.0, 1.0, 100)
    N = basis_matrix(xi, t, degree)
    assert np.allclose(np.asarray(N.sum(axis=1)), 1.0, atol=1e-12), \
        f"Partition of unity failed for degree {degree}"


# ── Non-negativity ────────────────────────────────────────────────────────────

def test_basis_nonnegative():
    """B-spline basis functions are non-negative."""
    t = make_knot_vector(8, 3)
    xi = jnp.linspace(0.0, 1.0, 200)
    N = basis_matrix(xi, t, 3)
    assert (np.asarray(N) >= -1e-14).all(), "Basis function went negative"


# ── Endpoint interpolation ────────────────────────────────────────────────────

def test_endpoint_interpolation():
    """Open B-splines interpolate the first and last control points."""
    for degree in [2, 3, 4]:
        n_ctrl = degree + 4
        t = make_knot_vector(n_ctrl, degree)
        N_left  = basis_matrix(jnp.array([0.0]), t, degree)
        N_right = basis_matrix(jnp.array([1.0]), t, degree)
        assert abs(float(N_left[0, 0])  - 1.0) < 1e-12, "Left endpoint not interpolated"
        assert abs(float(N_right[0, -1]) - 1.0) < 1e-12, "Right endpoint not interpolated"


# ── Derivatives ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("order", [1, 2])
def test_derivative_vs_finite_difference(order):
    """Analytical derivative matches central finite difference."""
    n_ctrl, degree = 8, 3
    t = make_knot_vector(n_ctrl, degree)
    # Evaluate at element midpoints — not at knot locations — to avoid
    # FD stepping across a knot boundary and picking up leakage.
    xi0 = jnp.array([0.1, 0.3, 0.5, 0.7])
    eps = 1e-5
    N_p = basis_deriv_matrix(xi0 + eps, t, degree, order=order - 1)
    N_m = basis_deriv_matrix(xi0 - eps, t, degree, order=order - 1)
    dN_fd = (N_p - N_m) / (2 * eps)
    dN    = basis_deriv_matrix(xi0, t, degree, order=order)
    assert np.allclose(np.asarray(dN), np.asarray(dN_fd), atol=1e-4), \
        f"Order-{order} derivative failed finite difference test"


def test_order_zero_equals_basis():
    """order=0 derivative returns the basis matrix unchanged."""
    n_ctrl, degree = 6, 2
    t = make_knot_vector(n_ctrl, degree)
    xi = jnp.linspace(0.0, 1.0, 30)
    assert np.allclose(
        np.asarray(basis_matrix(xi, t, degree)),
        np.asarray(basis_deriv_matrix(xi, t, degree, order=0))
    )


def test_derivative_order_exceeds_degree_is_zero():
    """Derivative of order > degree is identically zero."""
    n_ctrl, degree = 5, 2
    t = make_knot_vector(n_ctrl, degree)
    xi = jnp.linspace(0.0, 1.0, 20)
    d3N = basis_deriv_matrix(xi, t, degree, order=3)
    assert np.allclose(np.asarray(d3N), 0.0)
