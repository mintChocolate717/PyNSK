"""Tests for Gaussian quadrature (src/quadrature.py)."""

import numpy as np
import pytest

from src.bsplines import make_knot_vector
from src.quadrature import gauss_legendre, quadrature_points, recommended_n_gauss

# ── gauss_legendre ────────────────────────────────────────────────────────────


def test_gauss_legendre_point_count():
    for n in [1, 2, 3, 5]:
        xi, w = gauss_legendre(n)
        assert len(xi) == n
        assert len(w) == n


def test_gauss_legendre_weights_sum_to_two():
    """Weights must sum to 2 (length of [-1, 1])."""
    for n in [1, 2, 3, 4, 5]:
        _, w = gauss_legendre(n)
        assert np.isclose(float(w.sum()), 2.0, atol=1e-14)


@pytest.mark.parametrize("poly_degree", [1, 3, 5, 7])
def test_gauss_legendre_exact_integration(poly_degree):
    """n points integrates polynomials of degree 2n-1 exactly."""
    n = (poly_degree + 1) // 2 + 1
    xi, w = gauss_legendre(n)
    # Integrate x^poly_degree from -1 to 1
    numerical = float((xi**poly_degree * w).sum())
    analytical = 0.0 if poly_degree % 2 == 1 else 2.0 / (poly_degree + 1)
    assert np.isclose(numerical, analytical, atol=1e-12)


# ── quadrature_points ─────────────────────────────────────────────────────────


def test_quadrature_point_count():
    n_ctrl, degree, n_gauss = 6, 2, 3
    t = make_knot_vector(n_ctrl, degree)
    n_elements = len(np.unique(np.asarray(t))) - 1
    xi, r, w = quadrature_points(t, degree, n_gauss, R_max=1.0)
    assert len(xi) == n_elements * n_gauss
    assert len(r) == n_elements * n_gauss
    assert len(w) == n_elements * n_gauss


def test_quadrature_pts_in_domain():
    """All quadrature points lie within [0, R_max]."""
    R_max = 5.0
    t = make_knot_vector(8, 3)
    xi, r, w = quadrature_points(t, 3, 4, R_max=R_max)
    assert (np.asarray(r) >= 0.0).all()
    assert (np.asarray(r) <= R_max + 1e-12).all()
    assert (np.asarray(xi) >= 0.0).all()
    assert (np.asarray(xi) <= 1.0 + 1e-12).all()


def test_r_equals_R_max_times_xi():
    """Physical coordinate r = R_max * xi exactly."""
    R_max = 3.7
    t = make_knot_vector(7, 3)
    xi, r, w = quadrature_points(t, 3, 3, R_max=R_max)
    assert np.allclose(np.asarray(r), R_max * np.asarray(xi), atol=1e-14)


def test_integrate_constant_gives_R_max():
    """∫_0^{R_max} dr = R_max (integrating f=1, no r² factor)."""
    R_max = 2.5
    t = make_knot_vector(6, 2)
    _, _, w = quadrature_points(t, 2, 3, R_max=R_max)
    assert np.isclose(float(w.sum()), R_max, atol=1e-12)


def test_integrate_r_squared():
    """∫_0^{R_max} r² dr = R_max³ / 3 (the spherical volume factor)."""
    R_max = 4.0
    t = make_knot_vector(8, 3)
    _, r, w = quadrature_points(t, 3, 4, R_max=R_max)
    result = float((r**2 * w).sum())
    assert np.isclose(result, R_max**3 / 3.0, atol=1e-10)


def test_integrate_polynomial_exact():
    """Quadrature integrates polynomials up to degree 2*n_gauss-1 exactly per element."""
    R_max = 1.0
    n_gauss = 4
    t = make_knot_vector(6, 3)
    _, r, w = quadrature_points(t, 3, n_gauss, R_max=R_max)
    # Integrate r^3 from 0 to R_max = R_max^4 / 4
    result = float((r**3 * w).sum())
    assert np.isclose(result, R_max**4 / 4.0, atol=1e-12)


def test_recommended_n_gauss():
    assert recommended_n_gauss(2) == 3
    assert recommended_n_gauss(3) == 4
    assert recommended_n_gauss(4) == 5
