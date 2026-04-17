"""Tests for ``phase_field_div`` (exposed as a public helper).

``phase_field_div(∂ρ/∂r, ∂²ρ/∂r², ∂ϑ/∂r, ϑ, r)`` computes the spherical
divergence ∇·(∇ρ/ϑ) written out under spherical symmetry::

    2/(r ϑ) ∂ρ/∂r − (1/ϑ²) (∂ϑ/∂r)(∂ρ/∂r) + (1/ϑ) ∂²ρ/∂r²
"""

import jax.numpy as jnp
import numpy as np

from src.residuals import phase_field_div


def test_phase_field_div_is_exported():
    """Sanity check: the public name imports cleanly."""
    import src.residuals as R

    assert hasattr(R, "phase_field_div")
    # The private alias is preserved for back-compat but should equal the
    # public function object.
    assert R._phase_field_div is R.phase_field_div


def test_constant_vartheta_reduces_to_laplacian():
    """With ϑ = const (∂ϑ/∂r = 0), the divergence reduces to (1/ϑ) Δρ.

    In spherical symmetry  Δρ = ∂²ρ/∂r² + (2/r) ∂ρ/∂r.
    """
    rng = np.random.default_rng(0)
    r = jnp.array(rng.uniform(0.5, 2.0, size=8))
    drho_dr = jnp.array(rng.uniform(-1.0, 1.0, size=8))
    d2rho_dr2 = jnp.array(rng.uniform(-1.0, 1.0, size=8))
    vartheta = jnp.ones(8) * 1.3
    dvartheta_dr = jnp.zeros(8)

    got = phase_field_div(drho_dr, d2rho_dr2, dvartheta_dr, vartheta, r)
    laplacian = d2rho_dr2 + (2.0 / r) * drho_dr
    expected = laplacian / vartheta

    assert jnp.allclose(got, expected, atol=1e-12)


def test_constant_rho_returns_zero():
    """With ρ = const (all radial derivatives of ρ are zero), the divergence vanishes."""
    r = jnp.linspace(0.5, 2.0, 6)
    drho_dr = jnp.zeros_like(r)
    d2rho_dr2 = jnp.zeros_like(r)
    vartheta = 0.7 + 0.3 * jnp.cos(r)  # arbitrary nonzero ϑ profile
    dvartheta_dr = -0.3 * jnp.sin(r)

    got = phase_field_div(drho_dr, d2rho_dr2, dvartheta_dr, vartheta, r)

    assert jnp.allclose(got, 0.0, atol=1e-14)


def test_matches_finite_difference_on_polynomial_field():
    """Compare against an FD evaluation of ∇·(∇ρ/ϑ) on a smooth polynomial.

    Let ρ(r) = a + b r + c r² + d r³ and ϑ(r) = ϑ0 + β r + γ r². Build the
    scalar field f(r) = (1/ϑ) ∂ρ/∂r and evaluate ∇·(f ê_r) in spherical
    symmetry via:  ∇·(f ê_r) = (1/r²) d/dr (r² f).
    """
    a, b, c, d = 0.4, 0.3, -0.2, 0.1
    vt0, beta, gma = 1.2, 0.2, -0.05

    def rho_of(r):
        return a + b * r + c * r**2 + d * r**3

    def drho_of(r):
        return b + 2 * c * r + 3 * d * r**2

    def d2rho_of(r):
        return 2 * c + 6 * d * r

    def vt_of(r):
        return vt0 + beta * r + gma * r**2

    def dvt_of(r):
        return beta + 2 * gma * r

    r = jnp.linspace(0.3, 1.8, 7)
    got = phase_field_div(drho_of(r), d2rho_of(r), dvt_of(r), vt_of(r), r)

    # FD of (1/r²) d/dr [ r² (1/ϑ) ∂ρ/∂r ] using a small step
    eps = 1e-6

    def g(rv):
        return rv**2 * drho_of(rv) / vt_of(rv)

    fd = (g(r + eps) - g(r - eps)) / (2 * eps) / r**2

    assert jnp.allclose(got, fd, atol=1e-7)
