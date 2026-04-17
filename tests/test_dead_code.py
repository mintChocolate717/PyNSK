"""Direct tests for ``free_energy_loc`` and ``total_energy``.

These helpers are not (yet) called by any residual but are required for
post-processing and diagnostics (Phase D). They stay in the public API.
"""

import jax
import jax.numpy as jnp

from src.constitutive import (
    chemical_potential,
    free_energy_loc,
    internal_energy_loc,
    pressure,
    total_energy,
)

# ── free_energy_loc ───────────────────────────────────────────────────────────


def test_free_energy_at_critical_point():
    """ψ_loc(ρ_c, ϑ_c) with (ρ_c = 1/3, ϑ_c = 1, γ = 1.4).

    At the critical point: ψ_loc = -ρ + (8ϑ/27) log(ρ/(1-ρ))
                                   - (8ϑ/(27(γ-1))) log(ϑ)
                                   + 8ϑ/(27(γ-1))
    With ϑ = 1, log(1) = 0, so the logarithm-ϑ terms drop and the
    constant stays. We compute the expected value analytically.
    """
    rho = 1.0 / 3.0
    vartheta = 1.0
    gamma = 1.4
    psi = free_energy_loc(rho, vartheta, gamma)

    # log(ρ/(1-ρ)) = log(1/2) = -log 2
    expected = (
        -rho
        + (8.0 * vartheta / 27.0) * jnp.log(rho / (1.0 - rho))
        + 8.0 * vartheta / (27.0 * (gamma - 1.0))
    )
    assert jnp.allclose(psi, expected, atol=1e-12)


def test_free_energy_chemical_potential_consistency():
    """ν_loc = ∂(ρ ψ_loc)/∂ρ  (thermodynamic identity).

    Note: in the van der Waals reduced form used here, the chemical
    potential equals ∂ψ_loc/∂ρ  + ψ_loc/ρ. We verify via autodiff.
    """
    rho = 0.4
    vartheta = 1.2
    gamma = 1.4

    def rho_psi(r):
        return r * free_energy_loc(r, vartheta, gamma)

    dpsi_rho = jax.grad(rho_psi)(rho)
    nu = chemical_potential(rho, vartheta, gamma)
    assert jnp.allclose(dpsi_rho, nu, atol=1e-10)


def test_free_energy_ideal_gas_limit():
    """ρ → 0: ψ_loc → -ρ + (8ϑ/27) log ρ + (purely-ϑ terms).

    At very small density the (1 − ρ) factor is essentially 1, so
    log(ρ/(1−ρ)) ≈ log ρ and we recover the ideal-gas-type form.
    """
    rho = 1e-5
    vartheta = 1.0
    gamma = 1.4
    psi = free_energy_loc(rho, vartheta, gamma)

    # Expected via the exact formula (no approximations on our side).
    expected = (
        -rho
        + (8.0 * vartheta / 27.0) * jnp.log(rho / (1.0 - rho))
        - (8.0 * vartheta / (27.0 * (gamma - 1.0))) * jnp.log(vartheta)
        + 8.0 * vartheta / (27.0 * (gamma - 1.0))
    )
    assert jnp.allclose(psi, expected, atol=1e-12)

    # Sanity: rho -> 0 limit has the dominant -|const|·log(1/rho) behaviour
    # so ψ_loc goes to -∞.
    assert psi < free_energy_loc(0.1, vartheta, gamma)


def test_free_energy_pressure_maxwell_identity():
    """p = ρ² ∂ψ_loc/∂ρ (Maxwell thermodynamic identity, isothermal).

    The van der Waals pressure here must match this derivative identity.
    """
    rho = 0.25
    vartheta = 1.1
    gamma = 1.4

    dpsi_drho = jax.grad(free_energy_loc, argnums=0)(rho, vartheta, gamma)
    p_expected = rho**2 * dpsi_drho
    p_got = pressure(rho, vartheta, gamma)
    assert jnp.allclose(p_got, p_expected, atol=1e-10)


# ── total_energy ──────────────────────────────────────────────────────────────


def test_total_energy_equals_components_sum():
    """ρE = ι_loc + (∂ρ/∂r)²/(2 We ρ) · ρ (really just plus the gradient
    term divided ρ out) + ½ u² — verify the definition end-to-end.
    """
    rho = 0.3
    vartheta = 1.0
    drho_dr = 0.5
    u_r = 0.2
    gamma = 1.4
    We = 50.0

    tot = total_energy(rho, vartheta, drho_dr, u_r, gamma, We)
    iota = internal_energy_loc(rho, vartheta, gamma)
    grad = drho_dr**2 / (2.0 * We * rho)
    kinetic = 0.5 * u_r**2
    assert jnp.allclose(tot, iota + grad + kinetic, atol=1e-12)


def test_total_energy_ideal_gas_no_gradient_no_velocity():
    """With ∂ρ/∂r = 0 and u = 0 the total energy collapses to ι_loc."""
    rho = 0.2
    vartheta = 1.5
    gamma = 1.4
    We = 10.0
    tot = total_energy(rho, vartheta, 0.0, 0.0, gamma, We)
    assert jnp.allclose(tot, internal_energy_loc(rho, vartheta, gamma), atol=1e-12)


def test_total_energy_gradient_term_sign():
    """Gradient energy is always non-negative (capillary stability)."""
    rho = 0.3
    vartheta = 1.0
    gamma = 1.4
    We = 100.0
    for grad in [-0.9, -0.1, 0.0, 0.3, 1.4]:
        tot_with = total_energy(rho, vartheta, grad, 0.0, gamma, We)
        tot_base = internal_energy_loc(rho, vartheta, gamma)
        assert tot_with >= tot_base - 1e-14


def test_total_energy_kinetic_scaling():
    """Kinetic contribution scales quadratically with velocity."""
    rho = 0.25
    vartheta = 1.0
    gamma = 1.4
    We = 10.0
    base = total_energy(rho, vartheta, 0.0, 0.0, gamma, We)
    for u in [0.1, 0.5, 1.5]:
        tot = total_energy(rho, vartheta, 0.0, u, gamma, We)
        assert jnp.allclose(tot - base, 0.5 * u**2, atol=1e-12)
