import jax
import jax.numpy as jnp

from src.constitutive import (
    entropy,
    heat_flux,
    internal_energy_loc,
    interstitial_working,
    kappa_star,
    korteweg_stress,
    pressure,
    viscous_stress,
)


def test_pressure_critical_point():
    p = pressure(1.0 / 3.0, 1.0, 1.4)
    assert jnp.allclose(p, 1.0 / 27.0, atol=1e-10)


def test_pressure_low_density_ideal_gas():
    rho = 0.01
    vartheta = 1.0
    gamma = 1.4
    p = pressure(rho, vartheta, gamma)
    expected_p = 8.0 * 1.0 * 0.01 / 27.0 - 0.01**2
    assert jnp.allclose(p, expected_p, atol=1e-4)


def test_pressure_spinodal_negative_dpdrho():
    rho = 1.0 / 3.0
    vartheta = 0.85
    gamma = 1.4
    dpdrho = jax.grad(pressure, argnums=0)(rho, vartheta, gamma)
    assert dpdrho < 0.0


def test_entropy_sign():
    s = entropy(0.1, 2.0, 1.4)
    assert s > 0.0


def test_internal_energy_loc():
    rho_vals = [0.1, 0.5, 0.9]
    vartheta_vals = [0.5, 1.0, 2.0]
    gamma = 1.4
    for rho in rho_vals:
        for vartheta in vartheta_vals:
            i_loc = internal_energy_loc(rho, vartheta, gamma)
            expected = -rho + 8.0 * vartheta / (27.0 * (gamma - 1.0))
            assert jnp.allclose(i_loc, expected, atol=1e-10)


def test_viscous_stress_traceless():
    du_dr_vals = [-1.0, 0.0, 2.5]
    u_r_vals = [-0.5, 0.0, 1.0]
    r_vals = [0.1, 1.0, 5.0]
    Re = 100.0
    for du_dr in du_dr_vals:
        for u_r in u_r_vals:
            for r in r_vals:
                tau_rr, tau_tt = viscous_stress(du_dr, u_r, r, Re)
                assert jnp.allclose(tau_rr + 2.0 * tau_tt, 0.0, atol=1e-10)


def test_viscous_stress_uniform_expansion():
    C = 2.5
    r = 2.0
    u_r = C * r
    du_dr = C
    Re = 100.0
    tau_rr, tau_tt = viscous_stress(du_dr, u_r, r, Re)
    assert jnp.allclose(tau_rr, 0.0, atol=1e-10)
    assert jnp.allclose(tau_tt, 0.0, atol=1e-10)


def test_tau_tt_is_neg_half_tau_rr():
    du_dr = 1.5
    u_r = 0.5
    r = 2.0
    Re = 100.0
    tau_rr, tau_tt = viscous_stress(du_dr, u_r, r, Re)
    assert jnp.allclose(tau_tt, -tau_rr / 2.0, atol=1e-10)


def test_korteweg_zero_gradient():
    rho = 0.5
    drho_dr = 0.0
    d2rho_dr2 = 0.0
    r = 1.0
    We = 100.0
    varsigma_rr, varsigma_tt = korteweg_stress(rho, drho_dr, d2rho_dr2, r, We)
    assert jnp.allclose(varsigma_rr, 0.0, atol=1e-10)
    assert jnp.allclose(varsigma_tt, 0.0, atol=1e-10)


def test_korteweg_tt_minus_rr():
    rho_vals = [0.1, 0.5, 0.9]
    drho_dr_vals = [-0.5, 0.0, 1.5]
    d2rho_dr2_vals = [-1.0, 0.0, 2.0]
    r = 1.0
    We = 10.0
    for rho in rho_vals:
        for drho_dr in drho_dr_vals:
            for d2rho_dr2 in d2rho_dr2_vals:
                varsigma_rr, varsigma_tt = korteweg_stress(rho, drho_dr, d2rho_dr2, r, We)
                assert jnp.allclose(varsigma_tt - varsigma_rr, (1.0 / We) * drho_dr**2, atol=1e-10)


def test_heat_flux_direction():
    dvartheta_dr = 1.0
    kappa = 0.5
    q_r = heat_flux(dvartheta_dr, kappa)
    assert q_r < 0.0


def test_kappa_star():
    Re = 1.0
    Pr = 1.0
    gamma = 1.4
    k = kappa_star(Re, Pr, gamma)
    expected = 8.0 * 1.4 / (27.0 * 0.4 * 1.0 * 1.0)
    assert jnp.allclose(k, expected, atol=1e-10)


def test_interstitial_incompressible():
    u_r = 1.0
    r = 2.0
    du_dr = -2.0 * u_r / r  # ensuring du_dr + 2*u_r/r == 0
    rho = 0.5
    drho_dr = 0.1
    We = 10.0
    Pi_r = interstitial_working(rho, du_dr, u_r, r, drho_dr, We)
    assert jnp.allclose(Pi_r, 0.0, atol=1e-10)


def test_interstitial_zero_gradient():
    rho = 0.5
    du_dr = 1.5
    u_r = 0.5
    r = 2.0
    drho_dr = 0.0
    We = 10.0
    Pi_r = interstitial_working(rho, du_dr, u_r, r, drho_dr, We)
    assert jnp.allclose(Pi_r, 0.0, atol=1e-10)
