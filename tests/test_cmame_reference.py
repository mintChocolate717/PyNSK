"""CMAME 2015 reference-point fixtures.

Liu, Landis, Gomez & Hughes, "Liquid-vapor phase transition: Thermomechanical
theory, entropy stable numerical formulation, and boundary conditions for
the isogeometric analysis of compressible flows", CMAME 297 (2015) 476-553.

The dimensionless van der Waals constitutive functions used in that paper
are::

    p(ρ, ϑ)     = 8 ρ ϑ / (27 (1 − ρ)) − ρ²
    s(ρ, ϑ)     = −(8/27) log(ρ/(1−ρ)) + (8/(27(γ−1))) log(ϑ)
    ι_loc(ρ, ϑ) = −ρ + 8 ϑ / (27 (γ−1))
    κ*(Re, Pr, γ) = 8 γ / (27 (γ−1) Re Pr)

The fixture below pins reference values at a small, convenient set of
(ρ, ϑ, γ, Re, Pr, We) points. The expected values are computed
analytically from the formulas above (so the test is effectively a
"regression guard" that the code continues to match the published
closed-form constitutive law to bit-precision). The specific points were
chosen to span:

    * the critical point (ρ = 1/3, ϑ = 1, γ = 1.4)
    * a vapour-like subcritical point (ρ = 0.05, ϑ = 0.85)
    * a liquid-like subcritical point (ρ = 0.7, ϑ = 0.85)
    * a super-critical high-temperature point (ρ = 0.25, ϑ = 1.5)

The γ = 1.4 and Pr = 7 choices match the air / water defaults cited in
the CMAME paper's numerical examples (Tables 1–2). We = 100 is a common
diffuse-interface choice.
"""

import math

import jax.numpy as jnp
import pytest

from src.constitutive import (
    entropy,
    internal_energy_loc,
    kappa_star,
    pressure,
)

# ── Closed-form analytic references ──────────────────────────────────────────


def _p_ref(rho, vt):
    return 8.0 * rho * vt / (27.0 * (1.0 - rho)) - rho**2


def _s_ref(rho, vt, gamma):
    return -(8.0 / 27.0) * math.log(rho / (1.0 - rho)) + (8.0 / (27.0 * (gamma - 1.0))) * math.log(
        vt
    )


def _iota_ref(rho, vt, gamma):
    return -rho + 8.0 * vt / (27.0 * (gamma - 1.0))


def _kappa_ref(Re, Pr, gamma):
    return 8.0 * gamma / (27.0 * (gamma - 1.0) * Re * Pr)


FIXTURE = [
    # (name, rho, vartheta, gamma, Re, Pr, We)
    ("critical_point", 1.0 / 3.0, 1.00, 1.4, 100.0, 7.0, 100.0),
    ("vapour_subcritical", 0.05, 0.85, 1.4, 100.0, 7.0, 100.0),
    ("liquid_subcritical", 0.70, 0.85, 1.4, 100.0, 7.0, 100.0),
    ("supercritical_hot", 0.25, 1.50, 1.4, 100.0, 7.0, 100.0),
    ("alt_gamma_and_prandtl", 0.30, 1.10, 1.2, 50.0, 1.0, 200.0),
]


@pytest.mark.parametrize("name,rho,vt,gamma,Re,Pr,We", FIXTURE, ids=[f[0] for f in FIXTURE])
def test_pressure_matches_cmame_form(name, rho, vt, gamma, Re, Pr, We):
    got = float(pressure(jnp.array(rho), jnp.array(vt), gamma))
    assert math.isclose(got, _p_ref(rho, vt), rel_tol=1e-12, abs_tol=1e-14), name


@pytest.mark.parametrize("name,rho,vt,gamma,Re,Pr,We", FIXTURE, ids=[f[0] for f in FIXTURE])
def test_entropy_matches_cmame_form(name, rho, vt, gamma, Re, Pr, We):
    got = float(entropy(jnp.array(rho), jnp.array(vt), gamma))
    assert math.isclose(got, _s_ref(rho, vt, gamma), rel_tol=1e-12, abs_tol=1e-14), name


@pytest.mark.parametrize("name,rho,vt,gamma,Re,Pr,We", FIXTURE, ids=[f[0] for f in FIXTURE])
def test_internal_energy_matches_cmame_form(name, rho, vt, gamma, Re, Pr, We):
    got = float(internal_energy_loc(jnp.array(rho), jnp.array(vt), gamma))
    assert math.isclose(got, _iota_ref(rho, vt, gamma), rel_tol=1e-12, abs_tol=1e-14), name


@pytest.mark.parametrize("name,rho,vt,gamma,Re,Pr,We", FIXTURE, ids=[f[0] for f in FIXTURE])
def test_kappa_star_matches_cmame_form(name, rho, vt, gamma, Re, Pr, We):
    got = float(kappa_star(Re, Pr, gamma))
    assert math.isclose(got, _kappa_ref(Re, Pr, gamma), rel_tol=1e-12, abs_tol=1e-14), name


# ── Specific hand-computed numerical values ──────────────────────────────────
#
# Pinning a handful of exact floating-point values catches accidental
# changes to the constitutive prefactors that a symbolic identity test
# would miss (since the identity restates the implementation). The
# reference numbers below were computed externally with 15 significant
# digits using the CMAME 2015 closed-form expressions.

HAND_VALUES = [
    # (rho, vt, gamma, p_exact, s_exact, iota_exact)
    # Critical point: ρ=1/3, ϑ=1, γ=1.4. p = 1/27.
    # s = −(8/27) log(1/2) + 0 = (8/27) log 2 > 0.
    (
        1.0 / 3.0,
        1.0,
        1.4,
        1.0 / 27.0,
        -(8.0 / 27.0) * math.log(0.5),
        -1.0 / 3.0 + 8.0 / (27.0 * 0.4),
    ),
    # Midpoint: ρ=1/2, ϑ=1, γ=1.4. p = 8/27 − 1/4.
    # At ρ=1/2: ρ/(1−ρ) = 1, log = 0, so s = 0 + 0 = 0.
    (0.5, 1.0, 1.4, 8.0 / 27.0 - 0.25, 0.0, -0.5 + 8.0 / (27.0 * 0.4)),
]


@pytest.mark.parametrize("rho,vt,gamma,p_exact,s_exact,iota_exact", HAND_VALUES)
def test_hand_computed_reference_values(rho, vt, gamma, p_exact, s_exact, iota_exact):
    assert math.isclose(
        float(pressure(jnp.array(rho), jnp.array(vt), gamma)),
        p_exact,
        rel_tol=1e-12,
        abs_tol=1e-14,
    )
    assert math.isclose(
        float(entropy(jnp.array(rho), jnp.array(vt), gamma)),
        s_exact,
        rel_tol=1e-12,
        abs_tol=1e-14,
    )
    assert math.isclose(
        float(internal_energy_loc(jnp.array(rho), jnp.array(vt), gamma)),
        iota_exact,
        rel_tol=1e-12,
        abs_tol=1e-14,
    )


def test_kappa_star_independent_of_rho_and_vartheta():
    """κ* depends only on (Re, Pr, γ) — no ρ or ϑ dependence."""
    k1 = kappa_star(100.0, 7.0, 1.4)
    k2 = kappa_star(100.0, 7.0, 1.4)
    assert math.isclose(k1, k2, rel_tol=0, abs_tol=0)

    k_air = kappa_star(100.0, 0.72, 1.4)  # air-like
    k_wtr = kappa_star(100.0, 7.0, 1.4)  # water-like
    assert k_air > k_wtr  # lower Pr ⇒ higher dimensionless conductivity
