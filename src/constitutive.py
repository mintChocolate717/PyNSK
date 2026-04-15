"""Constitutive relations for the 1D spherically-symmetric NSK model.

Primary scalar fields:
    rho       — density (dimensionless, 0 < ρ < 1 in the van der Waals EOS)
    vartheta  — temperature ϑ (dimensionless, ϑ > 0)

The θθ stress components (``tau_tt``, ``varsigma_tt``) refer to the *polar*
angle θ and must not be confused with the temperature ϑ.
"""
import os

import jax
import jax.numpy as jnp
from jax.experimental import checkify

# Runtime guards on physical input ranges (0 < ρ < 1, ϑ > 0) are off by
# default to keep hot paths branch-free. Flip the environment variable
# ``PYNSK_CHECK_INPUTS=1`` or call ``enable_input_checks(True)`` to turn
# them on — useful while debugging, but they degrade performance slightly
# and require JAX tracing through jax.debug.check.
_CHECK_INPUTS = os.environ.get("PYNSK_CHECK_INPUTS", "0") == "1"


def enable_input_checks(flag: bool) -> None:
    """Toggle runtime input-range checks at module scope."""
    global _CHECK_INPUTS
    _CHECK_INPUTS = bool(flag)


def input_checks_enabled() -> bool:
    return _CHECK_INPUTS


def _check_rho_vartheta(rho, vartheta):
    """If enabled, assert 0 < ρ < 1 and ϑ > 0 at runtime.

    On the eager / concrete path (typical unit-test usage) this raises a
    plain :class:`ValueError` so failures are easy to catch. Inside a
    traced (``jit`` / ``vmap``) computation we fall back to
    :func:`jax.experimental.checkify.check`, which is JIT-safe **provided
    the caller wraps the top-level function in ``checkify.checkify``**.
    When guards are disabled (the default) the fast path stays
    completely branch-free.
    """
    if not _CHECK_INPUTS:
        return

    rho_arr = jnp.asarray(rho)
    vartheta_arr = jnp.asarray(vartheta)

    if isinstance(rho_arr, jax.core.Tracer) or isinstance(vartheta_arr, jax.core.Tracer):
        # Traced path — defer to checkify (JIT-safe).
        checkify.check(
            jnp.all(rho_arr > 0.0) & jnp.all(rho_arr < 1.0),
            "constitutive: density outside (0, 1)",
        )
        checkify.check(
            jnp.all(vartheta_arr > 0.0),
            "constitutive: temperature not strictly positive",
        )
        return

    rho_np = jax.device_get(rho_arr)
    vartheta_np = jax.device_get(vartheta_arr)

    if not ((rho_np > 0.0).all() and (rho_np < 1.0).all()):
        raise ValueError(
            "constitutive input guard: density must satisfy 0 < rho < 1, "
            f"got rho={rho_np!r}"
        )
    if not (vartheta_np > 0.0).all():
        raise ValueError(
            "constitutive input guard: temperature must satisfy vartheta > 0, "
            f"got vartheta={vartheta_np!r}"
        )


def free_energy_loc(rho, vartheta, gamma):
    """Local Helmholtz free energy density ψ_loc(ρ, ϑ)."""
    _check_rho_vartheta(rho, vartheta)
    return (
        -rho
        + (8.0 * vartheta / 27.0) * jnp.log(rho / (1.0 - rho))
        - (8.0 * vartheta / (27.0 * (gamma - 1.0))) * jnp.log(vartheta)
        + 8.0 * vartheta / (27.0 * (gamma - 1.0))
    )


def pressure(rho, vartheta, gamma):
    """van der Waals pressure p(ρ, ϑ) in reduced units."""
    _check_rho_vartheta(rho, vartheta)
    return 8.0 * vartheta * rho / (27.0 * (1.0 - rho)) - rho**2


def entropy(rho, vartheta, gamma):
    """Local specific entropy s(ρ, ϑ)."""
    _check_rho_vartheta(rho, vartheta)
    return -(8.0 / 27.0) * jnp.log(rho / (1.0 - rho)) + (
        8.0 / (27.0 * (gamma - 1.0))
    ) * jnp.log(vartheta)


def chemical_potential(rho, vartheta, gamma):
    """Chemical potential ν_loc(ρ, ϑ) = ∂ψ_loc/∂ρ + ψ_loc/ρ (vdW form)."""
    _check_rho_vartheta(rho, vartheta)
    return (
        -2.0 * rho
        + 8.0 * vartheta / (27.0 * (1.0 - rho))
        + (8.0 * vartheta / 27.0) * jnp.log(rho / (1.0 - rho))
        - (8.0 * vartheta / (27.0 * (gamma - 1.0))) * jnp.log(vartheta)
        + 8.0 * vartheta / (27.0 * (gamma - 1.0))
    )


def internal_energy_loc(rho, vartheta, gamma):
    """Local internal energy density ι_loc(ρ, ϑ)."""
    _check_rho_vartheta(rho, vartheta)
    return -rho + 8.0 * vartheta / (27.0 * (gamma - 1.0))


def total_energy(rho, vartheta, drho_dr, u_r, gamma, We):
    """Total energy density ρE = ι_loc + (∇ρ)²/(2 We ρ) · ρ + ½ ρ u²

    The expression returned is *per unit volume* (already multiplied by ρ
    where appropriate). ``drho_dr`` enters through the Korteweg gradient
    contribution.
    """
    iota_loc = internal_energy_loc(rho, vartheta, gamma)
    return iota_loc + (drho_dr**2) / (2.0 * We * rho) + 0.5 * u_r**2


def viscous_stress(du_dr, u_r, r, Re):
    """Traceless Newtonian viscous stress (τ_rr, τ_θθ) at radius r."""
    tau_rr = (4.0 / (3.0 * Re)) * (du_dr - u_r / r)
    tau_tt = (2.0 / (3.0 * Re)) * (u_r / r - du_dr)
    return tau_rr, tau_tt


def korteweg_stress(rho, drho_dr, d2rho_dr2, r, We):
    """Korteweg capillary stress (ς_rr, ς_θθ) at radius r."""
    delta_rho = d2rho_dr2 + 2.0 * drho_dr / r
    varsigma_rr = (1.0 / We) * (rho * delta_rho - 0.5 * drho_dr**2)
    varsigma_tt = (1.0 / We) * (rho * delta_rho + 0.5 * drho_dr**2)
    return varsigma_rr, varsigma_tt


def kappa_star(Re, Pr, gamma):
    """Dimensionless heat conductivity κ* = 8γ / (27(γ-1) Re Pr)."""
    return 8.0 * gamma / (27.0 * (gamma - 1.0) * Re * Pr)


def heat_flux(dvartheta_dr, kappa):
    """Fourier heat flux q_r = -κ ∂ϑ/∂r."""
    return -kappa * dvartheta_dr


def interstitial_working(rho, du_dr, u_r, r, drho_dr, We):
    """Interstitial working Π_r = (1/We) ρ (∇·u) ∂ρ/∂r."""
    return (1.0 / We) * rho * (du_dr + 2.0 * u_r / r) * drho_dr
