"""
Post-processing diagnostics (D3).

Implements quantities commonly monitored during an NSK bubble simulation:

* ``bubble_radius``     — iso-contour of ``rho = threshold`` via linear
  interpolation on an r-grid.
* ``total_free_energy`` — volume integral of Ψ_loc + (∇ρ)²/(2·We·ρ)
  with the spherical r² measure.
* ``total_internal_energy`` — volume integral of ρE.
* ``mass_conservation_error`` — relative drift of ∫ρ r² dr over a history.
* ``entropy_production_rate`` — positive-semi-definite global budget.

The quadrature/cache interface used here is deliberately duck-typed so the
module works with whatever ``build_basis_cache`` ends up producing in
``src/assembler.py`` (Phase B) — the module pulls the following attributes
(or dict keys) from ``cache``:

    r_q    : (n_qp,)   physical radial coordinates of all quadrature points
    w_q    : (n_qp,)   quadrature weights INCLUDING the element Jacobian
                       (so ∫ f dr ≈ Σ f(r_q) w_q, with no r² factor)
    N_rho, dN_rho, d2N_rho, N_u, dN_u, N_vartheta, dN_vartheta, N_V, dN_V

Missing derivative matrices are tolerated when the quantity does not need
them.
"""
from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from src.constitutive import (
    entropy,
    free_energy_loc,
    heat_flux,
    interstitial_working,
    kappa_star,
    total_energy,
    viscous_stress,
)


# ----------------------------------------------------------------------
# cache helpers
# ----------------------------------------------------------------------

def _get(cache: Any, key: str, default=None):
    """Read ``key`` from cache whether it is a dict or an attribute bag."""
    if isinstance(cache, dict):
        return cache.get(key, default)
    return getattr(cache, key, default)


def _require(cache: Any, key: str):
    val = _get(cache, key)
    if val is None:
        raise AttributeError(f"cache is missing required entry {key!r}")
    return val


def _params_get(params: Any, key: str, default=None):
    if isinstance(params, dict):
        return params.get(key, default)
    return getattr(params, key, default)


# ----------------------------------------------------------------------
# bubble radius
# ----------------------------------------------------------------------

def bubble_radius(rho_field, r_grid, threshold: float = 0.5) -> float:
    """Locate the iso-contour ``rho = threshold`` on ``r_grid`` by linear
    interpolation.

    Returns the smallest radius at which ``rho`` crosses ``threshold``. If
    the field does not cross the threshold anywhere, returns ``nan``.
    """
    r = np.asarray(r_grid, dtype=np.float64).ravel()
    rho = np.asarray(rho_field, dtype=np.float64).ravel()
    if r.shape != rho.shape:
        raise ValueError("rho_field and r_grid must have matching shape")

    # find first index where a sign change in (rho - threshold) occurs
    s = rho - threshold
    for i in range(len(s) - 1):
        if s[i] == 0.0:
            return float(r[i])
        if s[i] * s[i + 1] < 0.0:
            # linear interpolation in r for the zero
            alpha = s[i] / (s[i] - s[i + 1])
            return float(r[i] + alpha * (r[i + 1] - r[i]))
    if s[-1] == 0.0:
        return float(r[-1])
    return float("nan")


# ----------------------------------------------------------------------
# energy integrals
# ----------------------------------------------------------------------

def _field_at_quadrature(cache: Any, ctrl: dict):
    r_q = jnp.asarray(_require(cache, "r_q"))
    w_q = jnp.asarray(_require(cache, "w_q"))

    N_rho = jnp.asarray(_require(cache, "N_rho"))
    dN_rho = jnp.asarray(_require(cache, "dN_rho"))
    N_u = jnp.asarray(_require(cache, "N_u"))
    N_vartheta = jnp.asarray(_require(cache, "N_vartheta"))

    rho_q = N_rho @ jnp.asarray(ctrl["rho"])
    drho_dr_q = dN_rho @ jnp.asarray(ctrl["rho"])
    u_q = N_u @ jnp.asarray(ctrl["u"])
    theta_q = N_vartheta @ jnp.asarray(ctrl["vartheta"])
    return r_q, w_q, rho_q, drho_dr_q, u_q, theta_q


def total_free_energy(ctrl: dict, cache: Any, params: Any) -> jnp.ndarray:
    """∫ [Ψ_loc(ρ, ϑ) + (∇ρ)² / (2·We·ρ)] · r² dr."""
    gamma = _params_get(params, "gamma")
    We = _params_get(params, "We")
    if gamma is None or We is None:
        raise ValueError("params must supply 'gamma' and 'We'")

    r_q, w_q, rho_q, drho_dr_q, _u_q, theta_q = _field_at_quadrature(cache, ctrl)
    psi_loc = free_energy_loc(rho_q, theta_q, gamma)
    gradient_energy = drho_dr_q**2 / (2.0 * We * rho_q)
    integrand = (psi_loc + gradient_energy) * r_q**2
    return jnp.sum(integrand * w_q)


def total_internal_energy(ctrl: dict, cache: Any, params: Any) -> jnp.ndarray:
    """∫ ρ E · r² dr with E including gradient and kinetic contributions."""
    gamma = _params_get(params, "gamma")
    We = _params_get(params, "We")
    if gamma is None or We is None:
        raise ValueError("params must supply 'gamma' and 'We'")

    r_q, w_q, rho_q, drho_dr_q, u_q, theta_q = _field_at_quadrature(cache, ctrl)
    E_q = total_energy(rho_q, theta_q, drho_dr_q, u_q, gamma, We)
    integrand = rho_q * E_q * r_q**2
    return jnp.sum(integrand * w_q)


# ----------------------------------------------------------------------
# mass-conservation monitor
# ----------------------------------------------------------------------

def _mass_integral(ctrl_rho, cache):
    """∫ ρ r² dr using the quadrature cache."""
    r_q = jnp.asarray(_require(cache, "r_q"))
    w_q = jnp.asarray(_require(cache, "w_q"))
    N_rho = jnp.asarray(_require(cache, "N_rho"))
    rho_q = N_rho @ jnp.asarray(ctrl_rho)
    return jnp.sum(rho_q * r_q**2 * w_q)


def mass_conservation_error(history, cache) -> jnp.ndarray:
    """Relative drift in ∫ρ r² dr.

    ``history`` is an iterable of ctrl dicts (snapshots at successive time
    levels). The reference mass is taken from the first snapshot.

    Returns a JAX array of shape ``(n_snap,)`` containing
    ``|M_k - M_0| / M_0`` at each time level.
    """
    ctrl_list = list(history)
    if not ctrl_list:
        return jnp.zeros((0,))
    M0 = _mass_integral(ctrl_list[0]["rho"], cache)
    errors = [jnp.abs(_mass_integral(c["rho"], cache) - M0) / jnp.abs(M0)
              for c in ctrl_list]
    return jnp.stack(errors)


# ----------------------------------------------------------------------
# entropy production
# ----------------------------------------------------------------------

def entropy_production_rate(
    ctrl: dict, ctrl_dot: dict, cache: Any, params: Any
) -> jnp.ndarray:
    """Global entropy production rate.

    Aggregates the positive-semi-definite viscous, thermal, and Korteweg
    dissipation contributions:

        d_t S = ∫ (1/ϑ) [ τ_rr ∂u/∂r - q_r ∂(1/ϑ)/∂r · ϑ² - Π_r ] r² dr

    This is a diagnostic-only estimate; the full discrete balance requires
    the assembled residual and is returned by the solver.
    """
    Re = _params_get(params, "Re")
    We = _params_get(params, "We")
    Pr = _params_get(params, "Pr")
    gamma = _params_get(params, "gamma")
    if None in (Re, We, Pr, gamma):
        raise ValueError("params must supply 'Re', 'We', 'Pr', 'gamma'")

    r_q = jnp.asarray(_require(cache, "r_q"))
    w_q = jnp.asarray(_require(cache, "w_q"))

    N_rho = jnp.asarray(_require(cache, "N_rho"))
    dN_rho = jnp.asarray(_require(cache, "dN_rho"))
    N_u = jnp.asarray(_require(cache, "N_u"))
    dN_u = jnp.asarray(_require(cache, "dN_u"))
    N_vartheta = jnp.asarray(_require(cache, "N_vartheta"))
    dN_vartheta = jnp.asarray(_require(cache, "dN_vartheta"))

    rho = jnp.asarray(ctrl["rho"])
    u = jnp.asarray(ctrl["u"])
    theta = jnp.asarray(ctrl["vartheta"])

    rho_q = N_rho @ rho
    drho_dr_q = dN_rho @ rho
    u_q = N_u @ u
    du_dr_q = dN_u @ u
    theta_q = N_vartheta @ theta
    dtheta_dr_q = dN_vartheta @ theta

    tau_rr, _ = viscous_stress(du_dr_q, u_q, r_q, Re)
    kappa = kappa_star(Re, Pr, gamma)
    q_r = heat_flux(dtheta_dr_q, kappa)
    Pi_r = interstitial_working(rho_q, du_dr_q, u_q, r_q, drho_dr_q, We)

    # mechanical dissipation (viscous) and thermal dissipation (Fourier)
    viscous_diss = tau_rr * du_dr_q / theta_q
    thermal_diss = kappa * dtheta_dr_q**2 / theta_q**2
    korteweg_flux = -Pi_r / theta_q

    # silence unused-var warning while keeping import
    _ = q_r
    _ = entropy(rho_q, theta_q, gamma)

    integrand = (viscous_diss + thermal_diss + korteweg_flux) * r_q**2
    return jnp.sum(integrand * w_q)
