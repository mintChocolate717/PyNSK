"""Element residual assembly for the 1D spherically-symmetric NSK system.

Naming convention:
    * ``vartheta`` — temperature ϑ and quantities derived from it
      (``dvartheta_dr``, ``N_vartheta``, ``ctrl_vartheta``).
    * ``tt`` / ``theta`` suffixes on stresses (``tau_tt``, ``sigma_tt``,
      ``ς_θθ``) refer to the **polar angle** θ. These stay named ``tt``.
"""
import jax.numpy as jnp

from src.constitutive import (
    chemical_potential,
    entropy,
    heat_flux,
    interstitial_working,
    kappa_star,
    korteweg_stress,
    viscous_stress,
)


def phase_field_div(drho_dr, d2rho_dr2, dvartheta_dr, vartheta, r):
    """Spherical divergence ∇·(∇ρ/ϑ) at quadrature points.

    Eq. (eq:disc-phase-div)::

        2/(r ϑ) ∂ρ/∂r  −  (1/ϑ²) (∂ϑ/∂r)(∂ρ/∂r)  +  (1/ϑ) ∂²ρ/∂r²

    Args:
        drho_dr:      first radial derivative of density at quad points
        d2rho_dr2:    second radial derivative of density
        dvartheta_dr: first radial derivative of temperature ϑ
        vartheta:     temperature ϑ at quad points
        r:            physical radial coordinate at quad points
    """
    return (
        2.0 / (r * vartheta) * drho_dr
        - 1.0 / vartheta**2 * dvartheta_dr * drho_dr
        + 1.0 / vartheta * d2rho_dr2
    )


# Backwards-compatible alias (private spelling) in case any external caller
# still imports it. Prefer ``phase_field_div`` in new code.
_phase_field_div = phase_field_div


def element_residual_mass(
    N_rho,
    dN_rho,
    N_u,
    ctrl_rho,
    ctrl_rho_dot,
    ctrl_u,
    r_q,
    w_q,
    J_e,
):
    """Element residual for the mass equation R^{ρ,e}_C.

    Implements Eq. (mass-eq:element_residual)::

        R^ρ_C = Σ_q [N^ρ_C ρ̇ − (dN^ρ_C/dr) ρ u_r] r_q² J_e w_q

    Args:
        N_rho:       (n_qp, n_rho)  basis values for density field
        dN_rho:      (n_qp, n_rho)  first derivatives of density basis
        N_u:         (n_qp, n_u)    basis values for velocity field
        ctrl_rho:    (n_rho,)       density control points at current time
        ctrl_rho_dot:(n_rho,)       time derivative of density control points
        ctrl_u:      (n_u,)         velocity control points
        r_q:         (n_qp,)        physical quadrature locations
        w_q:         (n_qp,)        quadrature weights (reference element)
        J_e:         scalar         element Jacobian dr/dξ

    Returns:
        (n_rho,) element residual vector
    """
    rho_q = N_rho @ ctrl_rho
    rho_dot_q = N_rho @ ctrl_rho_dot
    u_q = N_u @ ctrl_u

    # integrand per quadrature point: shape (n_qp, n_rho)
    integrand = rho_dot_q[:, None] * N_rho - (rho_q * u_q)[:, None] * dN_rho

    weights = r_q**2 * J_e * w_q
    return jnp.einsum("q,qc->c", weights, integrand)


def element_residual_momentum(
    N_u,
    dN_u,
    N_rho,
    dN_rho,
    d2N_rho,
    N_vartheta,
    dN_vartheta,
    N_V,
    ctrl_rho,
    ctrl_rho_dot,
    ctrl_u,
    ctrl_u_dot,
    ctrl_vartheta,
    ctrl_V,
    r_q,
    w_q,
    J_e,
    Re,
    We,
    gamma,
    b_r,
):
    """Element residual for the momentum equation R^{u,e}_C.

    Implements Eq. (momentum-eq:element_residual).

    Args:
        N_u, dN_u:                 (n_qp, n_u)        velocity basis and derivatives
        N_rho, dN_rho, d2N_rho:    (n_qp, n_rho)      density basis up to 2nd derivative
        N_vartheta, dN_vartheta:   (n_qp, n_vartheta) temperature ϑ basis and derivatives
        N_V:                       (n_qp, n_V)        auxiliary variable basis
        ctrl_*:                    control point arrays for each field
        ctrl_rho_dot:              (n_rho,) time derivative of density
        ctrl_u_dot:                (n_u,)   time derivative of velocity
        r_q, w_q, J_e:             quadrature data
        Re, We, gamma:             dimensionless parameters
        b_r:                       scalar radial body force (per unit mass)

    Returns:
        (n_u,) element residual vector
    """
    rho_q = N_rho @ ctrl_rho
    drho_dr_q = dN_rho @ ctrl_rho
    d2rho_dr2_q = d2N_rho @ ctrl_rho
    rho_dot_q = N_rho @ ctrl_rho_dot

    u_q = N_u @ ctrl_u
    du_dr_q = dN_u @ ctrl_u
    u_dot_q = N_u @ ctrl_u_dot

    vartheta_q = N_vartheta @ ctrl_vartheta
    dvartheta_dr_q = dN_vartheta @ ctrl_vartheta

    V_q = N_V @ ctrl_V

    div_q = phase_field_div(drho_dr_q, d2rho_dr2_q, dvartheta_dr_q, vartheta_q, r_q)

    # η = ρVϑ + ½ρu² + (1/We)ρϑ∇·(∇ρ/ϑ)
    eta_q = (
        rho_q * V_q * vartheta_q
        + 0.5 * rho_q * u_q**2
        + (1.0 / We) * rho_q * vartheta_q * div_q
    )

    # ξ = Vϑ + ½u² + (1/We)ϑ∇·(∇ρ/ϑ)
    xi_q = V_q * vartheta_q + 0.5 * u_q**2 + (1.0 / We) * vartheta_q * div_q

    # H = ρs  (volumetric entropy density)
    H_q = rho_q * entropy(rho_q, vartheta_q, gamma)

    tau_rr_q, tau_tt_q = viscous_stress(du_dr_q, u_q, r_q, Re)
    varsigma_rr_q, varsigma_tt_q = korteweg_stress(rho_q, drho_dr_q, d2rho_dr2_q, r_q, We)

    # ∂(ρu_r)/∂t
    d_rhou_dt_q = rho_dot_q * u_q + rho_q * u_dot_q

    # dN^u/dr coefficient: [−ρu² − η + τ_rr + ς_rr]
    coeff_dN = -rho_q * u_q**2 - eta_q + tau_rr_q + varsigma_rr_q

    # N^u coefficient: [∂(ρu)/∂t − ξ ∂ρ/∂r − H ∂ϑ/∂r + (2/r)(τ_θθ + ς_θθ − η) − ρb_r]
    coeff_N = (
        d_rhou_dt_q
        - xi_q * drho_dr_q
        - H_q * dvartheta_dr_q
        + (2.0 / r_q) * (tau_tt_q + varsigma_tt_q - eta_q)
        - rho_q * b_r
    )

    weights = r_q**2 * J_e * w_q
    integrand = coeff_dN[:, None] * dN_u + coeff_N[:, None] * N_u
    return jnp.einsum("q,qc->c", weights, integrand)


def element_residual_energy(
    N_vartheta,
    dN_vartheta,
    N_rho,
    dN_rho,
    d2N_rho,
    N_u,
    dN_u,
    N_V,
    ctrl_rho,
    ctrl_rho_dot,
    ctrl_u,
    ctrl_u_dot,
    ctrl_vartheta,
    ctrl_vartheta_dot,
    ctrl_V,
    r_q,
    w_q,
    J_e,
    Re,
    We,
    gamma,
    Pr,
    b_r,
    r_s,
):
    """Element residual for the energy equation R^{E,e}_C.

    Implements Eq. (energy-eq:element_residual).
    ∂(ρE)/∂t is computed analytically from primary-field time derivatives.

    Args:
        N_vartheta, dN_vartheta:   (n_qp, n_vartheta) temperature ϑ basis and derivatives
        N_rho, dN_rho, d2N_rho:    (n_qp, n_rho)
        N_u, dN_u:                 (n_qp, n_u)
        N_V:                       (n_qp, n_V)
        ctrl_*_dot:                time derivatives of each field's control points
        r_q, w_q, J_e:             quadrature data
        Re, We, gamma, Pr:         dimensionless parameters
        b_r:                       scalar radial body force
        r_s:                       scalar volumetric heat source

    Returns:
        (n_vartheta,) element residual vector
    """
    rho_q = N_rho @ ctrl_rho
    drho_dr_q = dN_rho @ ctrl_rho
    d2rho_dr2_q = d2N_rho @ ctrl_rho
    rho_dot_q = N_rho @ ctrl_rho_dot
    drho_dot_dr_q = dN_rho @ ctrl_rho_dot

    u_q = N_u @ ctrl_u
    du_dr_q = dN_u @ ctrl_u
    u_dot_q = N_u @ ctrl_u_dot

    vartheta_q = N_vartheta @ ctrl_vartheta
    dvartheta_dr_q = dN_vartheta @ ctrl_vartheta
    vartheta_dot_q = N_vartheta @ ctrl_vartheta_dot

    V_q = N_V @ ctrl_V

    div_q = phase_field_div(drho_dr_q, d2rho_dr2_q, dvartheta_dr_q, vartheta_q, r_q)

    H_q = rho_q * entropy(rho_q, vartheta_q, gamma)

    # β = ρVϑ − ϑH + (1/2We)(∂ρ/∂r)² + ρu² + (1/We)ρϑ∇·(∇ρ/ϑ)
    beta_q = (
        rho_q * V_q * vartheta_q
        - vartheta_q * H_q
        + 0.5 / We * drho_dr_q**2
        + rho_q * u_q**2
        + (1.0 / We) * rho_q * vartheta_q * div_q
    )

    tau_rr_q, _ = viscous_stress(du_dr_q, u_q, r_q, Re)
    varsigma_rr_q, _ = korteweg_stress(rho_q, drho_dr_q, d2rho_dr2_q, r_q, We)

    kappa = kappa_star(Re, Pr, gamma)
    q_r_q = heat_flux(dvartheta_dr_q, kappa)
    Pi_r_q = interstitial_working(rho_q, du_dr_q, u_q, r_q, drho_dr_q, We)

    # ∂(ρE)/∂t  from  ρE = −ρ² + 8ρϑ/(27(γ−1)) + (∂ρ/∂r)²/(2We) + ½ρu²
    d_rhoE_dt_q = (
        (-2.0 * rho_q + 8.0 * vartheta_q / (27.0 * (gamma - 1.0)) + 0.5 * u_q**2) * rho_dot_q
        + 8.0 * rho_q / (27.0 * (gamma - 1.0)) * vartheta_dot_q
        + drho_dr_q / We * drho_dot_dr_q
        + rho_q * u_q * u_dot_q
    )

    # dN^ϑ/dr coefficient: [−βu_r + τ_rr u_r + ς_rr u_r − q_r − Π_r]
    coeff_dN = (-beta_q + tau_rr_q + varsigma_rr_q) * u_q - q_r_q - Pi_r_q

    # N^ϑ coefficient: [∂(ρE)/∂t − ρb_r u_r − ρr_s]
    coeff_N = d_rhoE_dt_q - rho_q * b_r * u_q - rho_q * r_s

    weights = r_q**2 * J_e * w_q
    integrand = coeff_dN[:, None] * dN_vartheta + coeff_N[:, None] * N_vartheta
    return jnp.einsum("q,qc->c", weights, integrand)


def element_residual_auxiliary(
    N_V,
    dN_V,
    N_rho,
    dN_rho,
    N_u,
    N_vartheta,
    ctrl_rho,
    ctrl_u,
    ctrl_vartheta,
    ctrl_V,
    r_q,
    w_q,
    J_e,
    We,
    gamma,
):
    """Element residual for the auxiliary equation R^{V,e}_C.

    Implements Eq. (auxiliary-eq:element_residual)::

        R^V_C = Σ_q { N^V [V − (1/ϑ)(ν_loc − ½u²)]
                      − (dN^V/dr) [1/(We ϑ) ∂ρ/∂r] } r_q² J_e w_q

    Args:
        N_V, dN_V:      (n_qp, n_V)        auxiliary basis and derivatives
        N_rho, dN_rho:  (n_qp, n_rho)
        N_u:            (n_qp, n_u)
        N_vartheta:     (n_qp, n_vartheta)
        ctrl_*:         control point arrays
        r_q, w_q, J_e:  quadrature data
        We, gamma:      dimensionless parameters

    Returns:
        (n_V,) element residual vector
    """
    rho_q = N_rho @ ctrl_rho
    drho_dr_q = dN_rho @ ctrl_rho
    u_q = N_u @ ctrl_u
    vartheta_q = N_vartheta @ ctrl_vartheta
    V_q = N_V @ ctrl_V

    nu_loc_q = chemical_potential(rho_q, vartheta_q, gamma)

    # N^V coefficient: [V − (1/ϑ)(ν_loc − ½u²)]
    coeff_N = V_q - (1.0 / vartheta_q) * (nu_loc_q - 0.5 * u_q**2)

    # dN^V/dr coefficient: −1/(We ϑ) ∂ρ/∂r
    coeff_dN = -(1.0 / (We * vartheta_q)) * drho_dr_q

    weights = r_q**2 * J_e * w_q
    integrand = coeff_N[:, None] * N_V + coeff_dN[:, None] * dN_V
    return jnp.einsum("q,qc->c", weights, integrand)
