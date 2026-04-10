import jax.numpy as jnp

def free_energy_loc(rho, theta, gamma):
    return -rho + (8.0 * theta / 27.0) * jnp.log(rho / (1.0 - rho)) - (8.0 * theta / (27.0 * (gamma - 1.0))) * jnp.log(theta) + 8.0 * theta / (27.0 * (gamma - 1.0))

def pressure(rho, theta, gamma):
    return 8.0 * theta * rho / (27.0 * (1.0 - rho)) - rho**2

def entropy(rho, theta, gamma):
    return -(8.0 / 27.0) * jnp.log(rho / (1.0 - rho)) + (8.0 / (27.0 * (gamma - 1.0))) * jnp.log(theta)

def chemical_potential(rho, theta, gamma):
    return -2.0 * rho + 8.0 * theta / (27.0 * (1.0 - rho)) + (8.0 * theta / 27.0) * jnp.log(rho / (1.0 - rho)) - (8.0 * theta / (27.0 * (gamma - 1.0))) * jnp.log(theta) + 8.0 * theta / (27.0 * (gamma - 1.0))

def internal_energy_loc(rho, theta, gamma):
    return -rho + 8.0 * theta / (27.0 * (gamma - 1.0))

def total_energy(rho, theta, drho_dr, u_r, gamma, We):
    iota_loc = internal_energy_loc(rho, theta, gamma)
    return iota_loc + (drho_dr**2) / (2.0 * We * rho) + 0.5 * u_r**2

def viscous_stress(du_dr, u_r, r, Re):
    tau_rr = (4.0 / (3.0 * Re)) * (du_dr - u_r / r)
    tau_tt = (2.0 / (3.0 * Re)) * (u_r / r - du_dr)
    return tau_rr, tau_tt

def korteweg_stress(rho, drho_dr, d2rho_dr2, r, We):
    delta_rho = d2rho_dr2 + 2.0 * drho_dr / r
    varsigma_rr = (1.0 / We) * (rho * delta_rho - 0.5 * drho_dr**2)
    varsigma_tt = (1.0 / We) * (rho * delta_rho + 0.5 * drho_dr**2)
    return varsigma_rr, varsigma_tt

def kappa_star(Re, Pr, gamma):
    return 8.0 * gamma / (27.0 * (gamma - 1.0) * Re * Pr)

def heat_flux(dtheta_dr, kappa):
    return -kappa * dtheta_dr

def interstitial_working(rho, du_dr, u_r, r, drho_dr, We):
    return (1.0 / We) * rho * (du_dr + 2.0 * u_r / r) * drho_dr
