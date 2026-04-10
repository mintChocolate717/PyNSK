# PLAN.md

## Goal
Implement `src/constitutive.py` — all constitutive relations for the van der Waals NSK fluid in dimensionless form, as pure JAX functions. Then write `tests/test_constitutive.py` with analytical verification tests.

## Context
PyNSK solver project for bubble cavitation using NSK equations with spherical symmetry. Math is fully derived and Gemini-verified in the LaTeX repo. All equations are in **dimensionless form** following Liu et al. (CMAME 2015), reference scales: ρ_ref = b, p_ref = ab², θ_ref = 8ab/(27R).

JAX float64 enabled globally in `src/__init__.py`. Existing: `src/bsplines.py`, `src/quadrature.py`, both with passing tests.

## Files In Scope
- `src/constitutive.py` — CREATE
- `tests/test_constitutive.py` — CREATE

## Steps

### Step 1: `src/constitutive.py`

All functions: pure JAX, no defaults, no side effects. Start with `import jax.numpy as jnp`.

**Thermodynamics** (from Ψ_loc = -ρ + (8θ/27)ln(ρ/(1-ρ)) - (8θ/(27(γ-1)))lnθ + 8θ/(27(γ-1))):

- `free_energy_loc(rho, theta, gamma)` → Ψ_loc
- `pressure(rho, theta, gamma)` → p = 8θρ/[27(1-ρ)] - ρ²
- `entropy(rho, theta, gamma)` → s = -(8/27)ln(ρ/(1-ρ)) + (8/(27(γ-1)))lnθ
- `chemical_potential(rho, theta, gamma)` → ν_loc = -2ρ + 8θ/[27(1-ρ)] + (8θ/27)ln(ρ/(1-ρ)) - (8θ/(27(γ-1)))lnθ + 8θ/(27(γ-1))
- `internal_energy_loc(rho, theta, gamma)` → ι_loc = -ρ + 8θ/(27(γ-1))
- `total_energy(rho, theta, drho_dr, u_r, gamma, We)` → E = ι_loc + (drho_dr²)/(2·We·ρ) + ½u_r²

**Mechanical stresses** (spherical symmetry):

- `viscous_stress(du_dr, u_r, r, Re)` → (tau_rr, tau_tt)
  - tau_rr = (4/(3Re))(du_dr - u_r/r)
  - tau_tt = (2/(3Re))(u_r/r - du_dr)
- `korteweg_stress(rho, drho_dr, d2rho_dr2, r, We)` → (varsigma_rr, varsigma_tt)
  - delta_rho = d2rho_dr2 + 2*drho_dr/r  (spherical Laplacian)
  - varsigma_rr = (1/We)*(rho*delta_rho - 0.5*drho_dr²)
  - varsigma_tt = (1/We)*(rho*delta_rho + 0.5*drho_dr²)

**Thermal fluxes** (spherical symmetry):

- `kappa_star(Re, Pr, gamma)` → κ* = 8γ/[27(γ-1)·Re·Pr]
- `heat_flux(dtheta_dr, kappa)` → q_r = -kappa * dtheta_dr
- `interstitial_working(rho, du_dr, u_r, r, drho_dr, We)` → Π_r = (1/We)*ρ*(du_dr + 2u_r/r)*drho_dr

### Step 2: `tests/test_constitutive.py`

Import: `import jax.numpy as jnp`, `import jax`, `from src.constitutive import ...`
Use `pytest`, assert with `jnp.allclose(..., atol=1e-10)` or plain `==` for exact.
Use gamma=1.4 throughout unless noted.

Tests:

1. `test_pressure_critical_point` — p(1/3, 1.0, 1.4) ≈ 1/27
2. `test_pressure_low_density_ideal_gas` — at rho=0.01, theta=1.0: p ≈ 8*1.0*0.01/27 - 0.01**2
3. `test_pressure_spinodal_negative_dpdrho` — at rho=1/3, theta=0.85: dp/drho < 0 (use jax.grad)
4. `test_entropy_sign` — s(0.1, 2.0, 1.4) > 0
5. `test_internal_energy_loc` — ι_loc(rho, theta, gamma) == -rho + 8*theta/(27*(gamma-1)) for several values
6. `test_viscous_stress_traceless` — tau_rr + 2*tau_tt == 0 for arbitrary inputs
7. `test_viscous_stress_uniform_expansion` — u_r = C*r → du_dr = C, u_r/r = C → tau_rr = 0, tau_tt = 0
8. `test_tau_tt_is_neg_half_tau_rr` — tau_tt == -tau_rr/2
9. `test_korteweg_zero_gradient` — drho_dr=0, d2rho_dr2=0 → varsigma_rr=0, varsigma_tt=0
10. `test_korteweg_tt_minus_rr` — varsigma_tt - varsigma_rr = (1/We)*drho_dr² for any inputs
11. `test_heat_flux_direction` — dtheta_dr=1.0 → q_r < 0
12. `test_kappa_star` — kappa_star(1.0, 1.0, 1.4) == 8*1.4/(27*0.4*1.0*1.0)
13. `test_interstitial_incompressible` — du_dr + 2*u_r/r = 0 → Pi_r = 0
14. `test_interstitial_zero_gradient` — drho_dr=0 → Pi_r = 0

## Success Criteria
- [ ] `pytest tests/test_constitutive.py` — all tests pass
- [ ] `ruff check src/constitutive.py tests/test_constitutive.py` — clean
- [ ] No default argument values anywhere
- [ ] All imports at top of file
- [ ] Write results to EXECUTION_LOG.md as you go
