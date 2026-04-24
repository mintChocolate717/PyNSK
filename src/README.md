# src — Solver Modules

Core physics and numerics. All modules are pure JAX/NumPy; no global state except the float64 flag in `__init__.py`.

## Files

### `__init__.py`
Enables JAX float64 globally via `jax.config.update("jax_enable_x64", True)`. Must be imported before any JAX computation — never remove.

---

### `bsplines.py`
B-spline basis functions for IGA spatial discretization on the reference interval [0, 1].

| Function | Description |
|---|---|
| `make_knot_vector(n_elements, degree)` | Open-uniform knot vector with clamped endpoints |
| `basis_matrix(xi_pts, n_elements, degree)` | B_A(ξ_q) → array shape [n_quad, n_basis] |
| `basis_deriv_matrix(xi_pts, n_elements, degree, order)` | nth-order derivatives of all basis functions |

Degree p ≥ 2 is required: the Korteweg capillary term needs ∂²ρ/∂r², which demands C¹ continuity across element boundaries.  
Basis matrices are constant JAX arrays (no autodiff through them). Physics autodiff happens in `residuals.py`.

---

### `quadrature.py`
Gaussian quadrature for the spherically symmetric radial domain [0, R_max].

| Function | Description |
|---|---|
| `gauss_legendre(n)` | Gauss-Legendre points and weights on [−1, 1] |
| `quadrature_points(mesh, n_gauss)` | Maps reference element quadrature → physical r coordinates |
| `recommended_n_gauss(degree)` | Returns degree+1 (exact for mass matrix integrands) |

Returned weights include element Jacobian J_e = (r_{e+1} − r_e)/2. The r² spherical factor is applied per-residual in `residuals.py`, not here.

---

### `constitutive.py`
Pure JAX functions for the van der Waals NSK fluid in dimensionless form. Reference scales fixed at the van der Waals critical point (ρ_ref = b, p_ref = ab², ϑ_ref = 8ab/27R). Implements Liu et al. (CMAME 2015) constitutive relations.

**Thermodynamic potentials:**

| Function | Output |
|---|---|
| `free_energy_loc(ρ, ϑ, γ)` | Helmholtz free energy per unit mass Ψ |
| `pressure(ρ, ϑ, γ)` | Equation of state: p = 8ϑρ/[27(1−ρ)] − ρ² |
| `entropy(ρ, ϑ, γ)` | Specific entropy σ |
| `chemical_potential(ρ, ϑ, γ)` | Gibbs potential ν_loc |
| `internal_energy_loc(ρ, ϑ, γ)` | Specific internal energy |

**Mechanical stresses (spherical symmetry — two independent components each):**

| Function | Output |
|---|---|
| `viscous_stress(du_dr, u_r, r, Re)` | τ_rr, τ_θθ — Newtonian; traceless (Stokes hypothesis) |
| `korteweg_stress(ρ, dρ_dr, d2ρ_dr2, We)` | ς_rr, ς_θθ — capillary; ∝ 1/We |

**Thermal fluxes:**

| Function | Output |
|---|---|
| `kappa_star(γ, Re, Pr)` | Effective thermal conductivity κ* = 8γ/[27(γ−1)·Re·Pr] |
| `heat_flux(dϑ_dr, γ, Re, Pr)` | Fourier heat flux q_r = −κ*·∂ϑ/∂r |
| `interstitial_working(ρ, u_r, du_dr, r, dρ_dr, We)` | Interstitial energy flux Π_r (Dunn–Serrin term) |

---

### `residuals.py`
Element-level weak form residuals for all 4 PDEs of the spherically symmetric NSK system. Each function returns a 1D JAX array of length n_basis_element (one entry per control point in the element).

| Function | PDE | Residual label |
|---|---|---|
| `element_residual_mass(...)` | Continuity (mass conservation) | R^ρ,e_C |
| `element_residual_momentum(...)` | Linear momentum (with viscous + Korteweg stress) | R^u,e_C |
| `element_residual_energy(...)` | Total energy (internal + kinetic + interfacial) | R^E,e_C |
| `element_residual_auxiliary(...)` | Auxiliary variable V = μ/ϑ + ½u² (phase-field coupling) | R^V,e_C |

Quadrature assembly uses `jnp.einsum("q,qc->c", weights, integrand)` — vectorized over quadrature points, no Python loops.  
The r² spherical Jacobian is folded into each integrand here.

**Depends on:** `constitutive.py`, `bsplines.py`, `quadrature.py`  
**Used by:** global assembly (Milestone 4 — not yet implemented)
