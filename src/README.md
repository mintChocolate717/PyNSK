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
| `make_knot_vector(n_ctrl, degree)` | Open-uniform knot vector with clamped endpoints |
| `basis_matrix(xi_pts, knots, degree)` | B_A(ξ_q) → array shape [n_quad, n_basis] |
| `basis_deriv_matrix(xi_pts, knots, degree, order)` | nth-order derivatives of all basis functions |

Degree p ≥ 2 is required: the Korteweg capillary term needs ∂²ρ/∂r², which demands C¹ continuity across element boundaries.  
Basis matrices are constant JAX arrays (no autodiff through them). Physics autodiff happens in `residuals.py`.

---

### `quadrature.py`
Gaussian quadrature for the spherically symmetric radial domain [0, R_max].

| Function | Description |
|---|---|
| `gauss_legendre(n)` | Gauss-Legendre points and weights on [−1, 1] |
| `quadrature_points(knots, degree, n_gauss, R_max)` | Maps reference element quadrature → physical r coordinates |
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
| `total_energy(ρ, ϑ, ∂ρ/∂r, u_r, γ, We)` | Total energy E = ι + (∇ρ)²/(2Weρ) + ½u² |

**Mechanical stresses (spherical symmetry — two independent components each):**

| Function | Output |
|---|---|
| `viscous_stress(du_dr, u_r, r, Re)` | τ_rr, τ_θθ — Newtonian; traceless (Stokes hypothesis) |
| `korteweg_stress(ρ, dρ_dr, d2ρ_dr2, We)` | ς_rr, ς_θθ — capillary; ∝ 1/We |

**Thermal fluxes:**

| Function | Output |
|---|---|
| `kappa_star(Re, Pr, γ)` | Effective thermal conductivity κ* = 8γ/[27(γ−1)·Re·Pr] |
| `heat_flux(dϑ_dr, κ)` | Fourier heat flux q_r = −κ*·∂ϑ/∂r |
| `interstitial_working(ρ, du_dr, u_r, r, dρ_dr, We)` | Interstitial energy flux Π_r (Dunn–Serrin term) |

**Input guards** (opt-in, off by default):

| Function | Description |
|---|---|
| `enable_input_checks(flag)` | Enable/disable runtime range checks on ρ and ϑ |
| `input_checks_enabled()` | Returns current flag state |

Guards assert 0 < ρ < 1 and ϑ > 0. Enable via `enable_input_checks(True)` or `PYNSK_CHECK_INPUTS=1`. Guards route through `jax.experimental.checkify` so they compose with JIT.

---

### `residuals.py`
Element-level weak form residuals for all 4 PDEs of the spherically symmetric NSK system. Each function returns a 1D JAX array of length n_basis_element (one entry per control point in the element).

| Function | PDE | Residual label |
|---|---|---|
| `element_residual_mass(...)` | Continuity (mass conservation) | R^ρ,e_C |
| `element_residual_momentum(...)` | Linear momentum (with viscous + Korteweg stress) | R^u,e_C |
| `element_residual_energy(...)` | Total energy (internal + kinetic + interfacial) | R^E,e_C |
| `element_residual_auxiliary(...)` | Auxiliary variable V = μ/ϑ + ½u² (phase-field coupling) | R^V,e_C |

Also exports `phase_field_div(dρ_dr, d²ρ_dr², dϑ_dr, ϑ, r)` — computes the spherical ∇·(∇ρ/ϑ) = (2/rϑ)∂ρ/∂r − (1/ϑ²)(∂ϑ/∂r)(∂ρ/∂r) + (1/ϑ)∂²ρ/∂r². Available as `residuals.phase_field_div` (and `residuals._phase_field_div` for back-compat).

Quadrature assembly uses `jnp.einsum("q,qc->c", weights, integrand)` — vectorized over quadrature points, no Python loops.  
The r² spherical Jacobian is folded into each integrand here.

**Depends on:** `constitutive.py`, `bsplines.py`, `quadrature.py`  
**Used by:** `assembler.py`

---

### `assembler.py`
Global residual assembler. Couples the four scalar fields (ρ, u_r, ϑ, V) into a single global residual vector of length 4·n_ctrl, ordered block-wise: `[R_ρ | R_u | R_ϑ | R_V]`.

| Function / class | Description |
|---|---|
| `element_connectivity(n_ctrl, degree)` | IEN local-to-global DOF map, shape (n_elem, p+1) |
| `build_basis_cache(knots, degree, n_gauss, R_max)` | Precomputes element-local N, dN, d²N, r, w, J, IEN — call once, reuse every Newton iteration |
| `assemble_residual(ctrl, ctrl_dot, cache, params)` | Global 4-block residual via `jax.vmap` over elements + `segment_sum` scatter-add |
| `apply_dirichlet(R, K, dof_indices, values, current)` | Standard row/col zero-out + unit diagonal + residual adjustment |
| `symmetry_bc_at_origin(n_ctrl)` | Essential BC: u_r(0) = 0 (only the first u-DOF constrained; ρ and ϑ have natural BCs at r=0) |
| `dirichlet_far_field(n_ctrl, rho_inf, vartheta_inf, u_inf)` | Far-field BCs at r=R_max on ρ, u, ϑ (V is algebraic, left free) |

The vmap loop is a single traced operation compatible with `jax.jacfwd`; assembling K_tan = ∂R/∂d_{n+1} is a one-liner.  
Sparsity note: currently dense `segment_sum`. A TODO(sparse) marker in `assemble_residual` identifies the single site to swap for BCOO/CSR storage.

**Depends on:** `residuals.py`, `bsplines.py`, `quadrature.py`  
**Used by:** `solver.py`, simulation drivers

---

### `solver.py`
Generalized-α time integrator (JWH 2000 first-order form) and Newton–Raphson solver.

| Class / Function | Description |
|---|---|
| `GenAlphaParams(rho_inf)` | Frozen dataclass: α_m = (2−ρ∞)/(1+ρ∞), α_f = 1/(1+ρ∞), γ = ½+α_m−α_f. `rho_inf` ∈ [0,1]: 1 → midpoint (no dissipation), 0 → asymptotic annihilation of highest-frequency mode. **Note:** these are the JWH (2000) first-order formulas, which differ from the Chung & Hulbert (1993) second-order structural dynamics formulas — the LaTeX currently shows the second-order form and needs updating. |
| `newton_solve(residual_fn, d0, d_dot0, tol, max_iter, damping)` | Newton–Raphson with `jax.jacfwd` tangent and backtracking line-search (≤4 halvings per iteration). Convergence when ‖R‖_∞ < tol AND ‖δd‖/max(‖d‖,1) < tol. Returns (d_new, d_dot_new, iters, residual_history). |
| `TimeStepper(residual_fn, cache, params, genalpha, dt)` | Generalized-α driver. `.step(state)` advances one dt; `.run(state, n_steps, callbacks)` returns full history. Passes `cache=None, params=None` to select "flat residual" mode (used by ODE tests). |
| `TimeState(d, d_dot, t)` | State dataclass carried between time steps. |
| `apply_dirichlet_flat(R, K, dof_indices, values, current)` | Same convention as `assembler.apply_dirichlet` but accepts plain iterables; useful for solver-level unit tests. |
| `save_state(state, path, dt, seed)` | Persist TimeState to NPZ checkpoint. |
| `load_state(path)` | Load checkpoint → (TimeState, dt, seed). |
| `spectrum(residual_fn, d, d_dot, k)` | k largest-magnitude eigenvalues of the Newton tangent via `jnp.linalg.eigvals`. Diagnostic only; for large problems use SciPy ARPACK. |

**Depends on:** JAX (no hard dep on `assembler.py` — tested against analytic ODEs independently)

---

### `config.py`
YAML → validated `Problem` dataclass. Raises `ConfigError` (a `ValueError` subclass) on any missing field or out-of-range value; never silently defaults physics.

| Function / class | Description |
|---|---|
| `load_problem(path)` | Parse a YAML file → `Problem` |
| `from_dict(raw)` | Validate a plain dict → `Problem` (useful in tests) |
| `dump_problem(problem, path)` | Round-trip write back to YAML |
| `Problem` | Frozen nested dataclass containing: `MeshSpec` (n_ctrl, R_max), `DiscretizationSpec` (degree, n_gauss), `TimeSpec` (dt, t_end, rho_inf), `MaterialSpec` (Re, We, Pr, gamma), `InitialSpec` (kind, R_bubble, interface_width, rho_liq, rho_vap, vartheta_0), `BoundarySpec` (inner, outer), `OutputSpec` (path, every, format) |

An example YAML lives at `examples/bubble_collapse.yaml`. `n_gauss` defaults to `degree + 1` if omitted.

---

### `initial_conditions.py`
Initial condition utilities for the spherical bubble problem.

| Function | Description |
|---|---|
| `from_bspline_projection(field_fn, knots, degree, n_quad, R_max)` | L² projection of a physical-space function f(r) onto the B-spline space. Solves M c = b (unweighted L², no r² factor). Polynomial functions representable by the basis are recovered exactly. |
| `bubble_profile(...)` | Analytic tanh ρ profile + equilibrium V, uniform ϑ, zero u. Two modes: (1) pointwise on an r-array; (2) control-point projection when `knots, degree, n_quad` are supplied. Returns dict with keys `rho, u, vartheta, V`. |

Density profile: ρ(r) = ½(ρ_liq + ρ_vap) + ½(ρ_liq − ρ_vap)·tanh((r − R_bubble)/w).  
V initialized from chemical equilibrium: V = ν_loc(ρ, ϑ)/ϑ.

---

### `io_vtk.py`
Per-timestep snapshot output for ParaView.

| Function | Description |
|---|---|
| `write_xdmf_timestep(path, t, r_grid, fields_dict)` | Primary: XDMF + HDF5 (requires `h5py`). Fallback: CSV + PVD index. Encodes time in filename suffix `_tNNNNNNNNN`. Creates parent directories automatically. |
| `read_csv_snapshot(path)` | Round-trip CSV reader → (r_array, fields_dict). |
| `available_backends()` | Returns tuple of active backends, e.g. `("xdmf-h5", "csv-pvd")`. |

The fallback CSV path is always present. `h5py` enables the primary XDMF path; `meshio` is detected but not yet used.

---

### `postprocess.py`
Simulation diagnostics. Cache interface is duck-typed (accepts both dataclass and dict) using keys: `r_q, w_q, N_rho, dN_rho, N_u, dN_u, N_vartheta, dN_vartheta`.

| Function | Description |
|---|---|
| `bubble_radius(rho_field, r_grid, threshold)` | Smallest r where ρ crosses `threshold` via linear interpolation. Returns `nan` if no crossing. |
| `total_free_energy(ctrl, cache, params)` | ∫[Ψ_loc(ρ,ϑ) + (∂ρ/∂r)²/(2·We·ρ)] r² dr |
| `total_internal_energy(ctrl, cache, params)` | ∫ρE r² dr (E includes gradient and kinetic terms) |
| `mass_conservation_error(history, cache)` | Relative drift ‖M_k − M_0‖/M_0 at each snapshot in `history` |
| `entropy_production_rate(ctrl, ctrl_dot, cache, params)` | Global viscous + thermal + Korteweg dissipation budget ∫(…) r² dr |

---

### `scales.py`
Dimensional-scale registry. Implements Liu-Landis-Gomez-Hughes (CMAME 2015) reference scales.

| Class / Function | Description |
|---|---|
| `ReferenceScales(rho_c, vartheta_c, L_c, u_c, p_c, Re, We, Pr, gamma)` | Frozen dataclass. Derives `t_c = L_c / u_c`. Validates all scales positive. Methods: `nondimensionalize(value, kind)`, `dimensionalize(value, kind)`. `kind` ∈ {density, velocity, temperature, pressure, time, length}. |
| `default_water_vapor_scales(L_c)` | Illustrative water-vapor scales (ρ_c=322 kg/m³, ϑ_c=647 K, p_c=22 MPa). Order-of-magnitude only — use explicit `ReferenceScales` for production runs. |

---

### `_repro.py`
Seed pinning and version snapshot for reproducible runs.

| Function | Description |
|---|---|
| `set_reproducible(seed)` | Seeds `random`, `numpy`, and JAX; returns dict with seed, numpy/jax version strings, and JAX PRNG key. Raises ValueError for negative seed. |
