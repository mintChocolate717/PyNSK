# TODO

## Milestone 1: Spatial Foundation
- [x] `src/bsplines.py` — basis matrix, first and second derivative matrices
- [x] `src/quadrature.py` — Gauss-Legendre points/weights, coordinate mapping from reference element to physical domain

## Milestone 2: Physics
- [x] `src/constitutive.py` — van der Waals EOS, Korteweg stress, viscous stress, heat flux, chemical potential
- [ ] Cross-check constitutive outputs against CMAME2015 Table of material parameters

## Milestone 3: Residuals (one PDE at a time)
- [x] `src/residuals.py` — mass residual R_mass
- [x] `src/residuals.py` — momentum residual R_momentum
- [x] `src/residuals.py` — energy residual R_energy
- [x] `src/residuals.py` — auxiliary (Korteweg/V) residual R_auxiliary
- [ ] Verify each residual via manufactured solution patch test (MMS tests exist but only for simple forcing cases; full smooth-solution convergence study pending)

## Milestone 4: Assembly & Solver
- [x] **Phase A** — YAML config loader, problem registry (`src/config.py`)
- [x] **Phase B** — `src/assembler.py`: element loop, dof map, global R assembly, Dirichlet BC helpers; `jax.vmap` element loop; dense `segment_sum` scatter (sparse TODO marked in-code)
- [x] **Phase C** — `src/solver.py`: Generalized-α (JWH 2000 first-order form), Newton-Raphson with `jax.jacfwd` tangent, backtracking line-search, checkpoint/restart, eigen-diagnostics
- [x] **Phase D** — `src/initial_conditions.py` (tanh bubble profile, L² B-spline projection), `src/scales.py` (ReferenceScales), `src/io_vtk.py` (XDMF/HDF5 + CSV fallback), `src/postprocess.py` (bubble_radius, energy integrals, mass conservation, entropy production)
- [x] **Phase E** — tooling polish: `pyproject.toml`, CI, regression skeleton, docs, `src/_repro.py`
- [ ] Verify K_tan against finite difference of R (Phase C exit criterion — autodiff tests pass but explicit FD check of assembled global tangent not yet done)
- [ ] Migrate Newton linear solve to `scipy.sparse.linalg` (GMRES + ILU) once Jacobian sparsity pattern is stable (current impl is dense)

## Milestone 5: Simulation
- [ ] Wire `scripts/run_sample_case.py` to a real bubble-collapse run (currently placeholder)
- [ ] Initial and boundary conditions wired end-to-end (config → IC → assembler → solver loop)
- [ ] Time integration convergence study (verify second-order accuracy in time on a full PDE problem)
- [ ] Spatial convergence study (h-refinement)
- [ ] Reproduce bubble collapse results

## LaTeX — open issues (Bubble-Cavitation repo)
- [x] **Fix temporal-discretization.tex**: JWH (2000) first-order parameters (α_m = (2−ρ∞)/(1+ρ∞), α_f = 1/(1+ρ∞)) confirmed correct in current file — resolved.
- [ ] Complete `numerical-implementation/` — boundary conditions, initial conditions, convergence criteria

## PyNSK naming audit — fixes needed (audit-reports/03-pynsk-variable-collisions.md)

**Hard (fix before first end-to-end run):**
- [ ] **H3 (crash)** — add `postprocess_cache_from_assembler(cache)` adapter in `postprocess.py`: flattens `(n_elem, n_gauss)` → `(n_qp,)`, renames `"r"`→`"r_q"`, `"w"`→`"w_q"`, `"N"`→`"N_rho"`/`"N_u"`/etc. Current assembler output is incompatible with all postprocess functions.
- [ ] **H2** — rename `dirichlet_far_field` parameter `rho_inf` → `rho_far` in `assembler.py:486` and `test_assembler.py:416,418,433` (silent wrong-BC risk if spectral radius ever passed here)
- [ ] **H1** — rename `GenAlphaParams.gamma` → `GenAlphaParams.gamma_ga` in `solver.py:50,58,234,246` (latent clash with adiabatic index `gamma` in constitutive/residuals)

**Soft:**
- [ ] **S1** — replace `theta`/`theta_q` → `vartheta`/`vartheta_q` in `postprocess.py:118,119,215,221,222`; `theta_v` → `vartheta_v` in `initial_conditions.py:126-128`
- [ ] **S2** — inline `float(self.rho_inf)` directly in `GenAlphaParams.__post_init__` (removes 3-line `rho` alias that shadows fluid density name)
- [ ] **S3** — rename loop variable `n` → `fname` in `io_vtk._write_csv_pvd` line 204
- [ ] **S4** — update `src/README.md` `korteweg_stress` signature to include `r` as 4th positional argument
- [ ] **S5** — remove `d_vth`/`d_vth_dot` abbreviation in `assembler._element_residuals` lines 210,215 (keep `ctrl_vartheta` names after IEN gather)

## Best-practice backlog
- [ ] **Sparse solvers** — migrate Newton linear solve to `scipy.sparse.linalg` once Jacobian sparsity pattern is stable (GMRES with ILU preconditioner); IEN connectivity already provides sparsity pattern
- [ ] **Eigendiagnostics** — optional SciPy ARPACK pass to monitor smallest/largest eigenvalues of K_tan during spinodal regions (dense `spectrum()` is available but not production-scale)
- [ ] **Regression fixtures** — generate NPZ reference files under `tests/regression/fixtures/` once solver runs end-to-end; wire `scripts/regen_regression_fixtures.py`
- [ ] **Coverage ratchet** — raise `fail_under` from 80 to 85 after end-to-end simulation works
- [ ] **Docs autodoc** — populate `docs/index.rst` automodule stubs once public API stabilises
