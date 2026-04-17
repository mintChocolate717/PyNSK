# TODO

## Milestone 1: Spatial Foundation
- [x] `src/bsplines.py` — basis matrix, first and second derivative matrices
- [x] `src/quadrature.py` — Gauss-Legendre points/weights, coordinate mapping from reference element to physical domain

## Milestone 2: Physics
- [x] `src/constitutive.py` — van der Waals EOS, Korteweg stress, viscous stress, heat flux, chemical potential
- [x] Cross-check constitutive outputs against CMAME 2015 reference values (`tests/test_cmame_reference.py`)

## Milestone 3: Residuals (one PDE at a time)
- [x] `src/residuals.py` — mass residual R_mass
- [x] `src/residuals.py` — momentum residual R_momentum
- [x] `src/residuals.py` — energy residual R_energy
- [x] `src/residuals.py` — auxiliary (Korteweg/V) residual R_auxiliary
- [x] Verify each residual via manufactured solution patch test (`tests/test_mms.py`)
- [x] Expose `phase_field_div` with standalone tests (`tests/test_phase_field_div.py`)
- [ ] Symbolic weak-form cross-check — sympy expansion of strong form vs. the integrand used in `src/residuals.py` (Phase A7 timed out before this was written)

## Milestone 4: Assembly & Solver — parallel five-phase push
- [x] **Phase A** — θ→ϑ rename, MMS patch tests, autodiff-vs-FD Jacobian checks, CMAME reference fixtures, input-range guards, dead-code coverage
- [x] **Phase B** — `src/assembler.py`: element connectivity, basis cache, vectorized `jax.vmap` element loop, Dirichlet / symmetry BC helpers
- [x] **Phase C** — `src/solver.py`: General-α (Jansen–Whiting–Hulbert) + Newton–Raphson, AutoDiff K_tan via `jax.jacfwd`, TimeStepper, checkpoint / restart, `spectrum()` eigendiagnostic
- [x] **Phase D** — ICs, scales registry, postprocess (bubble radius, free-energy, conservation error, entropy production), VTK/XDMF I/O with CSV+PVD fallback, YAML config loader, convergence notebooks
- [x] **Phase E** — tooling polish: `pyproject.toml`, CI, regression skeleton, docs, `src/_repro.py`
- [x] Verify K_tan against finite difference of R (`tests/test_autodiff.py`)

## Milestone 5: Simulation
- [x] Initial conditions for bubble collapse problem (`src/initial_conditions.py`, tanh profile + L² projection)
- [ ] Boundary conditions wired end-to-end for a real bubble-collapse case (assembler + solver + config all connected)
- [ ] Time integration convergence study (`notebooks/03_temporal_convergence.ipynb` — skeleton exists, full-system sweep guarded on import)
- [ ] Spatial convergence study (`notebooks/02_spatial_convergence.ipynb` — density-projection rate is live; full-system MMS sweep guarded on import)
- [ ] Reproduce bubble collapse results (`notebooks/04_bubble_collapse.ipynb` — skeleton exists)

## Best-practice backlog
- [x] **VTK output** — `src/io_vtk.py` with XDMF+HDF5 primary and CSV+PVD fallback
- [x] **YAML config** — `src/config.py` with validated nested `Problem` dataclass; `examples/bubble_collapse.yaml`
- [x] **Eigendiagnostics** — `spectrum()` in `src/solver.py` for dense K_tan; TODO note to swap in ARPACK/Lanczos for large problems
- [x] **Conservation monitors** — `mass_conservation_error`, `total_free_energy`, `entropy_production_rate` in `src/postprocess.py`
- [x] **Reproducibility utility** — `src/_repro.py`
- [ ] **Sparse solvers** — migrate Newton linear solve to `scipy.sparse.linalg` / `jax.experimental.sparse` once Jacobian sparsity is stable; switch point marked `TODO(sparse)` at `segment_sum` scatters in `src/assembler.py`
- [ ] **Regression fixtures** — generate NPZ reference files under `tests/regression/fixtures/` once solver lands end-to-end; wire `scripts/regen_regression_fixtures.py`
- [ ] **Docs autodoc** — populate `docs/index.rst` automodule stubs once public API stabilises
- [ ] **Coverage ratchet** — raise `fail_under` from 80 → 85 after the simulation driver is wired
- [ ] **Strict mypy** — currently progressive (`warn_no_return`, `warn_unused_ignores`, `check_untyped_defs`, `warn_redundant_casts`); ratchet to full `strict = true` after annotating `src/assembler.py`, `src/solver.py`, `src/residuals.py`, `src/constitutive.py`

## Integration follow-ups (from Session 5)
- [ ] **Cache interface reconciliation** — `src/postprocess.py` duck-types the cache dict; verify its expected keys (`N_rho`, `dN_rho`, `N_u`, `dN_u`, `N_vartheta`, `dN_vartheta`, `N_V`, `dN_V`, `r_q`, `w_q`) match what `src/assembler.build_basis_cache` produces; add a thin adapter if not
- [ ] **JIT wrapping of `assemble_residual`** — cache carries Python ints, so callers need `jax.jit(..., static_argnames=("cache",))` or an array-only repack; document the pattern
- [ ] **Quadrature-weight convention** — `quadrature.quadrature_points` returns `w` with `J_e` folded in; `residuals.element_residual_*` multiply by `J_e` again; `assembler` unfolds before calling. Pick one convention and delete the other.
- [ ] **`apply_dirichlet` sign convention** — assembler uses `R[i] = current[i] − values[i]`; solver's `apply_dirichlet_flat` mirrors this; verify no downstream caller assumes the opposite sign

## LaTeX
- [ ] Complete `temporal-discretization/` — predictor/corrector formulas, parameter definitions (ρ∞ → αm, αf, γ)
- [ ] Complete `constitutive-laws/` — van der Waals free energy, all derived quantities
- [ ] Complete `numerical-implementation/` — boundary conditions, initial conditions, convergence criteria
