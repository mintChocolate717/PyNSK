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
- [ ] Verify each residual via manufactured solution patch test

## Milestone 4: Assembly & Solver — parallel five-phase push
- [in-flight] **Phase A** — YAML config loader, CLI entry point, problem registry (`src/config.py`, `src/problems/`)
- [in-flight] **Phase B** — `src/assembler.py`: element loop, dof map, global R assembly, sparse Jacobian pattern
- [in-flight] **Phase C** — `src/solver.py`: General-α predictor/corrector, Newton-Raphson, AutoDiff K_tan (`jax.jacobian` of assembled R w.r.t. `d_{n+1}`), line search
- [in-flight] **Phase D** — boundary / initial conditions, conservation monitors, VTK + NPZ I/O (`src/io.py`, `src/postprocess.py`)
- [x] **Phase E** — tooling polish: `pyproject.toml`, CI, regression skeleton, docs, `src/_repro.py`
- [ ] Verify K_tan against finite difference of R (Phase C exit criterion)

## Milestone 5: Simulation
- [ ] Initial and boundary conditions for bubble collapse problem
- [ ] Time integration convergence study (verify second-order accuracy in time)
- [ ] Spatial convergence study (h-refinement)
- [ ] Reproduce bubble collapse results

## Best-practice backlog (added Session 4)
- [ ] **VTK output** — per-timestep `.vtu` export of ρ, u_r, ϑ, p for ParaView (`src/io.py`)
- [ ] **YAML config** — declarative case files parsed into a typed `Config` dataclass (Phase A)
- [ ] **Sparse solvers** — migrate Newton linear solve to `scipy.sparse.linalg` (GMRES with ILU preconditioner) once Jacobian sparsity pattern is stable
- [ ] **Eigendiagnostics** — optional SciPy ARPACK pass to monitor smallest/largest eigenvalues of K_tan during spinodal regions
- [ ] **Conservation monitors** — running mass / momentum / total-energy drift logged per step; surfaced in `postprocess`
- [ ] **Regression fixtures** — generate NPZ reference files under `tests/regression/fixtures/` once solver lands; wire `scripts/regen_regression_fixtures.py`
- [ ] **Docs autodoc** — populate `docs/index.rst` automodule stubs once public API stabilises
- [ ] **Coverage ratchet** — raise `fail_under` from 80 to 85 after Phase D merges

## LaTeX
- [ ] Complete `temporal-discretization/` — predictor/corrector formulas, parameter definitions (ρ∞ → αm, αf, γ)
- [ ] Complete `constitutive-laws/` — van der Waals free energy, all derived quantities
- [ ] Complete `numerical-implementation/` — boundary conditions, initial conditions, convergence criteria
