# TODO

## Milestone 1: Spatial Foundation
- [x] `src/bsplines.py` — basis matrix, first and second derivative matrices
- [ ] `src/quadrature.py` — Gauss-Legendre points/weights, coordinate mapping from reference element to physical domain

## Milestone 2: Physics
- [ ] `src/constitutive.py` — van der Waals EOS, Korteweg stress, viscous stress, heat flux, chemical potential
- [ ] Cross-check constitutive outputs against CMAME2015 Table of material parameters

## Milestone 3: Residuals (one PDE at a time)
- [ ] `src/residuals.py` — mass residual R_mass
- [ ] `src/residuals.py` — momentum residual R_momentum
- [ ] `src/residuals.py` — energy residual R_energy
- [ ] `src/residuals.py` — auxiliary (Korteweg/V) residual R_auxiliary
- [ ] Verify each residual via manufactured solution patch test

## Milestone 4: Assembly & Solver
- [ ] `src/assembler.py` — element loop, global R assembly
- [ ] `src/solver.py` — General-α predictor, corrector, Newton-Raphson loop
- [ ] `src/solver.py` — AutoDiff K_tan (differentiate assembled R w.r.t. d_{n+1})
- [ ] Verify K_tan against finite difference of R

## Milestone 5: Simulation
- [ ] Initial and boundary conditions for bubble collapse problem
- [ ] Time integration convergence study (verify second-order accuracy in time)
- [ ] Spatial convergence study (h-refinement)
- [ ] Reproduce bubble collapse results

## LaTeX
- [ ] Complete `temporal-discretization/` — predictor/corrector formulas, parameter definitions (ρ∞ → αm, αf, γ)
- [ ] Complete `constitutive-laws/` — van der Waals free energy, all derived quantities
- [ ] Complete `numerical-implementation/` — boundary conditions, initial conditions, convergence criteria
