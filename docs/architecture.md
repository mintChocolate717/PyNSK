# PyNSK Architecture

PyNSK is organised as a short stack of functional JAX modules. Each layer
depends only on the layers below it; there are no cycles. This reflects
the derivation order in the companion LaTeX (Liu, Landis, Gomez &
Hughes, *CMAME 297 (2015) 476–553*).

## Layer diagram

```
        +--------------------------------------------------+
        |           postprocess / io (future)              |
        |      VTK output, NPZ snapshots, diagnostics      |
        +--------------------------------------------------+
                             ^
                             |
        +--------------------------------------------------+
        |                solver (future)                   |
        |   General-alpha predictor/corrector, Newton,     |
        |   AutoDiff K_tan = jacobian(R) w.r.t. d_{n+1}    |
        +--------------------------------------------------+
                             ^
                             |
        +--------------------------------------------------+
        |               assembler (future)                 |
        |    element loop -> global residual R(d, d_dot)   |
        +--------------------------------------------------+
                             ^
                             |
        +--------------------------------------------------+
        |              residuals (src/residuals.py)        |
        |  R_mass, R_momentum, R_energy, R_auxiliary       |
        +--------------------------------------------------+
                  ^                             ^
                  |                             |
     +------------+------------+   +------------+-------------+
     |  constitutive (src/     |   | bsplines + quadrature    |
     |  constitutive.py)       |   | (src/bsplines.py,        |
     |  vdW free energy, p,    |   |  src/quadrature.py)      |
     |  nu, s, tau, varsigma,  |   | basis_matrix, deriv,     |
     |  q_r, Pi_r              |   | Gauss-Legendre mapping   |
     +-------------------------+   +--------------------------+
```

## Module responsibilities

| Module | Purpose | Key exports |
| ------ | ------- | ----------- |
| `src/bsplines.py` | Open-uniform knot vectors and basis evaluation / derivatives as precomputed float64 matrices. | `make_knot_vector`, `basis_matrix`, `basis_deriv_matrix` |
| `src/quadrature.py` | Gauss–Legendre points/weights and reference-to-physical coordinate mapping. | `gauss_legendre`, `quadrature_points`, `recommended_n_gauss` |
| `src/constitutive.py` | Dimensionless thermodynamic and mechanical closure relations for the van der Waals NSK fluid. | `pressure`, `entropy`, `chemical_potential`, `viscous_stress`, `korteweg_stress`, `heat_flux`, `interstitial_working` |
| `src/residuals.py` | Element-level weak-form residuals for mass, momentum, energy, and the auxiliary Korteweg equation. | `element_residual_{mass,momentum,energy,auxiliary}` |
| `src/assembler.py` *(future, Phase C)* | Scatters element residuals into the global vector. | — |
| `src/solver.py` *(future, Phase D)* | General-α time integration and Newton–Raphson with JAX tangent stiffness. | — |
| `src/io.py` / `src/postprocess.py` *(future)* | VTK export, NPZ snapshots, conservation monitors. | — |
| `src/_repro.py` | Seed pinning + version snapshot for reproducible runs. | `set_reproducible` |

## Design principles

1. **Functional, not object-oriented.** Each module exports pure
   functions over JAX arrays. State lives in caller-held arrays so that
   `jax.jit` / `jax.jacobian` / `jax.vmap` compose cleanly.
2. **Float64 everywhere.** Enabled globally in `src/__init__.py` — do
   not override per-module.
3. **Analytic where possible, AD where not.** Constitutive derivatives
   are written out by hand against the LaTeX; the tangent stiffness is
   obtained once, at assembly, via `jax.jacobian`.
4. **No implicit defaults for physics.** `Re`, `We`, `Pr`, `gamma` are
   always explicit arguments; there are no module-level constants.

## Reference

Liu, J., Landis, C. M., Gomez, H., Hughes, T. J. R.
*Liquid–vapor phase transition: Thermomechanical theory, entropy stable
numerical formulation, and boiling simulations.*
Comput. Methods Appl. Mech. Engrg. **297** (2015) 476–553.
