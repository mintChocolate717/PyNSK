# PyNSK Architecture

PyNSK is organised as a short stack of functional JAX modules. Each layer
depends only on the layers below it; there are no cycles. This reflects
the derivation order in the companion LaTeX (Liu, Landis, Gomez &
Hughes, *CMAME 297 (2015) 476–553*).

## Layer diagram

```
        +--------------------------------------------------+
        |           postprocess / io                       |
        |      src/postprocess.py, src/io_vtk.py           |
        |   bubble_radius, free energy, mass error,        |
        |   entropy production, XDMF/CSV snapshots         |
        +--------------------------------------------------+
                             ^
                             |
        +--------------------------------------------------+
        |                   solver                         |
        |              src/solver.py                       |
        |   GenAlphaParams, TimeStepper, newton_solve,     |
        |   save/load checkpoint, eigen-diagnostics        |
        +--------------------------------------------------+
                             ^
                             |
        +--------------------------------------------------+
        |                 assembler                        |
        |             src/assembler.py                     |
        |   element_connectivity, build_basis_cache,       |
        |   assemble_residual, apply_dirichlet BCs         |
        +--------------------------------------------------+
                             ^
                             |
        +--------------------------------------------------+
        |              residuals (src/residuals.py)        |
        |  R_mass, R_momentum, R_energy, R_auxiliary,      |
        |  phase_field_div                                  |
        +--------------------------------------------------+
                  ^                             ^
                  |                             |
     +------------+------------+   +------------+-------------+
     |  constitutive (src/     |   | bsplines + quadrature    |
     |  constitutive.py)       |   | (src/bsplines.py,        |
     |  vdW free energy, p,    |   |  src/quadrature.py)      |
     |  nu, s, tau, varsigma,  |   | basis_matrix, deriv,     |
     |  q_r, Pi_r, input guards|   | Gauss-Legendre mapping   |
     +-------------------------+   +--------------------------+

        +--------------------------------------------------+
        |          config / initial / scales               |
        |  src/config.py      YAML → Problem dataclass     |
        |  src/initial_conditions.py  tanh bubble IC,      |
        |                             B-spline projection  |
        |  src/scales.py      ReferenceScales, nondim      |
        +--------------------------------------------------+

        +--------------------------------------------------+
        |               utilities                          |
        |  src/_repro.py   seed pinning, version snapshot  |
        +--------------------------------------------------+
```

## Module responsibilities

| Module | Purpose | Key exports |
| ------ | ------- | ----------- |
| `src/bsplines.py` | Open-uniform knot vectors and basis evaluation / derivatives as precomputed float64 matrices. | `make_knot_vector`, `basis_matrix`, `basis_deriv_matrix` |
| `src/quadrature.py` | Gauss–Legendre points/weights and reference-to-physical coordinate mapping. | `gauss_legendre`, `quadrature_points`, `recommended_n_gauss` |
| `src/constitutive.py` | Dimensionless thermodynamic and mechanical closure relations for the van der Waals NSK fluid. Input-range guards (opt-in). | `pressure`, `entropy`, `chemical_potential`, `viscous_stress`, `korteweg_stress`, `heat_flux`, `interstitial_working`, `enable_input_checks` |
| `src/residuals.py` | Element-level weak-form residuals for mass, momentum, energy, and the auxiliary Korteweg equation. Also exports `phase_field_div`. | `element_residual_{mass,momentum,energy,auxiliary}`, `phase_field_div` |
| `src/assembler.py` | Scatters element residuals into the global 4-block vector via `jax.vmap` + `segment_sum`. Provides IEN, basis cache, and Dirichlet BC helpers. | `assemble_residual`, `build_basis_cache`, `apply_dirichlet`, `symmetry_bc_at_origin`, `dirichlet_far_field` |
| `src/solver.py` | Generalized-α time integration (JWH 2000 first-order form) and Newton–Raphson with `jax.jacfwd` tangent. Checkpoint/restart and eigen-diagnostics. | `GenAlphaParams`, `TimeStepper`, `newton_solve`, `save_state`, `load_state`, `spectrum` |
| `src/config.py` | YAML → validated `Problem` frozen dataclass. Raises `ConfigError` on any invalid field. | `load_problem`, `from_dict`, `dump_problem`, `Problem` |
| `src/initial_conditions.py` | Analytic tanh bubble density profile + B-spline L² projection for initial control-point arrays. | `bubble_profile`, `from_bspline_projection` |
| `src/io_vtk.py` | Per-timestep snapshot output: XDMF+HDF5 (primary) or CSV+PVD (fallback). | `write_xdmf_timestep`, `read_csv_snapshot`, `available_backends` |
| `src/postprocess.py` | Simulation diagnostics: bubble radius, free energy, mass conservation, entropy production. Duck-typed cache interface. | `bubble_radius`, `total_free_energy`, `total_internal_energy`, `mass_conservation_error`, `entropy_production_rate` |
| `src/scales.py` | Liu-Landis-Gomez-Hughes reference scales and nondimensionalization helpers. | `ReferenceScales`, `default_water_vapor_scales` |
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
5. **Solver independent of assembler.** `solver.py` has no hard import
   of `assembler.py`; it accepts any callable `R(d, d_dot, t)` so the
   integrator can be tested against analytic ODEs without the full PDE.

## Known LaTeX / code discrepancy

`temporal-discretization.tex` currently shows the Chung & Hulbert (1993)
second-order structural dynamics parameters:
```
α_m = (2ρ∞ − 1)/(ρ∞ + 1),   α_f = ρ∞/(ρ∞ + 1)
```
`solver.py` correctly implements the Jansen–Whiting–Hulbert (2000)
**first-order** form, which applies to the NSK system written as ḋ = …:
```
α_m = (2 − ρ∞)/(1 + ρ∞),   α_f = 1/(1 + ρ∞)
```
These are different formulas. For ρ∞ = 0 the second-order form gives
α_m = −1, γ = −0.5 (unstable); the first-order form gives α_m = 2,
γ = 1.5 (correct maximum dissipation). The LaTeX file needs updating
to match the code.

## Reference

Liu, J., Landis, C. M., Gomez, H., Hughes, T. J. R.
*Liquid–vapor phase transition: Thermomechanical theory, entropy stable
numerical formulation, and boiling simulations.*
Comput. Methods Appl. Mech. Engrg. **297** (2015) 476–553.
