# tests — Test Suite

~190 tests across 18 modules. Run with `pytest tests/` from the repo root.

Each test module mirrors a `src/` module. Tests verify mathematical identities, physical limits, known values, and equilibrium conditions — not just "does it run."

Fast suite (skips slow regression cases):
```bash
pytest tests/ -v --ignore=tests/regression
```

---

## `test_bsplines.py` — 8 tests

| Test | What it checks |
|---|---|
| Knot vector length and clamping | Structural: open-end conditions satisfied |
| Partition of unity: Σ B_A(ξ) = 1 | Completeness identity |
| Non-negativity: B_A ≥ 0 everywhere | Required for stable interpolation |
| Endpoint interpolation | Open splines pass through first/last control point |
| Derivative order 1 | Matches finite-difference approximation |
| Derivative order 2 | Matches finite-difference approximation |
| Derivative order 0 | Returns original basis (identity) |
| Derivative order > degree | Returns zero (exact annihilation) |

---

## `test_quadrature.py` — ~10 tests

| Test | What it checks |
|---|---|
| Point and weight count | Structural |
| Weights sum to 2 on [−1, 1] | Normalization |
| Exact polynomial integration to degree 2n−1 | Quadrature exactness property |
| Quadrature domain bounds | All r_q satisfy 0 ≤ r_q ≤ R_max |
| Coordinate mapping r = R_max·ξ | Affine map correctness |
| ∫ dr = R_max | Volume normalization |
| ∫ r² dr = R_max³/3 | Spherical volume factor |
| ∫ r³ dr = R_max⁴/4 | Polynomial exactness |

---

## `test_constitutive.py` — 14 tests

| Test | What it checks |
|---|---|
| Critical point p(1/3, 1, 1.4) = 1/27 | Known van der Waals critical value |
| Ideal gas limit (low density) | p → ρϑ as ρ → 0 |
| Spinodal region dp/dρ < 0 | Phase instability exists in correct range |
| Entropy sign | σ > 0 for physical states |
| Internal energy formula | Matches thermodynamic identity e = Ψ + ϑσ |
| Viscous stress traceless: τ_rr + 2τ_θθ = 0 | Stokes hypothesis |
| Uniform expansion u_r = Cr → τ = 0 | Incompressible-limit zero viscous stress |
| τ_θθ = −τ_rr/2 | Symmetry from tracelessness |
| Korteweg zero when ∂ρ/∂r = 0, ∂²ρ/∂r² = 0 | No-gradient limit |
| ς_θθ − ς_rr = (1/We)·(∂ρ/∂r)² | Capillary tensor identity |
| Heat flux direction: q_r < 0 for ∂ϑ/∂r > 0 | Fourier's law sign |
| Thermal conductivity formula | κ* = 8γ/[27(γ−1)·Re·Pr] |
| Interstitial working zero for ∇·u = 0 | Incompressible-flow limit |
| Interstitial working zero when ∂ρ/∂r = 0 | No-gradient limit |

---

## `test_residuals.py` — 12 tests

| Test | What it checks |
|---|---|
| Mass residual shape | Output is [n_basis] |
| Mass: zero for u=0, ρ̇=0 | Static equilibrium |
| Mass: constant ρ̇ gives ∫ N_C·r² dr | Verifies spatial integration |
| Momentum residual shape | Output is [n_basis] |
| Momentum: zero for static uniform flow | No-force equilibrium |
| Momentum: uniform expansion u_r = Cr → zero | Physical limit |
| Energy residual shape | Output is [n_basis] |
| Energy: zero at full thermodynamic equilibrium | u=0, no time derivatives, no source |
| Energy: nonzero under disequilibrium | Non-trivial residual |
| Auxiliary residual shape | Output is [n_basis] |
| Auxiliary: zero at chemical equilibrium (∂ρ/∂r = 0) | V = ν_loc/ϑ − ½u²/ϑ |
| Auxiliary: nonzero out of equilibrium | Non-trivial residual |

---

## `test_assembler.py` — 13 tests

| Test | What it checks |
|---|---|
| IEN shape | (n_elem, p+1) |
| IEN overlap | Consecutive elements share exactly `degree` DOFs |
| IEN monotone and bounds | Rows are [e, e+1, …, e+p]; min=0, max=n_ctrl−1 |
| Basis cache shapes | N, dN, d²N, r, w, J, IEN all correct shapes |
| Basis cache partition of unity | Σ N_A = 1 at every quadrature point |
| Basis cache r range | All r_q ∈ [0, R_max] |
| Single-element mesh equals element residual | 1-element global assembly matches manual element call for all 4 blocks |
| Assembly linearity in ctrl_dot | R is affine in time derivatives; verifies scatter-add correctness |
| Assembly zero on equilibrium | Uniform ρ, ϑ, V=ν_loc/ϑ, u=0 → R≡0 on interior DOFs; far-field BC absorbs pressure term |
| Dirichlet application | After apply_dirichlet, Newton step drives constrained DOFs to exact prescribed values |
| Dirichlet residual-only (K=None) | Residual modified, K returned as None |
| Symmetry BC at origin | Pins first u-DOF only; other fields have natural BCs at r=0 |
| Dirichlet far-field | Constrains last ρ, u, ϑ DOFs; V left free |

---

## `test_autodiff.py` — 8 tests

Autodiff vs. central finite-difference cross-check on all four element residuals. Each test freezes all but one control-point array and verifies `jax.jacfwd` matches FD to ~1e-6.

| Test | Residual | Differentiated w.r.t. |
|---|---|---|
| `test_autodiff_mass_vs_fd` | mass | ctrl_rho |
| `test_autodiff_momentum_vs_fd_wrt_u` | momentum | ctrl_u |
| `test_autodiff_momentum_vs_fd_wrt_rho` | momentum | ctrl_rho (2nd-derivative terms; 5e-5 tol) |
| `test_autodiff_energy_vs_fd_wrt_vartheta` | energy | ctrl_vartheta |
| `test_autodiff_energy_vs_fd_wrt_u` | energy | ctrl_u |
| `test_autodiff_auxiliary_vs_fd_wrt_V` | auxiliary | ctrl_V |
| `test_autodiff_auxiliary_vs_fd_wrt_rho` | auxiliary | ctrl_rho |
| `test_reverse_mode_matches_forward_mode_mass` | mass | jacrev == jacfwd |

---

## `test_cmame_reference.py` — ~27 test cases (parametrized)

Pins constitutive outputs against the closed-form CMAME 2015 expressions at 5 reference points: critical point (ρ=1/3, ϑ=1), vapour-subcritical, liquid-subcritical, supercritical-hot, and an alt-γ/Pr case.

| Test group | What it checks |
|---|---|
| `test_pressure_matches_cmame_form` [×5] | p(ρ, ϑ) = 8ρϑ/[27(1−ρ)] − ρ² to 1e-12 |
| `test_entropy_matches_cmame_form` [×5] | s = −(8/27)log(ρ/(1−ρ)) + (8/[27(γ−1)])log(ϑ) |
| `test_internal_energy_matches_cmame_form` [×5] | ι = −ρ + 8ϑ/[27(γ−1)] |
| `test_kappa_star_matches_cmame_form` [×5] | κ* = 8γ/[27(γ−1)Re·Pr] |
| `test_hand_computed_reference_values` [×2] | Critical point (ρ=1/3) and midpoint (ρ=1/2) exact float values |
| `test_kappa_star_independent_of_rho_and_vartheta` | κ* depends only on Re, Pr, γ; lower Pr ⇒ higher κ* |

---

## `test_config.py` — 11 tests

| Test | What it checks |
|---|---|
| Example file loads | `examples/bubble_collapse.yaml` parses to a valid Problem |
| Round-trip | `from_dict` → `dump_problem` → `load_problem` produces equal Problem |
| Missing section raises ConfigError | Removing `time` section raises with matching message |
| Missing field raises ConfigError | Removing `material.We` raises |
| Invalid value raises | `rho_inf = 2.0` (out of [0,1]) raises |
| Degree vs n_ctrl inconsistency | degree=5, n_ctrl=2 raises ConfigError |
| Default n_gauss | Omitting `n_gauss` defaults to degree+1 |
| Default output format | Omitting `format` defaults to "xdmf" |
| Load nonexistent file | FileNotFoundError raised |
| Invalid boundary type | `boundary.inner = "wibble"` raises ConfigError |
| YAML syntax valid | `bubble_collapse.yaml` parses via `yaml.safe_load` |

---

## `test_dead_code.py` — 8 tests

Tests for `free_energy_loc` and `total_energy` (not directly called by residuals but used by postprocessing and diagnostics).

| Test | What it checks |
|---|---|
| Free energy at critical point | Exact value at (ρ=1/3, ϑ=1) |
| Free energy / chemical potential consistency | ν_loc = ∂(ρΨ_loc)/∂ρ via autodiff |
| Free energy ideal-gas limit | Formula correct as ρ→0 |
| Free energy pressure Maxwell identity | p = ρ²·∂Ψ/∂ρ via autodiff |
| Total energy equals components sum | E = ι + (∇ρ)²/(2Weρ) + ½u² |
| Total energy: zero gradient + zero velocity → ι_loc | Simplification check |
| Total energy: gradient term always ≥ 0 | Capillary stability |
| Total energy: kinetic term scales as ½u² | Quadratic velocity dependence |

---

## `test_initial_conditions.py` — 10 tests

| Test | What it checks |
|---|---|
| Interface center | ρ(R_bubble) = ½(ρ_liq + ρ_vap) (tanh(0) = 0) |
| Far-field liquid | r >> R_bubble → ρ → ρ_liq |
| Center vapor | r = 0 → ρ → ρ_vap |
| Uniform temperature | ϑ ≡ ϑ_0 everywhere; u ≡ 0 |
| Keys | Output dict has {rho, u, vartheta, V} |
| Projection recovers constant | f = c → all control points = c |
| Projection recovers linear | f = 2r+1 exact for p≥1 |
| Projection recovers quadratic | f = ½r²+3r−2 exact for p=2 |
| Control-point mode shape | Returns arrays of length n_ctrl |
| Error without r or knots | ValueError raised |

---

## `test_input_guards.py` — 8 tests

Tests for opt-in runtime range checks on `constitutive.py` (via `enable_input_checks`).

| Test | What it checks |
|---|---|
| Guards disabled by default | `input_checks_enabled()` returns False |
| Valid inputs accepted | No raise for ρ=0.3, ϑ=1.1 |
| Bad ρ raises | ρ=1.2 (≥1) triggers check |
| Bad ϑ raises | ϑ=−0.1 triggers check |
| Zero ρ raises | ρ=0.0 triggers check |
| JIT works with guards off | `jax.jit(pressure)` compiles and returns finite value |
| Checkify wraps traced guards | Guards funnel through `jax.experimental.checkify` inside JIT |
| Toggle roundtrip | Enable/disable/enable leaves flag in expected state |

---

## `test_io_vtk.py` — 7 tests

| Test | What it checks |
|---|---|
| Available backends contains fallback | "csv-pvd" always present |
| Write creates files | Both output files exist after write |
| Multiple timesteps PVD collection | PVD indexes ≥2 datasets (CSV fallback path) |
| CSV roundtrip | r and all field arrays survive write→read exactly |
| Field shape mismatch raises | ValueError when field length ≠ len(r_grid) |
| Creates parent directory | Nested output paths auto-created |
| XDMF parses as valid XML | Generated .xdmf file has correct root tag (h5py path) |

---

## `test_mms.py` — 7 tests

Manufactured-solution (MMS) patch tests using a two-node linear hat basis over a single element [0.5, 1.5] and sympy for analytic reference integrals.

| Test | What it checks |
|---|---|
| Mass: linear ρ, linear u | R^ρ_C matches ∫[N_C ρ̇ − dN_C/dr · ρu] r² dr exactly |
| Auxiliary: zero gradient, constant V | R^V_C = const · ∫N_C r² dr (sympy) |
| Auxiliary: linear ρ, constant V | Both N and dN terms; reference via scipy.integrate.quad |
| Mass convergence rate | Residual differences decrease monotonically under h-refinement |
| Momentum: constant body force | R^u_C = −b_r·ρ·∫N_C r² dr (all stress terms drop) |
| Energy: constant heat source | R^E_C = −ρ·r_s·∫N_C r² dr (all flux terms drop) |
| Mass patch test: const ρ, u=0 | R^ρ ≡ 0 for any mesh |

---

## `test_phase_field_div.py` — 4 tests

Tests for `phase_field_div(dρ/dr, d²ρ/dr², dϑ/dr, ϑ, r)` — spherical ∇·(∇ρ/ϑ).

| Test | What it checks |
|---|---|
| Public name exported | `residuals.phase_field_div` exists and equals `residuals._phase_field_div` |
| Constant ϑ reduces to Laplacian | Result = (1/ϑ)(∂²ρ/∂r² + 2/r · ∂ρ/∂r) when ∂ϑ/∂r=0 |
| Constant ρ returns zero | All ρ-derivative terms zero → result=0 |
| Matches FD on polynomial field | Compares against (1/r²) d/dr[r²(1/ϑ)∂ρ/∂r] via central differences |

---

## `test_postprocess.py` — 12 tests

Uses a lightweight local cache (duck-typed dataclass) built from bsplines/quadrature, independent of `assembler.py`.

| Test | What it checks |
|---|---|
| Bubble radius linear interpolation | Crossing at correct r for a step-function ρ |
| Bubble radius exact on node | Returns exact r when ρ(r)=threshold at a grid point |
| Bubble radius no crossing | Returns nan |
| Total free energy finite and real | Positive finite value for physical state |
| Total internal energy scaling with ρ | Doubling ρ increases ‖ρE‖ by >50% |
| Mass conservation on steady state | error=0 when ρ never changes |
| Mass conservation detects drift | error>0 when ρ increases |
| Mass integral = sphere volume × density | ∫ρ r² dr = R_max³/3 for ρ≡1 |
| Entropy production: rest state = 0 | No velocity, no gradients → zero |
| Entropy production finite with gradients | Non-trivial linear-u profile gives finite value |
| Dict cache supported | Postprocess functions accept dict cache as well as dataclass |
| Missing param raises | ValueError when 'We' omitted from params |

---

## `test_repro.py` — 6 tests

| Test | What it checks |
|---|---|
| Returns seed in snapshot | snap["seed"] == supplied seed |
| Seeds numpy legacy RNG | Two calls with same seed produce identical draws |
| Seeds JAX | Same seed → same PRNG key |
| Different seeds differ | Distinct jax_prng_key for seed 0 vs 1 |
| Rejects negative seed | ValueError raised |
| Records version strings | snap contains numpy and jax version strings |

---

## `test_scales.py` — ~17 test cases (parametrized)

| Test | What it checks |
|---|---|
| Derived time scale | t_c = L_c / u_c |
| Round-trip float [×6 kinds] | nondim then dim recovers original value to 1e-14 |
| Round-trip array [×6 kinds] | Same for numpy arrays |
| Nondim density → 1 | rho_c / rho_c = 1 |
| Unknown kind raises | ValueError for kind="mass" |
| Negative scale raises | ValueError on construction |
| Default water-vapor scales constructible | t_c > 0; pressure round-trip works |

---

## `test_solver.py` — ~15 test cases (parametrized)

| Test | What it checks |
|---|---|
| GenAlpha parameters consistency [×5 rho_inf values] | α_m, α_f, γ match their analytic formulas |
| GenAlpha rejects out-of-range rho_inf | ValueError for rho_inf<0 or >1 |
| Second-order accuracy | Global error on ẏ=λy halves by ~4 under dt-refinement |
| High-frequency damping | rho_inf=0 damps fast mode orders of magnitude more than rho_inf=1 |
| Newton linear: 1 iteration | Converges in exactly one step for linear R(x)=Ax−b |
| Newton quadratic convergence | y²−2=0 shows quadratic convergence rate |
| Newton backtracking activates | x³−1=0 from far start converges monotonically |
| Checkpoint roundtrip | save_state/load_state recovers d, d_dot, t, dt, seed exactly |
| Spectrum shape | spectrum() returns (k,) complex array; largest two magnitudes match known eigenvalues |
| Step history recording | Every step appends {newton_iters, residual_norms} to history |
| Stepper wires to assembler | Confirms assembler symbols present (skips if Phase B not available) |
