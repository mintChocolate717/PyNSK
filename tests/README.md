# tests — Test Suite

44 tests across 4 modules. Run with `pytest tests/` from the repo root.

Each test module mirrors a `src/` module. Tests verify mathematical identities, physical limits, known values, and equilibrium conditions — not just "does it run."

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

## `test_quadrature.py` — 10 tests

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
