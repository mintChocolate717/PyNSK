# DEVLOG

## [2026-04-10] — Session 1: Architecture, Setup, B-splines

### Decisions
- **Code alongside LaTeX**, section by section. Each residual is implemented and verified immediately after its LaTeX derivation is complete.
- **JAX throughout** (switched from PyTorch): functional autodiff via `jax.jacobian(f)(x)` is cleaner for forming K_tan than tape-based autograd. Float64 enabled globally in `src/__init__.py`.
- **Basis matrices are constants**: `bsplines.py` returns plain float64 JAX arrays (no grad). JAX differentiates through the physics only. Field values are `N @ d` where `d` is a JAX array passed to `jax.jacobian`.
- **K_tan via JAX AutoDiff**: `d_dot_{n+1}` expressed as function of `d_{n+1}` via kinematic update before differentiating so both αf and αm contributions are captured in one `jax.jacobian` call.

### Completed
- `src/bsplines.py`: `make_knot_vector`, `basis_matrix`, `basis_deriv_matrix`
- `tests/test_bsplines.py`: partition of unity, non-negativity, endpoint interpolation, finite-difference derivative checks
- `environment.yml`, `TODO.md`, `DEVLOG.md`

### LaTeX changes (Bubble-Cavitation repo)
- Fixed algorithm: added explicit corrector formula `d_dot_{n+1}^{i+1} = d_dot_{n+1}^i + Δd / (γ Δt)`
- Fixed K_tan AutoDiff clarification

---

## [2026-04-10] — Session 2: Quadrature, Repo Split

### Decisions
- Split into two repos: `Bubble-Cavitation` (LaTeX/Overleaf) and `PyNSK` (solver)
- GitHub Actions added: LaTeX compile + PDF artifact on `Bubble-Cavitation`; pytest + ruff on `PyNSK`

### Completed
- `src/quadrature.py`: `gauss_legendre`, `quadrature_points`, `recommended_n_gauss`
  - Coordinate mapping: r(ξ_ref) = (r_{e+1}-r_e)/2 * ξ_ref + (r_{e+1}+r_e)/2, J_e = (r_{e+1}-r_e)/2
  - r² spherical factor intentionally excluded — applied per-residual
- `tests/test_quadrature.py`: 13 tests — weight sums, exact polynomial integration, r=R_max*xi, ∫r²dr = R³/3

### LaTeX sections covered
- `spatial-discretization-Bsplines/coordinate-mapping.tex` → `src/quadrature.py`

---

## [2026-04-10] — Session 3: Constitutive Laws, Residuals

### Decisions
- **\vartheta** for temperature throughout LaTeX (avoids conflict with polar angle \theta in τ_{θθ}, ς_{θθ} subscripts).
- **Interstitial working sign**: Π_r = (1/We)ρ(∂u_r/∂r + 2u_r/r)(∂ρ/∂r) — positive sign follows Liu et al. Eq.(95).
- **∂(ρE)/∂t** computed analytically by expanding ρE term by term; avoids AD through constitutive functions at assembly time.

### Completed
- `src/constitutive.py` — Milestone 2:
  - `free_energy_loc`, `pressure`, `entropy`, `chemical_potential`, `internal_energy_loc`, `total_energy`
  - `viscous_stress`, `korteweg_stress`, `kappa_star`, `heat_flux`, `interstitial_working`
- `tests/test_constitutive.py` — Milestone 2 tests
- `src/residuals.py` — Milestone 3:
  - `element_residual_mass`, `element_residual_momentum`, `element_residual_energy`, `element_residual_auxiliary`
  - All use `jnp.einsum("q,qc->c", weights, integrand)` pattern; vectorized over quadrature points
- `tests/test_residuals.py` — 12 tests, 12 passing (commit 8ce10fc)

### LaTeX sections covered (Bubble-Cavitation repo)
- `constitutive-laws/thermodynamics/nondimensionalization.tex` — reference scales, Re/We/κ*
- `constitutive-laws/thermodynamics/free-energy.tex` — Ψ_loc, Ψ_cap
- `constitutive-laws/thermodynamics/derived-quantities.tex` — p, s, ν, ι, E
- `constitutive-laws/mechanical/viscous-stress.tex` — τ_rr, τ_{θθ}
- `constitutive-laws/mechanical/korteweg-stress.tex` — ς_rr, ς_{θθ}
- `constitutive-laws/thermal/heat-flux.tex` — q_r
- `constitutive-laws/thermal/interstitial-working.tex` — Π_r
