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
