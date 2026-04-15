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

---

## [2026-04-15] — Session 4: Five-phase parallel push, Phase E polish

### The plan
With Milestones 1–3 done, the remaining work (assembler, solver, I/O,
regression, tooling) was split into five phases that can progress in
parallel without stepping on each other's source files:

- **Phase A** — input layer: YAML config loader, CLI entry point, problem
  registry.
- **Phase B** — assembly layer: `src/assembler.py`, sparse pattern
  bookkeeping, dof maps.
- **Phase C** — solver layer: `src/solver.py` with General-α +
  Newton–Raphson, AutoDiff K_tan, eigen-diagnostics.
- **Phase D** — physics closure + I/O: boundary/initial conditions, VTK
  output, conservation monitors.
- **Phase E** — infrastructure & polish *(this session)*: authoritative
  `pyproject.toml`, CI upgrade, regression skeleton, docs, trackers.

Each phase runs on its own branch (`phaseX/<slug>`) and merges to `main`
independently. Source-file scope is disjoint; where overlap is
unavoidable (trackers, README) Phase E owns the edit and the others
rebase.

### Completed (Phase E)
- `pyproject.toml` — authoritative config: ruff (E/F/I/B/NPY/RUF/UP/SIM,
  line 100, py313), per-file ignores, pytest (`testpaths`,
  `--strict-markers`, `regression` marker), coverage (`source=src`,
  branch, `fail_under=80`), mypy strict with JAX overrides.
- `.github/workflows/ci.yml` — jobs split into `lint`, `typecheck`
  (continue-on-error), `test` (matrix on 3.13, coverage artifact
  upload), `docs-build` (optional), and a manually-triggered
  `regression` job gated on `workflow_dispatch`.
- `tests/regression/` — package skeleton with three skipped placeholders
  (manufactured solution, bubble-collapse smoke, conservation budget)
  plus a `fixtures/` directory + README documenting the NPZ format.
- `src/_repro.py` + `tests/test_repro.py` — `set_reproducible(seed)`
  seeds `random`, numpy, JAX and returns a version snapshot. The one
  new `src/*.py` allowed this session (greenfield, no conflict).
- `CONTRIBUTING.md` — naming (θ vs ϑ → `vartheta`), type-hint style,
  testing expectations, branch/commit conventions.
- `docs/architecture.md` — ASCII layer diagram
  (bsplines/quadrature → constitutive → residuals → assembler → solver
  → postprocess/io) + module table + CMAME 2015 citation.
- `docs/conf.py`, `docs/index.rst` — Sphinx skeleton, autodoc-ready.
- `scripts/` — `run_sample_case.py` placeholder and
  `regen_regression_fixtures.py` skeleton.
- `.pre-commit-config.yaml` — ruff, ruff-format, check-yaml,
  end-of-file-fixer, trailing-whitespace, check-added-large-files.
- `README.md` — installation, tests, sample-case command, citation
  block, repo layout.
- `TODO.md` — Phases A–D marked `[in-flight]`, new tracked items
  (VTK output, YAML config, sparse solvers, eigendiagnostics,
  conservation monitors, regression fixtures).

### Decisions
- Coverage threshold starts at **80 %** (not 85 %) so Phases B–D can
  land without immediately failing CI; ratchet after Phase D merges.
- mypy is **non-blocking** until the assembler/solver types settle —
  strict mode is configured but the job carries `continue-on-error`.
- Regression tests use a dedicated `regression` pytest marker plus a
  CI job that only fires on `workflow_dispatch`; they never run on
  routine PRs.
- Fixture NPZ files under `tests/regression/fixtures/` stay in git only
  while small (< 200 kB); larger ones will move to Git LFS.
