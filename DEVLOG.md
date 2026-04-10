# DEVLOG

## [2026-04-10] — Session 1: Architecture, Setup, B-splines

### Decisions
- **Code alongside LaTeX**, section by section. Each residual is implemented and verified immediately after its LaTeX derivation is complete.
- **PyTorch throughout**: all residual functions return `torch.Tensor` so AutoDiff can differentiate the assembled residual w.r.t. control point DOFs to form K_tan in one shot.
- **Basis matrices are constants**: `bsplines.py` returns plain float64 tensors (no grad). AutoDiff only traces through the physics (residual computation), not through basis evaluation. Field values are then `N @ d` where `d.requires_grad=True`.
- **K_tan via AutoDiff**: before differentiating, `d_dot_{n+1}` is expressed as a function of `d_{n+1}` through the kinematic update so the full chain rule is captured in a single `torch.autograd.functional.jacobian` call.

### Architecture
```
src/
  bsplines.py      ← done
  quadrature.py    ← next
  constitutive.py
  residuals.py
  assembler.py
  solver.py
tests/
  test_bsplines.py ← done
```

### Completed
- `src/bsplines.py`: `make_knot_vector`, `basis_matrix`, `basis_deriv_matrix`
- `tests/test_bsplines.py`: partition of unity, non-negativity, endpoint interpolation, finite-difference derivative checks
- `environment.yml`, `TODO.md`, `DEVLOG.md`

### Algorithm gaps fixed (numerical-implementation.tex)
- Added explicit corrector formula: `d_dot_{n+1}^{i+1} = d_dot_{n+1}^i + Δd / (γ Δt)`
- Clarified K_tan AutoDiff: `d_dot_{n+1}` must be substituted via kinematic update before differentiating so both αf and αm contributions are captured
