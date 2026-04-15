# Contributing to PyNSK

Thanks for your interest! PyNSK is a small research solver; we prioritise
readability and reproducibility over breadth. This document collects the
conventions we follow so a new contributor can match the existing style.

## Naming conventions

- `theta` (θ) is always the **polar angle** in spherical coordinates
  (e.g. the subscript in `tau_tt`, the basis for `varsigma_tt`).
- `vartheta` (ϑ) is the **temperature** — spelled out in code to avoid
  collisions with the polar-angle `theta`. Matches the LaTeX derivation
  (`\vartheta`) and the Liu et al. CMAME 2015 notation.
- `rho` — density, `u_r` — radial velocity, `r` — radial coordinate.
- `Re`, `We`, `Pr`, `gamma` — Reynolds, Weber, Prandtl, specific-heat
  ratio. Upper-case exactly as in the LaTeX.
- B-spline degree is `p`, element count `n_el`, quadrature count
  `n_gauss`.
- Private/internal modules start with a single leading underscore
  (`src/_repro.py`).

## Type-hint style

- Every public function in `src/` carries a full type signature
  (`def f(x: jnp.ndarray, Re: float) -> jnp.ndarray: ...`).
- Prefer PEP 604 unions (`float | None`) and built-in generics
  (`list[int]`, `dict[str, np.ndarray]`).
- Use `jax.Array` (via `jax.typing.ArrayLike` for inputs, `jax.Array` for
  outputs) where JAX tracing matters. Plain `jnp.ndarray` is acceptable
  in older code but should not be introduced in new modules.
- Module-level `from __future__ import annotations` is allowed but not
  required — we target Python 3.13 so postponed evaluation is cheap.
- `mypy --strict` runs in CI (non-blocking for now). Please do not
  introduce `# type: ignore` comments without a short rationale.

## Testing expectations

- Every new function lands with a test. Unit tests live in `tests/` and
  mirror the module layout (`src/foo.py` → `tests/test_foo.py`).
- Numerical checks use `jnp.allclose(..., atol=1e-10)` (or tighter) for
  closed-form identities and `rtol=1e-6` for float64 solver residuals.
- Long / end-to-end cases live in `tests/regression/` behind the
  `regression` marker and are only run manually in CI.
- Run the fast suite locally with:
  ```
  pytest tests/ -v --ignore=tests/regression
  ```
  and check coverage with `pytest --cov=src --cov-branch`.

## Branch & commit conventions

- Topic branches follow `phase<LETTER>/<slug>` (e.g. `phaseE/polish`) or
  `feat/<slug>`, `fix/<slug>` for smaller work.
- Commit subject lines: imperative, <= 72 chars, no trailing period.
  Body (wrapped at 72) explains the *why* when it is not obvious from
  the diff.
- Reference LaTeX sections in commit bodies when a commit implements an
  equation derived in the companion `Bubble-Cavitation` repository.
- One logical change per commit. Prefer `git commit --amend`/rebase over
  "fixup" commits on a topic branch before opening a PR.
- PR titles match the leading commit subject. PR descriptions list the
  tests that were added or updated.

## Tooling checklist before pushing

```
ruff check src/ tests/
ruff format --check src/ tests/
pytest tests/ -v --ignore=tests/regression
```

If you install `pre-commit` (`pip install pre-commit && pre-commit
install`), the same checks run automatically.
