# Regression fixtures

Reference result NPZ files consumed by `tests/regression/test_canonical_cases.py`
live here.

Each fixture is an `numpy.savez`-generated archive with a small, documented
set of arrays. Fixtures are versioned in git only if they are small
(< 200 kB). Larger reference blobs should be stored with Git LFS or an
external artifact store and fetched by the CI regression job.

## Planned fixtures

| Filename                      | Produced by                                   | Arrays                        |
| ----------------------------- | --------------------------------------------- | ----------------------------- |
| `manufactured_steady.npz`     | `test_manufactured_solution_steady_state`     | `r`, `rho`, `u_r`, `vartheta` |
| `bubble_collapse_smoke.npz`   | `test_bubble_collapse_smoke`                  | `r`, `rho_t_end`              |
| `conservation_budget.npz`     | `test_conservation_error_budget_short_run`    | `t`, `dmass`, `dmom`, `dE`    |

## Regenerating

Fixtures are regenerated manually whenever the underlying physics or
discretization changes in a way that intentionally alters the answer. The
regeneration script will live at `scripts/regen_regression_fixtures.py`
(not yet implemented).
