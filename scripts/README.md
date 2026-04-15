# scripts/

Small, single-purpose helper scripts used during development. Each
script is self-contained: run with `python scripts/<name>.py`.

## Inventory

| Script | Purpose |
| ------ | ------- |
| `run_sample_case.py` | Placeholder entry point for a full bubble-collapse simulation. Prints configuration and exits until the solver lands (Phase D). |
| `regen_regression_fixtures.py` | Regenerate the NPZ fixtures under `tests/regression/fixtures/`. Skeleton only. |

Scripts are not importable from `src/` and are not covered by CI
coverage.
