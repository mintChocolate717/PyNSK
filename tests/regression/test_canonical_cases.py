"""Canonical regression cases for PyNSK.

These tests exercise the full solver stack end-to-end and are expected to be
the slowest in the suite. They are gated behind a manual CI job (see
``.github/workflows/ci.yml``) and are all currently skipped — they will be
enabled as the solver, assembler, and I/O layers land from phases B-D.

Reference results (NPZ files) will live under ``tests/regression/fixtures/``
and are loaded via :func:`load_reference`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_reference(name: str) -> dict[str, np.ndarray]:
    """Load a reference NPZ fixture by (stem) name.

    Parameters
    ----------
    name
        Filename stem (without ``.npz``) under ``tests/regression/fixtures/``.

    Returns
    -------
    dict[str, numpy.ndarray]
        Arrays from the NPZ archive, keyed by their stored name.
    """
    path = FIXTURES_DIR / f"{name}.npz"
    if not path.exists():
        pytest.skip(f"Reference fixture {path} not available yet.")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


@pytest.mark.regression
@pytest.mark.skip(reason="awaiting solver integration")
def test_manufactured_solution_steady_state() -> None:
    """Method of manufactured solutions — steady state.

    Drives the PDE system with a prescribed smooth (rho, u_r, vartheta) field
    and verifies the assembled residual plus Newton solver converge to the
    reference solution within expected spatial convergence rates.
    """
    raise NotImplementedError


@pytest.mark.regression
@pytest.mark.skip(reason="awaiting solver integration")
def test_bubble_collapse_smoke() -> None:
    """Bubble collapse smoke test.

    Short (few time-steps) integration of the canonical bubble-collapse IC to
    detect regressions in assembly, time integration, or constitutive wiring.
    Compares against a small NPZ fixture (density profile at t_end).
    """
    raise NotImplementedError


@pytest.mark.regression
@pytest.mark.skip(reason="awaiting solver integration")
def test_conservation_error_budget_short_run() -> None:
    """Conservation-error budget over a short run.

    Verifies mass, momentum, and total-energy drift stay within documented
    tolerances after N timesteps of a closed-domain configuration.
    """
    raise NotImplementedError
