"""Placeholder entry point for the bubble-collapse sample case.

The real solver lands in Phase D. For now this script exists so the
README can advertise a stable command and so CI smoke-tests can import
:mod:`src` without touching physics-dependent code.

Usage::

    python scripts/run_sample_case.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``src`` importable when running from a checkout.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src._repro import set_reproducible  # noqa: E402


def main() -> int:
    snapshot = set_reproducible(seed=0)
    print("PyNSK sample-case runner (placeholder).")
    print(f"  python : {snapshot['python']}")
    print(f"  numpy  : {snapshot['numpy']}")
    print(f"  jax    : {snapshot['jax']}")
    print()
    print("The solver is not yet wired up (Phase D). This script will be")
    print("extended to drive an end-to-end bubble-collapse run once the")
    print("assembler and time integrator land.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
