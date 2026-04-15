"""Regenerate regression NPZ fixtures.

Skeleton: the solver does not yet produce reference results, so this
script currently only emits an informational message. It will grow into
a small driver that runs each canonical case and writes an NPZ archive
into ``tests/regression/fixtures/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "regression" / "fixtures"


def main() -> int:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fixtures directory: {FIXTURES_DIR}")
    print("No fixtures regenerated — solver integration pending (Phase D).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
