"""Reproducibility helpers for PyNSK.

Tiny, dependency-light (stdlib + numpy + jax) utilities used to pin seeds
and log the version strings of every library that influences numerical
output. Kept deliberately small so it can be imported from tests and
scripts without pulling in the rest of the package.
"""

from __future__ import annotations

import logging
import os
import platform
import random
from typing import Any

logger = logging.getLogger(__name__)


def set_reproducible(seed: int = 0) -> dict[str, Any]:
    """Seed all RNGs PyNSK relies on and return a version snapshot.

    The function seeds :mod:`random`, :mod:`numpy`, and :mod:`jax` (via a
    fresh :class:`jax.random.PRNGKey`) and logs/returns the version strings
    of the libraries that influence numerical output. The returned snapshot
    is useful for embedding in saved result archives so stale fixtures can
    be detected.

    Parameters
    ----------
    seed
        Non-negative integer seed. Defaults to ``0``.

    Returns
    -------
    dict[str, Any]
        A dictionary with the seed, a JAX ``PRNGKey`` (as a list of ints),
        and version strings for ``python``, ``numpy``, ``jax`` and
        ``jaxlib``. Unavailable components are recorded as ``"unavailable"``.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    random.seed(seed)
    # Pin PYTHONHASHSEED for subprocesses spawned from this process.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    snapshot: dict[str, Any] = {
        "seed": seed,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    try:
        import numpy as np

        # Intentionally seed the legacy global RNG — downstream code and
        # tests may rely on ``np.random.rand`` etc. for reproducibility.
        np.random.seed(seed)  # noqa: NPY002
        snapshot["numpy"] = np.__version__
    except ImportError:  # pragma: no cover - numpy is a hard dep in practice
        snapshot["numpy"] = "unavailable"

    try:
        import jax

        key = jax.random.PRNGKey(seed)
        snapshot["jax"] = jax.__version__
        snapshot["jax_prng_key"] = [int(x) for x in key]
    except ImportError:  # pragma: no cover
        snapshot["jax"] = "unavailable"

    try:
        import jaxlib

        snapshot["jaxlib"] = jaxlib.__version__
    except ImportError:  # pragma: no cover
        snapshot["jaxlib"] = "unavailable"

    logger.info("PyNSK reproducibility snapshot: %s", snapshot)
    return snapshot
