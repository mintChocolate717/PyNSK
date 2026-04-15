"""Tests for :mod:`src._repro`."""

from __future__ import annotations

import numpy as np
import pytest

from src._repro import set_reproducible


def test_set_reproducible_returns_seed() -> None:
    snap = set_reproducible(42)
    assert snap["seed"] == 42
    assert "numpy" in snap
    assert "jax" in snap


def test_set_reproducible_seeds_numpy() -> None:
    # Verifying that the legacy global RNG is seeded — `noqa: NPY002`.
    set_reproducible(123)
    a = np.random.rand(5)  # noqa: NPY002
    set_reproducible(123)
    b = np.random.rand(5)  # noqa: NPY002
    assert np.array_equal(a, b)


def test_set_reproducible_seeds_jax() -> None:
    import jax

    snap_a = set_reproducible(7)
    snap_b = set_reproducible(7)
    assert snap_a["jax_prng_key"] == snap_b["jax_prng_key"]

    # Sanity: the key should actually be a valid PRNGKey.
    key = jax.random.PRNGKey(7)
    assert snap_a["jax_prng_key"] == [int(x) for x in key]


def test_set_reproducible_different_seeds_differ() -> None:
    a = set_reproducible(0)
    b = set_reproducible(1)
    assert a["jax_prng_key"] != b["jax_prng_key"]


def test_set_reproducible_rejects_negative_seed() -> None:
    with pytest.raises(ValueError):
        set_reproducible(-1)


def test_set_reproducible_records_versions() -> None:
    snap = set_reproducible(0)
    assert isinstance(snap["python"], str)
    assert isinstance(snap["numpy"], str)
    assert isinstance(snap["jax"], str)
    assert snap["numpy"] != "unavailable"
    assert snap["jax"] != "unavailable"
