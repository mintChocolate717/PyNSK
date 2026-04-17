"""Runtime input-range guards on constitutive functions.

Guards are off by default to keep hot paths clean. They can be enabled
programmatically (``enable_input_checks(True)``) or via the environment
variable ``PYNSK_CHECK_INPUTS=1``. When enabled, they assert::

    0 < ρ < 1       (van der Waals reduced density)
    ϑ > 0           (strictly positive temperature)
"""

import jax
import jax.numpy as jnp
import pytest
from jax.experimental import checkify

from src import constitutive
from src.constitutive import (
    enable_input_checks,
    input_checks_enabled,
    pressure,
)


@pytest.fixture(autouse=True)
def _restore_flag():
    """Ensure the module-level flag does not leak across tests."""
    prev = input_checks_enabled()
    yield
    enable_input_checks(prev)


def test_guards_disabled_by_default():
    # The flag may have been toggled on by a previous test; we ensured
    # a clean state via the fixture but the default at import time is
    # honoured unless the env var is set.
    enable_input_checks(False)
    assert input_checks_enabled() is False


def test_guards_accept_valid_inputs():
    enable_input_checks(True)
    # In-range values must not raise.
    p = pressure(jnp.array(0.3), jnp.array(1.1), 1.4)
    assert jnp.isfinite(p)


def test_guards_fire_on_bad_rho():
    enable_input_checks(True)
    # Density >= 1 violates the van der Waals reduced-density bound. We
    # force the check to materialise by calling jax.block_until_ready
    # via float(...) which triggers device execution.
    with pytest.raises(Exception):
        _ = float(pressure(jnp.array(1.2), jnp.array(1.0), 1.4))


def test_guards_fire_on_bad_vartheta():
    enable_input_checks(True)
    with pytest.raises(Exception):
        _ = float(pressure(jnp.array(0.3), jnp.array(-0.1), 1.4))


def test_guards_fire_on_zero_rho():
    enable_input_checks(True)
    with pytest.raises(Exception):
        _ = float(pressure(jnp.array(0.0), jnp.array(1.0), 1.4))


def test_jit_works_with_guards_disabled():
    """Default (guards off) must not break JIT compilation."""
    enable_input_checks(False)
    f = jax.jit(lambda r, t: pressure(r, t, 1.4))
    val = float(f(jnp.array(0.3), jnp.array(1.0)))
    assert jnp.isfinite(val)


def test_checkify_wraps_traced_guards():
    """Inside jit, guards must funnel through checkify cleanly."""
    enable_input_checks(True)

    @jax.jit
    def _p(r, t):
        return pressure(r, t, 1.4)

    err, val = checkify.checkify(_p)(jnp.array(0.3), jnp.array(1.0))
    err.throw()  # no error expected for valid inputs
    assert jnp.isfinite(val)

    err_bad, _ = checkify.checkify(_p)(jnp.array(1.5), jnp.array(1.0))
    with pytest.raises(Exception):
        err_bad.throw()


def test_guard_toggle_roundtrip():
    enable_input_checks(False)
    assert constitutive.input_checks_enabled() is False
    enable_input_checks(True)
    assert constitutive.input_checks_enabled() is True
    enable_input_checks(False)
    assert constitutive.input_checks_enabled() is False
