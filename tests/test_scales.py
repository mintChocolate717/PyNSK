"""Tests for src/scales.py (D2)."""
import numpy as np
import pytest

from src.scales import ReferenceScales, default_water_vapor_scales


@pytest.fixture
def scales() -> ReferenceScales:
    return ReferenceScales(
        rho_c=322.0,
        vartheta_c=647.0,
        L_c=1.0e-6,
        u_c=261.8,
        p_c=22.064e6,
        Re=100.0,
        We=1.0,
        Pr=7.0,
        gamma=1.4,
    )


def test_derived_time_scale(scales):
    assert np.isclose(scales.t_c, scales.L_c / scales.u_c)


@pytest.mark.parametrize(
    "kind",
    ["density", "velocity", "temperature", "pressure", "time", "length"],
)
def test_round_trip_float(scales, kind):
    v = 123.456
    out = scales.dimensionalize(scales.nondimensionalize(v, kind), kind)
    assert np.isclose(out, v, rtol=1e-14, atol=0.0)


@pytest.mark.parametrize(
    "kind",
    ["density", "velocity", "temperature", "pressure", "time", "length"],
)
def test_round_trip_array(scales, kind):
    v = np.array([1.0, 2.0, 3.0, 4.0])
    out = scales.dimensionalize(scales.nondimensionalize(v, kind), kind)
    np.testing.assert_allclose(out, v, rtol=1e-14)


def test_nondim_density(scales):
    # density = rho_c → dimensionless = 1
    assert np.isclose(scales.nondimensionalize(scales.rho_c, "density"), 1.0)


def test_unknown_kind_raises(scales):
    with pytest.raises(ValueError):
        scales.nondimensionalize(1.0, "mass")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        scales.dimensionalize(1.0, "mass")  # type: ignore[arg-type]


def test_negative_scale_raises():
    with pytest.raises(ValueError):
        ReferenceScales(
            rho_c=-1.0, vartheta_c=1.0, L_c=1.0, u_c=1.0, p_c=1.0,
            Re=1.0, We=1.0, Pr=1.0, gamma=1.4,
        )


def test_default_water_vapor_is_constructible():
    s = default_water_vapor_scales(L_c=1.0e-5)
    assert s.L_c == 1.0e-5
    assert s.t_c > 0.0
    # round-trip still works
    x = 3.0
    assert np.isclose(s.dimensionalize(s.nondimensionalize(x, "pressure"), "pressure"), x)
