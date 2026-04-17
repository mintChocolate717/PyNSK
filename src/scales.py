"""
Dimensional-scale registry (FEA best practice).

Implements the Liu-Landis-Gomez-Hughes reference scales (CMAME 2015) used
to non-dimensionalise the spherically-symmetric Navier-Stokes-Korteweg
equations, and the four dimensionless groups {Re, We, Pr, gamma} that
parameterise the resulting dimensionless system.

Reference scales (van der Waals EOS, with molecular-length scale L_c):

    rho_c  = b               (co-volume density)
    p_c    = a * b**2        (van der Waals pressure)
    theta_c = 8 * a * b / (27 * R_gas)
    u_c    = sqrt(p_c / rho_c) = b * sqrt(a)
    t_c    = L_c / u_c
    L_c    = user-chosen (e.g., bubble radius, capillary length)

Dimensionless groups (functions of the fluid transport properties and L_c):

    Re = rho_c * u_c * L_c / mu
    We = rho_c * u_c**2 * L_c**2 / lambda_cap    (capillary number inverse)
    Pr = mu * c_p / k_thermal
    gamma = c_p / c_v

All values are stored as Python floats and behave identically on CPU/GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Kind = Literal["density", "velocity", "temperature", "pressure", "time", "length"]

_ALLOWED_KINDS: tuple[str, ...] = (
    "density",
    "velocity",
    "temperature",
    "pressure",
    "time",
    "length",
)


@dataclass(frozen=True)
class ReferenceScales:
    """Liu-Landis-Gomez-Hughes reference scales and dimensionless groups.

    Attributes
    ----------
    rho_c :
        Density scale [kg/m^3].
    vartheta_c :
        Temperature scale [K].
    L_c :
        Length scale [m].
    u_c :
        Velocity scale [m/s].
    p_c :
        Pressure scale [Pa].
    t_c :
        Time scale [s] = L_c / u_c.
    Re, We, Pr, gamma :
        Dimensionless groups derived from the fluid and geometry.
    """

    rho_c: float
    vartheta_c: float
    L_c: float
    u_c: float
    p_c: float
    t_c: float = field(init=False)

    Re: float
    We: float
    Pr: float
    gamma: float

    def __post_init__(self) -> None:
        # derived time scale
        object.__setattr__(self, "t_c", self.L_c / self.u_c)
        # positivity sanity checks
        for name in ("rho_c", "vartheta_c", "L_c", "u_c", "p_c"):
            val = getattr(self, name)
            if val <= 0.0:
                raise ValueError(f"Reference scale {name!r} must be positive, got {val!r}")
        for name in ("Re", "We", "Pr", "gamma"):
            val = getattr(self, name)
            if val <= 0.0:
                raise ValueError(f"Dimensionless group {name!r} must be positive, got {val!r}")

    # ------------------------------------------------------------------
    # conversions
    # ------------------------------------------------------------------
    def _scale(self, kind: Kind) -> float:
        if kind == "density":
            return self.rho_c
        if kind == "velocity":
            return self.u_c
        if kind == "temperature":
            return self.vartheta_c
        if kind == "pressure":
            return self.p_c
        if kind == "time":
            return self.t_c
        if kind == "length":
            return self.L_c
        raise ValueError(f"Unknown kind {kind!r}; expected one of {_ALLOWED_KINDS}")

    def nondimensionalize(self, dim_value, kind: Kind):
        """Convert a dimensional value to dimensionless form.

        ``dim_value`` may be a float, numpy array, or JAX array. The return
        type matches the input type.
        """
        return dim_value / self._scale(kind)

    def dimensionalize(self, nondim_value, kind: Kind):
        """Convert a dimensionless value back to physical units."""
        return nondim_value * self._scale(kind)


def default_water_vapor_scales(L_c: float = 1.0e-6) -> ReferenceScales:
    """Representative water-vapor van der Waals scales (illustrative).

    Values are order-of-magnitude only; used by tests and examples where a
    concrete set of scales is needed. Users wiring up a production problem
    should construct ``ReferenceScales`` explicitly.
    """
    rho_c = 322.0  # kg/m^3  (water critical density, approx)
    vartheta_c = 647.0  # K       (water critical temperature)
    p_c = 22.064e6  # Pa      (water critical pressure)
    u_c = (p_c / rho_c) ** 0.5
    return ReferenceScales(
        rho_c=rho_c,
        vartheta_c=vartheta_c,
        L_c=L_c,
        u_c=u_c,
        p_c=p_c,
        Re=100.0,
        We=1.0,
        Pr=7.0,
        gamma=1.4,
    )
