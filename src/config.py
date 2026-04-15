"""
Problem-specification loader (D5): YAML -> validated ``Problem`` dataclass.

Schema (all sections required unless marked optional):

    mesh:
        n_ctrl:   int (>= degree + 1)
        R_max:    float (> 0)
    discretization:
        degree:   int (>= 1)
        n_gauss:  int (>= degree + 1)      optional; default degree + 1
    time:
        dt:       float (> 0)
        t_end:    float (> 0)
        rho_inf:  float in [0, 1]          Generalized-α spectral radius
    material:
        Re:       float (> 0)
        We:       float (> 0)
        Pr:       float (> 0)
        gamma:    float (> 1)
    initial:
        kind:            str (currently only "bubble")
        R_bubble:        float (> 0)
        interface_width: float (> 0)
        rho_liq:         float (> 0)
        rho_vap:         float (> 0)
        vartheta_0:      float (> 0)
    boundary:
        inner:    str (one of "symmetry", "dirichlet")
        outer:    str (one of "free", "dirichlet")
    output:
        path:      str
        every:     int (>= 1)
        format:    str (one of "xdmf", "csv")  optional; default "xdmf"

A ``Problem`` instance is a frozen nested dataclass. Each section keeps
its own dataclass for clarity and discoverability.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when the YAML config is missing fields or has invalid values."""


# ----------------------------------------------------------------------
# sections
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class MeshSpec:
    n_ctrl: int
    R_max: float


@dataclass(frozen=True)
class DiscretizationSpec:
    degree: int
    n_gauss: int


@dataclass(frozen=True)
class TimeSpec:
    dt: float
    t_end: float
    rho_inf: float


@dataclass(frozen=True)
class MaterialSpec:
    Re: float
    We: float
    Pr: float
    gamma: float


@dataclass(frozen=True)
class InitialSpec:
    kind: str
    R_bubble: float
    interface_width: float
    rho_liq: float
    rho_vap: float
    vartheta_0: float


@dataclass(frozen=True)
class BoundarySpec:
    inner: str
    outer: str


@dataclass(frozen=True)
class OutputSpec:
    path: str
    every: int
    format: str = "xdmf"


@dataclass(frozen=True)
class Problem:
    mesh: MeshSpec
    discretization: DiscretizationSpec
    time: TimeSpec
    material: MaterialSpec
    initial: InitialSpec
    boundary: BoundarySpec
    output: OutputSpec
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        out = asdict(self)
        return out


# ----------------------------------------------------------------------
# validation helpers
# ----------------------------------------------------------------------

def _require_section(raw: dict, name: str) -> dict:
    if name not in raw:
        raise ConfigError(f"Missing required section {name!r}")
    section = raw[name]
    if not isinstance(section, dict):
        raise ConfigError(f"Section {name!r} must be a mapping, got {type(section).__name__}")
    return section


def _require_keys(section: dict, name: str, required: tuple[str, ...]) -> None:
    missing = [k for k in required if k not in section]
    if missing:
        raise ConfigError(
            f"Section {name!r} is missing required field(s): {missing}"
        )


def _coerce(value: Any, typ: type, name: str) -> Any:
    try:
        return typ(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Field {name!r} cannot be coerced to {typ.__name__}: {exc}") from exc


def _positive(value: float, name: str) -> float:
    if value <= 0.0:
        raise ConfigError(f"Field {name!r} must be positive, got {value}")
    return value


def _nonneg(value: float, name: str) -> float:
    if value < 0.0:
        raise ConfigError(f"Field {name!r} must be non-negative, got {value}")
    return value


# ----------------------------------------------------------------------
# loader
# ----------------------------------------------------------------------

def _load_mesh(raw: dict) -> MeshSpec:
    s = _require_section(raw, "mesh")
    _require_keys(s, "mesh", ("n_ctrl", "R_max"))
    n_ctrl = _coerce(s["n_ctrl"], int, "mesh.n_ctrl")
    R_max = _coerce(s["R_max"], float, "mesh.R_max")
    if n_ctrl < 2:
        raise ConfigError("mesh.n_ctrl must be >= 2")
    _positive(R_max, "mesh.R_max")
    return MeshSpec(n_ctrl=n_ctrl, R_max=R_max)


def _load_discretization(raw: dict, mesh: MeshSpec) -> DiscretizationSpec:
    s = _require_section(raw, "discretization")
    _require_keys(s, "discretization", ("degree",))
    degree = _coerce(s["degree"], int, "discretization.degree")
    if degree < 1:
        raise ConfigError("discretization.degree must be >= 1")
    if mesh.n_ctrl < degree + 1:
        raise ConfigError(
            f"mesh.n_ctrl ({mesh.n_ctrl}) must be >= degree + 1 ({degree + 1})"
        )
    n_gauss = _coerce(s.get("n_gauss", degree + 1), int, "discretization.n_gauss")
    if n_gauss < degree + 1:
        raise ConfigError("discretization.n_gauss must be >= degree + 1")
    return DiscretizationSpec(degree=degree, n_gauss=n_gauss)


def _load_time(raw: dict) -> TimeSpec:
    s = _require_section(raw, "time")
    _require_keys(s, "time", ("dt", "t_end", "rho_inf"))
    dt = _positive(_coerce(s["dt"], float, "time.dt"), "time.dt")
    t_end = _positive(_coerce(s["t_end"], float, "time.t_end"), "time.t_end")
    rho_inf = _coerce(s["rho_inf"], float, "time.rho_inf")
    if not (0.0 <= rho_inf <= 1.0):
        raise ConfigError("time.rho_inf must be in [0, 1]")
    return TimeSpec(dt=dt, t_end=t_end, rho_inf=rho_inf)


def _load_material(raw: dict) -> MaterialSpec:
    s = _require_section(raw, "material")
    _require_keys(s, "material", ("Re", "We", "Pr", "gamma"))
    Re = _positive(_coerce(s["Re"], float, "material.Re"), "material.Re")
    We = _positive(_coerce(s["We"], float, "material.We"), "material.We")
    Pr = _positive(_coerce(s["Pr"], float, "material.Pr"), "material.Pr")
    gamma = _coerce(s["gamma"], float, "material.gamma")
    if gamma <= 1.0:
        raise ConfigError("material.gamma must be > 1")
    return MaterialSpec(Re=Re, We=We, Pr=Pr, gamma=gamma)


def _load_initial(raw: dict) -> InitialSpec:
    s = _require_section(raw, "initial")
    _require_keys(
        s, "initial",
        ("kind", "R_bubble", "interface_width", "rho_liq", "rho_vap", "vartheta_0"),
    )
    kind = str(s["kind"])
    if kind != "bubble":
        raise ConfigError(f"initial.kind must be 'bubble' (got {kind!r})")
    R_bubble = _positive(_coerce(s["R_bubble"], float, "initial.R_bubble"), "initial.R_bubble")
    interface_width = _positive(
        _coerce(s["interface_width"], float, "initial.interface_width"),
        "initial.interface_width",
    )
    rho_liq = _positive(_coerce(s["rho_liq"], float, "initial.rho_liq"), "initial.rho_liq")
    rho_vap = _positive(_coerce(s["rho_vap"], float, "initial.rho_vap"), "initial.rho_vap")
    vartheta_0 = _positive(
        _coerce(s["vartheta_0"], float, "initial.vartheta_0"), "initial.vartheta_0"
    )
    return InitialSpec(
        kind=kind, R_bubble=R_bubble, interface_width=interface_width,
        rho_liq=rho_liq, rho_vap=rho_vap, vartheta_0=vartheta_0,
    )


_INNER_BC = ("symmetry", "dirichlet")
_OUTER_BC = ("free", "dirichlet")


def _load_boundary(raw: dict) -> BoundarySpec:
    s = _require_section(raw, "boundary")
    _require_keys(s, "boundary", ("inner", "outer"))
    inner = str(s["inner"])
    outer = str(s["outer"])
    if inner not in _INNER_BC:
        raise ConfigError(f"boundary.inner must be one of {_INNER_BC}, got {inner!r}")
    if outer not in _OUTER_BC:
        raise ConfigError(f"boundary.outer must be one of {_OUTER_BC}, got {outer!r}")
    return BoundarySpec(inner=inner, outer=outer)


_OUTPUT_FORMATS = ("xdmf", "csv")


def _load_output(raw: dict) -> OutputSpec:
    s = _require_section(raw, "output")
    _require_keys(s, "output", ("path", "every"))
    path = str(s["path"])
    every = _coerce(s["every"], int, "output.every")
    if every < 1:
        raise ConfigError("output.every must be >= 1")
    fmt = str(s.get("format", "xdmf"))
    if fmt not in _OUTPUT_FORMATS:
        raise ConfigError(f"output.format must be one of {_OUTPUT_FORMATS}")
    return OutputSpec(path=path, every=every, format=fmt)


def load_problem(path: str | Path) -> Problem:
    """Parse a YAML file and return a validated ``Problem`` instance."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open() as fh:
        raw = yaml.safe_load(fh) or {}
    return from_dict(raw)


def from_dict(raw: dict) -> Problem:
    """Validate and construct a ``Problem`` from a plain Python dict."""
    if not isinstance(raw, dict):
        raise ConfigError("Top-level config must be a mapping")

    mesh = _load_mesh(raw)
    disc = _load_discretization(raw, mesh)
    time = _load_time(raw)
    mat = _load_material(raw)
    ic = _load_initial(raw)
    bc = _load_boundary(raw)
    out = _load_output(raw)
    meta = dict(raw.get("meta", {})) if isinstance(raw.get("meta"), dict) else {}
    return Problem(
        mesh=mesh, discretization=disc, time=time, material=mat,
        initial=ic, boundary=bc, output=out, meta=meta,
    )


def dump_problem(problem: Problem, path: str | Path) -> None:
    """Write a ``Problem`` back out to YAML (round-trip support)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as fh:
        yaml.safe_dump(problem.to_dict(), fh, sort_keys=False)


__all__ = [
    "BoundarySpec",
    "ConfigError",
    "DiscretizationSpec",
    "InitialSpec",
    "MaterialSpec",
    "MeshSpec",
    "OutputSpec",
    "Problem",
    "TimeSpec",
    "dump_problem",
    "from_dict",
    "load_problem",
]


# Suppress unused-imports warnings from static analysis
_ = fields
