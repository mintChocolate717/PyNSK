"""Tests for src/config.py (D5)."""
from pathlib import Path

import pytest
import yaml

from src.config import (
    ConfigError,
    Problem,
    dump_problem,
    from_dict,
    load_problem,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "bubble_collapse.yaml"


def _minimal_dict():
    return {
        "mesh": {"n_ctrl": 16, "R_max": 1.0},
        "discretization": {"degree": 2},
        "time": {"dt": 1.0e-3, "t_end": 1.0, "rho_inf": 0.5},
        "material": {"Re": 100.0, "We": 1.0, "Pr": 7.0, "gamma": 1.4},
        "initial": {
            "kind": "bubble", "R_bubble": 0.3, "interface_width": 0.05,
            "rho_liq": 0.6, "rho_vap": 0.05, "vartheta_0": 0.85,
        },
        "boundary": {"inner": "symmetry", "outer": "free"},
        "output": {"path": "out/run", "every": 10},
    }


def test_example_file_loads():
    problem = load_problem(EXAMPLE)
    assert isinstance(problem, Problem)
    assert problem.mesh.n_ctrl > 0
    assert problem.initial.kind == "bubble"
    assert problem.output.format == "xdmf"


def test_round_trip(tmp_path):
    problem = from_dict(_minimal_dict())
    path = tmp_path / "roundtrip.yaml"
    dump_problem(problem, path)
    problem2 = load_problem(path)
    assert problem2 == problem


def test_missing_required_section_raises():
    raw = _minimal_dict()
    del raw["time"]
    with pytest.raises(ConfigError, match="time"):
        from_dict(raw)


def test_missing_required_field_raises():
    raw = _minimal_dict()
    del raw["material"]["We"]
    with pytest.raises(ConfigError, match="We"):
        from_dict(raw)


def test_invalid_value_raises():
    raw = _minimal_dict()
    raw["time"]["rho_inf"] = 2.0  # out of [0, 1]
    with pytest.raises(ConfigError, match="rho_inf"):
        from_dict(raw)


def test_degree_vs_nctrl_consistency():
    raw = _minimal_dict()
    raw["mesh"]["n_ctrl"] = 2
    raw["discretization"]["degree"] = 5
    with pytest.raises(ConfigError, match="degree"):
        from_dict(raw)


def test_default_n_gauss():
    raw = _minimal_dict()
    # drop n_gauss if present; loader should default to degree + 1
    raw["discretization"].pop("n_gauss", None)
    problem = from_dict(raw)
    assert problem.discretization.n_gauss == problem.discretization.degree + 1


def test_default_output_format():
    raw = _minimal_dict()
    problem = from_dict(raw)
    assert problem.output.format == "xdmf"


def test_load_nonexistent_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_problem(tmp_path / "nope.yaml")


def test_invalid_boundary_raises():
    raw = _minimal_dict()
    raw["boundary"]["inner"] = "wibble"
    with pytest.raises(ConfigError, match="boundary.inner"):
        from_dict(raw)


def test_yaml_is_valid_syntax():
    """Defensive: the example file must be parseable by safe_load."""
    with EXAMPLE.open() as fh:
        raw = yaml.safe_load(fh)
    assert isinstance(raw, dict)
