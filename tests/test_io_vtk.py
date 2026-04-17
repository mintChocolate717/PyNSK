"""Tests for src/io_vtk.py (D4)."""

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from src.io_vtk import available_backends, read_csv_snapshot, write_xdmf_timestep


def _sample_fields(n: int = 16):
    r = np.linspace(0.0, 1.0, n)
    rho = 0.3 + 0.1 * r
    u = 0.0 * r
    vartheta = 0.9 * np.ones_like(r)
    V = 0.2 * np.ones_like(r)
    return r, {"rho": rho, "u": u, "vartheta": vartheta, "V": V}


def test_available_backends_contains_fallback():
    backs = available_backends()
    assert "csv-pvd" in backs


def test_write_creates_files(tmp_path):
    r, fields = _sample_fields()
    out = write_xdmf_timestep(tmp_path / "snap", t=0.123, r_grid=r, fields_dict=fields)
    for p in out.values():
        assert Path(p).exists()


def test_write_multiple_timesteps_pvd_collection(tmp_path):
    """When the CSV fallback is used, the .pvd should index all timesteps."""
    r, fields = _sample_fields()
    out1 = write_xdmf_timestep(tmp_path / "snap", t=0.1, r_grid=r, fields_dict=fields)
    out2 = write_xdmf_timestep(tmp_path / "snap", t=0.2, r_grid=r, fields_dict=fields)

    # If the primary (h5) backend is active, the function does not emit a pvd;
    # skip in that case.
    if "pvd" not in out1 or "pvd" not in out2:
        pytest.skip("Primary XDMF/H5 backend active; PVD collection not generated")

    pvd_path = Path(out2["pvd"])
    tree = ET.parse(pvd_path)
    datasets = tree.getroot().find("Collection").findall("DataSet")
    assert len(datasets) >= 2


def test_csv_roundtrip(tmp_path):
    r, fields = _sample_fields()
    out = write_xdmf_timestep(tmp_path / "snap", t=0.5, r_grid=r, fields_dict=fields)
    if "csv" not in out:
        pytest.skip("Primary XDMF/H5 backend active; CSV roundtrip N/A")

    r_back, fields_back = read_csv_snapshot(out["csv"])
    np.testing.assert_allclose(r_back, r)
    for name in fields:
        np.testing.assert_allclose(fields_back[name], fields[name])


def test_field_shape_mismatch_raises(tmp_path):
    r = np.linspace(0.0, 1.0, 8)
    bad = {"rho": np.zeros(7)}  # wrong length
    with pytest.raises(ValueError):
        write_xdmf_timestep(tmp_path / "snap", t=0.0, r_grid=r, fields_dict=bad)


def test_creates_parent_directory(tmp_path):
    r, fields = _sample_fields()
    nested = tmp_path / "deep" / "nest" / "snap"
    write_xdmf_timestep(nested, t=0.0, r_grid=r, fields_dict=fields)
    assert nested.parent.exists()


def test_xdmf_parses_when_h5_available(tmp_path):
    """If h5py is present, the generated .xdmf must be well-formed XML."""
    r, fields = _sample_fields()
    out = write_xdmf_timestep(tmp_path / "snap", t=0.0, r_grid=r, fields_dict=fields)
    if "xdmf" not in out:
        pytest.skip("h5py not installed; XDMF path not exercised")
    tree = ET.parse(out["xdmf"])
    root = tree.getroot()
    assert root.tag.endswith("Xdmf")
