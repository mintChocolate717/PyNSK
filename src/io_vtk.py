"""
ParaView-ready output for the 1D radial solver (FEA best practice).

Primary path
------------
If ``h5py`` is available, ``write_xdmf_timestep`` writes an XDMF + HDF5
pair containing a 1D line mesh (the radial grid) with the supplied point
data (ρ, u, ϑ, V, or any user-supplied fields). ParaView opens the .xdmf
file directly.

Fallback path
-------------
If ``h5py`` is not installed, the function writes plain CSV files
(``<stem>_t<step>.csv``) with r and each field as columns, plus a
``<stem>.pvd`` wrapper referencing each CSV. ParaView's native CSV + PVD
reader can load the time series, although it requires the user to select
the "Line Chart View". This fallback is documented both here and in the
generated file headers.
"""
from __future__ import annotations

import csv
import importlib.util
import os
from collections.abc import Sequence
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

_H5PY_AVAILABLE = importlib.util.find_spec("h5py") is not None
_MESHIO_AVAILABLE = importlib.util.find_spec("meshio") is not None


def _as_float_1d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).ravel()
    return arr


def write_xdmf_timestep(
    path: str | os.PathLike,
    t: float,
    r_grid,
    fields_dict: dict,
) -> dict[str, Path]:
    """Write a single snapshot for ParaView.

    Parameters
    ----------
    path :
        Output basename. Time is encoded in the suffix ``_tNNNNNN``. The
        function will create ``path.parent`` if it does not exist. Any
        existing suffix on ``path`` is stripped and replaced.
    t :
        Physical time value attached to the snapshot.
    r_grid :
        1D array of radial coordinates, shape (N,).
    fields_dict :
        Mapping ``name -> array of shape (N,)`` with point data.

    Returns
    -------
    Mapping of output-kind → file path, e.g.::

        {"xdmf": .../snap_t000001.xdmf, "h5": .../snap_t000001.h5}

    or, in CSV fallback mode::

        {"csv": .../snap_t000001.csv, "pvd": .../snap.pvd}
    """
    r_grid = _as_float_1d(r_grid)
    n = r_grid.shape[0]
    fields = {name: _as_float_1d(arr) for name, arr in fields_dict.items()}
    for name, arr in fields.items():
        if arr.shape[0] != n:
            raise ValueError(
                f"Field {name!r} has shape {arr.shape}, expected ({n},)"
            )

    base = Path(os.fspath(path))
    base_dir = base.parent
    base_dir.mkdir(parents=True, exist_ok=True)
    stem = base.stem

    # encode time as an integer step counter derived from t
    # caller is free to format the filename; we accept this simple rule.
    step = round(t * 1.0e6) if abs(t) < 1e3 else round(t)
    tag = f"{stem}_t{step:09d}"

    if _H5PY_AVAILABLE:
        return _write_xdmf_h5(base_dir, stem, tag, t, r_grid, fields)
    return _write_csv_pvd(base_dir, stem, tag, t, r_grid, fields)


# ----------------------------------------------------------------------
# XDMF + HDF5 path
# ----------------------------------------------------------------------

def _write_xdmf_h5(
    base_dir: Path, stem: str, tag: str, t: float,
    r_grid: np.ndarray, fields: dict[str, np.ndarray],
) -> dict[str, Path]:
    import h5py  # local import, optional dependency

    h5_path = base_dir / f"{tag}.h5"
    xdmf_path = base_dir / f"{tag}.xdmf"

    with h5py.File(h5_path, "w") as fh:
        # 1D points embedded as (N, 3) with y=z=0 for XDMF compatibility
        coords = np.zeros((r_grid.size, 3), dtype=np.float64)
        coords[:, 0] = r_grid
        fh.create_dataset("geometry/XYZ", data=coords)
        # topology: polyline with N-1 edges
        edges = np.stack([np.arange(r_grid.size - 1),
                          np.arange(1, r_grid.size)], axis=1)
        fh.create_dataset("topology/edges", data=edges.astype(np.int64))
        for name, arr in fields.items():
            fh.create_dataset(f"fields/{name}", data=arr)

    xdmf = ET.Element("Xdmf", Version="3.0")
    domain = ET.SubElement(xdmf, "Domain")
    grid = ET.SubElement(
        domain, "Grid", Name=stem, GridType="Uniform",
    )
    ET.SubElement(grid, "Time", Value=f"{t:.17g}")

    topology = ET.SubElement(
        grid, "Topology",
        TopologyType="Polyline", NumberOfElements=str(r_grid.size - 1),
        NodesPerElement="2",
    )
    ET.SubElement(
        topology, "DataItem",
        Dimensions=f"{r_grid.size - 1} 2", NumberType="Int",
        Precision="8", Format="HDF",
    ).text = f"{h5_path.name}:/topology/edges"

    geom = ET.SubElement(grid, "Geometry", GeometryType="XYZ")
    ET.SubElement(
        geom, "DataItem",
        Dimensions=f"{r_grid.size} 3", NumberType="Float",
        Precision="8", Format="HDF",
    ).text = f"{h5_path.name}:/geometry/XYZ"

    for name in fields:
        attr = ET.SubElement(
            grid, "Attribute",
            Name=name, AttributeType="Scalar", Center="Node",
        )
        ET.SubElement(
            attr, "DataItem",
            Dimensions=f"{r_grid.size}", NumberType="Float",
            Precision="8", Format="HDF",
        ).text = f"{h5_path.name}:/fields/{name}"

    tree = ET.ElementTree(xdmf)
    ET.indent(tree, space="  ")
    tree.write(xdmf_path, encoding="utf-8", xml_declaration=True)
    return {"xdmf": xdmf_path, "h5": h5_path}


# ----------------------------------------------------------------------
# CSV + PVD fallback
# ----------------------------------------------------------------------

def _write_csv_pvd(
    base_dir: Path, stem: str, tag: str, t: float,
    r_grid: np.ndarray, fields: dict[str, np.ndarray],
) -> dict[str, Path]:
    csv_path = base_dir / f"{tag}.csv"
    pvd_path = base_dir / f"{stem}.pvd"

    field_names = list(fields)
    with csv_path.open("w", newline="") as fh:
        fh.write(
            f"# t={t:.17g} (CSV fallback: h5py not available)\n"
        )
        writer = csv.writer(fh)
        writer.writerow(["r", *field_names])
        for i in range(r_grid.size):
            writer.writerow(
                [float(r_grid[i])] + [float(fields[n][i]) for n in field_names]
            )

    # Update (or create) the .pvd wrapper by re-scanning existing CSVs in the dir
    _rewrite_pvd(pvd_path, base_dir, stem)
    return {"csv": csv_path, "pvd": pvd_path}


def _rewrite_pvd(pvd_path: Path, base_dir: Path, stem: str) -> None:
    """Rebuild the PVD index from all ``<stem>_tNNNNN.csv`` siblings."""
    files: list[tuple[int, str]] = []
    prefix = f"{stem}_t"
    for entry in sorted(base_dir.iterdir()):
        if entry.is_file() and entry.suffix == ".csv" and entry.stem.startswith(prefix):
            tag = entry.stem[len(prefix):]
            try:
                step = int(tag)
            except ValueError:
                continue
            files.append((step, entry.name))

    vtk = ET.Element("VTKFile", type="Collection", version="0.1")
    coll = ET.SubElement(vtk, "Collection")
    for step, fname in files:
        ET.SubElement(
            coll, "DataSet",
            timestep=str(step), group="", part="0", file=fname,
        )

    tree = ET.ElementTree(vtk)
    ET.indent(tree, space="  ")
    tree.write(pvd_path, encoding="utf-8", xml_declaration=True)


def read_csv_snapshot(path: str | os.PathLike) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Round-trip helper: parse a CSV snapshot written by the fallback path."""
    p = Path(os.fspath(path))
    with p.open() as fh:
        # skip any ``#`` header lines
        line = fh.readline()
        while line.startswith("#"):
            line = fh.readline()
        headers = [h.strip() for h in line.rstrip("\n").split(",")]
        rows = [[float(x) for x in row.rstrip("\n").split(",")]
                for row in fh if row.strip()]
    data = np.array(rows, dtype=np.float64)
    r = data[:, 0]
    fields = {h: data[:, i + 1] for i, h in enumerate(headers[1:])}
    return r, fields


def available_backends() -> Sequence[str]:
    """Report which backends the runtime found on import.

    Useful for logging at simulation start-up.
    """
    out = []
    if _H5PY_AVAILABLE:
        out.append("xdmf-h5")
    if _MESHIO_AVAILABLE:
        out.append("meshio")
    out.append("csv-pvd")
    return tuple(out)
