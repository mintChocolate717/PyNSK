# PyNSK

An IGA (Isogeometric Analysis) solver for the Navier-Stokes-Korteweg (NSK) equations under spherical symmetry, targeting bubble nucleation and cavitation.

Extends the thermomechanical framework of Liu, Landis, Gomez & Hughes (CMAME 2015) to the spherically symmetric bubble cavitation problem.

## Features

- B-spline spatial discretization (Galerkin/IGA), degree p ≥ 2 for C¹ continuity
- General-α temporal integration (second-order, controllable high-frequency dissipation)
- Newton-Raphson solver with automatic tangent stiffness via JAX autodiff
- van der Waals constitutive model with Korteweg capillary stress
- JAX float64 throughout — enabled globally in `src/__init__.py`

## Installation

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate bubble-cavitation
pip install -e ".[dev]"
```

### Pip-only

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Python >= 3.13 is required.

## Running tests

Fast suite (skips the slow regression cases):

```bash
pytest tests/ -v --ignore=tests/regression
```

Full suite including regression:

```bash
pytest tests/ -v
```

Coverage report:

```bash
pytest tests/ --cov=src --cov-branch --cov-report=term-missing \
    --ignore=tests/regression
```

## Running a sample case

A placeholder driver lives at `scripts/run_sample_case.py`. It prints
the reproducibility snapshot today and will be wired to the full
bubble-collapse run once Phase D lands:

```bash
python scripts/run_sample_case.py
```

## Repository layout

```
src/                 solver modules (pure JAX, functional)
tests/               fast unit tests
tests/regression/    long end-to-end cases (manual CI job)
scripts/             helper drivers (not imported by src)
docs/                Sphinx skeleton + architecture overview
notebooks/           exploratory Jupyter notebooks
```

See `docs/architecture.md` for the layered module diagram.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for naming conventions, type-hint
style, testing expectations, and branch/commit rules.

## Citation

If you use PyNSK in academic work, please cite both this repository and
the underlying CMAME 2015 formulation:

```bibtex
@software{pynsk,
  title  = {PyNSK: Isogeometric Analysis solver for the 1D spherically-symmetric Navier--Stokes--Korteweg equations},
  author = {{PyNSK developers}},
  year   = {2026}
}

@article{LiuLandisGomezHughes2015,
  title   = {Liquid--vapor phase transition: Thermomechanical theory, entropy stable numerical formulation, and boiling simulations},
  author  = {Liu, Ju and Landis, Chad M. and Gomez, Hector and Hughes, Thomas J. R.},
  journal = {Computer Methods in Applied Mechanics and Engineering},
  volume  = {297},
  pages   = {476--553},
  year    = {2015},
  doi     = {10.1016/j.cma.2015.09.007}
}
```

## Reference

Liu, J., Landis, C.M., Gomez, H., Hughes, T.J.R. — *Liquid–vapor phase transition: Thermomechanical theory, entropy stable numerical formulation, and boiling simulations*. Comput. Methods Appl. Mech. Engrg. 297 (2015) 476–553.
