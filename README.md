# PyNSK

An IGA (Isogeometric Analysis) solver for the Navier-Stokes-Korteweg (NSK) equations under spherical symmetry, targeting bubble nucleation and cavitation.

Extends the thermomechanical framework of Liu, Landis, Gomez & Hughes (CMAME 2015) to the spherically symmetric bubble cavitation problem.

## Features

- B-spline spatial discretization (Galerkin/IGA), degree p ≥ 2 for C¹ continuity
- General-α temporal integration (second-order, controllable high-frequency dissipation)
- Newton-Raphson solver with automatic tangent stiffness via JAX autodiff
- van der Waals constitutive model with Korteweg capillary stress

## Installation

```bash
conda env create -f environment.yml
conda activate bubble-cavitation
```

## Running Tests

```bash
pytest tests/
```

## Reference

Liu, J., Landis, C.M., Gomez, H., Hughes, T.J.R. — *Liquid–vapor phase transition: Thermomechanical theory, entropy stable numerical formulation, and boiling simulations*. Comput. Methods Appl. Mech. Engrg. 297 (2015) 476–553.
