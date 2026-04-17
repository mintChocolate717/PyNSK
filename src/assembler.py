"""
Global residual assembler for 1D spherically-symmetric NSK IGA.

Couples the four scalar fields (ρ, u_r, ϑ, V) — each discretised with the
same open-uniform B-spline basis of degree p on [0, R_max] — into a single
global residual vector of length 4*n_ctrl, ordered block-wise:

    R_global = [ R_ρ   |   R_u   |   R_ϑ   |   R_V ]  ∈ ℝ^{4 n_ctrl}.

The element loop is vectorised with ``jax.vmap`` so the whole assembly is
a single traced operation, compatible with ``jax.jacfwd`` / ``jax.jit``.

Notation
--------
``vartheta`` is the thermodynamic temperature.  The θθ components of the
stress tensors keep the legacy names ``tau_tt`` / ``sigma_tt`` — in those
symbols *θ* refers to the polar angle, not temperature.

Sparsity (future work)
----------------------
The current implementation accumulates into dense JAX arrays so that the
whole residual remains a pure function compatible with ``jax.jacfwd``.
Each B-spline residual has bandwidth ≤ p+1, so the natural successor is
BCOO / CSR storage.  Search for "TODO(sparse)" below for the single
accumulation site that needs swapping.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from src.bsplines import basis_deriv_matrix, basis_matrix
from src.quadrature import quadrature_points
from src.residuals import (
    element_residual_auxiliary,
    element_residual_energy,
    element_residual_mass,
    element_residual_momentum,
)

# ── Element connectivity (IEN) ──────────────────────────────────────────────


def element_connectivity(n_ctrl: int, degree: int) -> jnp.ndarray:
    """Local-to-global DOF map (IEN) for an open-uniform B-spline mesh.

    The open-uniform knot vector ``make_knot_vector(n_ctrl, degree)`` has
    ``n_elements = n_ctrl - degree`` non-zero knot spans.  Element *e*
    supports the control points ``e, e+1, …, e+degree`` — consecutive
    elements overlap on *degree* control points (IGA C^{p-1} continuity
    across element boundaries for simple interior knots).

    Args:
        n_ctrl: number of control points in the basis.
        degree: polynomial degree p.

    Returns:
        IEN: int array of shape ``(n_elements, degree + 1)`` where
             ``IEN[e, a]`` is the global DOF index of local shape function
             *a* on element *e*.
    """
    n_elements = n_ctrl - degree
    if n_elements < 1:
        raise ValueError(f"n_ctrl ({n_ctrl}) must exceed degree ({degree}) by >=1")
    rows = np.arange(n_elements)[:, None]
    cols = np.arange(degree + 1)[None, :]
    return jnp.asarray(rows + cols, dtype=jnp.int32)


# ── Basis cache ─────────────────────────────────────────────────────────────


def build_basis_cache(
    knots: jnp.ndarray,
    degree: int,
    n_gauss: int,
    R_max: float,
) -> dict:
    """Precompute element-local basis matrices at all quadrature points.

    All four primary fields share the same basis, so only one set of
    matrices is stored.  Data is reshaped to ``(n_elements, n_gauss, …)``
    and sliced to the *degree+1* non-zero basis columns per element using
    the IEN array, so that a subsequent ``vmap`` over elements sees
    identically-shaped small matrices.

    Args:
        knots:   B-spline knot vector, shape ``(n_ctrl + degree + 1,)``.
        degree:  polynomial degree p.
        n_gauss: Gauss-Legendre points per element.
        R_max:   outer radius of the physical domain.

    Returns:
        dict with keys
            ``'xi'``  : (n_elements, n_gauss) parametric coords,
            ``'r'``   : (n_elements, n_gauss) physical coords,
            ``'w'``   : (n_elements, n_gauss) quadrature weights
                        (element Jacobian already folded in),
            ``'J'``   : (n_elements,)         element Jacobian dr/dξ_ref,
            ``'N'``   : (n_elements, n_gauss, degree+1) basis values,
            ``'dN'``  : (n_elements, n_gauss, degree+1) first derivatives
                        with respect to *r* (already divided by dr/dxi_param
                        — see note below),
            ``'d2N'`` : (n_elements, n_gauss, degree+1) second derivatives
                        with respect to *r*,
            ``'IEN'`` : (n_elements, degree+1) int connectivity.

    Derivative convention
    ---------------------
    ``basis_deriv_matrix`` returns derivatives w.r.t. the parametric
    variable ξ ∈ [0, 1].  The physical radius is ``r = R_max * ξ``, so
    ``dN/dr = (1/R_max) dN/dξ`` and similarly for the second derivative.
    Those chain-rule factors are applied here, so downstream element
    residual routines receive basis matrices already differentiated with
    respect to *r*.
    """
    n_ctrl = int(knots.shape[0]) - degree - 1
    n_elements = n_ctrl - degree

    xi_pts, r_pts, w_pts = quadrature_points(knots, degree, n_gauss, R_max)

    N_flat = basis_matrix(xi_pts, knots, degree)  # (n_pts, n_ctrl)
    dN_xi_flat = basis_deriv_matrix(xi_pts, knots, degree, 1)  # d/dξ
    d2N_xi_flat = basis_deriv_matrix(xi_pts, knots, degree, 2)  # d²/dξ²

    # chain rule: r = R_max * ξ → d/dr = (1/R_max) d/dξ
    dN_flat = dN_xi_flat / R_max
    d2N_flat = d2N_xi_flat / (R_max * R_max)

    IEN = element_connectivity(n_ctrl, degree)  # (n_elem, p+1)

    xi = xi_pts.reshape(n_elements, n_gauss)
    r = r_pts.reshape(n_elements, n_gauss)
    w = w_pts.reshape(n_elements, n_gauss)

    # Unique knot spans give the element Jacobian dr/d(ξ_ref).  ξ_ref is
    # the Gauss reference coord on [-1,1].  J_e = R_max * (ξ_b − ξ_a) / 2.
    t = np.asarray(knots)
    unique_knots = np.unique(t)
    J = jnp.asarray(
        R_max * (unique_knots[1:] - unique_knots[:-1]) / 2.0,
        dtype=jnp.float64,
    )

    # Slice basis columns per element.  N_flat has shape (n_pts, n_ctrl);
    # we gather the (p+1) non-zero columns for each element via IEN.
    N_flat_re = N_flat.reshape(n_elements, n_gauss, n_ctrl)
    dN_flat_re = dN_flat.reshape(n_elements, n_gauss, n_ctrl)
    d2N_flat_re = d2N_flat.reshape(n_elements, n_gauss, n_ctrl)

    elem_idx = jnp.arange(n_elements)[:, None, None]
    qp_idx = jnp.arange(n_gauss)[None, :, None]
    col_idx = IEN[:, None, :]  # (n_elem,1,p+1)

    N = N_flat_re[elem_idx, qp_idx, col_idx]  # (n_elem, n_gauss, p+1)
    dN = dN_flat_re[elem_idx, qp_idx, col_idx]
    d2N = d2N_flat_re[elem_idx, qp_idx, col_idx]

    return {
        "xi": xi,
        "r": r,
        "w": w,
        "J": J,
        "N": N,
        "dN": dN,
        "d2N": d2N,
        "IEN": IEN,
        "n_ctrl": n_ctrl,
        "n_elements": int(n_elements),
        "n_gauss": int(n_gauss),
        "degree": int(degree),
    }


# ── Element-level helpers (vmapped) ─────────────────────────────────────────


def _element_residuals(
    N_e,
    dN_e,
    d2N_e,
    r_e,
    w_e,
    J_e,
    IEN_e,
    ctrl_rho,
    ctrl_u,
    ctrl_vartheta,
    ctrl_V,
    ctrl_rho_dot,
    ctrl_u_dot,
    ctrl_vartheta_dot,
    Re,
    We,
    Pr,
    gamma,
    b_r,
    r_s,
):
    """Compute the four element residual vectors for a single element.

    All basis matrices are the same since every field shares the mesh, so
    ``N_e, dN_e, d2N_e`` are passed once and reused for every field.
    """
    # Gather element-local control DOFs through the IEN map.
    d_rho = ctrl_rho[IEN_e]
    d_u = ctrl_u[IEN_e]
    d_vth = ctrl_vartheta[IEN_e]
    d_V = ctrl_V[IEN_e]

    d_rho_dot = ctrl_rho_dot[IEN_e]
    d_u_dot = ctrl_u_dot[IEN_e]
    d_vth_dot = ctrl_vartheta_dot[IEN_e]

    R_rho = element_residual_mass(
        N_e,
        dN_e,
        N_e,
        d_rho,
        d_rho_dot,
        d_u,
        r_e,
        w_e,
        J_e,
    )
    R_u = element_residual_momentum(
        N_e,
        dN_e,
        N_e,
        dN_e,
        d2N_e,
        N_e,
        dN_e,
        N_e,
        d_rho,
        d_rho_dot,
        d_u,
        d_u_dot,
        d_vth,
        d_V,
        r_e,
        w_e,
        J_e,
        Re,
        We,
        gamma,
        b_r,
    )
    R_vth = element_residual_energy(
        N_e,
        dN_e,
        N_e,
        dN_e,
        d2N_e,
        N_e,
        dN_e,
        N_e,
        d_rho,
        d_rho_dot,
        d_u,
        d_u_dot,
        d_vth,
        d_vth_dot,
        d_V,
        r_e,
        w_e,
        J_e,
        Re,
        We,
        gamma,
        Pr,
        b_r,
        r_s,
    )
    R_V = element_residual_auxiliary(
        N_e,
        dN_e,
        N_e,
        dN_e,
        N_e,
        N_e,
        d_rho,
        d_u,
        d_vth,
        d_V,
        r_e,
        w_e,
        J_e,
        We,
        gamma,
    )
    return R_rho, R_u, R_vth, R_V


# NOTE: ``w_e`` here is the quadrature weight *already* multiplied by the
# element Jacobian inside ``quadrature_points``.  The element-residual
# functions in ``src/residuals.py`` still take ``J_e`` separately and do
# ``J_e * w_q`` internally — to avoid double-counting, we divide the
# cached weight by J_e before handing it to those functions (or,
# equivalently, supply J_e=1.0 and keep the pre-scaled weight).  We go
# with the simpler route: supply the raw reference-element weight.
def _element_residuals_wrapper(cache_slice, ctrl, ctrl_dot, params):
    (N_e, dN_e, d2N_e, r_e, w_folded_e, J_e, IEN_e) = cache_slice
    # Un-fold Jacobian: w_folded = w_ref * J_e  →  w_ref = w_folded / J_e
    w_ref_e = w_folded_e / J_e
    return _element_residuals(
        N_e,
        dN_e,
        d2N_e,
        r_e,
        w_ref_e,
        J_e,
        IEN_e,
        ctrl["rho"],
        ctrl["u"],
        ctrl["vartheta"],
        ctrl["V"],
        ctrl_dot["rho"],
        ctrl_dot["u"],
        ctrl_dot["vartheta"],
        params["Re"],
        params["We"],
        params["Pr"],
        params["gamma"],
        params["b_r"],
        params["r_s"],
    )


# ── Global assembly ─────────────────────────────────────────────────────────


def assemble_residual(
    ctrl: dict,
    ctrl_dot: dict,
    cache: dict,
    params: dict,
) -> jnp.ndarray:
    """Assemble the global 4-block residual vector.

    Args:
        ctrl:     dict with ``'rho'``, ``'u'``, ``'vartheta'``, ``'V'`` each
                  of shape ``(n_ctrl,)`` — current control-point values.
        ctrl_dot: dict with ``'rho'``, ``'u'``, ``'vartheta'`` each
                  ``(n_ctrl,)`` — time derivatives.  ``'V'`` is an
                  auxiliary (algebraic) variable and does not require a
                  time derivative; if supplied it is ignored.
        cache:    basis cache from :func:`build_basis_cache`.
        params:   dict with ``Re``, ``We``, ``Pr``, ``gamma``, ``b_r``, ``r_s``.

    Returns:
        Global residual vector of shape ``(4 * n_ctrl,)`` ordered as
        ``[R_ρ | R_u | R_ϑ | R_V]``.
    """
    n_ctrl = cache["n_ctrl"]
    IEN = cache["IEN"]
    N = cache["N"]
    dN = cache["dN"]
    d2N = cache["d2N"]
    r = cache["r"]
    w = cache["w"]
    J = cache["J"]

    cache_slice = (N, dN, d2N, r, w, J, IEN)

    # Vectorised element loop.  First axis of each array (except ctrl
    # dicts and params) is the element index.
    vmap_fn = jax.vmap(
        _element_residuals_wrapper,
        in_axes=((0, 0, 0, 0, 0, 0, 0), None, None, None),
    )

    R_rho_e, R_u_e, R_vth_e, R_V_e = vmap_fn(
        cache_slice, ctrl, ctrl_dot, params
    )  # each: (n_elements, degree+1)

    # Scatter-add element contributions into global vectors.
    # TODO(sparse): when moving to ``jax.experimental.sparse`` or
    # ``scipy.sparse``, replace the four dense ``segment_sum`` calls
    # below with accumulation into BCOO/CSR buffers (the IEN indices
    # already define the sparsity pattern).
    flat_idx = IEN.reshape(-1)  # (n_elements*(p+1),)

    def _scatter(R_e_block):
        return jax.ops.segment_sum(R_e_block.reshape(-1), flat_idx, num_segments=n_ctrl)

    R_rho = _scatter(R_rho_e)
    R_u = _scatter(R_u_e)
    R_vth = _scatter(R_vth_e)
    R_V = _scatter(R_V_e)

    return jnp.concatenate([R_rho, R_u, R_vth, R_V])


# ── Boundary-condition helpers ──────────────────────────────────────────────


def apply_dirichlet(
    R: jnp.ndarray,
    K: jnp.ndarray | None,
    dof_indices: jnp.ndarray,
    values: jnp.ndarray,
    current: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Impose essential (Dirichlet) boundary conditions.

    Standard row/column zero-out on the stiffness matrix with unit
    diagonal, and a matching adjustment on the residual so that the
    modified system solves exactly the prescribed values.

    The residual is overwritten with ``current[i] - values[i]`` at each
    constrained DOF so that, after a Newton update ``δ = -K^{-1} R``, the
    new value satisfies ``current_new = values`` exactly.

    Args:
        R:           global residual, shape ``(n_dof,)``.
        K:           global tangent, shape ``(n_dof, n_dof)``; pass
                     ``None`` if only the residual is being modified
                     (e.g. when evaluating a residual-only transient).
        dof_indices: 1-D int array of DOF indices to constrain.
        values:      prescribed values, same shape as ``dof_indices``.
        current:     current DOF values (so we can set the residual to
                     the signed difference).

    Returns:
        ``(R_modified, K_modified)``.  ``K_modified`` is ``None`` if
        ``K`` was ``None`` on input.
    """
    dof_indices = jnp.asarray(dof_indices, dtype=jnp.int32)
    values = jnp.asarray(values, dtype=R.dtype)
    current = jnp.asarray(current, dtype=R.dtype)

    # Residual: R_i = current_i - values_i at constrained DOFs.
    R_mod = R.at[dof_indices].set(current[dof_indices] - values)

    if K is None:
        return R_mod, None

    n = K.shape[0]
    # Zero the rows and columns, then place 1 on the diagonal.
    mask = jnp.zeros(n, dtype=jnp.bool_).at[dof_indices].set(True)
    K_mod = jnp.where(mask[:, None], 0.0, K)
    K_mod = jnp.where(mask[None, :], 0.0, K_mod)
    K_mod = K_mod.at[dof_indices, dof_indices].set(1.0)
    return R_mod, K_mod


def _block_offsets(n_ctrl: int) -> dict:
    """Block DOF offsets for [ρ | u | ϑ | V]."""
    return {
        "rho": 0,
        "u": n_ctrl,
        "vartheta": 2 * n_ctrl,
        "V": 3 * n_ctrl,
    }


def symmetry_bc_at_origin(n_ctrl: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Essential BC at r=0: impose ``u_r(0) = 0`` on the first u-control point.

    Only the radial velocity is constrained at the origin.

    * ``∂ρ/∂r = 0`` and ``∂ϑ/∂r = 0`` are **natural** conditions of the
      weak form (they correspond to vanishing diffusive fluxes at r=0
      and are satisfied automatically when the surface integrals
      vanish by the ``r²`` factor in the spherical volume element).
      No explicit treatment is required for those two fields.
    * ``V`` is an auxiliary algebraic variable and needs no boundary
      condition at the origin.

    Returns:
        ``(dof_indices, values)`` suitable to hand to
        :func:`apply_dirichlet` — a single entry fixing
        ``u[0] = 0`` (global DOF index ``n_ctrl``).
    """
    offsets = _block_offsets(n_ctrl)
    dof_indices = jnp.asarray([offsets["u"] + 0], dtype=jnp.int32)
    values = jnp.asarray([0.0], dtype=jnp.float64)
    return dof_indices, values


def dirichlet_far_field(
    n_ctrl: int,
    rho_inf: float,
    vartheta_inf: float,
    u_inf: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Essential BCs at r=R_max: ``ρ=ρ_∞``, ``ϑ=ϑ_∞``, ``u=u_∞`` (default 0).

    The far-field auxiliary variable ``V`` is left free (it is an
    algebraic field whose value is determined by the state).

    Returns:
        ``(dof_indices, values)`` for :func:`apply_dirichlet`.
    """
    offsets = _block_offsets(n_ctrl)
    last = n_ctrl - 1
    dof_indices = jnp.asarray(
        [offsets["rho"] + last, offsets["u"] + last, offsets["vartheta"] + last],
        dtype=jnp.int32,
    )
    values = jnp.asarray([rho_inf, u_inf, vartheta_inf], dtype=jnp.float64)
    return dof_indices, values
