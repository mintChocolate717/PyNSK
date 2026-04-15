"""Time-integration and nonlinear-solver infrastructure for PyNSK.

Implements a Jansen–Whiting–Hulbert generalized-α integrator for first-order
systems R(ḋ, d, t) = 0, a Newton–Raphson driver that builds the tangent via
``jax.jacfwd``, a :class:`TimeStepper` that glues the two together, plus
checkpoint/restart helpers and a small eigen-diagnostic utility.

The module intentionally avoids any hard dependence on ``src.assembler``:
integrator-level tests wrap small analytic ODEs and exercise the same code
paths as the full PDE, so this phase can be validated before Phase B is merged.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Generalized-α parameters
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GenAlphaParams:
    """Jansen–Whiting–Hulbert generalized-α parameters for first-order systems.

    Parameters are derived from a single input ``rho_inf`` ∈ [0, 1], the
    spectral radius at the high-frequency limit. ``rho_inf = 1`` recovers the
    undamped midpoint rule; ``rho_inf = 0`` yields asymptotic annihilation of
    the highest-frequency mode.

    Formulas (Chung & Hulbert, first-order form)::

        α_m = (2 − ρ∞) / (1 + ρ∞)
        α_f = 1       / (1 + ρ∞)
        γ   = 1/2 + α_m − α_f

    These values simultaneously guarantee second-order accuracy and
    unconditional stability on linear problems.
    """

    rho_inf: float
    alpha_m: float = field(init=False)
    alpha_f: float = field(init=False)
    gamma: float = field(init=False)

    def __post_init__(self) -> None:
        if not (0.0 <= self.rho_inf <= 1.0):
            raise ValueError("rho_inf must lie in [0, 1]")
        rho = float(self.rho_inf)
        object.__setattr__(self, "alpha_m", (2.0 - rho) / (1.0 + rho))
        object.__setattr__(self, "alpha_f", 1.0 / (1.0 + rho))
        object.__setattr__(
            self, "gamma", 0.5 + self.alpha_m - self.alpha_f
        )


# ─────────────────────────────────────────────────────────────────────────────
# Newton–Raphson
# ─────────────────────────────────────────────────────────────────────────────


def newton_solve(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d0: jnp.ndarray,
    d_dot0: jnp.ndarray,
    tol: float = 1e-8,
    max_iter: int = 20,
    damping: float = 1.0,
):
    """Newton–Raphson solve of ``residual_fn(d) = 0``.

    The caller supplies a residual that is already a function of a single
    flat vector (typically the generalized-α α-level DOF vector); this keeps
    the interface agnostic of whether ``d`` refers to positions, velocities,
    or a combined state.

    The tangent ``K = ∂R/∂d`` is computed by ``jax.jacfwd``. Convergence is
    declared when both::

        ‖R‖_∞ < tol           and         ‖δd‖ / max(‖d‖, 1) < tol

    A light backtracking line-search is applied: if a step increases
    ‖R‖, the step is halved up to four times before being accepted anyway.

    Parameters
    ----------
    residual_fn : callable
        Returns the residual vector given the current iterate.
    d0, d_dot0 : arrays
        Initial iterate and its time derivative; ``d_dot0`` is returned
        unchanged and is exposed purely for symmetry with callers that
        wrap both quantities in a state.
    tol : float
        Absolute residual and relative-increment tolerance.
    max_iter : int
        Maximum number of outer Newton iterations.
    damping : float
        Global multiplier applied to the Newton step before line-search.

    Returns
    -------
    d_new : jnp.ndarray
    d_dot_new : jnp.ndarray
        Returned unchanged; real time-derivative updates happen in
        :class:`TimeStepper`.
    iters : int
        Number of Newton iterations performed.
    residual_history : list[float]
        ‖R‖_∞ after each iteration (including the initial residual).
    """
    d = jnp.asarray(d0)
    R = residual_fn(d)
    r_norm = float(jnp.linalg.norm(R, ord=jnp.inf))
    history = [r_norm]

    iters = 0
    for _it in range(1, max_iter + 1):
        if r_norm < tol:
            break
        iters = _it

        K = jax.jacfwd(residual_fn)(d)
        try:
            delta = jnp.linalg.solve(K, -R)
        except Exception:  # pragma: no cover — fall back to lstsq on singular K
            delta, *_ = jnp.linalg.lstsq(K, -R)

        step = damping * delta

        # Backtracking line-search
        d_trial = d + step
        R_trial = residual_fn(d_trial)
        r_trial = float(jnp.linalg.norm(R_trial, ord=jnp.inf))

        backtracks = 0
        while r_trial > r_norm and backtracks < 4:
            step = 0.5 * step
            d_trial = d + step
            R_trial = residual_fn(d_trial)
            r_trial = float(jnp.linalg.norm(R_trial, ord=jnp.inf))
            backtracks += 1

        d_norm = float(jnp.linalg.norm(d, ord=2))
        step_norm = float(jnp.linalg.norm(step, ord=2))
        rel_increment = step_norm / max(d_norm, 1.0)

        d = d_trial
        R = R_trial
        r_norm = r_trial
        history.append(r_norm)

        if r_norm < tol and rel_increment < tol:
            break

    return d, jnp.asarray(d_dot0), iters, history


# ─────────────────────────────────────────────────────────────────────────────
# Time stepper
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TimeState:
    """State carried between time steps.

    ``d`` and ``d_dot`` are flat vectors; higher-level code (e.g. the
    dict-based assembler interface) is expected to serialize to this layout
    before invoking the stepper.
    """

    d: jnp.ndarray
    d_dot: jnp.ndarray
    t: float


class TimeStepper:
    """Generalized-α driver for first-order semi-discrete systems.

    The integrator expects a residual callable of the form::

        R = residual_fn(d_alpha, d_dot_alpha, t_alpha, cache, params)

    where all non-DOF arguments are forwarded from :attr:`cache` and
    :attr:`params`. Newton's method closes over the α-level state and solves
    for the end-of-step acceleration ``ḋ_{n+1}``, from which ``d_{n+1}`` is
    recovered via the γ-update.

    The class also supports a "flat residual" mode — useful for verification
    on toy ODEs and unit tests — in which ``residual_fn`` has the reduced
    signature ``R(d_alpha, d_dot_alpha, t_alpha)``. Pass ``cache`` / ``params``
    as ``None`` to select it.
    """

    def __init__(
        self,
        residual_fn: Callable[..., jnp.ndarray],
        cache: Any,
        params: Any,
        genalpha: GenAlphaParams,
        dt: float,
    ) -> None:
        self.residual_fn = residual_fn
        self.cache = cache
        self.params = params
        self.genalpha = genalpha
        self.dt = float(dt)
        self.step_history: list[dict] = []

    # ------------------------------------------------------------------ core

    def _eval_residual(self, d, d_dot, t):
        if self.cache is None and self.params is None:
            return self.residual_fn(d, d_dot, t)
        return self.residual_fn(d, d_dot, t, self.cache, self.params)

    def step(self, state: TimeState) -> TimeState:
        """Advance the state by ``dt`` using one generalized-α step."""
        a = self.genalpha
        dt = self.dt
        d_n, d_dot_n, t_n = state.d, state.d_dot, state.t

        # Predictor: keep d_dot_{n+1} = d_dot_n initially; then d_{n+1} is the
        # γ-update.  (Standard choice — "same-velocity" predictor.)
        d_dot_pred = d_dot_n
        d_pred = d_n + dt * d_dot_n  # γ * (d_dot_pred - d_dot_n) = 0

        # Newton on d_dot_{n+1}: parametrize the α-levels through ḋ_{n+1}.
        def residual_of_acc(d_dot_np1):
            d_np1 = d_n + dt * d_dot_n + a.gamma * dt * (d_dot_np1 - d_dot_n)
            # Jansen convention: α weights the new level.
            d_alpha = (1.0 - a.alpha_f) * d_n + a.alpha_f * d_np1
            d_dot_alpha = (1.0 - a.alpha_m) * d_dot_n + a.alpha_m * d_dot_np1
            t_alpha = t_n + a.alpha_f * dt
            return self._eval_residual(d_alpha, d_dot_alpha, t_alpha)

        d_dot_np1, _, iters, res_hist = newton_solve(
            residual_of_acc,
            d_dot_pred,
            d_dot_pred,
        )
        d_np1 = d_n + dt * d_dot_n + a.gamma * dt * (d_dot_np1 - d_dot_n)

        self.step_history.append(
            {
                "t_start": t_n,
                "t_end": t_n + dt,
                "newton_iters": iters,
                "residual_norms": res_hist,
            }
        )

        # Silence the unused-predictor reference (kept as documentation).
        _ = d_pred

        return TimeState(d=d_np1, d_dot=d_dot_np1, t=t_n + dt)

    # ------------------------------------------------------------------ bulk

    def run(
        self,
        state0: TimeState,
        n_steps: int,
        callbacks: Sequence[Callable[[TimeState], None]] = (),
    ) -> list[TimeState]:
        """Run ``n_steps`` generalized-α steps, returning the full history."""
        history = [state0]
        state = state0
        for _ in range(n_steps):
            state = self.step(state)
            for cb in callbacks:
                cb(state)
            history.append(state)
        return history


# ─────────────────────────────────────────────────────────────────────────────
# Dirichlet application helper (flat form)
# ─────────────────────────────────────────────────────────────────────────────


def apply_dirichlet_flat(
    R: jnp.ndarray,
    K: jnp.ndarray | None,
    dof_indices: Iterable[int],
    values: Iterable[float],
    current: Iterable[float],
):
    """Strong BC application in flat form.

    Sets ``R[i] = current[i] − values[i]`` at each prescribed DOF; if a
    tangent ``K`` is supplied, zeros the corresponding rows and columns and
    places a unit diagonal. Mirrors the convention used by the assembler.
    """
    dof = jnp.asarray(list(dof_indices), dtype=jnp.int32)
    vals = jnp.asarray(list(values))
    curr = jnp.asarray(list(current))

    R_new = R.at[dof].set(curr - vals)
    if K is None:
        return R_new, None

    K_new = K.at[dof, :].set(0.0)
    K_new = K_new.at[:, dof].set(0.0)
    K_new = K_new.at[dof, dof].set(1.0)
    return R_new, K_new


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint / restart
# ─────────────────────────────────────────────────────────────────────────────


def save_state(state: TimeState, path: str, dt: float | None = None, seed: int = 0) -> None:
    """Persist a :class:`TimeState` to ``path`` using ``numpy.savez``."""
    np.savez(
        path,
        d=np.asarray(state.d),
        d_dot=np.asarray(state.d_dot),
        t=np.asarray(state.t),
        dt=np.asarray(dt if dt is not None else 0.0),
        seed=np.asarray(seed),
    )


def load_state(path: str) -> tuple[TimeState, float, int]:
    """Load a previously-saved checkpoint, returning (state, dt, seed)."""
    data = np.load(path if path.endswith(".npz") else path + ".npz")
    state = TimeState(
        d=jnp.asarray(data["d"]),
        d_dot=jnp.asarray(data["d_dot"]),
        t=float(data["t"]),
    )
    return state, float(data["dt"]), int(data["seed"])


# ─────────────────────────────────────────────────────────────────────────────
# Eigen-diagnostics
# ─────────────────────────────────────────────────────────────────────────────


def spectrum(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d: jnp.ndarray,
    d_dot: jnp.ndarray,
    k: int = 6,
) -> jnp.ndarray:
    """Return the ``k`` largest-magnitude eigenvalues of the Newton tangent.

    Uses a dense ``jnp.linalg.eigvals`` on ``∂R/∂d`` linearized at ``d``.
    Sufficient for small verification problems; production-scale runs should
    instead use a matrix-free Lanczos / Arnoldi iteration (e.g.
    ``scipy.sparse.linalg.eigs`` wrapped around a JVP).

    Parameters
    ----------
    residual_fn : callable
        Residual of a single flat argument ``d`` (i.e. with ``d_dot`` and
        time already bound — typically the α-level closure from
        :meth:`TimeStepper.step`).
    d, d_dot : arrays
        Linearization point.
    k : int
        Number of eigenvalues to return.

    Returns
    -------
    jnp.ndarray of shape (k,)
        Complex eigenvalues sorted by descending magnitude.
    """
    del d_dot  # kept in the signature for symmetry; the closure owns it
    K = jax.jacfwd(residual_fn)(d)
    eigs = jnp.linalg.eigvals(K)
    order = jnp.argsort(-jnp.abs(eigs))
    return eigs[order][:k]
