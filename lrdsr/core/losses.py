from __future__ import annotations

import numpy as np


def huber_loss(residual: np.ndarray, delta: float = 1.5) -> np.ndarray:
    """Squared below `delta`, linear above it.

    Used rather than squared error because a single badly-explained window
    should not be able to drag a whole regime's law towards it — with `K`
    laws competing for every window, one outlier otherwise moves the
    partition, not just the fit.
    """
    residual = np.asarray(residual, dtype=float)
    a = np.abs(residual)
    return np.where(a <= delta, 0.5 * residual**2, delta * (a - 0.5 * delta))


def robust_scale(values: np.ndarray, eps: float = 1e-9) -> float:
    """A scale estimate that degrades gracefully: MAD, then IQR, then sd.

    Each fallback catches a case the one before it cannot: the MAD is zero
    when over half the values are identical (quantised data does this), and
    the IQR is zero when over three quarters are. Returning zero would make
    every scaled residual infinite, so the chain ends at `eps`.
    """
    values = np.asarray(values, dtype=float)
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    if not np.isfinite(mad) or mad < eps:
        q75, q25 = np.nanpercentile(values, [75, 25])
        mad = (q75 - q25) / 1.349
    if not np.isfinite(mad) or mad < eps:
        mad = np.nanstd(values)
    return float(max(mad, eps))


def normalize_cost_matrix(cost: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Robust global scaling while preserving zero as zero cost."""
    cost = np.asarray(cost, dtype=float)
    finite = cost[np.isfinite(cost)]
    if finite.size == 0:
        raise ValueError("Cost matrix has no finite values")
    scale = np.nanmedian(finite)
    if scale <= eps:
        scale = robust_scale(finite, eps=eps)
    return cost / max(scale, eps)


def aggregate_window_residual(
    y: np.ndarray,
    pred: np.ndarray,
    robust_delta: float = 1.5,
    scale: float | None = None,
) -> float:
    """Mean Huber residual of one window under one candidate mechanism.

    The residual is divided by a scale before the Huber loss so that
    ``robust_delta`` is not tied to the units of y. Which scale is used decides
    what the score can see:

    ``scale=None`` (the former ``residual_scale='per_window'`` behaviour)
        the window's OWN robust scale. This makes the score essentially
        scale-invariant: a residual r and 10*r receive the same score, so the
        magnitude of the misfit -- usually the main evidence that a window
        belongs to another mechanism -- is divided out, and only the residual's
        shape survives.
    ``scale=<float>`` (a shared scale, e.g. the noise level)
        magnitude information is preserved and windows are comparable across
        candidate mechanisms.

    See ``GroupedDCSR(residual_scale=...)``, which supplies a shared scale.
    """
    residual = np.asarray(y) - np.asarray(pred)
    s = robust_scale(residual) if scale is None else max(float(scale), 1e-9)
    return float(np.mean(huber_loss(residual / s, delta=robust_delta)))


def joint_cost(
    geometry_cost: np.ndarray,
    equation_cost: np.ndarray,
    alpha_geom: float,
    physics_cost: np.ndarray | None = None,
    lambda_phys: float = 0.0,
    complexity_cost: np.ndarray | None = None,
    beta_complexity: float = 0.0,
) -> np.ndarray:
    """The assignment cost: what each window would pay to join each regime.

    .. code-block:: text

        J[w,k] = alpha_geom       * D_geom(z_w, k)        # where it sits
               + (1 - alpha_geom) * D_eq(y_w, f_k(X_w))   # what explains it
               + lambda_phys      * D_phys(w, k)          # optional anchor
               + beta_complexity  * C(f_k)                # expression size

    Every term is **normalised by its own median before weighting**, which is
    the detail that makes `alpha_geom` mean anything: a geometry cost in
    standardised feature units and an equation cost in dB are not otherwise
    comparable, and the weight would silently be a units conversion rather
    than a choice. Zero stays zero under that scaling, so a perfectly
    explained window still costs nothing.

    At ``alpha_geom = 0`` assignment is purely mechanism-based. At
    ``alpha_geom = 1`` with Mahalanobis geometry this reduces exactly to a
    hard-EM GMM E-step, which is the sanity anchor at the other end.

    Returns
    -------
    (W, K) array
        Cost of assigning each window to each regime; the loop takes the
        row-wise argmin.
    """
    if not 0.0 <= alpha_geom <= 1.0:
        raise ValueError("alpha_geom must be in [0, 1]")

    g = normalize_cost_matrix(geometry_cost)
    e = normalize_cost_matrix(equation_cost)
    out = alpha_geom * g + (1.0 - alpha_geom) * e

    if physics_cost is not None and lambda_phys > 0:
        out = out + lambda_phys * normalize_cost_matrix(physics_cost)

    if complexity_cost is not None and beta_complexity > 0:
        c = np.asarray(complexity_cost, dtype=float)
        c = c / max(float(np.nanmedian(c)), 1e-9)
        out = out + beta_complexity * c

    return out
