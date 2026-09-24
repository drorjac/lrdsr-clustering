from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The default Huber constant, in units of the noise sd: 95% efficiency under
#: Gaussian noise, and the best of the sweep in
#: ``experiments/losses/huber_delta.py`` (tuning seeds) across four noise laws.
HUBER_DELTA = 1.345


def huber_loss(residual: np.ndarray, delta: float = HUBER_DELTA) -> np.ndarray:
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


#: MAD -> standard deviation for Gaussian data: 1 / Phi^-1(3/4).
MAD_TO_SD = 1.4826


def noise_scale(values: np.ndarray, eps: float = 1e-9) -> float:
    """The noise level in standard-deviation units: the normal-consistent MAD.

    This is the scale a loss is applied at. :func:`robust_scale` returns the
    raw MAD, which is ``0.6745 sigma`` for Gaussian noise; dividing residuals by
    it made a Huber ``delta = 1.5`` really ``~1.0 sigma``, where Huber is only
    ~90% efficient (``lrdsr.theory.losses``), and cost the default estimator
    about 30% in window error under Gaussian noise. With this scale every
    tuning constant means what it means in the robust-statistics literature.
    The fallbacks of :func:`robust_scale` (IQR / 1.349, sd) are already in sd
    units, so only the MAD branch is rescaled.
    """
    values = np.asarray(values, dtype=float)
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    if np.isfinite(mad) and mad >= eps:
        return float(MAD_TO_SD * mad)
    return robust_scale(values, eps=eps)


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
    robust_delta: float = HUBER_DELTA,
    scale: float | None = None,
    loss: str = "huber",
    loss_params: dict | None = None,
    scale_convention: str = "sd",
) -> float:
    """Mean loss of one window's residual under one candidate mechanism.

    The residual is divided by a scale before the loss so that
    ``robust_delta`` (and every other loss constant) is not tied to the units
    of y. Which scale is used decides what the score can see:

    ``scale=None`` (the former ``residual_scale='per_window'`` behaviour)
        the window's OWN robust scale -- :func:`noise_scale`, in sd units, or
        the raw MAD with ``scale_convention="mad"`` (the pre-2026-09-24
        behaviour, kept so older numbers can be reproduced). This makes the score essentially
        scale-invariant: a residual r and 10*r receive the same score, so the
        magnitude of the misfit -- usually the main evidence that a window
        belongs to another mechanism -- is divided out, and only the residual's
        shape survives.
    ``scale=<float>`` (a shared scale, e.g. the noise level)
        magnitude information is preserved and windows are comparable across
        candidate mechanisms.

    ``loss`` picks the per-sample penalty from :data:`LOSSES`. The default is
    ``"huber"`` with ``robust_delta = 1.345`` in units of the noise sd (95%
    Gaussian efficiency), chosen on the tuning seeds in
    ``experiments/losses/huber_delta.py``. Before 2026-09-24 it was 1.5 in
    raw-MAD units, i.e. ~1.0 sd; ``scale_convention="mad"`` with
    ``robust_delta=1.5`` reproduces that.

    See ``GroupedDCSR(residual_scale=..., loss=...)``, which supplies both.
    """
    residual = np.asarray(y) - np.asarray(pred)
    if scale is None:
        s = robust_scale(residual) if scale_convention == "mad" else noise_scale(residual)
    else:
        s = max(float(scale), 1e-9)
    params = dict(loss_params or {})
    if loss == "huber":
        params.setdefault("delta", robust_delta)
    return float(np.mean(loss_value(residual / s, loss, **params)))


# ==========================================================================
# the loss family
# ==========================================================================
# Each loss is a per-sample penalty rho(r) on a residual already divided by a
# scale, with its influence function psi = rho' and psi' alongside. psi and
# psi' are what the theory needs: for a small gap, the window error of a
# decision made with loss rho is Q(sqrt(n rho eta)/2), with the efficiency
#
#     eta = sigma^2 (E psi'(e))^2 / E psi(e)^2        (e the noise, scaled)
#
# (``lrdsr.theory.losses``). Squared loss has eta = sigma^2 / Var(e), i.e. 1
# under Gaussian noise and much less under heavy tails; the likelihood-ratio
# loss -log p(e) attains the maximum, sigma^2 times the Fisher information.
#
# The tuning constants are the textbook 95%-Gaussian-efficiency ones.

def squared_loss(residual: np.ndarray) -> np.ndarray:
    """``r^2 / 2``: the Gaussian negative log-likelihood, up to a constant."""
    r = np.asarray(residual, dtype=float)
    return 0.5 * r ** 2


def absolute_loss(residual: np.ndarray) -> np.ndarray:
    """``|r|``: the Laplace negative log-likelihood, up to a constant."""
    return np.abs(np.asarray(residual, dtype=float))


def cauchy_loss(residual: np.ndarray, c: float = 2.3849) -> np.ndarray:
    """``c^2/2 log(1 + (r/c)^2)``: redescending in influence, never in value."""
    r = np.asarray(residual, dtype=float)
    return 0.5 * c ** 2 * np.log1p((r / c) ** 2)


def tukey_loss(residual: np.ndarray, c: float = 4.685) -> np.ndarray:
    """Tukey's biweight: bounded, so one wild sample costs at most ``c^2/6``."""
    r = np.asarray(residual, dtype=float)
    u = np.clip(np.abs(r) / c, 0.0, 1.0)
    return (c ** 2 / 6.0) * (1.0 - (1.0 - u ** 2) ** 3)


def student_t_loss(residual: np.ndarray, nu: float = 4.0) -> np.ndarray:
    """``(nu+1)/2 log(1 + r^2/nu)``: the Student-t negative log-likelihood.

    The one member of the family whose shape is a *parameter*: ``nu -> inf``
    is the squared loss, ``nu = 1`` is Cauchy. :func:`learn_loss` fits ``nu``
    to the residuals, which is how the loss is learned rather than chosen.
    """
    r = np.asarray(residual, dtype=float)
    nu = float(nu)
    return 0.5 * (nu + 1.0) * np.log1p(r ** 2 / nu)


def _psi(r: np.ndarray, loss: str, **p) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    if loss == "squared":
        return r
    if loss == "absolute":
        return np.sign(r)
    if loss == "huber":
        d = p.get("delta", HUBER_DELTA)
        return np.clip(r, -d, d)
    if loss == "cauchy":
        c = p.get("c", 2.3849)
        return r / (1.0 + (r / c) ** 2)
    if loss == "tukey":
        c = p.get("c", 4.685)
        return np.where(np.abs(r) <= c, r * (1.0 - (r / c) ** 2) ** 2, 0.0)
    if loss == "student_t":
        nu = p.get("nu", 4.0)
        return (nu + 1.0) * r / (nu + r ** 2)
    raise ValueError(f"unknown loss {loss!r}; expected one of {LOSSES}")


def _psi_prime(r: np.ndarray, loss: str, **p) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    if loss == "squared":
        return np.ones_like(r)
    if loss == "absolute":
        # a distribution, not a function: E psi' = 2 p(0). Callers that need
        # the expectation use loss_efficiency, which handles it.
        return np.zeros_like(r)
    if loss == "huber":
        return (np.abs(r) <= p.get("delta", HUBER_DELTA)).astype(float)
    if loss == "cauchy":
        c = p.get("c", 2.3849)
        u = (r / c) ** 2
        return (1.0 - u) / (1.0 + u) ** 2
    if loss == "tukey":
        c = p.get("c", 4.685)
        u = (r / c) ** 2
        return np.where(np.abs(r) <= c, (1.0 - u) * (1.0 - 5.0 * u), 0.0)
    if loss == "student_t":
        nu = p.get("nu", 4.0)
        return (nu + 1.0) * (nu - r ** 2) / (nu + r ** 2) ** 2
    raise ValueError(f"unknown loss {loss!r}; expected one of {LOSSES}")


_RHO = {
    "squared": lambda r, **p: squared_loss(r),
    "absolute": lambda r, **p: absolute_loss(r),
    "huber": lambda r, **p: huber_loss(r, delta=p.get("delta", HUBER_DELTA)),
    "cauchy": lambda r, **p: cauchy_loss(r, c=p.get("c", 2.3849)),
    "tukey": lambda r, **p: tukey_loss(r, c=p.get("c", 4.685)),
    "student_t": lambda r, **p: student_t_loss(r, nu=p.get("nu", 4.0)),
}

#: Every per-sample loss the assignment cost can use.
LOSSES = tuple(_RHO)


def loss_value(residual: np.ndarray, loss: str = "huber", **params) -> np.ndarray:
    """``rho(r)`` for a named loss, elementwise, on a scaled residual."""
    if loss not in _RHO:
        raise ValueError(f"unknown loss {loss!r}; expected one of {LOSSES}")
    return _RHO[loss](residual, **params)


def loss_psi(residual: np.ndarray, loss: str = "huber", **params) -> np.ndarray:
    """The influence function ``psi = rho'``: how hard one sample pulls."""
    return _psi(residual, loss, **params)


def loss_psi_prime(residual: np.ndarray, loss: str = "huber", **params) -> np.ndarray:
    """``psi'``, the curvature the small-gap theory needs."""
    return _psi_prime(residual, loss, **params)


# ==========================================================================
# learning the loss
# ==========================================================================
@dataclass(frozen=True)
class LearnedLoss:
    """A loss fitted to residuals: its name, its parameters, and its scale.

    ``loss``/``params`` go straight into :func:`loss_value`; ``scale`` is what
    residuals are divided by first. ``loglik`` is the mean per-sample
    log-likelihood of the noise model the loss is the negative log of, which
    is what the candidates were compared on.
    """
    loss: str
    params: dict
    scale: float
    loglik: float


def fit_student_t(residuals: np.ndarray, nu_bounds=(0.8, 200.0),
                  max_samples: int = 20_000, seed: int = 0) -> tuple[float, float, float]:
    """Maximum-likelihood ``(nu, scale, mean loglik)`` of a zero-mean Student-t.

    Zero location because a residual under a fitted law has mean zero by
    construction; freeing it only lets the fit trade location against tails.
    ``scale`` is profiled for each ``nu`` by the EM fixed point
    ``s^2 = mean(w r^2)``, ``w = (nu+1)/(nu + r^2/s^2)``, and ``nu`` is found
    by a bounded 1-D search on the profiled likelihood -- which is smooth, so
    this is robust where a joint 2-D optimiser is not.
    """
    from scipy.optimize import minimize_scalar
    from scipy.special import gammaln

    r = np.asarray(residuals, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if r.size > max_samples:
        r = np.random.default_rng(seed).choice(r, max_samples, replace=False)
    r2 = r ** 2
    s0 = robust_scale(r) * 1.4826

    def profile(nu):
        s2 = s0 ** 2
        for _ in range(60):
            w = (nu + 1.0) / (nu + r2 / s2)
            s2_new = max(float(np.mean(w * r2)), 1e-18)
            if abs(s2_new - s2) <= 1e-10 * s2:
                s2 = s2_new
                break
            s2 = s2_new
        ll = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi * s2)
              - 0.5 * (nu + 1) * np.log1p(r2 / (nu * s2)))
        return float(np.mean(ll)), float(np.sqrt(s2))

    lo, hi = np.log(nu_bounds[0]), np.log(nu_bounds[1])
    res = minimize_scalar(lambda t: -profile(np.exp(t))[0], bounds=(lo, hi),
                          method="bounded", options={"xatol": 1e-3})
    nu = float(np.exp(res.x))
    ll, s = profile(nu)
    return nu, s, ll


def learn_loss(residuals: np.ndarray, candidates=("squared", "absolute", "student_t"),
               ) -> LearnedLoss:
    """Pick the loss whose noise model best explains ``residuals``.

    A loss is a noise model in disguise: squared is Gaussian, absolute is
    Laplace, ``student_t`` is Student-t with its ``nu`` fitted. Each is fitted
    by maximum likelihood (zero mean) and the best mean log-likelihood wins.
    The Student-t nests the Gaussian (``nu -> inf``) and approaches Cauchy
    (``nu = 1``), so on its own it already interpolates between the squared
    and a redescending loss; the other two are there so a clean or a Laplace
    residual is named for what it is.

    No label is read: the residuals are those of the *current* assignment.
    """
    r = np.asarray(residuals, dtype=float).ravel()
    r = r[np.isfinite(r)]
    fits = []
    if "squared" in candidates:
        s = float(np.sqrt(np.mean(r ** 2))) + 1e-12
        ll = float(np.mean(-0.5 * np.log(2 * np.pi * s ** 2) - 0.5 * r ** 2 / s ** 2))
        fits.append(LearnedLoss("squared", {}, s, ll))
    if "absolute" in candidates:
        b = float(np.mean(np.abs(r))) + 1e-12
        ll = float(np.mean(-np.log(2 * b) - np.abs(r) / b))
        fits.append(LearnedLoss("absolute", {}, b, ll))
    if "student_t" in candidates:
        nu, s, ll = fit_student_t(r)
        fits.append(LearnedLoss("student_t", {"nu": nu}, s, ll))
    if not fits:
        raise ValueError(f"no known candidate in {candidates}")
    return max(fits, key=lambda f: f.loglik)


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
