"""Mechanism space: the window statistic in which regimes are actually apart.

The paper's thesis is that *similarity in data space is not similarity of
mechanism*. This module applies that thesis to the one step of LR-DSR that
was still done in data space -- the initial partition -- by moving every
window to a coordinate system whose axes are **law coefficients** rather than
data summaries.

The construction
----------------
Write each window's design in a library ``Phi_w = [N_w | B_w]``: `N_w` the
columns whose coefficient is a per-window **nuisance** (a path term whose
strength varies with the link), `B_w` the columns that could carry a regime.
Profile the nuisance out of the window, subtract the law that fits *every*
window, put the signal columns on a common orthonormal footing, and project:

.. code-block:: text

    B~_w = (I - P_{N_w}) B_w,  y~_w = (I - P_{N_w}) y_w     profile out
    r_w  = y~_w - B~_w beta_bar(group of w)                  remove the shared
    G    = mean_w B~_w^T B~_w = V diag(lam) V^T              pooled Gram
    C_w  = B~_w V_r lam_r^-1/2                               common footing
    s_w  = C_w^T r_w                                         the statistic

with the truncation ``lam_r > tol * lam_max`` discarding directions the
library cannot resolve, and the ``s_w`` centred over windows at the end.

Two details earn their place, and each was put there by a measurement:

*Subtract the pooled law first.* The projection ``C_w^T y_w`` has mean
``(C_w^T C_w) gamma_k``, and ``C_w^T C_w`` is the identity only on average;
a shared component then leaks back through the per-window design
fluctuation. Removing it globally first leaves a leak proportional to the
*gap* instead, which is the thing being measured anyway. Without this step a
shared term 64x the gap costs the whole signal.

*Remove it per group, not per window.* ``groups`` names what a nuisance is
constant over -- for sensor data, the sensor. Everything constant within a
sensor (its gain, its offset, its calibration) is then absorbed, and the
regime gap survives untouched because it is exactly the thing that varies
*within* the sensor. Profiling the same term out of each window separately
looks equivalent and is not: when the nuisance is nearly collinear with the
regime gap over the observed inputs, a per-window profile deletes the signal
along with the nuisance. Use ``nuisance`` only for a term whose strength
genuinely moves window to window.

*Project, do not solve.* Reading the window's coefficients off with
``(C_w^T C_w)^-1`` would remove that dependence exactly, but the library is
deliberately collinear and the inverse amplifies noise in its weak
directions far more than the leak it removes (measured: it loses every pair).

Why this is the right space
---------------------------
If the laws lie in the span of ``B`` -- ``f_k = B beta_k`` -- then with
``gamma_k = L^T beta_k``

    s_w ~ N( gamma_{z_w} - gamma_bar,  sigma^2 (C_w^T C_w)^-1 ),
    C_w^T C_w = I  on average,

so the windows form an **isotropic** Gaussian mixture whose centres are
separated by

    || gamma_j - gamma_k ||^2 = (beta_j - beta_k)^T G (beta_j - beta_k)
                              = n E[g^2] = n rho sigma^2 ,

i.e. exactly the separation of ``lrdsr.theory``. Three things follow, and
each is verified in ``experiments/``:

1. **The Bayes error of clustering in this space is the theory's oracle
   error** ``Q(sqrt(n rho)/2)``. Plain K-means -- the simplest clustering
   there is -- is the *correct* rule here, because the construction is what
   makes the mixture isotropic. In data space no clustering is correct.
2. **Any component shared by the regimes cancels**, exactly as it does for
   the oracle: a common ``u`` in the span of ``B`` moves every ``beta_k`` by
   the same vector, and the centring removes it.
3. **A nuisance whose coefficient varies per window is removed too.** A
   day's overall level, for example, leaves the statistic altogether.

The invariance in (2) holds for shared components **inside** the library
span. A shared component outside it is absorbed by a window-dependent
best approximation and survives as jitter, which is the honest limit of the
construction and is measured in ``experiments/estimator/shared_component.py``.
"""
from __future__ import annotations

import numpy as np

from .backends import FastSymbolicRegressor

#: The one robustness constant: the fraction of most extreme windows that do
#: not vote for a cluster centre or for a reported spread. Fixed at the
#: conventional 1% a priori, never tuned -- see :func:`inliers`.
TRIM = 0.01

__all__ = [
    "inliers",
    "library_basis",
    "mechanism_features",
    "mechanism_init",
    "mechanism_noise",
    "rho_from_partition",
    "separation_matrix",
    "separation_unlabelled",
]


def library_basis(
    X: np.ndarray,
    feature_names: list[str] | None = None,
    include_trig: bool = True,
    include_log: bool = True,
    include_interactions: bool = True,
) -> np.ndarray:
    """The fast backend's own term library as a design matrix (intercept first).

    Using the *same* library the regressor searches is what makes the
    statement "the laws lie in the span of ``B``" mean something: a law the
    backend could find is a law this space can see.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    reg = FastSymbolicRegressor(
        feature_names=feature_names, include_trig=include_trig,
        include_log=include_log, include_interactions=include_interactions,
    )
    terms = reg._make_library(X)          # the library IS the interface here
    return np.column_stack([np.ones(len(X))] + [t.values for t in terms])


def _as_matrix(fn, X_flat: np.ndarray) -> np.ndarray | None:
    if fn is None:
        return None
    M = np.asarray(fn(X_flat), dtype=float)
    return M[:, None] if M.ndim == 1 else M


def mechanism_features(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    basis=None,
    nuisance=None,
    groups: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    ridge: float = 1e-6,
    center: bool = True,
) -> np.ndarray:
    """Per-window whitened law coefficients: the mechanism-space coordinates.

    Parameters
    ----------
    X_seq, y_seq : (W, n, d), (W, n)
        The windows, as the estimator sees them.
    basis : callable ``X_flat -> (N, p)``, optional
        The signal columns ``B``. Defaults to :func:`library_basis`, i.e. the
        symbolic backend's own term library.
    nuisance : callable ``X_flat -> (N, q)``, optional
        Columns whose coefficient is free **per window** and profiled out
        before anything else. Use it only for a term whose strength really
        does move window to window: anything constant within a link belongs
        in ``basis`` with ``groups`` instead, because a per-window profile
        also deletes whatever part of the signal is collinear with it.
    groups : (W,) array, optional
        Which windows share a nuisance -- the link, on real data. The pooled
        law and the centring are computed within each group, so a per-group
        gain, length or path coefficient is absorbed while the within-group
        regime contrast is kept. Default: one group.
    ridge : float
        Eigenvalue floor, relative to the largest, for the pooled Gram. The
        library is deliberately collinear (``x``, ``x^3``, ``sin x``), so the
        Gram is near-singular by construction; directions below the floor are
        dropped rather than inverted.
    center : bool
        Subtract the mean over windows, which is what removes a shared
        component (and is a no-op for K-means, but not for what is reported).

    Returns
    -------
    (W, p) array
        ``s_w``, in units of the noise: under the model its covariance is
        ``sigma^2 I``, so a Euclidean distance here is a likelihood ratio.
    """
    X_seq = np.asarray(X_seq, dtype=float)
    y_seq = np.asarray(y_seq, dtype=float)
    if X_seq.ndim != 3:
        raise ValueError("X_seq must be (W, n, d)")
    W, n, d = X_seq.shape
    if y_seq.shape != (W, n):
        raise ValueError("y_seq must be (W, n) and match X_seq")

    X_flat = X_seq.reshape(-1, d)
    B_all = (library_basis(X_flat, feature_names=feature_names)
             if basis is None else _as_matrix(basis, X_flat))
    N_all = _as_matrix(nuisance, X_flat)
    if B_all.shape[0] != W * n:
        raise ValueError("basis must return one row per sample")
    p = B_all.shape[1]

    Bt = np.empty((W, n, p))
    yt = np.empty((W, n))
    for w in range(W):
        sl = slice(w * n, (w + 1) * n)
        Bw, yw = B_all[sl], y_seq[w]
        if N_all is not None:
            Nw = N_all[sl]
            # profile the nuisance out of BOTH the response and the signal
            # columns, so what is left is orthogonal to it in this window.
            coef, *_ = np.linalg.lstsq(Nw, np.column_stack([yw, Bw]), rcond=None)
            fit = Nw @ coef
            yw = yw - fit[:, 0]
            Bw = Bw - fit[:, 1:]
        Bt[w], yt[w] = Bw, yw

    # The law that fits every window of a group at once: what the regimes
    # SHARE there. One group unless the caller says otherwise.
    R = np.empty_like(yt)
    idx = (np.zeros(W, dtype=int) if groups is None
           else np.asarray(groups).reshape(-1))
    if idx.shape[0] != W:
        raise ValueError("groups must have one entry per window")
    for gid in np.unique(idx):
        sel = np.flatnonzero(idx == gid)
        Bg = Bt[sel].reshape(-1, p)
        beta_bar, *_ = np.linalg.lstsq(Bg, yt[sel].reshape(-1), rcond=None)
        R[sel] = yt[sel] - (Bg @ beta_bar).reshape(len(sel), n)

    # Common footing: eigen-whitening of the pooled Gram, truncated at the
    # directions the library can actually resolve. After this the pooled Gram
    # is the identity, so a Euclidean distance is a likelihood ratio.
    G = np.einsum("wnp,wnq->pq", Bt, Bt) / W
    lam, V = np.linalg.eigh(G)
    keep = lam > max(ridge, 1e-12) * float(lam.max())
    if not np.any(keep):
        raise ValueError("the basis has no resolvable direction on this design")
    M = V[:, keep] / np.sqrt(lam[keep])
    S = np.einsum("wnp,wn->wp", Bt @ M, R)
    if center:
        for gid in np.unique(idx):
            sel = np.flatnonzero(idx == gid)
            S[sel] -= S[sel].mean(axis=0, keepdims=True)
    return S


def mechanism_init(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    n_clusters: int,
    seed: int = 0,
    n_init: int = 30,
    trim: float = TRIM,
    **kwargs,
) -> np.ndarray:
    """K-means in mechanism space: the initial partition LR-DSR starts from.

    Plain K-means and **no standardisation**: the whitening has already made
    the within-regime covariance isotropic, and rescaling the columns
    afterwards would undo exactly that (measured: it costs the polynomial
    pair its whole signal). The centres are computed on the inner
    ``1 - trim`` of the windows and then every window is assigned to the
    nearest one (:func:`inliers`).
    """
    from sklearn.cluster import KMeans

    S = mechanism_features(X_seq, y_seq, **kwargs)
    keep = inliers(S, trim)
    km = KMeans(n_clusters=n_clusters, n_init=n_init,
                random_state=seed).fit(S[keep])
    return km.predict(S)


def mechanism_noise(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    basis=None,
    nuisance=None,
    feature_names: list[str] | None = None,
) -> float:
    """Noise level ``sigma``, from the per-window residual of a full fit.

    Each window is fitted with **all** the columns and its own coefficients,
    so whichever regime generated it is absorbed and what remains is noise.
    Pooled over windows with the right degrees of freedom. No law and no
    partition is assumed, which is what lets the separation measured in
    mechanism space be reported in units of sigma.
    """
    X_seq = np.asarray(X_seq, dtype=float)
    y_seq = np.asarray(y_seq, dtype=float)
    W, n, d = X_seq.shape
    X_flat = X_seq.reshape(-1, d)
    B_all = (library_basis(X_flat, feature_names=feature_names)
             if basis is None else _as_matrix(basis, X_flat))
    N_all = _as_matrix(nuisance, X_flat)
    M_all = B_all if N_all is None else np.column_stack([N_all, B_all])
    dof = max(n - M_all.shape[1], 1)
    rss = 0.0
    for w in range(W):
        Mw = M_all[w * n:(w + 1) * n]
        coef, *_ = np.linalg.lstsq(Mw, y_seq[w], rcond=None)
        rss += float(np.sum((y_seq[w] - Mw @ coef) ** 2))
    return float(np.sqrt(rss / (W * dof)))


def separation_matrix(
    S: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    sigma: float | None = None,
    unbiased: bool = True,
) -> np.ndarray:
    """Pairwise centre distances in mechanism space.

    With ``sigma`` from :func:`mechanism_noise` the entries are in units of
    the noise, and entry ``(j, k)`` estimates ``sqrt(n rho_{jk})`` -- which is
    what ties this space to ``lrdsr.theory``.

    ``unbiased`` subtracts the estimation variance of the two centres --
    the within-group scatter divided by the group sizes. The plain distance
    between two sample means of ``p``-vectors is biased upwards, by enough to
    matter at exactly the small separations where the question is
    interesting. The correction is measured from the groups themselves, so it
    needs neither ``sigma`` nor an isotropy assumption.
    """
    S = np.asarray(S, dtype=float)
    labels = np.asarray(labels, dtype=int)
    p = S.shape[1]
    sizes = np.array([int(np.sum(labels == k)) for k in range(n_clusters)])
    centres = np.vstack([
        S[labels == k].mean(axis=0) if sizes[k] else np.full(p, np.nan)
        for k in range(n_clusters)
    ])
    # total within-group variance of each group, i.e. trace of its covariance
    scatter = np.array([
        float(np.sum(S[labels == k].var(axis=0, ddof=1))) if sizes[k] > 1 else 0.0
        for k in range(n_clusters)
    ])
    D = np.zeros((n_clusters, n_clusters))
    for j in range(n_clusters):
        for k in range(n_clusters):
            d2 = float(np.sum((centres[j] - centres[k]) ** 2))
            if unbiased and j != k and sizes[j] and sizes[k]:
                d2 -= scatter[j] / sizes[j] + scatter[k] / sizes[k]
            D[j, k] = np.sqrt(max(d2, 0.0))
    return D if sigma is None else D / float(sigma)


def separation_unlabelled(
    S: np.ndarray,
    sigma: float,
    window_len: int,
    pi: float = 0.5,
    trim: float = TRIM,
) -> dict:
    """Separation-to-noise ratio ``rho`` from unlabelled windows.

    Everything above needs to know which window came from which regime. This
    does not, and that is what makes the theory testable where no label
    exists.

    Under the model a window's statistic has covariance ``sigma^2 I``
    whatever its regime, so **all** the extra spread of ``S`` is the regimes
    pulling apart. For a two-regime mixture with shares ``pi, 1 - pi``

    .. code-block:: text

        tr Cov(S) = p sigma^2 + pi (1 - pi) || Delta ||^2 ,
        || Delta ||^2 = n rho sigma^2

    so ``rho`` follows from the trace. The trace of a sample covariance is
    unbiased, which the largest eigenvalue is not (it carries the
    Marchenko-Pastur inflation ``(1 + sqrt(p/W))^2``), so the trace is what
    is used.

    What it means on real data. The identity charges *every* source of
    window-to-window spread in the law to the regime gap: real links also
    differ from each other, and events differ within a link. ``rho`` from
    this estimator is therefore an **upper bound** on the separation the two
    regimes actually have, and the error it implies is a **lower bound** on
    what any method can achieve -- the direction that matters for a claim
    that something is undetectable.

    The most extreme ``trim`` of windows are left out of the trace: a
    variance is not robust, and on real sensor data a handful of bad windows
    otherwise set it (measured in an earlier application: ``rho`` of 62
    untrimmed against 13 trimmed, a factor of five from 1% of the windows).

    ``pi`` is unknown without labels. The default ``0.5`` maximises
    ``pi(1-pi)`` and so gives the smallest ``rho`` consistent with the
    observed spread; ``pi_from_split`` reports what the leading direction
    suggests instead, for the sensitivity.

    Returns
    -------
    dict
        ``rho`` (at the given ``pi``), ``excess`` (the excess trace),
        ``trace``, ``noise_trace``, ``pi_from_split`` and ``rho_at_split``.
    """
    S = np.asarray(S, dtype=float)[inliers(S, trim)]
    W, p = S.shape
    sigma = float(sigma)
    trace = float(np.sum(S.var(axis=0, ddof=1)))
    noise_trace = p * sigma ** 2
    excess = trace - noise_trace
    denom = pi * (1.0 - pi) * window_len * sigma ** 2

    # A share read off the leading direction, for the sensitivity only: the
    # split of the top principal score at its own K-means threshold.
    Sc = S - S.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Sc, full_matrices=False)
    t = Sc @ vt[0]
    cut = 0.5 * (t[t >= np.median(t)].mean() + t[t < np.median(t)].mean())
    share = float(np.mean(t >= cut))
    share = min(max(share, 1.0 / W), 1.0 - 1.0 / W)
    d_split = share * (1.0 - share) * window_len * sigma ** 2

    return {
        "rho": float(max(excess, 0.0) / denom) if denom > 0 else float("nan"),
        "excess": float(excess),
        "trace": trace,
        "noise_trace": float(noise_trace),
        "n_dims": int(p),
        "n_windows": int(W),
        "pi_from_split": share,
        "rho_at_split": float(max(excess, 0.0) / d_split) if d_split > 0 else float("nan"),
    }


def rho_from_partition(
    S: np.ndarray,
    labels: np.ndarray,
    sigma: float,
    window_len: int,
    trim: float = TRIM,
) -> float:
    """``rho`` from the distance between two ESTIMATED regime centres.

    The companion to :func:`separation_unlabelled`, and still label-free: the
    partition is the one K-means found in this same space. Where the regimes
    are resolvable it is the sharper of the two (measured on the simulator:
    within 1.5% of the truth for ``rho >= 0.5``, against a steady +19% for
    the trace). Where they are not, K-means splits the noise anyway and this
    estimate inflates, while the trace does not -- so the two are reported
    together and their agreement is itself the diagnostic.
    """
    S = np.asarray(S, dtype=float)
    keep = inliers(S, trim)
    d = separation_matrix(S[keep], np.asarray(labels)[keep], 2,
                          sigma=sigma, unbiased=True)[0, 1]
    return float(d ** 2 / max(window_len, 1))


def inliers(S: np.ndarray, trim: float = TRIM) -> np.ndarray:
    """Boolean mask dropping the ``trim`` fraction of largest radii.

    Real data supply outliers in quantity -- a sensor fault, a hardware
    event, one input far outside the usual range -- and the largest
    ``||s_w||`` can be several hundred times the noise scale. Left in, plain
    K-means spends its two clusters separating
    0.2% of the windows from the other 99.8%: outlier detection, not regime
    discovery. Trimming the extreme radii before the centres are computed is
    the standard remedy, and the same idea as the Huber loss the assignment
    cost already uses -- bound what one bad window can do, do not let it
    choose the partition.

    Every window is still assigned; only the centres are protected. On clean
    Gaussian windows, where 1% of the radii are ordinary, this changes
    nothing measurable.
    """
    S = np.asarray(S, dtype=float)
    if not 0.0 <= trim < 0.5:
        raise ValueError("trim must be in [0, 0.5)")
    if trim == 0.0:
        return np.ones(len(S), dtype=bool)
    r = np.linalg.norm(S - np.median(S, axis=0, keepdims=True), axis=1)
    return r <= np.quantile(r, 1.0 - trim)
