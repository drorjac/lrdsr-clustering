"""Kernel and function-space bases: mechanism space without a symbolic library.

Everything in ``mechanism_space``, ``soft`` and ``classify`` takes a
``basis``: a callable ``X_flat -> (N, p)``. The default is the fast backend's
term library, which is what makes a recovered law a *named* expression -- and
what fails when the gap between two laws lies outside the library's span
(``experiments/problems``: ``sin 4x`` against ``sin 4.6x`` leaves 37% of the
gap outside, and every method pays for it).

This module supplies bases that span a function space instead of a list of
terms, so the span can be grown until it contains the gap:

==================  =======================================================
``FourierBasis``    intercept + ``H`` harmonics of a known period (a daily
                    or weekly cycle)
``CosineBasis``     the DCT-II basis ``cos(k pi u)`` on ``[lo, hi]``: a
                    Fourier basis with no wrap-around, for signals that are
                    not periodic (a time index)
``LegendreBasis``   orthogonal polynomials on ``[lo, hi]``
``NystromBasis``    the kernel feature map ``k(x, C) U L^{-1/2}`` of an RBF
                    kernel at ``m`` centres: the finite-rank RKHS
``RandomFourierBasis``  random features of the same RBF kernel
==================  =======================================================

The kernel view in one line: with a Nystrom basis, the mechanism-space
distance between two windows is the RKHS distance between their fitted
functions, measured under the pooled design. The window kernel
``S S^T`` (:func:`window_kernel`) is then a kernel **between windows** that
compares their laws, which any kernel machine can use.

What a basis costs. A basis of rank ``p`` puts ``p`` noise dimensions into
mechanism space whatever it captures of the gap. Growing ``p`` trades bias
(gap outside the span) for variance (noise dimensions) -- in clustering the
variance side is the high-dimensional K-means cost, in classification it is
the ``2p/m`` of ``lrdsr.theory.classification``. :func:`select_rank` chooses
``p`` without a label.

Bases that need data to be defined (``Nystrom`` centres, the ranges of the
polynomial bases) are fitted on the first inputs they see unless
:meth:`fit` is called first; a classifier calls ``fit`` on training inputs
only.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial import legendre

__all__ = [
    "CosineBasis",
    "FourierBasis",
    "LegendreBasis",
    "NystromBasis",
    "RandomFourierBasis",
    "loo_error",
    "make_basis",
    "select_rank",
    "window_kernel",
]


def _flat(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return X[:, None] if X.ndim == 1 else X


class _Basis:
    """Shared plumbing: lazily fitted, callable, with a rank and names."""

    intercept = True

    def fit(self, X_flat):
        return self

    @property
    def fitted(self) -> bool:
        return True

    def __call__(self, X_flat) -> np.ndarray:
        X = _flat(X_flat)
        if not self.fitted:
            self.fit(X)
        return self.transform(X)

    def transform(self, X):  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def rank(self) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(rank={self.rank})"


class FourierBasis(_Basis):
    """``[1, sin(h w x), cos(h w x)]_{h=1..H}`` of the first input, ``w = 2 pi / period``.

    ``period = 2 pi`` is the convention of ``experiments/realdata``, where
    ``x`` is already the hour as an angle.
    """

    def __init__(self, n_harmonics: int, period: float = 2 * np.pi, column: int = 0):
        self.n_harmonics = int(n_harmonics)
        self.period = float(period)
        self.column = column

    @property
    def rank(self) -> int:
        return 1 + 2 * self.n_harmonics

    @property
    def names(self) -> list[str]:
        return ["1"] + [f"{f}({h}wx)" for h in range(1, self.n_harmonics + 1)
                        for f in ("sin", "cos")]

    def transform(self, X):
        x = X[:, self.column] * (2 * np.pi / self.period)
        cols = [np.ones_like(x)]
        for h in range(1, self.n_harmonics + 1):
            cols += [np.sin(h * x), np.cos(h * x)]
        return np.column_stack(cols)


class _Ranged(_Basis):
    """A 1-D basis on ``[lo, hi]``: the range is data unless given."""

    def __init__(self, n_terms: int, lo: float | None = None, hi: float | None = None,
                 column: int = 0):
        self.n_terms = int(n_terms)
        self.lo, self.hi = lo, hi
        self.column = column

    @property
    def fitted(self) -> bool:
        return self.lo is not None and self.hi is not None

    def fit(self, X_flat):
        x = _flat(X_flat)[:, self.column]
        if self.lo is None:
            self.lo = float(np.min(x))
        if self.hi is None:
            self.hi = float(np.max(x))
        if self.hi <= self.lo:
            self.hi = self.lo + 1.0
        return self

    def _u(self, X) -> np.ndarray:
        return (X[:, self.column] - self.lo) / (self.hi - self.lo)

    @property
    def rank(self) -> int:
        return self.n_terms


class CosineBasis(_Ranged):
    """``cos(k pi u)``, ``k = 0..n_terms-1``, ``u`` the input rescaled to ``[0, 1]``.

    The DCT-II basis: complete for a smooth function on an interval, with
    no periodic wrap-around, so it is the Fourier basis for a time index.
    ``k = 0`` is the intercept.
    """

    def transform(self, X):
        u = self._u(X)
        return np.cos(np.pi * np.outer(u, np.arange(self.n_terms)))


class LegendreBasis(_Ranged):
    """Legendre polynomials ``P_0..P_{n_terms-1}`` of the input on ``[-1, 1]``."""

    def transform(self, X):
        u = 2.0 * self._u(X) - 1.0
        return legendre.legvander(u, self.n_terms - 1)


class NystromBasis(_Basis):
    """The rank-``m`` feature map of an RBF kernel: ``[1, k(x, C) U L^{-1/2}]``.

    Parameters
    ----------
    n_centers : int
        ``m``. In one input the centres are the ``m`` evenly spaced quantiles
        of the pooled inputs; in several, K-means centres.
    bandwidth : float, optional
        Kernel length scale. Default: ``scale`` times the median distance
        between neighbouring centres, so resolution grows with ``m`` and the
        rank is the one knob.
    scale : float
        See ``bandwidth``. ``1.0`` makes adjacent bumps overlap by about
        half, a smooth but not collinear basis.
    tol : float
        Eigenvalues of ``K_CC`` below ``tol * max`` are dropped: the kernel
        matrix at close centres is near-singular by construction.
    spacing : ``"quantile"`` or ``"uniform"``
        One input only: centres at evenly spaced quantiles of the pooled
        inputs (default -- resolution where the data are), or evenly spaced
        over their range (resolution everywhere). With a skewed design, such
        as wind speeds, quantile centres starve the sparse end of the range
        and the law is erratic there (measured in ``experiments/wind``).
    """

    def __init__(self, n_centers: int, bandwidth: float | None = None,
                 scale: float = 1.0, tol: float = 1e-8, seed: int = 0,
                 columns=None, spacing: str = "quantile"):
        if spacing not in ("quantile", "uniform"):
            raise ValueError("spacing must be 'quantile' or 'uniform'")
        self.spacing = spacing
        self.n_centers = int(n_centers)
        self.bandwidth = bandwidth
        self.scale = scale
        self.tol = tol
        self.seed = seed
        self.columns = columns
        self.centers_ = None

    @property
    def fitted(self) -> bool:
        return self.centers_ is not None

    def _cols(self, X):
        return X if self.columns is None else X[:, self.columns]

    def fit(self, X_flat):
        X = self._cols(_flat(X_flat))
        m = self.n_centers
        if X.shape[1] == 1 and self.spacing == "uniform":
            lo, hi = float(X[:, 0].min()), float(X[:, 0].max())
            C = (lo + (hi - lo) * (np.arange(m) + 0.5) / m)[:, None]
        elif X.shape[1] == 1:
            C = np.quantile(X[:, 0], (np.arange(m) + 0.5) / m)[:, None]
        else:
            from sklearn.cluster import KMeans
            sub = X if len(X) <= 20000 else X[np.random.default_rng(
                self.seed).choice(len(X), 20000, replace=False)]
            C = KMeans(m, n_init=4, random_state=self.seed).fit(sub).cluster_centers_
        self.scale_ = X.std(axis=0) + 1e-12
        Cs = C / self.scale_
        if self.bandwidth is None:
            D = np.sqrt(((Cs[:, None, :] - Cs[None, :, :]) ** 2).sum(-1))
            np.fill_diagonal(D, np.inf)
            self.bw_ = self.scale * float(np.median(D.min(axis=1))) if m > 1 else 1.0
        else:
            self.bw_ = float(self.bandwidth)
        self.centers_ = Cs
        Kcc = self._k(Cs, Cs)
        lam, U = np.linalg.eigh(Kcc)
        keep = lam > self.tol * lam.max()
        self.map_ = U[:, keep] / np.sqrt(lam[keep])
        return self

    def _k(self, A, B):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-0.5 * d2 / self.bw_ ** 2)

    @property
    def rank(self) -> int:
        return 1 + (self.map_.shape[1] if self.fitted else self.n_centers)

    def transform(self, X):
        Xs = self._cols(X) / self.scale_
        F = self._k(Xs, self.centers_) @ self.map_
        return np.column_stack([np.ones(len(X)), F])


class RandomFourierBasis(_Basis):
    """``[1, sqrt(2/D) cos(W x + b)]``: random features of an RBF kernel.

    The bandwidth defaults to the per-input standard deviation of the data
    over ``scale``; ``W`` is drawn once, from ``seed``, at ``fit``.
    """

    def __init__(self, n_features: int, bandwidth: float | None = None,
                 scale: float = 4.0, seed: int = 0):
        self.n_features = int(n_features)
        self.bandwidth = bandwidth
        self.scale = scale
        self.seed = seed
        self.W_ = None

    @property
    def fitted(self) -> bool:
        return self.W_ is not None

    def fit(self, X_flat):
        X = _flat(X_flat)
        rng = np.random.default_rng(self.seed)
        bw = (X.std(axis=0) / self.scale + 1e-12 if self.bandwidth is None
              else np.full(X.shape[1], float(self.bandwidth)))
        self.W_ = rng.normal(size=(X.shape[1], self.n_features)) / bw[:, None]
        self.b_ = rng.uniform(0, 2 * np.pi, self.n_features)
        return self

    @property
    def rank(self) -> int:
        return 1 + self.n_features

    def transform(self, X):
        F = np.sqrt(2.0 / self.n_features) * np.cos(X @ self.W_ + self.b_)
        return np.column_stack([np.ones(len(X)), F])


BASES = {"fourier": FourierBasis, "cosine": CosineBasis, "legendre": LegendreBasis,
         "nystrom": NystromBasis, "rff": RandomFourierBasis}


def make_basis(kind: str, size: int, **kwargs) -> _Basis:
    """``make_basis("nystrom", 12)`` etc. ``size`` is the family's own knob
    (harmonics, terms, centres or features)."""
    if kind not in BASES:
        raise KeyError(f"unknown basis {kind!r}; expected one of {list(BASES)}")
    return BASES[kind](size, **kwargs)


# ==========================================================================
# choosing the rank without a label
# ==========================================================================
def _windows(X_seq, y_seq):
    """Normalise ``(W, n, d)`` arrays or lists of ragged windows to lists."""
    if isinstance(X_seq, np.ndarray) and X_seq.ndim == 3:
        return [X_seq[w] for w in range(len(X_seq))], [np.asarray(y_seq[w], float)
                                                        for w in range(len(X_seq))]
    if isinstance(X_seq, np.ndarray) and X_seq.ndim == 2:
        return [X_seq[w][:, None] for w in range(len(X_seq))], [
            np.asarray(y_seq[w], float) for w in range(len(X_seq))]
    return [_flat(x) for x in X_seq], [np.asarray(v, float) for v in y_seq]


def loo_error(X_seq, y_seq, basis, nuisance=None, ridge: float = 1e-8) -> float:
    """Pooled leave-one-sample-out error of a *per-window* fit in ``basis``.

    Each window is fitted with its own coefficients, so whichever law made
    it is absorbed; the leave-one-out residual of a linear smoother is
    ``r_i / (1 - h_ii)``, exact and free. Windows with fewer samples than
    columns are skipped. No partition and no label is used: this asks how
    rich a basis a single window's law needs, which is the resolution the
    regimes' gap can at most have.
    """
    Xs, ys = _windows(X_seq, y_seq)
    if hasattr(basis, "fit") and not getattr(basis, "fitted", True):
        basis.fit(np.vstack(Xs))
    tot, cnt = 0.0, 0
    for Xw, yw in zip(Xs, ys, strict=True):
        B = basis(Xw)
        if nuisance is not None:
            N = np.asarray(nuisance(Xw), float)
            B = np.column_stack([N[:, None] if N.ndim == 1 else N, B])
        n, p = B.shape
        if n <= p + 1:
            continue
        # thin SVD: stable hat diagonal for a collinear basis
        U, s, _ = np.linalg.svd(B, full_matrices=False)
        keep = s > np.sqrt(ridge) * s.max()
        U = U[:, keep]
        h = np.sum(U ** 2, axis=1)
        r = yw - U @ (U.T @ yw)
        tot += float(np.sum((r / np.clip(1.0 - h, 1e-6, None)) ** 2))
        cnt += n
    return tot / max(cnt, 1)


def select_rank(X_seq, y_seq, kind: str, sizes, nuisance=None, rule: str = "min",
                **basis_kwargs) -> tuple[int, list[dict]]:
    """The basis size with the smallest pooled LOO error (:func:`loo_error`).

    ``rule="1se"`` takes the smallest size within one standard error of the
    minimum instead -- the conventional guard against a flat curve. Returns
    ``(size, table)`` with one row per size.
    """
    Xs, ys = _windows(X_seq, y_seq)
    pooled = np.vstack(Xs)
    rows = []
    for s in sizes:
        b = make_basis(kind, s, **basis_kwargs).fit(pooled)
        per = []
        for Xw, yw in zip(Xs, ys, strict=True):
            e = loo_error([Xw], [yw], b, nuisance=nuisance)
            per.append(e)
        per = np.asarray(per)
        rows.append({"kind": kind, "size": int(s), "rank": b.rank,
                     "loo": float(per.mean()),
                     "loo_se": float(per.std(ddof=1) / np.sqrt(len(per)))})
    losses = np.array([r["loo"] for r in rows])
    best = int(np.argmin(losses))
    if rule == "1se":
        thr = losses[best] + rows[best]["loo_se"]
        best = next(i for i, v in enumerate(losses) if v <= thr)
    elif rule != "min":
        raise ValueError("rule must be 'min' or '1se'")
    return rows[best]["size"], rows


def window_kernel(S: np.ndarray, kind: str = "linear", gamma: float | None = None
                  ) -> np.ndarray:
    """A kernel **between windows** from their mechanism-space coordinates.

    ``linear``: ``S S^T``, the inner product of the windows' fitted laws
    (their RKHS inner product when ``S`` came from a Nystrom basis).
    ``rbf``: ``exp(-gamma ||s_w - s_v||^2)``, with ``gamma`` defaulting to
    one over twice the median squared distance.
    """
    S = np.asarray(S, dtype=float)
    if kind == "linear":
        return S @ S.T
    if kind == "rbf":
        sq = np.sum(S ** 2, axis=1)
        D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * S @ S.T, 0.0)
        if gamma is None:
            gamma = 1.0 / (2.0 * np.median(D2[np.triu_indices_from(D2, 1)]) + 1e-12)
        return np.exp(-gamma * D2)
    raise ValueError("kind must be 'linear' or 'rbf'")
