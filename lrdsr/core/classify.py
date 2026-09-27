"""Supervised LR-DSR: classify a window by the law that explains it.

Clustering asks which of ``K`` unknown laws produced each window. When some
windows come with a class, the same model gives a classifier: fit one law
per class, and send a new window to the class whose law explains it best.
That is the generative (discriminant-analysis) classifier of the mixture
the rest of the project clusters, and it keeps what the method is for:

* **Windows may have different designs, or different lengths.** A window is
  scored by its own residual under each law at *its own* inputs, so a series
  with missing hours, irregular sampling times or a different length is
  scored as is -- nothing is interpolated onto a grid first.
* **The dimension that is paid for is the basis, not the window.** The plug-in
  error of a ``p``-dimensional law with ``m`` training windows per class is
  in ``lrdsr.theory.classification``; a raw-profile classifier pays the
  same price with ``p`` equal to the window length.

Two estimators:

``LawClassifier``     one law per class in a basis (dense least squares, or a
                      named expression from the fast symbolic backend),
                      per-window nuisance profiled out, Gaussian likelihood.
``MechanismFeatures`` the mechanism-space statistic as a scikit-learn
                      transformer fitted on training windows only: the
                      *feature domain* in which any classifier can run.

Windows are passed as ``(W, n, d)`` arrays or as lists of ``(n_w, d)`` arrays
with matching lists of ``(n_w,)`` responses.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .backends import FastSymbolicRegressor
from .kernel import _windows
from .soft import library_terms

__all__ = ["LawClassifier", "MechanismFeatures", "as_windows", "intercept"]


def as_windows(X_seq, y_seq):
    """``(list of (n_w, d) inputs, list of (n_w,) responses)`` from either layout."""
    return _windows(X_seq, y_seq)


def intercept(X_flat) -> np.ndarray:
    """The per-window level as a nuisance: one constant column."""
    return np.ones((len(X_flat), 1))


def _as_matrix(fn, X) -> np.ndarray:
    M = np.asarray(fn(X), dtype=float)
    return M[:, None] if M.ndim == 1 else M


class _Design:
    """Evaluates the basis and profiles the nuisance out, window by window."""

    def __init__(self, basis, feature_names, nuisance):
        self.basis = basis
        self.feature_names = feature_names
        self.nuisance = intercept if nuisance == "intercept" else nuisance

    def fit(self, Xs):
        if self.basis is not None and hasattr(self.basis, "fit"):
            self.basis.fit(np.vstack(Xs))
        return self

    def B(self, Xw):
        if self.basis is None:
            return library_terms(Xw, feature_names=self.feature_names)[0]
        return _as_matrix(self.basis, Xw)

    def profile(self, Xw, yw, Bw=None):
        """``(B~, y~, dof_lost)``: both residualised on the window's nuisance."""
        Bt, yt, q = self.profile_batch(Xw[None], yw[None],
                                       None if Bw is None else Bw[None])
        return Bt[0], yt[0], q

    def profile_batch(self, X, y, B=None):
        """The same for ``g`` windows of one length: ``(g, n, p)``, ``(g, n)``."""
        g, n, d = X.shape
        if B is None:
            B = self.B(X.reshape(-1, d)).reshape(g, n, -1)
        if self.nuisance is None:
            return B, y, 0
        N = _as_matrix(self.nuisance, X.reshape(-1, d)).reshape(g, n, -1)
        Q, _ = np.linalg.qr(N)
        Qt = np.swapaxes(Q, 1, 2)
        return (B - Q @ (Qt @ B), y - (Q @ (Qt @ y[..., None]))[..., 0], N.shape[2])

    def profile_all(self, Xs, ys, Bs=None):
        """:meth:`profile_batch` over ragged windows, grouped by length."""
        out = [None] * len(Xs)
        lengths = np.array([len(v) for v in ys])
        for n in np.unique(lengths):
            idx = np.flatnonzero(lengths == n)
            X = np.stack([Xs[i] for i in idx])
            y = np.stack([ys[i] for i in idx])
            B = None if Bs is None else np.stack([Bs[i] for i in idx])
            Bt, yt, q = self.profile_batch(X, y, B)
            for j, i in enumerate(idx):
                out[i] = (Bt[j], yt[j], q)
        return out


class LawClassifier:
    """``L`` laws per class; a window goes to the class whose laws explain it.

    A real class is rarely one law: a working day in winter and in summer, a
    gesture made early or late. So each class is itself a mixture of
    regimes, found with the same hard loop LR-DSR clusters with (fit a law per
    regime, reassign each window to the law that explains it best), and a
    window's class likelihood is the mixture over that class's laws. This is
    mixture discriminant analysis with laws for components. ``L`` spans a
    family:

    ==============  ======================================================
    ``L = 1``       one law per class: the linear (LDA) rule, the setting
                    of ``lrdsr.theory.classification``
    ``L = 2, 3..``  each class a mixture of regimes
    ``L = "all"``   every training window its own law: nearest neighbour in
                    law space, with each window's law smoothed by the basis
    ==============  ======================================================

    Parameters
    ----------
    basis : callable ``X_flat -> (N, p)``, optional
        The law's columns. Default: the fast backend's library. Any basis
        of ``lrdsr.core.kernel`` works, and is fitted on training inputs.
    nuisance : ``"intercept"``, callable or None
        Columns whose coefficient is free per window and profiled out of
        both the fit and the score. ``"intercept"`` (default) makes the
        window's level a nuisance, so the class is decided by *shape*.
    laws_per_class : int or ``"all"``
        ``L``, above. A class with fewer windows than ``L`` gets one law per
        window.
    law : ``"dense"`` or ``"symbolic"``
        ``dense``: ridge least squares on the whole basis. ``symbolic``
        (``L = 1`` only): the fast backend's forward selection with BIC per
        class, on the nuisance-profiled responses -- a *named* expression
        per class, searched in the backend's own library.
    sigma : ``"shared"`` or ``"per_class"``
        One noise level (the linear rule) or one per class (quadratic).
    priors : ``"empirical"`` or ``"uniform"``
    ridge : float
        Relative ridge on the normal equations (per law, and per window for
        ``L = "all"``).
    """

    def __init__(self, basis=None, feature_names=None, nuisance="intercept",
                 laws_per_class=1, law: str = "dense", sigma: str = "shared",
                 priors: str = "empirical", ridge: float = 1e-6,
                 max_terms: int = 5, max_iter: int = 10, random_state: int = 0):
        if law not in ("dense", "symbolic"):
            raise ValueError("law must be 'dense' or 'symbolic'")
        if law == "symbolic" and laws_per_class != 1:
            raise ValueError("law='symbolic' supports laws_per_class=1 only")
        if not (laws_per_class == "all" or int(laws_per_class) >= 1):
            raise ValueError("laws_per_class must be a positive int or 'all'")
        if sigma not in ("shared", "per_class"):
            raise ValueError("sigma must be 'shared' or 'per_class'")
        if priors not in ("empirical", "uniform"):
            raise ValueError("priors must be 'empirical' or 'uniform'")
        self.basis = basis
        self.feature_names = feature_names
        self.nuisance = nuisance
        self.laws_per_class = laws_per_class
        self.law = law
        self.sigma = sigma
        self.priors = priors
        self.ridge = ridge
        self.max_terms = max_terms
        self.max_iter = max_iter
        self.random_state = random_state

    # --------------------------------------------------- sufficient statistics
    @staticmethod
    def _stats(prof):
        """Per window: ``G = B~^T B~``, ``a = B~^T y~``, ``yy = |y~|^2``, dof."""
        G = np.stack([Bt.T @ Bt for Bt, _, _ in prof])
        a = np.stack([Bt.T @ yt for Bt, yt, _ in prof])
        yy = np.array([float(yt @ yt) for _, yt, _ in prof])
        dof = np.array([len(yt) - q for _, yt, q in prof], float)
        return G, a, yy, dof

    def _solve(self, G, a):
        p = G.shape[-1]
        lam = self.ridge * max(np.trace(G) / p, 1e-12)
        return np.linalg.solve(G + lam * np.eye(p), a)

    @staticmethod
    def _rss_stats(G, a, yy, coef):
        """``(W, J)`` residual sums of squares of every window under every law.

        At a fixed design every window has the same Gram, and the quadratic
        term is one row computed once instead of ``W * J`` of them.
        """
        if np.allclose(G, G[:1], rtol=1e-10, atol=1e-12):
            quad = np.einsum("jp,pq,jq->j", coef, G[0], coef)[None, :]
        else:
            quad = np.einsum("wpq,jq->wjp", G, coef, optimize=True)
            quad = np.einsum("wjp,jp->wj", quad, coef, optimize=True)
        return np.maximum(yy[:, None] - 2.0 * a @ coef.T + quad, 0.0)

    def _class_laws(self, G, a, yy):
        """The hard loop inside one class: ``(coef (L, p), weights (L,))``."""
        W = len(G)
        L = W if self.laws_per_class == "all" else min(int(self.laws_per_class), W)
        if L == 1:
            return self._solve(G.sum(0), a.sum(0))[None, :], np.ones(1)
        own = np.stack([self._solve(G[w], a[w]) for w in range(W)])
        if L == W:
            return own, np.full(W, 1.0 / W)
        # start: K-means on the windows' own laws in pooled-whitened
        # coordinates (mechanism space), then fit / reassign until stable
        from sklearn.cluster import KMeans
        lam, V = np.linalg.eigh(G.mean(0))
        Gh = V @ np.diag(np.sqrt(np.clip(lam, 0, None))) @ V.T
        lab = KMeans(L, n_init=10, random_state=self.random_state).fit_predict(own @ Gh)
        for _ in range(self.max_iter):
            ks = np.unique(lab)
            coef = np.stack([self._solve(G[lab == k].sum(0), a[lab == k].sum(0))
                             for k in ks])
            new = ks[np.argmin(self._rss_stats(G, a, yy, coef), axis=1)]
            if np.array_equal(new, lab):
                break
            lab = new
        ks, counts = np.unique(lab, return_counts=True)
        coef = np.stack([self._solve(G[lab == k].sum(0), a[lab == k].sum(0)) for k in ks])
        return coef, counts / counts.sum()

    # ------------------------------------------------------------------ fit
    def fit(self, X_seq, y_seq, labels):
        Xs, ys = as_windows(X_seq, y_seq)
        labels = np.asarray(labels)
        if len(labels) != len(Xs):
            raise ValueError("one label per window")
        self.classes_ = np.unique(labels)
        self.design_ = _Design(self.basis, self.feature_names, self.nuisance).fit(Xs)
        prof = self.design_.profile_all(Xs, ys)
        cls = np.searchsorted(self.classes_, labels)
        if self.law == "symbolic":
            self.models_ = []
            for c in range(len(self.classes_)):
                idx = np.flatnonzero(cls == c)
                self.models_.append(FastSymbolicRegressor(
                    feature_names=self.feature_names, max_terms=self.max_terms).fit(
                    np.vstack([Xs[i] for i in idx]),
                    np.concatenate([prof[i][1] for i in idx])))
            R = self._rss_symbolic(Xs, ys)
            own = R[np.arange(len(Xs)), cls]
            dof = np.array([len(yt) - q for _, yt, q in prof], float)
            self.coef_, self.law_class_ = None, np.arange(len(self.classes_))
            self.law_weight_ = np.ones(len(self.classes_))
        else:
            G, a, yy, dof = self._stats(prof)
            coefs, owner, weights = [], [], []
            for c in range(len(self.classes_)):
                idx = np.flatnonzero(cls == c)
                cf, wt = self._class_laws(G[idx], a[idx], yy[idx])
                coefs.append(cf)
                owner += [c] * len(cf)
                weights.append(wt)
            self.coef_ = np.vstack(coefs)
            self.law_class_ = np.asarray(owner)
            self.law_weight_ = np.concatenate(weights)
            R = self._rss_stats(G, a, yy, self.coef_)
            # a window's own law (L = "all") fits it perfectly in-sample, so
            # the noise level is read off each window's best law in ANOTHER
            # position when that is the case: the leave-one-out residual
            if self.laws_per_class == "all":
                np.fill_diagonal(R, np.inf)
            own = np.array([R[w, self.law_class_ == cls[w]].min()
                            for w in range(len(Xs))])
            own = np.where(np.isfinite(own), own, np.nan)
        K = len(self.classes_)
        if self.sigma == "shared":
            s2 = np.nansum(own) / max(dof[np.isfinite(own)].sum(), 1.0)
            self.sigma_ = np.full(K, np.sqrt(max(s2, 1e-24)))
        else:
            self.sigma_ = np.array([
                np.sqrt(max(np.nansum(own[cls == c])
                            / max(dof[(cls == c) & np.isfinite(own)].sum(), 1.0), 1e-24))
                for c in range(K)])
        counts = np.bincount(cls, minlength=K).astype(float)
        self.log_prior_ = (np.log(counts / counts.sum()) if self.priors == "empirical"
                           else np.full(K, -np.log(K)))
        return self

    # ---------------------------------------------------------------- score
    def _rss_symbolic(self, Xs, ys) -> np.ndarray:
        K = len(self.models_)
        out = np.empty((len(Xs), K))
        lengths = np.array([len(v) for v in ys])
        for n in np.unique(lengths):
            idx = np.flatnonzero(lengths == n)
            X = np.stack([Xs[i] for i in idx])
            y = np.stack([ys[i] for i in idx])
            flat = X.reshape(-1, X.shape[2])
            P = np.stack([m.predict(flat) for m in self.models_], axis=-1)
            Pt, yt, _ = self.design_.profile_batch(X, y, P.reshape(len(idx), n, K))
            out[idx] = np.sum((yt[..., None] - Pt) ** 2, axis=1)
        return out

    def law_rss(self, X_seq, y_seq, batch: int = 2048) -> np.ndarray:
        """``(W, J)``: each window's residual sum of squares under every law."""
        Xs, ys = as_windows(X_seq, y_seq)
        if self.law == "symbolic":
            return self._rss_symbolic(Xs, ys)
        out = []
        for s in range(0, len(Xs), batch):
            G, a, yy, _ = self._stats(self.design_.profile_all(Xs[s:s + batch],
                                                               ys[s:s + batch]))
            out.append(self._rss_stats(G, a, yy, self.coef_))
        return np.vstack(out)

    def window_loglik(self, X_seq, y_seq) -> np.ndarray:
        """``(W, K)`` Gaussian log-likelihood of each window under each class:
        the mixture over the class's laws."""
        Xs, ys = as_windows(X_seq, y_seq)
        R = self.law_rss(Xs, ys)
        q = 0 if self.design_.nuisance is None else _as_matrix(
            self.design_.nuisance, Xs[0][:1]).shape[1]
        n = np.array([len(yw) - q for yw in ys], float)[:, None]
        s = self.sigma_[self.law_class_][None, :]
        per_law = (np.log(self.law_weight_)[None, :] - 0.5 * R / s ** 2
                   - n * np.log(s) - 0.5 * n * np.log(2 * np.pi))
        out = np.empty((len(Xs), len(self.classes_)))
        for c in range(len(self.classes_)):
            out[:, c] = logsumexp(per_law[:, self.law_class_ == c], axis=1)
        return out

    def predict_proba(self, X_seq, y_seq) -> np.ndarray:
        J = self.window_loglik(X_seq, y_seq) + self.log_prior_[None, :]
        return np.exp(J - logsumexp(J, axis=1, keepdims=True))

    def predict(self, X_seq, y_seq) -> np.ndarray:
        return self.classes_[np.argmax(self.window_loglik(X_seq, y_seq)
                                       + self.log_prior_[None, :], axis=1)]

    @property
    def n_laws(self) -> int:
        return len(self.law_class_)

    def expressions(self) -> list[str]:
        """Each law: the selected expression (symbolic) or the basis
        coefficients above 1% of the largest (dense)."""
        if self.law == "symbolic":
            return [m.expression() for m in self.models_]
        names = getattr(self.basis, "names", None)
        out = []
        for b in self.coef_:
            nm = names or [f"b{j}" for j in range(len(b))]
            big = np.abs(b) >= 1e-2 * max(np.abs(b).max(), 1e-12)
            out.append(" ".join(f"{c:+.3g}*{t}" for c, t, k in zip(b, nm, big, strict=True)
                                if k).lstrip("+") or "0")
        return out


class MechanismFeatures:
    """Mechanism-space coordinates as a train/test feature map.

    ``lrdsr.core.mechanism_space.mechanism_features`` computes the statistic
    transductively over the windows it is given. Here the pooled Gram, the
    whitening and the pooled law are learned on training windows and applied
    unchanged to new ones, so the coordinates can feed any scikit-learn
    classifier without leaking the test set into the representation.

    ``mode="project"`` is the statistic of ``mechanism_space`` (``C_w^T r_w``,
    covariance ``sigma^2 C_w^T C_w``, the identity on average). ``"solve"``
    reads each window's whitened coefficients off by ridge least squares
    instead, which removes the design dependence -- the right choice when
    windows are sampled at different inputs or have different lengths, and
    the wrong one for a collinear library (``mechanism_space`` explains why).
    """

    def __init__(self, basis=None, feature_names=None, nuisance="intercept",
                 mode: str = "project", tol: float = 1e-6, solve_ridge: float = 1e-3,
                 subtract_pooled: bool = True):
        if mode not in ("project", "solve"):
            raise ValueError("mode must be 'project' or 'solve'")
        self.basis = basis
        self.feature_names = feature_names
        self.nuisance = nuisance
        self.mode = mode
        self.tol = tol
        self.solve_ridge = solve_ridge
        self.subtract_pooled = subtract_pooled

    def fit(self, X_seq, y_seq, labels=None):
        Xs, ys = as_windows(X_seq, y_seq)
        self.design_ = _Design(self.basis, self.feature_names, self.nuisance).fit(Xs)
        prof = self.design_.profile_all(Xs, ys)
        nbar = float(np.mean([len(yw) for yw in ys]))
        # the pooled Gram per nbar samples: a window of the average length has
        # C^T C = I on average, whatever its length distribution
        G = sum(Bt.T @ Bt for Bt, _, _ in prof) / len(prof)
        lam, V = np.linalg.eigh(G)
        keep = lam > self.tol * lam.max()
        self.M_ = V[:, keep] / np.sqrt(lam[keep])
        self.nbar_ = nbar
        if self.subtract_pooled:
            B = np.vstack([p[0] for p in prof])
            y = np.concatenate([p[1] for p in prof])
            self.beta_bar_ = np.linalg.lstsq(B, y, rcond=None)[0]
        else:
            self.beta_bar_ = np.zeros(G.shape[0])
        self.center_ = 0.0
        self.center_ = self._raw(prof).mean(axis=0)
        return self

    def _raw(self, prof):
        out = []
        for Bt, yt, _ in prof:
            C = Bt @ self.M_
            r = yt - Bt @ self.beta_bar_
            if self.mode == "project":
                out.append(C.T @ r * np.sqrt(self.nbar_ / max(len(r), 1)))
            else:
                A = C.T @ C
                A += self.solve_ridge * np.eye(A.shape[0]) * max(np.trace(A) / A.shape[0], 1e-12)
                out.append(np.linalg.solve(A, C.T @ r) * np.sqrt(self.nbar_))
        return np.asarray(out) - self.center_

    def transform(self, X_seq, y_seq) -> np.ndarray:
        Xs, ys = as_windows(X_seq, y_seq)
        return self._raw(self.design_.profile_all(Xs, ys))

    def fit_transform(self, X_seq, y_seq, labels=None) -> np.ndarray:
        return self.fit(X_seq, y_seq).transform(X_seq, y_seq)
