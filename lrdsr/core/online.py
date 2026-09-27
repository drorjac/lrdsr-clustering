"""Real-time LR-DSR: assign each window as it arrives, and keep the laws current.

The batch estimators see every window before deciding anything. A stream
does not wait: a window arrives, it must be assigned now, and the laws must
absorb it without refitting from scratch. Two real-time estimators live here,
at two time scales.

:class:`OnlineLRDSR` -- **window by window.**
    Each regime keeps the sufficient statistics of a library regression,
    ``A_k = sum Phi^T Phi``, ``b_k = sum Phi^T y``, ``c_k = sum y^T y``, so the
    law and its noise level are one small solve away after every window
    (recursive least squares). A new window is scored under every law by its
    Gaussian log-likelihood, assigned to the MAP regime, and folded into that
    regime's statistics with an optional forgetting factor ``lambda`` for
    laws that drift. A window that **no** law explains -- its chi-square
    residual statistic is improbable under every regime -- is held in a
    novelty buffer; once enough mutually consistent ones accumulate, a new
    regime is born from them. The number of regimes is therefore not fixed.

:class:`CusumSegmenter` -- **sample by sample.**
    With the laws known (or learned), a single stream that switches regime
    at unknown times is segmented by CUSUM on the per-sample log-likelihood
    ratio. The theory is classical and checkable: with per-sample
    Kullback-Leibler divergence ``KL = rho / 2`` between two Gaussian laws of
    separation ``rho``, the detection delay after a switch is
    ``~ h / KL = 2h / rho`` samples, and false alarms arrive no more often
    than once every ``~ e^h`` samples (``lrdsr.theory.sequential``).

Neither reads a label. ``OnlineLRDSR`` can be warm-started from any batch fit
(:meth:`OnlineLRDSR.warm_start`), which is the natural deployment: learn the
laws on history, then run.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import chi2

from .soft import SoftLRDSR, library_terms


@dataclass
class Assignment:
    """What :meth:`OnlineLRDSR.partial_fit` decides about one window."""
    t: int
    label: int                  # -1 while the window sits in the novelty buffer
    posterior: np.ndarray       # over the regimes that existed at the time
    loglik: np.ndarray          # per regime
    p_value: float              # chi-square fit of the best regime
    novel: bool                 # no regime explains it
    spawned: int | None         # index of a regime born at this step


class OnlineLRDSR:
    """Streaming clustering by law, with regime birth.

    Parameters
    ----------
    basis : callable ``X_flat -> (N, p)``, optional
        Regression columns; default the fast backend's library.
    feature_names : list[str], optional
    forgetting : float in (0, 1]
        ``lambda``: each regime's statistics are multiplied by it before a
        window is added, so its memory is ``~ 1/(1-lambda)`` windows. ``1``
        never forgets (the laws are stationary).
    novelty_alpha : float
        A window is novel when its best regime's chi-square p-value is below
        this. ``0`` disables regime birth.
    novelty_patience : int
        How many novel windows must be buffered before a regime is born.
    max_clusters : int
        No more regimes than this, however much is novel.
    ridge : float
        Relative ridge on each regime's normal equations.
    prior_count : float
        Pseudo-count added to every regime's weight, so a new regime is not
        dismissed for being new.
    nuisance : ``"intercept"``, callable or None
        Columns whose coefficient is free per window, profiled out of each
        window's design and response before it is scored or absorbed (the
        ``LawClassifier`` convention). ``"intercept"`` makes a window's level
        a nuisance -- a day of traffic whose volume is high overall is not a
        new regime. The chi-square degrees of freedom drop accordingly.

    Windows may have different lengths (a day with missing hours): every
    statistic is per window, and a data-dependent ``basis`` (a
    ``NystromBasis``) is fitted on the warm-start inputs.
    """

    def __init__(self, basis=None, feature_names: list[str] | None = None,
                 forgetting: float = 1.0, novelty_alpha: float = 1e-3,
                 novelty_patience: int = 6, max_clusters: int = 8,
                 ridge: float = 1e-6, prior_count: float = 2.0, nuisance=None):
        if not 0.0 < forgetting <= 1.0:
            raise ValueError("forgetting must be in (0, 1]")
        self.basis = basis
        self.feature_names = feature_names
        self.forgetting = float(forgetting)
        self.novelty_alpha = float(novelty_alpha)
        self.novelty_patience = int(novelty_patience)
        self.max_clusters = int(max_clusters)
        self.ridge = ridge
        self.prior_count = prior_count
        self.nuisance = nuisance
        self.A_: list[np.ndarray] = []
        self.b_: list[np.ndarray] = []
        self.c_: list[float] = []
        self.m_: list[float] = []            # effective number of samples
        self.count_: list[float] = []        # effective number of windows
        self.buffer_: list[tuple[int, np.ndarray, np.ndarray]] = []
        self.t_ = 0
        self.log_: list[dict] = []

    # --------------------------------------------------------------- design
    def _phi(self, X_w: np.ndarray) -> np.ndarray:
        X_w = np.asarray(X_w, dtype=float)
        if X_w.ndim == 1:
            X_w = X_w[:, None]
        if self.basis is None:
            return library_terms(X_w, feature_names=self.feature_names)[0]
        P = np.asarray(self.basis(X_w), dtype=float)
        return P[:, None] if P.ndim == 1 else P

    def _design(self, X_w, y_w):
        """``(Phi~, y~, dof)``: the window's design and response with the
        nuisance profiled out, and the samples left after profiling."""
        Phi = self._phi(X_w)
        y = np.asarray(y_w, dtype=float).ravel()
        if self.nuisance is None:
            return Phi, y, len(y)
        from .classify import _as_matrix, intercept
        N = _as_matrix(intercept if self.nuisance == "intercept" else self.nuisance,
                       np.asarray(X_w, float).reshape(len(y), -1))
        Qn, _ = np.linalg.qr(N)
        return (Phi - Qn @ (Qn.T @ Phi), y - Qn @ (Qn.T @ y),
                max(len(y) - N.shape[1], 1))

    @property
    def n_clusters(self) -> int:
        return len(self.A_)

    def _coef(self, k: int) -> np.ndarray:
        A = self.A_[k]
        p = A.shape[0]
        return np.linalg.solve(A + self.ridge * max(np.trace(A) / p, 1e-12) * np.eye(p),
                               self.b_[k])

    def _sigma(self, k: int) -> float:
        beta = self._coef(k)
        rss = self.c_[k] - 2 * beta @ self.b_[k] + beta @ self.A_[k] @ beta
        dof = max(self.m_[k] - self.A_[k].shape[0], 1.0)
        return float(np.sqrt(max(rss / dof, 1e-12)))

    @property
    def coef_(self) -> np.ndarray:
        return np.vstack([self._coef(k) for k in range(self.n_clusters)])

    @property
    def sigma_(self) -> np.ndarray:
        return np.array([self._sigma(k) for k in range(self.n_clusters)])

    @property
    def weights_(self) -> np.ndarray:
        c = np.asarray(self.count_) + self.prior_count
        return c / c.sum()

    # ----------------------------------------------------------- bookkeeping
    def _add(self, k: int, Phi: np.ndarray, y: np.ndarray, weight: float = 1.0,
             dof: int | None = None):
        lam = self.forgetting
        self.A_[k] = lam * self.A_[k] + weight * Phi.T @ Phi
        self.b_[k] = lam * self.b_[k] + weight * Phi.T @ y
        self.c_[k] = lam * self.c_[k] + weight * float(y @ y)
        self.m_[k] = lam * self.m_[k] + weight * (len(y) if dof is None else dof)
        self.count_[k] = lam * self.count_[k] + weight

    def _new_regime(self, p: int):
        self.A_.append(np.zeros((p, p)))
        self.b_.append(np.zeros(p))
        self.c_.append(0.0)
        self.m_.append(0.0)
        self.count_.append(0.0)
        return self.n_clusters - 1

    # ------------------------------------------------------------ warm start
    def warm_start(self, X_seq: np.ndarray, y_seq: np.ndarray,
                   labels: np.ndarray | None = None, n_clusters: int | None = None,
                   random_state: int = 0) -> OnlineLRDSR:
        """Initialise the regimes from a batch of history.

        With ``labels`` (e.g. a ``GroupedDCSR`` or ``SoftLRDSR`` partition) the
        statistics are accumulated per label; without, a :class:`SoftLRDSR`
        with ``n_clusters`` regimes is fitted first and its responsibilities
        weight each window. Either way no true label is involved.
        """
        from .kernel import _windows

        Xs, ys = _windows(X_seq, y_seq)
        if self.basis is not None and hasattr(self.basis, "fit") and \
                not getattr(self.basis, "fitted", True):
            self.basis.fit(np.vstack(Xs))
        equal = len({len(v) for v in ys}) == 1
        if labels is None:
            if n_clusters is None:
                raise ValueError("pass labels or n_clusters")
            if equal and self.nuisance is None:
                # the original route, unchanged: soft EM on equal-length windows
                res = SoftLRDSR(n_clusters, basis=self.basis,
                                feature_names=self.feature_names,
                                random_state=random_state).fit(np.stack(Xs), np.stack(ys))
                R = res.responsibilities
            else:
                # ragged windows or a nuisance: each window's own whitened law
                # (mechanism features, solve mode), K-means, hard statistics
                from sklearn.cluster import KMeans

                from .classify import MechanismFeatures
                S = MechanismFeatures(basis=self.basis, feature_names=self.feature_names,
                                      nuisance=self.nuisance, mode="solve").fit_transform(Xs, ys)
                lab = KMeans(n_clusters, n_init=30, random_state=random_state).fit_predict(S)
                R = np.eye(n_clusters)[lab]
        else:
            labels = np.asarray(labels, int)
            R = np.eye(labels.max() + 1)[labels]
        p = self._phi(Xs[0]).shape[1]
        for _ in range(R.shape[1]):
            self._new_regime(p)
        lam, self.forgetting = self.forgetting, 1.0      # history is not forgotten
        for w in range(len(Xs)):
            Phi, y, dof = self._design(Xs[w], ys[w])
            for k in range(R.shape[1]):
                if R[w, k] > 1e-6:
                    self._add(k, Phi, y, R[w, k], dof=dof)
        self.forgetting = lam
        return self

    # ----------------------------------------------------------------- score
    def score_window(self, X_w, y_w) -> tuple[np.ndarray, np.ndarray]:
        """``(loglik, chi2 statistic)`` of one window under every regime."""
        Phi, y, n = self._design(X_w, y_w)
        L, Q = np.empty(self.n_clusters), np.empty(self.n_clusters)
        for k in range(self.n_clusters):
            s = self._sigma(k)
            q = float(np.sum((y - Phi @ self._coef(k)) ** 2)) / s ** 2
            Q[k] = q
            L[k] = -0.5 * q - n * np.log(s) - 0.5 * n * np.log(2 * np.pi)
        return L, Q

    def predict_proba(self, X_w, y_w) -> np.ndarray:
        L, _ = self.score_window(X_w, y_w)
        joint = np.log(self.weights_) + L
        return np.exp(joint - logsumexp(joint))

    # ----------------------------------------------------------- partial fit
    def partial_fit(self, X_w: np.ndarray, y_w: np.ndarray, update: bool = True) -> Assignment:
        """Assign one window now; fold it into its law unless it is novel."""
        if self.n_clusters == 0:
            raise RuntimeError("no regimes yet: call warm_start first")
        Phi, y, n = self._design(X_w, y_w)
        L, Q = self.score_window(X_w, y_w)
        joint = np.log(self.weights_) + L
        post = np.exp(joint - logsumexp(joint))
        k = int(np.argmax(post))
        p_val = float(chi2.sf(Q[k], df=n))
        novel = self.novelty_alpha > 0 and p_val < self.novelty_alpha
        spawned = None
        label = k
        if novel:
            label = -1
            self.buffer_.append((self.t_, Phi, y, n))
            spawned = self._maybe_spawn(n)
            if spawned is not None:
                label = spawned
        elif update:
            self._add(k, Phi, y, dof=n)
        a = Assignment(self.t_, label, post, L, p_val, bool(novel), spawned)
        self.log_.append({"t": self.t_, "label": label, "map": k, "p_value": p_val,
                          "novel": bool(novel), "spawned": spawned,
                          "n_clusters": self.n_clusters,
                          "max_posterior": float(post.max()),
                          "entropy": float(-np.sum(post * np.log(np.clip(post, 1e-300, 1))))})
        self.t_ += 1
        return a

    def _maybe_spawn(self, n: int) -> int | None:
        """Birth a regime from the buffer if it is large and self-consistent.

        One law is fitted to every buffered window together; the regime is
        born only if that law explains the windows at the noise level the
        existing regimes measure (each window's chi-square p-value under it
        clears ``novelty_alpha`` for 80% of them), so a buffer of unrelated
        junk does not become a regime. Otherwise the
        oldest buffered window is dropped and the buffer slides.
        """
        if len(self.buffer_) < self.novelty_patience:
            return None
        if self.n_clusters >= self.max_clusters:
            self.buffer_.pop(0)
            return None
        p = self.buffer_[0][1].shape[1]
        A = sum(Phi.T @ Phi for _, Phi, _, _ in self.buffer_)
        b = sum(Phi.T @ y for _, Phi, y, _ in self.buffer_)
        beta = np.linalg.solve(A + self.ridge * max(np.trace(A) / p, 1e-12) * np.eye(p), b)
        rss = [float(np.sum((y - Phi @ beta) ** 2)) for _, Phi, y, _ in self.buffer_]
        dofs = [d for _, _, _, d in self.buffer_]
        # Judge the candidate law against the noise level the EXISTING regimes
        # measure, not against the buffer's own residual spread: estimated
        # from the buffer, the noise absorbs whatever the joint law fails to
        # explain, and any mixture of ordinary windows looks "consistent".
        w = np.asarray(self.count_) + 1e-12
        s2 = float(np.sum(w * self.sigma_ ** 2) / np.sum(w))
        # each buffered window against its OWN degrees of freedom (windows
        # may differ in length); identical to the old test when they do not
        ok = np.mean([chi2.sf(r / s2, df=d) >= self.novelty_alpha
                      for r, d in zip(rss, dofs, strict=True)])
        if ok < 0.8:
            self.buffer_.pop(0)
            return None
        k = self._new_regime(p)
        for _, Phi, y, d in self.buffer_:
            self._add(k, Phi, y, dof=d)
        self.buffer_.clear()
        return k

    def fit_stream(self, X_seq: np.ndarray, y_seq: np.ndarray, update: bool = True,
                   true_labels_for_eval: np.ndarray | None = None) -> pd.DataFrame:
        """Run :meth:`partial_fit` over windows in order; return the timeline.

        ``true_labels_for_eval`` is only copied into the returned frame.
        """
        from .kernel import _windows

        Xs, ys = _windows(X_seq, y_seq)
        start = len(self.log_)
        for w in range(len(Xs)):
            self.partial_fit(Xs[w], ys[w], update=update)
        df = pd.DataFrame(self.log_[start:])
        if true_labels_for_eval is not None:
            df["truth"] = np.asarray(true_labels_for_eval)
        return df


# ==========================================================================
# sample-level: CUSUM segmentation between known laws
# ==========================================================================
class CusumSegmenter:
    """Segment one stream sample by sample into regimes with known laws.

    ``laws`` is a list of callables ``x -> f_k(x)`` and ``sigma`` the noise
    level (scalar or one per law). The segmenter holds a current regime ``j``
    and, for every alternative ``k``, the CUSUM statistic

    .. code-block:: text

        S_k(t) = max(0, S_k(t-1) + log p_k(y_t | x_t) - log p_j(y_t | x_t))

    and switches to ``argmax_k S_k`` the moment it exceeds ``h``. All
    statistics restart at a switch.

    ``h`` trades delay for false alarms: delay ``~ h / KL`` samples after a
    true switch, false alarms at most every ``~ e^h`` samples in a stable
    regime.
    """

    def __init__(self, laws, sigma, threshold: float = 5.0, start: int = 0):
        self.laws = list(laws)
        K = len(self.laws)
        self.sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (K,)).copy()
        self.threshold = float(threshold)
        self.start = int(start)

    def run(self, x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """Segment ``(x_t, y_t)``; one row per sample with state and alarms."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        T = len(y)
        K = len(self.laws)
        mu = np.vstack([np.asarray(f(x), dtype=float).reshape(-1) for f in self.laws])
        ll = -0.5 * ((y[None, :] - mu) / self.sigma[:, None]) ** 2 - np.log(self.sigma)[:, None]
        state = np.empty(T, dtype=int)
        alarm = np.zeros(T, dtype=bool)
        stat = np.empty(T)
        S = np.zeros(K)
        j = self.start
        for t in range(T):
            S = np.maximum(0.0, S + ll[:, t] - ll[j, t])
            S[j] = 0.0
            k = int(np.argmax(S))
            if S[k] > self.threshold:
                j = k
                alarm[t] = True
                S[:] = 0.0
            state[t] = j
            stat[t] = S.max()
        return pd.DataFrame({"t": np.arange(T), "state": state, "alarm": alarm,
                             "cusum": stat})
