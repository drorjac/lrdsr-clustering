"""Soft LR-DSR: the same problem, solved by EM instead of hard reassignment.

``GroupedDCSR`` makes every decision hard: a window belongs to one law, and a
law is fitted to the windows that belong to it. That is the classification
EM of a mixture of regressions. This module is the ordinary EM of the same
model, over the same term library the fast backend searches:

.. code-block:: text

    y_w = Phi_w beta_k + e_w,   e ~ N(0, sigma_k^2)  or  t_nu(0, sigma_k)
    z_w ~ Categorical(pi)

    E-step   r_wk  ∝  pi_k p(y_w | Phi_w, beta_k, sigma_k) ^ (1/T)
    M-step   beta_k = (sum_w r_wk Phi_w^T U_w Phi_w)^-1 sum_w r_wk Phi_w^T U_w y_w
             sigma_k, nu_k, pi_k  by weighted maximum likelihood

``U_w`` is identity for Gaussian noise and the Student-t sample weights
otherwise (the ECM step). ``T`` is a temperature: ``T > 1`` flattens the
posteriors, and annealing it to 1 is deterministic annealing -- the standard
cure for EM committing early to a bad partition.

Three things it gives that the hard loop does not:

* **responsibilities** -- a window near the boundary says so, which is what the
  online clusterer and the plots use as uncertainty;
* **a learning curve** -- the observed-data log-likelihood is monotone under
  EM at ``T = 1``, so ``history`` is a real objective, not a changed-fraction;
* **a learned loss** -- with ``noise="student_t"`` each regime's tail weight
  ``nu_k`` is fitted, so the loss a window is scored with is learned per law.

What it gives up: the law is a dense combination of library terms rather
than a selected expression. :meth:`SoftResult.expressions` prints the
coefficients above a threshold, and ``GroupedDCSR`` remains the estimator for
a *named* law.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp

from .backends import FastSymbolicRegressor
from .mechanism_space import mechanism_init

NOISE_MODELS = ("gaussian", "student_t")


def library_terms(X_flat: np.ndarray, feature_names=None, include_trig=True,
                  include_log=True, include_interactions=True):
    """``(Phi, names)``: the fast backend's library, intercept first, with names."""
    X_flat = np.asarray(X_flat, dtype=float)
    if X_flat.ndim == 1:
        X_flat = X_flat[:, None]
    reg = FastSymbolicRegressor(feature_names=feature_names,
                                include_trig=include_trig, include_log=include_log,
                                include_interactions=include_interactions)
    terms = reg._make_library(X_flat)
    Phi = np.column_stack([np.ones(len(X_flat))] + [t.values for t in terms])
    return Phi, ["1"] + [t.name for t in terms]


@dataclass
class SoftResult:
    """What :class:`SoftLRDSR` returns.

    ``labels`` is the MAP partition; ``responsibilities`` the full posterior
    ``(W, K)``; ``coef`` one row of library coefficients per regime;
    ``sigma``, ``weights`` and (Student-t only) ``nu`` per regime;
    ``history`` one row per EM iteration with the log-likelihood.
    """
    labels: np.ndarray
    responsibilities: np.ndarray
    coef: np.ndarray
    sigma: np.ndarray
    weights: np.ndarray
    nu: np.ndarray | None
    term_names: list[str]
    history: pd.DataFrame
    loglik: float
    window_loglik: np.ndarray = field(repr=False)

    @property
    def entropy(self) -> np.ndarray:
        """Posterior entropy per window, in nats: 0 = certain."""
        r = np.clip(self.responsibilities, 1e-300, 1.0)
        return -np.sum(r * np.log(r), axis=1)

    def expressions(self, tol: float = 1e-2) -> list[str]:
        """Each regime's law, dropping coefficients below ``tol`` x the largest."""
        out = []
        for b in self.coef:
            big = np.abs(b) >= tol * max(np.abs(b).max(), 1e-12)
            parts = [f"{c:+.4g}*{n}" if n != "1" else f"{c:+.4g}"
                     for c, n, keep in zip(b, self.term_names, big, strict=True) if keep]
            out.append(" ".join(parts).lstrip("+") or "0")
        return out


class SoftLRDSR:
    """EM for a mixture of library regressions over windows.

    Parameters
    ----------
    n_clusters : int
    noise : ``"gaussian"`` or ``"student_t"``
        The per-regime noise model. ``student_t`` learns ``nu_k``: the loss is
        learned, not chosen.
    basis : callable ``X_flat -> (N, p)``, optional
        The regression columns. Default: the fast backend's own library
        (:func:`library_terms`), intercept included.
    init : ``"mechanism"``, ``"random"`` or an array of labels
        Where EM starts. ``"mechanism"`` is K-means in mechanism space, the
        same start ``GroupedDCSR`` uses.
    temperature : float
        Initial temperature ``T0``. Annealed geometrically to 1 over
        ``anneal_iters`` iterations; ``T0 = 1`` is plain EM.
    ridge : float
        Relative ridge on the weighted normal equations: the library is
        collinear by design.
    """

    def __init__(self, n_clusters: int, noise: str = "gaussian", basis=None,
                 feature_names: list[str] | None = None, init="mechanism",
                 max_iter: int = 200, tol: float = 1e-7, temperature: float = 1.0,
                 anneal_iters: int = 20, ridge: float = 1e-8, min_weight: float = 1e-3,
                 nu_bounds=(1.0, 200.0), random_state: int = 0):
        if noise not in NOISE_MODELS:
            raise ValueError(f"noise must be one of {NOISE_MODELS}")
        if temperature < 1.0:
            raise ValueError("temperature must be >= 1")
        self.n_clusters = n_clusters
        self.noise = noise
        self.basis = basis
        self.feature_names = feature_names
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.temperature = float(temperature)
        self.anneal_iters = anneal_iters
        self.ridge = ridge
        self.min_weight = min_weight
        self.nu_bounds = nu_bounds
        self.random_state = random_state

    # ---------------------------------------------------------------- design
    def _design(self, X_seq):
        W, n, d = X_seq.shape
        flat = X_seq.reshape(-1, d)
        if self.basis is None:
            Phi, names = library_terms(flat, feature_names=self.feature_names)
        else:
            Phi = np.asarray(self.basis(flat), dtype=float)
            Phi = Phi[:, None] if Phi.ndim == 1 else Phi
            names = [f"b{j}" for j in range(Phi.shape[1])]
        return Phi.reshape(W, n, -1), names

    def _initial_labels(self, X_seq, y_seq):
        K = self.n_clusters
        if isinstance(self.init, np.ndarray):
            labels = np.asarray(self.init, int)
            if labels.shape != (X_seq.shape[0],):
                raise ValueError("init array must have one label per window")
            return labels
        if self.init == "random":
            return np.random.default_rng(self.random_state).integers(0, K, X_seq.shape[0])
        if self.init == "mechanism":
            return mechanism_init(X_seq, y_seq, K, seed=self.random_state,
                                  feature_names=self.feature_names)
        raise ValueError("init must be 'mechanism', 'random' or an array")

    # ---------------------------------------------------------------- M-step
    def _m_step(self, Phi, y, R, U):
        """Weighted least squares per regime; ``U`` the per-sample t weights."""
        _, n, p = Phi.shape
        K = self.n_clusters
        coef = np.zeros((K, p))
        sigma = np.zeros(K)
        for k in range(K):
            wk = R[:, k][:, None] * (U[k] if U is not None else 1.0)   # (W, n)
            A = np.einsum("wn,wnp,wnq->pq", wk, Phi, Phi)
            b = np.einsum("wn,wnp,wn->p", wk, Phi, y)
            A += self.ridge * np.trace(A) / p * np.eye(p)
            coef[k] = np.linalg.solve(A, b)
            res = y - Phi @ coef[k]
            num = float(np.sum(wk * res ** 2))
            den = float(np.sum(R[:, k]) * n)
            sigma[k] = np.sqrt(max(num / max(den, 1e-12), 1e-12))
        return coef, sigma

    def _update_nu(self, res2_over_s2, R, nu_old):
        """1-D weighted ML for each regime's ``nu`` (the learned tail)."""
        from scipy.optimize import minimize_scalar

        nu = np.array(nu_old, dtype=float)
        lo, hi = np.log(self.nu_bounds[0]), np.log(self.nu_bounds[1])
        for k in range(self.n_clusters):
            q = res2_over_s2[k]                       # (W, n)
            w = R[:, k][:, None] * np.ones_like(q)

            def nll(t, q=q, w=w):
                v = np.exp(t)
                ll = (gammaln((v + 1) / 2) - gammaln(v / 2) - 0.5 * np.log(v * np.pi)
                      - 0.5 * (v + 1) * np.log1p(q / v))
                return -float(np.sum(w * ll)) / max(float(w.sum()), 1e-12)

            nu[k] = float(np.exp(minimize_scalar(nll, bounds=(lo, hi), method="bounded",
                                                 options={"xatol": 1e-3}).x))
        return nu

    # ---------------------------------------------------------------- E-step
    def _window_loglik(self, Phi, y, coef, sigma, nu):
        """``(W, K)`` log p(y_w | regime k), and the scaled squared residuals."""
        W, n, _ = Phi.shape
        L = np.zeros((W, self.n_clusters))
        q_all = []
        for k in range(self.n_clusters):
            q = (y - Phi @ coef[k]) ** 2 / sigma[k] ** 2
            q_all.append(q)
            if self.noise == "gaussian":
                L[:, k] = -0.5 * np.sum(q, axis=1) - n * np.log(sigma[k]) \
                          - 0.5 * n * np.log(2 * np.pi)
            else:
                v = nu[k]
                L[:, k] = (n * (gammaln((v + 1) / 2) - gammaln(v / 2)
                                - 0.5 * np.log(v * np.pi) - np.log(sigma[k]))
                           - 0.5 * (v + 1) * np.sum(np.log1p(q / v), axis=1))
        return L, q_all

    # ------------------------------------------------------------------- fit
    def fit(self, X_seq: np.ndarray, y_seq: np.ndarray,
            true_labels_for_eval: np.ndarray | None = None) -> SoftResult:
        """Run EM. ``true_labels_for_eval`` is logged per iteration, never used."""
        X_seq = np.asarray(X_seq, dtype=float)
        y = np.asarray(y_seq, dtype=float)
        if X_seq.ndim == 2:
            X_seq = X_seq[:, :, None]
        W = X_seq.shape[0]
        K = self.n_clusters
        Phi, names = self._design(X_seq)

        labels0 = self._initial_labels(X_seq, y)
        R = np.full((W, K), 1e-3)
        R[np.arange(W), labels0] = 1.0
        R /= R.sum(axis=1, keepdims=True)
        U = None
        nu = np.full(K, 30.0) if self.noise == "student_t" else None

        history, prev = [], -np.inf
        coef, sigma = self._m_step(Phi, y, R, U)
        pi = np.clip(R.mean(axis=0), self.min_weight, None)
        pi /= pi.sum()
        for it in range(self.max_iter):
            T = (self.temperature ** max(0.0, 1.0 - it / max(self.anneal_iters, 1))
                 if self.temperature > 1.0 else 1.0)
            L, q_all = self._window_loglik(Phi, y, coef, sigma, nu)
            joint = np.log(pi)[None, :] + L
            loglik = float(np.sum(logsumexp(joint, axis=1)))
            logR = joint / T
            R = np.exp(logR - logsumexp(logR, axis=1, keepdims=True))

            row = {"iteration": it, "loglik": loglik, "temperature": T,
                   "mean_entropy": float(np.mean(-np.sum(
                       R * np.log(np.clip(R, 1e-300, 1)), axis=1)))}
            for k in range(K):
                row[f"sigma_{k}"] = float(sigma[k])
                if nu is not None:
                    row[f"nu_{k}"] = float(nu[k])
            if true_labels_for_eval is not None:
                from .evaluation import clustering_metrics
                row.update(clustering_metrics(true_labels_for_eval, R.argmax(axis=1)))
            history.append(row)

            # M-step
            if self.noise == "student_t":
                U = [(nu[k] + 1.0) / (nu[k] + q_all[k]) for k in range(K)]
            coef, sigma = self._m_step(Phi, y, R, U)
            pi = np.clip(R.mean(axis=0), self.min_weight, None)
            pi /= pi.sum()
            if self.noise == "student_t":
                _, q_new = self._window_loglik(Phi, y, coef, sigma, nu)
                nu = self._update_nu(q_new, R, nu)

            if T == 1.0 and np.isfinite(prev) and abs(loglik - prev) <= self.tol * abs(prev):
                break
            prev = loglik if T == 1.0 else -np.inf

        L, _ = self._window_loglik(Phi, y, coef, sigma, nu)
        joint = np.log(pi)[None, :] + L
        R = np.exp(joint - logsumexp(joint, axis=1, keepdims=True))
        return SoftResult(
            labels=R.argmax(axis=1), responsibilities=R, coef=coef, sigma=sigma,
            weights=pi, nu=nu, term_names=names, history=pd.DataFrame(history),
            loglik=float(np.sum(logsumexp(joint, axis=1))), window_loglik=L,
        )


def predict_soft(result: SoftResult, X_seq: np.ndarray, y_seq: np.ndarray,
                 feature_names=None, basis=None) -> np.ndarray:
    """Posterior responsibilities of new windows under a fitted soft model."""
    model = SoftLRDSR(result.coef.shape[0],
                      noise="student_t" if result.nu is not None else "gaussian",
                      basis=basis, feature_names=feature_names)
    X_seq = np.asarray(X_seq, dtype=float)
    if X_seq.ndim == 2:
        X_seq = X_seq[:, :, None]
    Phi, _ = model._design(X_seq)
    L, _ = model._window_loglik(Phi, np.asarray(y_seq, float), result.coef,
                                result.sigma, result.nu)
    joint = np.log(result.weights)[None, :] + L
    return np.exp(joint - logsumexp(joint, axis=1, keepdims=True))
