"""Estimation when every regime is a DIFFERENT function family.

The zoo's pairs are mostly one family at two parameter values. Here the
regimes are different functions -- a sine against an exponential against a
bump -- so a method that assumes one model form for every cluster is at a
disadvantage by construction, and a symbolic search per cluster should not be.

Pipelines compared (same SR engine and library everywhere, ``K`` given):

* ``sr_pooled``      SR on all windows pooled: one function.
* ``poly_mix``       one ASSUMED model for every cluster: a mixture of cubic
                     regressions (hard EM, started from per-window cubic fits).
* ``summary_sr``     cluster window summaries (K-means), then SR per cluster.
* ``sr_per_window``  SR per window, cluster the fitted curves, SR per cluster.
* ``mech_sr``        the model's coordinates only: K-means in mechanism space,
                     then SR per cluster (no loop).
* ``lrdsr``          the model: coordinates + the fit/reassign loop,
                     ``alpha_geom = 0`` as in the zoo block.
* ``soft_em``        the model's soft-EM variant (``SoftLRDSR``), its
                     partition then SR per cluster.
* ``lrdsr_sr``       the model's partition, then a larger SR search per
                     cluster (``max_terms = 8``) for the final laws.

Declared before the run (2026-09-29):
  F1  on the new family problems lrdsr has lower mean assignment error than
      every non-oracle pipeline except possibly mech_sr, at rho <= 0.25.
  F2  poly_mix has higher mean law error than lrdsr on the family problems
      (a cubic cannot represent exp / bump / tanh laws).
  F3  lrdsr_sr has law error within 0.02 of lrdsr (the partition, not the
      search size, limits the law).
  F4  summary_sr is the worst of the partition pipelines on the zoo, where
      laws share their marginal moments by design.

Writes ``results/srbaseline/families_*.csv``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from experiments.common.fitting import fit_lrdsr, window_features
from experiments.problems import zoo
from experiments.problems.zoo import Problem, _u, _x
from experiments.srbaseline.run import (
    GRID_N, N_WINDOWS, REPORT_SEEDS, RESULTS, WINDOW_LEN, _curves, _law_error, _sr,
)
from lrdsr.core.backends import FastSymbolicRegressor
from lrdsr.core.evaluation import aligned_accuracy
from lrdsr.core.mechanism_space import mechanism_init
from lrdsr.core.soft import SoftLRDSR

RHO_GRID = (0.1, 0.25, 1.0)

FAMILIES: tuple[Problem, ...] = (
    Problem("sine_vs_exp",
            (lambda X: np.sin(2 * _x(X)), lambda X: 0.3 * np.exp(0.8 * _x(X)) - 0.6),
            ("sin(2x)", "0.3*exp(0.8x) - 0.6"), _u(-2.0, 2.0), 1, (-2.0, 2.0), False,
            "oscillation against growth; exp is not a library term"),
    Problem("bump_vs_parabola",
            (lambda X: np.exp(-_x(X) ** 2), lambda X: 1.0 - 0.35 * _x(X) ** 2),
            ("exp(-x^2)", "1 - 0.35x^2"), _u(-2.0, 2.0), 1, (-2.0, 2.0), False,
            "two symmetric humps, a Gaussian and a parabola"),
    Problem("log_vs_sqrt",
            (lambda X: np.log1p(_x(X)), lambda X: 0.75 * np.sqrt(_x(X))),
            ("log(1+x)", "0.75*sqrt(x)"), _u(0.0, 4.0), 1, (0.0, 4.0), False,
            "two concave growths that nearly coincide"),
    Problem("tanh_vs_linear_vs_cubic",
            (lambda X: np.tanh(2 * _x(X)), lambda X: 0.6 * _x(X),
             lambda X: 0.15 * _x(X) ** 3),
            ("tanh(2x)", "0.6x", "0.15x^3"), _u(-2.0, 2.0), 1, (-2.0, 2.0), False,
            "K = 3, three odd laws of three families"),
    Problem("five_families",
            (lambda X: _x(X), lambda X: _x(X) ** 2 - 1.0,
             lambda X: 1.2 * np.sin(2.5 * _x(X)), lambda X: 2.0 * np.exp(-_x(X) ** 2) - 0.5,
             lambda X: np.abs(_x(X)) - 0.8),
            ("x", "x^2 - 1", "1.2 sin(2.5x)", "2exp(-x^2) - 0.5", "|x| - 0.8"),
            _u(-2.0, 2.0), 1, (-2.0, 2.0), False,
            "K = 5: line, parabola, sine, bump, kink"),
    Problem("two_input_families",
            (lambda X: X[..., 0] * np.sin(X[..., 1]), lambda X: X[..., 0] + np.cos(X[..., 1]),
             lambda X: 0.4 * X[..., 0] ** 2 - 0.3 * X[..., 1]),
            ("x1*sin(x2)", "x1 + cos(x2)", "0.4x1^2 - 0.3x2"), _u(-2.0, 2.0, d=2), 2,
            (-2.0, 2.0), False, "K = 3, two inputs, three forms"),
)


def _cubic(X):
    x = X.reshape(-1, X.shape[-1])
    cols = [np.ones(len(x))]
    for j in range(x.shape[1]):
        cols += [x[:, j], x[:, j] ** 2, x[:, j] ** 3]
    return np.column_stack(cols)


def poly_mix(X, y, K, seed, iters=30):
    """Mixture of cubic regressions, hard EM: ONE assumed model form."""
    W = len(y)
    B = np.stack([_cubic(X[w]) for w in range(W)])
    own = np.stack([np.linalg.lstsq(B[w], y[w], rcond=None)[0] for w in range(W)])
    lab = KMeans(K, n_init=10, random_state=seed).fit_predict(
        StandardScaler().fit_transform(own))
    coef = None
    for _ in range(iters):
        coef = []
        for k in range(K):
            idx = np.where(lab == k)[0]
            if len(idx) == 0:
                idx = np.arange(W)
            coef.append(np.linalg.lstsq(B[idx].reshape(-1, B.shape[-1]),
                                        y[idx].reshape(-1), rcond=None)[0])
        rss = np.stack([((y - B @ c) ** 2).sum(1) for c in coef], 1)
        new = rss.argmin(1)
        if np.array_equal(new, lab):
            break
        lab = new
    return lab, [lambda Xg, c=c: _cubic(Xg) @ c for c in coef]


def _sr_per_cluster(X, y, lab, K, names, **kw):
    d = X.shape[-1]
    out = []
    for k in range(K):
        idx = np.where(lab == k)[0]
        if len(idx) == 0:
            idx = np.arange(len(y))
        m = FastSymbolicRegressor(feature_names=names, **{"max_terms": 5, **kw})
        out.append(m.fit(X[idx].reshape(-1, d), y[idx].reshape(-1)).predict)
    return out


def one(problem, rho, seed, suite) -> dict:
    sigma = zoo.sigma_for_rho(problem, rho)
    X, y, z = zoo.make_windows(problem, sigma, N_WINDOWS, WINDOW_LEN, seed)
    names, K, d = problem.feature_names, problem.K, problem.d
    Xg = zoo.eval_grid(problem, GRID_N)
    row = {"suite": suite, "problem": problem.name, "K": K, "rho": rho, "seed": seed}

    def score(tag, lab, laws, t0):
        row[f"{tag}_error"] = 1.0 - aligned_accuracy(z, lab)
        row[f"{tag}_law_error"] = _law_error(problem, z, lab, laws)
        row[f"{tag}_secs"] = time.time() - t0

    t = time.time()
    score("oracle", zoo.oracle_labels(problem, X, y), list(problem.laws), t)
    t = time.time()
    pooled = _sr(X.reshape(-1, d), y.reshape(-1), names)
    score("sr_pooled", np.zeros(len(y), int), [pooled.predict], t)
    t = time.time()
    lab, laws = poly_mix(X, y, K, seed)
    score("poly_mix", lab, laws, t)
    t = time.time()
    Zf, fnames = window_features(X[:, :, :1], y)
    lab = KMeans(K, n_init=10, random_state=seed).fit_predict(
        StandardScaler().fit_transform(Zf))
    score("summary_sr", lab, _sr_per_cluster(X, y, lab, K, names), t)
    t = time.time()
    C = _curves([_sr(X[w], y[w], names) for w in range(len(y))], Xg)
    lab = KMeans(K, n_init=10, random_state=seed).fit_predict(C)
    score("sr_per_window", lab, _sr_per_cluster(X, y, lab, K, names), t)
    t = time.time()
    lab = mechanism_init(X, y, K, seed=seed)
    score("mech_sr", lab, _sr_per_cluster(X, y, lab, K, names), t)
    t = time.time()
    res = fit_lrdsr(Zf, fnames, X, y, names, K, seed, alpha_geom=0.0)
    lab = np.asarray(res.labels)
    score("lrdsr", lab, [m.predict for m in res.models], t)
    t = time.time()
    score("lrdsr_sr", lab, _sr_per_cluster(X, y, lab, K, names, max_terms=8), t)
    t = time.time()
    soft = SoftLRDSR(K, feature_names=names, random_state=seed).fit(X, y)
    lab = np.asarray(soft.labels)
    score("soft_em", lab, _sr_per_cluster(X, y, lab, K, names), t)
    return row


TAGS = ["oracle", "soft_em", "lrdsr", "lrdsr_sr", "mech_sr", "sr_per_window", "summary_sr",
        "poly_mix", "sr_pooled"]


def run() -> pd.DataFrame:
    rows = []
    for suite, probs in (("families", FAMILIES), ("zoo", zoo.PROBLEMS)):
        for p in probs:
            for rho in RHO_GRID:
                for s in REPORT_SEEDS:
                    rows.append(one(p, rho, s, suite))
            print(f"  {suite:9s} {p.name:24s} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "families_per_seed.csv", index=False)
    for what in ("error", "law_error"):
        t = df.groupby(["suite", "rho"])[[f"{g}_{what}" for g in TAGS]].mean()
        t.columns = TAGS
        t.to_csv(RESULTS / f"families_{what}.csv")
        print(f"\n{what} (mean over problems x 3 seeds)\n{t.round(3).to_string()}")
    fam = df[df.suite == "families"]
    for what in ("error", "law_error"):
        t = fam.groupby("problem")[[f"{g}_{what}" for g in TAGS]].mean()
        t.columns = TAGS
        print(f"\nfamilies by problem, {what}\n{t.round(3).to_string()}")
    g = df.groupby(["suite", "problem", "rho"])
    for other in TAGS[3:-1]:
        m = g[[f"lrdsr_error", f"{other}_error"]].mean()
        w = (m.lrdsr_error <= m[f"{other}_error"] + 1e-12).groupby(level=0).agg(["sum", "size"])
        print(f"lrdsr <= {other:14s} assignment cells:",
              {s: f"{r['sum']}/{r['size']}" for s, r in w.iterrows()})
    return df


if __name__ == "__main__":
    run()
