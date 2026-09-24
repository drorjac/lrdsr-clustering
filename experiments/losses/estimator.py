"""Does the estimator inherit the loss's efficiency?

V8 prices a loss for an oracle that knows both laws. The estimator does not
know them: it fits the laws on its own partition, scores every window by a
loss, and the loss therefore enters twice -- in the decision, and through the
residual scale and the fit it is computed against. This block asks whether
the ordering V8 predicts survives that.

The pair battery of ``experiments.estimator.benchmark`` (identical input
marginals, so only the law separates the regimes), with the Gaussian noise
replaced by four noise laws, all scaled to the same **variance** so that
``rho = E[gap^2] / Var(e)`` means the same thing in every column:

* ``gaussian``, ``laplace``, ``student_t3`` and ``contaminated_10``
  (10% of samples with 10x the standard deviation).

Arms, all initialised from the same mechanism-space partition, all with
``alpha_geom = 0`` (the geometry carries nothing here by construction):

* ``GroupedDCSR`` with ``loss`` in huber (the default), squared, cauchy,
  tukey, student_t (nu = 4) and ``learned``;
* ``SoftLRDSR`` with Gaussian and with Student-t noise (the latter learns
  one ``nu`` per regime);
* ``mechanism_kmeans``, the starting partition itself;
* two oracles that know both laws: ``oracle_squared`` (the smaller residual
  sum, the rule of V1-V7) and ``oracle_lrt`` (the likelihood ratio under the
  true noise law -- the actual Bayes rule, which the squared oracle is not
  once the noise is not Gaussian).

Writes ``loss_estimator.csv`` (per seed) and ``loss_estimator_summary.csv``.
Seeds {11, 23, 42}; nothing is tuned.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from experiments.common.fitting import window_features
from experiments.estimator.benchmark import (
    PAIR_BY_NAME,
    _matched_error,
    _pair_gap_ms,
)
from lrdsr.core.mechanism_space import mechanism_init
from lrdsr.core.model import GroupedDCSR
from lrdsr.core.soft import SoftLRDSR
from lrdsr.protocol import REPORT_SEEDS
from lrdsr.theory.losses import RESULTS, noise_families

PAIRS = ("polynomial", "oscillatory", "saturation")
NOISES = ("gaussian", "laplace", "student_t3", "contaminated_10")
RHO_GRID = (0.1, 0.25)
WINDOW_LEN = 64
N_WINDOWS = 150
HARD_LOSSES = {"huber": {}, "squared": {}, "cauchy": {}, "tukey": {},
               "student_t": {"nu": 4.0}, "learned": {}}


def make_windows(pair, noise, rho: float, seed: int, m: int = N_WINDOWS,
                 n: int = WINDOW_LEN):
    """``m`` windows of the pair, noise of law ``noise`` scaled to hit ``rho``.

    Returns ``(X_seq, y, z, x, sigma)``; ``sigma`` is the noise's standard
    deviation (its variance fixes ``rho``). ``z`` is for metrics only.
    """
    rng = np.random.default_rng(seed)
    sigma = float(np.sqrt(_pair_gap_ms(pair) / rho))
    x = rng.uniform(*pair.x_range, size=(m, n))
    z = rng.integers(0, 2, size=m)
    clean = np.where(z[:, None] == 0, pair.f0(x), pair.f1(x))
    e = noise.sample(rng, (m, n)) / noise.sd * sigma
    return x[:, :, None], clean + e, z, x, sigma


def oracle_squared(x, y, pair):
    res = np.column_stack([np.sum((y - f(x)) ** 2, axis=1) for f in (pair.f0, pair.f1)])
    return res.argmin(axis=1)


def oracle_lrt(x, y, pair, noise, sigma):
    """The Bayes rule under the true noise law: the larger log-likelihood."""
    k = noise.sd / sigma                       # residual -> natural units
    ll = np.column_stack([np.sum(noise.logpdf((y - f(x)) * k), axis=1)
                          for f in (pair.f0, pair.f1)])
    return ll.argmax(axis=1)


def _hard(X, y, Z, init, seed, loss, params):
    model = GroupedDCSR(
        n_clusters=2, alpha_geom=0.0, beta_complexity=0.002, max_iter=10, tol=0.01,
        backend="fast", backend_kwargs={"max_terms": 5}, random_state=seed,
        min_windows_per_cluster=4, init=init, geom_metric="mahalanobis",
        score_mode="cross_fit", residual_scale="global", loss=loss,
        loss_params=params)
    return model.fit(X, y, Z, feature_names=["x"])


def run(seeds=REPORT_SEEDS, verbose: bool = True) -> pd.DataFrame:
    fams = noise_families()
    rows = []
    t0 = time.time()
    for pname in PAIRS:
        pair = PAIR_BY_NAME[pname]
        for nname in NOISES:
            noise = fams[nname]
            for rho in RHO_GRID:
                for seed in seeds:
                    X, y, z, x, sigma = make_windows(pair, noise, rho, seed)
                    Z, _ = window_features(X, y)
                    init = mechanism_init(X, y, 2, seed=seed, feature_names=["x"])
                    base = {"pair": pname, "noise": nname, "rho": rho, "seed": seed}
                    err = {
                        "oracle_squared": _matched_error(z, oracle_squared(x, y, pair)),
                        "oracle_lrt": _matched_error(z, oracle_lrt(x, y, pair, noise, sigma)),
                        "mechanism_kmeans": _matched_error(z, init),
                    }
                    for loss, params in HARD_LOSSES.items():
                        try:
                            res = _hard(X, y, Z, init, seed, loss, params)
                            err[f"hard_{loss}"] = _matched_error(z, res.labels)
                        except RuntimeError:          # a cluster emptied
                            err[f"hard_{loss}"] = np.nan
                    for noise_model in ("gaussian", "student_t"):
                        res = SoftLRDSR(2, noise=noise_model, feature_names=["x"],
                                        init=init, random_state=seed).fit(X, y)
                        err[f"soft_{noise_model}"] = _matched_error(z, res.labels)
                    for arm, e in err.items():
                        rows.append({**base, "arm": arm, "matched_error": e})
                if verbose:
                    last = pd.DataFrame(rows[-len(err) * len(seeds):])
                    print(f"  {pname:12s} {nname:16s} rho={rho:<5g} "
                          + " ".join(f"{a.replace('hard_', 'h:').replace('soft_', 's:')}="
                                     f"{v:.3f}" for a, v in
                                     last.groupby("arm", sort=False).matched_error.mean().items())
                          + f"  [{time.time() - t0:.0f}s]", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "loss_estimator.csv", index=False)
    summary = summarise(df)
    summary.to_csv(RESULTS / "loss_estimator_summary.csv", index=False)
    if verbose:
        print(summary.pivot_table(index="arm", columns="noise",
                                  values="mean_error").round(3).to_string())
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Mean matched error per arm x noise (pairs, rhos and seeds pooled),
    with the excess over the true Bayes rule and the rank within the noise."""
    g = (df.groupby(["noise", "arm"])
         .agg(mean_error=("matched_error", "mean"),
              sd_error=("matched_error", "std"),
              cells=("matched_error", "size"))
         .reset_index())
    lrt = g[g.arm == "oracle_lrt"].set_index("noise").mean_error
    g["excess_over_lrt"] = g.mean_error - g.noise.map(lrt)
    est = ~g.arm.str.startswith("oracle")
    g["rank_among_estimators"] = np.nan
    g.loc[est, "rank_among_estimators"] = (g[est].groupby("noise").mean_error
                                           .rank(method="min"))
    return g


if __name__ == "__main__":
    run()
