"""Every method on every problem in the zoo, across the separation knob.

    python -m experiments problems

For each problem of ``experiments.problems.zoo`` and each ``rho`` (the
separation of its *closest* pair of laws), 150 windows of 48 samples, three
reporting seeds. The methods, all given the same windows and the same ``K``:

==================  =======================================================
``oracle``          the Bayes rule with the TRUE laws and equal priors: the
                    ceiling no estimator beats (up to sampling noise)
``lrdsr``           the hard loop, ``experiments.common.fitting.fit_lrdsr``,
                    ``alpha_geom = 0`` (the geometry carries nothing here)
``soft_em``         ``lrdsr.core.soft.SoftLRDSR``, Gaussian noise
``mechanism_kmeans`` K-means in mechanism space, no law fitted
``geometry``        the BEST of the seven geometric baselines on plain window
                    summaries, chosen after the fact against the truth -- an
                    optimistic bound on what clustering the data can do
``profile_kmeans``  K-means on each window's ``y`` ordered by ``x``: the raw
                    curve, no summary (one-input problems only)
==================  =======================================================

The truth reaches the metrics and nothing else. Recovered laws from the
hard loop are scored by their NMSE against the matched true law on a dense
grid, and their expressions are kept.

Writes ``results/problems/problems_{raw,summary,laws,catalog}.csv``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from experiments.common.fitting import fit_lrdsr, window_features
from experiments.problems.zoo import (
    PROBLEMS,
    eval_grid,
    gap_outside_library,
    in_library_residual,
    make_windows,
    oracle_labels,
    sigma_for_rho,
)
from lrdsr import paths
from lrdsr.core.baselines import geometry_baselines
from lrdsr.core.evaluation import aligned_accuracy, clustering_metrics, nmse
from lrdsr.core.mechanism_space import mechanism_init
from lrdsr.core.soft import SoftLRDSR

RESULTS = paths.results_dir("problems", figs=False)
REPORT_SEEDS = (11, 23, 42)
RHO_GRID = (0.1, 0.25, 1.0)
N_WINDOWS = 150
WINDOW_LEN = 48
METHODS = ("oracle", "lrdsr", "soft_em", "mechanism_kmeans", "geometry",
           "profile_kmeans")


def _err(true, pred) -> float:
    return 1.0 - aligned_accuracy(true, pred)


def _mapping(true, pred, K) -> dict:
    """Hungarian map predicted slot -> true slot."""
    C = np.zeros((K, K), int)
    for t, p in zip(true, pred, strict=True):
        C[int(t), int(p)] += 1
    r, c = linear_sum_assignment(-C)
    return {int(cc): int(rr) for rr, cc in zip(r, c, strict=True)}


def _profile_kmeans(X, y, K, seed):
    order = np.argsort(X[..., 0], axis=1)
    prof = np.take_along_axis(y, order, axis=1)
    return KMeans(n_clusters=K, n_init=10, random_state=seed).fit_predict(prof)


def run_cell(problem, seed, rho_grid=RHO_GRID, n_windows=N_WINDOWS,
             window_len=WINDOW_LEN):
    """All methods on one problem and seed, at every rho. Returns row lists."""
    rows, laws = [], []
    K, names = problem.K, problem.feature_names
    grid = eval_grid(problem)
    for rho in rho_grid:
        sigma = sigma_for_rho(problem, rho)
        X, y, z = make_windows(problem, sigma, n_windows, window_len, seed)
        Zf, fnames = window_features(X, y)

        labels, secs = {}, {}
        t = time.time()
        labels["oracle"] = oracle_labels(problem, X, y)
        secs["oracle"] = time.time() - t
        t = time.time()
        hard = fit_lrdsr(Zf, fnames, X, y, names, K, seed, alpha_geom=0.0)
        labels["lrdsr"], secs["lrdsr"] = hard.labels, time.time() - t
        t = time.time()
        soft = SoftLRDSR(K, feature_names=names, random_state=seed).fit(X, y)
        labels["soft_em"], secs["soft_em"] = soft.labels, time.time() - t
        t = time.time()
        labels["mechanism_kmeans"] = mechanism_init(X, y, K, seed=seed,
                                                    feature_names=names)
        secs["mechanism_kmeans"] = time.time() - t
        t = time.time()
        geom = geometry_baselines(Zf, K, seed=seed)
        geom_err = {b: _err(z, lab) for b, lab in geom.items()}
        best = min(geom_err, key=geom_err.get)
        labels["geometry"], secs["geometry"] = geom[best], time.time() - t
        if problem.d == 1:
            t = time.time()
            labels["profile_kmeans"] = _profile_kmeans(X, y, K, seed)
            secs["profile_kmeans"] = time.time() - t

        for method, lab in labels.items():
            m = clustering_metrics(z, lab)
            rows.append({
                "problem": problem.name, "K": K, "d": problem.d, "rho": rho,
                "sigma": sigma, "seed": seed, "method": method,
                "matched_error": 1.0 - m["aligned_accuracy"], "ARI": m["ARI"],
                "NMI": m["NMI"], "seconds": secs[method],
                "best_geometry_baseline": best if method == "geometry" else "",
            })

        mp = _mapping(z, hard.labels, K)
        for slot, model in enumerate(hard.models):
            k = mp.get(slot, slot)
            laws.append({
                "problem": problem.name, "rho": rho, "seed": seed, "true_slot": k,
                "truth": problem.truths[k], "recovered": model.expression(),
                "complexity": float(model.complexity()),
                "law_nmse": nmse(problem.laws[k](grid), model.predict(grid)),
            })
    return rows, laws


def catalog() -> pd.DataFrame:
    return pd.DataFrame([{
        "problem": p.name, "K": p.K, "d": p.d, "truths": " | ".join(p.truths),
        "why": p.why, "in_library": p.in_library,
        "law_outside_library": in_library_residual(p),
        "gap_outside_library": gap_outside_library(p),
        "x_lo": p.x_range[0], "x_hi": p.x_range[1],
    } for p in PROBLEMS])


def summarise(raw: pd.DataFrame, laws: pd.DataFrame) -> pd.DataFrame:
    wide = (raw.pivot_table(index=["problem", "K", "d", "rho", "seed"],
                            columns="method", values="matched_error")
               .reset_index())
    for m in METHODS:
        if m != "oracle" and m in wide:
            wide[f"{m}_gap"] = wide[m] - wide["oracle"]
    agg = wide.groupby(["problem", "K", "d", "rho"]).mean(numeric_only=True)
    agg = agg.drop(columns="seed").reset_index()
    ln = (laws.groupby(["problem", "rho"]).law_nmse.median()
              .rename("lrdsr_law_nmse_median").reset_index())
    return agg.merge(ln, on=["problem", "rho"], how="left")


def run(args=None, n_jobs: int = -1) -> pd.DataFrame:
    t0 = time.time()
    cat = catalog()
    cat.to_csv(RESULTS / "problems_catalog.csv", index=False)
    print(cat[["problem", "K", "d", "in_library", "gap_outside_library"]]
          .to_string(index=False), flush=True)

    out = Parallel(n_jobs=n_jobs)(
        delayed(run_cell)(p, s) for p in PROBLEMS for s in REPORT_SEEDS)
    raw = pd.DataFrame([r for rows, _ in out for r in rows])
    laws = pd.DataFrame([r for _, lw in out for r in lw])
    raw.to_csv(RESULTS / "problems_raw.csv", index=False)
    laws.to_csv(RESULTS / "problems_laws.csv", index=False)
    summ = summarise(raw, laws)
    summ.to_csv(RESULTS / "problems_summary.csv", index=False)

    with pd.option_context("display.width", 200, "display.precision", 3):
        print(summ[["problem", "rho", *[m for m in METHODS if m in summ],
                    "lrdsr_law_nmse_median"]].to_string(index=False))
    print(f"\n[problems] {len(raw)} rows in {(time.time() - t0) / 60:.1f} min "
          f"-> {RESULTS}")
    return summ


if __name__ == "__main__":
    run()
