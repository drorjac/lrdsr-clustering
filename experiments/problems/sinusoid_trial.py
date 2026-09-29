"""Trial: a library term with a fitted frequency, on every zoo problem.

    python -m experiments.problems.sinusoid_trial

The zoo's one failure is ``high_frequency`` (``sin 4x`` vs ``sin 4.6x``):
37% of the gap lies outside the span of the fixed library, whose fastest
term is ``sin 2x``. ``SinusoidSymbolicRegressor`` (backend ``fast_sin``) adds
``sin(a x)`` and ``cos(a x)`` with the frequency ``a`` fitted by variable
projection. The problem, the objective and the loop are unchanged: only the
per-cluster fit has a richer vocabulary. The start is left on the plain
library, so any gain is the loop's.

Same design as ``experiments.problems.run``: 150 windows of 48 samples,
``rho`` in {0.1, 0.25, 1.0}, the reporting seeds, the hard loop with the
project's settings and ``alpha_geom = 0``, once per backend on the same
windows.

Declared before the run (2026-09-29):
  S1  on high_frequency the fitted-frequency loop's gap to the oracle, mean
      over rho and seeds, is below 0.02 (library: 0.237; kernel soft EM,
      mission 10: 0.051).
  S2  on frequency_shift the gap does not grow by more than 0.01.
  S3  over the other ten problems the mean gap changes by at most 0.005,
      and no problem's mean gap grows by more than 0.02.
  S4  on high_frequency at rho = 1 every seed recovers both frequencies to
      within 0.05 of 4 and 4.6.

Stage two -- standard methods, head to head (``--baselines``). On the two
sine problems, the same windows, every arm:

* ``oracle``, ``lrdsr`` (library) and ``lrdsr_sin`` (fitted frequency);
* ``soft_em`` and ``mechanism_kmeans`` (the method's own family, library basis);
* ``geometry`` -- the best of seven clusterings of window summaries, picked
  after the fact against the truth (an optimistic bound);
* ``profile_kmeans`` -- K-means on each window's ``y`` ordered by ``x``;
* ``sr_per_window`` and ``sr_per_window_sin`` -- plain symbolic regression,
  one law per window, K-means on the fitted curves; the second with the same
  fitted-frequency vocabulary, so the vocabulary is not credited to LR-DSR;
* ``periodogram_kmeans`` -- the signal-processing standard: each window's
  Lomb-Scargle periodogram (irregular sampling) over the same frequency
  grid, K-means on it; ``peak_kmeans`` -- K-means on the peak frequency alone.

Declared before stage two ran (2026-09-29):
  B1  on high_frequency window summaries, the raw profile and SR per window
      (library) err at >= 0.30 at every rho (the zoo's picture, replicated).
  B2  on high_frequency periodogram_kmeans is within 0.05 of lrdsr_sin at
      rho = 1 and worse by more than 0.05 at rho = 0.1.
  B3  sr_per_window_sin is worse than lrdsr_sin by more than 0.05 at every
      rho <= 0.25: 48 samples pin a frequency down worse than a cluster does.
  B4  on frequency_shift (sin x vs sin 1.3x, closer than one period over the
      range) both periodogram arms are worse than lrdsr_sin by more than 0.1
      at every rho.

Nothing is tuned: the frequency grid (0.5 to 8 in steps of 0.05) and the
safeguards were fixed on hand-made laws before this run, never on the zoo.
Writes ``results/problems/sinusoid_trial_{raw,summary,verdict}.csv`` and, for
stage two, ``sinusoid_baselines_{raw,verdict}.csv``.
"""
from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from experiments.common.fitting import window_features
from experiments.problems.zoo import (
    PROBLEMS,
    make_windows,
    oracle_labels,
    sigma_for_rho,
)
from lrdsr import GroupedDCSR, paths
from lrdsr.core.backends import make_symbolic_regressor
from lrdsr.core.baselines import geometry_baselines
from lrdsr.core.evaluation import aligned_accuracy
from lrdsr.core.mechanism_space import mechanism_init
from lrdsr.core.soft import SoftLRDSR

RESULTS = paths.results_dir("problems", figs=False)
REPORT_SEEDS = (11, 23, 42)
RHO_GRID = (0.1, 0.25, 1.0)
N_WINDOWS, WINDOW_LEN = 150, 48
BACKENDS = ("fast", "fast_sin")


def fit(backend, X, y, Z, names, K, seed):
    """``experiments.common.fitting.fit_lrdsr`` with the backend as a knob."""
    return GroupedDCSR(
        n_clusters=K, alpha_geom=0.0, beta_complexity=0.002, max_iter=10, tol=0.01,
        backend=backend, backend_kwargs={"max_terms": 5}, random_state=seed,
        min_windows_per_cluster=4, init="mechanism", geom_metric="mahalanobis",
        score_mode="cross_fit", residual_scale="global",
    ).fit(X, y, Z, feature_names=list(names))


def run_cell(problem, seed):
    rows = []
    for rho in RHO_GRID:
        X, y, z = make_windows(problem, sigma_for_rho(problem, rho), N_WINDOWS, WINDOW_LEN, seed)
        Zf, _ = window_features(X, y)
        oracle_err = 1.0 - float(np.mean(oracle_labels(problem, X, y) == z))
        for backend in BACKENDS:
            t = time.time()
            res = fit(backend, X, y, Zf, problem.feature_names, problem.K, seed)
            exprs = [m.expression() for m in res.models]
            freqs = sorted(float(f) for e in exprs for f in re.findall(r"(?:sin|cos)\(([\d.]+)\*", e))
            rows.append({
                "problem": problem.name, "K": problem.K, "rho": rho, "seed": seed,
                "backend": backend, "error": 1.0 - aligned_accuracy(z, res.labels),
                "oracle": oracle_err, "seconds": time.time() - t,
                "laws": " | ".join(exprs), "fitted_freqs": " ".join(f"{f:.4g}" for f in freqs),
            })
    return rows


def summarise(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.assign(gap=raw.error - raw.oracle)
    wide = raw.pivot_table(index=["problem", "rho", "seed"], columns="backend",
                           values=["gap", "seconds"]).reset_index()
    wide.columns = ["_".join(c).strip("_") for c in wide.columns]
    return (wide.groupby("problem")[["gap_fast", "gap_fast_sin", "seconds_fast", "seconds_fast_sin"]]
                .mean().reset_index())


def verdicts(raw: pd.DataFrame, summ: pd.DataFrame) -> pd.DataFrame:
    s = summ.set_index("problem")
    d = s.gap_fast_sin - s.gap_fast
    other = d.drop(["high_frequency", "frequency_shift"])
    hf1 = raw[(raw.problem == "high_frequency") & (raw.rho == 1.0) & (raw.backend == "fast_sin")]

    def hits(fs):
        f = np.array([float(v) for v in fs.split()]) if fs else np.array([])
        return all(f.size and np.min(np.abs(f - t)) <= 0.05 for t in (4.0, 4.6))

    out = [
        ("S1", "high_frequency gap < 0.02", float(s.gap_fast_sin["high_frequency"]),
         bool(s.gap_fast_sin["high_frequency"] < 0.02)),
        ("S2", "frequency_shift gap grows by <= 0.01", float(d["frequency_shift"]),
         bool(d["frequency_shift"] <= 0.01)),
        ("S3", "other ten: mean change <= 0.005, none worse by > 0.02",
         float(other.mean()), bool(abs(other.mean()) <= 0.005 and other.max() <= 0.02)),
        ("S4", "rho = 1: both frequencies within 0.05, every seed",
         float(np.mean([hits(f) for f in hf1.fitted_freqs])),
         bool(all(hits(f) for f in hf1.fitted_freqs))),
    ]
    return pd.DataFrame(out, columns=["prediction", "claim", "value", "held"])


# ==========================================================================
# stage two: standard methods on the sine problems
# ==========================================================================
SINE_PROBLEMS = ("high_frequency", "frequency_shift")
FREQ_GRID = np.arange(0.5, 8.0001, 0.05)


def _kmeans(F, K, seed):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    F = StandardScaler().fit_transform(np.asarray(F, dtype=float).reshape(len(F), -1))
    return KMeans(K, n_init=10, random_state=seed).fit_predict(F)


def _periodogram(X, y):
    from scipy.signal import lombscargle
    return np.vstack([lombscargle(X[w, :, 0], y[w] - y[w].mean(), FREQ_GRID, normalize=True)
                      for w in range(len(y))])


def _sr_per_window(backend, X, y, names, grid, K, seed):
    curves = np.vstack([make_symbolic_regressor(backend, feature_names=names, max_terms=5)
                        .fit(X[w], y[w]).predict(grid) for w in range(len(y))])
    return _kmeans(curves, K, seed)


def baselines_cell(problem, seed):
    from experiments.problems.zoo import eval_grid
    rows = []
    names, K = problem.feature_names, problem.K
    grid = eval_grid(problem, 200)
    for rho in RHO_GRID:
        X, y, z = make_windows(problem, sigma_for_rho(problem, rho), N_WINDOWS, WINDOW_LEN, seed)
        Zf, _ = window_features(X, y)
        order = np.argsort(X[..., 0], axis=1)
        P = _periodogram(X, y)
        geom = geometry_baselines(Zf, K, seed=seed)
        labels = {
            "oracle": oracle_labels(problem, X, y),
            "lrdsr": fit("fast", X, y, Zf, names, K, seed).labels,
            "lrdsr_sin": fit("fast_sin", X, y, Zf, names, K, seed).labels,
            "soft_em": SoftLRDSR(K, feature_names=names, random_state=seed).fit(X, y).labels,
            "mechanism_kmeans": mechanism_init(X, y, K, seed=seed, feature_names=names),
            "geometry": min(geom.values(), key=lambda lab: 1.0 - aligned_accuracy(z, lab)),
            "profile_kmeans": _kmeans(np.take_along_axis(y, order, axis=1), K, seed),
            "sr_per_window": _sr_per_window("fast", X, y, names, grid, K, seed),
            "sr_per_window_sin": _sr_per_window("fast_sin", X, y, names, grid, K, seed),
            "periodogram_kmeans": _kmeans(P, K, seed),
            "peak_kmeans": _kmeans(FREQ_GRID[np.argmax(P, axis=1)], K, seed),
        }
        for method, lab in labels.items():
            rows.append({"problem": problem.name, "rho": rho, "seed": seed, "method": method,
                         "error": 1.0 - aligned_accuracy(z, lab)})
    return rows


def baseline_verdicts(raw: pd.DataFrame) -> pd.DataFrame:
    m = raw.groupby(["problem", "rho", "method"]).error.mean().unstack("method")
    hf, fs = m.loc["high_frequency"], m.loc["frequency_shift"]
    b1 = hf[["geometry", "profile_kmeans", "sr_per_window"]].min().min()
    d_hi = hf.loc[1.0, "periodogram_kmeans"] - hf.loc[1.0, "lrdsr_sin"]
    d_lo = hf.loc[0.1, "periodogram_kmeans"] - hf.loc[0.1, "lrdsr_sin"]
    b3 = (hf.sr_per_window_sin - hf.lrdsr_sin).loc[[0.1, 0.25]]
    b4 = min((fs.periodogram_kmeans - fs.lrdsr_sin).min(), (fs.peak_kmeans - fs.lrdsr_sin).min())
    out = [
        ("B1", "summaries, raw profile, SR per window: error >= 0.30, every rho",
         float(b1), bool(b1 >= 0.30)),
        ("B2", "periodogram within 0.05 at rho = 1, worse by > 0.05 at rho = 0.1",
         float(d_hi), bool(d_hi <= 0.05 and d_lo > 0.05)),
        ("B3", "SR per window (fitted frequency) worse by > 0.05 at rho <= 0.25",
         float(b3.min()), bool((b3 > 0.05).all())),
        ("B4", "frequency_shift: periodograms worse by > 0.1, every rho",
         float(b4), bool(b4 > 0.1)),
    ]
    return pd.DataFrame(out, columns=["prediction", "claim", "value", "held"])


def run_baselines(n_jobs: int = -1) -> pd.DataFrame:
    from experiments.problems.zoo import PROBLEM_BY_NAME
    out = Parallel(n_jobs=n_jobs)(delayed(baselines_cell)(PROBLEM_BY_NAME[p], s)
                                  for p in SINE_PROBLEMS for s in REPORT_SEEDS)
    raw = pd.DataFrame([r for rows in out for r in rows])
    raw.to_csv(RESULTS / "sinusoid_baselines_raw.csv", index=False)
    v = baseline_verdicts(raw)
    v.to_csv(RESULTS / "sinusoid_baselines_verdict.csv", index=False)
    with pd.option_context("display.width", 220, "display.precision", 3):
        print(raw.groupby(["problem", "rho", "method"]).error.mean().unstack("rho").round(3))
        print(v.to_string(index=False))
    return raw


def run(args=None, n_jobs: int = -1) -> pd.DataFrame:
    t0 = time.time()
    out = Parallel(n_jobs=n_jobs)(delayed(run_cell)(p, s) for p in PROBLEMS for s in REPORT_SEEDS)
    raw = pd.DataFrame([r for rows in out for r in rows])
    raw.to_csv(RESULTS / "sinusoid_trial_raw.csv", index=False)
    summ = summarise(raw)
    summ.to_csv(RESULTS / "sinusoid_trial_summary.csv", index=False)
    v = verdicts(raw, summ)
    v.to_csv(RESULTS / "sinusoid_trial_verdict.csv", index=False)
    with pd.option_context("display.width", 200, "display.precision", 4):
        print(summ.to_string(index=False))
        print(v.to_string(index=False))
        print(raw[(raw.problem == "high_frequency") & (raw.backend == "fast_sin")]
              [["rho", "seed", "fitted_freqs"]].to_string(index=False))
    print(f"\n[sinusoid trial] {len(raw)} rows in {(time.time() - t0) / 60:.1f} min -> {RESULTS}")
    return summ


if __name__ == "__main__":
    import sys
    if "--baselines" in sys.argv:
        run_baselines()
    else:
        run()
