"""Is LR-DSR better than plain symbolic regression? Estimation, then classification.

The method's own baselines so far were geometric (cluster window summaries)
or raw-profile (classify the 24-hour curve). The natural competitor is
symbolic regression itself, used the way it is used without this method.
Every SR baseline runs the SAME engine LR-DSR runs (``FastSymbolicRegressor``,
same library, ``max_terms = 5``), so the comparison is about the latent-regime
structure, not the search.

Task 1 -- estimation (unlabelled; the twelve problems of the zoo):

* ``sr_pooled``      one SR law fitted to every window pooled: what SR gives
                     when the regimes are ignored.
* ``sr_per_window``  one SR law per window, windows clustered by their fitted
                     curves on a common grid (K-means, ``K`` given), then one
                     SR law refitted per cluster: SR first, clustering second.
* ``lrdsr``          the project default (``experiments.common.fitting``),
                     ``alpha_geom = 0`` as in the zoo block.
* ``oracle``         the true laws (the Bayes ceiling).

Scored by matched assignment error and by the law error: RMS distance on the
problem's evaluation grid between each true law and the law recovered for it,
in units of the smallest gap between two true laws (1.0 = as far off as the
laws are from each other).

Task 2 -- classification (labelled; train / test windows of the zoo):

* ``sr_per_class``   one SR law per class, a window goes to the smallest RSS.
* ``sr_window_1nn``  a law per window, 1-NN between fitted curves.
* ``summary_svm``    RBF SVM on the window summaries every block uses.
* ``law_L1``, ``law_L2``  ``LawClassifier`` with one / two laws per class.
* ``oracle``         the true laws.

Two class structures: ``one_law`` (class = regime) and ``two_laws`` (the
four-way problem folded into two classes of two laws each: a class is a set
of laws, which is what ``L > 1`` is for).

Declared before the run (2026-09-29):
  P1  sr_pooled is near chance on assignment and has law error >= 0.5.
  P2  lrdsr has lower mean assignment error than sr_per_window at rho <= 0.25,
      and lower mean law error at every rho.
  P3  on two_laws, law_L2 beats sr_per_class at every m; on one_law, law_L1
      and sr_per_class are within 0.02 of each other at m >= 20.
  P4  sr_window_1nn is worse than law_L1 at every m (one noisy law per
      window is a worse estimate than one pooled law per class).

Nothing is tuned; seeds {11, 23, 42}. Writes ``results/srbaseline/``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from experiments.common.fitting import fit_lrdsr, window_features
from experiments.problems import zoo
from lrdsr import paths
from lrdsr.core.backends import FastSymbolicRegressor
from lrdsr.core.classify import LawClassifier
from lrdsr.core.evaluation import aligned_accuracy

RESULTS = paths.results_dir("srbaseline", figs=False)
REPORT_SEEDS = (11, 23, 42)
RHO_GRID = (0.1, 0.25, 1.0)
N_WINDOWS = 150
WINDOW_LEN = 48
SR_KW = {"max_terms": 5}
GRID_N = 200
M_TRAIN = (5, 20, 80)
N_TEST = 400
CLASSIFY_PROBLEMS = ("frequency_shift", "saturation", "power_exponent",
                     "three_way", "moment_matched", "rescaled_copy")
CLASSIFY_RHO = 0.25


def _sr(X, y, names):
    return FastSymbolicRegressor(feature_names=names, **SR_KW).fit(X, y)


def _match(true, pred, K):
    from scipy.optimize import linear_sum_assignment
    C = np.zeros((K, K))
    for t, p in zip(true, pred):
        if p < K:
            C[t, p] += 1
    r, c = linear_sum_assignment(-C)
    return dict(zip(r, c))  # true -> pred


def _min_gap(problem) -> float:
    G = zoo.pairwise_gap_ms(problem)
    return float(np.sqrt(G[~np.eye(problem.K, dtype=bool)].min()))


def _law_error(problem, true, pred, laws) -> float:
    """Mean over true regimes of RMS(true law - its matched recovered law) on
    the evaluation grid, in units of the smallest true gap."""
    Xg = zoo.eval_grid(problem)
    mp = _match(true, pred, problem.K)
    errs = []
    for k, f in enumerate(problem.laws):
        g = laws[min(mp.get(k, 0), len(laws) - 1)]
        errs.append(np.sqrt(np.mean((f(Xg) - g(Xg)) ** 2)))
    return float(np.mean(errs) / _min_gap(problem))


def _curves(models, Xg):
    return np.vstack([m.predict(Xg) for m in models])


# ============================================================ task 1: estimation
def estimation_one(problem, rho, seed) -> dict:
    sigma = zoo.sigma_for_rho(problem, rho)
    X, y, z = zoo.make_windows(problem, sigma, N_WINDOWS, WINDOW_LEN, seed)
    names = problem.feature_names
    K = problem.K
    Xg = zoo.eval_grid(problem, GRID_N)
    row = {"problem": problem.name, "K": K, "rho": rho, "seed": seed,
           "in_library": problem.in_library}

    def score(tag, lab, laws, secs):
        row[f"{tag}_error"] = 1.0 - aligned_accuracy(z, lab)
        row[f"{tag}_law_error"] = _law_error(problem, z, lab, laws)
        row[f"{tag}_secs"] = secs

    t = time.time()
    orc = zoo.oracle_labels(problem, X, y)
    score("oracle", orc, list(problem.laws), time.time() - t)

    t = time.time()
    pooled = _sr(X.reshape(-1, problem.d), y.reshape(-1), names)
    score("sr_pooled", np.zeros(N_WINDOWS, int), [pooled.predict], time.time() - t)

    t = time.time()
    per = [_sr(X[w], y[w], names) for w in range(N_WINDOWS)]
    C = _curves(per, Xg)
    lab = KMeans(K, n_init=10, random_state=seed).fit_predict(C)
    refit = []
    for k in range(K):
        idx = np.where(lab == k)[0]
        refit.append(_sr(X[idx].reshape(-1, problem.d), y[idx].reshape(-1), names).predict
                     if len(idx) else pooled.predict)
    score("sr_per_window", lab, refit, time.time() - t)

    t = time.time()
    Zf, fnames = window_features(X[:, :, :1], y)
    # alpha_geom = 0: the zoo's documented value, the geometry carries nothing
    res = fit_lrdsr(Zf, fnames, X, y, names, K, seed, alpha_geom=0.0)
    score("lrdsr", np.asarray(res.labels), [m.predict for m in res.models],
          time.time() - t)
    return row


def run_estimation() -> pd.DataFrame:
    rows = []
    for p in zoo.PROBLEMS:
        for rho in RHO_GRID:
            for s in REPORT_SEEDS:
                rows.append(estimation_one(p, rho, s))
        print(f"  estimation  {p.name:16s} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "estimation_per_seed.csv", index=False)
    return df


# ======================================================== task 2: classification
def _two_law_classes(z):
    """four_way folded: laws {0, 1} are class 0, laws {2, 3} class 1."""
    return (z >= 2).astype(int)


def classify_one(problem, structure, m, seed) -> list[dict]:
    sigma = zoo.sigma_for_rho(problem, CLASSIFY_RHO)
    n_classes = 2 if structure == "two_laws" else problem.K
    # draw enough windows to take m per class for training
    Xa, ya, za = zoo.make_windows(problem, sigma, 12 * m * problem.K + N_TEST,
                                  WINDOW_LEN, seed)
    ca = _two_law_classes(za) if structure == "two_laws" else za
    rng = np.random.default_rng(seed)
    tr = np.concatenate([rng.permutation(np.where(ca == c)[0])[:m]
                         for c in range(n_classes)])
    te = np.setdiff1d(np.arange(len(ca)), tr)[:N_TEST]
    Xtr, ytr, ctr = Xa[tr], ya[tr], ca[tr]
    Xte, yte, cte = Xa[te], ya[te], ca[te]
    names = problem.feature_names
    Xg = zoo.eval_grid(problem, GRID_N)
    out = []

    def add(method, pred, secs):
        out.append({"problem": problem.name, "structure": structure, "m": m,
                    "seed": seed, "method": method,
                    "error": float(np.mean(pred != cte)), "secs": secs})

    # oracle: the class of the true law with the smallest RSS
    rss = np.column_stack([np.sum((yte - f(Xte)) ** 2, 1) for f in problem.laws])
    law_cls = np.array([_two_law_classes(np.array([k]))[0]
                        if structure == "two_laws" else k for k in range(problem.K)])
    add("oracle", law_cls[rss.argmin(1)], 0.0)

    t = time.time()
    laws = [_sr(Xtr[ctr == c].reshape(-1, problem.d), ytr[ctr == c].reshape(-1), names)
            for c in range(n_classes)]
    rss = np.column_stack([np.sum((yte - g.predict(Xte.reshape(-1, problem.d))
                                   .reshape(yte.shape)) ** 2, 1) for g in laws])
    add("sr_per_class", rss.argmin(1), time.time() - t)

    t = time.time()
    Ctr = _curves([_sr(Xtr[w], ytr[w], names) for w in range(len(tr))], Xg)
    Cte = _curves([_sr(Xte[w], yte[w], names) for w in range(len(te))], Xg)
    add("sr_window_1nn",
        KNeighborsClassifier(1).fit(Ctr, ctr).predict(Cte), time.time() - t)

    t = time.time()
    Ftr, _ = window_features(Xtr[:, :, :1], ytr)
    Fte, _ = window_features(Xte[:, :, :1], yte)
    svm = make_pipeline(StandardScaler(), SVC(C=10.0, gamma="scale"))
    add("summary_svm", svm.fit(Ftr, ctr).predict(Fte), time.time() - t)

    for L in (1, 2):
        t = time.time()
        clf = LawClassifier(feature_names=names, nuisance=None, laws_per_class=L,
                            random_state=seed).fit(Xtr, ytr, ctr)
        add(f"law_L{L}", clf.predict(Xte, yte), time.time() - t)
    return out


def run_classification() -> pd.DataFrame:
    rows = []
    jobs = [(zoo.PROBLEM_BY_NAME[n], "one_law") for n in CLASSIFY_PROBLEMS]
    jobs.append((zoo.PROBLEM_BY_NAME["four_way"], "two_laws"))
    for p, st in jobs:
        for m in M_TRAIN:
            for s in REPORT_SEEDS:
                rows.extend(classify_one(p, st, m, s))
        print(f"  classify    {p.name:16s} {st} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "classification_per_seed.csv", index=False)
    return df


def summarise(est: pd.DataFrame, cls: pd.DataFrame) -> None:
    tags = ["oracle", "lrdsr", "sr_per_window", "sr_pooled"]
    a = est.groupby("rho")[[f"{t}_error" for t in tags]].mean()
    b = est.groupby("rho")[[f"{t}_law_error" for t in tags]].mean()
    a.to_csv(RESULTS / "estimation_assignment.csv")
    b.to_csv(RESULTS / "estimation_law.csv")
    print("\nTask 1 -- estimation: assignment error (mean over 12 problems x 3 seeds)")
    print(a.round(3).to_string())
    print("\nTask 1 -- estimation: law error (units of the smallest true gap)")
    print(b.round(3).to_string())
    pp = est.groupby("problem")[["oracle_error", "lrdsr_error", "sr_per_window_error",
                                 "lrdsr_law_error", "sr_per_window_law_error"]].mean()
    pp.to_csv(RESULTS / "estimation_by_problem.csv")
    print("\nper problem (all rho):")
    print(pp.round(3).to_string())
    c = (cls.groupby(["structure", "m", "method"])["error"].mean()
            .unstack("method"))
    c = c[["oracle", "law_L1", "law_L2", "sr_per_class", "sr_window_1nn", "summary_svm"]]
    c.to_csv(RESULTS / "classification_summary.csv")
    print("\nTask 2 -- classification error (mean over problems x 3 seeds)")
    print(c.round(3).to_string())


def run(args=None) -> None:
    est = run_estimation()
    cls = run_classification()
    summarise(est, cls)


if __name__ == "__main__":
    run()
