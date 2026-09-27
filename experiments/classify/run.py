"""Classification: a class is a set of laws. Theory, then 24 real datasets.

    python -m experiments classify

==================  =======================================================
``v10``             the plug-in learning curve of a law classifier,
                    predicted and simulated (``lrdsr.theory.classification``)
``benchmark``       24 UCR datasets, the archive's own train/test split:
                    law classifiers (CV on train), the feature domain
                    (mechanism features into logistic / SVM), raw-profile
                    baselines, and the archive's published 1-NN ED / DTW
``fewshot``         ``m`` = 1, 2, 5, 10 labelled series per class: the V10
                    prediction on real data -- a law pays for its basis
                    dimension, a raw profile for its length
``irregular``       each series keeps a random 50%, 25% or 10% of its
                    samples: law methods score what was observed, raw
                    methods interpolate first
``clustering``      the unsupervised problem on the same data (train+test,
                    ``K`` the number of classes): mechanism K-means and soft
                    EM in a cosine basis against K-means on the raw profile
==================  =======================================================

Protocol. Hyperparameters of every method are chosen by stratified CV on
the training split only (``methods.cv_law``, ``GridSearchCV``); the test
split is touched once, to score. In ``fewshot`` there is no CV at ``m = 1``,
so the basis size is chosen **without labels** from the training windows'
own leave-one-out fit (``lrdsr.core.kernel.select_rank``) and ``L`` is
fixed a priori (one law per class, and one law per window). Randomness
(which series are sampled or kept) uses the reporting seeds.

The dataset list and its split into ``daily`` and ``shape`` groups were
fixed before any run (``experiments/classify/ucr.py``).

Writes ``results/classify/``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans

from experiments.classify import methods as M
from experiments.classify import ucr
from lrdsr import paths
from lrdsr.core.evaluation import clustering_metrics
from lrdsr.protocol import REPORT_SEEDS

RESULTS = paths.results_dir("classify", figs=False)
FEWSHOT_M = (1, 2, 5, 10)
FEWSHOT_DRAWS = 5                       # training sets per seed and m
KEEP = (1.0, 0.5, 0.25, 0.1)
N_JOBS = 8


def _err(y, pred) -> float:
    return float(np.mean(np.asarray(pred) != np.asarray(y)))


# ==========================================================================
# the benchmark
# ==========================================================================
def benchmark_one(name: str) -> tuple[list[dict], dict]:
    s = ucr.load(name)
    t0 = time.time()
    Xs, ys = ucr.to_windows(s.train)
    rows = []

    def add(method, family, pred, secs, **extra):
        rows.append({"dataset": name, "group": s.group, "method": method,
                     "family": family, "error": _err(s.y_test, pred),
                     "seconds": secs, **extra})

    configs = {}
    for kind in ("cosine", "nystrom"):
        t = time.time()
        cfg, cv_acc = M.cv_law(Xs, ys, s.y_train, s.length, kind=kind)
        pred, model = M.law_predict(s.train, s.y_train, s.test, cfg, kind=kind)
        configs[kind] = cfg
        add(f"law_{kind}", "law", pred, time.time() - t, basis_size=cfg["size"],
            L=str(cfg["L"]), nuisance=str(cfg["nuisance"]), cv_accuracy=cv_acc,
            n_laws=model.n_laws)
    t = time.time()
    try:
        pred, model = M.law_symbolic_predict(s.train, s.y_train, s.test)
        add("law_symbolic", "law", pred, time.time() - t, n_laws=model.n_laws,
            expressions=" | ".join(model.expressions())[:400])
    except Exception as exc:                        # noqa: BLE001 - reported, not hidden
        print(f"  [{name}] law_symbolic failed: {exc}", flush=True)
    t = time.time()
    cfg = configs["cosine"]
    for clf, pred in M.mech_predict(s.train, s.y_train, s.test, cfg["size"],
                                    cfg["nuisance"]).items():
        add(f"mech_{clf}", "feature domain", pred, time.time() - t, basis_size=cfg["size"])
    t = time.time()
    for clf, pred in M.raw_predict(s.train, s.y_train, s.test, s.length).items():
        add(f"raw_{clf}", "raw profile", pred, time.time() - t)
    print(f"  [{name}] benchmark {time.time() - t0:.0f}s", flush=True)
    return rows, {"dataset": name, **{f"{k}_{f}": str(v) for k, c in configs.items()
                                      for f, v in c.items()}}


def run_benchmark(names=ucr.DATASETS):
    out = Parallel(n_jobs=N_JOBS)(delayed(benchmark_one)(n) for n in names)
    df = pd.DataFrame([r for rows, _ in out for r in rows])
    cfg = pd.DataFrame([c for _, c in out])
    pub = ucr.published()
    for col, method in (("published_ed_error", "published_1nn_ed"),
                        ("published_dtw_error", "published_1nn_dtw")):
        p = pub[["dataset", col]].rename(columns={col: "error"})
        p["method"], p["family"] = method, "published"
        p["group"] = p["dataset"].map(ucr.GROUP)
        df = pd.concat([df, p], ignore_index=True)
    return df, cfg


# ==========================================================================
# few-shot: V10 on real data
# ==========================================================================
def _draw(y, m, rng):
    idx = []
    for c in np.unique(y):
        pool = np.flatnonzero(y == c)
        idx += list(rng.choice(pool, size=min(m, len(pool)), replace=False))
    return np.asarray(idx)


def fewshot_one(name: str) -> list[dict]:
    s = ucr.load(name)
    rows = []
    Xt, yt = ucr.to_windows(s.test)
    for seed in REPORT_SEEDS:
        rng = np.random.default_rng([seed, len(name)])
        for m in FEWSHOT_M:
            if np.bincount(s.y_train).min() < m:
                continue
            for draw in range(FEWSHOT_DRAWS):
                idx = _draw(s.y_train, m, rng)
                tr = [s.train[i] for i in idx]
                ytr = s.y_train[idx]
                size = M.loo_size(tr)
                Xs, ys = ucr.to_windows(tr)
                for L in (1, "all"):
                    law = M.LawClassifier(basis=M.CosineBasis(size), nuisance=None,
                                          laws_per_class=L).fit(Xs, ys, ytr)
                    rows.append({"method": f"law_L{L}", "error": _err(
                        s.y_test, law.predict(Xt, yt)), "basis_size": size})
                for k, pred in M.raw_predict(tr, ytr, s.test, s.length,
                                             which="fast").items():
                    rows.append({"method": f"raw_{k}", "error": _err(s.y_test, pred),
                                 "basis_size": s.length})
                for r in rows[-4:]:
                    r.update({"dataset": name, "group": s.group, "seed": seed,
                              "m": m, "draw": draw})
    print(f"  [{name}] fewshot done", flush=True)
    return rows


def run_fewshot(names=ucr.DATASETS) -> pd.DataFrame:
    out = Parallel(n_jobs=N_JOBS)(delayed(fewshot_one)(n) for n in names)
    return pd.DataFrame([r for rows in out for r in rows])


# ==========================================================================
# irregular sampling
# ==========================================================================
def irregular_one(name: str, cfg: dict) -> list[dict]:
    s = ucr.load(name)
    law_cfg = {"size": int(cfg["cosine_size"]),
               "L": cfg["cosine_L"] if cfg["cosine_L"] == "all" else int(cfg["cosine_L"]),
               "nuisance": None if cfg["cosine_nuisance"] == "None" else "intercept"}
    rows = []
    for seed in REPORT_SEEDS:
        for keep in KEEP:
            if keep == 1.0 and seed != REPORT_SEEDS[0]:
                continue                       # nothing random at keep = 1
            tr = ucr.subsample(s.train, keep, seed) if keep < 1 else s.train
            te = ucr.subsample(s.test, keep, seed + 1) if keep < 1 else s.test
            # one law PER WINDOW (L = "all") cannot be richer than half the
            # window's kept samples; pooled class laws can. Cap only the former
            # (a rule of the method, not tuned per dataset). The feature map
            # reads one set of coefficients per window, so it is capped too.
            # The cap follows the TYPICAL window (median kept samples): a few
            # nearly empty series must not shrink the basis of every other
            # one, and the ridge keeps their own fits finite.
            typical = int(np.median([np.isfinite(v).sum() for v in tr + te]))
            cap = int(min(law_cfg["size"], max(typical // 2, 2)))
            cfg_k = dict(law_cfg, size=cap) if law_cfg["L"] == "all" else law_cfg
            pred, _ = M.law_predict(tr, s.y_train, te, cfg_k)
            res = {"law_cosine": pred}
            for k, p in M.raw_predict(tr, s.y_train, te, s.length, which="fast").items():
                res[f"raw_{k}"] = p
            for mode in ("project", "solve"):
                p = M.mech_predict(tr, s.y_train, te, cap, cfg_k["nuisance"],
                                   mode=mode, which=("logistic",))["logistic"]
                res[f"mech_logistic_{mode}"] = p
            for method, pred in res.items():
                rows.append({"dataset": name, "group": s.group, "keep": keep,
                             "seed": seed, "method": method,
                             "error": _err(s.y_test, pred), "basis_size": cfg_k["size"]})
    print(f"  [{name}] irregular done", flush=True)
    return rows


def run_irregular(configs: pd.DataFrame, names=ucr.DATASETS) -> pd.DataFrame:
    cfg = configs.set_index("dataset")
    todo = [n for n in names if n in cfg.index]
    out = Parallel(n_jobs=N_JOBS)(delayed(irregular_one)(n, cfg.loc[n].to_dict())
                                  for n in todo)
    return pd.DataFrame([r for rows in out for r in rows])


# ==========================================================================
# clustering
# ==========================================================================
def clustering_one(name: str) -> list[dict]:
    from lrdsr.core.classify import MechanismFeatures
    from lrdsr.core.kernel import CosineBasis
    from lrdsr.core.soft import SoftLRDSR

    s = ucr.load(name)
    series = s.train + s.test
    y = np.r_[s.y_train, s.y_test]
    K = s.n_classes
    rows = []
    for keep in (1.0, 0.25):
        for seed in REPORT_SEEDS:
            ser = ucr.subsample(series, keep, seed) if keep < 1 else series
            Xs, ys = ucr.to_windows(ser)
            size = M.loo_size(ser)
            res = {}
            for mode in ("project", "solve"):
                S = MechanismFeatures(basis=CosineBasis(size), nuisance=None,
                                      mode=mode).fit_transform(Xs, ys)
                res[f"mech_kmeans_{mode}"] = KMeans(K, n_init=10,
                                                    random_state=seed).fit_predict(S)
            # soft EM takes one design length; a dataset with missing values
            # (ragged windows) has no such arm
            if keep == 1.0 and len({len(v) for v in ys}) == 1:
                X3, Y3 = np.stack(Xs), np.stack(ys)
                res["soft_em_cosine"] = SoftLRDSR(
                    K, basis=CosineBasis(size), init=res["mech_kmeans_project"],
                    random_state=seed).fit(X3, Y3).labels
            G = ucr.to_grid(ser, s.length)
            res["raw_kmeans"] = KMeans(K, n_init=10, random_state=seed).fit_predict(G)
            for method, lab in res.items():
                m = clustering_metrics(y, lab)
                rows.append({"dataset": name, "group": s.group, "keep": keep,
                             "seed": seed, "method": method, "K": K, "basis_size": size,
                             "ARI": m["ARI"], "NMI": m["NMI"]})
    print(f"  [{name}] clustering done", flush=True)
    return rows


def run_clustering(names=ucr.DATASETS) -> pd.DataFrame:
    out = Parallel(n_jobs=N_JOBS)(delayed(clustering_one)(n) for n in names)
    return pd.DataFrame([r for rows in out for r in rows])


# ==========================================================================
def run(args=None) -> None:
    """All stages, or only those named in ``args`` (``benchmark``, ``fewshot``,
    ``irregular``, ``clustering``; ``v10`` forces the theory rerun). A stage
    that is skipped reads its predecessor's CSV."""
    t0 = time.time()
    pd.DataFrame([ucr.describe(ucr.load(n)) for n in ucr.DATASETS]).to_csv(
        RESULTS / "ucr_datasets.csv", index=False)

    if not (RESULTS / "v10_learning_curve.csv").exists() or "v10" in (args or ()):
        from lrdsr.theory.classification import run_v10
        print("\n--- V10: the learning curve ---", flush=True)
        run_v10()

    stages = set(args or ()) & {"benchmark", "fewshot", "irregular", "clustering"}
    stages = stages or {"benchmark", "fewshot", "irregular", "clustering"}

    if "benchmark" in stages:
        print("\n--- benchmark ---", flush=True)
        bench, cfg = run_benchmark()
        bench.to_csv(RESULTS / "ucr_benchmark.csv", index=False)
        cfg.to_csv(RESULTS / "ucr_configs.csv", index=False)
    else:
        # "None" is a value here (no nuisance), not a missing entry
        bench = pd.read_csv(RESULTS / "ucr_benchmark.csv")
        cfg = pd.read_csv(RESULTS / "ucr_configs.csv", keep_default_na=False)
    piv = bench.pivot_table(index=["group", "dataset"], columns="method", values="error")
    print(piv.round(3).to_string())

    if "fewshot" in stages:
        print("\n--- few-shot ---", flush=True)
        fs = run_fewshot()
        fs.to_csv(RESULTS / "ucr_fewshot.csv", index=False)
        print(fs.groupby(["group", "m", "method"]).error.mean().unstack().round(3)
              .to_string())

    if "irregular" in stages:
        print("\n--- irregular sampling ---", flush=True)
        irr = run_irregular(cfg)
        irr.to_csv(RESULTS / "ucr_irregular.csv", index=False)
        print(irr.groupby(["group", "keep", "method"]).error.mean().unstack().round(3)
              .to_string())

    if "clustering" in stages:
        print("\n--- clustering ---", flush=True)
        cl = run_clustering()
        cl.to_csv(RESULTS / "ucr_clustering.csv", index=False)
        print(cl.groupby(["group", "keep", "method"]).ARI.mean().unstack().round(3)
              .to_string())
    print(f"\n[classify] {(time.time() - t0) / 60:.1f} min; CSVs -> {RESULTS}")


if __name__ == "__main__":
    import sys
    run(sys.argv[1:])
