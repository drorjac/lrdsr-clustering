"""Wind power curves: laws on windows whose designs all differ.

    python -m experiments wind

Kelmarsh SCADA (``sources``), six-hour turbine blocks (``windows``), scored
against physical proxies -- colder air, wake from a neighbour, and night as a
negative control -- because the archive has no operator labels.

==================  =======================================================
``transfer``        per proxy: train on turbines 1-3, test on 4-6, then
                    the reverse. The law classifier (Nystrom basis in wind
                    speed; size and ``L`` by CV on the training turbines),
                    mechanism features into logistic regression, and the
                    method of bins into logistic regression, 1-NN and
                    nearest centroid. Balanced error (the wake classes are
                    4:1), so no rule wins by predicting the majority.
``ceiling``         V1 at each test block's own wind speeds: the error the
                    two proxy laws would allow if the proxy were exactly a
                    change of law with independent noise. Far below the
                    achieved error means the proxy is not one law per class.
``coverage``        the prediction declared before the run: laws beat the
                    method of bins most on blocks that saw few wind-speed
                    bins. Balanced error by coverage tercile.
``physics``         the cold and warm laws below rated, against the air
                    density ratio the block temperatures imply.
``clusters``        no labels: mechanism K-means (K = 2) and K-means on the
                    binned curves, and how each partition lines up with
                    each proxy and with the season.
``online``          each turbine streamed in time order (warm start on its
                    first 300 blocks, label-free): births and when.
==================  =======================================================

Writes ``results/wind/``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.wind import windows as W
from experiments.wind.windows import PROXIES, bin_profile, build
from lrdsr import paths
from lrdsr.core.classify import LawClassifier, MechanismFeatures
from lrdsr.core.kernel import NystromBasis
from lrdsr.core.online import OnlineLRDSR
from lrdsr.protocol import REPORT_SEEDS

RESULTS = paths.results_dir("wind", figs=False)
SIZES = (8, 16)
LAWS = (1, 2, 4)
SPLITS = ((1, 2, 3), (4, 5, 6))
WARM_BLOCKS = 300


def _berr(y, p) -> float:
    return 1.0 - balanced_accuracy_score(y, p)


def _sub(b, idx):
    return [b.X[i] for i in idx], [b.y[i] for i in idx]


def _law(size, L, seed=0):
    return LawClassifier(basis=NystromBasis(size), nuisance=None, laws_per_class=L,
                         priors="uniform", random_state=seed)


def cv_law(X, y, lab, seed=0):
    folds = StratifiedKFold(3, shuffle=True, random_state=seed)
    best, best_s = None, -1.0
    for size in SIZES:
        for L in LAWS:
            s = []
            for tr, va in folds.split(np.zeros(len(lab)), lab):
                m = _law(size, L, seed).fit([X[i] for i in tr], [y[i] for i in tr], lab[tr])
                s.append(balanced_accuracy_score(
                    lab[va], m.predict([X[i] for i in va], [y[i] for i in va])))
            if np.mean(s) > best_s + 1e-9:
                best, best_s = (size, L), float(np.mean(s))
    return best


def _logit(S, lab, St):
    g = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=5000, class_weight="balanced")),
        {"logisticregression__C": [0.01, 0.1, 1, 10]}, cv=3)
    return g.fit(S, lab).predict(St)


def transfer_one(b, proxy, train_t, seed=REPORT_SEEDS[0]):
    lab_all = b.labels(proxy)
    turb = b.meta["turbine"].to_numpy()
    tr = np.flatnonzero(np.isin(turb, train_t) & (lab_all >= 0))
    te = np.flatnonzero(~np.isin(turb, train_t) & (lab_all >= 0))
    Xtr, ytr = _sub(b, tr)
    Xte, yte = _sub(b, te)
    ltr, lte = lab_all[tr], lab_all[te]
    size, L = cv_law(Xtr, ytr, ltr, seed)
    pred = {"law": _law(size, L, seed).fit(Xtr, ytr, ltr).predict(Xte, yte)}
    one = _law(size, 1, seed).fit(Xtr, ytr, ltr)
    pred["law_L1"] = one.predict(Xte, yte)
    fm = MechanismFeatures(basis=NystromBasis(size), nuisance=None, mode="solve")
    pred["mech_logistic"] = _logit(fm.fit_transform(Xtr, ytr), ltr, fm.transform(Xte, yte))
    Btr, Bte = bin_profile(Xtr, ytr), bin_profile(Xte, yte)
    pred["bins_logistic"] = _logit(Btr, ltr, Bte)
    pred["bins_1nn"] = KNeighborsClassifier(1).fit(Btr, ltr).predict(Bte)
    pred["bins_centroid"] = NearestCentroid().fit(Btr, ltr).predict(Bte)
    # V1 at each test block's own design, with the one-law-per-class fit
    g = [one.design_.B(x) @ (one.coef_[1] - one.coef_[0]) for x in Xte]
    s = float(one.sigma_[0])
    ceil = np.array([norm.sf(np.sqrt(np.sum(v ** 2)) / (2 * s)) for v in g])
    cov = b.meta["coverage"].to_numpy()[te]
    rows = []
    for method, p in pred.items():
        rows.append({"proxy": proxy, "train_turbines": "".join(map(str, train_t)),
                     "method": method, "balanced_error": _berr(lte, p),
                     "n_test": len(te), "share_1": float(lte.mean()),
                     "basis_size": size, "L": L, "ceiling_v1": float(ceil.mean())})
    tert = pd.qcut(cov, 3, labels=["narrow", "middle", "wide"], duplicates="drop")
    cov_rows = []
    for t in ("narrow", "middle", "wide"):
        m = np.asarray(tert == t)
        if m.sum() == 0 or len(np.unique(lte[m])) < 2:
            continue
        for method in ("law", "bins_logistic", "bins_1nn"):
            cov_rows.append({"proxy": proxy, "train_turbines": "".join(map(str, train_t)),
                             "coverage": t, "coverage_bins_mean": float(cov[m].mean()),
                             "method": method,
                             "balanced_error": _berr(lte[m], pred[method][m]),
                             "n": int(m.sum())})
    return rows, cov_rows


def run_transfer(b):
    jobs = [(p, t) for p in PROXIES for t in SPLITS]
    out = Parallel(n_jobs=len(jobs))(delayed(transfer_one)(b, p, t) for p, t in jobs)
    return (pd.DataFrame([r for rows, _ in out for r in rows]),
            pd.DataFrame([r for _, rows in out for r in rows]))


def run_physics(b):
    """Cold and warm power curves below rated against the air-density ratio.

    Below rated, power scales with air density; at one pressure that is
    ``T_warm / T_cold`` (kelvin), a single number for every wind speed. The
    ratio is read two ways: from the binned means of every sample (the
    method of bins, the physics reference) and from the two fitted laws.
    The Nystrom centres sit at quantiles of the pooled wind speed, so the
    laws resolve the sparse high-wind end worse than the bins do; the
    columns say which is which.
    """
    lab = b.labels("cold")
    idx = np.flatnonzero(lab >= 0)
    m = _law(16, 1).fit(*_sub(b, idx), lab[idx])
    edges = np.arange(4.0, 12.01, 0.5)
    mid = 0.5 * (edges[1:] + edges[:-1])
    F = m.design_.B(mid[:, None]) @ m.coef_.T                # classes 0 = warm, 1 = cold
    x = np.concatenate([b.X[i][:, 0] for i in idx])
    y = np.concatenate([b.y[i] for i in idx])
    c = np.repeat(lab[idx], [len(b.y[i]) for i in idx])
    k = np.digitize(x, edges) - 1
    binned = np.array([[y[(k == j) & (c == cls)].mean() for cls in (0, 1)]
                       for j in range(len(mid))])
    # post hoc, after the quantile law was seen to be erratic above 9 m/s:
    # the same law with centres spread evenly over the wind-speed range
    mu = LawClassifier(basis=NystromBasis(16, spacing="uniform"), nuisance=None,
                       priors="uniform").fit(*_sub(b, idx), lab[idx])
    Fu = mu.design_.B(mid[:, None]) @ mu.coef_.T
    t = b.meta.loc[idx]
    Tc = float(t.loc[t["cold"] == 1, "temp"].mean()) + 273.15
    Tw = float(t.loc[t["cold"] == 0, "temp"].mean()) + 273.15
    return pd.DataFrame({"ws": mid, "warm_bins": binned[:, 0], "cold_bins": binned[:, 1],
                         "bins_ratio": binned[:, 1] / binned[:, 0],
                         "warm_law": F[:, 0], "cold_law": F[:, 1],
                         "law_ratio": F[:, 1] / F[:, 0],
                         "law_uniform_ratio": Fu[:, 1] / Fu[:, 0],
                         "density_ratio": Tw / Tc,
                         "T_cold_C": Tc - 273.15, "T_warm_C": Tw - 273.15})


def run_clusters(b):
    rows = []
    B = bin_profile(b.X, b.y)
    S = MechanismFeatures(basis=NystromBasis(16), nuisance=None, mode="solve").fit_transform(
        b.X, b.y)
    season = np.isin(b.meta["month"].to_numpy(), (11, 12, 1, 2, 3)).astype(int)
    for seed in REPORT_SEEDS:
        for name, Z in (("mech_kmeans", S), ("bins_kmeans", B)):
            lab = KMeans(2, n_init=10, random_state=seed).fit_predict(Z)
            row = {"seed": seed, "method": name, "cluster_share": float(lab.mean())}
            for proxy in PROXIES:
                ref = b.labels(proxy)
                ok = ref >= 0
                row[f"ARI_{proxy}"] = adjusted_rand_score(ref[ok], lab[ok])
            row["ARI_winter"] = adjusted_rand_score(season, lab)
            row["ARI_calm"] = adjusted_rand_score(
                (b.meta["ws_mean"] < b.meta["ws_mean"].median()).astype(int), lab)
            rows.append(row)
    return pd.DataFrame(rows)


def raw_effects() -> dict:
    """The raw-data facts the design rests on, measured rather than quoted.

    Share of samples without the anemometer standard deviation (why
    turbulence is not a proxy); and, on generating samples at 5-9 m/s in
    0.5 m/s bins, the largest night/day power ratio departure (why night is
    a negative control) and the high/low turbulence power ratio at 5 m/s
    where the column exists.
    """
    from experiments.wind.sources import load_scada
    d = load_scada()
    out = {"ws_sd_missing": float(d["ws_sd"].isna().mean())}
    g = d[d["ws"].between(4.75, 9.25) & (d["power"] > 0)].copy()
    g["bin"] = (g["ws"] * 2).round() / 2
    h = g["time"].dt.hour
    night = (h >= 20) | (h < 6)
    r = g.groupby(["bin", night])["power"].mean().unstack()
    out["night_day_max_departure"] = float((r[True] / r[False] - 1).abs().max())
    t = g.dropna(subset=["ws_sd"])
    t = t.assign(hi=(t["ws_sd"] / t["ws"]) > (t["ws_sd"] / t["ws"]).median())
    r2 = t[t["bin"] == 5.0].groupby("hi")["power"].mean()
    out["turbulence_ratio_5ms"] = float(r2[True] / r2[False])
    return out


def run_online(b, seed=REPORT_SEEDS[0]):
    rows = []
    for turb in sorted(b.meta["turbine"].unique()):
        idx = np.flatnonzero(b.meta["turbine"].to_numpy() == turb)
        X, y = _sub(b, idx)
        on = OnlineLRDSR(basis=NystromBasis(16), novelty_alpha=1e-3)
        on.warm_start(X[:WARM_BLOCKS], y[:WARM_BLOCKS], n_clusters=2, random_state=seed)
        tl = on.fit_stream(X[WARM_BLOCKS:], y[WARM_BLOCKS:])
        meta = b.meta.iloc[idx[WARM_BLOCKS:]].reset_index(drop=True)
        wsq = b.meta["ws_mean"].rank(pct=True).to_numpy()[idx[WARM_BLOCKS:]]
        for _, r in tl[tl["spawned"].notna()].iterrows():
            rows.append({"turbine": int(turb), "born_at": str(meta.loc[r.name, "block"]),
                         "month": int(meta.loc[r.name, "month"]),
                         "regime": int(r["spawned"]),
                         "ws_mean": float(meta.loc[r.name, "ws_mean"]),
                         "ws_percentile": float(wsq[r.name]),
                         "temp": float(meta.loc[r.name, "temp"])})
        rows.append({"turbine": int(turb), "born_at": "", "month": -1, "regime": -1,
                     "streamed": len(tl), "final_regimes": on.n_clusters,
                     "buffered_share": float((tl["label"] < 0).mean())})
    return pd.DataFrame(rows)


def run(args=None) -> None:
    t0 = time.time()
    b = build()
    pd.DataFrame([{"blocks": len(b.X), "first_block": str(b.meta["block"].min()),
                   "last_block": str(b.meta["block"].max()),
                   "median_bins_covered": float(b.meta["coverage"].median()),
                   "bins_total": len(W.BIN_EDGES) - 1,
                   **{f"{p}_{k}": int((b.labels(p) == k).sum())
                      for p in PROXIES for k in (0, 1)}, **raw_effects()}]).to_csv(
        RESULTS / "wind_coverage.csv", index=False)
    print(f"  {len(b.X)} blocks", flush=True)

    tr, cv = run_transfer(b)
    tr.to_csv(RESULTS / "wind_transfer.csv", index=False)
    cv.to_csv(RESULTS / "wind_coverage_terciles.csv", index=False)
    print(tr.groupby(["proxy", "method"])[["balanced_error", "ceiling_v1"]].mean()
          .round(3).to_string())
    print(cv.groupby(["proxy", "coverage", "method"]).balanced_error.mean().unstack()
          .round(3).to_string())

    ph = run_physics(b)
    ph.to_csv(RESULTS / "wind_physics.csv", index=False)
    print(ph.round(3).to_string(index=False))

    cl = run_clusters(b)
    cl.to_csv(RESULTS / "wind_clusters.csv", index=False)
    print(cl.groupby("method").mean(numeric_only=True).round(3).to_string())

    on = run_online(b)
    on.to_csv(RESULTS / "wind_online.csv", index=False)
    print(on.to_string(index=False))
    print(f"\n[wind] {(time.time() - t0) / 60:.1f} min; CSVs -> {RESULTS}")


if __name__ == "__main__":
    run()
