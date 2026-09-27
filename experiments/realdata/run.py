"""Real data: do the daily laws of two count series split by day type?

    python -m experiments.realdata.run

Two public hourly series (``sources``), one day per window (``windows``),
and every estimator in the project run on them unchanged. The reference is
the calendar day type -- a **proxy**, used for scoring only.

==================  =======================================================
``methods``         K = 2: LR-DSR (hard, ``alpha_geom`` 0 and 0.25),
                    SoftLRDSR (Gaussian, Student-t, and Gaussian on a
                    4-harmonic Fourier basis), K-means in mechanism space
                    (library and Fourier), the seven geometry baselines on
                    window summaries, and K-means on the raw 24-hour profile
``k_selection``     K = 1..8 by SoftLRDSR BIC, no label consulted, and what
                    the clusters at K = 2, 3 and the chosen K are in calendar,
                    weather and season
``laws``            each regime's recovered daily curve and expression,
                    beside the calendar-mean profiles
``rho``             the separation measured without labels, and the error
                    ceiling it implies, beside the error achieved
``online``          OnlineLRDSR warm-started on the first 60 days, then
                    streamed day by day in calendar order
``disagreements``  every day the law-based partition and the calendar
                    disagree on, by date
``learned_loss``    which noise family real residuals pick
``gaps``            the 616 traffic days with sensor gaps, scored at their
                    observed hours (``experiments.realdata.gaps``)
==================  =======================================================

A fact that shapes every number here: on these series every window has the
**same design** -- the 24 hours. With a fixed design, mechanism space over
a basis that spans the day is a linear isometry of the centred profile, so
K-means on the raw profile *is* mechanism-space K-means (``mech_fourier``
and ``raw_profile_kmeans`` agree exactly). The advantage the method has on
the simulator -- windows that look different because their inputs differ --
has nothing to act on, and the raw-profile baseline is the honest bar.

Writes ``results/realdata/realdata_*.csv``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm
from sklearn.cluster import KMeans

from experiments.common.fitting import fit_lrdsr, window_features
from experiments.realdata.windows import HOURS, DayWindows, load
from lrdsr import paths
from lrdsr.core.baselines import geometry_baselines
from lrdsr.core.evaluation import aligned_accuracy, clustering_metrics
from lrdsr.core.mechanism_space import (
    mechanism_features,
    mechanism_init,
    mechanism_noise,
    rho_from_partition,
    separation_unlabelled,
)
from lrdsr.core.model import GroupedDCSR
from lrdsr.core.online import OnlineLRDSR
from lrdsr.core.soft import SoftLRDSR
from lrdsr.protocol import REPORT_SEEDS

RESULTS = paths.results_dir("realdata", figs=False)
DATASETS = ("bike", "traffic")
K_GRID = (1, 2, 3, 4, 5, 6, 7, 8)
WARMUP_DAYS = 60
N_HARMONICS = 4
X_NAMES = ["x"]


def fourier(X_flat: np.ndarray, H: int = N_HARMONICS) -> np.ndarray:
    """Intercept plus ``H`` harmonics of the daily cycle: a basis that spans it.

    The fast library has one full harmonic and ``sin 2x`` only, so a
    two-peak commuting day is at the edge of what it can express; this is
    the basis a practitioner would reach for, used as a second arm and never
    in place of the library.
    """
    x = np.asarray(X_flat, dtype=float)[:, 0]
    cols = [np.ones_like(x)]
    for h in range(1, H + 1):
        cols += [np.sin(h * x), np.cos(h * x)]
    return np.column_stack(cols)


FOURIER_NAMES = ["1"] + [f"{f}({h}x)" for h in range(1, N_HARMONICS + 1)
                         for f in ("sin", "cos")]


def _err(ref, pred) -> float:
    return 1.0 - aligned_accuracy(ref, pred)


def _to_reference(ref: np.ndarray, pred: np.ndarray) -> dict:
    """Map each predicted cluster to a reference class (Hungarian when square)."""
    pred = np.asarray(pred, int)
    ks = np.unique(pred[pred >= 0])
    C = np.array([[np.sum((pred == k) & (ref == c)) for c in (0, 1)] for k in ks])
    if len(ks) == 2:
        r, c = linear_sum_assignment(-C)
        return {int(ks[i]): int(j) for i, j in zip(r, c, strict=True)}
    return {int(k): int(C[i].argmax()) for i, k in enumerate(ks)}


# ==========================================================================
# K = 2: every method
# ==========================================================================
def _methods(d: DayWindows, seed: int) -> dict[str, np.ndarray]:
    X, y = d.X_seq, d.y_seq
    Z, names = window_features(X, y)
    out = {}
    for a in (0.0, 0.25):
        out[f"lrdsr_alpha{a:g}"] = fit_lrdsr(Z, names, X, y, X_NAMES, 2, seed,
                                             alpha_geom=a).labels
    out["soft_gaussian"] = SoftLRDSR(2, feature_names=X_NAMES,
                                     random_state=seed).fit(X, y).labels
    out["soft_student_t"] = SoftLRDSR(2, noise="student_t", feature_names=X_NAMES,
                                      random_state=seed).fit(X, y).labels
    out["soft_fourier"] = SoftLRDSR(2, basis=fourier, random_state=seed).fit(X, y).labels
    out["mech_kmeans"] = mechanism_init(X, y, 2, seed=seed, feature_names=X_NAMES)
    out["mech_fourier"] = mechanism_init(X, y, 2, seed=seed, basis=fourier)
    for name, lab in geometry_baselines(Z, 2, seed=seed).items():
        out["geom_" + name] = lab
    out["raw_profile_kmeans"] = KMeans(2, n_init=30, random_state=seed).fit_predict(
        d.raw_profile)
    return out


FAMILY = {"lrdsr": "LR-DSR", "soft": "soft EM", "mech": "mechanism K-means",
          "geom": "geometry", "raw": "raw profile"}


def run_methods(datasets) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for d in datasets:
        for seed in REPORT_SEEDS:
            t0 = time.time()
            for method, lab in _methods(d, seed).items():
                m = clustering_metrics(d.reference, lab)
                rows.append({"dataset": d.name, "seed": seed, "method": method,
                             "family": FAMILY[method.split("_")[0]],
                             "ARI": m["ARI"], "NMI": m["NMI"],
                             "matched_error": _err(d.reference, lab)})
            print(f"  [{d.name}] seed {seed}: {time.time() - t0:.1f}s", flush=True)
    per_seed = pd.DataFrame(rows)
    summ = (per_seed.groupby(["dataset", "method", "family"], sort=False)
            .agg(ARI_mean=("ARI", "mean"), ARI_sd=("ARI", "std"),
                 NMI_mean=("NMI", "mean"), error_mean=("matched_error", "mean"),
                 error_min=("matched_error", "min"), error_max=("matched_error", "max"))
            .reset_index())
    # the best geometry baseline, per dataset, as its own row (chosen per seed
    # would be selecting on the reference; this is the best MEAN, reported)
    best = []
    for ds, g in summ[summ.family == "geometry"].groupby("dataset"):
        b = g.sort_values("ARI_mean", ascending=False).iloc[0].copy()
        b["method"] = "geom_best (" + b["method"][5:] + ")"
        best.append(b)
    summ = pd.concat([summ, pd.DataFrame(best)], ignore_index=True)
    return summ, per_seed


# ==========================================================================
# K without labels
# ==========================================================================
def _bic(res, W: int) -> float:
    K, p = res.coef.shape
    n_par = K * p + K + (K - 1)
    return float(-2.0 * res.loglik + n_par * np.log(W))


def run_k_selection(datasets, seed: int = REPORT_SEEDS[0]):
    """K from BIC, with no label consulted.

    BIC counts every hour as an independent sample. Hours within a day are
    strongly autocorrelated, so the evidence for one more regime is
    overcounted roughly by the ratio of 24 to the effective sample size, and
    a real series has more than two ways a day can go (weather, season,
    school holidays). Expect BIC to keep asking for more clusters; what the
    extra clusters *are* is the informative part, hence the contingency.
    """
    rows, cont, chosen = [], [], {}
    for d in datasets:
        for basis_name, basis in (("library", None), ("fourier", fourier)):
            # K = 1..8 although K = 2 is the hypothesis: the question is
            # whether the likelihood ever says "enough", and on these series
            # it does not (see the module docstring of run_k_selection).
            fits = {}
            for K in K_GRID:
                # best of the reporting seeds by LIKELIHOOD -- no label involved
                cands = [SoftLRDSR(K, basis=basis, feature_names=X_NAMES,
                                   random_state=s).fit(d.X_seq, d.y_seq)
                         for s in REPORT_SEEDS]
                res = max(cands, key=lambda r: r.loglik)
                fits[K] = res
                rows.append({"dataset": d.name, "basis": basis_name, "K": K,
                             "loglik": res.loglik, "BIC": _bic(res, len(d.y_seq)),
                             "ARI_vs_daytype": clustering_metrics(d.reference, res.labels)["ARI"]
                             if K > 1 else 0.0})
            g = pd.DataFrame([r for r in rows if r["dataset"] == d.name
                              and r["basis"] == basis_name])
            kbest = int(g.loc[g.BIC.idxmin(), "K"])
            chosen[(d.name, basis_name)] = (kbest, fits[kbest], fits)
            for K in sorted({2, 3, kbest}):
                cont += _contingency(d, fits[K].labels, basis_name, K, kbest)
    df = pd.DataFrame(rows)
    df["chosen"] = False
    for (ds, b), (k, _, _) in chosen.items():
        df.loc[(df.dataset == ds) & (df.basis == b) & (df.K == k), "chosen"] = True
    return df, pd.DataFrame(cont), chosen


def _contingency(d: DayWindows, labels, basis, K, kbest) -> list[dict]:
    """Cluster x (day type, weekday, season, year, weather) counts, plus a profile."""
    m = d.meta.assign(cluster=labels)
    out = []
    variables = ["day_type", "weekday", "season", "year"]
    if "weather" in m:
        variables.append("weather")
    if "precip" in m:
        m = m.assign(precip_class=pd.cut(m["precip"], [-1, 0.0, 2.0, 1e9],
                                         labels=["dry", "light", "wet"]).astype(str))
        variables.append("precip_class")
    for v in variables:
        tab = m.groupby(["cluster", v], observed=True).size()
        for (k, val), n in tab.items():
            out.append({"dataset": d.name, "basis": basis, "K": K,
                        "is_chosen_K": K == kbest, "cluster": int(k),
                        "variable": v, "value": str(val), "n_days": int(n)})
    for k, g in m.groupby("cluster"):
        out.append({"dataset": d.name, "basis": basis, "K": K, "is_chosen_K": K == kbest,
                    "cluster": int(k), "variable": "summary",
                    "value": (f"n={len(g)}; working={g.working.mean():.2f}; "
                              f"level={g.level.mean():.2f}; "
                              f"dates={g.date.min():%Y-%m-%d}..{g.date.max():%Y-%m-%d}; "
                              f"holidays={','.join(sorted(set(g.holiday.astype(str)) - {''}))[:120]}"),
                    "n_days": int(len(g))})
    return out


# ==========================================================================
# laws, separation
# ==========================================================================
def run_laws(datasets, seed: int = REPORT_SEEDS[0]):
    hours = np.arange(HOURS)
    grid = (2 * np.pi * hours / HOURS)[:, None]
    curves, exprs = [], []
    for d in datasets:
        Z, names = window_features(d.X_seq, d.y_seq)
        hard = fit_lrdsr(Z, names, d.X_seq, d.y_seq, X_NAMES, 2, seed, alpha_geom=0.0)
        soft = SoftLRDSR(2, basis=fourier, random_state=seed).fit(d.X_seq, d.y_seq)
        soft.term_names = FOURIER_NAMES
        for method, lab, predict, expr in (
                ("lrdsr_alpha0", hard.labels,
                 lambda k: hard.models[k].predict(grid), lambda k: hard.models[k].expression()),
                ("soft_fourier", soft.labels, lambda k: fourier(grid) @ soft.coef[k],
                 lambda k: soft.expressions()[k])):
            mp = _to_reference(d.reference, lab)
            for k in range(2):
                ref_class = "working" if mp.get(k, 0) == 1 else "off"
                sel = lab == k
                for h, v in zip(hours, predict(k), strict=True):
                    curves.append({"dataset": d.name, "method": method, "regime": ref_class,
                                   "hour": int(h), "value": float(v)})
                exprs.append({"dataset": d.name, "method": method, "regime": ref_class,
                              "expression": expr(k), "n_days": int(sel.sum()),
                              "share_working": float(d.reference[sel].mean())})
        for c, nm in ((1, "working"), (0, "off")):
            prof = d.y_seq[d.reference == c].mean(axis=0)
            for h, v in zip(hours, prof, strict=True):
                curves.append({"dataset": d.name, "method": "calendar_mean", "regime": nm,
                               "hour": int(h), "value": float(v)})
    return pd.DataFrame(curves), pd.DataFrame(exprs)


def run_rho(datasets, summary: pd.DataFrame, seed: int = REPORT_SEEDS[0]) -> pd.DataFrame:
    """Label-free separation, and the ceiling it implies, per basis.

    Two warnings go with every number. Hours within a day are not
    independent -- the residual is smooth -- so ``sigma`` per sample
    understates the noise a window really carries, and ``n = 24`` overstates
    the evidence. And the trace estimator charges *all* spread in law to the
    regime gap (seasonal shape changes, weather), so the rho it gives is an
    upper bound. The implied error is therefore a loose **lower** bound;
    the useful comparison is its order of magnitude against the achieved.
    """
    rows = []
    for d in datasets:
        best = summary[(summary.dataset == d.name)
                       & ~summary.method.str.startswith("geom_best")]
        best = best.sort_values("error_mean").iloc[0]
        for basis_name, kw in (("library", {"feature_names": X_NAMES}),
                               ("fourier", {"basis": fourier})):
            S = mechanism_features(d.X_seq, d.y_seq, **kw)
            sigma = mechanism_noise(d.X_seq, d.y_seq, **kw)
            u = separation_unlabelled(S, sigma, HOURS)
            lab = mechanism_init(d.X_seq, d.y_seq, 2, seed=seed, **kw)
            rho_p = rho_from_partition(S, lab, sigma, HOURS)
            rho_cal = rho_from_partition(S, d.reference, sigma, HOURS)
            rows.append({
                "dataset": d.name, "basis": basis_name, "sigma": sigma,
                "n_dims": u["n_dims"], "rho_trace": u["rho"],
                "rho_partition": rho_p, "rho_calendar": rho_cal,
                "ceiling_trace": float(norm.sf(np.sqrt(HOURS * u["rho"]) / 2)),
                "ceiling_partition": float(norm.sf(np.sqrt(HOURS * rho_p) / 2)),
                "ceiling_calendar": float(norm.sf(np.sqrt(HOURS * rho_cal) / 2)),
                "pi_from_split": u["pi_from_split"],
                "best_method": best["method"], "best_error": float(best["error_mean"]),
            })
    return pd.DataFrame(rows)


# ==========================================================================
# real time
# ==========================================================================
def run_online(datasets, batch_per_seed: pd.DataFrame, seed: int = REPORT_SEEDS[0]):
    """Warm-start on ``WARMUP_DAYS`` days, then assign each later day as it comes.

    Scoring maps each online cluster to its majority day type over the
    stream (many-to-one once regimes are born); days held in the novelty
    buffer are counted as errors, since they got no label when they arrived.
    """
    rows, summ = [], []
    for d in datasets:
        X, y, ref = d.X_seq, d.y_seq, d.reference
        for alpha in (1e-3, 1e-6):
            on = OnlineLRDSR(feature_names=X_NAMES, novelty_alpha=alpha).warm_start(
                X[:WARMUP_DAYS], y[:WARMUP_DAYS], n_clusters=2, random_state=seed)
            tl = on.fit_stream(X[WARMUP_DAYS:], y[WARMUP_DAYS:],
                               true_labels_for_eval=ref[WARMUP_DAYS:])
            meta = d.meta.iloc[WARMUP_DAYS:].reset_index(drop=True)
            # ``map`` (the MAP regime) would shadow DataFrame.map in the CSV
            tl = tl.rename(columns={"map": "map_regime"})
            tl["date"] = meta["date"].dt.strftime("%Y-%m-%d")
            tl["day_type"] = meta["day_type"]
            mp = _to_reference(ref[WARMUP_DAYS:], tl["label"].to_numpy())
            tl["as_daytype"] = tl["label"].map(lambda k, mp=mp: mp.get(int(k), -1))
            tl["correct"] = tl["as_daytype"] == tl["truth"]
            tl["rolling_accuracy_30"] = tl["correct"].rolling(30, min_periods=10).mean()
            tl["dataset"] = d.name
            tl["novelty_alpha"] = alpha
            rows.append(tl)
            births = tl[tl["spawned"].notna()]
            batch = batch_per_seed[(batch_per_seed.dataset == d.name)
                                   & (batch_per_seed.method == "lrdsr_alpha0")]
            born = []
            for _, b in births.iterrows():
                k = int(b["spawned"])
                g = tl[tl["label"] == k]
                born.append(f"regime {k} born {b['date']} "
                            f"({len(g)} days: " + ", ".join(
                                f"{t}={n}" for t, n in g["day_type"].value_counts().items())
                            + f"; {g['date'].min()}..{g['date'].max()})")
            summ.append({
                "dataset": d.name, "novelty_alpha": alpha, "warmup_days": WARMUP_DAYS,
                "streamed_days": len(tl), "final_regimes": on.n_clusters,
                "births": len(births), "buffered_days": int((tl["label"] < 0).sum()),
                "online_accuracy": float(tl["correct"].mean()),
                "online_ARI": clustering_metrics(tl["truth"], tl["label"])["ARI"],
                "batch_error_lrdsr_all_days": float(batch.matched_error.mean()),
                "birth_log": " | ".join(born),
            })
    return pd.concat(rows, ignore_index=True), pd.DataFrame(summ)


def run_disagreements(datasets, seed: int = REPORT_SEEDS[0]) -> pd.DataFrame:
    """Every day where the best law-based partition and the calendar disagree.

    At a label-free separation this large (``realdata_rho.csv``) the noise
    ceiling is ~0, so the residual error is not noise: it is days whose
    calendar type is not their behaviour. Listing them is the check.
    """
    rows = []
    for d in datasets:
        lab = SoftLRDSR(2, basis=fourier, random_state=seed).fit(d.X_seq, d.y_seq).labels
        mp = _to_reference(d.reference, lab)
        as_type = np.array([mp[int(k)] for k in lab])
        for i in np.flatnonzero(as_type != d.reference):
            m = d.meta.iloc[i]
            rows.append({"dataset": d.name, "date": f"{m['date']:%Y-%m-%d}",
                         "weekday": int(m["weekday"]), "day_type": m["day_type"],
                         "holiday": m["holiday"],
                         "behaves_like": "working" if as_type[i] == 1 else "off"})
    return pd.DataFrame(rows)


def run_learned_loss(datasets, seed: int = REPORT_SEEDS[0]) -> pd.DataFrame:
    """GroupedDCSR with ``loss='learned'``: what noise family do residuals pick?"""
    rows = []
    for d in datasets:
        Z, _ = window_features(d.X_seq, d.y_seq)
        for loss in ("huber", "learned"):
            model = GroupedDCSR(
                n_clusters=2, alpha_geom=0.0, beta_complexity=0.002, max_iter=10,
                tol=0.01, backend="fast", backend_kwargs={"max_terms": 5},
                random_state=seed, min_windows_per_cluster=4, init="mechanism",
                geom_metric="mahalanobis", score_mode="cross_fit",
                residual_scale="global", loss=loss)
            res = model.fit(d.X_seq, d.y_seq, Z, feature_names=X_NAMES)
            ll = getattr(model, "learned_loss_", None)
            rows.append({"dataset": d.name, "loss": loss,
                         "picked": ll.loss if ll else "huber",
                         "nu": float(ll.params.get("nu", np.nan)) if ll else np.nan,
                         "scale": float(ll.scale) if ll else np.nan,
                         "loglik_per_sample": float(ll.loglik) if ll else np.nan,
                         "iterations": len(res.history),
                         "ARI": clustering_metrics(d.reference, res.labels)["ARI"],
                         "matched_error": _err(d.reference, res.labels)})
    return pd.DataFrame(rows)


# ==========================================================================
def run(args=None) -> None:
    t0 = time.time()
    datasets = [load(n) for n in DATASETS]
    pd.DataFrame([d.coverage for d in datasets]).to_csv(
        RESULTS / "realdata_coverage.csv", index=False)
    for d in datasets:
        print(f"  {d.name}: {d.coverage}")

    print("\n--- K = 2, every method ---", flush=True)
    summ, per_seed = run_methods(datasets)
    summ.to_csv(RESULTS / "realdata_summary.csv", index=False)
    per_seed.to_csv(RESULTS / "realdata_per_seed.csv", index=False)
    print(summ[["dataset", "method", "ARI_mean", "error_mean"]].round(3).to_string(index=False))

    print("\n--- K without labels ---", flush=True)
    ksel, cont, _ = run_k_selection(datasets)
    ksel.to_csv(RESULTS / "realdata_k_selection.csv", index=False)
    cont.to_csv(RESULTS / "realdata_contingency.csv", index=False)
    print(ksel.round(3).to_string(index=False))
    print(cont[cont.variable == "summary"].to_string(index=False))

    print("\n--- laws ---", flush=True)
    curves, exprs = run_laws(datasets)
    curves.to_csv(RESULTS / "realdata_laws.csv", index=False)
    exprs.to_csv(RESULTS / "realdata_expressions.csv", index=False)
    print(exprs.to_string(index=False))

    print("\n--- separation without labels ---", flush=True)
    rho = run_rho(datasets, summ)
    rho.to_csv(RESULTS / "realdata_rho.csv", index=False)
    print(rho.round(4).to_string(index=False))

    print("\n--- real time ---", flush=True)
    tl, osum = run_online(datasets, per_seed)
    tl.to_csv(RESULTS / "realdata_online.csv", index=False)
    osum.to_csv(RESULTS / "realdata_online_summary.csv", index=False)
    print(osum.round(3).to_string(index=False))

    print("\n--- where the calendar and the law disagree ---", flush=True)
    dis = run_disagreements(datasets)
    dis.to_csv(RESULTS / "realdata_disagreements.csv", index=False)
    print(dis.to_string(index=False))

    print("\n--- learned loss ---", flush=True)
    ll = run_learned_loss(datasets)
    ll.to_csv(RESULTS / "realdata_learned_loss.csv", index=False)
    print(ll.round(3).to_string(index=False))

    print("\n--- the traffic days with sensor gaps ---", flush=True)
    from experiments.realdata import gaps
    gaps.run()
    print(f"\n[realdata] {(time.time() - t0) / 60:.1f} min; CSVs -> {RESULTS}")


if __name__ == "__main__":
    run()
