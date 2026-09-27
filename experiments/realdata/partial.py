"""V11 on real days: predict a partial day's error from the hours it has.

    python -m experiments.realdata.partial

``experiments/realdata/gaps.py`` showed the law route keeps its decision on
days with most hours missing. This asks the sharper question: can the
error of a partial day be **predicted**, from the laws and the observed
hours alone, before the day is scored (``lrdsr/theory/partial.py``)?

Per dataset (I-94 traffic, bike sharing) and reporting seed: the complete
days are split in half. On the training half the two day-laws are found
without labels (the realdata recipe: soft EM, Fourier basis, K = 2), a
``LawClassifier`` is fitted to them, and the residual covariance ``Sigma_k``
(24 x 24) of each regime's days about its law is estimated. On the held-out
half each day's **full-day decision** is the target -- label-free, and
itself almost never in doubt at a separation this large -- and the day is
re-scored under masks:

==============  =========================================================
``real``        the observed-hour set of a randomly drawn real I-94 gap day
``block``       a contiguous block of 6 or 12 hours missing, at each of the
                24 start hours: the same NUMBER of hours, different hours
``prefix``      hours 0..t-1 only: the day read as it arrives
==============  =========================================================

Each (day, mask) gets V11's predicted error for its target law and the
observed disagreement with the full-day decision; prefixes also get the
predicted and observed share of days decided at 99% posterior confidence.
``Sigma`` and the laws come from the training half only.

Writes ``results/realdata/realdata_partial_*.csv``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from experiments.realdata import gaps
from experiments.realdata.windows import HOURS, load
from lrdsr import paths
from lrdsr.core.classify import LawClassifier
from lrdsr.core.kernel import FourierBasis
from lrdsr.core.soft import SoftLRDSR
from lrdsr.protocol import REPORT_SEEDS
from lrdsr.theory.partial import decided_share, masked_error

RESULTS = paths.results_dir("realdata", figs=False)
BLOCKS = (6, 12)
CONFIDENCE = 0.99
CAL_BINS = (0.0, 0.002, 0.01, 0.03, 0.1, 0.3, 1.0)
XGRID = (2 * np.pi * np.arange(HOURS) / HOURS)[:, None]


def complete_days(name: str) -> np.ndarray:
    """``(W, 24)`` log1p counts of the complete days (level not removed)."""
    if name == "traffic":
        d = gaps.traffic_days()
        return np.stack([np.log1p(c) for c in d.loc[d.complete, "count"]])
    return load(name).y_seq                       # centred; the level is a nuisance


def real_masks() -> list[np.ndarray]:
    d = gaps.traffic_days()
    return [np.asarray(h) for h in d.loc[~d.complete, "hours"]]


def masks(seed: int, n_days: int, reals: list[np.ndarray]) -> list[tuple[str, str, np.ndarray]]:
    """Every mask family, as ``(family, label, observed hours)``; the real masks
    are one random draw per held-out day, the rest are shared by every day."""
    out = []
    for k in BLOCKS:
        for start in range(HOURS):
            miss = (start + np.arange(k)) % HOURS
            out.append(("block", f"{k}h@{start:02d}", np.setdiff1d(np.arange(HOURS), miss)))
    for t in range(2, HOURS + 1):
        out.append(("prefix", f"{t:02d}", np.arange(t)))
    return out


def fit_half(Y: np.ndarray, seed: int):
    """Laws, classifier and per-regime residual covariance from complete days."""
    Yc = Y - Y.mean(axis=1, keepdims=True)
    X = np.broadcast_to(XGRID, (len(Y), HOURS, 1)).copy()
    lab = SoftLRDSR(2, basis=FourierBasis(4), random_state=seed).fit(X, Yc).labels
    clf = LawClassifier(basis=FourierBasis(4), nuisance="intercept").fit(X, Y, lab)
    F = clf.design_.B(XGRID) @ clf.coef_.T                 # (24, 2) the two laws
    Sigma = []
    for k in range(2):
        f = F[:, k] - F[:, k].mean()
        R = Yc[lab == k] - f[None, :]
        Sigma.append(np.cov(R, rowvar=False))
    return clf, F, Sigma, lab


def run_dataset(name: str, reals, seeds=REPORT_SEEDS) -> pd.DataFrame:
    Y = complete_days(name)
    rows = []
    for seed in seeds:
        rng = np.random.default_rng([seed, len(name)])
        perm = rng.permutation(len(Y))
        tr, ev = Y[perm[: len(Y) // 2]], Y[perm[len(Y) // 2:]]
        clf, F, Sigma, _ = fit_half(tr, seed)
        s2 = float(clf.sigma_[0] ** 2)
        L = float(clf.log_prior_[1] - clf.log_prior_[0])
        g = F[:, 1] - F[:, 0]
        Xev = np.broadcast_to(XGRID, (len(ev), HOURS, 1)).copy()
        target = clf.predict(Xev, ev)
        fam = masks(seed, len(ev), reals)
        pick = rng.integers(0, len(reals), len(ev))
        per_day_real = [("real", "real", reals[j]) for j in pick]
        for fi, (family, label, H) in enumerate(fam + per_day_real):
            if family == "real":
                days = [fi - len(fam)]                   # one real mask per day
            else:
                days = range(len(ev))
            days = list(days)
            Xm = [XGRID[H] for _ in days]
            ym = [ev[d][H] for d in days]
            pred = clf.predict(Xm, ym)
            post = clf.predict_proba(Xm, ym)
            for j, d in enumerate(days):
                k = int(target[d])
                SH = Sigma[k][np.ix_(H, H)]
                row = {"dataset": name, "seed": seed, "family": family, "mask_label": label,
                       "n_hours": len(H), "target": k,
                       "predicted_error": masked_error(g[H], SH, s2, L, truth=k),
                       "error": bool(pred[j] != k),
                       "posterior_target": float(post[j, k])}
                if family == "prefix":
                    ok, bad = decided_share(g[H], SH, s2, L, truth=k, confidence=CONFIDENCE)
                    row.update(pred_decided_right=ok, pred_decided_wrong=bad,
                               decided_right=bool(post[j, k] >= CONFIDENCE),
                               decided_wrong=bool(post[j, 1 - k] >= CONFIDENCE))
                rows.append(row)
        print(f"  [{name}] seed {seed}: {len(ev)} held-out days", flush=True)
    return pd.DataFrame(rows)


def calibration(df: pd.DataFrame) -> pd.DataFrame:
    """Predicted against observed error, binned by the prediction."""
    d = df.assign(bin=pd.cut(df.predicted_error, CAL_BINS, include_lowest=True).astype(str))
    g = d.groupby(["dataset", "family", "bin"], observed=True).agg(
        n=("error", "size"), predicted=("predicted_error", "mean"),
        observed=("error", "mean")).reset_index()
    g["band"] = 2 * np.sqrt(np.clip(g.predicted * (1 - g.predicted), 1e-12, None) / g.n)
    g["in_band"] = (g.observed - g.predicted).abs() <= g.band
    return g


def by_mask(df: pd.DataFrame, family: str) -> pd.DataFrame:
    d = df[df.family == family]
    cols = {"predicted": ("predicted_error", "mean"), "observed": ("error", "mean"),
            "n": ("error", "size")}
    if family == "prefix":
        cols.update(pred_decided_right=("pred_decided_right", "mean"),
                    decided_right=("decided_right", "mean"),
                    pred_decided_wrong=("pred_decided_wrong", "mean"),
                    decided_wrong=("decided_wrong", "mean"))
    return d.groupby(["dataset", "mask_label"]).agg(**cols).reset_index()


def run(args=None) -> None:
    t0 = time.time()
    reals = real_masks()
    df = pd.concat([run_dataset(n, reals) for n in ("traffic", "bike")], ignore_index=True)
    cal = calibration(df)
    cal.to_csv(RESULTS / "realdata_partial_calibration.csv", index=False)
    print(cal.round(4).to_string(index=False))
    blk = by_mask(df, "block")
    blk.to_csv(RESULTS / "realdata_partial_blocks.csv", index=False)
    pre = by_mask(df, "prefix")
    pre.to_csv(RESULTS / "realdata_partial_prefix.csv", index=False)
    print(pre.round(3).to_string(index=False))
    # the ranking test: within a block length, does the predicted error order
    # the start hours the way the observed error does?
    rk = []
    for (ds, k), g in blk.assign(k=blk["mask_label"].str.split("h").str[0]).groupby(
            ["dataset", "k"]):
        rk.append({"dataset": ds, "block_hours": int(k),
                   "spearman": float(g.predicted.rank().corr(g.observed.rank())),
                   "worst_start_predicted": g.loc[g.predicted.idxmax(), "mask_label"],
                   "worst_start_observed": g.loc[g.observed.idxmax(), "mask_label"],
                   "min_observed": float(g.observed.min()),
                   "max_observed": float(g.observed.max())})
    rk = pd.DataFrame(rk)
    rk.to_csv(RESULTS / "realdata_partial_ranking.csv", index=False)
    print(rk.round(3).to_string(index=False))
    print(f"\n[realdata.partial] {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    run()
