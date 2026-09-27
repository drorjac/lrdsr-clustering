"""The days the first pass threw away: I-94 days with sensor gaps.

    python -m experiments.realdata.gaps

``windows.build_windows`` keeps a traffic day only if all 24 hours are
there, because every estimator it feeds needs one design for every window.
That drops 646 of 1860 days -- a third of the data, and not at random (2013
alone loses 221). A law does not need a complete day: it can be scored at
whichever hours were observed. So these are the first real windows in the
project with **different designs**, which is the setting the method is for.

One window per day, at its observed hours only:

.. code-block:: text

    x_t = 2 pi h_t / 24          h_t the observed hours of the day
    y_t = log1p(count_t)         NOT centred: the day's level is a nuisance,
                                 profiled out per window, because the mean
                                 over a partial day is biased by which hours
                                 are missing

Days with fewer than ``MIN_HOURS`` observed hours are left out (a rule fixed
before any result: below six hours the day's shape is not identified by a
four-harmonic law).

==================  =======================================================
``assign``          label-free: cluster the complete days (K = 2, Fourier
                    basis, the realdata recipe), turn the two clusters into
                    two laws, and assign each gap day by likelihood at its
                    own hours. Scored against the calendar, like the rest of
                    ``realdata``.
``supervised``      the same with the calendar as training labels on the
                    complete days (a classifier, not a clustering)
``transplant``      the controlled version: every complete day gets the gap
                    mask of a randomly drawn real gap day, so the answer at
                    24 hours is known. Agreement with the full-day decision
                    and with the calendar, by hours observed.
==================  =======================================================

Every arm compares the law route with the profile route a practitioner would
take instead -- impute the missing hours (circular linear interpolation, or
the hour's mean over complete days), centre, and use the same decision on
the completed profile.

Writes ``results/realdata/realdata_gaps_*.csv``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from experiments.realdata.run import N_HARMONICS, _to_reference
from experiments.realdata.sources import load_hourly
from experiments.realdata.windows import HOURS, traffic_hourly
from lrdsr import paths
from lrdsr.core.classify import LawClassifier
from lrdsr.core.kernel import FourierBasis
from lrdsr.core.soft import SoftLRDSR
from lrdsr.protocol import REPORT_SEEDS

RESULTS = paths.results_dir("realdata", figs=False)
MIN_HOURS = 6
HOUR_BINS = (5, 11, 17, 22, 23, 24)            # (5, 11], (11, 17], ... hours observed


# ==========================================================================
# the days, complete and partial
# ==========================================================================
def traffic_days() -> pd.DataFrame:
    """One row per day with >= MIN_HOURS observed hours: its hours, counts and
    calendar. ``complete`` marks the 24-hour days the first pass used."""
    h = traffic_hourly(load_hourly("traffic")).drop_duplicates(["date", "hour"])
    rows = []
    for date, g in h.groupby("date"):
        g = g.sort_values("hour")
        if len(g) < MIN_HOURS:
            continue
        rows.append({"date": date, "hours": g["hour"].to_numpy(),
                     "count": g["count"].to_numpy(float),
                     "working": bool(g["working"].iloc[0]),
                     "holiday": g["holiday"].iloc[0], "n_hours": len(g)})
    d = pd.DataFrame(rows)
    d["complete"] = d["n_hours"] == HOURS
    d["reference"] = d["working"].astype(int)
    return d.reset_index(drop=True)


def windows(hours_list, counts_list):
    """Ragged windows at the observed hours: ``x = 2 pi h / 24``, ``y = log1p``."""
    X = [(2 * np.pi * np.asarray(h) / HOURS)[:, None] for h in hours_list]
    y = [np.log1p(np.asarray(c, float)) for c in counts_list]
    return X, y


def impute(hours_list, counts_list, how: str, hour_mean=None) -> np.ndarray:
    """``(W, 24)`` centred log profiles, the missing hours filled in.

    ``linear``: circular linear interpolation between observed hours.
    ``mean``: each missing hour gets the complete days' mean at that hour,
    shifted to the day's own level over the hours it has.
    """
    out = np.empty((len(hours_list), HOURS))
    grid = np.arange(HOURS)
    for i, (h, c) in enumerate(zip(hours_list, counts_list, strict=True)):
        v = np.log1p(np.asarray(c, float))
        h = np.asarray(h)
        if how == "linear":
            hh = np.r_[h - HOURS, h, h + HOURS]
            out[i] = np.interp(grid, hh, np.r_[v, v, v])
        elif how == "mean":
            full = hour_mean.copy()
            full += (v - hour_mean[h]).mean()
            full[h] = v
            out[i] = full
        else:
            raise ValueError(how)
    return out - out.mean(axis=1, keepdims=True)


def _fourier():
    return FourierBasis(N_HARMONICS)


# ==========================================================================
# the three arms
# ==========================================================================
def _complete_arrays(days):
    c = days[days.complete]
    X = np.stack([(2 * np.pi * np.arange(HOURS) / HOURS)[:, None]] * len(c))
    Y = np.stack([np.log1p(v) for v in c["count"]])
    return c, X, Y


def _cluster_complete(days, seed):
    """K = 2 on complete days, the realdata recipe: soft EM in a Fourier basis
    on the centred profile (level removed, as in ``windows.build_windows``)."""
    c, X, Y = _complete_arrays(days)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    return c, SoftLRDSR(2, basis=_fourier(), random_state=seed).fit(X, Yc).labels


def _profile_rule(prof_train, lab_train, kind):
    """The decision a profile method makes on a completed profile."""
    if kind == "centroid":
        C = np.stack([prof_train[lab_train == k].mean(0) for k in np.unique(lab_train)])
        classes = np.unique(lab_train)
        return lambda P: classes[np.argmin(((P[:, None, :] - C[None]) ** 2).sum(-1), 1)]
    if kind == "logistic":
        m = LogisticRegression(C=1.0, max_iter=5000).fit(prof_train, lab_train)
        return m.predict
    raise ValueError(kind)


def _decide(days_train, lab_train, days_eval, kind, hour_mean):
    """Every route's labels for ``days_eval``: law, and profile after each imputation."""
    Xtr, ytr = windows(days_train["hours"], days_train["count"])
    law = LawClassifier(basis=_fourier(), nuisance="intercept").fit(Xtr, ytr, lab_train)
    Xe, ye = windows(days_eval["hours"], days_eval["count"])
    out = {"law": law.predict(Xe, ye)}
    prof_tr = impute(days_train["hours"], days_train["count"], "linear")
    rule = _profile_rule(prof_tr, lab_train, kind)
    for how in ("linear", "mean"):
        out[f"profile_{how}"] = rule(impute(days_eval["hours"], days_eval["count"], how,
                                            hour_mean))
    return out


def _hour_mean(days):
    _, _, Y = _complete_arrays(days)
    return Y.mean(axis=0)


def run_assign(days, seed=REPORT_SEEDS[0]) -> pd.DataFrame:
    """Label-free: clusters of complete days -> laws -> partial days."""
    c, lab = _cluster_complete(days, seed)
    gap = days[~days.complete]
    res = _decide(c, lab, gap, "centroid", _hour_mean(days))
    mp = _to_reference(c["reference"].to_numpy(), lab)     # cluster -> day type
    rows = []
    for route, pred in res.items():
        as_type = np.array([mp[int(k)] for k in pred])
        for i, (_, d) in enumerate(gap.iterrows()):
            rows.append({"arm": "assign", "route": route, "date": f"{d.date:%Y-%m-%d}",
                         "n_hours": d.n_hours, "reference": d.reference,
                         "holiday": d.holiday, "predicted": int(as_type[i]),
                         "correct": bool(as_type[i] == d.reference)})
    return pd.DataFrame(rows)


def run_supervised(days) -> pd.DataFrame:
    c = days[days.complete]
    gap = days[~days.complete]
    res = _decide(c, c["reference"].to_numpy(), gap, "logistic", _hour_mean(days))
    rows = []
    for route, pred in res.items():
        for i, (_, d) in enumerate(gap.iterrows()):
            rows.append({"arm": "supervised", "route": route,
                         "date": f"{d.date:%Y-%m-%d}", "n_hours": d.n_hours,
                         "reference": d.reference, "holiday": d.holiday,
                         "predicted": int(pred[i]), "correct": bool(pred[i] == d.reference)})
    return pd.DataFrame(rows)


def run_transplant(days, seeds=REPORT_SEEDS) -> pd.DataFrame:
    """Real gap masks on complete days. Train on half the complete days (full),
    evaluate the other half masked; the full-day decision is the target."""
    c = days[days.complete].reset_index(drop=True)
    masks = [np.asarray(h) for h in days.loc[~days.complete, "hours"]]
    hour_mean = _hour_mean(days)
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(c))
        tr, ev = c.iloc[perm[: len(c) // 2]], c.iloc[perm[len(c) // 2:]]
        # label-free labels on the training half, as in ``assign``
        _, Xtr, Ytr = _complete_arrays(tr.assign(complete=True))
        lab_tr = SoftLRDSR(2, basis=_fourier(), random_state=seed).fit(
            Xtr, Ytr - Ytr.mean(1, keepdims=True)).labels
        mp = _to_reference(tr["reference"].to_numpy(), lab_tr)
        full = _decide(tr, lab_tr, ev, "centroid", hour_mean)
        pick = rng.integers(0, len(masks), len(ev))
        mh = [np.intersect1d(masks[j], np.arange(HOURS)) for j in pick]
        masked = ev.assign(hours=mh, count=[ev["count"].iloc[i][mh[i]]
                                            for i in range(len(ev))],
                           n_hours=[len(m) for m in mh])
        part = _decide(tr, lab_tr, masked, "centroid", hour_mean)
        for route in part:
            for i in range(len(ev)):
                rows.append({"arm": "transplant", "seed": seed, "route": route,
                             "n_hours": int(masked["n_hours"].iloc[i]),
                             "reference": int(ev["reference"].iloc[i]),
                             "agrees_with_full_day": bool(part[route][i] == full[route][i]),
                             "agrees_with_law_full_day": bool(part[route][i] == full["law"][i]),
                             "correct": bool(mp[int(part[route][i])]
                                             == ev["reference"].iloc[i])})
    return pd.DataFrame(rows)


def _bin(n):
    return pd.cut(n, HOUR_BINS).astype(str)


def run(args=None) -> None:
    t0 = time.time()
    days = traffic_days()
    cov = {"days_with_min_hours": len(days), "complete": int(days.complete.sum()),
           "partial": int((~days.complete).sum()), "min_hours": MIN_HOURS,
           "working_share_partial": float(days.loc[~days.complete, "reference"].mean())}
    print(cov, flush=True)
    pd.DataFrame([cov]).to_csv(RESULTS / "realdata_gaps_coverage.csv", index=False)

    a = run_assign(days)
    s = run_supervised(days)
    both = pd.concat([a, s], ignore_index=True)
    both.to_csv(RESULTS / "realdata_gaps_days.csv", index=False)
    both["hours_bin"] = _bin(both.n_hours)
    summ = (both.groupby(["arm", "route", "hours_bin"])
            .agg(accuracy=("correct", "mean"), n_days=("correct", "size")).reset_index())
    tot = both.groupby(["arm", "route"]).agg(accuracy=("correct", "mean"),
                                             n_days=("correct", "size")).reset_index()
    tot["hours_bin"] = "all"
    summ = pd.concat([summ, tot], ignore_index=True)
    summ.to_csv(RESULTS / "realdata_gaps_summary.csv", index=False)
    print(summ.pivot_table(index=["arm", "hours_bin"], columns="route",
                           values="accuracy").round(3).to_string())

    tr = run_transplant(days)
    tr["hours_bin"] = _bin(tr.n_hours)
    tsum = (tr.groupby(["route", "hours_bin"])
            .agg(agrees_with_full_day=("agrees_with_full_day", "mean"),
                 accuracy=("correct", "mean"), n=("correct", "size")).reset_index())
    tsum.to_csv(RESULTS / "realdata_gaps_transplant.csv", index=False)
    print(tsum.pivot_table(index="hours_bin", columns="route",
                           values=["agrees_with_full_day", "accuracy"]).round(3).to_string())
    print(f"\n[realdata.gaps] {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    run()
