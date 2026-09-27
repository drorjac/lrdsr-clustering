"""Real time with a kernel basis and with windows that have gaps.

    python -m experiments.online.ragged

==================  =======================================================
``kernel_birth``    the zoo's ``high_frequency`` pair as a stream: warm
                    start on law 0 alone, law 1 appears at window 40. Is it
                    born, how fast, and how well are later windows sorted --
                    with the fast library (37% of the gap outside its span)
                    against a Nystrom basis (rank 17, fitted on the warm
                    start)? Control streams with law 0 only count false
                    births.
``traffic``         every I-94 day with >= 6 observed hours, in calendar
                    order, one day at a time, the level a nuisance: the
                    stream the first real-time pass could not run, because
                    it dropped the 616 days with sensor gaps.
==================  =======================================================

Writes ``results/online/online_ragged_*.csv``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from experiments.online.streams import novelty_power
from experiments.problems.zoo import PROBLEMS, sigma_for_rho
from experiments.realdata import gaps
from experiments.realdata.run import _to_reference
from lrdsr import paths
from lrdsr.core.kernel import FourierBasis, NystromBasis
from lrdsr.core.online import OnlineLRDSR
from lrdsr.core.soft import SoftLRDSR
from lrdsr.protocol import REPORT_SEEDS

RESULTS = paths.results_dir("online", figs=False)
WARM, ONSET, STREAM = 80, 40, 240
N_LEN = 48
RHOS = (0.5, 1.0, 2.0)
BIRTH_ALPHA = 1e-3            # the middle of experiments/online/streams.ALPHAS
ALPHA = 1e-6                  # the real-data stream, as in experiments/realdata
WARMUP_DAYS = 60


# ==========================================================================
# a newcomer the library cannot represent
# ==========================================================================
def _hf_stream(rho, seed, control=False):
    prob = next(p for p in PROBLEMS if p.name == "high_frequency")
    sigma = sigma_for_rho(prob, rho)
    rng = np.random.default_rng([seed, int(rho * 100), control])
    T = WARM + STREAM
    z = np.zeros(T, int)
    if not control:
        z[WARM + ONSET:] = rng.integers(0, 2, T - WARM - ONSET)
    X = prob.sample(rng, (T, N_LEN))
    y = np.stack([prob.laws[k](X[t]) for t, k in enumerate(z)])
    return X, y + rng.normal(0, sigma, y.shape), z


def birth_cell(rho, seed, basis_name, control):
    X, y, z = _hf_stream(rho, seed, control)
    basis = None if basis_name == "library" else NystromBasis(16)
    on = OnlineLRDSR(basis=basis, feature_names=["x"], novelty_alpha=BIRTH_ALPHA)
    on.warm_start(X[:WARM], y[:WARM], labels=np.zeros(WARM, int))
    tl = on.fit_stream(X[WARM:], y[WARM:], true_labels_for_eval=z[WARM:])
    births = tl.index[tl["spawned"].notna()].tolist()
    row = {"rho": rho, "seed": seed, "basis": basis_name, "control": control,
           "births": len(births),
           # the power a known-law chi-square test would have on one newcomer
           # window (experiments/online/streams.novelty_power)
           "predicted_power": novelty_power(rho, BIRTH_ALPHA, n=N_LEN)}
    if not control:
        first = births[0] if births else np.nan
        row["birth_delay"] = (first - ONSET) if births else np.nan
        # after the birth: does the newborn regime take law 1's windows?
        if births:
            after = tl.iloc[first + 1:]
            lab = after["label"].to_numpy()
            born = int(tl["spawned"].iloc[first])
            pred = np.where(lab == born, 1, np.where(lab >= 0, 0, -1))
            row["error_after_birth"] = float(np.mean(pred != after["truth"].to_numpy()))
            row["buffered_after_birth"] = float(np.mean(lab < 0))
    return row


def run_kernel_birth() -> pd.DataFrame:
    cells = [(r, s, b, c) for r in RHOS for s in REPORT_SEEDS
             for b in ("library", "kernel") for c in (False, True)]
    return pd.DataFrame(Parallel(n_jobs=-1)(delayed(birth_cell)(*c) for c in cells))


# ==========================================================================
# the full I-94 stream
# ==========================================================================
def run_traffic(seed=REPORT_SEEDS[0]) -> tuple[pd.DataFrame, pd.DataFrame]:
    days = gaps.traffic_days().sort_values("date").reset_index(drop=True)
    X, y = gaps.windows(days["hours"], days["count"])
    ref = days["reference"].to_numpy()
    tls, summ = [], []
    for arm, sel in (("all_days", np.ones(len(days), bool)),
                     ("complete_only", days["complete"].to_numpy())):
        idx = np.flatnonzero(sel)
        # the same warm start for both arms, the published recipe: soft EM in a
        # Fourier basis on the centred profiles of the first WARMUP_DAYS
        # COMPLETE days, handed over as labels -- so the arms differ only in
        # whether partial days are streamed
        comp = np.flatnonzero(days["complete"].to_numpy())
        warm = comp[:WARMUP_DAYS]
        Yw = np.stack([y[i] for i in warm])
        Xw = np.stack([X[i] for i in warm])
        lab = SoftLRDSR(2, basis=FourierBasis(4), random_state=seed).fit(
            Xw, Yw - Yw.mean(axis=1, keepdims=True)).labels
        rest = idx[idx > warm[-1]]
        on = OnlineLRDSR(basis=FourierBasis(4), nuisance="intercept", novelty_alpha=ALPHA)
        on.warm_start([X[i] for i in warm], [y[i] for i in warm], labels=lab)
        tl = on.fit_stream([X[i] for i in rest], [y[i] for i in rest],
                           true_labels_for_eval=ref[rest])
        tl = tl.rename(columns={"map": "map_regime"})
        tl["date"] = days["date"].iloc[rest].dt.strftime("%Y-%m-%d").to_numpy()
        tl["n_hours"] = days["n_hours"].iloc[rest].to_numpy()
        tl["complete"] = days["complete"].iloc[rest].to_numpy()
        mp = _to_reference(ref[rest], tl["label"].to_numpy())
        tl["as_daytype"] = tl["label"].map(lambda k, mp=mp: mp.get(int(k), -1))
        tl["correct"] = tl["as_daytype"] == tl["truth"]
        tl["arm"] = arm
        tls.append(tl)
        for part, m in (("complete", tl["complete"]), ("partial", ~tl["complete"]),
                        ("all", tl["complete"] | True)):
            g = tl[m]
            if len(g):
                summ.append({"arm": arm, "days": part, "n": len(g),
                             "accuracy": float(g["correct"].mean()),
                             "buffered": float((g["label"] < 0).mean()),
                             "births": int(tl["spawned"].notna().sum()),
                             "final_regimes": on.n_clusters})
    return pd.concat(tls, ignore_index=True), pd.DataFrame(summ)


def run(args=None) -> None:
    t0 = time.time()
    kb = run_kernel_birth()
    kb.to_csv(RESULTS / "online_ragged_kernel_birth.csv", index=False)
    print(kb.groupby(["rho", "basis", "control"]).mean(numeric_only=True).round(3).to_string())
    tl, s = run_traffic()
    tl.to_csv(RESULTS / "online_ragged_traffic.csv", index=False)
    s.to_csv(RESULTS / "online_ragged_traffic_summary.csv", index=False)
    print(s.round(4).to_string(index=False))
    print(f"\n[online.ragged] {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    run()
