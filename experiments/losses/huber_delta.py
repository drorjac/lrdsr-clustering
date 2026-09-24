"""Where the default Huber constant comes from: a sweep on the tuning seeds.

The assignment cost divides every residual by a scale before the loss. Until
2026-09-24 that scale was the raw MAD, ``0.6745 sigma`` under Gaussian noise,
so the default ``robust_delta = 1.5`` was really ``~1.0 sigma``: a Huber
loss only ~90% efficient under Gaussian noise (``lrdsr.theory.losses``), and
measured to cost ~30% in window error against the squared loss. The scale is
now the normal-consistent MAD (``lrdsr.core.losses.noise_scale``), so
``delta`` is in units of the noise sd, and this sweep picks its value.

It runs on ``TUNE_SEEDS`` {3, 7, 19}, per the protocol: the default is chosen
here and every reported number (``estimator.py`` and the other blocks) is
made on {11, 23, 42} with whatever was chosen. ``delta = 100`` is the squared
loss in all but name, and ``eta`` is the efficiency the V8 theory assigns to
each ``delta`` under each noise, for reading the sweep against.

Writes ``results/losses/loss_huber_delta.csv`` (per cell) and
``loss_huber_delta_summary.csv``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.estimator.benchmark import PAIR_BY_NAME, _matched_error
from experiments.losses.estimator import (
    NOISES,
    PAIRS,
    RHO_GRID,
    make_windows,
    oracle_lrt,
)
from lrdsr.core.mechanism_space import mechanism_init
from lrdsr.core.model import GroupedDCSR
from lrdsr.protocol import TUNE_SEEDS
from lrdsr.theory.losses import RESULTS, efficiency, noise_families

DELTAS = (1.0, 1.345, 1.5, 2.0, 3.0, 100.0)


def _cell(noise_name, pair_name, rho, seed):
    fam = noise_families()
    pair = PAIR_BY_NAME[pair_name]
    X, y, z, x, sigma = make_windows(pair, fam[noise_name], rho, seed)
    init = mechanism_init(X, y, 2, seed=seed, feature_names=["x"])
    Z = np.random.default_rng(0).normal(size=(len(y), 1))   # alpha_geom = 0: unused
    rows = []
    for d in DELTAS:
        model = GroupedDCSR(
            n_clusters=2, alpha_geom=0.0, beta_complexity=0.002,
            backend_kwargs={"max_terms": 5}, random_state=seed,
            min_windows_per_cluster=4, init=init, residual_scale="global",
            robust_delta=d)
        labels = model.fit(X, y, Z, feature_names=["x"]).labels
        rows.append({"noise": noise_name, "pair": pair_name, "rho": rho, "seed": seed,
                     "delta": d, "matched_error": _matched_error(z, labels)})
    rows.append({"noise": noise_name, "pair": pair_name, "rho": rho, "seed": seed,
                 "delta": np.nan, "matched_error": _matched_error(
                     z, oracle_lrt(x, y, pair, fam[noise_name], sigma))})
    return rows


def run(seeds=TUNE_SEEDS, n_jobs: int = -1) -> pd.DataFrame:
    from joblib import Parallel, delayed

    cells = [(nz, pn, rho, s) for nz in NOISES for pn in PAIRS for rho in RHO_GRID
             for s in seeds]
    out = Parallel(n_jobs=n_jobs)(delayed(_cell)(*c) for c in cells)
    df = pd.DataFrame([r for rows in out for r in rows])
    df["arm"] = np.where(df["delta"].isna(), "oracle_lrt", "huber")
    df.to_csv(RESULTS / "loss_huber_delta.csv", index=False)

    fam = noise_families()
    hub = df[df.arm == "huber"]
    summ = hub.groupby(["noise", "delta"], as_index=False).matched_error.mean()
    orc = df[df.arm == "oracle_lrt"].groupby("noise").matched_error.mean()
    summ["oracle_lrt"] = summ.noise.map(orc)
    summ["excess_over_lrt"] = summ.matched_error - summ.oracle_lrt
    summ["eta"] = [efficiency("huber", fam[n], delta=d) if d < 50 else 1.0
                   for n, d in zip(summ.noise, summ["delta"], strict=True)]
    summ["rank_in_noise"] = summ.groupby("noise").matched_error.rank(method="min")
    summ.to_csv(RESULTS / "loss_huber_delta_summary.csv", index=False)
    print(summ.pivot_table(index="noise", columns="delta",
                           values="matched_error").round(4).to_string())
    return summ


if __name__ == "__main__":
    run()
