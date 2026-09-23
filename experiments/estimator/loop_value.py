"""When is the alternating loop worth running?

Stage 1 reports a flat negative: LR-DSR's loop does not beat mechanism
K-means on the pair battery, mean gap {estGapMeanKm} against {estGapMean}.
That is true and it is also *conditional*, on an initialiser that is already
at the oracle. A tool that adds nothing to a perfect start has not been shown
to add nothing; it has been shown nothing was left to add.

So the question here is not "does the loop help" but **where**. The loop is
an operator on a partition, so it is characterised the way an operator is:
feed it partitions of known quality and measure what comes out.

    gain = error(partition in) - error(partition out)

``gain > 0`` is repair, ``gain < 0`` is damage, and the useful result is the
boundary between them.

Three axes, each a reason a starting partition might be poor in practice:

``corruption``  the oracle partition with a fraction ``q`` of windows
                flipped at random. A controlled quality axis from 0 (the
                ceiling) to 0.5 (chance). The oracle is a function of the
                data and the known laws -- the same object stage 1 uses as
                its ceiling everywhere -- so no stored label is read to
                build it. It is a **diagnostic input**, and no number from
                this arm is LR-DSR's accuracy on anything.
``initialiser`` the real initialisers, unmodified: mechanism space, the
                data-space BGMM it replaced, plain K-means on window
                summaries, and a random partition. This arm is the
                reportable one -- everything in it is label-free.
``regime``      separation ``rho``, class balance, and window length ``n``.
                A loop that only repairs at high ``rho`` is a different tool
                from one that repairs when the evidence is thin.

Protocol: explored on the tuning seeds {3, 7, 19}, reported on {11, 23, 42},
so "the loop helps above corruption X" is not a threshold chosen on the
numbers it is quoted with.

Writes ``results/estimator/loop_value_*.csv``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.estimator.benchmark import (
    ALPHA_GEOM_G2,
    PAIRS,
    _make_windows,
    _oracle_labels,
    _pair_gap_ms,
)
from lrdsr import paths, stats
from lrdsr.core.evaluation import aligned_accuracy
from lrdsr.core.mechanism_space import mechanism_init
from lrdsr.core.model import GroupedDCSR

RESULTS = paths.results_dir("estimator", figs=False)
TUNE_SEEDS = (3, 7, 19)
REPORT_SEEDS = (11, 23, 42)

#: Fraction of the oracle partition flipped before the loop sees it.
#: 0.5 is chance for two balanced classes.
CORRUPTION = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
#: Separations, from the benchmark's own grid.
RHO_GRID = (0.1, 0.25, 1.0)
#: The label-free initialisers, worst to best.
INITS = ("random", "kmeans", "bgmm", "mechanism")
N_WINDOWS = 150


def _err(true, pred):
    return 1.0 - aligned_accuracy(np.asarray(true), np.asarray(pred))


def _corrupt(labels, q, rng, k=2):
    """Flip a fraction ``q`` of the partition, uniformly at random."""
    out = np.asarray(labels, int).copy()
    if q <= 0:
        return out
    idx = rng.choice(len(out), size=int(round(q * len(out))), replace=False)
    out[idx] = rng.integers(0, k, size=len(idx))
    return out


def _loop(X_seq, y, Zg, init, seed, max_iter=10):
    """The alternating loop, started from ``init`` (name or array)."""
    model = GroupedDCSR(
        n_clusters=2, alpha_geom=ALPHA_GEOM_G2, beta_complexity=0.002,
        max_iter=max_iter, tol=0.01, backend="fast",
        backend_kwargs={"max_terms": 5}, random_state=seed,
        min_windows_per_cluster=4, init=init, geom_metric="mahalanobis",
        score_mode="cross_fit", residual_scale="global")
    return model.fit(X_seq, y, Zg, feature_names=["x"])


def run_corruption(seeds, rho_grid=RHO_GRID, verbose=True):
    """The controlled-quality axis: what does the loop do to a bad start?"""
    rows = []
    for pair in PAIRS:
        ms = _pair_gap_ms(pair)
        for rho in rho_grid:
            sigma = float(np.sqrt(ms / rho))
            for seed in seeds:
                X_seq, y, z, x, Zg = _make_windows(pair, sigma, N_WINDOWS, seed)
                oracle = _oracle_labels(x, y, [pair.f0, pair.f1])
                rng = np.random.default_rng(seed + 7777)
                for q in CORRUPTION:
                    start = _corrupt(oracle, q, rng)
                    res = _loop(X_seq, y, Zg, start, seed)
                    e_in, e_out = _err(z, start), _err(z, res.labels)
                    rows.append({
                        "arm": "corruption", "pair": pair.name, "rho": rho,
                        "seed": seed, "corruption": q,
                        "error_in": e_in, "error_out": e_out,
                        "gain": e_in - e_out,
                        "iterations": int(len(res.history)),
                        "moved": bool(len(res.history) > 1
                                      or res.history.changed_fraction.iloc[0] > 0),
                        "oracle_error": _err(z, oracle)})
            if verbose:
                h = [r for r in rows if r["pair"] == pair.name
                     and r["rho"] == rho]
                g = {q: np.mean([r["gain"] for r in h if r["corruption"] == q])
                     for q in CORRUPTION}
                print(f"  {pair.name:11s} rho {rho:<5g} gain by corruption: "
                      + "  ".join(f"{q:.2f}:{v:+.3f}" for q, v in g.items()),
                      flush=True)
    return pd.DataFrame(rows)


def run_initialisers(seeds, rho_grid=RHO_GRID, verbose=True):
    """The label-free arm: the initialisers anyone would actually use."""
    rows = []
    for pair in PAIRS:
        ms = _pair_gap_ms(pair)
        for rho in rho_grid:
            sigma = float(np.sqrt(ms / rho))
            for seed in seeds:
                X_seq, y, z, x, Zg = _make_windows(pair, sigma, N_WINDOWS, seed)
                rng = np.random.default_rng(seed + 31)
                for name in INITS:
                    if name == "random":
                        start = rng.integers(0, 2, size=len(y))
                    elif name == "mechanism":
                        start = mechanism_init(X_seq, y, 2, seed=seed)
                    else:
                        start = name           # the model builds it
                    res = _loop(X_seq, y, Zg, start, seed)
                    # the partition the loop was handed, however it was made
                    if isinstance(start, str):
                        probe = _loop(X_seq, y, Zg, start, seed, max_iter=1)
                        e_in = _err(z, probe.labels)
                    else:
                        e_in = _err(z, start)
                    rows.append({
                        "arm": "initialiser", "pair": pair.name, "rho": rho,
                        "seed": seed, "init": name,
                        "error_in": e_in, "error_out": _err(z, res.labels),
                        "gain": e_in - _err(z, res.labels),
                        "iterations": int(len(res.history))})
            if verbose:
                h = [r for r in rows if r["pair"] == pair.name
                     and r["rho"] == rho]
                g = {n: np.mean([r["gain"] for r in h if r["init"] == n])
                     for n in INITS}
                print(f"  {pair.name:11s} rho {rho:<5g} gain by init: "
                      + "  ".join(f"{n}:{v:+.3f}" for n, v in g.items()),
                      flush=True)
    return pd.DataFrame(rows)


def summarise(corr, init):
    """Where the loop repairs, and by how much, with the seed as replicate."""
    rows = []
    for q, h in corr.groupby("corruption"):
        g = h.gain.to_numpy()
        lo, hi = stats.boot_ci(lambda i: float(np.mean(g[i])), len(g))
        rows.append({"arm": "corruption", "level": float(q), "n": len(g),
                     "mean_gain": float(g.mean()), "lo": lo, "hi": hi,
                     "repairs": bool(lo > 0), "damages": bool(hi < 0),
                     "mean_error_in": float(h.error_in.mean()),
                     "mean_error_out": float(h.error_out.mean())})
    for name, h in init.groupby("init"):
        g = h.gain.to_numpy()
        lo, hi = stats.boot_ci(lambda i: float(np.mean(g[i])), len(g))
        rows.append({"arm": "initialiser", "level": name, "n": len(g),
                     "mean_gain": float(g.mean()), "lo": lo, "hi": hi,
                     "repairs": bool(lo > 0), "damages": bool(hi < 0),
                     "mean_error_in": float(h.error_in.mean()),
                     "mean_error_out": float(h.error_out.mean())})
    return pd.DataFrame(rows)


def run(verbose=True):
    out = {}
    for tag, seeds in (("tuning", TUNE_SEEDS), ("report", REPORT_SEEDS)):
        print(f"\n[loop_value] corruption axis, {tag} seeds", flush=True)
        c = run_corruption(seeds, verbose=verbose)
        print(f"\n[loop_value] initialiser axis, {tag} seeds", flush=True)
        i = run_initialisers(seeds, verbose=verbose)
        c["seed_set"], i["seed_set"] = tag, tag
        out[tag] = (c, i)
    corr = pd.concat([out[t][0] for t in out], ignore_index=True)
    init = pd.concat([out[t][1] for t in out], ignore_index=True)
    rep = summarise(corr[corr.seed_set == "report"],
                    init[init.seed_set == "report"])
    corr.to_csv(RESULTS / "loop_value_corruption.csv", index=False)
    init.to_csv(RESULTS / "loop_value_initialiser.csv", index=False)
    rep.to_csv(RESULTS / "loop_value_summary.csv", index=False)
    print(f"\n[loop_value] -> {RESULTS}")
    print(rep.round(3).to_string(index=False))
    return corr, init, rep


if __name__ == "__main__":
    run()
