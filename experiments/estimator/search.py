"""The search itself: which formula, found how, and at what complexity.

The benchmark reports how well windows are *assigned*. That takes the
symbolic search for granted, and it is the part a reader has not seen: LR-DSR
alternates *fit a law per cluster* with *reassign every window to the law that
explains it best*, so what the search does and what it costs decides the
whole loop.

Four blocks, each a process rather than a score:

``path``        the greedy forward selection, term by term. The backend
                builds a library of interpretable terms and adds the one
                that most improves BIC, until none does. Refitting with
                ``max_terms`` = 0, 1, 2, ... walks that path in public API
                only -- greedy selection is prefix-stable, which the run
                asserts rather than assumes.
``complexity``  what each extra term buys, in HELD-OUT error on fresh
                windows. The reference is not a term count but the true
                law's own held-out error: the question is how many terms
                the search needs to reach the noise floor, and whether BIC
                stops there, short of it, or past it.
``beta``        the assignment's complexity penalty, swept. ``beta = 0``
                lets the loop prefer an elaborate law; a large ``beta``
                forces parsimony on every slot at once. Reported as matched
                error AND as the complexity the loop settles on, because a
                penalty that changes neither is not doing anything.
``history``     the alternating loop, iteration by iteration: how much of
                the partition moves, what the assignment costs, and -- since
                these are synthetic laws -- how accurate it is at each step.
                The truth is read here for a curve, never for a fit.

One pair is deliberately **outside the library**: ``saturation`` needs
``exp`` and ``x^1.1``, and the library has neither. Its rows are the
measurement of what "the library does not contain the truth" costs, and they
are reported beside the others rather than dropped.

Writes ``results/estimator/search_{path,complexity,beta,history}.csv``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.estimator.benchmark import (
    PAIRS,
    REPORT_SEEDS,
    _make_windows,
    _pair_gap_ms,
)
from lrdsr import paths
from lrdsr.core.backends import FastSymbolicRegressor
from lrdsr.core.evaluation import aligned_accuracy
from lrdsr.core.model import GroupedDCSR

RESULTS = paths.results_dir("estimator", figs=False)

#: The separations the search is walked at. How many terms a search takes is
#: not a property of the law -- it is a property of how much noise the law is
#: seen through -- so the path is walked at three, from the benchmark's own
#: grid. ``RHO_SHOW`` is the one the loop blocks use.
RHO_GRID = (0.25, 1.0, 4.0)
RHO_SHOW = 1.0
#: How far the greedy path is walked. The backend's own default is 5; going
#: past it is how the run shows what BIC declines to take.
MAX_TERMS = 8
#: Windows per fit, and a second independent draw of the same size for the
#: held-out error.
N_WINDOWS = 150
#: The complexity penalties swept in ``beta``. 0.002 is what every other
#: experiment uses; the grid brackets it by three orders of magnitude either
#: way, which is the only way to show that the default is on a plateau.
BETA_GRID = (0.0, 0.002, 0.02, 0.2, 1.0, 5.0)


def _slot_data(pair, sigma, seed, slot):
    """The windows of ONE regime, flattened -- the search's actual input.

    The loop fits a law per cluster, so what the backend sees is the rows of
    one cluster. Giving it the true partition isolates the search from the
    assignment: any failure here is the library or BIC, never the labels.
    """
    X_seq, y, z, x, _ = _make_windows(pair, sigma, N_WINDOWS, seed)
    m = z == slot
    return x[m].reshape(-1, 1), y[m].reshape(-1)


def run_path(seeds=REPORT_SEEDS, verbose=True):
    """The greedy path, term by term, for every regime of every pair."""
    rows = []
    for pair in PAIRS:
      for rho in RHO_GRID:
        sigma = float(np.sqrt(_pair_gap_ms(pair) / rho))
        for slot, truth in ((0, pair.truth0), (1, pair.truth1)):
            for seed in seeds:
                Xtr, ytr = _slot_data(pair, sigma, seed, slot)
                Xte, yte = _slot_data(pair, sigma, seed + 1000, slot)
                prev = []
                for k in range(MAX_TERMS + 1):
                    m = FastSymbolicRegressor(max_terms=k,
                                              feature_names=["x"]).fit(Xtr, ytr)
                    names = list(m.term_names_)
                    # greedy forward selection is prefix-stable; if that ever
                    # stops being true the path below is not a path.
                    assert names[:len(prev)] == prev, (names, prev)
                    added = names[len(prev)] if len(names) > len(prev) else None
                    rows.append({
                        "pair": pair.name, "rho": rho, "slot": slot,
                        "truth": truth,
                        "seed": seed, "step": k, "n_terms": len(names),
                        "complexity": float(m.complexity()),
                        "term_added": added, "stalled": added is None and k > 0,
                        "bic": float(m.bic_),
                        "rmse_train": float(np.sqrt(np.mean(
                            (ytr - m.predict(Xtr)) ** 2))),
                        "rmse_held_out": float(np.sqrt(np.mean(
                            (yte - m.predict(Xte)) ** 2))),
                        "expression": m.expression(),
                    })
                    prev = names
                if verbose and seed == seeds[0]:
                    last = rows[-1]
                    print(f"  {pair.name:11s} rho {rho:<5g} slot {slot}  "
                          f"{last['n_terms']} terms: {last['expression'][:52]}",
                          flush=True)
    return pd.DataFrame(rows)


def run_complexity(path: pd.DataFrame, seeds=REPORT_SEEDS):
    """What each term buys, against the floor the true law itself leaves.

    The true law's held-out error is the only honest reference: a term count
    cannot be compared across pairs whose truths are not in the same library,
    and for ``saturation`` the truth is not in the library at all.
    """
    rows = []
    for pair in PAIRS:
      for rho in RHO_GRID:
        sigma = float(np.sqrt(_pair_gap_ms(pair) / rho))
        for slot, f in ((0, pair.f0), (1, pair.f1)):
            for seed in seeds:
                Xte, yte = _slot_data(pair, sigma, seed + 1000, slot)
                floor = float(np.sqrt(np.mean((yte - f(Xte[:, 0])) ** 2)))
                h = path[(path.pair == pair.name) & (path.rho == rho)
                         & (path.slot == slot) & (path.seed == seed)]
                bic_pick = int(h.loc[h.bic.idxmin(), "n_terms"])
                reach = h[h.rmse_held_out <= floor * 1.01]
                rows.append({
                    "pair": pair.name, "rho": rho, "slot": slot, "seed": seed,
                    "sigma": sigma, "floor_rmse": floor,
                    "bic_n_terms": bic_pick,
                    "bic_rmse": float(h[h.n_terms == bic_pick]
                                      .rmse_held_out.iloc[0]),
                    "bic_excess": float(h[h.n_terms == bic_pick]
                                        .rmse_held_out.iloc[0]) - floor,
                    "n_terms_to_floor": (int(reach.n_terms.min())
                                         if len(reach) else np.nan),
                    "best_rmse": float(h.rmse_held_out.min()),
                    "best_n_terms": int(h.loc[h.rmse_held_out.idxmin(),
                                              "n_terms"]),
                })
    return pd.DataFrame(rows)


def run_beta(seeds=REPORT_SEEDS, verbose=True):
    """Does the assignment's complexity penalty do anything?"""
    rows = []
    for pair in PAIRS:
        sigma = float(np.sqrt(_pair_gap_ms(pair) / RHO_SHOW))
        for beta in BETA_GRID:
            for seed in seeds:
                X_seq, y, z, x, Zg = _make_windows(pair, sigma, N_WINDOWS, seed)
                model = GroupedDCSR(
                    n_clusters=2, alpha_geom=0.0, beta_complexity=beta,
                    max_iter=10, tol=0.01, backend="fast",
                    backend_kwargs={"max_terms": 5}, random_state=seed,
                    min_windows_per_cluster=4, init="mechanism",
                    geom_metric="mahalanobis", score_mode="cross_fit",
                    residual_scale="global")
                res = model.fit(X_seq, y, Zg, feature_names=["x"])
                rows.append({
                    "pair": pair.name, "beta_complexity": beta, "seed": seed,
                    "matched_error": 1.0 - aligned_accuracy(z, res.labels),
                    "mean_complexity": float(np.mean(
                        [m.complexity() for m in res.models])),
                    "n_iterations": int(len(res.history)),
                    "expressions": " | ".join(m.expression()[:40]
                                              for m in res.models),
                })
            if verbose:
                h = [r for r in rows if r["pair"] == pair.name
                     and r["beta_complexity"] == beta]
                print(f"  {pair.name:11s} beta {beta:<6g} "
                      f"error {np.mean([r['matched_error'] for r in h]):.3f}  "
                      f"complexity {np.mean([r['mean_complexity'] for r in h]):.2f}",
                      flush=True)
    return pd.DataFrame(rows)


#: The loop is shown under two initialisers and two separations. At the easy
#: cell the mechanism initialiser is already right and the loop has no work
#: to do -- which is the point, and is only visible against an arm where it
#: does. ``bgmm`` is the data-space initialiser LR-DSR used before mechanism
#: space, kept reachable so the change is an ablation and not an assertion.
HISTORY_CELLS = (("mechanism", 1.0), ("mechanism", 0.1),
                 ("bgmm", 1.0), ("bgmm", 0.1))


def run_history(seeds=REPORT_SEEDS, verbose=True):
    """The alternating loop, iteration by iteration, under two initialisers.

    ``true_labels_for_eval`` puts accuracy in the history frame. It reaches
    the recorded curve and nothing else -- the fit never sees it.
    """
    rows = []
    for pair in PAIRS:
        for init, rho in HISTORY_CELLS:
            sigma = float(np.sqrt(_pair_gap_ms(pair) / rho))
            for seed in seeds:
                X_seq, y, z, x, Zg = _make_windows(pair, sigma, N_WINDOWS, seed)
                model = GroupedDCSR(
                    n_clusters=2, alpha_geom=0.0, beta_complexity=0.002,
                    max_iter=10, tol=0.01, backend="fast",
                    backend_kwargs={"max_terms": 5}, random_state=seed,
                    min_windows_per_cluster=4, init=init,
                    geom_metric="mahalanobis", score_mode="cross_fit",
                    residual_scale="global")
                res = model.fit(X_seq, y, Zg, feature_names=["x"],
                                true_labels_for_eval=z)
                h = res.history.copy()
                h.insert(0, "pair", pair.name)
                h.insert(1, "init", init)
                h.insert(2, "rho", rho)
                h.insert(3, "seed", seed)
                rows.append(h)
            if verbose:
                last = rows[-1]
                print(f"  {pair.name:11s} {init:9s} rho {rho:<5g} "
                      f"{len(last)} iterations, "
                      f"accuracy {last.aligned_accuracy.iloc[0]:.2f} -> "
                      f"{last.aligned_accuracy.iloc[-1]:.2f}", flush=True)
    return pd.concat(rows, ignore_index=True)


def run(verbose=True):
    print("\n[search] the greedy path", flush=True)
    P = run_path(verbose=verbose)
    print("\n[search] complexity against the noise floor", flush=True)
    C = run_complexity(P)
    print("\n[search] the assignment's complexity penalty", flush=True)
    B = run_beta(verbose=verbose)
    print("\n[search] the loop, iteration by iteration", flush=True)
    H = run_history()
    for name, df in (("path", P), ("complexity", C), ("beta", B),
                     ("history", H)):
        df.to_csv(RESULTS / f"search_{name}.csv", index=False)
    print(f"\n[search] {len(P)} path rows, {len(C)} complexity rows, "
          f"{len(B)} beta rows, {len(H)} iterations -> {RESULTS}")
    return P, C, B, H


if __name__ == "__main__":
    run()
