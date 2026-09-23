"""Does LR-DSR reach the oracle when the geometry carries nothing?

Four pairs of symbolic laws. In every pair both regimes draw ``x`` from the
IDENTICAL marginal, so neither the covariate nor a planted feature geometry
carries regime information and the equation term must carry the assignment
alone. LR-DSR is scored against the known-law oracle (the Bayes ceiling) and
against the best geometric baseline, on two feature sets:

* a planted, deliberately near-uninformative geometry -- the clean test;
* plain window summaries of ``(x, y)`` -- the fair one: they DO separate pairs
  that differ in magnitude, and that contrast is the point.

Writes ``results/estimator/``. Seeds {11, 23, 42}; nothing is tuned here
(``alpha_geom = 0`` is the principled value when geometry carries nothing).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from experiments.common.fitting import fit_lrdsr, window_features
from lrdsr import paths
from lrdsr.core.baselines import geometry_baselines
from lrdsr.core.evaluation import aligned_accuracy
from lrdsr.core.mechanism_space import (
    mechanism_features,
    mechanism_init,
    mechanism_noise,
    separation_matrix,
)

RESULTS = paths.results_dir("estimator", figs=False)
REPORT_SEEDS = (11, 23, 42)
ALPHA_GEOM_G2 = 0.0


def _matched_error(true: np.ndarray, pred: np.ndarray) -> float:
    """Per-window classification error after optimal permutation matching."""
    return 1.0 - aligned_accuracy(true, pred)


@dataclass(frozen=True)
class Pair:
    name: str
    f0: Callable[[np.ndarray], np.ndarray]
    f1: Callable[[np.ndarray], np.ndarray]
    truth0: str
    truth1: str
    x_range: tuple[float, float]
    why: str


PAIRS = (
    Pair("linear", lambda x: 2 * x + 1, lambda x: -x + 3,
         "2*x + 1", "-x + 3", (-2.0, 2.0), "sanity, large rho"),
    Pair("polynomial", lambda x: x ** 2 + x, lambda x: x ** 2 - x,
         "x^2 + x", "x^2 - x", (-2.0, 2.0), "large COMMON x^2 cancels"),
    Pair("oscillatory", lambda x: np.sin(x) + 0.2 * x, lambda x: np.cos(x) + 0.2 * x,
         "sin(x) + 0.2*x", "cos(x) + 0.2*x", (-2.0, 2.0),
         "common trend cancels, non-monotone gap"),
    # x^b needs x > 0, so this pair alone draws x ~ U[0.2, 4]; both regimes
    # still share the identical marginal, which is what G2 requires.
    Pair("saturation", lambda x: 0.4 * x ** 1.1,
         lambda x: 0.4 * x ** 1.1 + 1.5 * (1.0 - np.exp(-x / 1.0)),
         "0.4*x^1.1", "0.4*x^1.1 + 1.5*(1 - exp(-x))", (0.2, 4.0),
         "generic preview of the CML wet-antenna term"),
)
PAIR_BY_NAME = {p.name: p for p in PAIRS}

# The spec's common grid is {0.25, 1, 4}; 0.1 is added below it so there is a
# resolvable oracle error (and therefore a resolvable learning gap) at n = 64,
# where rho >= 0.25 is already near error-free.
RHO_GRID = (0.1, 0.25, 1.0, 4.0)
WINDOW_LEN = 64
N_WINDOWS = 150

#: The planted geometry every baseline clusters on: two Gaussian blobs with a
#: large spread, so P(Z | z=0) ~ P(Z | z=1), lifted to windows. The
#: equation term is then the ONLY thing that can recover the partition.
GEOM_CENTERS = np.array([[1.2, 0.0], [-1.2, 0.0]])
GEOM_OVERLAP = 6.0


def _pair_gap_ms(pair: Pair, seed: int = 0, n: int = 400_000) -> float:
    """E[(f1 - f0)^2] under x ~ U(pair.x_range) -- fixes sigma for a target rho."""
    x = np.random.default_rng(seed).uniform(*pair.x_range, n)
    return float(np.mean((pair.f1(x) - pair.f0(x)) ** 2))


def _make_windows(pair: Pair, sigma: float, m: int, seed: int):
    """m windows of ``WINDOW_LEN`` samples. z ~ Bernoulli(1/2) is ground truth
    (kept for metrics only). BOTH regimes draw x from the identical marginal,
    and the planted geometry ``Zg`` is deliberately near-uninformative, so
    neither the covariate nor the feature geometry carries a regime signal.

    Returns ``(X_seq, y, z, x, Zg)``.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(*pair.x_range, size=(m, WINDOW_LEN))
    z = rng.integers(0, 2, size=m)
    clean = np.where(z[:, None] == 0, pair.f0(x), pair.f1(x))
    y = clean + rng.normal(0.0, sigma, size=(m, WINDOW_LEN))
    Zg = rng.normal(GEOM_CENTERS[z], GEOM_OVERLAP, size=(m, 2))
    return x[:, :, None], y, z, x, Zg


def _oracle_labels(x: np.ndarray, y: np.ndarray, laws) -> np.ndarray:
    """Smaller residual sum under the two TRUE laws -- the Bayes rule."""
    res = np.column_stack([np.sum((y - f(x)) ** 2, axis=1) for f in laws])
    return res.argmin(axis=1)


def _match(true: np.ndarray, pred: np.ndarray) -> dict:
    """Hungarian mapping pred-label -> true-label (for slot alignment)."""
    from scipy.optimize import linear_sum_assignment
    k = max(int(true.max()), int(pred.max())) + 1
    conf = np.zeros((k, k), dtype=int)
    for t, p in zip(true, pred):
        conf[t, p] += 1
    row, col = linear_sum_assignment(-conf)
    return {int(c): int(r) for r, c in zip(row, col)}


def run_benchmark(m: int = N_WINDOWS, rho_grid=RHO_GRID) -> pd.DataFrame:
    """The pair battery: LR-DSR vs the oracle vs the best geometry baseline,
    on windows where the geometry is uninformative by construction.

    Committed prediction: on the planted near-uninformative geometry ``Zg``
    the best geometry baseline scores ~ chance (matched error ~ 0.5) on every
    pair. A second column reports the same baselines run on the
    ``window_features`` summary of ``(x, y)`` --
    those DO separate the magnitude pairs, and that contrast is the point.
    """
    rows, expr_rows = [], []
    for pair in PAIRS:
        ms = _pair_gap_ms(pair)
        for rho in rho_grid:
            sigma = float(np.sqrt(ms / rho))
            for seed in REPORT_SEEDS:
                X_seq, y, z, x, Zg = _make_windows(pair, sigma, m, seed)
                Zf, fnames = window_features(X_seq, y)

                oracle = _oracle_labels(x, y, [pair.f0, pair.f1])
                res = fit_lrdsr(Zg, ["g1", "g2"], X_seq, y, ["x"], 2, seed,
                                alpha_geom=ALPHA_GEOM_G2)
                # ABLATION: the same estimator initialised in data space,
                # which is what LR-DSR did before mechanism space.
                res_ds = fit_lrdsr(Zg, ["g1", "g2"], X_seq, y, ["x"], 2, seed,
                                   alpha_geom=ALPHA_GEOM_G2, init="bgmm")
                # The initialiser ALONE: K-means in mechanism space, no
                # symbolic fit and no alternation.
                mech_lab = mechanism_init(X_seq, y, 2, seed=seed)
                # The separation this space actually shows, in noise units;
                # theory says it is sqrt(n rho). The TRUE labels are used, so
                # this is a measurement reported at the end, never a fit.
                S = mechanism_features(X_seq, y)
                sig = mechanism_noise(X_seq, y)
                sep = float(separation_matrix(S, z, 2, sigma=sig)[0, 1])

                geom = geometry_baselines(Zg, 2, seed=seed)
                geom_err = {b: _matched_error(z, lab) for b, lab in geom.items()}
                best_b = min(geom_err, key=geom_err.get)
                feat = geometry_baselines(Zf, 2, seed=seed)
                feat_err = min(_matched_error(z, lab) for lab in feat.values())

                rows.append({
                    "contribution": 2, "pair": pair.name, "why": pair.why,
                    "rho": rho, "sigma": sigma, "seed": seed, "n_windows": m,
                    "window_len": WINDOW_LEN, "alpha_geom": ALPHA_GEOM_G2,
                    "oracle_error": _matched_error(z, oracle),
                    "lrdsr_error": _matched_error(z, res.labels),
                    "lrdsr_dataspace_error": _matched_error(z, res_ds.labels),
                    "mechanism_kmeans_error": _matched_error(z, mech_lab),
                    "separation_measured": sep,
                    "separation_predicted": float(np.sqrt(WINDOW_LEN * rho)),
                    "best_geometry_baseline": best_b,
                    "best_geometry_error": geom_err[best_b],
                    "best_window_feature_error": feat_err,
                })

                mapping = _match(z, np.asarray(res.labels))
                for pred_slot, model in enumerate(res.models):
                    true_slot = mapping.get(pred_slot, pred_slot)
                    expr_rows.append({
                        "pair": pair.name, "rho": rho, "seed": seed,
                        "true_slot": true_slot,
                        "truth": pair.truth0 if true_slot == 0 else pair.truth1,
                        "recovered": model.expression(),
                    })
    df = pd.DataFrame(rows)
    expr = pd.DataFrame(expr_rows)
    df.to_csv(RESULTS / "benchmark_per_seed.csv", index=False)
    expr.to_csv(RESULTS / "benchmark_expressions.csv", index=False)

    agg = (df.groupby(["pair", "rho"])
             .agg(oracle=("oracle_error", "mean"),
                  lrdsr=("lrdsr_error", "mean"),
                  lrdsr_sd=("lrdsr_error", "std"),
                  lrdsr_dataspace=("lrdsr_dataspace_error", "mean"),
                  mechanism_kmeans=("mechanism_kmeans_error", "mean"),
                  geometry=("best_geometry_error", "mean"),
                  window_features=("best_window_feature_error", "mean"),
                  sep_measured=("separation_measured", "mean"),
                  sep_predicted=("separation_predicted", "mean"))
             .reset_index())
    agg.to_csv(RESULTS / "benchmark_summary.csv", index=False)


    print(agg.to_string(index=False))
    rel = (df.separation_measured / df.separation_predicted - 1.0).abs()
    print(f"\nmean best-geometry error = {df.best_geometry_error.mean():.3f} "
          f"(prediction: ~0.5, chance); "
          f"mean LR-DSR-minus-oracle gap = "
          f"{(df.lrdsr_error - df.oracle_error).mean():+.3f} "
          f"(data-space init: {(df.lrdsr_dataspace_error - df.oracle_error).mean():+.3f})")
    print(f"mechanism-space separation vs sqrt(n rho): "
          f"median relative error {rel.median():.3f}, max {rel.max():.3f}")
    return df


def recovered_laws() -> pd.DataFrame:
    """One representative recovered expression per (pair, slot), beside the
    truth: the median rho, first report seed. Reads
    ``benchmark_expressions.csv``; runs the benchmark first if it is missing."""
    f = RESULTS / "benchmark_expressions.csv"
    if not f.exists():
        run_benchmark()
    expr = pd.read_csv(f)
    # one representative recovery per (pair, slot): the median-rho, first seed
    rho_mid = sorted(expr.rho.unique())[len(expr.rho.unique()) // 2]
    seed0 = sorted(expr.seed.unique())[0]
    view = (expr[(expr.rho == rho_mid) & (expr.seed == seed0)]
            .sort_values(["pair", "true_slot"]))
    view.to_csv(RESULTS / "benchmark_recovered_laws.csv", index=False)
    print(f"recovered laws at rho={rho_mid}, seed={seed0}:")
    print(view[["pair", "true_slot", "truth", "recovered"]].to_string(index=False))
    return view


def run(args=None) -> None:
    run_benchmark()
    recovered_laws()
    print(f"\n[estimator] CSVs -> {RESULTS}")


if __name__ == "__main__":
    run()
