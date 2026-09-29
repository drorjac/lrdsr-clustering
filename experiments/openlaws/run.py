"""Laws outside every library: a final open-ended search per regime.

    python -m experiments openlaws          # needs PySR (pip install "pysr<2")

The fast engine chooses among fixed terms, so a law outside its library can
only come back as a look-alike; the fitted-frequency term (``fast_sin``)
widens the vocabulary by one family. Open-ended symbolic regression (PySR)
builds formulas from operators and fits the constants inside them, but is
far too slow to run on every round of the loop. The design tested here:

1. LR-DSR finds the regimes with a fast engine (unchanged);
2. the optional block 5, ``lrdsr.core.refine.refine_laws``: PySR runs **once
   per final regime** and its law replaces the loop's only if it lowers the
   group's cost on the same objective;
3. one extra reassignment (block 4, once more).

The battery: four pairs, each law outside BOTH the fixed library and the
fitted-frequency family, written down before any fit.

==================  =====================================================
``chirp``           sin(0.8 x^2)  vs  sin(1.0 x^2)            x in [-3, 3]
``damped``          e^(-0.4x) sin(3x)  vs  e^(-0.1x) sin(3x)  x in [0, 4]
``am_sine``         x sin(4x)  vs  x sin(4.6x)                x in [-3, 3]
``lorentzian``      1/(1 + x^2)  vs  1/(1 + 2 x^2)            x in [-3, 3]
==================  =====================================================

150 windows of 48 samples, ``rho`` in {0.25, 1.0}, reporting seeds. Arms:
the oracle; LR-DSR with the library (``fast``) and with the fitted frequency
(``fast_sin``), each with its own laws; and the final step on the
``fast_sin`` partition -- PySR laws, and the labels after the extra round.

Law error: RMS distance between a true law and its matched recovered law on
a dense grid, in units of the RMS gap between the two true laws (1.0 = as
far off as the laws are from each other).

PySR: operators + - * / sin cos exp, maxsize 25, 40 iterations,
deterministic and serial with the seed as random state -- slower, but a run
that returns a different law each time cannot support a claim that a law
was recovered.

Changed after the smoke test, before the reported run (2026-09-29), and
said so: the smoke test (damped, seed 11 -- a REPORTING seed, which is why
this is recorded) showed PySR's default formula choice ("best") dropping the
decay envelope and the extra reassignment making labels worse. On tuning
seed 3 with the true partition, "best" and "accuracy" were compared with a
held-out 1-SE choice (fit on half the windows, keep the simplest formula
within one standard error of the best held-out error); 1-SE won and is used
(``pysr_holdout_engine``, 1000 samples per fit). The keep-only-if-better
guard of block 5 was added at the same time, and after one more check on
tuning seed 3 it compares the two laws on HELD-OUT windows (both fitted on
half of the group, judged on the other half): judged in-sample, the loop's
law always wins, because it was fitted on those windows. The predictions below are
unchanged; O2/O3 score the laws PySR FOUND (before the guard), O4 the labels
after block 5.

Declared before the run (2026-09-29):
  O1  the fast library's laws are look-alikes: law error >= 0.2 on chirp
      and am_sine at rho = 1.
  O2  PySR at the end recovers the law (law error < 0.1) in at least 75% of
      the (problem, seed, regime) cases at rho = 1.
  O3  ... and in at least 50% at rho = 0.25.
  O4  the extra reassignment with PySR laws never raises the mean error of
      a (problem, rho) cell by more than 0.005, and lowers it wherever the
      fast_sin partition was more than 0.05 above the oracle.

Writes ``results/openlaws/openlaws_{raw,laws,verdict}.csv``.  ~30-40 minutes.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from experiments.common.fitting import window_features
from experiments.problems.sinusoid_trial import fit as fit_lrdsr_backend
from experiments.problems.zoo import (
    Problem,
    _u,
    eval_grid,
    make_windows,
    oracle_labels,
    sigma_for_rho,
)
from lrdsr import paths
from lrdsr.core.evaluation import aligned_accuracy
from lrdsr.core.refine import pysr_holdout_engine, refine_laws

RESULTS = paths.results_dir("openlaws", figs=False)
REPORT_SEEDS = (11, 23, 42)
RHO_GRID = (0.25, 1.0)
N_WINDOWS, WINDOW_LEN = 150, 48
ENGINE = None  # built lazily in run(): PySR is imported only when this block runs
PYSR_OPTIONS = {
    "niterations": 40, "binary_operators": ["+", "-", "*", "/"],
    "unary_operators": ["sin", "cos", "exp"], "maxsize": 25,
    "model_selection": "best", "deterministic": True, "parallelism": "serial",
    "progress": False, "verbosity": 0,
}


def _x(X):
    return X[..., 0]


PROBLEMS = (
    Problem("chirp", (lambda X: np.sin(0.8 * _x(X) ** 2), lambda X: np.sin(1.0 * _x(X) ** 2)),
            ("sin(0.8*x^2)", "sin(1.0*x^2)"), _u(-3.0, 3.0), 1, (-3.0, 3.0), False,
            "frequency that grows with x: no fixed term, no single fitted frequency"),
    Problem("damped", (lambda X: np.exp(-0.4 * _x(X)) * np.sin(3 * _x(X)),
                       lambda X: np.exp(-0.1 * _x(X)) * np.sin(3 * _x(X))),
            ("exp(-0.4*x)*sin(3*x)", "exp(-0.1*x)*sin(3*x)"), _u(0.0, 4.0), 1, (0.0, 4.0), False,
            "same oscillation, different decay: the envelope is outside every library"),
    Problem("am_sine", (lambda X: _x(X) * np.sin(4 * _x(X)), lambda X: _x(X) * np.sin(4.6 * _x(X))),
            ("x*sin(4*x)", "x*sin(4.6*x)"), _u(-3.0, 3.0), 1, (-3.0, 3.0), False,
            "a fast sine whose amplitude grows with x"),
    Problem("lorentzian", (lambda X: 1 / (1 + _x(X) ** 2), lambda X: 1 / (1 + 2 * _x(X) ** 2)),
            ("1/(1+x^2)", "1/(1+2*x^2)"), _u(-3.0, 3.0), 1, (-3.0, 3.0), False,
            "a rational bump: smooth, so the library approximates it -- but not exactly"),
)


def _matching(z, labels, K) -> dict:
    """true regime -> predicted slot, Hungarian on the contingency table."""
    C = np.zeros((K, K))
    for t, p in zip(z, labels, strict=True):
        C[int(t), int(p)] += 1
    r, c = linear_sum_assignment(-C)
    return dict(zip(r, c, strict=True))


def law_errors(problem, z, labels, predict_fns) -> list[float]:
    """Per true regime: RMS(true - matched recovered) / RMS(gap between true laws)."""
    Xg = eval_grid(problem, 800)
    gap = np.sqrt(np.mean((problem.laws[1](Xg) - problem.laws[0](Xg)) ** 2))
    mp = _matching(z, labels, problem.K)
    return [float(np.sqrt(np.mean((f(Xg) - np.asarray(predict_fns[mp[k]](Xg)).ravel()) ** 2)) / gap)
            for k, f in enumerate(problem.laws)]


def run_cell(problem, rho, seed):
    X, y, z = make_windows(problem, sigma_for_rho(problem, rho), N_WINDOWS, WINDOW_LEN, seed)
    Zf, _ = window_features(X, y)
    rows, laws = [], []
    oracle_err = 1.0 - float(np.mean(oracle_labels(problem, X, y) == z))
    base = {"problem": problem.name, "rho": rho, "seed": seed, "oracle": oracle_err}

    fits = {be: fit_lrdsr_backend(be, X, y, Zf, ["x"], problem.K, seed) for be in ("fast", "fast_sin")}
    for be, res in fits.items():
        errs = law_errors(problem, z, res.labels, [m.predict for m in res.models])
        rows.append({**base, "arm": f"lrdsr_{be}", "error": 1.0 - aligned_accuracy(z, res.labels),
                     "law_error_mean": float(np.mean(errs)), "seconds": np.nan})
        for k, e in enumerate(errs):
            laws.append({**base, "arm": f"lrdsr_{be}", "true_law": problem.truths[k], "law_error": e,
                         "recovered": res.models[_matching(z, res.labels, problem.K)[k]].expression()})

    # optional block 5 on the fast_sin result: PySR once per group, kept only if it lowers the cost
    t = time.time()
    res = fits["fast_sin"]
    out = refine_laws(res, X, y, engine=ENGINE, seed=seed)
    secs = time.time() - t
    errs = law_errors(problem, z, out.labels, [m.predict for m in out.models])
    rows.append({**base, "arm": "refine_block5", "error": 1.0 - aligned_accuracy(z, out.labels),
                 "law_error_mean": float(np.mean(errs)), "seconds": secs,
                 "replaced": int(sum(out.replaced))})
    # the laws PySR FOUND (before the guard), scored on the partition they were searched on
    found = [f if f is not None else res.models[j] for j, f in enumerate(out.found_models)]
    found_errs = law_errors(problem, z, res.labels, [f.predict for f in found])
    mp = _matching(z, res.labels, problem.K)
    for k, e in enumerate(found_errs):
        j = mp[k]
        laws.append({**base, "arm": "pysr_found", "true_law": problem.truths[k], "law_error": e,
                     "recovered": out.expressions_found[j], "replaced": out.replaced[j],
                     "cost_loop": out.cost_before[j], "cost_found": out.cost_found[j]})
    return rows, laws


def verdicts(raw: pd.DataFrame, laws: pd.DataFrame) -> pd.DataFrame:
    lib = laws[(laws.arm == "lrdsr_fast") & (laws.rho == 1.0) & laws.problem.isin(["chirp", "am_sine"])]
    ps = laws[laws.arm == "pysr_found"]
    rec = ps.assign(ok=ps.law_error < 0.1).groupby("rho").ok.mean()
    cell = raw.groupby(["problem", "rho", "arm"]).error.mean().unstack("arm")
    delta = cell.refine_block5 - cell.lrdsr_fast_sin
    far = cell.lrdsr_fast_sin - raw.groupby(["problem", "rho"]).oracle.mean() > 0.05
    o4 = bool((delta <= 0.005).all() and (delta[far] < 0).all())
    out = [
        ("O1", "library laws are look-alikes: law error >= 0.2 (chirp, am_sine, rho = 1)",
         float(lib.law_error.min()), bool((lib.law_error >= 0.2).all())),
        ("O2", "PySR finds the law (< 0.1) in >= 75% of cases at rho = 1",
         float(rec.get(1.0, np.nan)), bool(rec.get(1.0, 0) >= 0.75)),
        ("O3", "... and in >= 50% at rho = 0.25", float(rec.get(0.25, np.nan)),
         bool(rec.get(0.25, 0) >= 0.5)),
        ("O4", "block 5: never worse by > 0.005, better where fast_sin was > 0.05 off",
         float(delta.max()), o4),
    ]
    return pd.DataFrame(out, columns=["prediction", "claim", "value", "held"])


def run(args=None) -> pd.DataFrame:
    global ENGINE
    ENGINE = pysr_holdout_engine(**{k: v for k, v in PYSR_OPTIONS.items() if k != "model_selection"})
    t0 = time.time()
    rows, laws = [], []
    for p in PROBLEMS:
        for rho in RHO_GRID:
            for s in REPORT_SEEDS:
                r, lw = run_cell(p, rho, s)
                rows += r
                laws += lw
                print(f"  {p.name:11s} rho={rho:<5} seed={s}  "
                      + "  ".join(f"{x['arm']}={x['error']:.3f}" for x in r), flush=True)
    raw, laws = pd.DataFrame(rows), pd.DataFrame(laws)
    raw.to_csv(RESULTS / "openlaws_raw.csv", index=False)
    laws.to_csv(RESULTS / "openlaws_laws.csv", index=False)
    v = verdicts(raw, laws)
    v.to_csv(RESULTS / "openlaws_verdict.csv", index=False)
    with pd.option_context("display.width", 220, "display.precision", 3, "display.max_colwidth", 60):
        print(raw.groupby(["problem", "rho", "arm"])[["error", "law_error_mean"]].mean().unstack("arm"))
        print(laws[(laws.arm == "pysr_found")][["problem", "rho", "seed", "true_law", "recovered", "law_error",
                                                "replaced"]]
              .to_string(index=False))
        print(v.to_string(index=False))
    print(f"\n[openlaws] {len(raw)} rows in {(time.time() - t0) / 60:.1f} min -> {RESULTS}")
    return raw


if __name__ == "__main__":
    run()
