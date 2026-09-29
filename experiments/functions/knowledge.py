"""the functions block -- what is declared mechanism knowledge worth, and when?

The three-law mixture of ``data.py`` -- $f_1$ quadratic, $f_2$ sine, $f_3$
cubic, geometry overlapping and laws unrelated -- fitted under three framings
of how much is known about $f_k$ before the data arrive:

==============  ==========================================================
``known``       the exact law, nothing fitted (``mode='known'``). The upper
                bound: what the estimator returns when the physics is
                perfect.
``known_form``  the FORM is known, its coefficients are not, fitted per
                regime by bounded least squares (``mode='factory'``). This
                is the CML case -- ``a R^b`` is known to be a power law and
                ``a, b`` are not -- reproduced with no rain physics in it.
``unknown``     full symbolic search (``mode='unknown'``), the project
                default backend.
==============  ==========================================================

Two knobs, and the second is the one the paper cares about:

* **noise.** Knowledge cannot matter where the signal is clean and geometry
  already solves the problem; the framings should separate as noise grows.
* **window length $n$.** The original benchmark was row-level -- one sample
  per label, ``n = 1`` -- which is the regime where a declared law is worth
  most, because nothing else can pin the regime down. Everything else in
  this paper is about $n$ samples sharing a label, where \\S2 says the
  evidence grows like $\\sqrt{n\\rho}$. Sweeping $n$ measures how fast
  knowing the law stops mattering.

Baselines at every cell: the best of seven geometric clusterings, and (for
$n > 1$) $K$-means in mechanism space, which declares nothing but a library.

Two heavier search paradigms -- PySR (genetic programming) and PhySO
(reinforcement learning) -- were run against this same battery in an
earlier, unpublished exploration. Neither beat the deterministic backend
there, and both cost hours, so they are not re-run here.

Writes ``results/functions/``. Seeds {11, 23, 42}.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.functions.data import DEFAULT_COMPONENTS, simulate_function_windows
from lrdsr import paths
from lrdsr.core.backends import ParametricPriorRegressor
from lrdsr.core.baselines import geometry_baselines
from lrdsr.core.evaluation import clustering_metrics
from lrdsr.core.mechanism_space import mechanism_init
from lrdsr.core.model import GroupedDCSR

RESULTS = paths.results_dir("functions", figs=False)
REPORT_SEEDS = (11, 23, 42)
TUNE_SEEDS = (3, 7, 19)
#: alpha_geom is problem-dependent everywhere in this project, and at n = 1 it
#: is decisive: a single sample cannot identify a law, so the geometry term is
#: the only thing left for the 'unknown' framing to stand on. It is therefore
#: selected here rather than assumed -- on the TUNING seeds, at the middle
#: noise level, averaged over the three framings so the choice cannot favour
#: one of them. (The archived sweep fixed it at the CLI default 0.25.)
ALPHA_CANDIDATES = (0.0, 0.25, 0.5)

N_WINDOWS = 150
#: n = 1 is the original row-level benchmark; 8 is a real CML window; 64 is
#: the window length the rest of the paper uses.
WINDOW_LENS = (1, 8, 64)
#: The archived sweep stopped at 1.2, which is saturated for every framing
#: once n > 1 -- at n = 64 the three laws are trivially separable there. The
#: range is extended so each window length has a cell where the question is
#: live; 0.12/0.6/1.2 are the original levels, unchanged.
NOISE_LEVELS = (("low", 0.12), ("medium", 0.6), ("high", 1.2),
                ("very high", 2.4), ("extreme", 4.8))
FRAMINGS = ("known", "known_form", "unknown")

#: Slot order is fixed by the mean of y, which is observable and label-free:
#: the population means are quadratic +1.79, sine -0.5, cubic -1.2, so
#: sorting ascending puts cubic, sine, quadratic in slots 0, 1, 2 and a
#: declared law lands on the regime it describes.
_SLOT_ORDER = ("cubic", "sine", "quadratic")


def _known_specs():
    """The exact three laws, nothing fitted, in slot order."""
    return [
        {"mode": "known", "expression": "0.35*x^3 - 1.2",
         "function": lambda X: 0.35 * X[:, 0] ** 3 - 1.2},
        {"mode": "known", "expression": "2.5*sin(2*x) - 0.5",
         "function": lambda X: 2.5 * np.sin(2.0 * X[:, 0]) - 0.5},
        {"mode": "known", "expression": "0.8*x^2 + 1.5*x + 0.5",
         "function": lambda X: 0.8 * X[:, 0] ** 2 + 1.5 * X[:, 0] + 0.5},
    ]


def _known_form_specs():
    """The form only, coefficients free and bounded inside their own family.

    The bounds are what stop a "known form" from quietly absorbing another
    regime's shape -- the same discipline the CML path prior uses.
    """
    def cubic_form(X, theta):
        return theta[0] * X[:, 0] ** 3 + theta[1]

    def sine_form(X, theta):
        return theta[0] * np.sin(theta[1] * X[:, 0]) + theta[2]

    def quad_form(X, theta):
        return theta[0] * X[:, 0] ** 2 + theta[1] * X[:, 0] + theta[2]

    return [
        {"mode": "factory", "expression": "a x^3 + b",
         "factory": lambda: ParametricPriorRegressor(
             prior_form=cubic_form, p0=[0.2, 0.0],
             bounds=([-3.0, -4.0], [3.0, 4.0]),
             expression_template="{0:.4g}*x^3 + {1:.3g}")},
        {"mode": "factory", "expression": "a sin(b x) + c",
         "factory": lambda: ParametricPriorRegressor(
             prior_form=sine_form, p0=[1.5, 1.5, 0.0],
             bounds=([0.2, 0.3, -4.0], [6.0, 4.0, 4.0]),
             expression_template="{0:.4g}*sin({1:.3g}*x) + {2:.3g}")},
        {"mode": "factory", "expression": "a x^2 + b x + c",
         "factory": lambda: ParametricPriorRegressor(
             prior_form=quad_form, p0=[0.5, 0.5, 0.0],
             bounds=([-3.0, -4.0, -4.0], [3.0, 4.0, 4.0]),
             expression_template="{0:.4g}*x^2 + {1:.3g}*x + {2:.3g}")},
    ]


SPECS = {"known": _known_specs, "known_form": _known_form_specs,
         "unknown": lambda: None}


def _fit(X_seq, y_seq, Z, labels, framing, seed, window_len, alpha_geom):
    """One LR-DSR fit under one framing.

    ``init`` is mechanism space wherever it is defined -- a window of one
    sample cannot carry a coefficient vector, so ``n = 1`` falls back to the
    data-space initialiser, which is the honest comparison and not a
    handicap: it is what the original benchmark used.
    """  # noqa: D401
    specs = SPECS[framing]()
    model = GroupedDCSR(
        n_clusters=3, alpha_geom=alpha_geom, backend="fast",
        backend_kwargs={"max_terms": 4}, random_state=seed,
        mechanism_specs=specs, max_iter=10, score_mode="cross_fit",
        residual_scale="global", min_windows_per_cluster=4,
        init="mechanism" if window_len > 1 else "bgmm",
        geom_metric="mahalanobis",
    )
    return model.fit(X_seq, y_seq, Z, feature_names=["x"],
                     true_labels_for_eval=labels,
                     # a declared law binds to a SLOT, so the slots need an
                     # observable order before anything is fitted
                     slot_order_key=(y_seq.mean(axis=1)
                                     if specs is not None else None))


def tune_alpha(n_windows: int = N_WINDOWS) -> dict:
    """Select ``alpha_geom`` per window length on the TUNING seeds.

    The middle noise level only, and the mean ARI over the three framings:
    a value chosen to suit one framing would decide the comparison the
    experiment exists to make.
    """
    _, noise = NOISE_LEVELS[len(NOISE_LEVELS) // 2]
    rows = []
    for n in WINDOW_LENS:
        for alpha in ALPHA_CANDIDATES:
            aris = []
            for seed in TUNE_SEEDS:
                X_seq, y_seq, Z, labels, _ = simulate_function_windows(
                    n_windows=n_windows, window_len=n, noise_std=noise,
                    seed=seed)
                for framing in FRAMINGS:
                    res = _fit(X_seq, y_seq, Z, labels, framing, seed, n, alpha)
                    aris.append(clustering_metrics(
                        labels, np.asarray(res.labels))["ARI"])
            rows.append({"window_len": n, "alpha_geom": alpha,
                         "noise_std": noise, "mean_ari_tuning_seeds":
                         float(np.mean(aris))})
        print(f"  [functions] alpha swept at n={n}", flush=True)
    sweep = pd.DataFrame(rows)
    sweep.to_csv(RESULTS / "knowledge_alpha_sweep.csv", index=False)
    best = {int(n): float(g.sort_values("mean_ari_tuning_seeds").alpha_geom.iloc[-1])
            for n, g in sweep.groupby("window_len")}
    print(sweep.round(3).to_string(index=False))
    print(f"  selected alpha_geom per n: {best}")
    return best


def run_ladder(n_windows: int = N_WINDOWS, alpha: dict | None = None) -> pd.DataFrame:
    """The knowledge ladder over (noise, window length, framing, seed).

    Committed prediction, written before the run: the framings are
    indistinguishable at low noise, they separate as noise grows (this is the
    archived result), and the separation SHRINKS as ``n`` grows, because
    ``n`` samples sharing a label is itself evidence about which law holds.
    """
    alpha = tune_alpha(n_windows) if alpha is None else alpha
    rows = []
    for level, noise in NOISE_LEVELS:
        for n in WINDOW_LENS:
            for seed in REPORT_SEEDS:
                X_seq, y_seq, Z, labels, eqs = simulate_function_windows(
                    n_windows=n_windows, window_len=n, noise_std=noise,
                    seed=seed)
                base = {"level": level, "noise_std": noise, "window_len": n,
                        "seed": seed, "n_windows": n_windows,
                        "alpha_geom": alpha[n],
                        "n_regimes": len(DEFAULT_COMPONENTS)}

                geo = geometry_baselines(Z, 3, seed=seed)
                best = max(geo.items(),
                           key=lambda kv: clustering_metrics(labels, kv[1])["ARI"])
                rows.append({**base, "framing": "geometry (best of 7)",
                             "declares": "nothing", "baseline": best[0],
                             **clustering_metrics(labels, best[1])})

                if n > 1:
                    lab = mechanism_init(X_seq, y_seq, 3, seed=seed)
                    rows.append({**base, "framing": "mechanism $K$-means",
                                 "declares": "a library",
                                 **clustering_metrics(labels, lab)})

                for framing in FRAMINGS:
                    res = _fit(X_seq, y_seq, Z, labels, framing, seed, n,
                               alpha[n])
                    lab = np.asarray(res.labels)
                    rows.append({
                        **base, "framing": framing,
                        "declares": {"known": "the exact law",
                                     "known_form": "the form",
                                     "unknown": "nothing"}[framing],
                        **clustering_metrics(labels, lab),
                        "expressions": " | ".join(m.expression()
                                                  for m in res.models)})
            print(f"  [functions] noise={level:<6} n={n:<3} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "knowledge_per_seed.csv", index=False)

    agg = (df.groupby(["level", "noise_std", "window_len", "framing", "declares"])
             .agg(ari=("ARI", "mean"), ari_sd=("ARI", "std"),
                  accuracy=("aligned_accuracy", "mean"))
             .reset_index()
             .sort_values(["noise_std", "window_len", "ari"], ascending=[1, 1, 0]))
    agg.to_csv(RESULTS / "knowledge_summary.csv", index=False)

    # what declared knowledge buys, per cell: known minus unknown
    w = agg.pivot_table(index=["level", "noise_std", "window_len"],
                        columns="framing", values="ari")
    gain = pd.DataFrame({
        "known_minus_unknown": w["known"] - w["unknown"],
        "form_minus_unknown": w["known_form"] - w["unknown"],
        "unknown_minus_geometry": w["unknown"] - w["geometry (best of 7)"],
    }).reset_index()
    if "mechanism $K$-means" in w:
        gain["known_minus_mechanism"] = (w["known"]
                                         - w["mechanism $K$-means"]).values
    gain.to_csv(RESULTS / "knowledge_gain.csv", index=False)

    print()
    print(agg.round(3).to_string(index=False))
    print()
    print(gain.round(3).to_string(index=False))
    return df


def run(args=None) -> None:
    run_ladder()


if __name__ == "__main__":
    run()
