"""Does the ESTIMATOR inherit the oracle's invariance to a shared term?

V4 is unambiguous about the oracle: adding any ``u(x)`` to every regime leaves
the gap, and so the error, exactly unchanged. Nothing said the same of an
estimator, and the previous version of LR-DSR did not have the property -- it
initialised in data space, where a large shared component is most of what a
window looks like.

Mechanism space is built to inherit the invariance: a component shared by the
regimes shifts every window's law by the same amount and is subtracted before
the windows are compared (``lrdsr.core.mechanism_space``). That is a
prediction with a number attached, and this is the experiment that could have
falsified it.

The sweep is the microwave link in miniature. The gap is a saturating
wet-antenna term; the shared component is a rain-path-like term multiplied by
``c``, so ``c`` is the path-to-antenna ratio a longer link would have. The
separation ``rho`` and the noise are held FIXED, so the oracle error is
constant by construction and every change is the estimator's.

Two shared components are used, and the contrast between them is the point:

* ``x^2``     -- INSIDE the symbolic library, where the invariance is claimed;
* ``sin(9x)`` -- OUTSIDE it, where it is not, and the honest limit shows.

Writes ``results/estimator/shared_component*.csv``. Seeds {11, 23, 42}.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from experiments.common.fitting import fit_lrdsr, window_features
from lrdsr import paths
from lrdsr.core.evaluation import aligned_accuracy
from lrdsr.core.mechanism_space import (
    mechanism_features,
    mechanism_init,
    mechanism_noise,
    separation_matrix,
    separation_unlabelled,
)

RESULTS = paths.results_dir("estimator", figs=False)
REPORT_SEEDS = (11, 23, 42)

WINDOW_LEN = 64
N_WINDOWS = 150
RHO = 0.25                      # fixed: the oracle error is the same everywhere
X_RANGE = (0.2, 4.0)
#: Shared-component multipliers. c = 0 is the pure-gap control.
C_GRID = (0.0, 1.0, 4.0, 16.0, 64.0)

#: The gap: the same saturating wet-antenna preview as the benchmark's
#: 'saturation' pair. It carries magnitude, so data-space summaries CAN find
#: it at c = 0 -- which is what makes the collapse at larger c a measurement
#: and not a rigged comparison.
def gap(x):
    return 1.5 * (1.0 - np.exp(-x))


SHARED = {
    "x^2": (lambda x: x ** 2, "inside the library"),
    "sin(9x)": (lambda x: np.sin(9.0 * x), "outside the library"),
}


def _matched_error(true, pred):
    return 1.0 - aligned_accuracy(true, pred)


def _windows(u, c, sigma, seed):
    """Both regimes share ``c * u(x)``; they differ only by ``gap(x)``."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(*X_RANGE, size=(N_WINDOWS, WINDOW_LEN))
    z = rng.integers(0, 2, size=N_WINDOWS)
    clean = c * u(x) + np.where(z[:, None] == 1, gap(x), 0.0)
    y = clean + rng.normal(0.0, sigma, size=(N_WINDOWS, WINDOW_LEN))
    return x[:, :, None], y, z, x


def _oracle(x, y, c, u):
    f0 = c * u(x)
    res = np.column_stack([np.sum((y - f) ** 2, axis=1) for f in (f0, f0 + gap(x))])
    return res.argmin(axis=1)


def run_sweep() -> pd.DataFrame:
    """Committed prediction, written before the run: with the shared component
    inside the library the mechanism-space error is FLAT in ``c``, while the
    data-space error rises from the oracle to chance over the same sweep.
    Outside the library the prediction is that mechanism space fails too, and
    that the failure is visible as a loss of measured separation."""
    xs = np.random.default_rng(0).uniform(*X_RANGE, 400_000)
    gap_ms = float(np.mean(gap(xs) ** 2))
    sigma = float(np.sqrt(gap_ms / RHO))

    rows = []
    for shared_name, (u, where) in SHARED.items():
        for c in C_GRID:
            for seed in REPORT_SEEDS:
                X_seq, y, z, x = _windows(u, c, sigma, seed)
                ratio = float(np.sqrt(np.mean((c * u(x)) ** 2) / gap_ms))
                oracle = _oracle(x, y, c, u)

                # Both arms see the SAME window features; only the initial
                # partition differs, so this isolates the representation.
                Zf, _ = window_features(X_seq, y)
                mech = fit_lrdsr(Zf, None, X_seq, y, ["x"], 2, seed,
                                 alpha_geom=0.0, init="mechanism")
                data = fit_lrdsr(Zf, None, X_seq, y, ["x"], 2, seed,
                                 alpha_geom=0.0, init="bgmm")
                # and the two representations clustered directly, no alternation
                mech_km = mechanism_init(X_seq, y, 2, seed=seed)
                data_km = KMeans(2, n_init=30, random_state=seed).fit_predict(
                    StandardScaler().fit_transform(Zf))

                S = mechanism_features(X_seq, y)
                sig = mechanism_noise(X_seq, y)
                unl = separation_unlabelled(S, sig, WINDOW_LEN)
                rows.append({
                    "shared": shared_name, "where": where, "c": c,
                    "shared_to_gap": ratio, "seed": seed, "rho": RHO,
                    "sigma": sigma, "window_len": WINDOW_LEN,
                    "n_windows": N_WINDOWS,
                    "oracle_error": _matched_error(z, oracle),
                    "lrdsr_mechanism_error": _matched_error(z, mech.labels),
                    "lrdsr_dataspace_error": _matched_error(z, data.labels),
                    "kmeans_mechanism_error": _matched_error(z, mech_km),
                    "kmeans_dataspace_error": _matched_error(z, data_km),
                    "separation_measured": float(
                        separation_matrix(S, z, 2, sigma=sig)[0, 1]),
                    "separation_predicted": float(np.sqrt(WINDOW_LEN * RHO)),
                    "rho_hat": unl["rho"],
                })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "shared_component.csv", index=False)

    agg = (df.groupby(["shared", "where", "c"])
             .agg(shared_to_gap=("shared_to_gap", "mean"),
                  oracle=("oracle_error", "mean"),
                  lrdsr_mechanism=("lrdsr_mechanism_error", "mean"),
                  lrdsr_mechanism_sd=("lrdsr_mechanism_error", "std"),
                  lrdsr_dataspace=("lrdsr_dataspace_error", "mean"),
                  lrdsr_dataspace_sd=("lrdsr_dataspace_error", "std"),
                  kmeans_mechanism=("kmeans_mechanism_error", "mean"),
                  kmeans_dataspace=("kmeans_dataspace_error", "mean"),
                  sep_measured=("separation_measured", "mean"),
                  sep_predicted=("separation_predicted", "mean"),
                  rho_hat=("rho_hat", "mean"))
             .reset_index())
    agg.to_csv(RESULTS / "shared_component_summary.csv", index=False)

    verdict = []
    for name, g in agg.groupby("shared"):
        g = g.sort_values("c")
        lo, hi = g.iloc[0], g.iloc[-1]
        verdict.append({
            "shared": name, "where": g["where"].iloc[0],
            "c_max": float(hi.c), "shared_to_gap_max": float(hi.shared_to_gap),
            "oracle_drift": float(g.oracle.max() - g.oracle.min()),
            "mechanism_at_c0": float(lo.kmeans_mechanism),
            "mechanism_at_cmax": float(hi.kmeans_mechanism),
            "mechanism_drift": float(g.kmeans_mechanism.max()
                                     - g.kmeans_mechanism.min()),
            "dataspace_at_c0": float(lo.kmeans_dataspace),
            "dataspace_at_cmax": float(hi.kmeans_dataspace),
            "dataspace_drift": float(g.kmeans_dataspace.max()
                                     - g.kmeans_dataspace.min()),
            "lrdsr_mechanism_drift": float(g.lrdsr_mechanism.max()
                                           - g.lrdsr_mechanism.min()),
            "lrdsr_dataspace_drift": float(g.lrdsr_dataspace.max()
                                           - g.lrdsr_dataspace.min()),
            "sep_at_c0": float(lo.sep_measured),
            "sep_at_cmax": float(hi.sep_measured),
        })
    ver = pd.DataFrame(verdict)
    ver.to_csv(RESULTS / "shared_component_verdict.csv", index=False)

    print(agg.round(4).to_string(index=False))
    print()
    print(ver.round(4).to_string(index=False))
    return df


def run(args=None) -> None:
    run_sweep()
    print(f"\n[estimator/shared_component] CSVs -> {RESULTS}")


if __name__ == "__main__":
    run()
