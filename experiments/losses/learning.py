"""Learning the loss: can the noise law be read off the residuals?

A loss is a noise model in disguise, so choosing one is a claim about the
noise -- a claim the residuals of a fitted law can test. Two ways of learning
it are measured here, neither of which reads a label.

``trace``
    The loss each estimator settles on, iteration by iteration:
    ``GroupedDCSR(loss='learned')`` re-fits a Gaussian / Laplace / Student-t
    noise model to the pooled residuals of its current partition every
    iteration (``history`` columns ``loss`` and ``loss_nu``);
    ``SoftLRDSR(noise='student_t')`` fits one ``nu`` per regime inside EM.
    Polynomial pair, ``rho = 0.25``, the four noise laws of
    ``experiments.losses.estimator``.

``family``
    ``lrdsr.core.losses.learn_loss`` on pure noise, where the answer is known:
    how often it names the right family, and what ``nu`` it fits, against
    sample size. The contaminated law is in no candidate family; the
    Student-t is the nearest heavy-tailed one, and that is what it should
    say.

Writes ``loss_learning_trace.csv`` and ``loss_learning_family.csv``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.common.fitting import window_features
from experiments.estimator.benchmark import PAIR_BY_NAME
from experiments.losses.estimator import _hard, make_windows
from lrdsr.core.losses import learn_loss
from lrdsr.core.mechanism_space import mechanism_init
from lrdsr.core.soft import SoftLRDSR
from lrdsr.protocol import REPORT_SEEDS
from lrdsr.theory.losses import RESULTS, noise_families

NOISES = ("gaussian", "laplace", "student_t3", "contaminated_10")
#: What the right answer is, where there is one.
TRUE_FAMILY = {"gaussian": "squared", "laplace": "absolute",
               "student_t3": "student_t", "contaminated_10": "student_t"}
TRUE_NU = {"gaussian": np.inf, "laplace": np.nan, "student_t3": 3.0,
           "contaminated_10": np.nan}
SAMPLE_SIZES = (100, 1_000, 10_000)
#: A Student-t this light-tailed is a Gaussian for every practical purpose:
#: the t nests the Gaussian, so on Gaussian residuals it ties with the
#: squared loss and wins about half the time by sampling noise, at ``nu``
#: near the bound. ``share_effective`` counts those picks as ``squared``.
NU_GAUSSIAN = 30.0
REPS = 30


def run_trace(seeds=REPORT_SEEDS, rho: float = 0.25) -> pd.DataFrame:
    fams = noise_families()
    pair = PAIR_BY_NAME["polynomial"]
    rows = []
    for nname in NOISES:
        for seed in seeds:
            X, y, z, _x, _s = make_windows(pair, fams[nname], rho, seed)
            Z, _ = window_features(X, y)
            init = mechanism_init(X, y, 2, seed=seed, feature_names=["x"])
            base = {"noise": nname, "seed": seed, "true_nu": TRUE_NU[nname]}
            h = _hard(X, y, Z, init, seed, "learned", {}).history
            for _, r in h.iterrows():
                rows.append({**base, "method": "hard_learned", "iteration": int(r.iteration),
                             "regime": -1, "family": r["loss"],
                             "nu": float(r["loss_nu"]) if "loss_nu" in h and
                             pd.notna(r.get("loss_nu")) else np.nan})
            s = SoftLRDSR(2, noise="student_t", feature_names=["x"], init=init,
                          random_state=seed).fit(X, y).history
            for _, r in s.iterrows():
                for k in (0, 1):
                    rows.append({**base, "method": "soft_student_t",
                                 "iteration": int(r.iteration), "regime": k,
                                 "family": "student_t", "nu": float(r[f"nu_{k}"])})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "loss_learning_trace.csv", index=False)
    return df


def run_family(seeds=REPORT_SEEDS) -> pd.DataFrame:
    fams = noise_families()
    rows = []
    for nname in NOISES:
        noise = fams[nname]
        for n in SAMPLE_SIZES:
            picks, nus = [], []
            for seed in seeds:
                rng = np.random.default_rng(seed)
                for _ in range(REPS):
                    got = learn_loss(noise.sample(rng, n))
                    picks.append(got.loss)
                    nus.append(got.params.get("nu", np.nan))
            picks = np.array(picks)
            nus = np.array(nus, dtype=float)
            eff = np.where((picks == "student_t") & (nus >= NU_GAUSSIAN), "squared", picks)
            for fam in ("squared", "absolute", "student_t"):
                sel = picks == fam
                rows.append({
                    "true_noise": nname, "n_samples": n, "chosen": fam,
                    "share": float(sel.mean()),
                    "share_effective": float(np.mean(eff == fam)),
                    "is_right": fam == TRUE_FAMILY[nname],
                    "median_nu": float(np.nanmedian(nus[sel])) if fam == "student_t"
                    and sel.any() else np.nan,
                    "trials": len(picks),
                })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "loss_learning_family.csv", index=False)
    return df


def run(verbose: bool = True):
    tr = run_trace()
    fam = run_family()
    if verbose:
        last = (tr.sort_values("iteration").groupby(["noise", "method", "seed", "regime"])
                .tail(1))
        print(last.groupby(["noise", "method"]).agg(
            family=("family", lambda v: v.value_counts().idxmax()),
            nu_median=("nu", "median")).to_string())
        print(fam.pivot_table(index=["true_noise", "n_samples"], columns="chosen",
                              values=["share", "share_effective"]).round(2).to_string())
    return tr, fam


if __name__ == "__main__":
    run()
