"""The kernel extension: mechanism space over an RKHS instead of a term library.

    python -m experiments kernel

The problem zoo has one real failure: ``high_frequency`` (``sin 4x`` against
``sin 4.6x``), where 37% of the *gap* lies outside the fast library's span
and every method pays for it. The library is a list of terms; a kernel basis
spans a function space whose resolution is one knob, its rank. So:

==================  =======================================================
``sweep``           every zoo problem, every separation, mechanism K-means
                    in a Nystrom (RBF) basis at ranks 4..48: error against
                    rank, and the share of the gap each rank leaves outside
                    its span. The bias-variance curve of the kernel basis.
``rule``            which *label-free* rank rule to use -- per-window
                    leave-one-out (``min`` / ``1se``), or the spectral SNR
                    of mechanism space (its maximum, the smallest rank
                    within one bootstrap SE of it, or the maximum of the
                    curve smoothed over neighbouring ranks) -- chosen on the
                    TUNING seeds
``zoo``             on the REPORTING seeds, with the chosen rule: kernel
                    mechanism K-means and soft EM in the kernel basis,
                    against the oracle, the library method and the raw
                    profile, on all twelve problems
==================  =======================================================

Writes ``results/kernel/``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans

from experiments.problems.run import _err, _profile_kmeans
from experiments.problems.zoo import (
    PROBLEMS,
    gap_outside_library,
    make_windows,
    oracle_labels,
    sigma_for_rho,
)
from lrdsr import paths
from lrdsr.core.kernel import NystromBasis, loo_error
from lrdsr.core.mechanism_space import (
    inliers,
    mechanism_features,
    mechanism_init,
    mechanism_noise,
)
from lrdsr.core.soft import SoftLRDSR
from lrdsr.protocol import REPORT_SEEDS, TUNE_SEEDS

RESULTS = paths.results_dir("kernel", figs=False)
RHO_GRID = (0.1, 0.25, 1.0)
N_WINDOWS = 150
WINDOW_LEN = 48
RANKS = (4, 8, 12, 16, 24, 32, 48)
RULES = ("loo_min", "loo_1se", "snr", "snr_1se", "snr_smooth")
#: bootstrap resamples of the windows for the SNR's standard error
N_BOOT = 50


def gap_outside_basis(problem, basis, n: int = 4000, seed: int = 0) -> float:
    """Worst share of a pairwise gap the basis cannot represent (as for the library)."""
    X = problem.sample(np.random.default_rng(seed), (n,))
    B = basis(X)
    F = [f(X) for f in problem.laws]
    worst = 0.0
    for j in range(problem.K):
        for k in range(j + 1, problem.K):
            g = F[j] - F[k]
            coef, *_ = np.linalg.lstsq(B, g, rcond=None)
            worst = max(worst, float(np.mean((g - B @ coef) ** 2) / np.mean(g ** 2)))
    return worst


def _kmeans(S, K, seed):
    keep = inliers(S)
    return KMeans(K, n_init=30, random_state=seed).fit(S[keep]).predict(S)


def _snr(S, sigma):
    """Excess spread of mechanism space over the noise, per root dimension.

    ``tr Cov(S) - p sigma^2`` is the regime spread the basis captures
    (``separation_unlabelled``); the high-dimensional clustering threshold
    compares it with ``sqrt(p)``, the fluctuation of ``p`` noise dimensions.
    """
    S = S[inliers(S)]
    p = S.shape[1]
    excess = float(np.sum(S.var(axis=0, ddof=1))) - p * sigma ** 2
    return excess / (np.sqrt(p) * sigma ** 2)


def _snr_se(S, sigma, seed):
    """Bootstrap standard error of :func:`_snr`, resampling windows."""
    rng = np.random.default_rng(seed)
    vals = [_snr(S[rng.integers(0, len(S), len(S))], sigma) for _ in range(N_BOOT)]
    return float(np.std(vals, ddof=1))


def sweep_cell(problem, rho, seed):
    """Every rank: error, gap outside the span, and the three rules' scores."""
    sigma = sigma_for_rho(problem, rho)
    X, y, z = make_windows(problem, sigma, N_WINDOWS, WINDOW_LEN, seed)
    rows = []
    flat = X.reshape(-1, X.shape[-1])
    for r in RANKS:
        if r >= WINDOW_LEN - 2:
            continue
        b = NystromBasis(r).fit(flat)
        S = mechanism_features(X, y, basis=b)
        lab = _kmeans(S, problem.K, seed)
        per = [loo_error(X[w:w + 1], y[w:w + 1], b) for w in range(len(X))]
        sig = mechanism_noise(X, y, basis=b)
        rows.append({
            "problem": problem.name, "K": problem.K, "rho": rho, "seed": seed,
            "basis_rank": b.rank, "basis_size": r, "error": _err(z, lab),
            "gap_outside": gap_outside_basis(problem, b),
            "loo": float(np.mean(per)),
            "loo_se": float(np.std(per, ddof=1) / np.sqrt(len(per))),
            "snr": _snr(S, sig),
            "snr_se": _snr_se(S, sig, seed),
        })
    return rows


def choose(g: pd.DataFrame, rule: str) -> int:
    """The size a rule picks within one (problem, rho, seed) sweep."""
    g = g.sort_values("basis_size")
    if rule == "loo_min":
        return int(g.loc[g.loo.idxmin(), "basis_size"])
    if rule == "loo_1se":
        best = g.loc[g.loo.idxmin()]
        return int(g[g.loo <= best.loo + best.loo_se].iloc[0]["basis_size"])
    if rule == "snr":
        return int(g.loc[g.snr.idxmax(), "basis_size"])
    if rule == "snr_1se":
        # the smallest rank whose SNR is within one bootstrap SE of the best
        best = g.loc[g.snr.idxmax()]
        return int(g[g.snr >= best.snr - best.snr_se].iloc[0]["basis_size"])
    if rule == "snr_smooth":
        # the maximum of the SNR curve averaged over neighbouring ranks
        sm = g.snr.rolling(3, center=True, min_periods=1).mean()
        return int(g.loc[sm.idxmax(), "basis_size"])
    raise ValueError(rule)


def run_sweep(seeds) -> pd.DataFrame:
    cells = [(p, rho, s) for p in PROBLEMS for rho in RHO_GRID for s in seeds]
    out = Parallel(n_jobs=-1)(delayed(sweep_cell)(*c) for c in cells)
    return pd.DataFrame([r for rows in out for r in rows])


def rule_table(sweep: pd.DataFrame) -> pd.DataFrame:
    """Per (problem, rho, seed): the error each rule gets, and the best rank's
    (chosen against the truth -- an optimistic bound, reported as such)."""
    rows = []
    for (pname, rho, seed), g in sweep.groupby(["problem", "rho", "seed"]):
        row = {"problem": pname, "rho": rho, "seed": seed,
               "best_rank_error": float(g.error.min())}
        for rule in RULES:
            s = choose(g, rule)
            row[f"{rule}_size"] = s
            row[f"{rule}_error"] = float(g[g["basis_size"] == s].error.iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def zoo_cell(problem, rho, seed, size):
    sigma = sigma_for_rho(problem, rho)
    X, y, z = make_windows(problem, sigma, N_WINDOWS, WINDOW_LEN, seed)
    K, names = problem.K, problem.feature_names
    flat = X.reshape(-1, X.shape[-1])
    b = NystromBasis(size).fit(flat)
    labels = {"oracle": oracle_labels(problem, X, y),
              "mech_library": mechanism_init(X, y, K, seed=seed, feature_names=names)}
    labels["mech_kernel"] = _kmeans(mechanism_features(X, y, basis=b), K, seed)
    labels["soft_em_kernel"] = SoftLRDSR(K, basis=b, init=labels["mech_kernel"],
                                         random_state=seed).fit(X, y).labels
    labels["soft_em_library"] = SoftLRDSR(K, feature_names=names,
                                          random_state=seed).fit(X, y).labels
    if problem.d == 1:
        labels["profile_kmeans"] = _profile_kmeans(X, y, K, seed)
    return [{"problem": problem.name, "K": K, "d": problem.d, "rho": rho, "seed": seed,
             "method": m, "basis_size": size, "matched_error": _err(z, lab)}
            for m, lab in labels.items()]


def run_zoo(rep_rules: pd.DataFrame, rule: str) -> pd.DataFrame:
    print(f"\n--- the zoo with the kernel basis (rule: {rule}) ---", flush=True)
    cells = [(p, row["rho"], row["seed"], int(row[f"{rule}_size"]))
             for p in PROBLEMS
             for _, row in rep_rules[rep_rules.problem == p.name].iterrows()]
    out = Parallel(n_jobs=-1)(delayed(zoo_cell)(*c) for c in cells)
    zoo = pd.DataFrame([r for rows in out for r in rows])
    orc = zoo[zoo.method == "oracle"].set_index(["problem", "rho", "seed"]).matched_error
    zoo["gap_to_oracle"] = zoo.matched_error - zoo.set_index(
        ["problem", "rho", "seed"]).index.map(orc)
    zoo.to_csv(RESULTS / "kernel_zoo.csv", index=False)
    summ = zoo.groupby(["problem", "method"]).gap_to_oracle.mean().unstack()
    summ.to_csv(RESULTS / "kernel_zoo_summary.csv")
    print(summ.round(3).to_string())
    print(zoo.groupby("method").gap_to_oracle.mean().round(4).to_string())
    return zoo


def run(args=None) -> None:
    t0 = time.time()
    cat = pd.DataFrame([{
        "problem": p.name, "K": p.K, "d": p.d,
        "gap_outside_library": gap_outside_library(p),
        **{f"gap_outside_nystrom_{r}": gap_outside_basis(
            p, NystromBasis(r).fit(p.sample(np.random.default_rng(0), (4000,))))
           for r in RANKS}} for p in PROBLEMS])
    cat.to_csv(RESULTS / "kernel_gap_catalog.csv", index=False)
    print(cat.round(3).to_string(index=False))

    print("\n--- rank sweep, tuning seeds ---", flush=True)
    tune = run_sweep(TUNE_SEEDS)
    tune.to_csv(RESULTS / "kernel_sweep_tune.csv", index=False)
    rules = rule_table(tune)
    rules.to_csv(RESULTS / "kernel_rules_tune.csv", index=False)
    means = {r: rules[f"{r}_error"].mean() for r in RULES}
    rule = min(means, key=means.get)
    print({k: round(v, 4) for k, v in means.items()}, "->", rule,
          f"(best rank, optimistic: {rules.best_rank_error.mean():.4f})")

    print("\n--- rank sweep, reporting seeds ---", flush=True)
    rep = run_sweep(REPORT_SEEDS)
    rep.to_csv(RESULTS / "kernel_sweep.csv", index=False)
    rep_rules = rule_table(rep)
    rep_rules["rule_chosen_on_tune"] = rule
    rep_rules.to_csv(RESULTS / "kernel_rules.csv", index=False)

    run_zoo(rep_rules, rule)
    print(f"\n[kernel] {(time.time() - t0) / 60:.1f} min; CSVs -> {RESULTS}")


if __name__ == "__main__":
    run()
