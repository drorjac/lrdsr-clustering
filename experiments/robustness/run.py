"""Robustness: noise that is correlated in time.

    python -m experiments robustness

The model is ``y = f_z(x) + e`` and everything so far drew ``e`` independent
sample to sample: the loss sums over samples, BIC counts every sample, and
the ceiling ``Q(sqrt(n rho)/2)`` is derived for independent noise. On a real
time series the *law* carries most of the time structure (the daily curve is
``f``), but the residual is still correlated: a day that runs high at 8 am
runs high at 9. This block changes the noise and nothing else, and asks how
much the independence assumption costs.

The design. The twelve problems of the zoo (``experiments.problems.zoo``),
150 windows of 48 samples, each window's inputs **sorted**, so the sample
order is the order of ``x`` -- as the hours of a day are. The noise is a
stationary AR(1) along that order,

    e_t = phi e_{t-1} + sqrt(1 - phi^2) sigma u_t,     Var e_t = sigma^2,

so the per-sample separation ``rho`` is the zoo's at every ``phi``: only the
correlation moves. The same innovations ``u`` are used at every ``phi``
(common random numbers), so the sweep changes the correlation and nothing
else.

The arms, all given the same windows and the same ``K``:

==================  =======================================================
``oracle_ols``      the true laws, smallest residual sum of squares: the
                    rule every estimator here approximates. Its ceiling
                    under correlated noise is the reference for "does the
                    method lose more than its own rule does?"
``oracle_gls``      the true laws AND the true ``phi``: residuals whitened
                    (Prais-Winsten) before they are summed. The Bayes rule
                    under AR(1) noise -- what a time-series-aware method
                    could reach. Not an estimator; a bound.
``lrdsr``           the hard loop, as in the zoo (``alpha_geom = 0``)
``soft_em``         ``SoftLRDSR``, Gaussian noise
``mechanism_kmeans`` K-means in mechanism space, no law fitted
==================  =======================================================

Theory, exact for two laws and Gaussian noise, per window with gap
``g = f_1 - f_0`` at its inputs and ``Sigma = sigma^2 phi^|i-j|``:

    OLS rule:  err = Q( |g|^2 / (2 sqrt(g^T Sigma g)) )
    GLS rule:  err = Q( sqrt(g^T Sigma^-1 g) / 2 )

With ``phi = 0`` both are V1's ``Q(|g|/(2 sigma))``. A smooth gap and smooth
noise overlap, so ``g^T Sigma g`` grows and the OLS error rises (fewer
effective samples); a gap that varies faster than the noise is *easier*
under GLS, which strips the slow noise out.

And on the real days (``experiments.realdata``): the lag-1 autocorrelation of
each day's residual about its law, so the ``phi`` grid can be read against
data.

Declared before the run (2026-09-29):
  R1  at phi = 0, oracle_gls and oracle_ols give identical labels, and every
      arm is within seed noise of the zoo (problems_summary.csv).
  R2  the exact formulas above predict the two oracles' mean error within
      0.02 in every two-law cell.
  R3  lrdsr's gap to oracle_ols, averaged over problems, stays below 0.05 at
      every phi <= 0.8: the method loses what its rule loses, little more.
  R4  oracle_gls beats oracle_ols by more as phi grows, and by the most on
      the problems whose gap varies fastest (high_frequency, moving_step).
  R5  the real days' residual lag-1 autocorrelation is above 0.5 on both
      datasets: the independence assumption is violated on real data.

Added after the run (2026-09-29), and labelled so in the verdicts: R2's band
turned out narrower than two Monte-Carlo standard errors at 150 windows x 3
seeds, and the innovations are shared across ``phi`` by design, so one
unlucky draw repeats down a problem's rows. ``theory_check`` re-tests the
formulas on 4000 windows per two-law cell, where the standard error is
~0.007. R2 stays reported as declared.

Reporting seeds only; nothing is tuned. Writes ``results/robustness/``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm

from experiments.common.fitting import fit_lrdsr, window_features
from experiments.problems.zoo import PROBLEMS, sigma_for_rho
from lrdsr import paths
from lrdsr.core.evaluation import aligned_accuracy
from lrdsr.core.mechanism_space import mechanism_init
from lrdsr.core.soft import SoftLRDSR

RESULTS = paths.results_dir("robustness", figs=False)
REPORT_SEEDS = (11, 23, 42)
RHO_GRID = (0.1, 0.25, 1.0)
PHI_GRID = (0.0, 0.3, 0.6, 0.8, 0.9, 0.95)
N_WINDOWS = 150
WINDOW_LEN = 48
METHODS = ("oracle_ols", "oracle_gls", "lrdsr", "soft_em", "mechanism_kmeans")


# ==========================================================================
# the noise, the two oracles, the theory
# ==========================================================================
def ar1(u: np.ndarray, phi: float) -> np.ndarray:
    """Stationary unit-variance AR(1) along the last axis, from innovations ``u``."""
    e = np.empty_like(u)
    e[..., 0] = u[..., 0]
    s = np.sqrt(1.0 - phi ** 2)
    for t in range(1, u.shape[-1]):
        e[..., t] = phi * e[..., t - 1] + s * u[..., t]
    return e


def whiten(r: np.ndarray, phi: float) -> np.ndarray:
    """Prais-Winsten: map AR(1) residuals to independent ones (up to scale)."""
    w = np.empty_like(r)
    w[..., 0] = np.sqrt(1.0 - phi ** 2) * r[..., 0]
    w[..., 1:] = r[..., 1:] - phi * r[..., :-1]
    return w


def ar1_cov(n: int, phi: float, sigma: float = 1.0) -> np.ndarray:
    i = np.arange(n)
    return sigma ** 2 * phi ** np.abs(i[:, None] - i[None, :])


def make_windows(problem, sigma, phi, n_windows=N_WINDOWS, window_len=WINDOW_LEN,
                 seed=11):
    """Zoo windows with sorted inputs and AR(1) noise. Returns ``(X, y, z)``.

    Inputs, labels and innovations depend on ``seed`` only, so the same seed
    gives the same windows at every ``phi`` and ``sigma``.
    """
    rng = np.random.default_rng([seed, problem.key, 7])
    X = problem.sample(rng, (n_windows, window_len))
    order = np.argsort(X[..., 0], axis=1)
    X = np.take_along_axis(X, order[..., None], axis=1)
    z = rng.integers(0, problem.K, size=n_windows)
    u = rng.normal(size=(n_windows, window_len))
    clean = np.empty((n_windows, window_len))
    for k, f in enumerate(problem.laws):
        m = z == k
        if m.any():
            clean[m] = f(X[m])
    return X, clean + sigma * ar1(u, phi), z


def oracle(problem, X, y, phi: float | None) -> np.ndarray:
    """True laws; ``phi=None`` the OLS rule, else residuals whitened first."""
    cost = []
    for f in problem.laws:
        r = y - f(X)
        if phi is not None:
            r = whiten(r, phi)
        cost.append(np.sum(r ** 2, axis=1))
    return np.argmin(np.column_stack(cost), axis=1)


def predicted_errors(problem, X, sigma, phi) -> tuple[float, float]:
    """Mean exact error of the OLS and GLS oracles over the windows (K = 2)."""
    S = ar1_cov(X.shape[1], phi, sigma)
    Si = np.linalg.inv(S)
    G = problem.laws[1](X) - problem.laws[0](X)
    gg = np.einsum("wi,wi->w", G, G)
    gSg = np.einsum("wi,ij,wj->w", G, S, G)
    gSig = np.einsum("wi,ij,wj->w", G, Si, G)
    # a window on which the two laws coincide (the kink, left of its hinge)
    # carries no evidence: either rule is a coin flip there
    ols = np.full(gg.shape, 0.5)
    ok = gSg > 0
    ols[ok] = norm.sf(gg[ok] / (2.0 * np.sqrt(gSg[ok])))
    gls = norm.sf(np.sqrt(gSig) / 2.0)
    return float(ols.mean()), float(gls.mean())


# ==========================================================================
# the sweep
# ==========================================================================
def run_cell(problem, seed):
    rows = []
    for rho in RHO_GRID:
        sigma = sigma_for_rho(problem, rho)
        for phi in PHI_GRID:
            X, y, z = make_windows(problem, sigma, phi, seed=seed)
            Zf, fnames = window_features(X, y)
            names = problem.feature_names
            labels = {
                "oracle_ols": oracle(problem, X, y, None),
                "oracle_gls": oracle(problem, X, y, phi),
                "lrdsr": fit_lrdsr(Zf, fnames, X, y, names, problem.K, seed,
                                   alpha_geom=0.0).labels,
                "soft_em": SoftLRDSR(problem.K, feature_names=names,
                                     random_state=seed).fit(X, y).labels,
                "mechanism_kmeans": mechanism_init(X, y, problem.K, seed=seed,
                                                   feature_names=names),
            }
            pred = predicted_errors(problem, X, sigma, phi) if problem.K == 2 else (np.nan, np.nan)
            for method, lab in labels.items():
                rows.append({
                    "problem": problem.name, "K": problem.K, "rho": rho, "phi": phi,
                    "seed": seed, "method": method,
                    "matched_error": 1.0 - aligned_accuracy(z, lab),
                    "pred_ols": pred[0], "pred_gls": pred[1],
                    "same_as_ols": float(np.mean(lab == labels["oracle_ols"])),
                })
    return rows


def theory_check(n_windows: int = 4000, seed: int = REPORT_SEEDS[0]) -> pd.DataFrame:
    """The two oracles against their exact formulas, on many windows (K = 2)."""
    rows = []
    for problem in (p for p in PROBLEMS if p.K == 2):
        for rho in RHO_GRID:
            sigma = sigma_for_rho(problem, rho)
            for phi in PHI_GRID:
                X, y, z = make_windows(problem, sigma, phi, n_windows=n_windows, seed=seed)
                pred_ols, pred_gls = predicted_errors(problem, X, sigma, phi)
                obs_ols = float(np.mean(oracle(problem, X, y, None) != z))
                obs_gls = float(np.mean(oracle(problem, X, y, phi) != z))
                rows.append({"problem": problem.name, "rho": rho, "phi": phi,
                             "windows": n_windows, "obs_ols": obs_ols, "pred_ols": pred_ols,
                             "obs_gls": obs_gls, "pred_gls": pred_gls,
                             "se_ols": np.sqrt(pred_ols * (1 - pred_ols) / n_windows),
                             "se_gls": np.sqrt(pred_gls * (1 - pred_gls) / n_windows)})
    return pd.DataFrame(rows)


def summarise(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["problem", "K", "rho", "phi"]
    wide = raw.pivot_table(index=[*keys, "seed"], columns="method",
                           values="matched_error").reset_index()
    pred = raw.groupby([*keys, "seed"])[["pred_ols", "pred_gls"]].first().reset_index()
    wide = wide.merge(pred, on=[*keys, "seed"])
    for m in ("lrdsr", "soft_em", "mechanism_kmeans"):
        wide[f"{m}_gap"] = wide[m] - wide["oracle_ols"]
    wide["gls_gain"] = wide["oracle_ols"] - wide["oracle_gls"]
    return wide.groupby(keys).mean(numeric_only=True).drop(columns="seed").reset_index()


# ==========================================================================
# how correlated are real residuals?
# ==========================================================================
def real_residual_acf(seed: int = REPORT_SEEDS[0], max_lag: int = 6) -> pd.DataFrame:
    """Autocorrelation of each real day's residual about its assigned law.

    The laws are the two-regime soft EM over the Fourier basis, the best
    arm of ``experiments.realdata``; each day's level is removed by the
    window construction already, and the residual is re-centred per day.
    """
    from experiments.realdata.run import fourier
    from experiments.realdata.windows import load

    rows = []
    for name in ("bike", "traffic"):
        d = load(name)
        res = SoftLRDSR(2, basis=fourier, noise="student_t", random_state=seed).fit(
            d.X_seq, d.y_seq)
        Phi = fourier(d.X_seq[0])
        R = d.y_seq - np.stack([Phi @ res.coef[k] for k in res.labels])
        R = R - R.mean(axis=1, keepdims=True)
        denom = np.sum(R ** 2, axis=1)
        for lag in range(1, max_lag + 1):
            acf = np.sum(R[:, lag:] * R[:, :-lag], axis=1) / denom
            rows.append({"dataset": name, "lag": lag, "acf_median": float(np.median(acf)),
                         "acf_q25": float(np.quantile(acf, 0.25)),
                         "acf_q75": float(np.quantile(acf, 0.75)), "days": len(acf)})
    return pd.DataFrame(rows)


def verdicts(summ: pd.DataFrame, acf: pd.DataFrame, th: pd.DataFrame) -> pd.DataFrame:
    s0 = summ[summ.phi == 0.0]
    two = summ[summ.K == 2]
    by_phi = summ.groupby("phi")[["lrdsr_gap", "gls_gain"]].mean()
    hf = summ[summ.problem.isin(["high_frequency", "moving_step"])].groupby("phi").gls_gain.mean()
    rest = summ[~summ.problem.isin(["high_frequency", "moving_step"])].groupby("phi").gls_gain.mean()
    lag1 = acf[acf.lag == 1].set_index("dataset").acf_median
    out = [
        ("R1", "phi = 0: GLS oracle = OLS oracle",
         float((s0.oracle_gls - s0.oracle_ols).abs().max()), bool((s0.oracle_gls == s0.oracle_ols).all())),
        ("R2", "theory within 0.02 of both oracles, two-law cells",
         float(max((two.oracle_ols - two.pred_ols).abs().max(), (two.oracle_gls - two.pred_gls).abs().max())),
         bool(((two.oracle_ols - two.pred_ols).abs() <= 0.02).all()
              and ((two.oracle_gls - two.pred_gls).abs() <= 0.02).all())),
        ("R3", "lrdsr gap to OLS oracle < 0.05 at every phi <= 0.8",
         float(by_phi.loc[by_phi.index <= 0.8, "lrdsr_gap"].max()),
         bool((by_phi.loc[by_phi.index <= 0.8, "lrdsr_gap"] < 0.05).all())),
        ("R4", "GLS gain grows with phi, largest on fast-varying gaps",
         float(hf.iloc[-1] - rest.iloc[-1]),
         bool(by_phi.gls_gain.is_monotonic_increasing and (hf.iloc[-1] > rest.iloc[-1]))),
        ("R5", "real residual lag-1 autocorrelation > 0.5 on both datasets",
         float(lag1.min()), bool((lag1 > 0.5).all())),
    ]
    z = np.concatenate([(th.obs_ols - th.pred_ols) / th.se_ols, (th.obs_gls - th.pred_gls) / th.se_gls])
    out.insert(2, ("R2b", "(added after the run) theory within 3 SE, 4000 windows",
                   float(np.mean(np.abs(z) <= 3.0)), bool(np.mean(np.abs(z) <= 3.0) >= 0.95)))
    return pd.DataFrame(out, columns=["prediction", "claim", "value", "held"])


def run(args=None, n_jobs: int = -1) -> pd.DataFrame:
    t0 = time.time()
    out = Parallel(n_jobs=n_jobs)(delayed(run_cell)(p, s) for p in PROBLEMS for s in REPORT_SEEDS)
    raw = pd.DataFrame([r for rows in out for r in rows])
    raw.to_csv(RESULTS / "ar_noise_raw.csv", index=False)
    summ = summarise(raw)
    summ.to_csv(RESULTS / "ar_noise_summary.csv", index=False)
    acf = real_residual_acf()
    acf.to_csv(RESULTS / "real_residual_acf.csv", index=False)
    th = theory_check()
    th.to_csv(RESULTS / "ar_noise_theory.csv", index=False)
    v = verdicts(summ, acf, th)
    v.to_csv(RESULTS / "ar_noise_verdict.csv", index=False)

    with pd.option_context("display.width", 200, "display.precision", 3):
        print(summ.groupby("phi")[["oracle_gls", "oracle_ols", "lrdsr", "soft_em",
                                   "mechanism_kmeans", "lrdsr_gap", "gls_gain"]].mean())
        print(acf[acf.lag <= 3].to_string(index=False))
        print(v.to_string(index=False))
    print(f"\n[robustness] {len(raw)} rows in {(time.time() - t0) / 60:.1f} min -> {RESULTS}")
    return summ


if __name__ == "__main__":
    run()
