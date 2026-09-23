"""The separation theory: what any method can and cannot do.

The ceiling this project measures everything against, in five blocks. None
of them involve an estimator -- they are statements about the problem.

``V1``  the window-error identity is **exact**, not asymptotic:
        ``P_err = Q(sqrt(D_n) / 2 sigma)``.
``V2``  going from the realised ``D_n`` to the population ``n rho`` is a
        Jensen step, priced by ``kappa = Var(g^2)/E[g^2]^2``. The correction
        is first order and says so.
``V3``  the ``K``-ary sandwich: with more than two laws the error is
        bracketed by the pairwise terms.
``V4``  a component present in **both** laws carries no information about
        which one produced a window, so the oracle is invariant to it
        however large it grows.
``V7``  mechanism space attains the bound, and the separation is measurable
        on unlabelled windows.

Carried over from the parent project with its application-specific block
(``V5``, which instantiated ``rho(L)`` for a particular physics) removed.
What is here is the part that holds for any two laws.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import roots_hermitenorm
from scipy.stats import norm

from .. import paths

#: CSVs are findings and are committed; the diagnostic figures drawn here are
#: intermediates -- the paper draws its own from the CSVs.
RESULTS = paths.RESULTS / "theory"
FIGURES = paths.CACHE_ROOT / "figures" / "theory"

#: Committed Monte-Carlo seeds. Tuning//smoke seeds are 3/7/19 and never appear
#: in a committed number.
REPORT_SEEDS = (11, 23, 42)
#: V2.4 gives the anomalous point its own 20 fresh seeds.
ANOMALY_SEEDS = tuple(range(101, 121))

Q = norm.sf                     # Q(t) = P(N(0,1) > t)


# ---------------------------------------------------------------------------
# designs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Design:
    """A gap shape on ``x ~ U[-1, 1]``, with its dispersion in closed form.

    ``kappa`` is computed analytically, never fitted — the whole point of V2 is
    that the correction's slope is *predicted* before the points are drawn.
    """
    name: str
    shape: Callable[[np.ndarray], np.ndarray]
    kappa: float
    note: str

    def gap(self, x: np.ndarray, rho: float, sigma: float = 1.0) -> np.ndarray:
        """The shape rescaled so that ``E[g^2] = rho * sigma^2`` exactly."""
        raw = self.shape(x)
        return raw * np.sqrt(rho * sigma**2 / self._mean_square())

    def _mean_square(self) -> float:
        return {"linear": 1.0 / 3.0, "sign": 1.0, "quadratic": 1.0 / 5.0}[self.name]


#: kappa = Var(g^2)/E[g^2]^2 on U[-1,1]:
#:   linear     g=x   : E[x^2]=1/3, E[x^4]=1/5  -> (1/5-1/9)/(1/9)     = 4/5
#:   sign       g=±1  : g^2 == 1, zero variance -> 0  (V2's bound is EQUALITY)
#:   quadratic  g=x^2 : E[x^4]=1/5, E[x^8]=1/9  -> (1/9-1/25)/(1/25)   = 16/9
DESIGNS = (
    Design("linear", lambda x: x, 4.0 / 5.0,
           "g(x) = x — the reference design"),
    Design("sign", lambda x: np.sign(x), 0.0,
           "g(x) = sign(x) — |g| constant, so Jensen's inequality is TIGHT"),
    Design("quadratic", lambda x: x**2, 16.0 / 9.0,
           "g(x) = x^2 — lopsided, large kappa"),
)
DESIGN_BY_NAME = {d.name: d for d in DESIGNS}


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _mkdirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def _save_fig(fig, stem: str) -> None:
    """Diagnostic figures, PNG and PDF, into the intermediate tier."""
    _mkdirs()
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{stem}.{ext}", dpi=180, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------------------------------------------------------------------------
# core quantities
# ---------------------------------------------------------------------------
def exact_error(design: Design, n: int, rho: float, seed: int,
                n_designs: int = 200_000, sigma: float = 1.0) -> float:
    """E_x[ Q(sqrt(D_n)/(2 sigma)) ] — the exact random-design Bayes error.

    Only designs are drawn; the noise is integrated out analytically, which is
    what makes this an *oracle* quantity rather than a simulation.
    """
    rng = _rng(seed)
    total, done = 0.0, 0
    chunk = max(1, min(n_designs, 4_000_000 // max(n, 1)))
    while done < n_designs:
        m = min(chunk, n_designs - done)
        x = rng.uniform(-1.0, 1.0, size=(m, n))
        d_n = np.sum(design.gap(x, rho, sigma) ** 2, axis=1)
        total += float(np.sum(Q(np.sqrt(d_n) / (2.0 * sigma))))
        done += m
    return total / n_designs


def simulated_error(design: Design, n: int, rho: float, seed: int,
                    n_windows: int = 200_000, sigma: float = 1.0) -> float:
    """Draw designs AND noise, run the min-residual rule, count mistakes.

    This is the honest end-to-end simulation V1 checks the exact formula
    against: an error occurs when ``sum_t eps_t g(x_t) < -D_n / 2``.
    """
    rng = _rng(seed)
    errors, done = 0, 0
    chunk = max(1, min(n_windows, 4_000_000 // max(n, 1)))
    while done < n_windows:
        m = min(chunk, n_windows - done)
        x = rng.uniform(-1.0, 1.0, size=(m, n))
        g = design.gap(x, rho, sigma)
        eps = rng.normal(0.0, sigma, size=(m, n))
        d_n = np.sum(g * g, axis=1)
        errors += int(np.sum(np.sum(eps * g, axis=1) < -0.5 * d_n))
        done += m
    return errors / n_windows


def population_error(n: int, rho: float) -> float:
    """The simple formula Q(0.5 sqrt(n rho)) — always optimistic."""
    return float(Q(0.5 * np.sqrt(n * rho)))


def _band(p: float, m: int) -> float:
    """1 s.d. binomial simulation noise; the spec's 'matches' is 3x this."""
    return float(np.sqrt(max(p, 1e-12) * (1.0 - min(p, 1 - 1e-12)) / m))


# ---------------------------------------------------------------------------
# V1 — does the exact formula hold?
# ---------------------------------------------------------------------------
def run_v1(args=None, n_windows: int = 200_000,
           n_grid=(8, 16, 32, 64, 128, 256)) -> pd.DataFrame:
    """Sanity anchor: simulated error must sit on the exact formula's curve."""
    rows = []
    for design in DESIGNS:
        for n in n_grid:
            for rho in (0.5,):
                m = n_windows * len(REPORT_SEEDS)
                sim = float(np.mean([
                    simulated_error(design, n, rho, seed, n_windows)
                    for seed in REPORT_SEEDS]))
                # Average the exact formula over the same seeds: it is itself a
                # Monte-Carlo estimate (over designs), so charging only the
                # simulation's noise to the comparison understates the band.
                exact = float(np.mean([exact_error(design, n, rho, seed, n_windows)
                                       for seed in REPORT_SEEDS]))
                # Band from the MODEL probability, not the observed one. At
                # n = 256 the true error is ~1e-7, so 6e5 windows expect 0.06
                # errors and observe 0; a band built from p_hat = 0 collapses to
                # nothing and the point can never pass, however right the theory
                # is. Points that expect fewer than 10 errors are BELOW
                # SIMULATION RESOLUTION and are reported as such, not as
                # failures -- they carry no information either way.
                band = np.sqrt(2.0) * _band(max(exact, 1.0 / m), m)
                expected_errors = exact * m
                resolvable = expected_errors >= 10.0
                rows.append({
                    "design": design.name, "kappa": design.kappa, "n": n, "rho": rho,
                    "simulated": sim, "exact_formula": exact,
                    "population_formula": population_error(n, rho),
                    "abs_gap": abs(sim - exact), "noise_band_1sd": band,
                    "expected_error_count": expected_errors,
                    "resolvable": resolvable,
                    "within_3_sd": bool(abs(sim - exact) <= 3.0 * band) if resolvable else None,
                })
    df = pd.DataFrame(rows)
    _mkdirs()
    df.to_csv(RESULTS / "v1_exactness.csv", index=False)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
    for ax, design in zip(axes, DESIGNS, strict=False):
        sub = df[df.design == design.name]
        ax.plot(sub.n, sub.exact_formula, lw=2, color="#0e7490",
                label="exact  $E_x[Q(\\sqrt{D_n}/2\\sigma)]$", zorder=2)
        ax.errorbar(sub.n, sub.simulated, yerr=3 * sub.noise_band_1sd, fmt="o",
                    ms=5, color="#1b1f24", capsize=3, lw=1,
                    label="simulated (min-residual)", zorder=3)
        ax.plot(sub.n, sub.population_formula, ls="--", lw=1.4, color="#b45309",
                label="simple  $Q(\\frac{1}{2}\\sqrt{n\\rho})$", zorder=1)
        ax.set(xscale="log", yscale="log", xlabel="window length $n$",
               title=f"{design.name}   $\\kappa$ = {design.kappa:.3f}")
        ax.grid(alpha=.25, lw=.6)
    axes[0].set_ylabel("P(error)")
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle("V1 — the exact random-design formula is the truth; "
                 "the simple formula is optimistic", y=1.02, fontsize=11)
    _save_fig(fig, "v1_exactness")
    print(df.to_string(index=False))
    res = df[df.resolvable]
    print(f"\nV1: {int(res.within_3_sd.sum())}/{len(res)} resolvable points within 3 s.d. "
          f"({len(df) - len(res)} points below simulation resolution — "
          f"fewer than 10 expected errors in {n_windows * len(REPORT_SEEDS):,} windows)")
    return df


# ---------------------------------------------------------------------------
# V2 — the simple formula: optimistic, and by exactly how much
# ---------------------------------------------------------------------------
def _n_star_bisect(design: Design, rho: float, target: float, seed: int,
                   n_designs: int, lo: int = 2, hi: int | None = None) -> float:
    """The n whose EXACT error equals ``target`` — CONTINUOUS, not the integer.

    Integer bisection alone is not good enough here. At rho = 2 the reference
    n0* is only 5.4, so rounding up to the next whole window would report a ~10%
    excess even for the ``sign`` design, whose kappa is exactly 0 and whose true
    excess is exactly 0. The rounding, not the design, would be driving V2's
    headline plot. So we bracket on integers and then interpolate in
    log(error), which is close to linear in n over one step.
    """
    # Bracket from n0* by doubling, NOT from a huge fixed ceiling. n* always
    # sits just above n0* = 4z^2/rho, so a fixed hi = 40000 would spend every
    # call evaluating exact_error on a 40000-sample window -- 2.4e9 elements
    # per call, and the block never finishes.
    if hi is None:
        n0 = 4.0 * float(norm.isf(target)) ** 2 / rho
        hi = max(4, int(np.ceil(n0)))
        for _ in range(12):
            if exact_error(design, hi, rho, seed, n_designs) <= target:
                break
            lo, hi = hi, hi * 2
        else:
            return float("nan")
    elif exact_error(design, hi, rho, seed, n_designs) > target:
        return float("nan")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if exact_error(design, mid, rho, seed, n_designs) <= target:
            hi = mid
        else:
            lo = mid
    e_lo = exact_error(design, lo, rho, seed, n_designs)
    e_hi = exact_error(design, hi, rho, seed, n_designs)
    if not (e_lo > target >= e_hi) or e_lo <= 0 or e_hi <= 0:
        return float(hi)
    w = (np.log(e_lo) - np.log(target)) / (np.log(e_lo) - np.log(e_hi))
    return float(lo + w * (hi - lo))


def run_v2(args=None, n_designs: int = 60_000, target: float = 0.05,
           rho_grid=(0.1, 0.25, 0.5, 1.0, 2.0)) -> pd.DataFrame:
    """Jensen's inequality, and the kappa-linear correction to ``n*``.

    The predicted line is drawn from ``kappa`` computed in closed form from the
    design — nothing is fitted, and the line exists before the points do.
    """
    z = float(norm.isf(target))
    rows = []
    for design in DESIGNS:
        for rho in rho_grid:
            n0 = 4.0 * z**2 / rho
            n_star = float(np.mean([
                _n_star_bisect(design, rho, target, seed, n_designs)
                for seed in REPORT_SEEDS]))
            # Jensen check at the population-optimal n, where it bites hardest.
            n_chk = max(2, round(n0))
            ex = float(np.mean([exact_error(design, n_chk, rho, s, n_designs)
                                for s in REPORT_SEEDS]))
            pop = population_error(n_chk, rho)
            rows.append({
                "design": design.name, "kappa": design.kappa, "rho": rho,
                "n0_star": n0, "n_star_measured": n_star,
                "excess_fraction": (n_star - n0) / n0,
                "predicted_excess": design.kappa * (1 + 1 / z**2) * rho / 16.0,
                "exact_at_n0": ex, "population_at_n0": pop,
                "jensen_holds": ex >= pop - 3 * _band(pop, n_designs * len(REPORT_SEEDS)),
                "jensen_slack": ex - pop,
            })
    df = pd.DataFrame(rows)
    _mkdirs()
    df.to_csv(RESULTS / "v2_kappa_correction.csv", index=False)

    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    colors = {"linear": "#0e7490", "sign": "#15803d", "quadratic": "#b45309"}
    for design in DESIGNS:
        sub = df[df.design == design.name]
        c = colors[design.name]
        a1.plot(sub.rho, sub.jensen_slack, "o-", color=c, ms=5, lw=1.4,
                label=f"{design.name} ($\\kappa$={design.kappa:.2f})")
        rr = np.linspace(0, max(rho_grid) * 1.05, 50)
        a2.plot(rr, design.kappa * (1 + 1 / z**2) * rr / 16.0, "--", color=c, lw=1.4,
                label=f"predicted, {design.name}")
        a2.plot(sub.rho, sub.excess_fraction, "o", color=c, ms=6)
    a1.axhline(0, color="#1b1f24", lw=1)
    a1.set(xlabel=r"$\rho$", ylabel="exact $-$ simple, at $n_0^*$",
           title="V2.1 — the simple formula is never pessimistic\n"
                 "(sign design: exactly tight, as Jensen predicts)")
    a1.legend(frameon=False, fontsize=8.5); a1.grid(alpha=.25, lw=.6)
    a2.set(xlabel=r"$\rho$", ylabel=r"$(n^* - n_0^*)\,/\,n_0^*$",
           title="V2.2 — extra samples needed is linear in $\\rho$,\n"
                 "with the slope $\\kappa$ predicts (lines drawn first)")
    a2.legend(frameon=False, fontsize=8.5); a2.grid(alpha=.25, lw=.6)
    _save_fig(fig, "v2_kappa_linearity")
    print(df.to_string(index=False))
    return df


def run_v2_anomaly(args=None, n_designs: int = 60_000,
                   target: float = 0.05, rho: float = 0.25) -> pd.DataFrame:
    """V2.4 — the rho = 0.25 point, rerun alone on 20 fresh seeds.

    Pre-registered: if the cloud covers the predicted line it was noise; if it
    does not, the correction's validity range is stated alongside the result rather
    than the point being dropped.
    """
    z = float(norm.isf(target))
    rows = []
    for design in DESIGNS:
        n0 = 4.0 * z**2 / rho
        for seed in ANOMALY_SEEDS:
            n_star = _n_star_bisect(design, rho, target, seed, n_designs)
            rows.append({"design": design.name, "kappa": design.kappa, "rho": rho,
                         "seed": seed, "n0_star": n0,
                         "n_star_measured": n_star,
                         "excess_fraction": (n_star - n0) / n0})
    df = pd.DataFrame(rows)
    pred = {d.name: d.kappa * (1 + 1 / z**2) * rho / 16.0 for d in DESIGNS}
    df["predicted_excess"] = df.design.map(pred)
    _mkdirs()
    df.to_csv(RESULTS / "v2_anomaly_rho025.csv", index=False)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for i, design in enumerate(DESIGNS):
        sub = df[df.design == design.name]
        ax.scatter(np.full(len(sub), i) + np.random.default_rng(0).uniform(-.09, .09, len(sub)),
                   sub.excess_fraction, s=22, alpha=.65, color="#0e7490",
                   zorder=3, label="20 seeds" if i == 0 else None)
        ax.hlines(pred[design.name], i - .28, i + .28, color="#b45309", lw=2.4,
                  zorder=4, label="predicted" if i == 0 else None)
        m, s = sub.excess_fraction.mean(), sub.excess_fraction.std()
        ax.errorbar(i, m, yerr=1.96 * s / np.sqrt(len(sub)), fmt="s", ms=7,
                    color="#1b1f24", capsize=4, zorder=5,
                    label="mean ± 95% CI" if i == 0 else None)
    ax.set_xticks(range(len(DESIGNS)))
    ax.set_xticklabels([d.name for d in DESIGNS])
    ax.set(ylabel=r"$(n^*-n_0^*)/n_0^*$",
           title=r"V2.4 — the anomalous $\rho$ = 0.25 point, rerun on 20 fresh seeds")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=.25, lw=.6, axis="y")
    _save_fig(fig, "v2_anomaly")

    summary = (df.groupby("design")
                 .agg(mean_excess=("excess_fraction", "mean"),
                      sd=("excess_fraction", "std"),
                      predicted=("predicted_excess", "first"))
                 .reset_index())
    summary["ci95"] = 1.96 * summary.sd / np.sqrt(len(ANOMALY_SEEDS))
    summary["covers_prediction"] = (
        (summary.predicted >= summary.mean_excess - summary.ci95)
        & (summary.predicted <= summary.mean_excess + summary.ci95))
    summary.to_csv(RESULTS / "v2_anomaly_verdict.csv", index=False)
    print(summary.to_string(index=False))
    return summary


# ---------------------------------------------------------------------------
# V3 — K > 2
# ---------------------------------------------------------------------------
def kary_correct_gh(t: float, K: int, nodes: int = 120) -> float:
    """P(correct) for K equidistant mechanisms, by Gauss-Hermite quadrature.

    ``E_W[ Phi(t*sqrt(2) + W)^{K-1} ]`` with ``W ~ N(0,1)``. For ``K = 2`` this
    collapses to ``Phi(t)``, i.e. the binary formula, which is the first thing
    the test asserts.
    """
    x, w = roots_hermitenorm(nodes)
    w = w / w.sum()
    return float(np.sum(w * norm.cdf(t * np.sqrt(2.0) + x) ** (K - 1)))


def _kary_correct_mc(t: float, K: int, seed: int, draws: int = 400_000) -> float:
    """Direct MC over the correlated Gaussian decision statistics."""
    rng = _rng(seed)
    # Z_k = <eps, g_jk>/(sigma sqrt(D)) is standard normal, and for an
    # equidistant (simplex) design <g_jk, g_jl> = D/2, so the Z_k are
    # equicorrelated at 1/2 -- generated exactly by one shared plus K-1
    # independent standard normals. The decision is CORRECT when every
    # Z_k > -t; getting that inequality backwards silently produces
    # Phi(t*sqrt2) instead of Phi(t) at K = 2, which is how this was caught.
    shared = rng.standard_normal(draws)[:, None]
    indep = rng.standard_normal((draws, K - 1))
    z = (shared + indep) / np.sqrt(2.0)
    return float(np.mean(np.all(z > -t, axis=1)))


def run_v3(args=None, n_designs: int = 60_000) -> pd.DataFrame:
    """Quadrature exactness, the sandwich, and a non-equidistant control."""
    rows = []
    for K in (3, 5):
        for t in (0.5, 1.0, 1.5, 2.0, 2.5):
            gh = kary_correct_gh(t, K)
            mc = float(np.mean([_kary_correct_mc(t, K, s) for s in REPORT_SEEDS]))
            # Two different checks, because they can answer different questions.
            # Against MC we can only claim agreement to MC noise (~7e-4 at
            # 4e5 draws) -- no amount of care makes a Monte-Carlo comparison a
            # 1e-6 statement. The quadrature's OWN exactness is established by
            # node convergence: 60 vs 240 nodes agreeing far below 1e-6.
            gh_coarse = kary_correct_gh(t, K, nodes=60)
            rows.append({"block": "quadrature", "K": K, "t": t,
                         "gauss_hermite": gh, "monte_carlo": mc,
                         "abs_diff": abs(gh - mc),
                         "mc_noise_3sd": 3 * _band(mc, 400_000 * 3),
                         "matches_mc": abs(gh - mc) < 3 * _band(mc, 400_000 * 3),
                         "gh_60_vs_240_nodes": abs(gh_coarse - gh),
                         "quadrature_exact_to_1e6": abs(gh_coarse - gh) < 1e-6})
    quad = pd.DataFrame(rows)

    # --- sandwich on a random design -------------------------------------
    srows = []
    design = DESIGN_BY_NAME["linear"]
    for K in (3, 5):
        for rho_min in (0.25, 0.5, 1.0):
            for n in (16, 32, 64, 128):
                rng = _rng(REPORT_SEEDS[0])
                x = rng.uniform(-1.0, 1.0, size=(n_designs, n))
                d_n = np.sum(design.gap(x, rho_min) ** 2, axis=1)
                t = np.sqrt(d_n) / 2.0
                pe = float(np.mean([1.0 - kary_correct_gh(float(ti), K)
                                    for ti in t[:4000]]))
                # The sandwich bounds the K-ary error by the BINARY error of
                # the same design, so Q must be the exact random-design value,
                # not the population approximation -- mixing the two makes the
                # upper bound fail wherever the design penalty is large.
                q = float(np.mean(Q(t[:4000])))
                q_pop = population_error(n, rho_min)
                lo, hi = (2.0 / K) * q, (K - 1) * q
                srows.append({"block": "sandwich", "K": K, "rho_min": rho_min, "n": n,
                              "P_e": pe, "lower_2_over_K_Q": lo,
                              "upper_K_minus_1_Q": hi,
                              "binary_Q_exact": q, "binary_Q_population": q_pop,
                              "inside": (pe >= lo) and (pe <= hi),
                              "equals_closest_pair_Q": abs(pe - q) < 1e-3})
    sand = pd.DataFrame(srows)

    _mkdirs()
    quad.to_csv(RESULTS / "v3_quadrature.csv", index=False)
    sand.to_csv(RESULTS / "v3_sandwich.csv", index=False)

    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for K, c in ((3, "#0e7490"), (5, "#b45309")):
        s = quad[quad.K == K]
        a1.semilogy(s.t, 1 - s.gauss_hermite, "-", color=c, lw=2, label=f"K={K} Gauss-Hermite")
        a1.semilogy(s.t, 1 - s.monte_carlo, "o", color=c, ms=5, mfc="none", label=f"K={K} direct MC")
    a1.set(xlabel="$t=\\sqrt{D_n}/2\\sigma$", ylabel="$P_e$",
           title="V3.1 — the quadrature formula is exact")
    a1.legend(frameon=False, fontsize=8.5); a1.grid(alpha=.25, lw=.6)

    s = sand[(sand.K == 5) & (sand.rho_min == 0.5)]
    a2.fill_between(s.n, s.lower_2_over_K_Q, s.upper_K_minus_1_Q, alpha=.20,
                    color="#0e7490", label="sandwich $[\\frac{2}{K}Q,\\;(K-1)Q]$")
    a2.semilogy(s.n, s.P_e, "o-", color="#1b1f24", ms=5, lw=1.5, label="measured $P_e$")
    a2.semilogy(s.n, [population_error(int(n), 0.5) for n in s.n], "--",
                color="#b45309", lw=1.4, label="closest-pair $Q$ alone")
    a2.set(xscale="log", xlabel="window length $n$", ylabel="$P_e$",
           title="V3.2 — $P_e$ sits inside the sandwich,\nand is NOT the closest-pair error")
    a2.legend(frameon=False, fontsize=8.5); a2.grid(alpha=.25, lw=.6)
    _save_fig(fig, "v3_kary")

    print(quad.to_string(index=False))
    print(f"\nsandwich holds at {int(sand.inside.sum())}/{len(sand)} points")
    return sand


# ---------------------------------------------------------------------------
# V4 — common-component cancellation
# ---------------------------------------------------------------------------
def run_v4(args=None, n_windows: int = 200_000, multiple: float = 50.0) -> pd.DataFrame:
    """Adding a shared component to every mechanism must change nothing.

    This is the invariance the separation argument rests on: a term
    is common to H0 and H1, so it cancels out of the gap and cannot help or
    hurt identification.
    """
    design = DESIGN_BY_NAME["linear"]
    rows = []
    for n in (16, 64, 256):
        for rho in (0.25, 1.0):
            plain = float(np.mean([simulated_error(design, n, rho, s, n_windows)
                                   for s in REPORT_SEEDS]))
            # With a common u(x) added to BOTH mechanisms the gap is unchanged,
            # so we verify by simulating the full residual rule with u present.
            shifted = 0.0
            # DIFFERENT seeds from the plain arm on purpose: with shared seeds
            # the two arms cancel to the bit and the check degenerates into
            # verifying arithmetic. Independent draws make it a real
            # Monte-Carlo comparison against the noise band.
            for s in (s0 + 1000 for s0 in REPORT_SEEDS):
                rng = _rng(s)
                errs, done = 0, 0
                chunk = max(1, min(n_windows, 4_000_000 // n))
                gap_rms = np.sqrt(rho)
                while done < n_windows:
                    m = min(chunk, n_windows - done)
                    x = rng.uniform(-1.0, 1.0, size=(m, n))
                    g = design.gap(x, rho)
                    u = multiple * gap_rms * np.cos(3.0 * x)   # shared, large
                    eps = rng.normal(0.0, 1.0, size=(m, n))
                    y = u + 0.5 * g + eps                       # truth = "+g/2"
                    r_true = np.sum((y - (u + 0.5 * g)) ** 2, axis=1)
                    r_alt = np.sum((y - (u - 0.5 * g)) ** 2, axis=1)
                    errs += int(np.sum(r_alt < r_true))
                    done += m
                shifted += errs / n_windows
            shifted /= len(REPORT_SEEDS)
            band = _band(plain, n_windows * len(REPORT_SEEDS))
            rows.append({"n": n, "rho": rho, "common_multiple": multiple,
                         "error_without_common": plain,
                         "error_with_common": shifted,
                         "abs_diff": abs(plain - shifted),
                         "noise_band_1sd": band,
                         "unchanged_within_3sd": abs(plain - shifted) <= 3 * band})
    df = pd.DataFrame(rows)
    _mkdirs()
    df.to_csv(RESULTS / "v4_common_component.csv", index=False)
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# V7 — mechanism space: the bound is attainable, and rho is measurable
# ---------------------------------------------------------------------------
#: V7 draws real windows rather than gap designs, so it is the one block whose
#: cost is set by the number of windows. 600 is where the label-free estimator
#: settles in this battery; at 150 windows its trace is too noisy to read. The
#: real bands have tens of thousands.
V7_WINDOWS = 600
V7_N = 64
V7_RHO_GRID = (0.05, 0.1, 0.25, 1.0)
#: The shared component the regimes are given, as a multiple of the gap's rms.
#: The oracle is invariant to it by V4; V7 asks whether the STATISTIC is.
V7_SHARED = (0.0, 16.0)


def _v7_windows(design: Design, rho: float, shared: float, seed: int,
                n: int = V7_N, m: int = V7_WINDOWS):
    """m windows on ``x ~ U[-1, 1]``, gap ``design.gap``, shared ``shared * x^2``.

    The shared component is ``x^2``: inside the symbolic library, which is
    where V4's invariance is claimed to carry over to the statistic. The
    'sign' design's gap is NOT in that library, which is the other half of
    the test.
    """
    rng = _rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(m, n))
    z = rng.integers(0, 2, size=m)
    g = design.gap(x, rho)                     # E[g^2] = rho, sigma = 1
    common = shared * np.sqrt(rho) * x ** 2
    clean = common + np.where(z[:, None] == 1, 0.5 * g, -0.5 * g)
    y = clean + rng.normal(0.0, 1.0, size=(m, n))
    return x[:, :, None], y, z, clean


def run_v7(args=None) -> pd.DataFrame:
    """Mechanism space: separation, attainability, and the label-free rho.

    Three predictions, all written before the run:

    V7.1  the distance between the regime centres in mechanism space is
          ``sqrt(n rho)`` in units of the noise -- the same ``n rho`` that
          governs the oracle in V1-V2;
    V7.2  plain K-means there reaches the *exact* oracle error of V1, so the
          detection bound is attainable without knowing either law;
    V7.3  ``rho`` can be read off UNLABELLED windows from the excess spread.

    And one boundary: all three hold when the gap and the shared component
    lie in the symbolic library, and V7 includes a gap ('sign') and a
    magnitude of shared component where that fails, so the limit is measured
    rather than asserted.
    """
    from ..core.evaluation import aligned_accuracy
    from ..core.mechanism_space import (
        mechanism_features,
        mechanism_init,
        mechanism_noise,
        separation_matrix,
        separation_unlabelled,
    )
    _mkdirs()

    rows = []
    for design in DESIGNS:
        for rho in V7_RHO_GRID:
            for shared in V7_SHARED:
                for seed in REPORT_SEEDS:
                    X, y, z, _ = _v7_windows(design, rho, shared, seed)
                    S = mechanism_features(X, y)
                    sig = mechanism_noise(X, y)
                    lab = mechanism_init(X, y, 2, seed=seed)
                    sep = float(separation_matrix(S, z, 2, sigma=sig)[0, 1])
                    unl = separation_unlabelled(S, sig, V7_N)
                    rows.append({
                        "design": design.name, "kappa": design.kappa,
                        "rho": rho, "shared_x_gap": shared, "seed": seed,
                        "n": V7_N, "n_windows": V7_WINDOWS,
                        "in_library": design.name != "sign",
                        "sigma_hat": sig, "dims": int(S.shape[1]),
                        "sep_measured": sep,
                        "sep_predicted": float(np.sqrt(V7_N * rho)),
                        "kmeans_error": float(1.0 - aligned_accuracy(z, lab)),
                        "oracle_exact": exact_error(design, V7_N, rho, seed),
                        "oracle_population": population_error(V7_N, rho),
                        "rho_hat": unl["rho"],
                        "rho_hat_at_split": unl["rho_at_split"],
                        "pi_hat": unl["pi_from_split"],
                    })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "v7_mechanism_space.csv", index=False)

    df["sep_rel"] = (df.sep_measured / df.sep_predicted - 1.0).abs()
    df["rho_rel"] = (df.rho_hat / df.rho - 1.0).abs()
    df["gap_to_oracle"] = df.kmeans_error - df.oracle_exact

    verdict = []
    for (in_lib,), g in df.groupby(["in_library"]):
        base = g[g.shared_x_gap == 0.0].groupby(["design", "rho"]).kmeans_error.mean()
        big = g[g.shared_x_gap == g.shared_x_gap.max()].groupby(
            ["design", "rho"]).kmeans_error.mean()
        verdict.append({
            "in_library": bool(in_lib),
            "designs": ",".join(sorted(g.design.unique())),
            "cells": len(g),
            "sep_rel_median": float(g.sep_rel.median()),
            "sep_rel_max": float(g.sep_rel.max()),
            "rho_rel_median": float(g.rho_rel.median()),
            "rho_rel_max": float(g.rho_rel.max()),
            "gap_to_oracle_median": float(g.gap_to_oracle.median()),
            "gap_to_oracle_max": float(g.gap_to_oracle.max()),
            "invariance_max_shift": float((big - base).abs().max()),
        })
    ver = pd.DataFrame(verdict)
    ver.to_csv(RESULTS / "v7_verdict.csv", index=False)

    print(df.groupby(["design", "rho", "shared_x_gap"])[
        ["sep_measured", "sep_predicted", "kmeans_error", "oracle_exact",
         "rho_hat"]].mean().round(4).to_string())
    print()
    print(ver.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# V6 — the two headline figures
# ---------------------------------------------------------------------------
def run_v6(args=None, n_designs: int = 40_000, target: float = 0.05) -> None:
    """Figure 1: the fan -- identical n*rho, different difficulty."""
    import matplotlib.pyplot as plt

    design = DESIGN_BY_NAME["linear"]
    z = float(norm.isf(target))

    # ---- Figure 1: error vs n*rho ---------------------------------------
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    nrho = np.logspace(np.log10(2), np.log10(400), 60)
    ax.plot(nrho, Q(0.5 * np.sqrt(nrho)), lw=2.6, color="#1b1f24", zorder=4,
            label=r"population  $Q(\frac{1}{2}\sqrt{n\rho})$  — all $\rho$ collapse here")
    palette = {0.1: "#0e7490", 0.5: "#7c3aed", 2.0: "#b45309"}
    for rho, c in palette.items():
        ns = np.unique(np.round(np.logspace(np.log10(max(2, 2 / rho)),
                                            np.log10(400 / rho), 12)).astype(int))
        ex = [exact_error(design, int(n), rho, REPORT_SEEDS[0], n_designs) for n in ns]
        ax.plot(ns * rho, ex, "o-", color=c, ms=4.5, lw=1.7, zorder=3,
                label=fr"exact random design, $\rho$={rho}")
        corr = [population_error(int(n), rho) *
                (1 + design.kappa * (1 + 1 / z**2) * rho / 16.0) for n in ns]
        ax.plot(ns * rho, corr, "--", color=c, lw=1.3, alpha=.85, zorder=2)
    ax.axhline(target, color="#15803d", lw=1.4, ls="-.", zorder=1,
               label=f"{target:.0%} target")
    ax.scatter([], [], s=48, facecolors="none", edgecolors="#1b1f24",
               label="reserved: LR-DSR (Phase 2)")
    ax.set(xscale="log", yscale="log", xlabel=r"$n\rho$",
           ylabel="P(error)", ylim=(1e-4, 0.6), xlim=(2, 90),
           title="Figure 1 — the fan: identical $n\\rho$, different difficulty\n"
                 "the exact curves rise above the population curve, and the gap grows with $\\rho$")
    ax.grid(alpha=.25, lw=.6)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.annotate("no rule can get below\nthe population curve", xy=(2.6, 1.9e-4),
                fontsize=8, color="#6b7280")
    ax.annotate("oracle-infeasible\nat that $\\rho$", xy=(23, .13), fontsize=8,
                color="#6b7280")
    _save_fig(fig, "f1_fan")

    print(f"wrote {FIGURES}/f1_fan.[png|pdf]")


# ---------------------------------------------------------------------------
#: The blocks, in order, with what each one settles.
BLOCKS = (
    ("V1", "the identity is exact, not asymptotic", "run_v1"),
    ("V2", "the population form needs a shape correction (kappa)", "run_v2"),
    ("V2.4", "the anomaly, rerun", "run_v2_anomaly"),
    ("V3", "the K-ary sandwich", "run_v3"),
    ("V4", "a shared component changes nothing", "run_v4"),
    ("V6", "the headline figure", "run_v6"),
    ("V7", "mechanism space attains the bound", "run_v7"),
)


def run_all(args=None) -> None:
    """Every block, in order. The oracle only -- no estimator anywhere."""
    for tag, what, fn in BLOCKS:
        print(f"\n=== {tag} — {what} ===", flush=True)
        globals()[fn](args)
    print(f"\nCSVs -> {RESULTS}/   figures -> {FIGURES}/")
