"""V8: what a loss costs -- the ceiling for a decision made with a robust loss.

The oracle of V1-V7 decides between two known laws by the smaller *squared*
residual, which is the likelihood-ratio rule only when the noise is
Gaussian. Any other loss ``rho`` (Huber, Cauchy, Tukey, ...) makes a different
rule,

.. code-block:: text

    decide f1  iff  T = sum_t [ rho(e0_t / s) - rho(e1_t / s) ] > 0 ,
    e0 = y - f0,   e1 = y - f1,   s a fixed scale,

and for a small per-sample gap ``delta = f1 - f0`` its error is, to leading
order,

.. code-block:: text

    P_err  =  Q( sqrt(n rho eta) / 2 ),

    eta    =  Var(e) (E psi'(u))^2 / ( s^2 E psi(u)^2 ),   u = e / s,

with ``psi = rho'``. Derivation: expand ``rho((e - delta)/s)`` to second
order in ``delta``; under ``f0`` the statistic has mean
``-E psi' sum delta^2 / 2 s^2`` and variance ``E psi^2 sum delta^2 / s^2``, and
``sum delta^2 = n rho Var(e)`` by the **convention of this module: rho is the
separation relative to the noise VARIANCE**, ``rho = E[delta^2] / Var(e)``.

Under that convention

* the squared loss has ``eta = 1`` for **every** noise -- its error depends on
  the noise only through the variance, which is its weakness under outliers;
* the likelihood-ratio loss ``-log p(e)`` has ``eta = Var(e) I(p)``, the
  Fisher information in units of the variance: the largest ``eta`` of any
  loss (Cramer-Rao), and exactly 1 for Gaussian noise;
* every other loss sits between, at ``eta = efficiency x Var(e) I(p)``, the
  familiar asymptotic relative efficiency of an M-estimator. The window
  error of a *classification* is therefore governed by the same number that
  governs the variance of an *estimate* -- a loss that is 95% efficient
  needs 1/0.95 as many samples per window for the same error.

The **scale** ``s`` is the noise's own population "normal-equivalent" MAD,
``median|e| / 0.6745`` (``s = sigma`` for Gaussian noise), so a tuning
constant such as Huber's ``delta = 1.345`` means what it means in the robust
statistics literature. ``GroupedDCSR`` uses the same scale
(``lrdsr.core.losses.noise_scale``); until 2026-09-24 it divided by the raw
MAD, ``0.6745 s``, which this module's efficiency table is what exposed --
pass ``scale=`` to :func:`efficiency` to evaluate that old convention.

``run_v8`` checks the formula by simulating the decision directly, with the
realised design integrated the same way V1 does (the prediction averages
``Q(sqrt(eta D_n / Var)/2)`` over the simulated designs, so the kappa
correction of V2 is not charged to the loss).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from itertools import pairwise

import numpy as np
import pandas as pd
from scipy import integrate
from scipy.stats import norm
from scipy.stats import t as student_t

from lrdsr import paths
from lrdsr.core.losses import HUBER_DELTA, loss_psi, loss_psi_prime, loss_value
from lrdsr.protocol import REPORT_SEEDS

Q = norm.sf
RESULTS = paths.results_dir("losses", figs=False)


# ==========================================================================
# noise families
# ==========================================================================
@dataclass(frozen=True)
class Noise:
    """A symmetric zero-mean noise law in its natural units.

    ``sample(rng, size)`` draws, ``pdf``/``logpdf`` evaluate, ``score`` is
    ``-d log p / de`` (the likelihood-ratio influence), ``var`` the variance.
    """
    name: str
    sample: Callable
    pdf: Callable
    logpdf: Callable
    score: Callable
    var: float
    what: str

    @property
    def sd(self) -> float:
        return float(np.sqrt(self.var))

    @cached_property
    def mad_scale(self) -> float:
        """Normal-equivalent MAD, ``median|e| / 0.6745``: the default ``s``."""
        from scipy.optimize import brentq
        med = brentq(lambda a: 2 * _quad(self.pdf, 0.0, a) - 0.5, 1e-9, 100 * self.sd)
        return float(med / norm.ppf(0.75))


def _gaussian(sd: float = 1.0) -> Noise:
    return Noise("gaussian",
                 lambda rng, size: rng.normal(0.0, sd, size),
                 lambda e: norm.pdf(e, scale=sd),
                 lambda e: norm.logpdf(e, scale=sd),
                 lambda e: np.asarray(e) / sd ** 2,
                 sd ** 2, "N(0, 1)")


def _contaminated(eps: float, k: float = 10.0) -> Noise:
    """``(1-eps) N(0,1) + eps N(0,k^2)``: a gross-error model."""
    def pdf(e):
        return (1 - eps) * norm.pdf(e) + eps * norm.pdf(e, scale=k)

    def logpdf(e):
        return np.logaddexp(np.log(1 - eps) + norm.logpdf(e),
                            np.log(eps) + norm.logpdf(e, scale=k))

    def score(e):
        # posterior weight of the wide component, computed in logs: at large
        # |e| both densities underflow and the ratio would be 0/0
        e = np.asarray(e, dtype=float)
        la = np.log(1 - eps) + norm.logpdf(e)
        lb = np.log(eps) + norm.logpdf(e, scale=k)
        wb = np.exp(lb - np.logaddexp(la, lb))
        return (1 - wb) * e + wb * e / k ** 2

    def sample(rng, size):
        out = rng.normal(0.0, 1.0, size)
        wild = rng.random(size) < eps
        out[wild] *= k
        return out

    return Noise(f"contaminated_{round(100 * eps)}", sample, pdf, logpdf, score,
                 (1 - eps) + eps * k ** 2, f"{1 - eps:g} N(0,1) + {eps:g} N(0,{k:g}^2)")


def _student(nu: float = 3.0) -> Noise:
    return Noise(f"student_t{nu:g}",
                 lambda rng, size: rng.standard_t(nu, size),
                 lambda e: student_t.pdf(e, nu),
                 lambda e: student_t.logpdf(e, nu),
                 lambda e: (nu + 1) * np.asarray(e) / (nu + np.asarray(e) ** 2),
                 nu / (nu - 2), f"t_{nu:g}")


def _laplace(b: float = 1.0) -> Noise:
    return Noise("laplace",
                 lambda rng, size: rng.laplace(0.0, b, size),
                 lambda e: np.exp(-np.abs(e) / b) / (2 * b),
                 lambda e: -np.abs(e) / b - np.log(2 * b),
                 lambda e: np.sign(e) / b,
                 2 * b ** 2, "Laplace(0, 1)")


def noise_families() -> dict[str, Noise]:
    """The five noise laws every loss is judged under, by name."""
    fams = [_gaussian(), _laplace(), _student(3.0), _contaminated(0.1),
            _contaminated(0.2)]
    return {f.name: f for f in fams}


# ==========================================================================
# efficiency
# ==========================================================================
#: The losses V8 compares, with the parameters they are used at. ``lrt`` is
#: ``-log p(e)`` of the true noise: not a loss one can choose without knowing
#: the noise, but the ceiling every other loss is measured against.
LOSS_GRID = {
    "squared": {},
    "absolute": {},
    "huber": {"delta": HUBER_DELTA},
    "cauchy": {},
    "tukey": {},
    "student_t": {"nu": 4.0},
}


def _quad(f, a, b):
    return integrate.quad(f, a, b, limit=400, epsabs=1e-12, epsrel=1e-10)[0]


def _expect(g, noise: Noise, breaks=()) -> float:
    """``E g(e)`` for a symmetric noise, integrating over ``e >= 0`` and doubling."""
    pts = sorted({0.0, *[abs(b) for b in breaks if np.isfinite(b)], np.inf})
    return 2.0 * sum(_quad(lambda e: g(e) * noise.pdf(e), a, b)
                     for a, b in pairwise(pts))


def efficiency(loss: str, noise: Noise, scale: float | None = None, **params) -> float:
    """``eta`` of a loss under a noise, by quadrature (see module docstring).

    ``scale`` defaults to the noise's normal-equivalent MAD. For ``absolute``
    the curvature is a point mass, ``E psi'(u) = 2 s p(0)``.
    """
    s = noise.mad_scale if scale is None else float(scale)
    if loss == "lrt":
        return lrt_efficiency(noise)
    # the kinks of psi, in e units, so quad does not straddle them
    kinks = {"huber": [params.get("delta", HUBER_DELTA) * s],
             "tukey": [params.get("c", 4.685) * s]}.get(loss, [])
    e_psi2 = _expect(lambda e: float(loss_psi(e / s, loss, **params)) ** 2, noise, kinks)
    if loss == "absolute":
        e_dpsi = 2.0 * s * float(noise.pdf(0.0))
    else:
        e_dpsi = _expect(lambda e: float(loss_psi_prime(e / s, loss, **params)), noise,
                         kinks)
    return float(noise.var * e_dpsi ** 2 / (s ** 2 * e_psi2))


def lrt_efficiency(noise: Noise) -> float:
    """``Var(e) I(p)``: the efficiency of the likelihood-ratio loss, the maximum."""
    if noise.name == "laplace":
        return float(noise.var)            # score is +-1/b, I = 1/b^2, b = 1
    fisher = _expect(lambda e: float(noise.score(e)) ** 2, noise)
    return float(noise.var * fisher)


def predicted_error(n: int, rho: float, eta: float) -> float:
    """The population form ``Q(sqrt(n rho eta) / 2)``."""
    return float(Q(0.5 * np.sqrt(n * rho * eta)))


# ==========================================================================
# simulation
# ==========================================================================
def gap(x: np.ndarray) -> np.ndarray:
    """The polynomial pair's gap, ``(x^2 + x) - (x^2 - x) = 2x`` on U[-2, 2]."""
    return 2.0 * x


GAP_MS = 16.0 / 3.0            # E[(2x)^2], x ~ U[-2, 2]


def _rho_of(loss: str, noise: Noise, s: float, params: dict):
    """The per-sample loss on a residual in e units."""
    if loss == "lrt":
        return lambda e: -noise.logpdf(e)
    return lambda e: loss_value(e / s, loss, **params)


def simulate_error(noise: Noise, n: int, rho: float, seed: int,
                   losses: dict | None = None, n_windows: int = 30_000,
                   scale: float | None = None) -> dict:
    """Monte-Carlo window error of the two-law rule under every loss at once.

    One draw of designs and noise per call, shared by all losses, so their
    differences are not simulation noise. Truth is ``f0`` for every window:
    the rule and the noise are symmetric, so the error under ``f1`` is the
    same. Returns ``{loss: (simulated, design_predicted)}`` plus ``D_n``
    statistics; ``design_predicted`` averages ``Q(sqrt(eta D_n / Var)/2)``
    over the simulated designs.
    """
    losses = dict(LOSS_GRID if losses is None else losses)
    losses.setdefault("lrt", {})
    s = noise.mad_scale if scale is None else float(scale)
    rng = np.random.default_rng(seed)
    amp = np.sqrt(rho * noise.var / GAP_MS)       # scales the gap to hit rho
    x = rng.uniform(-2.0, 2.0, (n_windows, n))
    delta = amp * gap(x)
    e = noise.sample(rng, (n_windows, n))
    d_n = np.sum(delta ** 2, axis=1)
    out = {}
    for name, params in losses.items():
        L = _rho_of(name, noise, s, params)
        T = np.sum(L(e) - L(e - delta), axis=1)        # > 0: prefers f1 -> error
        # ties (a bounded loss saturating on both sides) are broken by a coin
        err = np.mean(T > 0) + 0.5 * np.mean(T == 0)
        eta = efficiency(name, noise, scale=s, **params)
        pred = float(np.mean(Q(0.5 * np.sqrt(eta * d_n / noise.var))))
        out[name] = (float(err), pred, eta)
    return out


#: Losses whose psi is discontinuous: the second-order expansion behind
#: ``eta`` is then only first-order accurate in the gap (``|e| - |e - delta|``
#: equals ``delta sign(e)`` except on a band of width ``delta``), so they
#: approach the formula as ``delta``, not ``delta^2``. The LRT loss under
#: Laplace noise is the absolute loss.
NON_SMOOTH = {("absolute", None), ("lrt", "laplace")}

#: Local-gap bins for the verdict: rms per-sample gap in units of the scale.
GAP_BINS = (0.0, 0.1, 0.25, 0.5, np.inf)


def _smooth(loss: str, noise: str) -> bool:
    return (loss, None) not in NON_SMOOTH and (loss, noise) not in NON_SMOOTH


def run_v8(n_grid=(16, 64), nrho_grid=(0.25, 1.0, 2.25, 4.0, 9.0, 16.0, 25.0),
           n_windows: int = 30_000, verbose: bool = True) -> pd.DataFrame:
    """V8: predicted vs simulated window error, loss x noise x (n, n rho).

    Writes ``v8_efficiency.csv`` (one row per cell, seeds pooled) and
    ``v8_verdict.csv`` (per loss and local-gap bin: the share of resolvable
    cells the formula matched within 3 s.d. of the pooled simulation).

    ``local_gap = sqrt(rho Var(e)) / s`` is the rms per-sample gap in units
    of the loss's scale -- the quantity the expansion needs small. It is not
    ``n rho``: at fixed ``n rho`` it shrinks like ``1/sqrt(n)``, and at fixed
    ``rho`` it is ``sqrt(Var)/s`` times larger for heavy-tailed noise, whose
    variance is set by tails the robust scale ignores (8.7x for the 10%
    contamination).
    """
    rows = []
    fams = noise_families()
    for noise in fams.values():
        s = noise.mad_scale
        lrt_eta = lrt_efficiency(noise)
        for n in n_grid:
            for nrho in nrho_grid:
                rho = nrho / n
                per_seed = [simulate_error(noise, n, rho, seed, n_windows=n_windows)
                            for seed in REPORT_SEEDS]
                m = n_windows * len(REPORT_SEEDS)
                for name in per_seed[0]:
                    sim = float(np.mean([p[name][0] for p in per_seed]))
                    pred = float(np.mean([p[name][1] for p in per_seed]))
                    eta = per_seed[0][name][2]
                    band = float(np.sqrt(max(pred, 1.0 / m) * (1 - pred) / m))
                    resolvable = pred * m >= 10.0
                    rows.append({
                        "noise": noise.name, "loss": name, "n": n, "n_rho": nrho,
                        "rho": rho, "eta": eta, "eta_rel_lrt": eta / lrt_eta,
                        "local_gap": float(np.sqrt(rho * noise.var) / s),
                        "smooth_psi": _smooth(name, noise.name),
                        "simulated": sim, "predicted_design": pred,
                        "predicted_population": predicted_error(n, rho, eta),
                        "band_1sd": band, "z_gap": (sim - pred) / band,
                        "resolvable": resolvable,
                        "within_band": bool(abs(sim - pred) <= 3 * band) if resolvable else None,
                        "rel_gap": (sim - pred) / max(pred, 1e-12),
                    })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "v8_efficiency.csv", index=False)

    res = df[df.resolvable].copy()
    res["within_band"] = res["within_band"].astype(bool)
    res["gap_bin"] = pd.cut(res.local_gap, GAP_BINS, right=True).astype(str)
    verdict = (res.groupby(["loss", "gap_bin"], observed=True)
               .agg(cells=("within_band", "size"),
                    within=("within_band", "sum"),
                    median_abs_rel_gap=("rel_gap", lambda v: float(v.abs().median())),
                    worst_abs_rel_gap=("rel_gap", lambda v: float(v.abs().max())))
               .reset_index())
    verdict["share_within"] = verdict.within / verdict.cells
    verdict.to_csv(RESULTS / "v8_verdict.csv", index=False)
    if verbose:
        print(df.pivot_table(index="noise", columns="loss", values="eta",
                             aggfunc="first").round(3).to_string())
        print(verdict.to_string(index=False))
    return df
