"""V9 -- the real-time ceiling: how fast a regime switch can be seen.

The batch theory (``verification.py``) prices a *window*: ``n`` samples, one
law, error ``Q(sqrt(n rho)/2)``. A stream has no windows. Samples arrive one
at a time, the regime switches at an unknown moment, and the question
becomes *how many samples after the switch until it is noticed*, against
*how often it is "noticed" when nothing happened*.

The optimal answer is CUSUM on the per-sample log-likelihood ratio
(Lorden 1971, Moustakides 1986) -- ``lrdsr.core.online.CusumSegmenter``. For
two laws with gap ``g(x) = f_1(x) - f_0(x)`` and Gaussian noise ``sigma``,
the increment ``l_t = log p_1(y_t|x_t) - log p_0(y_t|x_t)`` has

.. code-block:: text

    after the switch    E l =  rho / 2 = KL        Var l = rho + kappa rho^2 / 4
    before it           E l = -rho / 2

with ``rho = E[g^2]/sigma^2`` -- the same separation the window theory uses
-- and ``kappa = Var(g^2)/E[g^2]^2`` the same design heterogeneity. So the
window-level separation also sets the real-time one: **per-sample KL is
rho/2**.

Three levels of prediction are made and all are checked:

*First order (Lorden).* Delay ``~ h / KL = 2h / rho``; false-alarm run
length ``>= e^h``. The delay is a large-``h`` asymptote and ignores the
overshoot of the statistic over ``h`` and the reflection at zero, so it is
expected to be off at small ``h``; ``e^h`` is a bound, not an estimate.

*Brownian (Siegmund 1985, eq. 2.56).* Standardise the increments to unit
variance, drift ``mu = E l / sqrt(Var l)``, threshold ``b = h / sqrt(Var l)``,
correct the threshold for overshoot, ``b+ = b + 2 * 0.583``, and use the
Brownian-motion run length

.. code-block:: text

    ARL(mu, b) = (exp(-2 mu b+) + 2 mu b+ - 1) / (2 mu^2)

at ``mu > 0`` (delay) and ``mu < 0`` (false alarms). The ``0.583``
(``= -zeta(1/2)/sqrt(2 pi)``) is Siegmund's overshoot constant for small
Gaussian increments.

*Tilted (the default).* The Brownian formula's exponent ``2 mu b`` is the
tilt root ``theta*`` (``E e^{-theta* l} = 1``) times the threshold. For a
log-likelihood ratio that root is **exactly 1** in both directions, whatever
the shape of the gap (``E_0 e^{l} = E_1 e^{-l} = 1``), while standardising by
the variance only gets it right for a constant gap: with ``kappa > 0`` the
increments are a Gaussian scale mixture and ``2 mu^2 / Var`` underestimates
the root. Writing the same formula in the LLR's own tilt,

.. code-block:: text

    h'    = h + 2 * 0.583 * sqrt(Var l)          overshoot, as above
    delay = (h' + exp(-h') - 1) / KL
    ARL   = (exp(h') - h' - 1) / KL

which coincides with the Brownian form when the gap is constant. The only
approximation left is the overshoot constant, which is a small-increment
value: the prediction should degrade as ``rho`` grows.

Delays are measured from a zero statistic (the switch happens when the
detector is fully "reset"), which is Lorden's worst case for CUSUM and the
quantity the formula describes. False-alarm run lengths are censored at a
cap; a censored cell is reported with its censored fraction and is excluded
from the formula comparison, because its mean is only a lower bound.

    python -m experiments online    # runs V9 with the stream experiments
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lrdsr import paths

RESULTS = paths.RESULTS / "online"
REPORT_SEEDS = (11, 23, 42)

#: Siegmund's overshoot constant, -zeta(1/2)/sqrt(2 pi).
OVERSHOOT = 0.5826

#: The pair V9 runs on: the polynomial pair's gap ``g(x) = 2x`` on U[-2, 2],
#: so ``E g^2 = 16/3`` and ``kappa = E x^4 / E[x^2]^2 - 1 = 4/5``.
X_RANGE = (-2.0, 2.0)
def F0(x):
    return x ** 2 + x


def F1(x):
    return x ** 2 - x


GAP_MS = 16.0 / 3.0
KAPPA = 0.8

RHO_GRID = (0.1, 0.25, 1.0, 4.0)
H_GRID = (2.0, 4.0, 6.0, 8.0, 10.0)


# ==========================================================================
# the formulas
# ==========================================================================
def kl_per_sample(rho: float) -> float:
    """Per-sample Kullback-Leibler divergence between the two laws: ``rho/2``."""
    return 0.5 * float(rho)


def increment_variance(rho: float, kappa: float = 0.0) -> float:
    """``Var l = rho + kappa rho^2 / 4`` (either side of the switch)."""
    return float(rho) + float(kappa) * float(rho) ** 2 / 4.0


def _siegmund(mu: float, b: float) -> float:
    """Siegmund's CUSUM run length for unit-variance increments of drift ``mu``."""
    bp = b + 2.0 * OVERSHOOT
    if abs(mu) < 1e-12:
        return bp ** 2
    return (np.exp(-2.0 * mu * bp) + 2.0 * mu * bp - 1.0) / (2.0 * mu ** 2)


def _h_prime(h: float, rho: float, kappa: float) -> float:
    return float(h) + 2.0 * OVERSHOOT * np.sqrt(increment_variance(rho, kappa))


METHODS = ("first_order", "brownian", "tilted")


def predicted_delay(h: float, rho: float, kappa: float = 0.0,
                    method: str = "tilted") -> float:
    """Expected detection delay in samples after a switch, from a zero statistic.

    ``method`` is ``"first_order"`` (Lorden/Wald, ``h / KL = 2h/rho``),
    ``"brownian"`` (Siegmund, variance-standardised) or ``"tilted"``
    (Siegmund in the log-likelihood ratio's exact tilt; the default).
    """
    if method == "first_order":
        return float(h) / kl_per_sample(rho)
    if method == "brownian":
        s = np.sqrt(increment_variance(rho, kappa))
        return float(_siegmund(kl_per_sample(rho) / s, h / s))
    if method == "tilted":
        hp = _h_prime(h, rho, kappa)
        return float((hp + np.exp(-hp) - 1.0) / kl_per_sample(rho))
    raise ValueError(f"method must be one of {METHODS}")


def predicted_arl(h: float, rho: float, kappa: float = 0.0,
                  method: str = "tilted") -> float:
    """Expected run length to a false alarm in a stable regime.

    ``"first_order"`` returns Lorden's bound ``e^h`` (the true value is at
    least this); ``"brownian"`` and ``"tilted"`` as in
    :func:`predicted_delay`.
    """
    if method == "first_order":
        return float(np.exp(h))
    if method == "brownian":
        s = np.sqrt(increment_variance(rho, kappa))
        return float(_siegmund(-kl_per_sample(rho) / s, h / s))
    if method == "tilted":
        hp = _h_prime(h, rho, kappa)
        return float((np.exp(hp) - hp - 1.0) / kl_per_sample(rho))
    raise ValueError(f"method must be one of {METHODS}")


# ==========================================================================
# simulation
# ==========================================================================
def _sigma(rho: float) -> float:
    return float(np.sqrt(GAP_MS / rho))


def first_alarm(regime: int, rho: float, h: float, reps: int, cap: int,
                rng: np.random.Generator, block: int = 4096) -> np.ndarray:
    """First alarm time (samples, 1-based) of CUSUM for ``F0 -> F1``.

    The stream is generated entirely from law ``regime`` and the detector
    starts in regime 0 with a zero statistic: ``regime=1`` measures the
    delay after a switch at time 0, ``regime=0`` the run length to a false
    alarm. Vectorised over replicates; identical, sample for sample, to
    ``CusumSegmenter`` with two laws (``tests/test_online.py`` checks it).
    Runs that never alarm within ``cap`` return ``cap + 1`` (censored).
    """
    sigma = _sigma(rho)
    S = np.zeros(reps)
    out = np.full(reps, cap + 1, dtype=np.int64)
    alive = np.ones(reps, dtype=bool)
    f = F1 if regime == 1 else F0
    t0 = 0
    while t0 < cap and alive.any():
        m = min(block, cap - t0)
        idx = np.flatnonzero(alive)
        x = rng.uniform(*X_RANGE, size=(len(idx), m))
        y = f(x) + rng.normal(0.0, sigma, size=x.shape)
        llr = (-(y - F1(x)) ** 2 + (y - F0(x)) ** 2) / (2 * sigma ** 2)
        s = S[idx]
        hit = np.full(len(idx), -1)
        for j in range(m):
            s = np.maximum(0.0, s + llr[:, j])
            new = (s > h) & (hit < 0)
            hit[new] = j
            if (hit >= 0).all():
                break
        S[idx] = s
        done = hit >= 0
        out[idx[done]] = t0 + hit[done] + 1
        alive[idx[done]] = False
        t0 += m
    return out


def run_v9(reps_delay: int = 4000, reps_arl: int = 400, cap: int = 200_000,
           verbose: bool = True) -> dict:
    """Detection delay and false-alarm run length against both predictions."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows_d, rows_a = [], []
    for i, rho in enumerate(RHO_GRID):
        for j, h in enumerate(H_GRID):
            rng = np.random.default_rng(REPORT_SEEDS[0] * 1000 + 10 * i + j)
            d = first_alarm(1, rho, h, reps_delay, cap=50_000, rng=rng)
            rows_d.append({
                "rho": rho, "h": h, "reps": reps_delay,
                "delay_mc": float(d.mean()),
                "delay_se": float(d.std(ddof=1) / np.sqrt(len(d))),
                "delay_first_order": predicted_delay(h, rho, KAPPA, "first_order"),
                "delay_brownian": predicted_delay(h, rho, KAPPA, "brownian"),
                "delay_tilted": predicted_delay(h, rho, KAPPA, "tilted"),
            })
            a = first_alarm(0, rho, h, reps_arl, cap=cap, rng=rng)
            cens = a > cap
            rows_a.append({
                "rho": rho, "h": h, "reps": reps_arl, "cap": cap,
                "censored_frac": float(cens.mean()),
                # with censoring this mean is a lower bound on the ARL
                "arl_mc": float(np.minimum(a, cap).mean()),
                "arl_se": float(np.minimum(a, cap).std(ddof=1) / np.sqrt(len(a))),
                "arl_lorden_bound": predicted_arl(h, rho, KAPPA, "first_order"),
                "arl_brownian": predicted_arl(h, rho, KAPPA, "brownian"),
                "arl_tilted": predicted_arl(h, rho, KAPPA, "tilted"),
            })
            if verbose:
                r, q = rows_d[-1], rows_a[-1]
                print(f"  V9 rho={rho:<5} h={h:<4} delay {r['delay_mc']:8.1f} "
                      f"(tilted {r['delay_tilted']:8.1f}, 2h/rho "
                      f"{r['delay_first_order']:8.1f})   ARL {q['arl_mc']:10.0f} "
                      f"(tilted {q['arl_tilted']:10.0f}, e^h "
                      f"{q['arl_lorden_bound']:8.0f}, censored "
                      f"{q['censored_frac']:.0%})", flush=True)
    D, A = pd.DataFrame(rows_d), pd.DataFrame(rows_a)
    for m in ("first_order", "brownian", "tilted"):
        D[f"rel_err_{m}"] = D[f"delay_{m}"] / D.delay_mc - 1.0
    for m in ("brownian", "tilted"):
        A[f"rel_err_{m}"] = A[f"arl_{m}"] / A.arl_mc - 1.0
    A["above_lorden_bound"] = A.arl_mc >= A.arl_lorden_bound
    unc = A[A.censored_frac == 0.0]
    v = {"delay_cells": len(D), "arl_cells": len(A),
         "arl_uncensored_cells": len(unc),
         # the bound must hold everywhere, censored or not (censoring only
         # lowers the measured mean)
         "arl_cells_above_lorden_bound": int(A.above_lorden_bound.sum())}
    for m in ("first_order", "brownian", "tilted"):
        e = D[f"rel_err_{m}"].abs()
        v[f"delay_median_abs_rel_err_{m}"] = float(e.median())
        v[f"delay_max_abs_rel_err_{m}"] = float(e.max())
    for m in ("brownian", "tilted"):
        e = unc[f"rel_err_{m}"].abs()
        v[f"arl_median_abs_rel_err_{m}"] = float(e.median()) if len(e) else np.nan
        v[f"arl_max_abs_rel_err_{m}"] = float(e.max()) if len(e) else np.nan
        small = unc[unc.rho <= 1.0][f"rel_err_{m}"].abs()
        v[f"arl_max_abs_rel_err_{m}_rho_le_1"] = float(small.max()) if len(small) else np.nan
    verdict = pd.DataFrame([v])
    D.to_csv(RESULTS / "v9_delay.csv", index=False)
    A.to_csv(RESULTS / "v9_arl.csv", index=False)
    verdict.to_csv(RESULTS / "v9_verdict.csv", index=False)
    if verbose:
        print(verdict.T.to_string(header=False))
    return {"delay": D, "arl": A, "verdict": verdict}
