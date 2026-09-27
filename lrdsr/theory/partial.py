"""V11 -- the error of a window observed at some of its inputs, before it is scored.

V1 prices a window by its realised design: with two known laws and gap
``g = f_1 - f_0``, the error is ``Q(|g|/(2 sigma))`` over the inputs it has.
Real windows break two of V1's assumptions at once -- a day of traffic is
missing hours, and the hours it has are **correlated** (a day that runs high
at 8 am runs high at 9) -- and the classifier actually used is least squares
with the day's level profiled out, not the likelihood-ratio rule under the
true noise. This module prices *that* classifier, exactly, for any set of
observed inputs ``H``.

The rule. With the level profiled (``P`` the centring projector on ``H``),
``g~ = P g_H``, the pooled per-sample variance ``s^2`` the classifier fitted
and log prior ratio ``L = log(pi_1 / pi_0)``, a window goes to law 1 when

.. code-block:: text

    (RSS_0 - RSS_1) / (2 s^2) + L  > 0,     RSS_0 - RSS_1 = |g~|^2 + 2 g~^T e

for a window of law 1 (sign flipped for law 0). With noise ``e ~ N(0, Sigma)``
the statistic is Gaussian, ``g~^T e`` has variance ``g~^T Sigma_H g~`` (``P``
is absorbed because ``g~`` is already centred), and

.. code-block:: text

    err_1(H) = Q( (|g~|^2 / 2 + s^2 L) / sqrt(g~^T Sigma_H g~) )
    err_0(H) = Q( (|g~|^2 / 2 - s^2 L) / sqrt(g~^T Sigma_H g~) )

With ``Sigma = sigma^2 I`` and ``L = 0`` this is V1's ``Q(|g~|/(2 sigma))``. The
only estimated input is ``Sigma``: the covariance of complete windows'
residuals about their law, over the full design, restricted to ``H``. It
carries the correlation of neighbouring hours *and* the day-to-day
variation of a law within its regime -- everything that is not the gap.

Two consequences are tested on real data (``experiments/realdata/partial.py``):
**which** inputs are missing matters, not how many (a missing morning peak
removes most of ``|g~|``, a missing night almost none); and a day read hour
by hour is decided when ``|g~_t|^2 / sqrt(g~_t^T Sigma g~_t)`` has grown
enough, so the hour of decision is predicted from the laws alone.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

Q = norm.sf


def centred_gap(g_H: np.ndarray) -> np.ndarray:
    """``P g``: the gap at the observed inputs with the level profiled out."""
    g_H = np.asarray(g_H, dtype=float)
    return g_H - g_H.mean()


def gap_moments(g_H: np.ndarray, Sigma_H: np.ndarray) -> tuple[float, float]:
    """``(|g~|^2, g~^T Sigma_H g~)`` for the observed inputs."""
    gt = centred_gap(g_H)
    return float(gt @ gt), float(gt @ np.asarray(Sigma_H, float) @ gt)


def masked_error(g_H, Sigma_H, s2: float, log_prior_ratio: float = 0.0,
                 truth: int = 1) -> float:
    """V11: error of the level-profiled least-squares rule for a window of
    law ``truth`` observed at ``H`` (``g_H`` the gap there, ``Sigma_H`` the
    noise covariance there)."""
    d2, v = gap_moments(g_H, Sigma_H)
    if v <= 0 or d2 <= 0:
        return 0.5
    shift = s2 * log_prior_ratio * (1.0 if truth == 1 else -1.0)
    return float(Q((0.5 * d2 + shift) / np.sqrt(v)))


def decided_share(g_H, Sigma_H, s2: float, log_prior_ratio: float = 0.0,
                  truth: int = 1, confidence: float = 0.99) -> tuple[float, float]:
    """``(P(decided correctly), P(decided wrongly))`` at posterior ``confidence``.

    The posterior log-odds of the true law is Gaussian with mean
    ``|g~|^2 / (2 s^2) +/- L`` and variance ``g~^T Sigma g~ / s^4``; a decision
    is made when it clears ``log(c / (1 - c))`` in either direction.
    """
    d2, v = gap_moments(g_H, Sigma_H)
    thr = np.log(confidence / (1.0 - confidence))
    if v <= 0:
        return 0.0, 0.0
    mu = 0.5 * d2 / s2 + (log_prior_ratio if truth == 1 else -log_prior_ratio)
    sd = np.sqrt(v) / s2
    return float(Q((thr - mu) / sd)), float(Q((thr + mu) / sd))
