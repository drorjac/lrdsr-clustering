"""The problem zoo: twelve regime-recovery problems, each hard for its own reason.

The estimator block asks whether LR-DSR reaches the ceiling on four pairs.
This asks the wider question -- *which shapes of problem* does it reach it
on -- with law families chosen to break a specific assumption each:

* the law is **not in the library** (a frequency shift, a kink, a power
  exponent, a damping envelope, a saturation curve), so the search can only
  return a surrogate;
* the gap is **outside the span** of the library altogether (a high
  frequency, a moving step), so no combination of terms can see all of it --
  the one kind of "out of library" that actually costs separation;
* the problem has **more than two regimes** (K = 3 and K = 4);
* the law has **two inputs** and the regimes differ only in how they combine;
* the two laws are a **rescaled copy** of each other, so the gap is
  proportional to the law and vanishes where it does;
* the two laws produce ``y`` with **matched first two moments**, so the
  distribution of the response alone says nothing.

Every problem draws its inputs from ONE marginal shared by all regimes, so
the covariate carries no regime information and the equation has to. The
noise level is set from the realised gap, so ``rho`` is a knob: for ``K > 2``
it is the **smallest** pairwise separation, i.e. the hardest pair sits at the
requested ``rho`` and the others above it.

``in_library`` says whether *every* true law lies in the span of the fast
backend's term library (``1, x, x^2, x^3, log1p|x|, sin x, cos x, sin 2x``
per input, plus pairwise products). It is checked, not asserted: see
:func:`in_library_residual`. "Out of the library" is a statement about the
*expression*, not about what the span can represent: the library is rich,
and :func:`gap_outside_library` measures how much of each **gap** it misses
-- which is what assignment runs on.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Callable

import numpy as np

from lrdsr.core.soft import library_terms

Law = Callable[[np.ndarray], np.ndarray]


def _u(lo, hi, d=1):
    """A sampler ``(rng, shape) -> X`` of shape ``shape + (d,)``, uniform."""
    def draw(rng, shape):
        return rng.uniform(lo, hi, size=(*shape, d))
    return draw


@dataclass(frozen=True)
class Problem:
    """One regime-recovery problem.

    ``laws[k](X)`` takes ``X`` of shape ``(..., d)`` and returns ``(...)``.
    ``sample(rng, shape)`` draws the inputs every regime shares.
    ``x_range`` is the plotting / evaluation range of the first input.
    """
    name: str
    laws: tuple[Law, ...]
    truths: tuple[str, ...]
    sample: Callable
    d: int
    x_range: tuple[float, float]
    in_library: bool
    why: str

    @property
    def K(self) -> int:
        return len(self.laws)

    @property
    def feature_names(self) -> list[str]:
        return ["x"] if self.d == 1 else [f"x{j + 1}" for j in range(self.d)]

    @property
    def key(self) -> int:
        """A stable integer for seeding, independent of registry order."""
        return zlib.crc32(self.name.encode())


def _x(X):
    return X[..., 0]


def _relu(v):
    return np.maximum(v, 0.0)


# matched moments: x ~ U(-a, a) gives E x = 0, Var x = a^2/3; the quadratic
# c (x^2 - a^2/3) has mean 0 and variance c^2 * 4 a^4 / 45, equal to a^2/3
# when c = sqrt(15) / (2 a).
_A_MM = 2.0
_C_MM = np.sqrt(15.0) / (2.0 * _A_MM)

PROBLEMS: tuple[Problem, ...] = (
    Problem(
        "frequency_shift",
        (lambda X: np.sin(_x(X)), lambda X: np.sin(1.3 * _x(X))),
        ("sin(x)", "sin(1.3*x)"), _u(-3.0, 3.0), 1, (-3.0, 3.0), False,
        "same shape, 30% faster: sin(1.3x) is not a library term"),
    Problem(
        "saturation",
        (lambda X: 1.0 - np.exp(-_x(X)), lambda X: np.tanh(_x(X))),
        ("1 - exp(-x)", "tanh(x)"), _u(0.0, 3.0), 1, (0.0, 3.0), False,
        "two saturating curves with the same slope at 0; neither in the library"),
    Problem(
        "kink",
        (lambda X: _relu(_x(X)), lambda X: _relu(_x(X) - 0.8)),
        ("max(0, x)", "max(0, x - 0.8)"), _u(-2.0, 2.0), 1, (-2.0, 2.0), False,
        "a hinge that moves; identical on x < 0, non-smooth, out of library"),
    Problem(
        "interaction",
        (lambda X: X[..., 0] * X[..., 1], lambda X: X[..., 0] + X[..., 1]),
        ("x1*x2", "x1 + x2"), _u(-2.0, 2.0, d=2), 2, (-2.0, 2.0), True,
        "two inputs; the regimes differ only in how they combine them"),
    Problem(
        "damping",
        (lambda X: np.sin(2 * _x(X)),
         lambda X: np.sin(2 * _x(X)) * np.exp(-0.3 * _x(X))),
        ("sin(2*x)", "sin(2*x)*exp(-0.3*x)"), _u(0.0, 4.0), 1, (0.0, 4.0), False,
        "an envelope; the gap is zero at x=0 and grows; the damped law is out of library"),
    Problem(
        "power_exponent",
        (lambda X: _x(X) ** 1.5, lambda X: _x(X) ** 1.8),
        ("x^1.5", "x^1.8"), _u(0.2, 4.0), 1, (0.2, 4.0), False,
        "two power laws, close exponents, both between library powers"),
    Problem(
        "three_way",
        (lambda X: _x(X) ** 2, lambda X: 0.5 * _x(X) ** 3,
         lambda X: 2.0 * np.sin(_x(X))),
        ("x^2", "0.5*x^3", "2*sin(x)"), _u(-2.0, 2.0), 1, (-2.0, 2.0), True,
        "K = 3; cubic and 2 sin(x) nearly coincide on |x| < 1"),
    Problem(
        "four_way",
        (lambda X: _x(X), lambda X: _x(X) ** 2 - 1.0,
         lambda X: 1.5 * np.sin(2 * _x(X)), lambda X: 2.0 * np.log1p(np.abs(_x(X))) - 1.0),
        ("x", "x^2 - 1", "1.5*sin(2*x)", "2*log1p(|x|) - 1"), _u(-2.0, 2.0), 1,
        (-2.0, 2.0), True,
        "K = 4, six pairs, the tightest one sets the noise"),
    Problem(
        "rescaled_copy",
        (lambda X: _x(X) ** 2 + np.sin(_x(X)),
         lambda X: 0.7 * (_x(X) ** 2 + np.sin(_x(X)))),
        ("x^2 + sin(x)", "0.7*(x^2 + sin(x))"), _u(-2.0, 2.0), 1, (-2.0, 2.0), True,
        "same shape, 70% amplitude: the gap is proportional to the law"),
    Problem(
        "moment_matched",
        (lambda X: _x(X), lambda X: _C_MM * (_x(X) ** 2 - _A_MM ** 2 / 3.0)),
        ("x", f"{_C_MM:.4g}*(x^2 - {_A_MM ** 2 / 3:.4g})"), _u(-_A_MM, _A_MM), 1,
        (-_A_MM, _A_MM), True,
        "y has the same mean and variance under both laws; only the shape differs"),
)
PROBLEMS = (*PROBLEMS,
    Problem(
        "high_frequency",
        (lambda X: np.sin(4.0 * _x(X)), lambda X: np.sin(4.6 * _x(X))),
        ("sin(4*x)", "sin(4.6*x)"), _u(-3.0, 3.0), 1, (-3.0, 3.0), False,
        "oscillation faster than any library term: ~37% of the gap is outside the span"),
    Problem(
        "moving_step",
        (lambda X: np.sign(_x(X) - 0.3), lambda X: np.sign(_x(X) + 0.3)),
        ("sign(x - 0.3)", "sign(x + 0.3)"), _u(-2.0, 2.0), 1, (-2.0, 2.0), False,
        "a discontinuity that moves; the laws differ only on |x| < 0.3"),
)
PROBLEM_BY_NAME = {p.name: p for p in PROBLEMS}


# ==========================================================================
# calibration and data
# ==========================================================================
def pairwise_gap_ms(problem: Problem, seed: int = 0, n: int = 400_000) -> np.ndarray:
    """``(K, K)`` matrix of ``E[(f_j - f_k)^2]`` under the input marginal."""
    X = problem.sample(np.random.default_rng(seed), (n,))
    F = np.vstack([f(X) for f in problem.laws])
    K = problem.K
    G = np.zeros((K, K))
    for j in range(K):
        for k in range(K):
            G[j, k] = float(np.mean((F[j] - F[k]) ** 2))
    return G


def sigma_for_rho(problem: Problem, rho: float) -> float:
    """Noise sd putting the *closest* pair of laws at separation ``rho``."""
    G = pairwise_gap_ms(problem)
    off = G[~np.eye(problem.K, dtype=bool)]
    return float(np.sqrt(off.min() / rho))


def realised_rho(problem: Problem, sigma: float) -> np.ndarray:
    """``(K, K)`` pairwise ``rho`` at noise ``sigma``, on a fresh draw."""
    return pairwise_gap_ms(problem, seed=1) / sigma ** 2


def make_windows(problem: Problem, sigma: float, n_windows: int, window_len: int,
                 seed: int):
    """Windows of one problem. Returns ``(X_seq, y, z)``.

    ``z`` is uniform over the ``K`` regimes and is ground truth for the
    metrics only. The same ``seed`` gives the same inputs and labels at every
    ``sigma`` (common random numbers), so a sweep over ``rho`` changes the
    noise and nothing else.
    """
    rng = np.random.default_rng([seed, problem.key])
    X = problem.sample(rng, (n_windows, window_len))
    z = rng.integers(0, problem.K, size=n_windows)
    clean = np.empty((n_windows, window_len))
    for k, f in enumerate(problem.laws):
        m = z == k
        if m.any():
            clean[m] = f(X[m])
    y = clean + rng.normal(0.0, 1.0, size=clean.shape) * sigma
    return X, y, z


def oracle_labels(problem: Problem, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The Bayes rule with equal priors: smallest residual under the TRUE laws."""
    res = np.column_stack([np.sum((y - f(X)) ** 2, axis=1) for f in problem.laws])
    return res.argmin(axis=1)


def in_library_residual(problem: Problem, n: int = 4000, seed: int = 0) -> float:
    """Worst relative residual of projecting each true law on the library.

    Zero (to rounding) means the law is an exact combination of library
    terms; the ``in_library`` flag of every problem is checked against this
    in the tests.
    """
    X = problem.sample(np.random.default_rng(seed), (n,))
    Phi, _ = library_terms(X, feature_names=problem.feature_names)
    worst = 0.0
    for f in problem.laws:
        t = f(X)
        coef, *_ = np.linalg.lstsq(Phi, t, rcond=None)
        worst = max(worst, float(np.mean((t - Phi @ coef) ** 2) / max(np.var(t), 1e-12)))
    return worst


def eval_grid(problem: Problem, n: int = 2000) -> np.ndarray:
    """Dense inputs for scoring a recovered law: a line for d=1, a draw for d=2."""
    if problem.d == 1:
        return np.linspace(*problem.x_range, n)[:, None]
    return problem.sample(np.random.default_rng(12345), (n,))


def gap_outside_library(problem: Problem, n: int = 4000, seed: int = 0) -> float:
    """Largest share of a pairwise GAP the library cannot represent.

    ``in_library_residual`` is about each law; this is about what separates
    them, which is what assignment runs on. For each pair the gap
    ``f_j - f_k`` is projected on the library and the unexplained share of
    ``E[gap^2]`` is reported; the worst pair is returned. A law can be out of
    the library and still have a gap the library resolves almost entirely.
    """
    X = problem.sample(np.random.default_rng(seed), (n,))
    Phi, _ = library_terms(X, feature_names=problem.feature_names)
    F = [f(X) for f in problem.laws]
    worst = 0.0
    for j in range(problem.K):
        for k in range(j + 1, problem.K):
            g = F[j] - F[k]
            coef, *_ = np.linalg.lstsq(Phi, g, rcond=None)
            worst = max(worst, float(np.mean((g - Phi @ coef) ** 2) / np.mean(g ** 2)))
    return worst
