"""V10 -- the learning curve of a law classifier: what a basis dimension costs.

With the two laws known, a window is classified with error
``Q(sqrt(n rho) / 2)`` -- the same ceiling clustering is held to. A
classifier does not know the laws; it estimates each from ``m`` labelled
windows. How much does that cost, and what does it depend on?

Work in mechanism space. With a basis of rank ``p`` that spans the gap, a
window is a point ``s ~ N(mu_class, sigma^2 I_p)`` and the two centres are
``D = sqrt(n rho)`` noise units apart. The law classifier with a shared
noise level and equal priors is the nearest-centroid rule with estimated
centres ``mu^_c = mu_c + e_c``, ``e_c ~ N(0, I_p / m)``. Rotating so that the
gap lies on the first axis, its error given the estimates depends on three
scalars only,

.. code-block:: text

    d = mu^_1 - mu^_0 = Delta + e,          e ~ N(0, 2 I / m)
    A = d^T Delta / (2 |d|) = (D^2 + D e_1) / (2 |d|),
    |d|^2 = (D + e_1)^2 + (2/m) chi^2_{p-1},
    z = d^T (e_0 + e_1) / (2 |d|) ~ N(0, 1/(2m))       the midpoint's offset

    P_err = E[ Q(A - z) + Q(A + z) ] / 2                           (exact)

and to first order in ``1/m``, concentrating ``|d|``,

.. code-block:: text

    P_err  ~  Q( D^2 / (2 sqrt(D^2 + 2 p / m)) )                   (first order)

-- Raudys's formula for the Euclidean-distance classifier, specialised to
the space where it is the right rule. Three readings:

* ``m -> inf`` gives the ceiling ``Q(D / 2)``: the classifier is the oracle
  once the laws are known.
* **The price is ``p / m``, not ``n / m``.** A raw-profile classifier on the
  same windows is the same rule with ``p = n`` (the window length). A law
  classifier pays for the basis dimension only, so with few labelled windows
  it wins by exactly the ratio of the two -- provided the basis spans the
  gap. If it spans only a share ``1 - g`` of it, ``D^2`` becomes
  ``(1 - g) D^2``: the bias the kernel bases of ``lrdsr.core.kernel`` trade
  against ``p``.
* The number of labelled windows needed to get within a factor of the
  ceiling grows linearly in ``p`` and inversely in ``D^2``.

``run_v10`` checks the exact form and the first-order form against the
actual ``LawClassifier`` on simulated windows, at fixed and at random
designs.

**At a random design two corrections are needed**, both measured and neither
tuned. (i) A law estimated from ``N = m n`` randomly placed samples has the
inverse-Wishart prediction variance ``N / (N - p - 1)`` times the fixed-design
one, so the design costs ``(p + 1) / n`` of a window per class:
``m_eff = m - (p + 1) / n``. (ii) The oracle itself sits above ``Q(D / 2)`` by
the Jensen gap of V1, which adds. :func:`plugin_error_random_design` is the
exact form with both; it holds while ``p / (m n)`` is small and fails as the
inverse-Wishart tail takes over (``v10_verdict`` bins it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from lrdsr import paths

Q = norm.sf
RESULTS = paths.RESULTS / "classify"
REPORT_SEEDS = (11, 23, 42)


def plugin_error_first_order(D2: float, p: int, m: float) -> float:
    """``Q(D^2 / (2 sqrt(D^2 + 2p/m)))``: Raudys's first-order plug-in error."""
    D2 = float(D2)
    return float(Q(D2 / (2.0 * np.sqrt(D2 + 2.0 * p / m))))


def plugin_error(D2: float, p: int, m: float, draws: int = 200_000,
                 seed: int = 0) -> float:
    """Expected plug-in error of nearest-centroid, by the three-scalar reduction.

    Exact for isotropic Gaussian windows in ``p`` dimensions with ``m``
    training windows per class and known noise; a Monte-Carlo average over
    the estimation noise only (the error given the estimates is closed
    form), so ``draws`` controls a small, smooth sampling error.
    """
    rng = np.random.default_rng(seed)
    D = np.sqrt(float(D2))
    e1 = rng.normal(0.0, np.sqrt(2.0 / m), draws)
    rest = (2.0 / m) * rng.chisquare(p - 1, draws) if p > 1 else 0.0
    dn = np.sqrt((D + e1) ** 2 + rest)
    A = (D * D + D * e1) / (2.0 * dn)
    z = rng.normal(0.0, np.sqrt(1.0 / (2.0 * m)), draws)
    return float(np.mean(0.5 * (Q(A - z) + Q(A + z))))


def plugin_error_random_design(D2: float, p: int, m: float, n: int,
                               oracle_gap: float = 0.0, **kw) -> float:
    """The exact form at ``m_eff = m - (p + 1) / n``, plus the oracle's own
    Jensen gap (``oracle_gap``, the simulated oracle minus ``Q(D/2)``)."""
    m_eff = m - (p + 1.0) / n
    if m_eff <= 0:
        return 0.5
    return plugin_error(D2, p, m_eff, **kw) + float(oracle_gap)


def labelled_windows_needed(D2: float, p: int, factor: float = 1.5) -> float:
    """Smallest ``m`` (first order) at which the error is within ``factor`` of
    the ceiling ``Q(D / 2)``. Grows linearly in ``p``."""
    target = factor * Q(np.sqrt(D2) / 2.0)
    lo, hi = 1e-3, 1e7
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if plugin_error_first_order(D2, p, mid) > target:
            lo = mid
        else:
            hi = mid
    return float(hi)


# ==========================================================================
# verification by simulation
# ==========================================================================
N_WINDOW = 48
P_GRID = (3, 7, 15, 31)
M_GRID = (1, 2, 4, 8, 32)
D2_GRID = (4.0, 9.0, 16.0)
N_TEST = 4000


def _laws(p_signal: int = 3):
    """Two laws whose gap lies in the first ``p_signal`` cosine terms."""
    from lrdsr.core.kernel import CosineBasis
    b = CosineBasis(p_signal, lo=0.0, hi=1.0)
    g = np.zeros(p_signal)
    g[1:] = 1.0                                    # gap orthogonal to the level
    base = np.zeros(p_signal)
    base[0], base[1] = 0.3, 0.5
    return b, base, base + g


def _simulate(design: str, D2: float, p: int, m: int, seed: int, n: int = N_WINDOW,
              n_test: int = N_TEST) -> tuple[float, float]:
    """(classifier error, oracle error) on ``n_test`` windows per class."""
    from lrdsr.core.classify import LawClassifier
    from lrdsr.core.kernel import CosineBasis

    rng = np.random.default_rng([seed, p, m, int(D2 * 100), design == "random"])
    b, beta0, beta1 = _laws()

    def draw(W):
        if design == "fixed":
            x = np.broadcast_to((np.arange(n) + 0.5) / n, (W, n))
        else:
            x = rng.uniform(0.0, 1.0, (W, n))
        return x[..., None].copy()

    # sigma from the POPULATION gap: n E[g^2] / sigma^2 = D2
    xx = np.linspace(0, 1, 20001)[:, None]
    gms = float(np.mean((b(xx) @ (beta1 - beta0)) ** 2))
    sigma = np.sqrt(n * gms / D2)

    def windows(W, cls):
        X = draw(W)
        B = b(X.reshape(-1, 1)).reshape(W, n, -1)
        beta = beta1 if cls else beta0
        return X, B @ beta + rng.normal(0.0, sigma, (W, n))

    Xtr0, ytr0 = windows(m, 0)
    Xtr1, ytr1 = windows(m, 1)
    Xte0, yte0 = windows(n_test, 0)
    Xte1, yte1 = windows(n_test, 1)
    clf = LawClassifier(basis=CosineBasis(p, lo=0.0, hi=1.0), nuisance=None,
                        priors="uniform").fit(
        np.concatenate([Xtr0, Xtr1]), np.concatenate([ytr0, ytr1]),
        np.r_[np.zeros(m, int), np.ones(m, int)])
    Xte, yte = np.concatenate([Xte0, Xte1]), np.concatenate([yte0, yte1])
    truth = np.r_[np.zeros(n_test, int), np.ones(n_test, int)]
    err = float(np.mean(clf.predict(Xte, yte) != truth))
    Bte = b(Xte.reshape(-1, 1)).reshape(len(Xte), n, -1)
    r0 = np.sum((yte - Bte @ beta0) ** 2, axis=1)
    r1 = np.sum((yte - Bte @ beta1) ** 2, axis=1)
    oracle = float(np.mean((r1 < r0).astype(int) != truth))
    return err, oracle


def _cell(design, D2, p, m, seeds, reps) -> dict:
    errs, orcs = [], []
    for seed in seeds:
        for r in range(reps):
            e, o = _simulate(design, D2, p, m, seed * 1000 + r, n_test=N_TEST // 4)
            errs.append(e)
            orcs.append(o)
    errs = np.asarray(errs)
    return {
        "design": design, "D2": D2, "p": p, "m": m,
        "simulated": float(errs.mean()),
        "band": float(2 * errs.std(ddof=1) / np.sqrt(len(errs))),
        "oracle_simulated": float(np.mean(orcs)),
        "ceiling": float(Q(np.sqrt(D2) / 2)),
        "predicted_exact": plugin_error(D2, p, m),
        "predicted_first_order": plugin_error_first_order(D2, p, m),
        "n_training_sets": len(errs),
    }


def run_v10(seeds=REPORT_SEEDS, reps: int = 40, n_jobs: int = -1) -> pd.DataFrame:
    """The learning curve, predicted and simulated, over ``(design, D^2, p, m)``.

    Each cell averages ``reps`` independent training sets per seed (the
    plug-in error is a mean over training sets, and one set is a single
    draw of it) against ``N_TEST / 4`` windows per class. ``band`` is two
    standard errors of that mean.
    """
    from joblib import Parallel, delayed

    RESULTS.mkdir(parents=True, exist_ok=True)
    cells = [(design, D2, p, m) for design in ("fixed", "random") for D2 in D2_GRID
             for p in P_GRID for m in M_GRID]
    rows = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_cell)(*c, seeds, reps) for c in cells)
    df = pd.DataFrame(rows)
    # the formula is for the plug-in rule relative to ITS ceiling; at a random
    # design the oracle itself sits above Q(D/2) (the Jensen gap of V1), so the
    # comparison there is made on the excess over the simulated oracle too.
    df["excess_simulated"] = df["simulated"] - df["oracle_simulated"]
    df["excess_predicted"] = df["predicted_exact"] - df["ceiling"]
    df = v10_verdict(df)
    df.to_csv(RESULTS / "v10_learning_curve.csv", index=False)
    return df


PMN_BINS = (0.0, 0.05, 0.1, 0.2, np.inf)


def v10_verdict(df: pd.DataFrame, n: int = N_WINDOW) -> pd.DataFrame:
    """Add the random-design prediction, and the in-band flag each form earns.

    ``predicted`` is the form that applies to the cell: the exact one at a
    fixed design, :func:`plugin_error_random_design` at a random one.
    """
    df = df.copy()
    df["p_over_mn"] = df["p"] / (df["m"] * n)
    df["pmn_bin"] = pd.cut(df["p_over_mn"], PMN_BINS).astype(str)
    df["predicted_random_design"] = [
        plugin_error_random_design(D2, p, m, n, og)
        for D2, p, m, og in zip(df["D2"], df["p"], df["m"],
                                df["oracle_simulated"] - df["ceiling"], strict=True)]
    df["predicted"] = np.where(df["design"] == "fixed", df["predicted_exact"],
                               df["predicted_random_design"])
    for col in ("predicted", "predicted_exact", "predicted_first_order"):
        df[f"in_band_{col.replace('predicted_', '') if col != 'predicted' else 'applicable'}"] = (
            (df["simulated"] - df[col]).abs() <= df["band"])
    return df
