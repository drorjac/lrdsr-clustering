"""The statistics the analyses share, kept apart from any one analysis.

Eight functions, no configuration and no data: a bootstrap, three tests, a
threshold fit, an OLS with intervals, and the fold assignment. They were the
last section of a 689-line module that also held the paths, the data loading,
the law library and the estimator, which meant anything wanting a bootstrap
imported all four.

**The unit of replication is the link, not the window.** Windows inside one
link share a storm, an antenna, a calibration and a length, so a per-window
test treats tens of thousands of correlated observations as independent and
will report p = 3e-45 for a 0.01 dB difference. ``paired_over_links`` and
``grouped_folds`` exist to make the right unit the easy one.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

#: Bootstrap replicates, and the project-wide seed for them. Fixed so that a
#: confidence interval is reproducible: a CI that moves when it is recomputed
#: cannot be compared across revisions.
N_BOOTSTRAP, BOOT_SEED = 2000, 20260908

#: Folds for grouped cross-validation, held out by link.
N_FOLDS = 10


def boot_ci(stat, n, n_boot=None, seed=None, alpha=0.05):
    """Percentile bootstrap of ``stat(idx)`` over ``n`` units."""
    n_boot = N_BOOTSTRAP if n_boot is None else n_boot
    rng = np.random.default_rng(BOOT_SEED if seed is None else seed)
    vals = np.array([stat(rng.integers(0, n, n)) for _ in range(n_boot)])
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return np.nan, np.nan
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


def paired_over_links(df, a, b, n_boot=None):
    """Does ``b`` beat ``a``? Paired per link, tested ACROSS links.

    The unit of replication is the LINK. Windows within a link share one
    storm, one antenna, one calibration and one length, so a per-window test
    treats tens of thousands of correlated observations as independent and
    will call a 0.01 dB difference p = 3e-45.
    """
    d = df[b].to_numpy() - df[a].to_numpy()
    n = len(d)
    if n < 6:
        return {"comparison": f"{b} vs {a}", "n_links": n,
                "mean_d": float(d.mean()), "lo": np.nan, "hi": np.nan,
                "median_d": float(np.median(d)),
                "win_rate": float((d < 0).mean()), "p": np.nan}
    lo, hi = boot_ci(lambda i: float(d[i].mean()), n, n_boot=n_boot)
    nz = d[d != 0]
    return {"comparison": f"{b} vs {a}", "n_links": n,
            "mean_d": float(d.mean()), "lo": lo, "hi": hi,
            "median_d": float(np.median(d)),
            "win_rate": float((d < 0).mean()),
            "p": float(wilcoxon(nz, alternative="less").pvalue)
            if len(nz) >= 6 else np.nan}


def mann_whitney(x, y, alternative="two-sided"):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) < 3 or len(y) < 3:
        return {"n_x": len(x), "n_y": len(y), "median_x": np.nan,
                "median_y": np.nan, "U": np.nan, "p": np.nan}
    U, p = mannwhitneyu(x, y, alternative=alternative)
    return {"n_x": len(x), "n_y": len(y),
            "median_x": float(np.median(x)), "median_y": float(np.median(y)),
            "U": float(U), "p": float(p)}


def sign_test(x, mu=0.0):
    """Two-sided sign test that the median of ``x`` differs from ``mu``.

    Used for "does the power law fit this link" -- a sign test, not a t-test,
    because window residuals within a link are skewed and heavy-tailed.
    """
    from scipy.stats import binomtest
    x = np.asarray(x, float)
    x = x[np.isfinite(x)] - mu
    n_pos, n = int((x > 0).sum()), int((x != 0).sum())
    if n < 5:
        return {"n": n, "n_pos": n_pos, "p": np.nan}
    return {"n": n, "n_pos": n_pos,
            "p": float(binomtest(n_pos, n, 0.5).pvalue)}


def permutation_spearman(x, y, n_perm=5000, seed=None):
    """Spearman with a permutation null (shuffle y across units)."""
    rng = np.random.default_rng(BOOT_SEED if seed is None else seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 6:
        return {"rho": np.nan, "p_perm": np.nan, "n": len(x)}
    rho = spearmanr(x, y)[0]
    null = np.array([spearmanr(x, rng.permutation(y))[0] for _ in range(n_perm)])
    p = float((np.abs(null) >= abs(rho)).mean())
    return {"rho": float(rho), "p_perm": max(p, 1.0 / n_perm),
            "p_is_bound": bool(p == 0.0), "n": len(x), "n_perm": n_perm}


def logistic_threshold(x, y, n_boot=400, seed=None, weights=None):
    """Logistic fit of ``y`` on ``x``; returns the x where p = 0.5, with CI.

    ``x`` is log length throughout, so the threshold comes back in km.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    x, y = np.asarray(x, float).reshape(-1, 1), np.asarray(y, int)
    if len(np.unique(y)) < 2 or len(y) < 8:
        return {"threshold_km": np.nan, "auc": np.nan, "lo": np.nan,
                "hi": np.nan, "slope": np.nan, "n": len(y)}

    def fit(idx):
        m = LogisticRegression(max_iter=400)
        w = None if weights is None else np.asarray(weights, float)[idx]
        m.fit(x[idx], y[idx], sample_weight=w)
        c, b0 = float(m.coef_[0][0]), float(m.intercept_[0])
        return (np.exp(-b0 / c) if abs(c) > 1e-9 else np.nan), c, m

    thr, slope, model = fit(np.arange(len(y)))
    auc = float(roc_auc_score(y, model.predict_proba(x)[:, 1]))
    # A near-zero logistic coefficient means length carries no information,
    # and exp(-b0/c) then explodes: a "threshold" of 0.000 km with a 10^55
    # upper bound is not a small threshold, it is no threshold. Report it as
    # undefined rather than as a number, and let the AUC say why.
    if abs(slope) < 1e-6 or not (np.exp(x.min()) * 1e-3 < thr
                                 < np.exp(x.max()) * 1e3):
        return {"threshold_km": np.nan, "auc": auc, "slope": slope,
                "lo": np.nan, "hi": np.nan, "n": len(y),
                "note": "length carries no information; threshold undefined"}
    rng = np.random.default_rng(BOOT_SEED if seed is None else seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        try:
            boots.append(fit(i)[0])
        except Exception:
            continue
    boots = np.array([b for b in boots if np.isfinite(b)])
    # Drop bootstrap replicates that blew up the same way.
    if len(boots):
        keep = (boots > np.exp(x.min()) * 1e-3) & (boots < np.exp(x.max()) * 1e3)
        boots = boots[keep]
    return {"threshold_km": float(thr), "auc": auc, "slope": slope,
            "lo": float(np.percentile(boots, 2.5)) if len(boots) > 20 else np.nan,
            "hi": float(np.percentile(boots, 97.5)) if len(boots) > 20 else np.nan,
            "n": len(y)}


def ols_ci(x, y, n_boot=None, seed=None):
    """Slope and intercept with a bootstrap CI over points."""
    n_boot = N_BOOTSTRAP if n_boot is None else n_boot
    x, y = np.asarray(x, float), np.asarray(y, float)

    def fit(i):
        X = np.column_stack([np.ones(len(i)), x[i]])
        return np.linalg.lstsq(X, y[i], rcond=None)[0]

    b0, b1 = fit(np.arange(len(x)))
    rng = np.random.default_rng(BOOT_SEED if seed is None else seed)
    B = np.array([fit(rng.integers(0, len(x), len(x))) for _ in range(n_boot)])
    pred = b0 + b1 * x
    return {"slope": float(b1), "intercept": float(b0),
            "slope_lo": float(np.percentile(B[:, 1], 2.5)),
            "slope_hi": float(np.percentile(B[:, 1], 97.5)),
            "int_lo": float(np.percentile(B[:, 0], 2.5)),
            "int_hi": float(np.percentile(B[:, 0], 97.5)),
            "r2": float(1 - np.sum((y - pred) ** 2) /
                        max(np.sum((y - y.mean()) ** 2), 1e-12)), "n": len(x)}


def grouped_folds(link, n_folds=None, seed=None):
    """Fold index per row, with whole links held out together."""
    n_folds = N_FOLDS if n_folds is None else n_folds
    uniq = np.array(sorted(set(link)))
    rng = np.random.default_rng(BOOT_SEED if seed is None else seed)
    m = dict(zip(rng.permutation(uniq), np.arange(len(uniq)) % n_folds,
                 strict=True))
    return np.array([m[link_id] for link_id in link])
