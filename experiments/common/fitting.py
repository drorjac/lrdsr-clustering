"""One LR-DSR fit with the project defaults, and the features every method sees.

Shared by every block that fits the model, so all of them
run the identical estimator.
"""
from __future__ import annotations

import numpy as np

from lrdsr.core.model import GroupedDCSR


def window_features(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Mechanism-agnostic window summaries: the geometry every method sees.

    Deliberately plain -- no residuals against any candidate law -- so the
    geometric baselines are honest and the comparison is about the equation
    term, not about who was handed a better representation.
    """
    x = X[:, :, 0]
    dy = np.diff(y, axis=1) if y.shape[1] > 1 else np.zeros_like(y[:, :1])
    Z = np.column_stack([
        y.mean(axis=1), y.std(axis=1), y.min(axis=1), y.max(axis=1),
        np.abs(dy).mean(axis=1), (x * y).mean(axis=1),
        np.percentile(y, 90, axis=1) - np.percentile(y, 10, axis=1),
    ])
    names = ["y_mean", "y_std", "y_min", "y_max", "y_absdiff", "xy_mean",
             "y_p90_p10"]
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0), names


#: The initialiser LR-DSR uses everywhere in the paper. ``"mechanism"``
#: clusters the windows by their whitened law coefficients
#: (``lrdsr.core.mechanism_space``); ``"bgmm"`` is the data-space initialiser
#: it replaced, kept reachable so the change is an ablation and not an
#: assertion.
INIT = "mechanism"


def fit_lrdsr(Z, names, X_seq, y_seq, x_names, K, seed, alpha_geom=0.25,
              init=INIT, mechanism_basis=None, mechanism_nuisance=None,
              mechanism_groups=None):
    """One LR-DSR fit: fast symbolic backend, cross-fitted window scoring,
    a shared (global) residual scale, initialised in mechanism space.

    ``mechanism_basis`` / ``mechanism_nuisance`` override the default library
    for the initial partition, e.g. a domain basis and a per-window nuisance
    term that should never reach the statistic.
    """
    model = GroupedDCSR(
        n_clusters=K, alpha_geom=alpha_geom, beta_complexity=0.002,
        max_iter=10, tol=0.01, backend="fast",
        backend_kwargs={"max_terms": 5}, random_state=seed,
        min_windows_per_cluster=4, init=init, geom_metric="mahalanobis",
        score_mode="cross_fit", residual_scale="global",
        mechanism_basis=mechanism_basis, mechanism_nuisance=mechanism_nuisance,
        mechanism_groups=mechanism_groups,
    )
    return model.fit(X_seq, y_seq, Z, feature_names=list(x_names))
