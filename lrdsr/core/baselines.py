"""Unsupervised clustering baselines for fair comparison with LR-DSR.

Fairness contract: every baseline receives exactly the same standardized
geometry/feature matrix Z that LR-DSR's geometric stage uses, the same
number of clusters, and the same random seed. None of them sees symbolic
residuals, physics scores, or true labels. K-means doubles as LR-DSR's
initializer, so "LR-DSR vs K-means" isolates the contribution of
mechanism-aware reassignment specifically.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering, Birch, KMeans, SpectralClustering
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import StandardScaler


def fuzzy_cmeans(
    Z: np.ndarray,
    n_clusters: int,
    m: float = 2.0,
    max_iter: int = 300,
    tol: float = 1e-5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuzzy c-means (Bezdek). Returns (hard labels [W], memberships [W, K]).

    Standard alternating updates of centers and memberships with fuzzifier
    ``m``; hard labels are the argmax memberships.
    """
    if m <= 1.0:
        raise ValueError("fuzzifier m must be > 1")
    Z = np.asarray(Z, dtype=float)
    rng = np.random.default_rng(seed)
    U = rng.dirichlet(np.ones(n_clusters), size=Z.shape[0])  # [W, K]

    for _ in range(max_iter):
        Um = U**m
        centers = (Um.T @ Z) / (Um.sum(axis=0)[:, None] + 1e-12)
        d2 = ((Z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1) + 1e-12
        inv = d2 ** (-1.0 / (m - 1.0))
        U_new = inv / inv.sum(axis=1, keepdims=True)
        if np.abs(U_new - U).max() < tol:
            U = U_new
            break
        U = U_new

    return U.argmax(axis=1), U


def geometry_baselines(
    Z: np.ndarray,
    n_clusters: int,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Run the standard unsupervised baselines on one feature matrix.

    Returns {name: hard labels [W]}. Z is standardized internally so every
    method sees the identical representation.
    """
    Zs = StandardScaler().fit_transform(np.asarray(Z, dtype=float))
    labels = {}
    labels["K-means"] = KMeans(
        n_clusters=n_clusters, n_init=30, random_state=seed
    ).fit_predict(Zs)
    labels["Fuzzy c-means"], _ = fuzzy_cmeans(Zs, n_clusters, seed=seed)
    labels["GMM (full cov)"] = GaussianMixture(
        n_components=n_clusters, covariance_type="full", n_init=5, random_state=seed
    ).fit_predict(Zs)
    labels["Bayesian GMM (full cov)"] = BayesianGaussianMixture(
        n_components=n_clusters, covariance_type="full", n_init=3,
        max_iter=300, random_state=seed,
    ).fit_predict(Zs)
    labels["Birch"] = Birch(n_clusters=n_clusters).fit_predict(Zs)
    labels["Spectral (RBF kernel)"] = SpectralClustering(
        n_clusters=n_clusters, affinity="rbf", assign_labels="kmeans",
        n_init=30, random_state=seed,
    ).fit_predict(Zs)
    labels["Ward agglomerative"] = AgglomerativeClustering(
        n_clusters=n_clusters, linkage="ward"
    ).fit_predict(Zs)
    return labels
