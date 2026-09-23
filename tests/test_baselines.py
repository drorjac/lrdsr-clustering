import numpy as np
from sklearn.datasets import make_blobs

from lrdsr.core.baselines import fuzzy_cmeans, geometry_baselines
from lrdsr.core.evaluation import clustering_metrics

EXPECTED = {"K-means", "Fuzzy c-means", "GMM (full cov)", "Bayesian GMM (full cov)",
            "Birch", "Spectral (RBF kernel)", "Ward agglomerative"}


def test_fuzzy_cmeans_recovers_separated_blobs():
    Z, y = make_blobs(n_samples=300, centers=3, cluster_std=0.4, random_state=0)
    labels, U = fuzzy_cmeans(Z, 3, seed=0)
    assert labels.shape == (300,)
    assert U.shape == (300, 3)
    assert np.allclose(U.sum(axis=1), 1.0)
    assert clustering_metrics(y, labels)["ARI"] > 0.95


def test_geometry_baselines_complete_deterministic_and_sane():
    Z, y = make_blobs(n_samples=240, centers=3, cluster_std=0.5, random_state=1)
    out1 = geometry_baselines(Z, 3, seed=1)
    out2 = geometry_baselines(Z, 3, seed=1)
    assert set(out1) == EXPECTED
    for name, labels in out1.items():
        assert labels.shape == (240,), name
        assert set(np.unique(labels)) <= set(range(3)), name
        assert np.array_equal(labels, out2[name]), f"{name} not deterministic"
        assert clustering_metrics(y, labels)["ARI"] > 0.9, name
