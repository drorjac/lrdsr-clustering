"""V11: the masked-window error formula against the actual classifier."""
from __future__ import annotations

import numpy as np
import pytest

from lrdsr.core.classify import LawClassifier
from lrdsr.core.kernel import FourierBasis
from lrdsr.theory.partial import Q, decided_share, masked_error


def _setup(seed=0, n_train=400):
    rng = np.random.default_rng(seed)
    x = (2 * np.pi * np.arange(24) / 24)[:, None]
    f0 = np.sin(x[:, 0])
    f1 = np.sin(x[:, 0]) + 0.35 * np.cos(2 * x[:, 0])
    # AR(1)-correlated noise: the case V11 exists for
    idx = np.arange(24)
    Sigma = 0.3 ** 2 * 0.6 ** np.abs(idx[:, None] - idx[None, :])
    Lc = np.linalg.cholesky(Sigma)
    z = rng.integers(0, 2, n_train)
    Y = np.where(z[:, None] == 1, f1, f0) + (Lc @ rng.normal(size=(24, n_train))).T
    Y = Y + rng.normal(0, 3, (n_train, 1))                 # a level: the nuisance
    X = np.broadcast_to(x, (n_train, 24, 1)).copy()
    clf = LawClassifier(basis=FourierBasis(3), nuisance="intercept").fit(X, Y, z)
    return rng, x, clf, Lc, Sigma


def test_iid_limit_is_v1():
    g = np.array([1.0, -1.0, 2.0, -2.0])
    assert masked_error(g, np.eye(4) * 0.5 ** 2, 0.25) == pytest.approx(
        Q(np.linalg.norm(g) / (2 * 0.5)))


@pytest.mark.parametrize("H", [np.arange(24), np.arange(6, 14), np.array([0, 3, 9, 15, 21])])
def test_masked_error_matches_the_classifier(H):
    rng, x, clf, Lc, Sigma = _setup()
    F = clf.design_.B(x) @ clf.coef_.T
    g = F[:, 1] - F[:, 0]
    s2 = float(clf.sigma_[0] ** 2)
    L = float(clf.log_prior_[1] - clf.log_prior_[0])
    n = 20000
    for truth in (0, 1):
        # windows of the FITTED law, so only the noise model is under test
        Y = F[:, truth] + (Lc @ rng.normal(size=(24, n))).T + rng.normal(0, 3, (n, 1))
        pred = clf.predict([x[H]] * n, [Y[i, H] for i in range(n)])
        sim = float(np.mean(pred != truth))
        th = masked_error(g[H], Sigma[np.ix_(H, H)], s2, L, truth=truth)
        assert abs(sim - th) <= 4 * np.sqrt(max(th * (1 - th), 1e-4) / n) + 2e-3


def test_decided_share_limits():
    g = np.linspace(-1, 1, 12)
    right, wrong = decided_share(g, np.eye(12) * 1e-6, 0.01, 0.0, truth=1)
    assert right == pytest.approx(1.0) and wrong == pytest.approx(0.0, abs=1e-12)
    right, wrong = decided_share(np.zeros(12), np.eye(12), 1.0)
    assert right == 0.0 and wrong == 0.0
