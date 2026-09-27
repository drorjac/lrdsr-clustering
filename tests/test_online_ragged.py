"""The online clusterer with ragged windows, a nuisance and a kernel basis."""
from __future__ import annotations

import numpy as np

from lrdsr.core.kernel import NystromBasis
from lrdsr.core.online import OnlineLRDSR


def _stream(W=200, n=32, seed=0, level=0.0, ragged=False):
    rng = np.random.default_rng(seed)
    X, Y, Z = [], [], []
    for _ in range(W):
        k = int(rng.integers(0, 2))
        m = int(rng.integers(12, n + 1)) if ragged else n
        x = np.sort(rng.uniform(-3, 3, m))
        f = np.sin(4 * x) if k == 0 else np.sin(4.6 * x)
        X.append(x[:, None])
        Y.append(f + rng.normal(0, 0.2, m) + rng.normal(0, level))
        Z.append(k)
    return X, Y, np.asarray(Z)


def test_ragged_stream_keeps_its_laws():
    X, Y, z = _stream(ragged=True)
    on = OnlineLRDSR(basis=NystromBasis(16), novelty_alpha=0.0)
    on.warm_start(X[:60], Y[:60], n_clusters=2, random_state=0)
    tl = on.fit_stream(X[60:], Y[60:], true_labels_for_eval=z[60:])
    acc = np.mean(tl["label"].to_numpy() == tl["truth"].to_numpy())
    assert max(acc, 1 - acc) > 0.95


def test_kernel_basis_is_fitted_on_the_warm_start():
    X, Y, _ = _stream(W=40)
    b = NystromBasis(12)
    OnlineLRDSR(basis=b, novelty_alpha=0.0).warm_start(X, Y, n_clusters=2)
    assert b.fitted and b.centers_.min() >= -3.0 / X[0].std() - 1


def test_a_level_shift_is_not_a_new_regime():
    X, Y, _ = _stream(W=160, level=2.0, seed=1)
    on = OnlineLRDSR(basis=NystromBasis(16), nuisance="intercept", novelty_alpha=1e-6)
    on.warm_start(X[:60], Y[:60], n_clusters=2, random_state=0)
    tl = on.fit_stream(X[60:], Y[60:])
    assert tl["spawned"].notna().sum() == 0 and on.n_clusters == 2


def test_without_the_nuisance_the_same_stream_breaks():
    """Unprofiled levels do not trigger births -- they inflate the noise the
    laws measure, which hides the regimes instead: accuracy collapses."""
    acc = {}
    for nu in ("intercept", None):
        X, Y, z = _stream(W=160, level=2.0, seed=1)
        on = OnlineLRDSR(basis=NystromBasis(16), nuisance=nu, novelty_alpha=1e-6)
        on.warm_start(X[:60], Y[:60], n_clusters=2, random_state=0)
        tl = on.fit_stream(X[60:], Y[60:], true_labels_for_eval=z[60:])
        a = np.mean(tl["label"].to_numpy() == tl["truth"].to_numpy())
        acc[nu] = max(a, 1 - a)
    assert acc["intercept"] > 0.95 and acc[None] < acc["intercept"] - 0.2
