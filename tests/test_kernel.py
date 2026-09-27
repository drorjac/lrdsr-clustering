"""The kernel bases: what they span, and the label-free rank rule's arithmetic."""
from __future__ import annotations

import numpy as np
import pytest

from lrdsr.core.kernel import (
    CosineBasis,
    FourierBasis,
    LegendreBasis,
    NystromBasis,
    RandomFourierBasis,
    loo_error,
    make_basis,
    select_rank,
    window_kernel,
)
from lrdsr.core.mechanism_space import mechanism_features


@pytest.mark.parametrize("basis,rank", [
    (FourierBasis(3), 7), (CosineBasis(6), 6), (LegendreBasis(5), 5),
    (NystromBasis(8), 9), (RandomFourierBasis(10), 11)])
def test_shapes_and_rank(basis, rank):
    X = np.random.default_rng(0).uniform(-2, 2, (300, 1))
    B = basis(X)
    assert B.shape == (300, rank) and basis.rank == rank
    assert np.all(np.isfinite(B))


def test_cosine_is_orthogonal_on_its_midpoint_grid():
    n = 48
    x = ((np.arange(n) + 0.5) / n)[:, None]
    B = CosineBasis(10, lo=0.0, hi=1.0)(x)
    G = B.T @ B
    np.testing.assert_allclose(G - np.diag(np.diag(G)), 0.0, atol=1e-10)


def test_nystrom_is_fitted_once_and_reused():
    rng = np.random.default_rng(1)
    b = NystromBasis(6).fit(rng.uniform(0, 1, (200, 1)))
    c0 = b.centers_.copy()
    b(rng.uniform(5, 6, (50, 1)))              # new inputs do not refit
    np.testing.assert_array_equal(b.centers_, c0)


def test_nystrom_spans_a_frequency_the_library_cannot():
    """The repair the kernel block is about: sin(4x) - sin(4.6x) on [-3, 3]."""
    x = np.linspace(-3, 3, 2000)[:, None]
    g = np.sin(4 * x[:, 0]) - np.sin(4.6 * x[:, 0])
    B = NystromBasis(24).fit(x)(x)
    coef, *_ = np.linalg.lstsq(B, g, rcond=None)
    assert np.mean((g - B @ coef) ** 2) / np.mean(g ** 2) < 1e-3


def test_loo_error_matches_brute_force():
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 1, (1, 20, 1))
    y = np.sin(3 * X[0, :, 0])[None] + rng.normal(0, 0.1, (1, 20))
    b = CosineBasis(5, lo=0.0, hi=1.0)
    fast = loo_error(X, y, b)
    B = b(X[0])
    brute = []
    for i in range(20):
        keep = np.arange(20) != i
        coef, *_ = np.linalg.lstsq(B[keep], y[0, keep], rcond=None)
        brute.append((y[0, i] - B[i] @ coef) ** 2)
    assert fast == pytest.approx(np.mean(brute), rel=1e-8)


def test_select_rank_prefers_enough_but_not_everything():
    rng = np.random.default_rng(3)
    x = np.broadcast_to(((np.arange(64) + 0.5) / 64)[None, :, None], (40, 64, 1))
    f = np.cos(3 * np.pi * x[..., 0])            # needs cosine term k = 3
    y = f + rng.normal(0, 0.3, f.shape)
    size, table = select_rank(x, y, "cosine", (2, 4, 8, 32))
    assert size in (4, 8) and len(table) == 4


def test_make_basis_rejects_unknown():
    with pytest.raises(KeyError):
        make_basis("wavelet", 4)


def test_mechanism_space_accepts_a_kernel_basis():
    rng = np.random.default_rng(4)
    X = rng.uniform(-3, 3, (60, 32, 1))
    z = rng.integers(0, 2, 60)
    y = np.where(z[:, None] == 1, np.sin(4 * X[..., 0]), np.sin(4.6 * X[..., 0]))
    y = y + rng.normal(0, 0.1, y.shape)
    b = NystromBasis(16).fit(X.reshape(-1, 1))
    S = mechanism_features(X, y, basis=b)
    assert S.shape[0] == 60
    K = window_kernel(S)
    np.testing.assert_allclose(K, K.T)
    R = window_kernel(S, "rbf")
    assert np.all((R > 0) & (R <= 1 + 1e-12))
