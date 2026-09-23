"""the estimator block -- the generic symbolic benchmark: invariants of the construction.

Mathematical invariants, not smoke tests, all on tiny configs.
"""
import numpy as np
import pytest

from experiments.estimator import benchmark as G


def test_windows_have_identical_x_marginals():
    """Both regimes of every pair must draw x from the identical marginal, so
    the covariate carries no regime signal."""
    for pair in G.PAIRS:
        _, _, z, x, _ = G._make_windows(pair, sigma=0.3, m=4000, seed=7)
        x0 = x[z == 0].ravel()
        x1 = x[z == 1].ravel()
        assert x0.mean() == pytest.approx(x1.mean(), abs=0.05)
        assert x0.std() == pytest.approx(x1.std(), abs=0.05)
        assert x0.min() == pytest.approx(x1.min(), abs=0.05)


def test_planted_geometry_is_near_uninformative():
    """The planted Zg must not separate the regimes on its own."""
    from lrdsr.core.baselines import geometry_baselines
    _, _, z, _, Zg = G._make_windows(G.PAIR_BY_NAME["polynomial"],
                                     sigma=0.3, m=400, seed=7)
    errs = [G._matched_error(z, lab)
            for lab in geometry_baselines(Zg, 2, seed=7).values()]
    assert min(errs) > 0.25          # no baseline gets close to perfect


def test_oracle_beats_or_matches_lrdsr_on_average():
    """The oracle sees the true laws, so it cannot do worse than LR-DSR by more
    than noise -- a sanity check on the matched-error plumbing."""
    pair = G.PAIR_BY_NAME["linear"]
    ms = G._pair_gap_ms(pair)
    sigma = float(np.sqrt(ms / 0.25))
    _X_seq, y, z, x, _Zg = G._make_windows(pair, sigma, m=120, seed=11)
    oracle = G._oracle_labels(x, y, [pair.f0, pair.f1])
    err = G._matched_error(z, oracle)
    assert 0.0 <= err <= 0.15        # rho = 0.25, n = 64 -> ~2-4% Bayes error


def test_matched_error_is_permutation_invariant():
    true = np.array([0, 0, 1, 1, 1])
    assert G._matched_error(true, np.array([1, 1, 0, 0, 0])) == 0.0
    assert G._matched_error(true, true) == 0.0
