"""The fitted-frequency term: it finds a sine, and only when a sine is there."""
import numpy as np

from lrdsr.core.backends import SinusoidSymbolicRegressor, make_symbolic_regressor


def _data(f, n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, (n, 1))
    return x, f(x[:, 0]) + rng.normal(0, 0.5, n)


def test_recovers_a_frequency_outside_the_library():
    m = SinusoidSymbolicRegressor(feature_names=["x"]).fit(*_data(lambda x: np.sin(4.6 * x)))
    assert len(m.terms_) == 1 and m.terms_[0][0] == "sin"
    assert abs(m.terms_[0][2] - 4.6) < 0.05


def test_keeps_the_library_law_when_no_sine_is_needed():
    m = SinusoidSymbolicRegressor(feature_names=["x"]).fit(*_data(lambda x: 0.8 * x**2 + 1.5 * x))
    assert all(t[0] == "lib" for t in m.terms_)
    assert sorted(m.term_names_) == ["(x)^2", "x"]


def test_a_phase_is_a_sine_and_a_cosine_at_one_frequency():
    m = SinusoidSymbolicRegressor(feature_names=["x"]).fit(*_data(lambda x: np.sin(4 * x + 1)))
    freqs = [t[2] for t in m.terms_ if t[0] != "lib"]
    assert len(freqs) == 2 and np.allclose(freqs, 4.0, atol=0.05)


def test_prediction_matches_the_fit_and_the_factory_builds_it():
    x, y = _data(lambda x: np.sin(4 * x) + 0.5 * np.sin(6.5 * x))
    m = make_symbolic_regressor("fast_sin", feature_names=["x"]).fit(x, y)
    assert isinstance(m, SinusoidSymbolicRegressor)
    assert np.mean((m.predict(x) - y) ** 2) < 0.3
