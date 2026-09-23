import numpy as np
import pytest

from lrdsr.core.backends import FastSymbolicRegressor, make_symbolic_regressor
from lrdsr.core.evaluation import aligned_accuracy, clustering_metrics, nmse
from lrdsr.core.model import GroupedDCSR, RowDCSR
from tests.function_mixture import simulate_function_mixture


def test_fast_backend_recovers_simple_law():
    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, 400)
    y = 0.8 * x**2 + 1.5 * x + 0.5 + rng.normal(0, 0.05, 400)
    model = FastSymbolicRegressor(feature_names=["x"])
    model.fit(x[:, None], y)
    assert nmse(y, model.predict(x[:, None])) < 0.05
    assert model.complexity() >= 2
    assert isinstance(model.expression(), str)


def test_backend_factory():
    assert isinstance(make_symbolic_regressor("fast"), FastSymbolicRegressor)
    with pytest.raises(ValueError):
        make_symbolic_regressor("nope")
    # Optional SR engines are recognised names: absent installs raise ImportError,
    # not the "unknown backend" ValueError.
    for name in ("pysr", "physo", "dso"):
        try:
            make_symbolic_regressor(name)
        except ImportError:
            pass
        except ValueError as e:
            raise AssertionError(
                f"{name!r} should be a recognised backend name") from e


def test_aligned_accuracy_permutation_invariant():
    true = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([2, 2, 0, 0, 1, 1])
    assert aligned_accuracy(true, pred) == 1.0


def test_row_dcsr_beats_kmeans_on_overlapping_mixture():
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    data = simulate_function_mixture(n_per_regime=150, geometry_overlap=1.2, seed=7)
    km = KMeans(n_clusters=3, n_init=10, random_state=7).fit_predict(
        StandardScaler().fit_transform(data.Z)
    )
    result = RowDCSR(
        n_clusters=3, alpha_geom=0.25, backend="fast",
        backend_kwargs={"max_terms": 4}, random_state=7, max_iter=6,
    ).fit(data.X, data.y, data.Z, feature_names=["x"], true_labels_for_eval=data.labels)

    ari_km = clustering_metrics(data.labels, km)["ARI"]
    ari_dcsr = clustering_metrics(data.labels, result.labels)["ARI"]
    assert ari_dcsr > ari_km
    assert not result.history.empty
    assert np.min(np.bincount(result.labels, minlength=3)) >= 3


def test_grouped_dcsr_shape_validation():
    model = GroupedDCSR(n_clusters=2)
    with pytest.raises(ValueError):
        model.fit(np.zeros((4, 5, 2)), np.zeros((3, 5)), np.zeros((4, 3)))


def test_physics_cost_hook_receives_labels_and_models():
    """The optional physics-anchor hook is called with the current state.

    The anchor itself is an ablation (it hurt ARI in the previous revision and
    is not part of the primary suite), but the hook is part of the core
    assignment loss, so its contract stays tested: it gets the current labels
    and models and returns a [W, K] cost matrix.
    """
    seen = {}

    def builder(labels, models=None):
        seen["labels"] = np.asarray(labels).copy()
        seen["n_models"] = len(models) if models is not None else 0
        return np.zeros((labels.shape[0], 3))

    data = simulate_function_mixture(n_per_regime=40, seed=7)
    X_seq = data.X[:, None, :]
    y_seq = data.y[:, None]
    GroupedDCSR(n_clusters=3, alpha_geom=0.5, lambda_phys=0.2, max_iter=2,
                random_state=7).fit(
        X_seq, y_seq, data.Z, feature_names=["x"], physics_cost_builder=builder)
    assert seen["labels"].shape[0] == data.y.shape[0]
    assert seen["n_models"] == 3


def test_grouped_dcsr_validates_init_and_geom_metric():
    with pytest.raises(ValueError):
        GroupedDCSR(n_clusters=2, init="bogus")
    with pytest.raises(ValueError):
        GroupedDCSR(n_clusters=2, geom_metric="bogus")


def test_grouped_dcsr_gmm_init_mahalanobis_runs():
    data = simulate_function_mixture(n_per_regime=60, seed=3)
    X_seq = data.X[:, None, :]
    y_seq = data.y[:, None]
    model = GroupedDCSR(n_clusters=3, alpha_geom=0.5, backend="fast",
                        backend_kwargs={"max_terms": 3}, max_iter=4,
                        random_state=3, init="gmm", geom_metric="mahalanobis")
    result = model.fit(X_seq, y_seq, data.Z, feature_names=["x"])
    assert result.labels.shape == (data.y.size,)
    assert set(result.labels) <= {0, 1, 2}


def test_superposition_regressor_recovers_mixing_weights():
    from lrdsr.core.backends import FastSymbolicRegressor, SuperpositionRegressor

    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(400, 1))
    f1 = FastSymbolicRegressor(max_terms=3).fit(X, (X[:, 0] ** 2).copy())
    f2 = FastSymbolicRegressor(max_terms=3).fit(X, np.sin(2 * X[:, 0]))
    y = 0.5 * f1.predict(X) + 1.5 * f2.predict(X) + 0.25

    sup = SuperpositionRegressor([f1, f2]).fit(X, y)
    assert np.allclose(sup.weights_, [0.5, 1.5], atol=0.05)
    assert abs(sup.intercept_ - 0.25) < 0.05
    assert sup.is_valid


def test_superposition_rejects_rescaled_single_component():
    from lrdsr.core.backends import FastSymbolicRegressor, SuperpositionRegressor

    rng = np.random.default_rng(1)
    X = rng.uniform(-2, 2, size=(300, 1))
    f1 = FastSymbolicRegressor(max_terms=3).fit(X, (X[:, 0] ** 2).copy())
    f2 = FastSymbolicRegressor(max_terms=3).fit(X, np.sin(2 * X[:, 0]))
    # Pure rescale of one component: not a genuine superposition.
    sup = SuperpositionRegressor([f1, f2]).fit(X, 1.4 * f1.predict(X))
    assert not sup.is_valid


def test_superposition_weights_are_non_negative_and_bounded():
    from lrdsr.core.backends import FastSymbolicRegressor, SuperpositionRegressor

    rng = np.random.default_rng(2)
    X = rng.uniform(-2, 2, size=(300, 1))
    f1 = FastSymbolicRegressor(max_terms=3).fit(X, (X[:, 0] ** 2).copy())
    f2 = FastSymbolicRegressor(max_terms=3).fit(X, np.sin(2 * X[:, 0]))
    sup = SuperpositionRegressor([f1, f2], max_weight=3.0).fit(
        X, -5.0 * f1.predict(X) + 20.0 * f2.predict(X))
    assert np.all(sup.weights_ >= 0.0)
    assert np.all(sup.weights_ <= 3.0 + 1e-9)


def test_superposition_off_by_default_and_backward_compatible():
    data = simulate_function_mixture(n_per_regime=60, seed=3)
    X_seq, y_seq = data.X[:, None, :], data.y[:, None]
    kw = {"n_clusters": 3, "alpha_geom": 0.5, "backend": "fast",
              "backend_kwargs": {"max_terms": 3}, "max_iter": 3, "random_state": 3}
    plain = GroupedDCSR(**kw).fit(X_seq, y_seq, data.Z, feature_names=["x"])
    explicit = GroupedDCSR(**kw, superposition=False).fit(
        X_seq, y_seq, data.Z, feature_names=["x"])
    assert np.array_equal(plain.labels, explicit.labels)
    with pytest.raises(ValueError):
        GroupedDCSR(n_clusters=2, superposition_margin=-0.1)
