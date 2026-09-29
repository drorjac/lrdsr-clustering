"""Optional block 5: a found law is kept only when it lowers the objective's cost."""
import numpy as np

from lrdsr import GroupedDCSR
from lrdsr.core.evaluation import aligned_accuracy
from lrdsr.core.refine import refine_laws

LAWS = (lambda x: np.exp(-0.4 * x) * np.sin(3 * x), lambda x: np.exp(-0.1 * x) * np.sin(3 * x))


def _windows(seed=0, W=80, n=40, sigma=0.15):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 4, (W, n, 1))
    z = rng.integers(0, 2, W)
    y = np.stack([LAWS[k](X[w, :, 0]) for w, k in enumerate(z)]) + rng.normal(0, sigma, (W, n))
    return X, y, z


def _fit(X, y):
    return GroupedDCSR(n_clusters=2, alpha_geom=0.0, init="mechanism", random_state=0).fit(
        X, y, np.zeros((len(y), 1)), feature_names=["x"])


def _engine_returning(fn_by_group):
    """A stand-in engine: returns a fixed law, chosen by which true law dominates the group."""
    def engine(Xw, yw, seed):
        errs = [np.mean((yw - f(Xw[..., 0])) ** 2) for f in LAWS]
        f = fn_by_group(int(np.argmin(errs)))
        return (lambda X, f=f: f(np.asarray(X)[:, 0])), "stand-in", 3.0
    return engine


def test_a_better_law_replaces_the_loops_and_labels_do_not_get_worse():
    X, y, z = _windows()
    res = _fit(X, y)
    out = refine_laws(res, X, y, engine=_engine_returning(lambda k: LAWS[k]))
    assert all(out.replaced)
    assert all(f < b for f, b in zip(out.cost_found, out.cost_before, strict=True))
    assert aligned_accuracy(z, out.labels) >= aligned_accuracy(z, res.labels) - 1e-9


def test_a_worse_law_is_rejected_and_nothing_moves():
    X, y, _ = _windows(seed=1)
    res = _fit(X, y)
    out = refine_laws(res, X, y, engine=_engine_returning(lambda k: (lambda x: 0 * x)))
    assert not any(out.replaced)
    assert out.models == list(res.models)
    assert np.array_equal(out.labels, res.labels)
