"""Optional block 5: after the loop, look for each group's law more deeply.

Block 1 of the loop chooses each law from a fixed vocabulary, because it runs
many times (every round, every group, every cross-fitting fold) and must be
fast. An open-ended search -- PySR, which builds formulas from operators and
fits the constants inside them -- can find laws that vocabulary cannot
write, but takes minutes per law. So it runs **once per final group**, after
the loop has settled the groups:

1. for each group, the engine searches a law on that group's windows;
2. the found law **replaces the loop's law only if it explains the group's
   windows better on the objective's own cost** (mean Huber of the residual
   at the shared noise scale), judged the way the loop judges -- never on a
   window the law was fitted on: the group's windows are split in half, the
   engine and a fresh copy of the loop's law both fit on one half, and they
   are compared on the other. It is one more "fit laws, labels fixed" step
   of the same descent, with a larger vocabulary, judged on the loop's own
   (cross-fitted) terms -- not a separate criterion;
3. optionally, one more "assign" step: every window to its cheapest law.

Not the default and not used by any committed result. The engine is
pluggable: :func:`pysr_holdout_engine` is the one provided (needs
``pip install "pysr<2"``, which installs Julia on first use).
"""
from __future__ import annotations

import copy
import tempfile
from dataclasses import dataclass, field

import numpy as np

from .losses import HUBER_DELTA, aggregate_window_residual, noise_scale

#: PySR settings used by :func:`pysr_holdout_engine` unless overridden.
PYSR_DEFAULTS = {
    "niterations": 40, "binary_operators": ["+", "-", "*", "/"],
    "unary_operators": ["sin", "cos", "exp"], "maxsize": 25,
    "deterministic": True, "parallelism": "serial", "progress": False, "verbosity": 0,
}


class _FoundLaw:
    """A law returned by an engine, with the interface the loop's laws have."""

    def __init__(self, predict, expression: str, complexity: float = np.nan):
        self._predict, self._expression, self._complexity = predict, expression, complexity

    def predict(self, X):
        with np.errstate(all="ignore"):
            out = np.asarray(self._predict(np.asarray(X, dtype=float)), dtype=float).reshape(-1)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def expression(self) -> str:
        return self._expression

    def complexity(self) -> float:
        return self._complexity


def pysr_holdout_engine(max_samples: int = 1000, **pysr_options):
    """PySR, with the formula chosen on held-out windows.

    PySR returns a list of candidate formulas of growing size. Its own
    default pick ("best") favours the simplest one that explains most of the
    variance, which drops small details that separate regimes (a decay
    envelope); picking the most accurate returns overfitted formulas. Here
    PySR is fitted on half of the group's windows and the formula kept is the
    **simplest within one standard error of the best error on the other
    half** (the 1-SE rule). Returns ``engine(Xw, yw, seed) -> (predict,
    expression, complexity)``.
    """
    options = {**PYSR_DEFAULTS, **pysr_options}

    def engine(Xw: np.ndarray, yw: np.ndarray, seed: int):
        from pysr import PySRRegressor

        rng = np.random.default_rng(seed)
        order = rng.permutation(len(yw))
        tr, va = order[: len(order) // 2], order[len(order) // 2:]
        d = Xw.shape[-1]
        Xtr, ytr = Xw[tr].reshape(-1, d), yw[tr].reshape(-1)
        Xva, yva = Xw[va].reshape(-1, d), yw[va].reshape(-1)
        idx = rng.choice(len(ytr), min(max_samples, len(ytr)), replace=False)
        with tempfile.TemporaryDirectory() as tmp:
            m = PySRRegressor(random_state=seed, output_directory=tmp, **options)
            m.fit(Xtr[idx], ytr[idx])
            eqs = m.equations_.reset_index(drop=True)
            scores = []
            for i in range(len(eqs)):
                with np.errstate(all="ignore"):
                    r2 = (yva - m.predict(Xva, index=i)) ** 2
                ok = np.all(np.isfinite(r2))
                scores.append((float(np.mean(r2)) if ok else np.inf,
                               float(np.std(r2) / np.sqrt(len(r2))) if ok else np.inf))
            best = int(np.argmin([s[0] for s in scores]))
            limit = scores[best][0] + scores[best][1]
            pick = min(i for i in range(len(eqs)) if scores[i][0] <= limit)  # ordered by size
            m.predict(Xva[:2], index=pick)          # compile while the temp dir exists

        def predict(X, m=m, i=pick):
            return m.predict(np.asarray(X, dtype=float).reshape(-1, d), index=i)

        return predict, str(eqs.loc[pick, "sympy_format"]), float(eqs.loc[pick, "complexity"])

    return engine


@dataclass
class RefineResult:
    labels: np.ndarray                 # after the optional reassignment
    models: list                       # the law kept per group (refined or the loop's)
    replaced: list[bool]               # per group: did the engine's law win?
    labels_before: np.ndarray
    cost_before: list[float] = field(default_factory=list)   # per group, the loop's law
    cost_found: list[float] = field(default_factory=list)    # per group, the engine's law
    expressions_found: list[str] = field(default_factory=list)
    found_models: list = field(default_factory=list)          # per group, the engine's law (or None)


def _window_costs(model, X_seq, y_seq, windows, scale, delta):
    return np.array([aggregate_window_residual(y_seq[w], model.predict(X_seq[w]),
                                               robust_delta=delta, scale=scale) for w in windows])


def refine_laws(result, X_seq, y_seq, engine=None, reassign: bool = True, seed: int = 0,
                robust_delta: float = HUBER_DELTA, min_windows: int = 4) -> RefineResult:
    """Block 5 (optional): search each final group's law with a heavier engine.

    ``result`` is what ``GroupedDCSR.fit`` returned. ``engine(Xw, yw, seed)``
    returns ``(predict, expression, complexity)`` for one group's windows;
    the default is :func:`pysr_holdout_engine`. Each group's windows are split
    in half (by ``seed``); the engine and a copy of the loop's law are fitted
    on one half and compared on the other by mean Huber cost at the shared
    noise scale. The found law replaces the loop's only if it is cheaper
    there -- a law is never judged on windows it was fitted on, as in the
    loop's cross-fitting. With ``reassign`` every window then moves to its
    cheapest kept law (the loop's "assign" step, once). ``cost_before`` /
    ``cost_found`` are those held-out costs.
    """
    X_seq, y_seq = np.asarray(X_seq, dtype=float), np.asarray(y_seq, dtype=float)
    engine = engine or pysr_holdout_engine()
    labels = np.asarray(result.labels).copy()
    K = len(result.models)
    resid = np.concatenate([y_seq[w] - result.models[labels[w]].predict(X_seq[w])
                            for w in range(len(y_seq))])
    scale = noise_scale(resid)

    d = X_seq.shape[-1]
    rows = []                          # per group: (kept, replaced?, cost loop, cost found, expr, found)
    for k in range(K):
        windows = np.flatnonzero(labels == k)
        loop_law = result.models[k]
        if len(windows) < 2 * min_windows:
            rows.append((loop_law, False, np.nan, np.nan, "", None))
            continue
        order = np.random.default_rng(seed + k).permutation(windows)
        fit_half, judge_half = order[: len(order) // 2], order[len(order) // 2:]
        loop_copy = copy.deepcopy(loop_law)
        loop_copy.fit(X_seq[fit_half].reshape(-1, d), y_seq[fit_half].reshape(-1))
        c_loop = float(_window_costs(loop_copy, X_seq, y_seq, judge_half, scale, robust_delta).mean())
        predict, expr, cx = engine(X_seq[fit_half], y_seq[fit_half], seed + k)
        law = _FoundLaw(predict, expr, cx)
        c_new = float(_window_costs(law, X_seq, y_seq, judge_half, scale, robust_delta).mean())
        win = c_new < c_loop
        rows.append((law if win else loop_law, bool(win), c_loop, c_new, expr, law))
    models, replaced, before, found, exprs, found_models = (list(c) for c in zip(*rows, strict=True))

    new_labels = labels
    if reassign and any(replaced):
        cost = np.column_stack([_window_costs(m, X_seq, y_seq, range(len(y_seq)), scale, robust_delta)
                                for m in models])
        new_labels = cost.argmin(axis=1)
    return RefineResult(labels=new_labels, models=models, replaced=replaced, labels_before=labels,
                        cost_before=before, cost_found=found, expressions_found=exprs,
                        found_models=found_models)
