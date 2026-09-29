from __future__ import annotations

from contextlib import contextmanager as contextlib_contextmanager
from dataclasses import dataclass
from typing import ClassVar

import numpy as np


class SymbolicRegressorBase:
    """The interface a mechanism must satisfy to be a regime's law.

    This is the project's extension point: anything implementing these four
    methods can be the per-regime model, so swapping the symbolic engine —
    or supplying a fixed parametric law with nothing fitted — needs no change
    to the estimator.

    `predict` carries the weight. The loop scores every window under every
    regime's law, so it is called once per (window, regime) per iteration,
    and a slow implementation shows up immediately.

    `complexity` feeds the `beta_complexity` term of the assignment cost, so
    a regressor that under-reports it will be preferred for the wrong reason.
    The default of 1.0 is deliberately flat: a subclass that does not model
    complexity should not appear cheaper than one that does.
    """

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit one law to the windows currently assigned to this regime."""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict `y` for a design matrix. Called once per window per regime."""
        raise NotImplementedError

    def expression(self) -> str:
        """The fitted law, human-readable. This is what the method is FOR."""
        raise NotImplementedError

    def complexity(self) -> float:
        """A scalar penalty for the expression's size, for the cost's `beta` term."""
        return 1.0


@dataclass
class _LibraryTerm:
    name: str
    values: np.ndarray


class FastSymbolicRegressor(SymbolicRegressorBase):
    """Small deterministic symbolic baseline.

    This is not intended to compete with a full SR engine. It builds a library
    of interpretable nonlinear terms and uses greedy forward BIC selection.
    The common interface lets the DCSR algorithm be tested without a heavy
    external symbolic-regression dependency.
    """

    def __init__(
        self,
        max_terms: int = 5,
        include_trig: bool = True,
        include_log: bool = True,
        include_interactions: bool = True,
        min_bic_improvement: float = 1e-6,
        feature_names: list[str] | None = None,
    ):
        self.max_terms = max_terms
        self.include_trig = include_trig
        self.include_log = include_log
        self.include_interactions = include_interactions
        self.min_bic_improvement = min_bic_improvement
        self.feature_names = feature_names

    @staticmethod
    def _safe_log_abs(x: np.ndarray) -> np.ndarray:
        return np.log1p(np.abs(x))

    def _make_library(self, X: np.ndarray) -> list[_LibraryTerm]:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        names = self.feature_names or [f"x{j}" for j in range(X.shape[1])]
        terms: list[_LibraryTerm] = []

        for j, name in enumerate(names):
            x = X[:, j]
            terms.extend([
                _LibraryTerm(name, x),
                _LibraryTerm(f"({name})^2", x**2),
                _LibraryTerm(f"({name})^3", x**3),
            ])
            if self.include_log:
                terms.append(_LibraryTerm(f"log1p(abs({name}))", self._safe_log_abs(x)))
            if self.include_trig:
                terms.extend([
                    _LibraryTerm(f"sin({name})", np.sin(x)),
                    _LibraryTerm(f"cos({name})", np.cos(x)),
                    _LibraryTerm(f"sin(2*{name})", np.sin(2.0 * x)),
                ])

        if self.include_interactions and X.shape[1] > 1:
            for i in range(X.shape[1]):
                for j in range(i + 1, X.shape[1]):
                    terms.append(
                        _LibraryTerm(
                            f"({names[i]})*({names[j]})",
                            X[:, i] * X[:, j],
                        )
                    )
        return terms

    @staticmethod
    def _bic(y: np.ndarray, pred: np.ndarray, p: int) -> float:
        n = len(y)
        rss = float(np.sum((y - pred) ** 2)) + 1e-12
        return n * np.log(rss / max(n, 1)) + p * np.log(max(n, 2))

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        lib = self._make_library(X)
        n = len(y)

        intercept = np.ones((n, 1))
        coef0, *_ = np.linalg.lstsq(intercept, y, rcond=None)
        best_pred = intercept @ coef0
        best_bic = self._bic(y, best_pred, 1)
        chosen: list[int] = []
        remaining = set(range(len(lib)))

        while len(chosen) < self.max_terms and remaining:
            candidate = None
            candidate_bic = best_bic
            candidate_coef = None

            for idx in sorted(remaining):
                cols = [np.ones(n)] + [lib[c].values for c in [*chosen, idx]]
                Phi = np.column_stack(cols)
                coef, *_ = np.linalg.lstsq(Phi, y, rcond=None)
                pred = Phi @ coef
                bic = self._bic(y, pred, Phi.shape[1])
                if bic < candidate_bic - self.min_bic_improvement:
                    candidate = idx
                    candidate_bic = bic
                    candidate_coef = coef

            if candidate is None:
                break

            chosen.append(candidate)
            remaining.remove(candidate)
            best_bic = candidate_bic
            self.coef_ = candidate_coef

        cols = [np.ones(n)] + [lib[c].values for c in chosen]
        Phi = np.column_stack(cols)
        self.coef_, *_ = np.linalg.lstsq(Phi, y, rcond=None)
        self.chosen_ = chosen
        self.term_names_ = [lib[c].name for c in chosen]
        self.bic_ = self._bic(y, Phi @ self.coef_, Phi.shape[1])
        return self

    def _design(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        lib = self._make_library(X)
        return np.column_stack(
            [np.ones(len(X))] + [lib[c].values for c in self.chosen_]
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._design(X) @ self.coef_

    def expression(self) -> str:
        pieces = [f"{self.coef_[0]:.5g}"]
        for coef, name in zip(self.coef_[1:], self.term_names_, strict=False):
            pieces.append(f"{coef:+.5g}*{name}")
        return " ".join(pieces)

    def complexity(self) -> float:
        return float(1 + len(self.chosen_))


class SinusoidSymbolicRegressor(FastSymbolicRegressor):
    """The fast library plus two term types whose frequency is FITTED.

    The fixed library has ``sin x``, ``cos x`` and ``sin 2x`` and nothing
    faster, so a law like ``sin(4.6 x)`` can only come back as a surrogate.
    Here each greedy round also offers, per input ``x_j``,

        b sin(a x_j)     and     b cos(a x_j)

    -- separate terms, as ``sin x`` and ``cos x`` are in the library; a phase
    is both at one frequency. The frequency ``a`` is chosen by variable
    projection: for every ``a`` on ``freq_grid`` the coefficient is least
    squares (vectorised over the grid through the current design's QR), the
    best grid point is refined by a bounded scalar search, and BIC charges
    the term two parameters, its coefficient and ``a``. It competes with the
    fixed terms on the same BIC, so it is chosen only when it explains the
    data better than they do.

    Three safeguards keep the term honest, each put there by a failure seen
    while building it:

    * a sinusoid must complete at least one period over the observed range
      of ``x_j`` (``a >= 2 pi / range``): a slower one is locally a
      polynomial, which the library already has;
    * a sinusoid is not offered when 80% of it is already explained by the
      terms chosen so far (variance inflation above 5): ``cos(1.05 x)`` with a
      large coefficient cancels against ``cos x`` into a parabola, which fits
      and explains nothing;
    * after the forward search a backward pass drops any term whose removal
      lowers BIC, and the plain library search is run too, the lower-BIC law
      being kept: greedy search can grab a slow cosine first because it
      imitates ``x^2``, and the sinusoid has to beat the library on the
      library's own terms.

    Only the per-cluster fit changes -- the problem, the objective and the
    loop are the ones :class:`GroupedDCSR` always runs. Not the default: every
    committed result uses the fixed library.
    """

    #: The largest share of a new sinusoid the chosen terms may already explain.
    MAX_EXPLAINED = 0.8
    _TRIG: ClassVar[dict] = {"sin": np.sin, "cos": np.cos}

    def __init__(self, freq_grid=None, **kwargs):
        super().__init__(**kwargs)
        self.freq_grid = (np.arange(0.5, 8.0001, 0.05) if freq_grid is None
                          else np.asarray(freq_grid, dtype=float))

    def _columns(self, X: np.ndarray, terms) -> list[np.ndarray]:
        lib = self._make_library(X) if any(t[0] == "lib" for t in terms) else None
        return [lib[t[1]].values if t[0] == "lib" else self._TRIG[t[0]](t[2] * X[:, t[1]])
                for t in terms]

    @staticmethod
    def _n_par(n_columns: int, terms) -> int:
        """Coefficients plus one per fitted frequency."""
        return n_columns + sum(t[0] != "lib" for t in terms)

    @classmethod
    def _rss_add(cls, Q, r, V):
        """RSS after adding each column of ``V`` to a design with orthonormal
        basis ``Q`` and residual ``r``; ``inf`` where the column is too
        collinear with that design."""
        v0 = (V * V).sum(0)
        V = V - Q @ (Q.T @ V)
        vv = (V * V).sum(0)
        rss = float(r @ r) - (V.T @ r) ** 2 / np.maximum(vv, 1e-12)
        return np.where(vv >= (1.0 - cls.MAX_EXPLAINED) * v0, rss, np.inf)

    def fit(self, X: np.ndarray, y: np.ndarray):
        from scipy.optimize import minimize_scalar

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        y = np.asarray(y, dtype=float).reshape(-1)
        n = len(y)
        lib = self._make_library(X)
        names = self.feature_names or [f"x{j}" for j in range(X.shape[1])]
        step = float(np.min(np.diff(self.freq_grid))) if len(self.freq_grid) > 1 else 0.1

        def bic_rss(rss, n_par):
            return n * np.log(max(rss, 1e-12) / n) + n_par * np.log(n)

        chosen: list[tuple] = []
        best_bic = self._bic(y, np.full(n, y.mean()), 1)
        while len(chosen) < self.max_terms:
            Phi = np.column_stack([np.ones(n), *self._columns(X, chosen)])
            n_par = self._n_par(Phi.shape[1], chosen)
            Q, _ = np.linalg.qr(Phi)
            r = y - Q @ (Q.T @ y)
            cand, cand_bic = None, best_bic - self.min_bic_improvement
            free = [i for i in range(len(lib)) if ("lib", i) not in chosen]
            if free:
                rss = self._rss_add(Q, r, np.column_stack([lib[i].values for i in free]))
                k = int(np.argmin(rss))
                if np.isfinite(rss[k]) and bic_rss(rss[k], n_par + 1) < cand_bic:
                    cand, cand_bic = ("lib", free[k]), bic_rss(rss[k], n_par + 1)
            for j in range(X.shape[1]):
                span = float(np.ptp(X[:, j]))
                if span <= 0:
                    continue
                lowest = 2.0 * np.pi / span
                grid = self.freq_grid[self.freq_grid >= lowest]
                if grid.size == 0:
                    continue
                A = np.outer(X[:, j], grid)
                for kind, fn in self._TRIG.items():
                    rss = self._rss_add(Q, r, fn(A))
                    if not np.isfinite(rss).any():
                        continue
                    a0 = float(grid[int(np.argmin(rss))])

                    def one(a, j=j, fn=fn, Q=Q, r=r):
                        return min(float(self._rss_add(Q, r, fn(a * X[:, j])[:, None])[0]), 1e300)

                    res = minimize_scalar(one, bounds=(max(a0 - step, lowest), a0 + step),
                                          method="bounded")
                    a, rss_a = ((float(res.x), float(res.fun)) if res.fun <= rss.min()
                                else (a0, float(rss.min())))
                    if bic_rss(rss_a, n_par + 2) < cand_bic:
                        cand, cand_bic = (kind, j, a), bic_rss(rss_a, n_par + 2)
            if cand is None:
                break
            chosen.append(cand)
            best_bic = cand_bic

        def bic_of(terms):
            P = np.column_stack([np.ones(n), *self._columns(X, terms)])
            c, *_ = np.linalg.lstsq(P, y, rcond=None)
            return self._bic(y, P @ c, self._n_par(P.shape[1], terms))

        # backward: drop any term whose removal lowers BIC, until none does
        while chosen:
            current = bic_of(chosen)
            b, i = min((bic_of(chosen[:i] + chosen[i + 1:]), i) for i in range(len(chosen)))
            if b >= current - self.min_bic_improvement:
                break
            chosen = chosen[:i] + chosen[i + 1:]

        # the sinusoids must beat the plain library on its own terms
        plain = FastSymbolicRegressor(
            max_terms=self.max_terms, include_trig=self.include_trig,
            include_log=self.include_log, include_interactions=self.include_interactions,
            min_bic_improvement=self.min_bic_improvement).fit(X, y)
        if plain.bic_ < bic_of(chosen) - self.min_bic_improvement:
            chosen = [("lib", c) for c in plain.chosen_]

        self.terms_ = chosen
        Phi = np.column_stack([np.ones(n), *self._columns(X, chosen)])
        self.coef_, *_ = np.linalg.lstsq(Phi, y, rcond=None)
        self.term_names_ = [lib[t[1]].name if t[0] == "lib" else f"{t[0]}({t[2]:.4g}*{names[t[1]]})"
                            for t in chosen]
        self.chosen_ = [t[1] for t in chosen if t[0] == "lib"]
        self.bic_ = self._bic(y, Phi @ self.coef_, self._n_par(Phi.shape[1], chosen))
        return self

    def _design(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        return np.column_stack([np.ones(len(X)), *self._columns(X, self.terms_)])

    def complexity(self) -> float:
        return float(self._n_par(1 + len(self.terms_), self.terms_))


class PySRRegressor(SymbolicRegressorBase):
    """Thin adapter for the PySR evolutionary symbolic regressor.

    Defaults are conservative and fully overridable through
    ``backend_kwargs``; the wrapper does not pin exact PySR options because
    the PySR API evolves between releases.
    """

    def __init__(self, feature_names: list[str] | None = None, **pysr_kwargs):
        try:
            import pysr
        except ImportError as exc:
            raise ImportError(
                "PySR is not installed. Install it (pip install pysr, which "
                "provisions its Julia backend on first use) and rerun with "
                "backend='pysr'."
            ) from exc
        options = {
            "niterations": 100,
            "binary_operators": ["+", "-", "*", "/"],
            "unary_operators": [],
            "maxsize": 20,
            "random_state": 7,
            "progress": False,
            "verbosity": 0,
        }
        options.update(pysr_kwargs)
        self.feature_names = feature_names
        self.model = pysr.PySRRegressor(**options)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        try:
            self.model.fit(X, y, variable_names=self.feature_names)
        except TypeError:  # older/newer PySR without variable_names in fit
            self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(np.asarray(X, dtype=float))).reshape(-1)

    def expression(self) -> str:
        try:
            return str(self.model.sympy())
        except Exception:
            try:
                return str(self.model.get_best()["equation"])
            except Exception:
                return str(self.model)

    def complexity(self) -> float:
        try:
            return float(self.model.get_best()["complexity"])
        except Exception:
            return 1.0


class KnownMechanismRegressor(SymbolicRegressorBase):
    """Fixed, fully known mechanism: no symbolic search is performed.

    Used for regimes whose complete functional form f_k(x) = h_k(x) is
    supplied. This is a special case / upper-bound benchmark: the known law
    is only used for prediction and regime scoring.
    """

    def __init__(self, function, expression_name: str | None = None,
                 complexity_value: float = 1.0):
        if not callable(function):
            raise ValueError("KnownMechanismRegressor requires a callable function(X)")
        self.function = function
        self.expression_name = expression_name
        self.complexity_value = float(complexity_value)

    def fit(self, X: np.ndarray, y: np.ndarray):
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.function(np.asarray(X, dtype=float))).reshape(-1)

    def expression(self) -> str:
        if self.expression_name is not None:
            return self.expression_name
        name = getattr(self.function, "__name__", "h")
        return f"known mechanism {name}(x)"

    def complexity(self) -> float:
        return self.complexity_value


class ResidualSymbolicRegressor(SymbolicRegressorBase):
    """Partially known mechanism: f_k(x) = h_k(x) + g_k(x).

    ``prior_function`` supplies the known part h_k; the wrapped
    ``residual_model`` discovers the unknown residual g_k by fitting
    r = y - h_k(X). Prediction returns h_k(X) + g_hat(X).
    """

    def __init__(self, prior_function, residual_model: SymbolicRegressorBase,
                 prior_expression: str | None = None,
                 prior_complexity: float = 1.0):
        if not callable(prior_function):
            raise ValueError("ResidualSymbolicRegressor requires a callable prior_function(X)")
        self.prior_function = prior_function
        self.residual_model = residual_model
        self.prior_expression = prior_expression
        self.prior_complexity = float(prior_complexity)

    def _prior(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.prior_function(np.asarray(X, dtype=float))).reshape(-1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        y = np.asarray(y, dtype=float).reshape(-1)
        self.residual_model.fit(X, y - self._prior(X))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._prior(X) + np.asarray(self.residual_model.predict(X)).reshape(-1)

    def expression(self) -> str:
        prior = self.prior_expression or getattr(
            self.prior_function, "__name__", "h"
        ) + "(x)"
        return f"{prior} + ({self.residual_model.expression()})"

    def complexity(self) -> float:
        return self.prior_complexity + float(self.residual_model.complexity())


class SuperpositionRegressor(SymbolicRegressorBase):
    """Composite mechanism: f_k(x) = sum_j w_j * f_j(x) + c.

    Wraps already-fitted component models (from other clusters) and fits only
    the mixing weights by ordinary least squares. This is the model-side
    counterpart of the simulators' mix-phenomena layer: when a regime really
    is a superposition of other regimes, this hypothesis fits it with a
    handful of free parameters instead of a fresh symbolic search, so it
    generalizes better and the recovered weights are directly interpretable.

    The component models are treated as frozen feature maps; ``fit`` never
    refits them, so composites cannot be built recursively from composites.

    Weights are constrained to be **non-negative and bounded** (a physical
    superposition adds severities, it does not subtract mechanisms), and the
    hypothesis is only considered *valid* when every component carries a
    material share of the explained signal — otherwise a "superposition" is
    just a rescaled copy of a single law, which would erode the very
    distinctions the equation term is supposed to provide.
    """

    def __init__(self, components: list[SymbolicRegressorBase],
                 component_labels: list[str] | None = None,
                 max_weight: float = 5.0,
                 min_share: float = 0.15):
        if len(components) < 2:
            raise ValueError("SuperpositionRegressor needs at least two components")
        self.components = list(components)
        self.component_labels = component_labels
        self.max_weight = float(max_weight)
        self.min_share = float(min_share)
        self.weights_ = None
        self.intercept_ = 0.0
        self.shares_ = None

    def _component_predictions(self, X: np.ndarray) -> np.ndarray:
        cols = [np.asarray(m.predict(X), dtype=float).reshape(-1) for m in self.components]
        return np.nan_to_num(np.column_stack(cols), nan=0.0, posinf=0.0, neginf=0.0)

    def _design(self, X: np.ndarray) -> np.ndarray:
        C = self._component_predictions(X)
        return np.column_stack([C, np.ones(C.shape[0])])

    def fit(self, X: np.ndarray, y: np.ndarray):
        from scipy.optimize import lsq_linear

        X = np.asarray(X, dtype=float)
        y = np.nan_to_num(np.asarray(y, dtype=float).reshape(-1))
        C = self._component_predictions(X)
        A = np.column_stack([C, np.ones(C.shape[0])])
        n = C.shape[1]
        lower = np.append(np.zeros(n), -np.inf)
        upper = np.append(np.full(n, self.max_weight), np.inf)
        sol = lsq_linear(A, y, bounds=(lower, upper), max_iter=100)
        self.weights_ = np.asarray(sol.x[:-1], dtype=float)
        self.intercept_ = float(sol.x[-1])

        # Share of explained variation carried by each component. A valid
        # superposition needs every component to contribute materially.
        scales = np.array([np.std(C[:, j]) for j in range(n)])
        contrib = np.abs(self.weights_) * scales
        total = contrib.sum()
        self.shares_ = contrib / total if total > 1e-12 else np.zeros(n)
        return self

    @property
    def is_valid(self) -> bool:
        """True when every component carries at least ``min_share`` of the fit."""
        if self.shares_ is None:
            return False
        return bool(np.all(self.shares_ >= self.min_share))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("SuperpositionRegressor is not fitted")
        A = np.nan_to_num(self._design(np.asarray(X, dtype=float)),
                          nan=0.0, posinf=0.0, neginf=0.0)
        return A @ np.append(self.weights_, self.intercept_)

    def expression(self) -> str:
        if self.weights_ is None:
            return "superposition (unfitted)"
        labels = self.component_labels or [f"f{j}" for j in range(len(self.components))]
        terms = [f"{w:.4g}*[{lab}: {m.expression()}]"
                 for w, lab, m in zip(self.weights_, labels, self.components, strict=False)]
        return " + ".join(terms) + f" + {self.intercept_:.4g}"

    def complexity(self) -> float:
        # Component structure is inherited, but only the weights are free
        # parameters here; charge the inherited structure at a discount so a
        # true composite is preferred over an unconstrained fresh search.
        inherited = sum(float(m.complexity()) for m in self.components)
        return 0.5 * inherited + float(len(self.components) + 1)


class DSORegressor(SymbolicRegressorBase):
    """Optional adapter for a compatible DeepSymbolicRegressor installation."""

    def __init__(self, config=None):
        try:
            from dso import DeepSymbolicRegressor
        except ImportError as exc:
            raise ImportError(
                "DSO is not installed. Install a compatible Deep Symbolic "
                "Optimization package/repository and rerun with backend='dso'."
            ) from exc
        self.model = DeepSymbolicRegressor() if config is None else DeepSymbolicRegressor(config)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(np.asarray(X), np.asarray(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(np.asarray(X))).reshape(-1)

    def expression(self) -> str:
        try:
            return self.model.program_.pretty()
        except Exception:
            return str(self.model.program_)

    def complexity(self) -> float:
        try:
            return float(self.model.program_.complexity)
        except Exception:
            return 1.0


class _PhySOFitTimeout(BaseException):
    """Raised by PhySORegressor's wall-clock guard.

    Deliberately a BaseException so it is not caught by physo's internal
    `except Exception` handlers around program execution.
    """


class PhySORegressor(SymbolicRegressorBase):
    """Thin adapter for PhySO (Physical Symbolic Optimization).

    https://github.com/wassimtenachi/physo -- reinforcement-learning symbolic
    regression with optional dimensional-analysis constraints. A third search
    paradigm alongside ``fast`` (greedy BIC) and ``pysr`` (genetic programming),
    used for **SR-engine sensitivity checks**, not as a default: it is slow
    (every DCSR iteration refits every cluster) and stochastic, so run it on a
    handful of bounded configurations, not the full sweep, and over several
    seeds.

    Defaults are conservative and fully overridable through ``backend_kwargs``
    (``run_config``, ``op_names``, ``epochs``, ``X_units``, ``y_units``, ...).
    The wrapper does not pin PhySO internals because the API evolves between
    releases; anything in ``physo_kwargs`` is forwarded verbatim to ``physo.SR``.
    """

    def __init__(self, feature_names: list[str] | None = None, **physo_kwargs):
        try:
            import physo  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "PhySO is not installed. Install it (pip install physo, which "
                "pulls in PyTorch) and rerun with backend='physo'."
            ) from exc
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "PhySO requires PyTorch. Install torch and rerun with "
                "backend='physo'."
            ) from exc
        self.feature_names = feature_names
        # Small, bounded search so this stays usable inside the alternating loop.
        # ``free_consts_names`` matters: without free constants PhySO cannot
        # represent a scaled law (a*x^2 + b), so the dimensionless SR needs a
        # handful declared. Everything here is overridable via backend_kwargs.
        #
        # NOTE on the monitors: physo.SR treats ``get_run_logger=None`` /
        # ``get_run_visualiser=None`` as "install the DEFAULTS", not as "off"
        # (see physo.SR, `if get_run_logger is None: get_run_logger = ...`).
        # The defaults write SR.log and re-render a ~1.5 MB SR_curves.png every
        # single epoch, which cost ~16% of fit time and littered the repo root.
        # Passing explicitly inert monitors is the only way to switch them off.
        self._opts = {
            "op_names": ["add", "sub", "mul", "div", "n2", "sqrt", "sin", "cos"],
            "free_consts_names": ["a", "b", "c"],
            "epochs": 20,
            # parallel_mode: physo defaults this to True. It is forced off here
            # because physo's parallelism re-executes the importing module in
            # each spawned worker; without a __main__ guard that re-runs the
            # caller. Left as a measured open question, not a settled verdict.
            "parallel_mode": False,
            "get_run_logger": self._quiet_logger,
            "get_run_visualiser": self._quiet_visualiser,
        }
        self._opts.update(physo_kwargs)
        self._verbose = bool(self._opts.pop("verbose", False))
        # Hard wall-clock guard on a single fit. Some in-loop fits have stalled
        # for hours with no epoch progress and no traceback; the cause is not
        # understood, so this bounds the damage to one skipped fit rather than
        # a long run. None disables it.
        self._fit_timeout_s = self._opts.pop("fit_timeout_s", None)
        self._program = None
        self._expr = "physo (unfitted)"

    @staticmethod
    def _quiet_logger():
        from physo.learn import monitoring
        return monitoring.RunLogger(save_path=None, do_save=False)

    @staticmethod
    def _quiet_visualiser():
        from physo.learn import monitoring
        return monitoring.RunVisualiser(
            epoch_refresh_rate=10**9, save_path=None,
            do_show=False, do_prints=False, do_save=False)

    @staticmethod
    @contextlib_contextmanager
    def _time_limit(seconds):
        """SIGALRM wall-clock cap on the wrapped block.

        Only armed on the main thread of a POSIX process -- signal.alarm is
        unavailable elsewhere, and there the block simply runs unguarded.
        """
        import signal
        import threading

        armed = (seconds and hasattr(signal, "SIGALRM")
                 and threading.current_thread() is threading.main_thread())
        if not armed:
            yield
            return

        def _raise(signum, frame):
            # BaseException, NOT TimeoutError: physo wraps program execution in
            # broad `except Exception` handlers, which silently swallow an
            # ordinary timeout and let the fit run on (measured: a 5 s cap on a
            # 135 s fit had no effect). A BaseException escapes those handlers.
            raise _PhySOFitTimeout(f"PhySO fit exceeded {seconds}s")

        previous = signal.signal(signal.SIGALRM, _raise)
        # REPEATING timer, not a one-shot alarm. physo executes each candidate
        # program inside a bare `except:` (physym/batch_execute.py:123,229),
        # which swallows even a BaseException -- a single alarm is absorbed and
        # the fit runs on regardless. Re-firing every second means the raise
        # eventually lands outside one of those blocks and propagates.
        signal.setitimer(signal.ITIMER_REAL, float(seconds), 1.0)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0, 0.0)
            signal.signal(signal.SIGALRM, previous)

    def fit(self, X: np.ndarray, y: np.ndarray):
        import contextlib
        import io

        import physo
        import torch

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        y = np.asarray(y, dtype=float).reshape(-1)
        names = self.feature_names or [f"x{j}" for j in range(X.shape[1])]

        opts = dict(self._opts)
        opts.setdefault("X_names", list(names))
        opts.setdefault("y_name", "y")
        # physo.SR expects (n_dim, n_samples) tensors.
        Xt = torch.tensor(X.T)
        yt = torch.tensor(y)
        sink = io.StringIO()
        cm = contextlib.nullcontext() if self._verbose else contextlib.redirect_stdout(sink)
        with cm, self._time_limit(self._fit_timeout_s):
            result = physo.SR(Xt, yt, **opts)
        # physo.SR returns (best_expression, logs) across known versions.
        self._program = result[0] if isinstance(result, (tuple, list)) else result

        for getter in ("get_infix_sympy", "get_infix_str", "get_infix_pretty", "__str__"):
            try:
                val = getattr(self._program, getter)()
                if val is not None:
                    self._expr = str(val)
                    break
            except Exception:
                continue
        return self

    def _execute(self, X: np.ndarray) -> np.ndarray:
        import torch

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        Xt = torch.tensor(X.T)
        out = self._program.execute(Xt)
        arr = out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)
        return np.asarray(arr, dtype=float).reshape(-1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._program is None:
            raise RuntimeError("PhySORegressor is not fitted")
        # PhySO programs carry their own optimised free constants; execute() is
        # the only reliable evaluator (the infix string names constants a/b/c,
        # so a sympy lambdify cannot stand in for it).
        return np.nan_to_num(self._execute(X), nan=0.0, posinf=0.0, neginf=0.0)

    def expression(self) -> str:
        return self._expr

    def complexity(self) -> float:
        for attr in ("size", "n_complexity", "complexity"):
            try:
                v = getattr(self._program, attr)
                return float(v() if callable(v) else v)
            except Exception:
                continue
        try:
            return float(len(self._program.tokens))
        except Exception:
            pass
        try:
            import sympy

            return float(sympy.count_ops(sympy.sympify(self._expr)) + 1)
        except Exception:
            return 1.0


def make_symbolic_regressor(
    backend: str = "fast",
    feature_names: list[str] | None = None,
    **kwargs,
) -> SymbolicRegressorBase:
    """Build a regressor by name: ``fast`` (default), ``pysr`` or ``physo``.

    ``fast`` is a deterministic library search — a fixed basis, fitted by
    least squares with a complexity penalty. It is the default everywhere and
    **every committed result in this project uses it**, because a genetic
    search that returns a different expression each run cannot support a
    claim that a particular law was recovered. The other two backends exist
    for sensitivity checks, and are optional dependencies.
    """
    backend = backend.lower()
    if backend == "fast":
        return FastSymbolicRegressor(feature_names=feature_names, **kwargs)
    if backend == "fast_sin":
        return SinusoidSymbolicRegressor(feature_names=feature_names, **kwargs)
    if backend == "pysr":
        return PySRRegressor(feature_names=feature_names, **kwargs)
    if backend == "physo":
        return PhySORegressor(feature_names=feature_names, **kwargs)
    if backend == "dso":
        return DSORegressor(**kwargs)
    raise ValueError(f"Unknown symbolic backend: {backend}")


class ParametricPriorRegressor(SymbolicRegressorBase):
    """Partially known mechanism whose PRIOR ITSELF HAS FREE PARAMETERS.

    ``ResidualSymbolicRegressor`` takes a fully specified prior ``h_k(x)``.
    Physics usually gives less than that: the functional *form* is known but
    its coefficients are not (the rain power law ``A = a R^b L`` is known to
    be a power law, while ``a`` and ``b`` depend on frequency, drop-size
    distribution and polarisation). This wrapper covers that case:

        f_k(x) = h(x; theta_hat) + g_k(x)

    where ``theta_hat`` is fitted by bounded least squares on the cluster's
    own data and ``g_k`` is discovered by the wrapped symbolic model on the
    remaining residual. Both steps see only the data currently assigned to
    the cluster -- no labels, no reference quantities.

    Parameters
    ----------
    prior_form : callable(X, theta) -> array
        The known functional form, vectorised over rows of ``X``.
    p0, bounds
        Initial guess and ``(lower, upper)`` bounds for ``theta``, passed to
        :func:`scipy.optimize.least_squares`. Bounds are what keeps a
        "physical" prior physical: an unbounded fit can absorb any residual
        and the knowledge stops being knowledge.
    residual_model : SymbolicRegressorBase or None
        Discovers ``g_k``. ``None`` means the prior alone is the mechanism,
        i.e. a *fitted-known* mechanism.
    expression_template : str, optional
        Human-readable form with ``{0}``, ``{1}``, ... placeholders for the
        fitted parameters, e.g. ``"{0:.4g} * R^{1:.3f} * L"``.
    """

    def __init__(self, prior_form, p0, bounds=None,
                 residual_model: SymbolicRegressorBase | None = None,
                 expression_template: str | None = None,
                 prior_complexity: float = 2.0):
        if not callable(prior_form):
            raise ValueError("prior_form must be callable(X, theta)")
        self.prior_form = prior_form
        self.p0 = np.asarray(p0, dtype=float)
        self.bounds = bounds if bounds is not None else (-np.inf, np.inf)
        self.residual_model = residual_model
        self.expression_template = expression_template
        self.prior_complexity = float(prior_complexity)
        self.theta_ = np.asarray(p0, dtype=float)

    def _prior(self, X: np.ndarray, theta=None) -> np.ndarray:
        theta = self.theta_ if theta is None else theta
        out = self.prior_form(np.asarray(X, dtype=float), theta)
        return np.nan_to_num(np.asarray(out, dtype=float).reshape(-1),
                             nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, X: np.ndarray, y: np.ndarray):
        from scipy.optimize import least_squares

        X = np.asarray(X, dtype=float)
        y = np.nan_to_num(np.asarray(y, dtype=float).reshape(-1))

        def residuals(theta):
            return self._prior(X, theta) - y

        try:
            sol = least_squares(residuals, self.p0, bounds=self.bounds,
                                max_nfev=200)
            self.theta_ = np.asarray(sol.x, dtype=float)
        except (ValueError, np.linalg.LinAlgError):
            self.theta_ = self.p0.copy()  # keep the prior at its nominal value

        if self.residual_model is not None:
            self.residual_model.fit(X, y - self._prior(X))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = self._prior(X)
        if self.residual_model is not None:
            pred = pred + np.asarray(self.residual_model.predict(X)).reshape(-1)
        return pred

    def expression(self) -> str:
        if self.expression_template is not None:
            prior = self.expression_template.format(*self.theta_)
        else:
            params = ", ".join(f"{t:.4g}" for t in self.theta_)
            prior = f"h(x; {params})"
        if self.residual_model is None:
            return prior
        return f"{prior} + ({self.residual_model.expression()})"

    def complexity(self) -> float:
        c = self.prior_complexity + float(self.theta_.size)
        if self.residual_model is not None:
            c += float(self.residual_model.complexity())
        return c
