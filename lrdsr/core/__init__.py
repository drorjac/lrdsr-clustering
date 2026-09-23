"""The LR-DSR algorithm: dataset-agnostic core.

Read these five files to understand the whole method:

- ``model.py``      -- ``GroupedDCSR`` / ``RowDCSR``: the alternating loop.
- ``backends.py``   -- symbolic-regression interface + mechanism-knowledge wrappers.
- ``losses.py``     -- the joint assignment cost and its residual scaling.
- ``baselines.py``  -- the 7 geometry-only baselines (fair-comparison contract).
- ``evaluation.py`` -- ARI / NMI / aligned accuracy / regression metrics.

Nothing in this package may read true labels except the optional
``true_labels_for_eval`` logging hook, and nothing here may import from an
application package (``lrdsr.cml``) or from ``experiments/``.
"""

from .backends import (
    DSORegressor,
    FastSymbolicRegressor,
    KnownMechanismRegressor,
    ParametricPriorRegressor,
    PySRRegressor,
    ResidualSymbolicRegressor,
    SuperpositionRegressor,
    SymbolicRegressorBase,
    make_symbolic_regressor,
)
from .baselines import fuzzy_cmeans, geometry_baselines
from .evaluation import aligned_accuracy, clustering_metrics, nmse, regression_metrics
from .losses import aggregate_window_residual, huber_loss, joint_cost, robust_scale
from .model import DCSRResult, GroupedDCSR, RowDCSR

__all__ = [
    "DCSRResult",
    "DSORegressor",
    "FastSymbolicRegressor",
    "GroupedDCSR",
    "KnownMechanismRegressor",
    "ParametricPriorRegressor",
    "PySRRegressor",
    "ResidualSymbolicRegressor",
    "RowDCSR",
    "SuperpositionRegressor",
    "SymbolicRegressorBase",
    "aggregate_window_residual",
    "aligned_accuracy",
    "clustering_metrics",
    "fuzzy_cmeans",
    "geometry_baselines",
    "huber_loss",
    "joint_cost",
    "make_symbolic_regressor",
    "nmse",
    "regression_metrics",
    "robust_scale",
]
