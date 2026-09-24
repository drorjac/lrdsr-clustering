"""The LR-DSR algorithm: dataset-agnostic core.

Read these five files to understand the whole method (and the two after
them for the soft and the real-time variants):

- ``model.py``      -- ``GroupedDCSR`` / ``RowDCSR``: the alternating loop.
- ``backends.py``   -- symbolic-regression interface + mechanism-knowledge wrappers.
- ``losses.py``     -- the joint assignment cost and its residual scaling.
- ``baselines.py``  -- the 7 geometry-only baselines (fair-comparison contract).
- ``evaluation.py`` -- ARI / NMI / aligned accuracy / regression metrics.
- ``soft.py``       -- ``SoftLRDSR``: EM with responsibilities and a learned loss.
- ``online.py``     -- ``OnlineLRDSR`` / ``CusumSegmenter``: real-time clustering.

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
from .losses import (
    LOSSES,
    LearnedLoss,
    aggregate_window_residual,
    huber_loss,
    joint_cost,
    learn_loss,
    loss_psi,
    loss_value,
    noise_scale,
    robust_scale,
)
from .model import DCSRResult, GroupedDCSR, RowDCSR
from .online import CusumSegmenter, OnlineLRDSR
from .soft import SoftLRDSR, SoftResult

__all__ = [
    "LOSSES",
    "CusumSegmenter",
    "DCSRResult",
    "DSORegressor",
    "FastSymbolicRegressor",
    "GroupedDCSR",
    "KnownMechanismRegressor",
    "LearnedLoss",
    "OnlineLRDSR",
    "ParametricPriorRegressor",
    "PySRRegressor",
    "ResidualSymbolicRegressor",
    "RowDCSR",
    "SoftLRDSR",
    "SoftResult",
    "SuperpositionRegressor",
    "SymbolicRegressorBase",
    "aggregate_window_residual",
    "aligned_accuracy",
    "clustering_metrics",
    "fuzzy_cmeans",
    "geometry_baselines",
    "huber_loss",
    "joint_cost",
    "learn_loss",
    "loss_psi",
    "loss_value",
    "make_symbolic_regressor",
    "nmse",
    "noise_scale",
    "regression_metrics",
    "robust_scale",
]
