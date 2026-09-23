"""The three-law function mixture, re-exported for the model tests.

The generator itself lives with the experiment that uses it
(``experiments/functions/data.py``, the functions block) so there is exactly one copy.
"""
from __future__ import annotations

from experiments.functions.data import (
    BASE_FUNCTIONS,
    DEFAULT_COMPONENTS,
    FunctionMixtureData,
    simulate_function_mixture,
    simulate_function_windows,
)

__all__ = ["BASE_FUNCTIONS", "DEFAULT_COMPONENTS", "FunctionMixtureData",
           "simulate_function_mixture", "simulate_function_windows"]
