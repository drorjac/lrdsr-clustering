"""Shared statistics: a reproducible percentile bootstrap.

The unit of replication is whatever is independent -- a seed, a day, a
turbine -- never a single sample: samples inside one window share its law
and its noise, so resampling them would report confidence the data do not
have.
"""
from __future__ import annotations

import numpy as np

#: Bootstrap replicates, and the project-wide seed for them. Fixed so that a
#: confidence interval is reproducible: a CI that moves when it is recomputed
#: cannot be compared across revisions.
N_BOOTSTRAP, BOOT_SEED = 2000, 20260908


def boot_ci(stat, n, n_boot=None, seed=None, alpha=0.05):
    """Percentile bootstrap of ``stat(idx)`` over ``n`` units."""
    n_boot = N_BOOTSTRAP if n_boot is None else n_boot
    rng = np.random.default_rng(BOOT_SEED if seed is None else seed)
    vals = np.array([stat(rng.integers(0, n, n)) for _ in range(n_boot)])
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return np.nan, np.nan
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))
