"""Wind blocks, the method of bins, and the proxies -- on a synthetic SCADA frame.

Offline: the frame is built here in the shape ``sources.load_scada`` returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.wind import windows as W
from lrdsr.core.kernel import NystromBasis


def _curve(ws):
    # a realistic shape near cut-in (~50 kW at 4.5 m/s): the downtime rule
    # must drop only the stopped rows, never a turbine that is just starting
    return np.clip((ws - 3.0) / (12.5 - 3.0), 0, 1) ** 2 * 2050


def _scada(days=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(1, 7):
        time = pd.date_range("2017-01-02", periods=144 * days, freq="10min")
        ws = np.clip(6 + 3 * np.sin(np.arange(len(time)) / 50) + rng.normal(0, 0.5, len(time)), 0, 25)
        p = _curve(ws) + rng.normal(0, 20, len(time))
        p[100:110] = 0.0                                 # downtime with wind
        rows.append(pd.DataFrame({"time": time, "ws": ws, "ws_sd": np.nan,
                                  "direction": rng.uniform(0, 360, len(time)),
                                  "power": p, "temp": 5.0 + t, "turbine": t}))
    return pd.concat(rows, ignore_index=True)


def test_blocks_drop_downtime_and_keep_ragged_designs(monkeypatch):
    monkeypatch.setattr(W, "_waked", lambda d: np.zeros(len(d), bool))
    b = W.build(_scada())
    assert len(b.X) == 6 * 4 * 4                        # 6 turbines x 4 days x 4 blocks
    assert all(len(x) == len(v) >= W.MIN_SAMPLES for x, v in zip(b.X, b.y, strict=True))
    # the downtime rows (wind above cut-in, no power) never reach a window
    assert min(v.min() for v in b.y) > -0.1
    assert set(b.meta["night"].unique()) <= {-1, 0, 1}
    assert (b.meta["cold"] >= 0).all()


def test_method_of_bins_interpolates_empty_bins():
    x = np.array([[5.2], [5.3], [8.1]])
    y = np.array([0.1, 0.2, 0.5])
    prof = W.bin_profile([x], [y])[0]
    centres = 0.5 * (W.BIN_EDGES[1:] + W.BIN_EDGES[:-1])
    assert prof[np.searchsorted(centres, 5.25)] == pytest.approx(0.15)
    mid = np.searchsorted(centres, 6.75)                # an empty bin between the two
    assert 0.15 < prof[mid] < 0.5
    assert prof[0] == pytest.approx(0.15) and prof[-1] == pytest.approx(0.5)


def test_uniform_centres_cover_a_skewed_design():
    x = np.random.default_rng(1).gamma(2.0, 2.0, 5000)[:, None]   # wind-like skew
    q = NystromBasis(12).fit(x).centers_[:, 0]
    u = NystromBasis(12, spacing="uniform").fit(x).centers_[:, 0]
    top = np.quantile(x, 0.99) / x.std()
    assert (u > 0.7 * top).sum() > (q > 0.7 * top).sum()
    with pytest.raises(ValueError):
        NystromBasis(4, spacing="log")
