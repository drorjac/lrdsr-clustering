"""The real-data windows: built right, and the cached archives unchanged.

The builder is tested on a small synthetic hourly frame, so these run
offline. Anything that needs the downloaded archives is skipped when they
are not in ``.cache/realdata/``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.realdata import sources
from experiments.realdata.windows import HOURS, build_windows, traffic_hourly


def _hourly(n_days=6, drop=None, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        date = pd.Timestamp("2020-01-06") + pd.Timedelta(days=d)   # a Monday
        for h in range(HOURS):
            if drop and (d, h) in drop:
                continue
            rows.append({"date": date, "hour": h,
                         "count": float(rng.integers(1, 500)),
                         "working": date.dayofweek < 5, "holiday": "",
                         "weather": 1, "temp": 0.5})
    return pd.DataFrame(rows)


def test_shape_and_centring():
    w = build_windows(_hourly(), "toy")
    assert w.X_seq.shape == (6, HOURS, 1) and w.y_seq.shape == (6, HOURS)
    np.testing.assert_allclose(w.y_seq.mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(w.X_seq[0, :, 0], 2 * np.pi * np.arange(HOURS) / HOURS)
    # Mon..Sat: five working days then a weekend day
    assert w.reference.tolist() == [1, 1, 1, 1, 1, 0]
    assert w.meta["day_type"].tolist()[-1] == "weekend"


def test_incomplete_days_dropped_or_zero_filled():
    drop = {(1, 3), (2, 3), (2, 4), (2, 5)}
    strict = build_windows(_hourly(drop=drop), "toy", max_zero_fill=0)
    assert strict.coverage["days_kept"] == 4
    assert strict.coverage["days_dropped_incomplete"] == 2
    lenient = build_windows(_hourly(drop=drop), "toy", max_zero_fill=1)
    assert lenient.coverage["days_kept"] == 5
    assert lenient.coverage["hours_zero_filled"] == 1


def test_duplicate_timestamps_collapse():
    h = _hourly(n_days=2)
    w = build_windows(pd.concat([h, h.iloc[:5]]), "toy")
    assert w.coverage["days_kept"] == 2


def test_traffic_holiday_propagates_from_midnight_row():
    t = pd.date_range("2016-07-04", periods=48, freq="h")      # a Monday holiday
    raw = pd.DataFrame({"date_time": t.astype(str),
                        "holiday": ["Independence Day"] + ["None"] * 47,
                        "temp": 290.0, "rain_1h": 0.0, "snow_1h": 0.0,
                        "traffic_volume": 1000})
    tidy = traffic_hourly(raw)
    first = tidy[tidy["date"] == pd.Timestamp("2016-07-04")]
    assert (first["holiday"] == "Independence Day").all()
    assert not first["working"].any()
    assert tidy[tidy["date"] == pd.Timestamp("2016-07-05")]["working"].all()


@pytest.mark.parametrize("name", list(sources.SOURCES))
def test_cached_archive_matches_checksum(name):
    if not sources.is_cached(name):
        pytest.skip(f"{name} archive not cached; run python -m experiments.realdata.run")
    sources.fetch(name)          # raises on a checksum mismatch


@pytest.mark.skipif(not sources.is_cached("bike"), reason="bike archive not cached")
def test_bike_windows_have_the_published_calendar():
    from experiments.realdata.windows import load
    w = load("bike")
    assert w.coverage["days_raw"] == 731
    assert 0.6 < w.reference.mean() < 0.75
    assert np.isfinite(w.y_seq).all()
