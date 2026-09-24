"""One day is one window: the real series in the shape the estimators take.

The construction, identical for both series:

.. code-block:: text

    x_t = 2 pi hour_t / 24                      t = 0..23
    y_t = log1p(count_t) - mean_t log1p(count_t)

``x`` on the circle makes the fast library's ``sin x``, ``cos x`` and
``sin 2x`` a (truncated) Fourier basis of the daily cycle, with the
polynomials still there for anything asymmetric. The per-day centring is
label-free and removes the day's overall level -- season, growth of the
scheme, a closed lane -- so what is left to cluster is the **shape** of the
daily law, which is the claim: a working day and a day off differ in *how*
traffic unfolds over the day, not in how much of it there is.

The reference label is the calendar day type, ``1`` for a working day and
``0`` for a weekend or public holiday. It is a **proxy** for the mechanism,
not the mechanism: a bridge day, a snow day or the week between Christmas
and New Year behaves like a day off while being a working day on paper. It
is used to score a partition and never reaches a fit.

Coverage. A day is kept only if all 24 hours are present after the rules
below; :func:`build_windows` records how many were dropped and why.

* ``bike`` -- the table aggregates trip logs, so an hour with no trip has no
  row. A day missing at most ``max_zero_fill`` hours gets those hours as
  zero rentals (almost all are 2-5 am in winter); a day missing more is an
  outage (e.g. Hurricane Sandy, 2012-10-29/30) and is dropped.
* ``traffic`` -- a detector never reports zero on an interstate, so a
  missing hour is a sensor gap and the day is dropped. Duplicate timestamps
  (one row per weather description; the count is identical across them, as
  checked) are collapsed to one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HOURS = 24


@dataclass
class DayWindows:
    """Days as windows. ``X_seq`` ``(W, 24, 1)``, ``y_seq`` ``(W, 24)``.

    ``reference`` is the calendar day type (1 = working day), for scoring
    only; ``meta`` has one row per day (date, weekday, day type, holiday,
    weather, season, level); ``coverage`` says what was dropped.
    """
    name: str
    X_seq: np.ndarray
    y_seq: np.ndarray
    reference: np.ndarray
    meta: pd.DataFrame
    coverage: dict

    @property
    def dates(self) -> pd.Series:
        return self.meta["date"]

    @property
    def raw_profile(self) -> np.ndarray:
        """The 24-dim centred log profile: the input of the raw-profile baseline."""
        return self.y_seq


def build_windows(hourly: pd.DataFrame, name: str, max_zero_fill: int = 0,
                  ) -> DayWindows:
    """Turn a tidy hourly frame into day windows.

    ``hourly`` needs ``date`` (datetime64, day resolution), ``hour`` (0..23),
    ``count`` and ``working`` (bool, per day); any other column is treated as
    per-hour meta and summarised per day by :func:`_day_meta`.
    """
    h = hourly.drop_duplicates(["date", "hour"]).copy()
    n_days_raw = h["date"].nunique()
    per_day = h.groupby("date")["hour"].nunique()
    missing = HOURS - per_day
    keep = missing[missing <= max_zero_fill].index
    h = h[h["date"].isin(keep)]
    grid = (h.pivot(index="date", columns="hour", values="count")
             .reindex(columns=range(HOURS)))
    filled = int(grid.isna().to_numpy().sum())
    grid = grid.fillna(0.0).sort_index()
    logc = np.log1p(grid.to_numpy(dtype=float))
    level = logc.mean(axis=1)
    y = logc - level[:, None]
    x = 2.0 * np.pi * np.arange(HOURS) / HOURS
    X_seq = np.broadcast_to(x[None, :, None], (len(grid), HOURS, 1)).copy()

    meta = _day_meta(h, grid.index)
    meta["level"] = level
    ref = meta["working"].astype(int).to_numpy()
    coverage = {"dataset": name, "days_raw": int(n_days_raw),
                "days_kept": int(len(grid)),
                "days_dropped_incomplete": int(n_days_raw - len(grid)),
                "hours_zero_filled": filled,
                "working_share": float(ref.mean())}
    return DayWindows(name, X_seq, y, ref, meta.reset_index(drop=True), coverage)


def _day_meta(h: pd.DataFrame, dates: pd.Index) -> pd.DataFrame:
    """One row per kept day: calendar, and weather summarised over 7-20 h."""
    g = h.groupby("date")
    meta = pd.DataFrame({"date": dates})
    meta["weekday"] = pd.to_datetime(meta["date"]).dt.dayofweek
    meta["month"] = pd.to_datetime(meta["date"]).dt.month
    meta["year"] = pd.to_datetime(meta["date"]).dt.year
    meta["season"] = meta["month"].map(
        lambda m: ("winter", "spring", "summer", "autumn")[(m % 12) // 3])
    meta["working"] = g["working"].first().reindex(dates).to_numpy().astype(bool)
    meta["holiday"] = g["holiday"].first().reindex(dates).to_numpy()
    meta["day_type"] = np.where(meta["working"], "working",
                                np.where(meta["holiday"].astype(str) != "",
                                         "holiday", "weekend"))
    day = h[(h["hour"] >= 7) & (h["hour"] <= 20)].groupby("date")
    if "weather" in h:
        # worst daytime weather category (bike: 1 clear .. 4 heavy rain/snow)
        meta["weather"] = day["weather"].max().reindex(dates).to_numpy()
    if "precip" in h:
        meta["precip"] = day["precip"].sum().reindex(dates).to_numpy()
    if "temp" in h:
        meta["temp"] = day["temp"].mean().reindex(dates).to_numpy()
    return meta


# ==========================================================================
# the two series
# ==========================================================================
def bike_hourly(raw: pd.DataFrame) -> pd.DataFrame:
    """UCI Bike Sharing ``hour.csv`` -> the tidy hourly frame.

    ``working`` is the dataset's own ``workingday`` flag (neither weekend nor
    a DC public holiday); ``weather`` its ``weathersit`` (1 clear .. 4 heavy
    precipitation); ``temp`` its normalised temperature.
    """
    out = pd.DataFrame({
        "date": pd.to_datetime(raw["dteday"]),
        "hour": raw["hr"].astype(int),
        "count": raw["cnt"].astype(float),
        "working": raw["workingday"].astype(bool),
        "weather": raw["weathersit"].astype(int),
        "temp": raw["temp"].astype(float),
    })
    hol = raw["holiday"].astype(bool)
    out["holiday"] = np.where(hol, "holiday", "")
    return out


def traffic_hourly(raw: pd.DataFrame) -> pd.DataFrame:
    """UCI Metro Interstate Traffic Volume -> the tidy hourly frame.

    The published ``holiday`` column names the holiday on the day's
    **midnight row only**; it is propagated to the whole date here, otherwise
    23 of a holiday's 24 hours would read as ordinary. ``working`` is Mon-Fri
    and not a named holiday. ``precip`` is rain + snow in mm/h, ``temp`` in
    kelvin.
    """
    t = pd.to_datetime(raw["date_time"])
    out = pd.DataFrame({
        "date": t.dt.normalize(),
        "hour": t.dt.hour.astype(int),
        "count": raw["traffic_volume"].astype(float),
        "precip": raw["rain_1h"].clip(upper=100).astype(float)
                  + raw["snow_1h"].astype(float),
        "temp": raw["temp"].astype(float),
    })
    names = (pd.Series(raw["holiday"].where(raw["holiday"].notna()
                                           & (raw["holiday"] != "None")).to_numpy(),
                       index=out.index)
             .groupby(out["date"]).transform(lambda s: s.dropna().iloc[0]
                                             if s.notna().any() else ""))
    out["holiday"] = names.fillna("").astype(str)
    out["working"] = (out["date"].dt.dayofweek < 5) & (out["holiday"] == "")
    return out


def load(name: str) -> DayWindows:
    """The day windows of ``bike`` or ``traffic``, from the verified cache."""
    from .sources import load_hourly
    raw = load_hourly(name)
    if name == "bike":
        return build_windows(bike_hourly(raw), "bike", max_zero_fill=2)
    if name == "traffic":
        return build_windows(traffic_hourly(raw), "traffic", max_zero_fill=0)
    raise KeyError(name)
