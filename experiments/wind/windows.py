"""Turbine blocks as windows, and the physical proxies they are scored against.

One window is one turbine over one 6-hour block (up to 36 ten-minute
samples): ``x`` the nacelle wind speed (m/s), ``y`` the power as a share of
rated. Every window has **its own design** -- the wind speeds it happened to
see -- which is the setting the method is for and the one in which no raw
profile exists: a power curve per block has to be binned (the IEC 61400-12
method of bins, ``bin_profile``) before a profile method can use it.

Rules, fixed before any fit:

* a sample is dropped when wind speed or power is missing, and as
  **downtime** when the wind is above cut-in (4.5 m/s) but the turbine makes
  under 10 kW -- a stopped turbine is not a power curve;
* a block is kept with at least ``MIN_SAMPLES`` samples left.

The proxies, each from covariates the fit never sees:

==============  =========================================================
``cold``        block mean ambient temperature below the median over all
                blocks. Physics: below rated, power scales with air
                density, ~1/T -- but temperature also tracks season, so
                this proxy carries more than density (measured, not assumed)
``waked``       the wind comes from a neighbouring turbine less than six
                rotor diameters away (+/- 15 deg): 1 if >= 70% of the
                block's samples are waked, 0 if <= 5%, else unscored
``night``       00-06 h (1) against 12-18 h (0) UTC; the other blocks are
                unscored. The raw effect is ~1-2%: a negative control
==============  =========================================================

Turbulence intensity was planned as a proxy and dropped: the anemometer
standard-deviation column is missing for 84% of the samples.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from experiments.wind.sources import RATED_KW, ROTOR_M, load_scada, turbines

BLOCK_H = 6
MIN_SAMPLES = 24
CUT_IN, DOWN_KW = 4.5, 10.0
WAKE_DIAMETERS, WAKE_HALF_ANGLE = 6.0, 15.0
BIN_EDGES = np.arange(3.0, 15.01, 0.5)          # the method of bins, 0.5 m/s
PROXIES = ("cold", "waked", "night")


@dataclass
class Blocks:
    """Ragged windows ``X`` (list of ``(n_w, 1)``), ``y`` (list), and ``meta``."""
    X: list
    y: list
    meta: pd.DataFrame

    def labels(self, proxy: str) -> np.ndarray:
        """The proxy as 0/1, with -1 where the block is not scored for it."""
        return self.meta[proxy].to_numpy()


def _waked(d: pd.DataFrame) -> np.ndarray:
    t = turbines()
    lat0 = np.deg2rad(t["Latitude"].mean())
    east = (t["Longitude"] - t["Longitude"].mean()).to_numpy() * 111320 * np.cos(lat0)
    north = (t["Latitude"] - t["Latitude"].mean()).to_numpy() * 110540
    out = np.zeros(len(d), bool)
    turb = d["turbine"].to_numpy()
    direction = d["direction"].to_numpy()
    for i in range(len(t)):
        for j in range(len(t)):
            dx, dy = east[j] - east[i], north[j] - north[i]
            if i == j or np.hypot(dx, dy) > WAKE_DIAMETERS * ROTOR_M:
                continue
            bearing = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
            diff = np.abs((direction - bearing + 180) % 360 - 180)
            out |= (turb == i + 1) & (diff < WAKE_HALF_ANGLE)
    return out


def build(d: pd.DataFrame | None = None) -> Blocks:
    d = load_scada() if d is None else d
    d = d.dropna(subset=["ws", "power"]).copy()
    d = d[~((d["ws"] > CUT_IN) & (d["power"] < DOWN_KW))]
    d["waked"] = _waked(d) & d["direction"].notna().to_numpy()
    d["block"] = d["time"].dt.floor(f"{BLOCK_H}h")
    X, y, rows = [], [], []
    for (turb, blk), g in d.groupby(["turbine", "block"], sort=True):
        if len(g) < MIN_SAMPLES:
            continue
        g = g.sort_values("ws")
        X.append(g["ws"].to_numpy()[:, None])
        y.append((g["power"] / RATED_KW).to_numpy())
        share = float(g["waked"].mean())
        rows.append({"turbine": int(turb), "block": blk, "hour": blk.hour,
                     "month": blk.month, "n": len(g), "temp": float(g["temp"].mean()),
                     "ws_mean": float(g["ws"].mean()),
                     "coverage": int(np.unique(np.digitize(g["ws"], BIN_EDGES)).size),
                     "waked_share": share})
    meta = pd.DataFrame(rows)
    med = meta["temp"].median()
    meta["cold"] = np.where(meta["temp"].isna(), -1, (meta["temp"] < med).astype(int))
    meta["waked"] = np.where(meta["waked_share"] >= 0.7, 1,
                             np.where(meta["waked_share"] <= 0.05, 0, -1))
    meta["night"] = np.where(meta["hour"] == 0, 1, np.where(meta["hour"] == 12, 0, -1))
    return Blocks(X, y, meta)


def bin_profile(X: list, y: list) -> np.ndarray:
    """The method of bins per window: mean power per 0.5 m/s bin on
    ``BIN_EDGES``, empty bins linearly interpolated in wind speed, flat
    beyond the observed ones -- a fixed-length profile a raw method can use."""
    centres = 0.5 * (BIN_EDGES[1:] + BIN_EDGES[:-1])
    out = np.empty((len(X), len(centres)))
    for i, (x, v) in enumerate(zip(X, y, strict=True)):
        idx = np.digitize(x[:, 0], BIN_EDGES) - 1
        ok = (idx >= 0) & (idx < len(centres))
        sums = np.bincount(idx[ok], v[ok], len(centres))
        cnt = np.bincount(idx[ok], minlength=len(centres))
        have = cnt > 0
        if have.sum() == 0:
            out[i] = np.mean(v)
        elif have.sum() == 1:
            out[i] = (sums / np.maximum(cnt, 1))[have][0]
        else:
            out[i] = np.interp(centres, centres[have], sums[have] / cnt[have])
    return out
