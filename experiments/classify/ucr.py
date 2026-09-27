"""The UCR time-series archive, as windows: download, verify, parse.

Each series is one window: ``x`` its time index on ``[0, 1]`` (or ``2 pi t/n``
for the library, the realdata convention), ``y`` its values, the class the
archive's label. A series with missing values keeps only its observed
samples -- a **ragged** window, which the law classifier scores as is and a
raw-profile method has to interpolate first.

The datasets are chosen in two groups, fixed before any result was seen:

``daily``  a count or load over a daily (or weekly) cycle, the class a
           kind of day, place or appliance: the same kind of data as
           ``experiments/realdata`` and the setting the method is built for
``shape``  classic shape benchmarks (motion, ECG, spectra, simulated
           patterns), where a class is a *pattern* with phase jitter rather
           than a law -- the contrast, where no low-rank law should be
           expected to win

Archives come from timeseriesclassification.com and are cached in
``.cache/ucr/``; their sha256 is recorded in ``checksums.json`` beside this
file on first download and checked on every later load. The published 1-NN
error rates (``DataSummary.csv`` of the UCR 2018 archive) are fetched the same
way and used as reference rows only.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lrdsr import paths

CHECKSUMS = Path(__file__).with_name("checksums.json")
URL = "https://www.timeseriesclassification.com/aeon-toolkit/{name}.zip"
SUMMARY_URL = "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/DataSummary.csv"

DAILY = ("Chinatown", "ItalyPowerDemand", "MelbournePedestrian", "PowerCons",
         "DodgerLoopDay", "DodgerLoopWeekend", "DodgerLoopGame",
         "ElectricDevices", "SmallKitchenAppliances")
SHAPE = ("GunPoint", "ECG200", "Trace", "SyntheticControl", "CBF", "TwoPatterns",
         "Coffee", "Wafer", "Plane", "ArrowHead", "FaceFour", "Lightning2",
         "Lightning7", "SonyAIBORobotSurface1", "MoteStrain")
DATASETS = DAILY + SHAPE
GROUP = {**dict.fromkeys(DAILY, "daily"), **dict.fromkeys(SHAPE, "shape")}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(key: str, url: str, file: str) -> Path:
    path = paths.cache_dir("ucr") / file
    if not path.exists():
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                path.write_bytes(r.read())
        except OSError as exc:
            raise FileNotFoundError(
                f"{key}: {path.relative_to(paths.ROOT)} is not cached and the download "
                f"failed ({exc}). Fetch {url} by hand into that path.") from exc
    known = json.loads(CHECKSUMS.read_text()) if CHECKSUMS.exists() else {}
    digest = _sha256(path)
    if key not in known:
        known[key] = {"url": url, "sha256": digest, "bytes": path.stat().st_size}
        CHECKSUMS.write_text(json.dumps(dict(sorted(known.items())), indent=2) + "\n")
    elif known[key]["sha256"] != digest:
        raise ValueError(f"{key}: {path} has sha256 {digest}, checksums.json records "
                         f"{known[key]['sha256']}; delete it and refetch, and update "
                         f"checksums.json knowingly if upstream changed.")
    return path


def parse_ts(text: str) -> tuple[list[np.ndarray], np.ndarray]:
    """A univariate ``.ts`` file -> (series with NaN for missing, labels)."""
    series, labels, in_data = [], [], False
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("@data"):
            in_data = True
            continue
        if not in_data or line.startswith("@"):
            continue
        body, label = line.rsplit(":", 1)
        vals = [np.nan if v in ("?", "NaN", "nan", "") else float(v)
                for v in body.split(",")]
        series.append(np.asarray(vals))
        labels.append(label.strip())
    return series, np.asarray(labels)


@dataclass
class UCRSplit:
    """One dataset: series (NaN = missing) and labels, train and test."""
    name: str
    train: list[np.ndarray]
    y_train: np.ndarray
    test: list[np.ndarray]
    y_test: np.ndarray

    @property
    def group(self) -> str:
        return GROUP[self.name]

    @property
    def length(self) -> int:
        return max(len(s) for s in self.train + self.test)

    @property
    def n_classes(self) -> int:
        return len(np.unique(np.r_[self.y_train, self.y_test]))

    @property
    def missing_share(self) -> float:
        allv = np.concatenate(self.train + self.test)
        return float(np.isnan(allv).mean())


def load(name: str) -> UCRSplit:
    path = _fetch(name, URL.format(name=name), f"{name}.zip")
    with zipfile.ZipFile(path) as z:
        tr = parse_ts(z.read(f"{name}_TRAIN.ts").decode("utf-8", "replace"))
        te = parse_ts(z.read(f"{name}_TEST.ts").decode("utf-8", "replace"))
    classes = np.unique(np.r_[tr[1], te[1]])
    code = {c: i for i, c in enumerate(classes)}
    return UCRSplit(name, tr[0], np.array([code[c] for c in tr[1]]),
                    te[0], np.array([code[c] for c in te[1]]))


def published() -> pd.DataFrame:
    """UCR 2018 ``DataSummary.csv``: published 1-NN ED and DTW error rates."""
    path = _fetch("DataSummary", SUMMARY_URL, "DataSummary.csv")
    d = pd.read_csv(path, encoding="utf-8-sig")
    d.columns = [c.strip() for c in d.columns]
    out = pd.DataFrame({
        "dataset": d["Name"].str.strip(),
        "published_ed_error": pd.to_numeric(d["ED (w=0)"], errors="coerce"),
        "published_dtw_error": pd.to_numeric(
            d["DTW (learned_w)"].astype(str).str.split().str[0], errors="coerce"),
    })
    return out[out.dataset.isin(DATASETS)].reset_index(drop=True)


# ==========================================================================
# series -> windows
# ==========================================================================
def to_windows(series: list[np.ndarray], angle: bool = False):
    """Ragged windows: ``(X list of (n_w, 1), y list of (n_w,))``, observed samples only.

    ``x = t / (n - 1)`` on ``[0, 1]``, or ``2 pi t / n`` with ``angle=True``.
    """
    Xs, ys = [], []
    for s in series:
        n = len(s)
        t = np.arange(n, dtype=float)
        x = 2 * np.pi * t / n if angle else t / max(n - 1, 1)
        ok = np.isfinite(s)
        Xs.append(x[ok][:, None])
        ys.append(s[ok].astype(float))
    return Xs, ys


def to_grid(series: list[np.ndarray], length: int | None = None) -> np.ndarray:
    """The raw-profile view: every series on a common grid, gaps linearly
    interpolated (constant beyond the ends) -- what a profile method needs."""
    L = length or max(len(s) for s in series)
    grid = np.linspace(0.0, 1.0, L)
    out = np.empty((len(series), L))
    for i, s in enumerate(series):
        t = np.linspace(0.0, 1.0, len(s))
        ok = np.isfinite(s)
        out[i] = np.interp(grid, t[ok], s[ok]) if ok.sum() >= 2 else np.nanmean(s)
    return out


def subsample(series: list[np.ndarray], keep: float, seed: int) -> list[np.ndarray]:
    """Irregular sampling: each series keeps its own random ``keep`` share of
    its observed samples (at least 3), the rest set missing."""
    rng = np.random.default_rng(seed)
    out = []
    for s in series:
        obs = np.flatnonzero(np.isfinite(s))
        k = max(3, int(round(keep * len(obs))))
        sel = np.sort(rng.choice(obs, size=min(k, len(obs)), replace=False))
        t = np.full(len(s), np.nan)
        t[sel] = s[sel]
        out.append(t)
    return out


def describe(split: UCRSplit) -> dict:
    return {"dataset": split.name, "group": split.group, "length": split.length,
            "n_train": len(split.train), "n_test": len(split.test),
            "n_classes": split.n_classes, "missing_share": split.missing_share,
            "ragged": len({len(s) for s in split.train + split.test}) > 1}

