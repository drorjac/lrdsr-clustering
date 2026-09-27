"""Kelmarsh wind farm SCADA: download, verify, read the columns the study uses.

Kelmarsh (Northamptonshire, UK), six Senvion MM92 turbines, 2050 kW rated,
10-minute SCADA exported by Greenbyte; Plumley (2022), Zenodo record 5841834,
doi:10.5281/zenodo.5841834, CC-BY-4.0.

The yearly archives are cached in ``.cache/wind/`` (gitignored) and their
sha256 recorded in ``checksums.json`` beside this file on first download,
then verified on every load -- the ``experiments/realdata/sources.py``
pattern.

What the archives do **not** contain, checked before this study was
designed: usable curtailment, derating or icing labels. The per-cause
"Lost Production to Curtailment" columns are zero throughout 2016, 2017 and
2021; the power setpoint is empty before 2021 and afterwards a controller
reference that never caps output; the status log records one icing stop in
two years. The study is therefore scored against physical proxies
(``windows.py``), not operator labels.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from lrdsr import paths

CHECKSUMS = Path(__file__).with_name("checksums.json")
RECORD = "https://zenodo.org/records/5841834/files/{file}?download=1"
ARCHIVES = {"2016": "Kelmarsh_SCADA_2016_3082.zip", "2017": "Kelmarsh_SCADA_2017_3083.zip"}
STATIC = "Kelmarsh_WT_static.csv"
RATED_KW = 2050.0
ROTOR_M = 92.0
#: commercial operation began 2016-04-15 (the static file); earlier rows are
#: commissioning and are not used
START = pd.Timestamp("2016-04-15")

COLUMNS = {
    "# Date and time": "time",
    "Wind speed (m/s)": "ws",
    "Wind speed, Standard deviation (m/s)": "ws_sd",
    "Wind direction (°)": "direction",
    "Power (kW)": "power",
    "Nacelle ambient temperature (°C)": "temp",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(file: str) -> Path:
    path = paths.cache_dir("wind") / file
    url = RECORD.format(file=file)
    if not path.exists():
        try:
            with urllib.request.urlopen(url, timeout=1800) as r:
                path.write_bytes(r.read())
        except OSError as exc:
            raise FileNotFoundError(f"{file}: not cached and the download failed ({exc}); "
                                    f"fetch {url} into {path}") from exc
    known = json.loads(CHECKSUMS.read_text()) if CHECKSUMS.exists() else {}
    digest = _sha256(path)
    if file not in known:
        known[file] = {"url": url, "sha256": digest, "bytes": path.stat().st_size}
        CHECKSUMS.write_text(json.dumps(dict(sorted(known.items())), indent=2) + "\n")
    elif known[file]["sha256"] != digest:
        raise ValueError(f"{file}: sha256 {digest} does not match checksums.json; "
                         f"delete it and refetch, and update checksums.json knowingly")
    return path


def turbines() -> pd.DataFrame:
    """The static table: one row per turbine with position, hub height, rated power."""
    s = pd.read_csv(fetch(STATIC), encoding="utf-8-sig")
    s["turbine"] = np.arange(1, len(s) + 1)
    return s


def load_scada() -> pd.DataFrame:
    """Every 10-minute row the study uses, all turbines, from ``START`` on."""
    frames = []
    for file in ARCHIVES.values():
        with zipfile.ZipFile(fetch(file)) as z:
            for n in sorted(m for m in z.namelist() if m.startswith("Turbine_Data")):
                d = pd.read_csv(z.open(n), skiprows=9, usecols=list(COLUMNS))
                d = d.rename(columns=COLUMNS)
                d["turbine"] = int(n.split("_")[3])
                frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["time"] = pd.to_datetime(d["time"])
    return d[d["time"] >= START].sort_values(["turbine", "time"]).reset_index(drop=True)
