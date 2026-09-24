"""Where the real data comes from, and proof it is the same data next time.

Two public hourly count series, both with a day-type structure that is
plausibly a *change of law* rather than a change of level: a commuting day
has two rush-hour peaks, a weekend day has one broad afternoon hump.

==========  ================================================================
``bike``    UCI Bike Sharing (Capital Bikeshare, Washington DC, 2011-2012),
            ``hour.csv``: hourly rentals with the dataset's own
            ``workingday`` / ``holiday`` flags and weather.
``traffic`` UCI Metro Interstate Traffic Volume (I-94 westbound, Minneapolis
            - St Paul, 2012-2018): hourly vehicle counts with a ``holiday``
            name and weather.
==========  ================================================================

The archives are downloaded once into ``.cache/realdata/`` (gitignored). The
first download records each archive's sha256 in ``checksums.json`` next to
this file, which *is* committed; every later load verifies against it, so a
silently changed upstream file fails loudly instead of moving the results.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from lrdsr import paths

CHECKSUMS = Path(__file__).with_name("checksums.json")

SOURCES = {
    "bike": {
        "url": "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
        "file": "bike_sharing.zip",
        "member": "hour.csv",
        "cite": "Fanaee-T & Gama (2013), UCI ML Repository, doi:10.24432/C5W894",
    },
    "traffic": {
        "url": "https://archive.ics.uci.edu/static/public/492/"
               "metro+interstate+traffic+volume.zip",
        "file": "metro_traffic.zip",
        "member": "Metro_Interstate_Traffic_Volume.csv.gz",
        "cite": "Hogue (2019), UCI ML Repository, doi:10.24432/C5X60B",
    },
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _known() -> dict:
    return json.loads(CHECKSUMS.read_text()) if CHECKSUMS.exists() else {}


def raw_path(name: str) -> Path:
    """Where the archive for ``name`` lives in the cache (it may not exist)."""
    return paths.CACHE_ROOT / "realdata" / SOURCES[name]["file"]


def is_cached(name: str) -> bool:
    return raw_path(name).exists()


def fetch(name: str) -> Path:
    """The verified local archive for ``name``, downloading it if needed."""
    if name not in SOURCES:
        raise KeyError(f"unknown source {name!r}; expected one of {list(SOURCES)}")
    src = SOURCES[name]
    path = paths.cache_dir("realdata") / src["file"]
    if not path.exists():
        try:
            with urllib.request.urlopen(src["url"], timeout=60) as r:
                data = r.read()
        except OSError as exc:
            raise FileNotFoundError(
                f"{name}: {path.relative_to(paths.ROOT)} is not cached and the "
                f"download failed ({exc}).\nFetch it by hand from\n  {src['url']}\n"
                f"and save it at that path, then rerun."
            ) from exc
        path.write_bytes(data)
    digest = _sha256(path)
    known = _known()
    if name not in known:
        known[name] = {"url": src["url"], "sha256": digest,
                       "bytes": path.stat().st_size}
        CHECKSUMS.write_text(json.dumps(known, indent=2) + "\n")
    elif known[name]["sha256"] != digest:
        raise ValueError(
            f"{name}: {path} has sha256 {digest}, but checksums.json records "
            f"{known[name]['sha256']}. The upstream file changed or the download "
            f"is corrupt; delete it and refetch, and if upstream really changed, "
            f"update checksums.json knowingly -- every realdata result moves.")
    return path


def load_hourly(name: str) -> pd.DataFrame:
    """The raw hourly table for ``name``, exactly as published."""
    with zipfile.ZipFile(fetch(name)) as z, z.open(SOURCES[name]["member"]) as f:
        if SOURCES[name]["member"].endswith(".gz"):
            return pd.read_csv(gzip.open(f))
        return pd.read_csv(f)
