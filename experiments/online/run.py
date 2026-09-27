"""Clustering by law in real time: the ceiling, then the estimators.

    python -m experiments online
    python -m experiments.online.live           # watch one stream being clustered

======  =================================================================
V9      how fast a switch between two laws can be detected sample by
        sample: CUSUM delay and false-alarm run length against Lorden's
        first-order law, Siegmund's Brownian approximation, and the same
        approximation in the log-likelihood ratio's exact tilt
stream  ``OnlineLRDSR`` window by window -- stationary, a regime born
        mid-stream, a drifting law, latency -- and ``CusumSegmenter`` on
        one sample stream with learned laws (``experiments.online.streams``)
======  =================================================================

Writes ``results/online/``.
"""
from __future__ import annotations

import time

from experiments.online import streams
from lrdsr.theory import sequential


def run(args=None) -> None:
    t0 = time.time()
    print("\n=== V9: the real-time ceiling ===", flush=True)
    sequential.run_v9()
    print(f"  [V9] {time.time() - t0:.0f} s", flush=True)
    print("\n=== streams ===", flush=True)
    streams.run()
    print("\n=== ragged windows and a kernel basis ===", flush=True)
    from experiments.online import ragged
    ragged.run()
    print(f"\n[online] CSVs -> {streams.RESULTS}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    run()
