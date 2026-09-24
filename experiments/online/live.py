"""Watch LR-DSR cluster a stream in real time.

    python -m experiments.online.live                  # a window, if there is a display
    python -m experiments.online.live --save           # figures/online_live.gif
    python -m experiments.online.live --save out.gif --windows 120 --seed 23

Two laws are learned from 40 unlabelled windows; then windows arrive one at
a time. Each is assigned the moment it arrives and folded into its law, so
the curves on the left move as the stream goes. After 60 windows a third
law starts to appear: its first windows are grey (no law explains them, they
wait in the novelty buffer), and a few windows later a third regime is born
for them. The right panel is the timeline, the truth above the assignment.

The animation *is* the run: the model is advanced inside the animation
callback (``lrdsr.viz.animate_stream``), not replayed from a log.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from lrdsr import paths


def _stream(seed: int, n_windows: int):
    from experiments.online.streams import LAWS, make_windows, newcomer_law, sigma_for

    rng = np.random.default_rng(seed)
    sigma = sigma_for(1.0)
    laws = (LAWS[0], LAWS[1], newcomer_law(6.0))
    zh = rng.integers(0, 2, 40)
    Xh, yh = make_windows(zh, sigma, rng, laws=laws)
    appear = min(60, n_windows // 2)
    z = np.r_[rng.integers(0, 2, appear), rng.integers(0, 3, n_windows - appear)]
    X, y = make_windows(z, sigma, rng, laws=laws)
    return Xh, yh, X, y, z


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--save", nargs="?", const=str(paths.ROOT / "figures" / "online_live.gif"),
                    default=None, help="write a GIF instead of (or when there is no) display")
    ap.add_argument("--windows", type=int, default=150)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--fps", type=int, default=8)
    a = ap.parse_args(argv)

    import matplotlib
    headless = sys.platform != "darwin" and not os.environ.get("DISPLAY")
    if a.save or headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from lrdsr import viz
    from lrdsr.core.online import OnlineLRDSR

    viz.style()
    Xh, yh, X, y, z = _stream(a.seed, a.windows)
    model = OnlineLRDSR(feature_names=["x"]).warm_start(Xh, yh, n_clusters=2,
                                                       random_state=a.seed)
    anim = viz.animate_stream(model, X, y, truth=z, x_range=(-2, 2),
                              interval=int(1000 / a.fps))
    if a.save or headless:
        from matplotlib.animation import PillowWriter
        out = a.save or str(paths.ROOT / "figures" / "online_live.gif")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        anim.save(out, writer=PillowWriter(fps=a.fps), dpi=80)
        print(f"wrote {out}  ({model.n_clusters} regimes at the end)")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
