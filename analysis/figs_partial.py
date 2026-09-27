"""V11 and the streams with gaps, read from ``results/realdata`` and ``results/online``.

    python -m analysis.figs_partial      # write them to figures/
"""
from __future__ import annotations

import matplotlib as mpl

if __name__ == "__main__":
    mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.plots import OUT, RES, C, W, style

DS = {"traffic": ("I-94 traffic", C["mech"], "o"), "bike": ("bike sharing", C["feat"], "s")}


def _csv(block: str, name: str) -> pd.DataFrame:
    return pd.read_csv(RES / block / name)


def v11_which_hours(k: int = 12):
    """V11: which hours are missing matters, and the laws predict which.

    A block of ``k`` consecutive hours removed from every held-out complete
    day, at each start hour. Lines: the error V11 predicts from the laws and
    the residual covariance of the training days; markers: the observed
    disagreement with the day's full-day decision (3 seeds).
    """
    style()
    b = _csv("realdata", "realdata_partial_blocks.csv")
    b = b[b["mask_label"].str.startswith(f"{k}h@")].copy()
    b["start"] = b["mask_label"].str.split("@").str[1].astype(int)
    fig, ax = plt.subplots(figsize=(W, 3.0))
    for ds, (lab, col, mk) in DS.items():
        g = b[b.dataset == ds].sort_values("start")
        ax.plot(g.start, g.predicted, "-", color=col, label=f"{lab}, V11 predicted")
        ax.plot(g.start, g.observed, mk, color=col, mfc="white", ls="none",
                label=f"{lab}, observed")
    ax.set(xlabel=f"first missing hour (a {k}-hour block is missing)",
           ylabel="error vs full-day decision", xticks=range(0, 24, 3))
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    return fig


def v11_early_decision():
    """A day read hour by hour: share decided at 99% posterior, predicted vs observed."""
    style()
    p = _csv("realdata", "realdata_partial_prefix.csv")
    p["hours"] = p["mask_label"].astype(int)
    fig, ax = plt.subplots(figsize=(W, 3.0))
    for ds, (lab, col, mk) in DS.items():
        g = p[p.dataset == ds].sort_values("hours")
        ax.plot(g.hours, g.pred_decided_right, "-", color=col, label=f"{lab}, predicted")
        ax.plot(g.hours, g.decided_right, mk, color=col, mfc="white", ls="none",
                label=f"{lab}, observed")
    ax.set(xlabel="hours observed since midnight", ylabel="share of days decided\n"
           "correctly at 99% confidence", xlim=(1.5, 12.5), ylim=(-0.02, 1.02))
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    return fig


def online_kernel_birth():
    """A newcomer law the library cannot represent: births, by basis and separation."""
    style()
    kb = _csv("online", "online_ragged_kernel_birth.csv")
    g = kb[~kb.control].groupby(["rho", "basis"]).agg(
        born=("births", lambda s: np.mean(s > 0)), power=("predicted_power", "first"))
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for basis, col, mk in (("kernel", C["mech"], "o"), ("library", C["feat"], "s")):
        h = g.xs(basis, level="basis")
        ax.plot(h.index, h.born, "-", marker=mk, color=col,
                label=f"{'Nystrom basis' if basis == 'kernel' else 'term library'}")
    pw = g.xs("kernel", level="basis").power
    ax.plot(pw.index, pw.values, ":", color=C["oracle"],
            label="per-window power of the test, laws known")
    ax.set(xscale="log", xlabel="separation of the newcomer, rho",
           ylabel="share of streams with a birth", ylim=(-0.05, 1.05))
    ax.set_xticks(list(pw.index))
    ax.set_xticklabels([f"{r:g}" for r in pw.index])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    return fig


FIGURES = {"v11_which_hours": v11_which_hours, "v11_early_decision": v11_early_decision,
           "online_kernel_birth": online_kernel_birth}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGURES.items():
        fig = fn()
        fig.savefig(OUT / f"{name}.png")
        plt.close(fig)
        print(f"  [fig] figures/{name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
