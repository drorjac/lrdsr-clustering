"""The wind power-curve figures, read from ``results/wind/``.

    python -m analysis.figs_wind      # write them to figures/
"""
from __future__ import annotations

import matplotlib as mpl

if __name__ == "__main__":
    mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.plots import OUT, RES, C, W, style

PROXY_TITLE = {"cold": "cold vs warm", "waked": "waked vs free", "night": "night vs day\n(control)"}
ARMS = (("law", "law classifier", C["mech"]),
        ("mech_logistic", "mechanism features + logistic", "#7b5ea7"),
        ("bins_1nn", "method of bins + 1-NN", C["oracle"]),
        ("bins_logistic", "method of bins + logistic", C["geom"]))


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RES / "wind" / name)


def wind_transfer():
    """Balanced error on turbines never trained on, per proxy (mean of both
    directions); the dash is V1's ceiling for a clean two-law proxy."""
    style()
    t = _csv("wind_transfer.csv")
    m = t.groupby(["proxy", "method"]).agg(err=("balanced_error", "mean"),
                                           ceil=("ceiling_v1", "mean"))
    fig, ax = plt.subplots(figsize=(W, 3.2))
    x = np.arange(len(PROXY_TITLE))
    w = 0.19
    for i, (meth, lab, col) in enumerate(ARMS):
        v = [m.loc[(p, meth), "err"] for p in PROXY_TITLE]
        ax.bar(x + (i - 1.5) * w, v, w * 0.9, color=col, label=lab)
    for j, p in enumerate(PROXY_TITLE):
        ax.plot([j - 2 * w, j + 2 * w], [m.loc[(p, "law"), "ceil"]] * 2, color=C["bad"],
                lw=1.6, ls="--", zorder=5, label="V1 ceiling" if j == 0 else None)
    ax.axhline(0.5, color=C["geom"], lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(list(PROXY_TITLE.values()))
    ax.set(ylabel="balanced error, held-out turbines", ylim=(0, 0.55))
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    return fig


def wind_physics():
    """Cold over warm power at the same wind speed, against the air-density ratio.

    Binned means of every sample (the physics reference), the law with
    quantile-placed Nystrom centres (used everywhere else) and with evenly
    spaced ones; the flat line is the density ratio the block temperatures
    imply at one pressure.
    """
    style()
    p = _csv("wind_physics.csv")
    fig, ax = plt.subplots(figsize=(W, 3.0))
    ax.plot(p.ws, p.bins_ratio, "o", color=C["oracle"], mfc="white", label="binned means")
    ax.plot(p.ws, p.law_ratio, "-", color=C["feat"], label="law, quantile centres")
    ax.plot(p.ws, p.law_uniform_ratio, "-", color=C["mech"], label="law, even centres")
    ax.axhline(p.density_ratio.iloc[0], color=C["geom"], lw=2,
               label=f"air density ratio ({p.density_ratio.iloc[0]:.3f})")
    ax.set(xlabel="wind speed (m/s)", ylabel="cold / warm power", ylim=(0.95, 1.3))
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    return fig


FIGURES = {"wind_transfer": wind_transfer, "wind_physics": wind_physics}


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
