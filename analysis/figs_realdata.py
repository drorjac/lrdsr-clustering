"""The real-data figures, read from ``results/realdata/``.

Three pictures: what the two daily laws are, how every method does against
the calendar, and the real-time pass laid out over the calendar.

    python -m analysis.figs_realdata      # write them to figures/
"""
from __future__ import annotations

import matplotlib as mpl

if __name__ == "__main__":
    mpl.use("Agg")       # headless only as a script; notebooks keep inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.plots import OUT, RES, C, W, style

DATASET_TITLE = {"bike": "Bike Sharing (DC, 2011-12)",
                 "traffic": "I-94 traffic (MN, 2012-18)"}
REGIME_COLOR = {"working": C["mech"], "off": C["feat"]}


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RES / "realdata" / name)


def realdata_laws():
    """The recovered daily law of each regime, over the calendar-mean profile.

    Solid: the law LR-DSR fitted to each cluster (library basis); dotted:
    the soft-EM law on the Fourier basis; grey band: the mean centred
    profile of the calendar's working days and days off. The clusters are
    named by the day type most of their days have -- a name, not a label
    the fit saw.
    """
    style()
    laws = _csv("realdata_laws.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W + 1.2, 3.1), sharey=True)
    for ax, (ds, g) in zip(axes, laws.groupby("dataset", sort=False), strict=False):
        for regime, col in REGIME_COLOR.items():
            cal = g[(g.method == "calendar_mean") & (g.regime == regime)]
            ax.plot(cal.hour, cal.value, color=col, lw=6, alpha=0.18,
                    label=f"calendar mean, {regime}")
            for method, ls in (("lrdsr_alpha0", "-"), ("soft_fourier", ":")):
                m = g[(g.method == method) & (g.regime == regime)]
                ax.plot(m.hour, m.value, ls, color=col, lw=1.7,
                        label=f"{'LR-DSR' if method.startswith('lrdsr') else 'soft EM, Fourier'}"
                              f", {regime}")
        ax.set(title=DATASET_TITLE.get(ds, ds), xlabel="hour of day",
               xticks=range(0, 25, 6))
    axes[0].set_ylabel("centred log1p(count)")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return fig


def realdata_methods():
    """ARI against the calendar day type, every method, both series (mean of 3 seeds)."""
    style()
    s = _csv("realdata_summary.csv")
    s = s[~s.method.str.startswith("geom_best")]
    fam_col = {"LR-DSR": C["mech"], "soft EM": "#4a7a4a", "mechanism K-means": "#7b5ea7",
               "geometry": C["geom"], "raw profile": C["oracle"]}
    fig, axes = plt.subplots(1, 2, figsize=(W + 1.2, 3.6), sharex=True)
    for ax, (ds, g) in zip(axes, s.groupby("dataset", sort=False), strict=False):
        g = g.sort_values("ARI_mean")
        y = np.arange(len(g))
        ax.barh(y, g.ARI_mean, xerr=g.ARI_sd.fillna(0), color=[fam_col[f] for f in g.family],
                height=0.7, error_kw={"lw": 0.8})
        ax.set_yticks(y)
        ax.set_yticklabels([m.replace("geom_", "") for m in g.method], fontsize=7)
        ax.set(title=DATASET_TITLE.get(ds, ds), xlabel="ARI vs calendar day type",
               xlim=(0, 1))
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in fam_col.values()]
    axes[1].legend(handles, list(fam_col), fontsize=7, loc="lower right")
    fig.tight_layout()
    return fig


def realdata_online():
    """The real-time pass over the calendar: assigned regime against day type.

    One row per series (``novelty_alpha = 1e-3``). Top strip: the calendar
    day type; bottom strip: the regime each day got **the moment it
    arrived**, grey while held in the novelty buffer; vertical lines mark
    regime births. The line is the 30-day rolling accuracy.
    """
    style()
    tl = _csv("realdata_online.csv")
    tl = tl[tl.novelty_alpha == tl.novelty_alpha.max()]
    pal = [C["mech"], C["feat"], "#4a7a4a", "#7b5ea7", "#b03a2e", "#2a9d8f"]
    fig, axes = plt.subplots(2, 1, figsize=(W + 1.2, 4.4))
    for ax, (ds, g) in zip(axes, tl.groupby("dataset", sort=False), strict=False):
        dates = pd.to_datetime(g["date"])
        truth_col = [REGIME_COLOR["working"] if t == 1 else REGIME_COLOR["off"]
                     for t in g["truth"]]
        lab = g["label"].astype(int)
        # colour an online cluster by the day type it maps to, births distinct
        lab_col = []
        for k, a in zip(lab, g["as_daytype"], strict=True):
            if k < 0:
                lab_col.append("#9a9a9a")
            elif k <= 1:
                lab_col.append(REGIME_COLOR["working"] if a == 1 else REGIME_COLOR["off"])
            else:
                lab_col.append(pal[k % len(pal)])
        ax.bar(dates, 1.0, bottom=1.15, width=1.0, color=truth_col, lw=0)
        ax.bar(dates, 1.0, bottom=0.0, width=1.0, color=lab_col, lw=0)
        ax.plot(dates, g["rolling_accuracy_30"] - 1.25, color=C["oracle"], lw=0.8)
        for d in pd.to_datetime(g.loc[g["spawned"].notna(), "date"]):
            ax.axvline(d, color=C["oracle"], ls=":", lw=1.1)
        acc = g["correct"].mean()
        ax.set(title=f"{DATASET_TITLE.get(ds, ds)} -- streamed, accuracy {acc:.3f}",
               ylim=(-1.3, 2.2))
        ax.set_yticks([-0.75, 0.5, 1.65])
        ax.set_yticklabels(["30-day acc.\n(0.5..1)", "assigned", "calendar"], fontsize=7.5)
        ax.spines["left"].set_visible(False)
    fig.tight_layout()
    return fig


FIGURES = {"realdata_laws": realdata_laws, "realdata_methods": realdata_methods,
           "realdata_online": realdata_online}


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
