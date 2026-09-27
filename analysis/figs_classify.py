"""The classification figures, read from ``results/classify/``.

    python -m analysis.figs_classify      # write them to figures/
"""
from __future__ import annotations

import matplotlib as mpl

if __name__ == "__main__":
    mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analysis.plots import OUT, RES, C, W, style

P_STYLE = {3: ("o", ":"), 7: ("s", "--"), 15: ("^", "-."), 31: ("D", "-")}
#: label offsets (points) where two dataset names would collide
NUDGE = {"GunPoint": (8, 3), "MelbournePedestrian": (-40, 9)}
GROUP_TITLE = {"daily": "daily-cycle data (9 datasets)",
               "shape": "shape benchmarks (15 datasets)"}


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RES / "classify" / name)


def v10_learning_curve(D2: float = 9.0):
    """V10: the plug-in error against labelled windows per class, ``n rho = 9``.

    Markers: the ``LawClassifier`` simulated (120 training sets per point,
    two-standard-error bars). Lines: the exact three-scalar form. Grey: the
    ceiling ``Q(sqrt(n rho)/2)``. Left, a fixed design; right, a random one,
    where the prediction carries the two corrections of
    ``lrdsr.theory.classification`` (design cost ``(p+1)/n``, oracle's Jensen gap).
    """
    style()
    d = _csv("v10_learning_curve.csv")
    d = d[d.D2 == D2]
    fig, axes = plt.subplots(1, 2, figsize=(W + 0.6, 3.1), sharey=True)
    for ax, design in zip(axes, ("fixed", "random"), strict=True):
        g = d[d.design == design]
        for p, (mk, ls) in P_STYLE.items():
            h = g[g.p == p].sort_values("m")
            ax.errorbar(h.m, h.simulated, yerr=h.band, fmt=mk, color=C["mech"],
                        mfc="white", ms=5, capsize=0, lw=0.8)
            ax.plot(h.m, h.predicted, ls, color=C["mech"], lw=1.2, label=f"p = {p}")
        ax.axhline(g.ceiling.iloc[0], color=C["geom"], lw=2, zorder=0,
                   label="ceiling (laws known)")
        ax.set(xscale="log", xlabel="labelled windows per class, m",
               title=f"{design} design", ylim=(0, 0.52))
        ax.set_xticks([1, 2, 4, 8, 32])
        ax.set_xticklabels(["1", "2", "4", "8", "32"])
    axes[0].set_ylabel("classification error")
    axes[0].legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    return fig


def classify_benchmark():
    """24 UCR datasets: the law classifier against the best raw-profile baseline.

    One point per dataset, test error on the archive's own split. x: the best
    law classifier (cosine or Nystrom basis, hyperparameters by CV on train);
    y: the best raw-profile classifier (1-NN, centroid, logistic, SVM, forest,
    each tuned on train). Above the diagonal the law wins. Best-of is taken on
    each side alike, so neither side is favoured by the selection.
    """
    style()
    b = _csv("ucr_benchmark.csv")
    piv = b.pivot_table(index=["group", "dataset"], columns="method", values="error")
    law = piv[["law_cosine", "law_nystrom"]].min(axis=1)
    raw = piv[[c for c in piv.columns if c.startswith("raw_")]].min(axis=1)
    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    for grp, mk, col in (("daily", "o", C["mech"]), ("shape", "s", C["feat"])):
        sel = law.index.get_level_values(0) == grp
        ax.plot(law[sel], raw[sel], mk, color=col, mfc=col if grp == "daily" else "white",
                label=GROUP_TITLE[grp])
        for (_g, name), x, y in zip(law[sel].index, law[sel], raw[sel], strict=True):
            if abs(x - y) > 0.03:
                ax.annotate(name, (x, y), fontsize=6.3,
                            xytext=NUDGE.get(name, (3, 2)),
                            textcoords="offset points", color="#444444")
    lim = max(law.max(), raw.max()) * 1.05
    ax.plot([0, lim], [0, lim], color=C["geom"], lw=1)
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="best law classifier, test error",
           ylabel="best raw-profile classifier, test error")
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    return fig


def classify_fewshot():
    """Few labelled series: mean test error against ``m`` per class, by group.

    ``law L=all``: nearest law (each training series smoothed by a cosine
    basis whose size was chosen without labels); ``law L=1``: one law per
    class; the raw 1-NN and nearest centroid are the same two rules on the
    full-length profile. The V10 prediction is that the law versions pay for
    the basis size and the raw ones for the series length.
    """
    style()
    f = _csv("ucr_fewshot.csv")
    arms = (("law_Lall", "law, nearest law (L = all)", C["mech"], "o", "-"),
            ("law_L1", "law, one per class (L = 1)", C["mech"], "s", "--"),
            ("raw_1nn_ed", "raw profile, 1-NN", C["oracle"], "^", "-"),
            ("raw_centroid", "raw profile, nearest centroid", C["oracle"], "v", "--"))
    fig, axes = plt.subplots(1, 2, figsize=(W + 0.6, 3.1), sharey=True)
    for ax, grp in zip(axes, ("daily", "shape"), strict=True):
        g = f[f.group == grp]
        # only datasets that have every m, so a curve is one set of datasets
        full = g.groupby("dataset").m.nunique()
        g = g[g.dataset.isin(full[full == full.max()].index)]
        for method, lab, c, mk, ls in arms:
            h = g[g.method == method].groupby("m").error.mean()
            ax.plot(h.index, h.values, ls, marker=mk, color=c, mfc="white", label=lab)
        ax.set(xscale="log", xlabel="labelled series per class, m",
               title=f"{grp}: the {g.dataset.nunique()} datasets with every m")
        ax.set_xticks([1, 2, 5, 10])
        ax.set_xticklabels(["1", "2", "5", "10"])
    axes[0].set_ylabel("mean test error")
    axes[0].legend(fontsize=7.5)
    fig.tight_layout()
    return fig


def classify_irregular():
    """Irregular sampling: each series keeps a random share of its samples.

    Mean test error by the share kept. The law classifier scores the samples
    that were kept, at their own times; the raw-profile rules and the feature
    map's logistic regression see the series after linear interpolation or
    through the mechanism map.
    """
    style()
    r = _csv("ucr_irregular.csv")
    arms = (("law_cosine", "law classifier (observed samples only)", C["mech"], "o", "-"),
            ("mech_logistic_solve", "mechanism features (solve) + logistic",
             C["feat"], "s", "--"),
            ("raw_1nn_ed", "interpolate, raw 1-NN", C["oracle"], "^", "-"),
            ("raw_centroid", "interpolate, nearest centroid", C["oracle"], "v", ":"))
    fig, axes = plt.subplots(1, 2, figsize=(W + 0.6, 3.1), sharey=True)
    for ax, grp in zip(axes, ("daily", "shape"), strict=True):
        g = r[r.group == grp]
        for method, lab, c, mk, ls in arms:
            h = g[g.method == method].groupby("keep").error.mean()
            ax.plot(h.index, h.values, ls, marker=mk, color=c, mfc="white", label=lab)
        ax.set(xscale="log", xlabel="share of samples kept", title=GROUP_TITLE[grp])
        ax.set_xticks([0.1, 0.25, 0.5, 1.0])
        ax.set_xticklabels(["10%", "25%", "50%", "100%"])
        ax.invert_xaxis()
    axes[0].set_ylabel("mean test error")
    axes[0].legend(fontsize=7.2)
    fig.tight_layout()
    return fig


FIGURES = {"v10_learning_curve": v10_learning_curve,
           "classify_benchmark": classify_benchmark,
           "classify_fewshot": classify_fewshot,
           "classify_irregular": classify_irregular}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGURES.items():
        if not (RES / "classify").exists():
            break
        try:
            fig = fn()
        except FileNotFoundError as exc:
            print(f"  [fig] {name} skipped: {exc}")
            continue
        fig.savefig(OUT / f"{name}.png")
        plt.close(fig)
        print(f"  [fig] figures/{name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
