"""Figures for the problem zoo (``results/problems/``).

``problem_atlas`` draws the laws live from the registry, so a change to a
problem changes the picture; the other two read only the committed CSVs.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.plots import RES, C, style

PAL = ("#1f5fa8", "#d9822b", "#4a7a4a", "#7b5ea7")
LABEL = {"oracle": "oracle", "lrdsr": "LR-DSR (hard)", "soft_em": "soft EM",
         "mechanism_kmeans": "mechanism K-means", "geometry": "best geometry",
         "profile_kmeans": "profile K-means"}
COLOR = {"oracle": C["oracle"], "lrdsr": C["mech"], "soft_em": "#4a7a4a",
         "mechanism_kmeans": "#7b5ea7", "geometry": C["geom"],
         "profile_kmeans": C["feat"]}


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RES / "problems" / name)


def problem_atlas():
    """Every problem's laws, one panel each; two-input laws at ``x2 = 1``."""
    style()
    from experiments.problems.zoo import PROBLEMS
    n = len(PROBLEMS)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(11.5, 2.35 * rows))
    ax = ax.ravel()
    for i, p in enumerate(PROBLEMS):
        x = np.linspace(*p.x_range, 500)
        X = x[:, None] if p.d == 1 else np.column_stack([x, np.ones_like(x)])
        for k, (f, t) in enumerate(zip(p.laws, p.truths, strict=True)):
            ax[i].plot(x, f(X), color=PAL[k % len(PAL)], lw=1.7, label=t)
        tag = "" if p.in_library else "  [out of library]"
        ax[i].set_title(f"{p.name}{tag}", fontsize=9)
        ax[i].legend(fontsize=6.5, loc="best")
        if p.d == 2:
            ax[i].set_xlabel("$x_1$  ($x_2 = 1$)", fontsize=8)
    for a in ax[n:]:
        a.axis("off")
    fig.tight_layout()
    return fig


def problem_error_vs_rho():
    """Matched error against ``rho`` per problem, every method, oracle in black."""
    style()
    s = _csv("problems_summary.csv")
    probs = list(dict.fromkeys(s["problem"]))
    cols = 4
    rows = int(np.ceil(len(probs) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(11.5, 2.4 * rows), sharey=True)
    ax = ax.ravel()
    for i, p in enumerate(probs):
        g = s[s["problem"] == p].sort_values("rho")
        for m in LABEL:
            if m not in g or g[m].isna().all():
                continue
            ax[i].plot(g["rho"], g[m], "-o", ms=3, color=COLOR[m],
                       lw=2.2 if m == "oracle" else 1.3, label=LABEL[m])
        ax[i].set(xscale="log", title=p, ylim=(-0.02, 0.55))
        ax[i].title.set_fontsize(9)
    for a in ax[len(probs):]:
        a.axis("off")
    for a in ax[::cols]:
        a.set_ylabel("matched error")
    ax[0].legend(fontsize=6.5, loc="upper right")
    fig.supxlabel(r"separation $\rho$ of the closest pair", fontsize=9)
    fig.tight_layout()
    return fig


def problem_gap_heatmap():
    """Gap to the oracle, method x problem, averaged over rho and seeds."""
    style()
    s = _csv("problems_summary.csv")
    methods = [m for m in LABEL if m != "oracle" and f"{m}_gap" in s]
    M = s.groupby("problem")[[f"{m}_gap" for m in methods]].mean()
    M = M.loc[M[f"{methods[0]}_gap"].sort_values().index]
    fig, ax = plt.subplots(figsize=(7.4, 0.32 * len(M) + 1.4))
    im = ax.imshow(M.to_numpy(), cmap="Reds", vmin=0, vmax=0.4, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 0.25 else "black")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([LABEL[m] for m in methods], rotation=20, ha="right")
    ax.set_yticks(range(len(M)))
    ax.set_yticklabels(M.index)
    ax.set_title(r"excess error over the oracle (mean over $\rho$, seeds)")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    return fig


FIGURES = {
    "problem_atlas": problem_atlas,
    "problem_error_vs_rho": problem_error_vs_rho,
    "problem_gap_heatmap": problem_gap_heatmap,
}


if __name__ == "__main__":
    from lrdsr import paths
    out = paths.ROOT / "figures"
    out.mkdir(exist_ok=True)
    for name, fn in FIGURES.items():
        f = fn()
        f.savefig(out / f"{name}.png")
        plt.close(f)
        print(f"  [fig] figures/{name}.png")
