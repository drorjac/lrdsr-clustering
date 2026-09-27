"""The kernel-extension figures, read from ``results/kernel/``.

    python -m analysis.figs_kernel      # write them to figures/
"""
from __future__ import annotations

import matplotlib as mpl

if __name__ == "__main__":
    mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.plots import OUT, RES, C, W, style

RHO_STYLE = {0.1: ("o", ":"), 0.25: ("s", "--"), 1.0: ("^", "-")}


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RES / "kernel" / name)


def kernel_rank_sweep(problem: str = "high_frequency"):
    """Bias against variance for a kernel basis, on the zoo's one failure.

    Left: the share of the gap between the two laws that a Nystrom basis of
    each rank leaves outside its span; the library's share for comparison.
    Right: mechanism K-means error against rank at each separation
    (reporting seeds), the oracle as a floor, and the rank the label-free
    SNR rule picks (filled markers).
    """
    style()
    sw = _csv("kernel_sweep.csv")
    sw = sw[sw.problem == problem]
    cat = _csv("kernel_gap_catalog.csv").set_index("problem").loc[problem]
    rules = _csv("kernel_rules.csv")
    rules = rules[rules.problem == problem]
    zoo = _csv("kernel_zoo.csv")
    orc = zoo[(zoo.problem == problem) & (zoo.method == "oracle")].groupby("rho").matched_error.mean()

    fig, (a, b) = plt.subplots(1, 2, figsize=(W + 0.6, 3.0))
    g = sw.groupby("basis_rank").gap_outside.mean()
    a.plot(g.index, 100 * g.values, "o-", color=C["mech"], label="Nystrom (RBF) basis")
    a.axhline(100 * cat.gap_outside_library, color=C["feat"], ls="--",
              label=f"term library ({100 * cat.gap_outside_library:.0f}%)")
    a.set(xlabel="basis rank", ylabel="% of the gap outside the span",
          title="what the basis cannot see", yscale="symlog", ylim=(0, 150))
    a.legend(loc="upper right")
    for rho, (mk, ls) in RHO_STYLE.items():
        e = sw[sw.rho == rho].groupby("basis_rank").error.mean()
        b.plot(e.index, e.values, ls, marker=mk, color=C["mech"], mfc="white",
               label=f"rho = {rho:g}")
        b.axhline(orc.get(rho, np.nan), color=C["oracle"], lw=0.8, ls=ls)
        pick = rules[rules.rho == rho]
        pr = pick.snr_size.map(lambda s: s + 1).mean()        # size -> rank
        pe = pick.snr_error.mean()
        b.plot([pr], [pe], marker=mk, color=C["mech"], ms=7)
    b.set(xlabel="basis rank", ylabel="matched error",
          title="error; filled = SNR rule, black = oracle")
    b.legend(loc="upper right")
    fig.tight_layout()
    return fig


def kernel_zoo():
    """Gap to the oracle on every zoo problem: term library against kernel basis."""
    style()
    s = pd.read_csv(RES / "kernel" / "kernel_zoo_summary.csv").set_index("problem")
    order = s["mech_library"].sort_values().index
    arms = (("mech_library", "mechanism K-means, library", C["feat"], "o"),
            ("mech_kernel", "mechanism K-means, kernel", C["mech"], "s"),
            ("soft_em_library", "soft EM, library", C["feat"], "^"),
            ("soft_em_kernel", "soft EM, kernel", C["mech"], "D"))
    fig, ax = plt.subplots(figsize=(W, 3.6))
    y = np.arange(len(order))
    for i, (col, lab, colr, mk) in enumerate(arms):
        filled = "kernel" in col
        ax.plot(s.loc[order, col], y + (i - 1.5) * 0.16, mk, color=colr,
                mfc=colr if filled else "white", ls="none", label=lab)
    ax.axvline(0, color=C["oracle"], lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set(xlabel="matched error minus the oracle's (mean of 3 rho x 3 seeds)")
    ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    return fig


FIGURES = {"kernel_rank_sweep": kernel_rank_sweep, "kernel_zoo": kernel_zoo}


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
