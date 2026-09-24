"""Figures for the losses block, built from ``results/losses/``.

    FIGURES = {"loss_efficiency": ..., "loss_estimator": ..., "loss_learning": ...}

Each function returns a Figure and reads only the committed CSVs.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from analysis.plots import C, H, W, _csv, style

NOISE_ORDER = ("gaussian", "laplace", "student_t3", "contaminated_10", "contaminated_20")
LOSS_ORDER = ("squared", "absolute", "huber", "cauchy", "tukey", "student_t", "lrt")
PAL = {"squared": C["data"], "absolute": "#8c6d31", "huber": C["mech"],
       "cauchy": "#2a9d8f", "tukey": "#7b5ea7", "student_t": C["feat"],
       "lrt": C["oracle"], "learned": C["ok"]}


def loss_efficiency():
    """V8: each loss's efficiency, and the formula against the simulation.

    Left: ``eta / eta_LRT`` -- the share of the best possible separation a
    loss keeps under each noise. Right: every resolvable cell, simulated
    error against the formula, coloured by the per-sample gap in scale
    units; the small-gap cells sit on the diagonal, the large-gap ones fall
    away from it, which is where the expansion stops being an approximation.
    """
    style()
    d = _csv("losses", "v8_efficiency.csv")
    eta = d.drop_duplicates(["noise", "loss"]).pivot(index="noise", columns="loss",
                                                     values="eta_rel_lrt")
    eta = eta.loc[[n for n in NOISE_ORDER if n in eta.index]]
    fig, ax = plt.subplots(1, 2, figsize=(W + 2.4, H + 0.4),
                           gridspec_kw={"width_ratios": [1.5, 1]})
    losses = [lo for lo in LOSS_ORDER if lo in eta.columns and lo != "lrt"]
    width = 0.8 / len(losses)
    xs = np.arange(len(eta))
    for i, lo in enumerate(losses):
        ax[0].bar(xs + (i - (len(losses) - 1) / 2) * width, eta[lo], width,
                  color=PAL[lo], label=lo)
    ax[0].axhline(1.0, color=C["oracle"], lw=0.9, ls="--")
    ax[0].set(xticks=xs, ylim=(0, 1.08), ylabel=r"$\eta / \eta_{\mathrm{LRT}}$",
              title="(a) the share of the ceiling a loss keeps")
    ax[0].set_xticklabels([n.replace("contaminated_", "contam. ").replace("_", " ")
                           for n in eta.index], fontsize=8)
    ax[0].legend(ncol=3, fontsize=7.5, loc="lower left")

    r = d[d.resolvable.astype(str) == "True"]
    sc = ax[1].scatter(r.predicted_design, r.simulated, c=np.log10(r.local_gap), s=9,
                       cmap="viridis", lw=0)
    lo_, hi_ = 1e-4, 0.5
    ax[1].plot([lo_, hi_], [lo_, hi_], color=C["oracle"], lw=0.9)
    ax[1].set(xscale="log", yscale="log", xlim=(lo_, hi_), ylim=(lo_, hi_),
              xlabel=r"predicted $E_x\,Q(\sqrt{\eta D_n/\mathrm{Var}}/2)$",
              ylabel="simulated error", title="(b) formula vs simulation")
    cb = fig.colorbar(sc, ax=ax[1], fraction=0.05)
    cb.set_label(r"$\log_{10}$ per-sample gap / scale")
    fig.tight_layout()
    return fig


def loss_estimator():
    """The estimator under non-Gaussian noise, one bar per arm per noise."""
    style()
    s = _csv("losses", "loss_estimator_summary.csv")
    arms = ["oracle_lrt", "oracle_squared", "mechanism_kmeans", "hard_huber",
            "hard_squared", "hard_cauchy", "hard_tukey", "hard_student_t",
            "hard_learned", "soft_gaussian", "soft_student_t"]
    col = {"oracle_lrt": C["oracle"], "oracle_squared": "#777777",
           "mechanism_kmeans": C["geom"], "hard_huber": PAL["huber"],
           "hard_squared": PAL["squared"], "hard_cauchy": PAL["cauchy"],
           "hard_tukey": PAL["tukey"], "hard_student_t": PAL["student_t"],
           "hard_learned": PAL["learned"], "soft_gaussian": "#e8a3a0",
           "soft_student_t": "#a6c8a0"}
    noises = [n for n in NOISE_ORDER if n in set(s.noise)]
    fig, ax = plt.subplots(figsize=(W + 2.4, H + 0.5))
    width = 0.84 / len(arms)
    for i, a in enumerate(arms):
        v = s[s.arm == a].set_index("noise").mean_error.reindex(noises)
        ax.bar(np.arange(len(noises)) + (i - (len(arms) - 1) / 2) * width, v, width,
               color=col[a], label=a.replace("hard_", "LR-DSR ").replace("soft_", "soft EM "))
    ax.set(xticks=range(len(noises)), ylabel="matched error",
           title="the estimator under four noise laws (pairs, rho and seeds pooled)")
    ax.set_xticklabels([n.replace("_", " ") for n in noises])
    ax.legend(ncol=4, fontsize=7.2, loc="upper left")
    fig.tight_layout()
    return fig


def loss_learning():
    """Learning the loss: the fitted nu, and the family named from residuals."""
    style()
    tr = _csv("losses", "loss_learning_trace.csv")
    fam = _csv("losses", "loss_learning_family.csv")
    fig, ax = plt.subplots(1, 2, figsize=(W + 2.0, H + 0.3))
    ncol = {"gaussian": C["mech"], "laplace": "#8c6d31", "student_t3": C["feat"],
            "contaminated_10": C["bad"]}
    for n, c in ncol.items():
        t = tr[(tr.noise == n) & (tr.method == "soft_student_t")]
        g = t.groupby("iteration").nu.median()
        ax[0].plot(g.index, g.values, color=c, label=f"{n.replace('_', ' ')}")
        h = tr[(tr.noise == n) & (tr.method == "hard_learned")].dropna(subset=["nu"])
        if len(h):
            gh = h.groupby("iteration").nu.median()
            ax[0].plot(gh.index, gh.values, "o--", ms=3, color=c, lw=1)
    ax[0].axhline(3.0, color=C["oracle"], lw=0.8, ls=":")
    ax[0].text(ax[0].get_xlim()[1] * 0.98, 3.2, r"true $\nu$ of $t_3$", ha="right",
               fontsize=7.5)
    ax[0].set(yscale="log", xlabel="iteration", ylabel=r"learned $\nu$ (median)",
              title=r"(a) $\nu$: soft EM (solid), LR-DSR learned (dashed)")
    ax[0].legend(fontsize=7.5)

    noises = list(ncol)
    sizes = sorted(fam.n_samples.unique())
    right = (fam[fam.is_right.astype(str) == "True"]
             .pivot(index="true_noise", columns="n_samples", values="share_effective")
             .reindex(noises))
    for n in noises:
        ax[1].plot(sizes, right.loc[n, sizes], "-o", color=ncol[n], ms=4,
                   label=n.replace("_", " "))
    ax[1].set(xscale="log", ylim=(-0.03, 1.05), xlabel="residual samples",
              ylabel="share naming the right family",
              title="(b) the family, from residuals alone")
    ax[1].legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    return fig


FIGURES = {
    "loss_efficiency": loss_efficiency,
    "loss_estimator": loss_estimator,
    "loss_learning": loss_learning,
}
