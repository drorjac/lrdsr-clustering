"""Every figure, built from ``results/``.

Each function returns a matplotlib Figure and reads only ``results/`` -- or,
for the two live demonstrations (``mechanism_vs_data_space`` and
``function_atlas``), generates its own windows in a second, so they show the
method running rather than a picture of it.

    python -m analysis.plots        # write them all to figures/
"""
from __future__ import annotations

import matplotlib as mpl

if __name__ == "__main__":
    # Headless when run as the figure build; left alone when imported, so a
    # notebook that imports these figures keeps its inline backend.
    mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lrdsr import paths

RES = paths.RESULTS
#: One gallery for the whole project.
OUT = paths.ROOT / "figures"
W, H = 7.4, 3.2                       # a screen, not a column

C = {"oracle": "#222222", "mech": "#1f5fa8", "data": "#b03a2e",
     "geom": "#b0b0b0", "feat": "#d9822b", "ok": "#4a7a4a", "bad": "#b03a2e"}


def style() -> None:
    mpl.rcParams.update({
        "font.family": "serif", "font.size": 9.5, "axes.titlesize": 10,
        "axes.labelsize": 9.5, "legend.fontsize": 8.5, "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.8,
        "lines.linewidth": 1.6, "lines.markersize": 5,
        "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 160,
        "savefig.bbox": "tight", "mathtext.fontset": "cm"})


def _csv(section: str, name: str) -> pd.DataFrame:
    return pd.read_csv(RES / section / name)


def mechanism_vs_data_space(rho: float = 0.25, seed: int = 11):
    """The paper's thesis in one picture, computed live in a second.

    Left: the windows as any clustering would see them -- two plain summaries
    of ``(x, y)``. Right: the same windows in mechanism space. The colour is
    the regime, which no method is told. The point is that the left panel has
    no structure to find and the right panel is two clouds.
    """
    style()
    from experiments.common.fitting import window_features
    from experiments.estimator.benchmark import (
        PAIR_BY_NAME,
        _make_windows,
        _pair_gap_ms,
    )
    from lrdsr.core.mechanism_space import mechanism_features

    pair = PAIR_BY_NAME["polynomial"]
    sigma = float(np.sqrt(_pair_gap_ms(pair) / rho))
    X, y, z, _x, _Zg = _make_windows(pair, sigma, 400, seed)

    Zf, names = window_features(X, y)
    S = mechanism_features(X, y)
    _u, _sv, vt = np.linalg.svd(S - S.mean(0), full_matrices=False)
    P = (S - S.mean(0)) @ vt[:2].T

    fig, ax = plt.subplots(1, 2, figsize=(W, 3.3))
    for k, (name, col) in enumerate((("regime 0: $x^2+x$", "#1f5fa8"),
                                     ("regime 1: $x^2-x$", "#d9822b"))):
        m = z == k
        ax[0].scatter(Zf[m, 0], Zf[m, 1], s=11, alpha=0.65, lw=0, color=col,
                      label=name)
        ax[1].scatter(P[m, 0], P[m, 1], s=11, alpha=0.65, lw=0, color=col,
                      label=name)
    ax[0].set(xlabel=f"window {names[0]}", ylabel=f"window {names[1]}",
              title="data space: what a clustering sees")
    ax[1].set(xlabel="mechanism space, PC1", ylabel="PC2",
              title=r"mechanism space: centres $\sqrt{n\rho}$ apart")
    for a in ax:
        a.legend(loc="upper right")
    fig.suptitle(f"the same 400 windows, $\\rho$ = {rho}", y=1.02, fontsize=10)
    fig.tight_layout()
    return fig


def separation_identity():
    """Every cell in which the identity was measured, against sqrt(n rho)."""
    style()
    V7 = _csv("theory", "v7_mechanism_space.csv")
    B = _csv("estimator", "benchmark_per_seed.csv")
    fig, ax = plt.subplots(1, 2, figsize=(W, 3.1))

    lim = [0.8, 20]
    ax[0].plot(lim, lim, "-", color=C["oracle"], lw=1.0, zorder=1)
    for (design, in_lib), g in V7.groupby(["design", "in_library"]):
        ax[0].scatter(g.sep_predicted, g.sep_measured, s=26, zorder=3,
                      marker="o" if in_lib else "s",
                      color=C["mech"] if in_lib else C["bad"], alpha=0.8,
                      label=f"{design}" + ("" if in_lib else "  (outside)"))
    ax[0].set(xscale="log", yscale="log", xlim=lim, ylim=lim,
              xlabel=r"$\sqrt{n\rho}$  (theory)",
              ylabel="measured centre separation / $\\hat\\sigma$",
              title="V7: the separation identity")
    ax[0].legend(loc="upper left")

    rel = (B.separation_measured / B.separation_predicted - 1.0) * 100
    for pair, g in B.groupby("pair"):
        r = (g.separation_measured / g.separation_predicted - 1.0) * 100
        ax[1].scatter(g.rho, r, s=26, alpha=0.8, label=pair)
    ax[1].axhline(0, color=C["oracle"], lw=1.0)
    ax[1].set(xscale="log", xlabel=r"$\rho$",
              ylabel="relative error of the identity (%)",
              title=f"§4.2: median |error| = {rel.abs().median():.1f}%")
    ax[1].legend(loc="upper right", ncol=2)
    fig.tight_layout()
    return fig


def attainability():
    """K-means in mechanism space against the EXACT oracle, cell by cell."""
    style()
    V7 = _csv("theory", "v7_mechanism_space.csv")
    fig, ax = plt.subplots(1, 2, figsize=(W, 3.1))
    lim = [-0.01, 0.30]
    ax[0].plot(lim, lim, "-", color=C["oracle"], lw=1.0, zorder=1)
    for (design, in_lib), g in V7.groupby(["design", "in_library"]):
        ax[0].scatter(g.oracle_exact, g.kmeans_error, s=26, zorder=3,
                      marker="o" if in_lib else "s",
                      color=C["mech"] if in_lib else C["bad"], alpha=0.8,
                      label=f"{design}" + ("" if in_lib else "  (outside)"))
    ax[0].set(xlim=lim, ylim=lim, xlabel="exact oracle error",
              ylabel="$K$-means in mechanism space",
              title="the bound is attainable, with no label")
    ax[0].legend(loc="upper left")

    g = V7.groupby(["rho", "shared_x_gap"]).kmeans_error.mean().unstack()
    for c in g.columns:
        ax[1].plot(g.index, g[c], "-o",
                   color=C["mech"] if c == 0 else C["feat"],
                   label=f"shared component = {c:g}$\\times$ the gap")
    ax[1].plot(g.index, V7.groupby("rho").oracle_exact.mean(), "--",
               color=C["oracle"], label="exact oracle")
    ax[1].set(xscale="log", xlabel=r"$\rho$", ylabel="assignment error",
              title="and invariant to a shared component")
    ax[1].legend(loc="upper right")
    fig.tight_layout()
    return fig


def estimator_gap():
    """Distance to the oracle, per pair, for every arm -- the §4.2 headline."""
    style()
    S = _csv("estimator", "benchmark_summary.csv")
    pairs = list(S.pair.unique())
    fig, ax = plt.subplots(1, len(pairs), figsize=(W, 3.0), sharey=True)
    arms = (("lrdsr", "LR-DSR (mechanism init)", C["mech"], "-o"),
            ("lrdsr_dataspace", "LR-DSR (data-space init)", C["data"], "-s"),
            ("mechanism_kmeans", "$K$-means in mechanism space", C["ok"], "-^"),
            ("window_features", "best clustering, window features", C["feat"], "-v"))
    for a, p in zip(ax, pairs, strict=True):
        g = S[S.pair == p].sort_values("rho")
        a.plot(g.rho, g.oracle, "--", color=C["oracle"], label="oracle")
        for col, name, c, fmt in arms:
            a.plot(g.rho, g[col], fmt, color=c, ms=4, label=name)
        a.axhline(0.5, color="#dddddd", lw=0.8)
        a.set(xscale="log", xlabel=r"$\rho$", title=p)
    ax[0].set_ylabel("matched assignment error")
    h, lab = ax[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    return fig


def shared_component_sweep():
    """The invariance, and the library boundary where it stops."""
    style()
    A = _csv("estimator", "shared_component_summary.csv")
    fig, ax = plt.subplots(1, 2, figsize=(W, 3.1), sharey=True)
    for a, (shared, g) in zip(ax, A.groupby("shared"), strict=True):
        g = g.sort_values("c")
        x = np.maximum(g.shared_to_gap, 0.3)
        a.plot(x, g.kmeans_mechanism, "-o", color=C["mech"],
               label="$K$-means, mechanism space")
        a.plot(x, g.kmeans_dataspace, "-^", color=C["feat"],
               label="$K$-means, data space")
        a.plot(x, g.lrdsr_mechanism, "-s", color=C["ok"], ms=4,
               label="LR-DSR (mechanism init)")
        a.axhline(float(g.oracle.mean()), color=C["oracle"], ls="--",
                  label="oracle (fixed by construction)")
        a.axhline(0.5, color="#dddddd", lw=0.8)
        a.set(xscale="log", xlabel="shared component / gap (rms)",
              title=f"shared term: {shared}  ({g['where'].iloc[0]})")
    ax[0].set_ylabel("matched assignment error")
    ax[0].legend(loc="upper left")
    fig.tight_layout()
    return fig


def the_three_laws():
    """f1, f2, f3 -- and the geometry a clustering would have to work from."""
    style()
    from experiments.functions.data import (
        BASE_FUNCTIONS,
        DEFAULT_COMPONENTS,
        simulate_function_mixture,
    )
    d = simulate_function_mixture(n_per_regime=200, noise_std=0.6, seed=11)
    names = [c[0] for c in DEFAULT_COMPONENTS]
    cols = ["#1f5fa8", "#d9822b", "#4a7a4a"]
    fig, ax = plt.subplots(1, 2, figsize=(W, 3.2))
    x = np.linspace(-2.2, 2.2, 300)
    for k, (nm, c) in enumerate(zip(names, cols, strict=True)):
        m = d.labels == k
        eq, fn = BASE_FUNCTIONS[nm]
        ax[0].scatter(d.X[m, 0], d.y[m], s=7, alpha=0.35, lw=0, color=c)
        ax[0].plot(x, fn(x), "-", color=c, lw=2,
                   label=f"$f_{k+1}$  {nm}:  ${eq}$")
        ax[1].scatter(d.Z[m, 0], d.Z[m, 1], s=9, alpha=0.5, lw=0, color=c,
                      label=f"$f_{k+1}$ {nm}")
    ax[0].set(xlabel="$x$", ylabel="$y$",
              title="three laws, one input distribution")
    ax[0].legend(loc="upper left", fontsize=7.5)
    ax[1].set(xlabel="$z_1$", ylabel="$z_2$",
              title="the geometry a clustering sees: overlapping")
    ax[1].legend(loc="upper right")
    fig.tight_layout()
    return fig


def knowledge_ladder():
    """What declared knowledge buys, and how fast n makes it irrelevant."""
    style()
    A = _csv("functions", "knowledge_summary.csv")
    G = _csv("functions", "knowledge_gain.csv")
    ns = sorted(A.window_len.unique())
    order = [("known", "the exact law", C["oracle"], "--o"),
             ("known_form", "the form only", C["ok"], "-s"),
             ("unknown", "nothing (full search)", C["mech"], "-o"),
             ("mechanism $K$-means", "a library only", C["feat"], "-^"),
             ("geometry (best of 7)", "nothing (geometry only)", C["geom"], ":v")]
    fig, ax = plt.subplots(1, len(ns) + 1, figsize=(W + 1.6, 3.4),
                           sharey=True)
    handles = {}
    for a, n in zip(ax[:-1], ns, strict=True):
        g = A[A.window_len == n]
        for key, lab, c, fmt in order:
            h = g[g.framing == key].sort_values("noise_std")
            if len(h):
                line, = a.plot(h.noise_std, h.ari, fmt, color=c, ms=4)
                handles.setdefault(lab, line)
        a.set(xscale="log", xlabel=r"noise $\sigma$",
              title=f"$n$ = {n}" + ("   (row-level)" if n == 1 else ""))
    ax[0].set_ylabel("ARI against the true regime")
    fig.legend(handles.values(), [f"declares: {k}" for k in handles],
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.09),
               fontsize=8)

    gg = G.sort_values("noise_std")
    for n, h in gg.groupby("window_len"):
        ax[-1].plot(h.noise_std, h.known_minus_unknown, "-o", ms=4,
                    label=f"$n$ = {n}")
    ax[-1].axhline(0, color="k", lw=0.8)
    ax[-1].set(xscale="log", xlabel=r"noise $\sigma$",
               title="what the exact law buys")
    ax[-1].set_ylabel("ARI(known) $-$ ARI(unknown)")
    ax[-1].legend(loc="upper right", title="window length")
    fig.tight_layout()
    return fig


def f1_competing_families():
    """F1: two parametric families, against the covariate you could threshold.

    The disturbance is ``c x^b`` added to ``a x^b s``; ``c/a`` is how big it
    is. The honest competitor is not another clustering, it is a threshold on
    the covariate ``s`` itself -- if that separates the regimes, no law
    comparison was needed. Left panel: the disturbance belongs to a whole
    group, so ``s`` still carries it. Right panel: it is assigned per window,
    so ``s`` carries nothing and only the law can tell them apart.
    """
    style()
    A = _csv("functions", "f1_summary.csv")
    V = _csv("functions", "f1_verdict.csv")
    order = [("LR-DSR (form declared)", "LR-DSR (form declared)", C["mech"], "-o"),
             ("best threshold on s", "best threshold on $s$", C["bad"], "--s"),
             ("K-means in mechanism space", "mechanism $K$-means", C["feat"], "-^"),
             ("Bayesian GMM (full cov)", "Bayesian GMM", C["geom"], ":v")]
    titles = {"per_group": "disturbance per group\n($s$ still carries it)",
              "per_window": "disturbance per window\n($s$ carries nothing)"}
    asg = ["per_group", "per_window"]
    ticks = sorted(A.c_over_a.unique())
    labels = [f"{t:g}" for t in ticks]
    fig, ax = plt.subplots(1, 3, figsize=(W + 1.4, 3.3))
    for a, key in zip(ax[:2], asg, strict=True):
        g = A[A.assignment == key]
        for m, lab, c, fmt in order:
            h = g[g.method == m].sort_values("c_over_a")
            a.plot(h.c_over_a, h.ARI, fmt, color=c, ms=4, label=lab)
        a.set(xscale="log", xlabel="$c/a$   (disturbance size)",
              title=titles[key], ylim=(-0.1, 1.05),
              xticks=ticks, xticklabels=labels)
        a.minorticks_off()
    ax[0].set_ylabel("ARI against the true regime")
    ax[1].legend(loc="upper left", fontsize=7.5)

    for key, c, fmt in [("per_group", C["oracle"], "-o"),
                        ("per_window", C["mech"], "--s")]:
        h = V[V.assignment == key].sort_values("c_over_a")
        ax[2].plot(h.c_over_a, h.gain_over_threshold, fmt, color=c, ms=4,
                   label=key.replace("_", " "))
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].fill_between([0.2, 5], -0.6, 0, color=C["bad"], alpha=0.07)
    ax[2].set(xscale="log", xlim=(0.2, 5), xlabel="$c/a$",
              title="gain over the covariate",
              xticks=ticks, xticklabels=labels)
    ax[2].minorticks_off()
    ax[2].set_ylabel("ARI(LR-DSR) $-$ ARI(threshold)")
    ax[2].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def f2_heterogeneous_null():
    """F2: is a wrong $K$ the method's fault, or the framing's?

    The null is contaminated with 1--3 extra mechanisms (a step, an
    oscillation, a drift) that are all still "null". Asked for $K=2$, LR-DSR
    must merge unlike things. Asked for the true $K$ and merged afterwards,
    it does not. The gap between those two curves is the price of the
    framing, and it is invisible until the noise is high enough to matter.
    """
    style()
    A = _csv("functions", "f2_summary.csv")
    ktrue = (A[A.method.str.startswith("LR-DSR (K=") & A.method.str.contains("merged")]
             .groupby(["noise_std", "n_disturbances"]).mcc.mean().reset_index())
    base = (A[~A.method.str.startswith("LR-DSR")]
            .groupby(["noise_std", "n_disturbances"]).mcc.max().reset_index())
    noises = sorted(A.noise_std.unique())
    fig, ax = plt.subplots(1, len(noises), figsize=(W + 0.6, 3.1), sharey=True)
    for a, s in zip(ax, noises, strict=True):
        k2 = A[(A.noise_std == s) & (A.method == "LR-DSR (K=2)")]
        a.plot(k2.n_disturbances, k2.mcc, "-o", color=C["mech"], ms=4,
               label="LR-DSR, told $K=2$")
        h = ktrue[ktrue.noise_std == s]
        a.plot(h.n_disturbances, h.mcc, "--s", color=C["oracle"], ms=4,
               label="LR-DSR, told $K_{\\mathrm{true}}$, then merged")
        b = base[base.noise_std == s]
        a.plot(b.n_disturbances, b.mcc, ":v", color=C["geom"], ms=4,
               label="best of 8 baselines")
        a.set(xlabel="extra null mechanisms", xticks=[0, 1, 2, 3],
              title=rf"$\sigma$ = {s:g}", ylim=(0.35, 1.03))
    ax[0].set_ylabel("MCC, signal vs null")
    ax[0].legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    return fig


def f3_closing_pair():
    """F3: what happens as two regimes become the same law.

    One regime is a scaled copy of another; ``scale`` moves it from clearly
    different (0.2) to nearly identical (0.8). The failure is not graceful
    degradation into noise -- the pair merges and an innocent third regime is
    split to keep $K$ at four.
    """
    style()
    A = _csv("functions", "f3_summary.csv")
    Cf = _csv("functions", "f3_confusions.csv")
    order = [("LR-DSR", "LR-DSR", C["mech"], "-o"),
             ("Bayesian GMM (full cov)", "Bayesian GMM", C["geom"], ":v"),
             ("K-means in mechanism space", "mechanism $K$-means", C["feat"], "-^"),
             ("Ward agglomerative", "Ward", C["geom"], "--x")]
    fig, ax = plt.subplots(1, 2, figsize=(W, 3.1))
    for m, lab, c, fmt in order:
        h = A[A.method == m].sort_values("scale")
        ax[0].plot(h.scale, h.ARI, fmt, color=c, ms=4, label=lab)
    ax[0].set(xlabel="scale of the copy   (1.0 = identical)",
              title="the hard pair closes", ylim=(-0.05, 1.05))
    ax[0].set_ylabel("ARI against the true regime")
    ax[0].legend(loc="lower left", fontsize=8)

    r = (Cf.groupby(["scale", "regime"]).recovered_fraction.min()
           .unstack("regime"))
    for reg, c in [("scaled", C["bad"]), ("other", C["feat"]),
                   ("base", C["mech"]), ("lagged", C["geom"])]:
        ax[1].plot(r.index, r[reg], "-o", color=c, ms=4, label=reg)
    ax[1].set(xlabel="scale of the copy", ylim=(-0.05, 1.05),
              title="which regime pays")
    ax[1].set_ylabel("fraction of the regime recovered")
    ax[1].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return fig


def function_atlas():
    """Every law stage 1 uses, drawn -- the exploration before the method.

    Read out of the modules that define them, so a change to a family
    changes this figure. Four panels, four questions: what does the library
    look like, what does a nested family look like, what does a
    heterogeneous null look like, and what does a closing pair look like.
    """
    style()
    from experiments.functions import scenarios as S
    from experiments.functions.data import BASE_FUNCTIONS, DEFAULT_COMPONENTS

    fig, ax = plt.subplots(1, 4, figsize=(W + 4.4, 2.9))
    used = {c[0] for c in DEFAULT_COMPONENTS}
    pal = ["#1f5fa8", "#b03a2e", "#4a7a4a", "#7b5ea7", "#d9822b"]

    # (a) the base library, on the generator's own x range
    x = np.linspace(-2.2, 2.2, 400)
    for i, (nm, (_eq, f)) in enumerate(BASE_FUNCTIONS.items()):
        ax[0].plot(x, f(x), "-" if nm in used else "--", color=pal[i],
                   lw=2.0 if nm in used else 1.3,
                   label=nm + ("" if nm in used else " (reserve)"))
    ax[0].set(title="(a) the base library", xlabel="$x$", ylabel="$f(x)$")

    # (b) F1: the extra term does not scale with s, so its SHARE falls
    xf = np.linspace(*(0.2, 5.0), 400)
    c = S.F1_A * 1.0
    for i, sv in enumerate((S.F1_S[0], S.F1_S[-1])):
        h0 = S.F1_A * xf ** S.F1_B * sv
        ax[1].plot(xf, h0, "-", color=pal[i], lw=1.8, label=f"$H_0$, $s$={sv:g}")
        ax[1].plot(xf, h0 + c * xf ** S.F1_B, ":", color=pal[i], lw=1.8,
                   label=f"$H_1$, $s$={sv:g}")
    ax[1].set(title="(b) F1 -- nested, $c/a$ = 1", xlabel="$x$", yscale="log")

    # (c) F2: the null is several mechanisms, none like the signal
    x2 = np.linspace(-2.0, 2.0, 400)
    for i, (nm, fac, is_sig, _e, _n) in enumerate(S.F2_REGIMES):
        ax[2].plot(x2, fac(1.0)(x2), "-" if is_sig else "--", color=pal[i],
                   lw=2.2 if is_sig else 1.3,
                   label=nm if is_sig else f"{nm} (counts as null)")
    ax[2].set(title="(c) F2 -- a null of many kinds", xlabel="$x$")

    # (d) F3: the hard pair, at both ends of its knob
    x3 = np.sort(np.linspace(-2.0, 2.0, 400))
    base = 0.8 * x3 ** 2 + 1.5 * x3
    acc, lag = 0.0, np.zeros_like(x3)
    for i, v in enumerate(base):
        acc = 0.85 * acc + 0.15 * v
        lag[i] = acc
    ax[3].plot(x3, base, "-", color=pal[0], lw=2.2, label="base")
    for j, sc in enumerate((S.F3_SCALES[0], S.F3_SCALES[-1])):
        ax[3].plot(x3, sc * base, ":", color=pal[1], lw=1.4 + 0.8 * j,
                   label=f"scaled, $s$={sc:g}")
    ax[3].plot(x3, 2.2 * np.sin(2.0 * x3), "--", color=pal[2], lw=1.3,
               label="other")
    ax[3].plot(x3, 3.0 * lag, "-.", color=pal[3], lw=1.3, label="lagged")
    ax[3].set(title="(d) F3 -- a pair that closes", xlabel="$x$")

    for a in ax:
        a.legend(fontsize=7.2, loc="best")
    fig.tight_layout()
    return fig


def search_path():
    """The search trace: which term entered, in what order, and what it bought.

    The backend is greedy forward selection over a library of interpretable
    terms, scored by BIC. Each row is one regime of one pair; each cell is
    one step of its search, labelled with the term that entered and shaded by
    how much of the total BIC drop that step delivered. The search stops when
    no library term improves BIC -- so a short row is not a failure, it is
    the stopping rule working.

    Read the rows against the truth printed beside them. Where they differ
    and the row is still short, the library did not contain the truth and BIC
    declined to approximate it further: ``saturation`` is that case, by
    construction.
    """
    style()
    P = _csv("estimator", "search_path.csv")
    P = P[P.seed == sorted(P.seed.unique())[0]]
    rhos = sorted(P.rho.unique())
    keys = [(p, s) for p in dict.fromkeys(P.pair) for s in (0, 1)]
    nmax = int(P.n_terms.max())

    fig, ax = plt.subplots(1, len(rhos), figsize=(3.0 * len(rhos) + 2.2, 4.0),
                           sharey=True)
    for c, rho in enumerate(rhos):
        a = ax[c]
        a.set_xlim(-0.5, nmax - 0.5)
        a.set_ylim(len(keys) - 0.5, -0.5)
        for r, (pair, slot) in enumerate(keys):
            h = (P[(P.pair == pair) & (P.slot == slot) & (P.rho == rho)]
                 .sort_values("step"))
            if h.empty:
                continue
            bic = h.bic.to_numpy()
            span = max(bic[0] - bic.min(), 1e-9)
            steps = h[h.term_added.notna()]
            for j, (_, row) in enumerate(steps.iterrows()):
                drop = (bic[j] - bic[j + 1]) / span
                a.add_patch(plt.Rectangle((j - 0.46, r - 0.42), 0.92, 0.84,
                                          color=plt.cm.Blues(0.25 + 0.7 * drop),
                                          ec="w", lw=1.2))
                a.text(j, r, row.term_added, ha="center", va="center",
                       fontsize=7.4,
                       color="w" if drop > 0.45 else "#1b1f24")
            a.text(len(steps) - 0.35, r, "  stop", ha="left", va="center",
                   fontsize=7.0, color="#999999")
        a.set_xticks(range(nmax))
        a.set_xticklabels([f"step {i + 1}" for i in range(nmax)], fontsize=8)
        a.set_title(f"$\\rho$ = {rho:g}", fontsize=9.5)
        for sp in a.spines.values():
            sp.set_visible(False)
        a.tick_params(length=0)
    ax[0].set_yticks(range(len(keys)))
    ax[0].set_yticklabels(
        [f"{p}  [{s}]  ${P[(P.pair == p) & (P.slot == s)].truth.iloc[0]}$"
         for p, s in keys], fontsize=7.8)
    fig.suptitle("the greedy search, step by step      "
                 "shade: share of the BIC drop that step delivered",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def search_complexity():
    """What the search takes, against what the truth leaves behind.

    Left: the number of terms BIC takes, per pair, as the noise falls. Right:
    how far the returned law is from the floor the TRUE law leaves -- zero
    means the search has nothing left to find, whatever its expression looks
    like.
    """
    style()
    C = _csv("estimator", "search_complexity.csv")
    pairs = list(dict.fromkeys(C.pair))
    rhos = sorted(C.rho.unique())
    fig, ax = plt.subplots(1, 2, figsize=(W + 1.0, 3.1))
    x = np.arange(len(pairs))
    for i, rho in enumerate(rhos):
        h = C[C.rho == rho].groupby("pair", sort=False)
        ax[0].bar(x + (i - 1) * 0.27, [h.bic_n_terms.mean()[p] for p in pairs],
                  0.25, color=plt.cm.viridis(0.15 + 0.35 * i),
                  label=f"$\\rho$ = {rho:g}")
        ax[1].bar(x + (i - 1) * 0.27,
                  [h.bic_excess.mean()[p] / h.floor_rmse.mean()[p] for p in pairs],
                  0.25, color=plt.cm.viridis(0.15 + 0.35 * i))
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(pairs, rotation=18)
    ax[0].set(ylabel="terms BIC takes", title="what the search spends")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set(ylabel="(held-out RMSE $-$ floor) / floor",
              title="what is left on the table")
    ax[0].legend(fontsize=8)
    ax[1].text(0.02, 0.95, "floor = the TRUE law's own held-out error",
               transform=ax[1].transAxes, fontsize=7.5, color="#555555",
               va="top")
    fig.tight_layout()
    return fig


def search_beta():
    """Does the assignment's complexity penalty do anything?

    ``beta`` prices a law's size inside the assignment cost. Swept over four
    orders of magnitude around the value every other experiment uses.
    """
    style()
    B = _csv("estimator", "search_beta.csv")
    g = B.groupby(["pair", "beta_complexity"]).agg(
        err=("matched_error", "mean"), cx=("mean_complexity", "mean"),
        it=("n_iterations", "mean")).reset_index()
    fig, ax = plt.subplots(1, 3, figsize=(W + 1.6, 3.0))
    for i, pair in enumerate(dict.fromkeys(B.pair)):
        h = g[g.pair == pair].sort_values("beta_complexity")
        b = h.beta_complexity.replace(0.0, h.beta_complexity[
            h.beta_complexity > 0].min() / 4)      # 0 drawn at the left edge
        for j, (col, lab) in enumerate((("err", "matched error"),
                                        ("cx", "complexity of the fitted law"),
                                        ("it", "iterations to converge"))):
            ax[j].plot(b, h[col], "-o", ms=4, lw=1.5,
                       color=plt.cm.tab10(i), label=pair if j == 0 else None)
            ax[j].set(xscale="log", xlabel=r"$\beta_{\rm complexity}$",
                      ylabel=lab)
    default = 0.002
    for a in ax:
        a.axvline(default, color="#999999", lw=1.0, ls=":")
        a.text(default * 1.25, a.get_ylim()[1], " default", fontsize=7.2,
               color="#777777", va="top")
    ax[0].axhline(0.5, color=C["bad"], lw=0.9, ls="--")
    ax[0].text(b.min(), 0.5, " chance", fontsize=7.2, color=C["bad"],
               va="bottom")
    ax[0].legend(fontsize=7.5)
    fig.suptitle("the complexity penalty, swept over four orders of magnitude",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def search_loop():
    """The alternating loop, iteration by iteration, under two initialisers.

    Fit a law per cluster, score every window under every law, reassign,
    repeat. Solid is the mechanism initialiser, dashed the data-space one it
    replaced. The top row is a separation where the answer is reachable, the
    bottom one where it is not -- and the loop behaves differently in a way
    the final accuracy alone does not show.
    """
    style()
    H = _csv("estimator", "search_history.csv")
    pairs = list(dict.fromkeys(H.pair))
    fig, ax = plt.subplots(2, 3, figsize=(W + 1.8, 5.2), sharex=True)
    for r, rho in enumerate(sorted(H.rho.unique(), reverse=True)):
        for i, pair in enumerate(pairs):
            for init, ls in (("mechanism", "-"), ("bgmm", "--")):
                h = (H[(H.pair == pair) & (H.rho == rho) & (H.init == init)]
                     .groupby("iteration").mean(numeric_only=True))
                if h.empty:
                    continue
                kw = {"color": plt.cm.tab10(i), "ls": ls, "lw": 1.5, "ms": 3.5}
                ax[r, 0].plot(h.index, h.mean_assignment_cost, "o", **kw,
                              label=pair if (init == "mechanism" and r == 0)
                              else None)
                ax[r, 1].plot(h.index, h.changed_fraction.clip(lower=1e-3),
                              "o", **kw)
                ax[r, 2].plot(h.index, h.aligned_accuracy, "o", **kw)
        ax[r, 1].axhline(0.01, color="#999999", lw=1.0, ls=":")
        ax[r, 1].set_yscale("log")
        ax[r, 0].set_ylabel(f"$\\rho$ = {rho:g}\nmean assignment cost",
                            fontsize=8.5)
        ax[r, 1].set_ylabel("fraction of windows that moved", fontsize=8.5)
        ax[r, 2].set_ylabel("accuracy (never fitted)", fontsize=8.5)
    for c, t in enumerate(("what the loop minimises", "the stopping rule",
                           "what it was worth")):
        ax[0, c].set_title(t, fontsize=9.5)
        ax[1, c].set_xlabel("iteration")
    ax[1, 1].text(0.3, 0.0115, "tol", fontsize=7.2, color="#777777")
    ax[0, 0].legend(fontsize=7.5)
    proxy = [plt.Line2D([], [], color="#444444", ls=ls, lw=1.5)
             for ls in ("-", "--")]
    ax[0, 2].legend(proxy, ["mechanism", "data space"], fontsize=7.5,
                    title="initialiser", title_fontsize=7.5, loc="lower right")
    ax[0, 0].text(0.5, 0.5, "the mechanism arm is ONE point:\n"
                            "nothing moved, so the loop stopped",
                  transform=ax[0, 0].transAxes, fontsize=7.4, color="#666666",
                  ha="center")
    fig.tight_layout()
    return fig


def loop_value():
    """Where the alternating loop earns its place.

    Stage 1's negative -- the loop does not beat mechanism K-means -- is
    measured at one point: a start that is already at the oracle. This is
    the same operator measured across starts. Left: the gain against how
    corrupted the starting partition is. Right: what the loop does to each
    initialiser anyone would actually use.
    """
    style()
    C_ = _csv("estimator", "loop_value_corruption.csv")
    I_ = _csv("estimator", "loop_value_initialiser.csv")
    C_, I_ = C_[C_.seed_set == "report"], I_[I_.seed_set == "report"]
    S = _csv("estimator", "loop_value_summary.csv")

    fig, ax = plt.subplots(1, 3, figsize=(W + 2.4, 3.1))

    # (a) gain against corruption, with the bootstrap band
    h = S[S.arm == "corruption"].copy()
    h["level"] = h.level.astype(float)
    h = h.sort_values("level")
    ax[0].fill_between(h.level, h.lo, h.hi, color=C["mech"], alpha=.22)
    ax[0].plot(h.level, h.mean_gain, "-o", ms=4.5, color=C["mech"])
    ax[0].axhline(0, color="k", lw=0.9)
    ax[0].set(xlabel="fraction of the start corrupted",
              ylabel="gain = error in $-$ error out",
              title="the loop repairs in proportion\nto what is broken")
    ax[0].text(0.02, 0.96, "above 0: repair", transform=ax[0].transAxes,
               fontsize=7.5, color=C["ok"], va="top")

    # (b) in vs out: the contraction. The horizontal bands ARE the finding.
    for r, g in C_.groupby("rho"):
        p_ = ax[1].plot(g.error_in, g.error_out, "o", ms=3.5, alpha=.55,
                        label=fr"$\rho$ = {r:g}")[0]
        ax[1].axhline(g.error_out.median(), color=p_.get_color(), lw=1.2,
                      ls="--", alpha=.8)
    lim = [0, max(C_.error_in.max(), C_.error_out.max()) * 1.05]
    ax[1].plot(lim, lim, "-", color="#999999", lw=1.0)
    ax[1].text(lim[1] * .52, lim[1] * .60, "no change", fontsize=7.2,
               color="#777777", rotation=38)
    ax[1].set(xlabel="error of the partition handed in",
              ylabel="error of the partition handed out",
              title="the output does not depend\non the input")
    ax[1].legend(fontsize=7.5, loc="lower right")

    # (c) the label-free initialisers
    g = I_.groupby("init").agg(inn=("error_in", "mean"),
                               out=("error_out", "mean")).reindex(
        ["random", "kmeans", "bgmm", "mechanism"]).dropna()
    xp = np.arange(len(g))
    ax[2].bar(xp - 0.2, g.inn, 0.38, color="#b0b0b0", label="initialiser alone")
    ax[2].bar(xp + 0.2, g.out, 0.38, color=C["mech"], label="after the loop")
    for i, (a, b) in enumerate(zip(g.inn, g.out, strict=True)):
        ax[2].annotate("", (i + 0.2, b), (i - 0.2, a),
                       arrowprops={"arrowstyle": "->", "lw": 1.1,
                                   "color": C["ok"] if b < a else C["bad"]})
    ax[2].set_xticks(xp)
    ax[2].set_xticklabels(g.index, rotation=18)
    ax[2].set(ylabel="matched error", title="on the initialisers\nanyone would use")
    ax[2].legend(fontsize=7.5)
    fig.tight_layout()
    return fig


ALL = {
    "mechanism_vs_data_space": mechanism_vs_data_space,
    "separation_identity": separation_identity,
    "attainability": attainability,
    "shared_component_sweep": shared_component_sweep,
    "estimator_gap": estimator_gap,
    "loop_value": loop_value,
    "search_path": search_path,
    "search_complexity": search_complexity,
    "search_beta": search_beta,
    "search_loop": search_loop,
    "function_atlas": function_atlas,
    "the_three_laws": the_three_laws,
    "knowledge_ladder": knowledge_ladder,
    "f1_competing_families": f1_competing_families,
    "f2_heterogeneous_null": f2_heterogeneous_null,
    "f3_closing_pair": f3_closing_pair,
}


#: The figure modules of the blocks added after this file: problems, losses,
#: online and realdata. Each exposes ``FIGURES = {name: fn}``; they import
#: ``style``/``C`` from here, so they are collected lazily in :func:`all_figures`
#: rather than imported at the top.
BLOCK_FIGURES = ("analysis.figs_problems", "analysis.figs_losses",
                 "analysis.figs_online", "analysis.figs_realdata",
                 "analysis.figs_kernel", "analysis.figs_classify")


def all_figures() -> dict:
    """Every figure in the project: the ones above, then each block's."""
    import importlib

    out = dict(ALL)
    for mod in BLOCK_FIGURES:
        out.update(importlib.import_module(mod).FIGURES)
    return out


def main(argv=None) -> int:
    """``python -m analysis.plots [name-prefix ...]``: write figures to figures/."""
    import sys
    only = (sys.argv[1:] if argv is None else argv) or None
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in all_figures().items():
        if only and not any(name.startswith(o) for o in only):
            continue
        fig = fn()
        fig.savefig(OUT / f"{name}.png")
        plt.close(fig)
        print(f"  [fig] figures/{name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
