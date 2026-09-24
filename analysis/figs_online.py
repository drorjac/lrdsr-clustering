"""Figures for the online block, built from ``results/online/``.

``FIGURES`` maps a name to a function returning a Figure, in the same
contract as ``analysis.plots.ALL``. ``online_stream`` also runs one short
stream live (a second) so the timeline shows the estimator, not a log.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.plots import C, H, W, plt, style
from lrdsr import paths

RES = paths.RESULTS / "online"


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RES / name)


def online_sequential():
    """V9: detection delay and false-alarm run length against the theory."""
    style()
    D, A = _csv("v9_delay.csv"), _csv("v9_arl.csv")
    fig, ax = plt.subplots(1, 2, figsize=(W + 1.2, H + 0.3))
    pal = ["#1f5fa8", "#d9822b", "#4a7a4a", "#7b5ea7"]
    hh = np.linspace(D.h.min(), D.h.max(), 100)
    from lrdsr.theory.sequential import KAPPA, predicted_arl, predicted_delay
    for i, (rho, g) in enumerate(D.groupby("rho")):
        ax[0].errorbar(g.h, g.delay_mc, yerr=1.96 * g.delay_se, fmt="o", ms=4,
                       color=pal[i], label=rf"$\rho$ = {rho:g}")
        ax[0].plot(hh, [predicted_delay(h, rho, KAPPA) for h in hh], "-",
                   color=pal[i], lw=1.2)
        ax[0].plot(hh, 2 * hh / rho, ":", color=pal[i], lw=1.0)
    ax[0].set(yscale="log", xlabel="threshold $h$", ylabel="delay (samples)",
              title="delay after a switch: MC (dots), tilted (—), $2h/\\rho$ (···)")
    ax[0].legend(loc="lower right")
    for i, (rho, g) in enumerate(A.groupby("rho")):
        cens = g.censored_frac > 0
        ax[1].plot(g.h[~cens], g.arl_mc[~cens], "o", ms=4, color=pal[i],
                   label=rf"$\rho$ = {rho:g}")
        ax[1].plot(g.h[cens], g.arl_mc[cens], "o", ms=5, mfc="none", color=pal[i])
        ax[1].plot(hh, [predicted_arl(h, rho, KAPPA) for h in hh], "-",
                   color=pal[i], lw=1.2)
    ax[1].plot(hh, np.exp(hh), "--", color=C["oracle"], lw=1.2, label="$e^h$ (bound)")
    ax[1].set(yscale="log", xlabel="threshold $h$", ylabel="run length (samples)",
              title="false-alarm run length (hollow: censored)")
    ax[1].legend(loc="upper left")
    fig.tight_layout()
    return fig


def online_birth_drift():
    """Regime birth against the novelty test's power; tracking a drifting law."""
    style()
    B = _csv("stream_birth_summary.csv")
    B = B[B["stream"] == "newcomer"]
    T = _csv("stream_drift_time.csv")
    fig, ax = plt.subplots(1, 3, figsize=(W + 3.2, H + 0.2))
    pal = {1e-2: "#b03a2e", 1e-3: "#1f5fa8", 1e-4: "#4a7a4a"}
    for (alpha, pat), g in B.groupby(["novelty_alpha", "novelty_patience"]):
        mk = "o" if pat == B.novelty_patience.min() else "s"
        ax[0].plot(g.predicted_power, g.frac_born, mk, color=pal.get(alpha, "k"), ms=5,
                   label=rf"$\alpha$={alpha:g}, patience {pat}")
        ok = g.frac_born > 0
        ax[1].plot(g.predicted_delay_newcomer_windows[ok],
                   g.mean_birth_delay_newcomer_windows[ok], mk,
                   color=pal.get(alpha, "k"), ms=5)
    ax[0].set(xlabel="novelty test power (non-central $\\chi^2$)",
              ylabel="fraction of runs with a birth", title="(a) is the newcomer born?",
              ylim=(-0.05, 1.05), xscale="log")
    ax[0].legend(fontsize=7, loc="upper left")
    lim = [1, max(40, float(np.nanmax(B.predicted_delay_newcomer_windows[B.frac_born > 0])))]
    ax[1].plot(lim, lim, "-", color=C["oracle"], lw=0.8)
    ax[1].set(xscale="log", yscale="log", xlabel="predicted: patience / power",
              ylabel="measured (newcomer windows)", title="(b) how soon")
    for i, (arm, g) in enumerate(T.groupby("arm", sort=False)):
        g = g.groupby("drift_offset", as_index=False).mean(numeric_only=True)
        ax[2].plot(g["drift_offset"], g.tracking_rmse, "-o", ms=3,
                   color=["#999999", "#b03a2e", "#d9822b", "#1f5fa8"][i % 4], label=arm)
    ax[2].set(xlabel="intercept drift $c(t)$", ylabel="law RMSE",
              title="(c) tracking a drifting law")
    ax[2].legend(loc="upper left")
    fig.tight_layout()
    return fig


def online_stream():
    """One stream clustered live (timeline), and the stationary scoreboard."""
    style()
    from experiments.online.live import _stream
    from lrdsr import viz
    from lrdsr.core.online import OnlineLRDSR

    Xh, yh, X, y, z = _stream(11, 150)
    on = OnlineLRDSR(feature_names=["x"]).warm_start(Xh, yh, n_clusters=2, random_state=11)
    tl = on.fit_stream(X, y, true_labels_for_eval=z)
    # present regimes in the truth's colours: match slots once, over the stream
    from experiments.online.streams import _mapping
    m = _mapping(z, tl["label"].to_numpy())
    tl["label"] = [m.get(int(v), int(v)) if v >= 0 else -1 for v in tl["label"]]

    S = _csv("stream_stationary_summary.csv")
    fig = plt.figure(figsize=(W + 1.6, H + 2.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.1], hspace=0.55)
    viz.plot_stream(tl, ax=fig.add_subplot(gs[0]),
                    title="a stream clustered as it arrives: a third law appears at "
                          "window 60 (grey = held as novel, dotted = regime born)")
    ax = fig.add_subplot(gs[1])
    order = ["oracle", "batch_soft", "batch_lrdsr", "online_history160", "online",
             "online_frozen"]
    order = [o for o in order if o in set(S.method)]
    rhos = sorted(S.rho.unique())
    wbar = 0.8 / len(order)
    cols = [C["oracle"], "#7b5ea7", C["mech"], "#4a7a4a", C["feat"], C["geom"]]
    for i, meth in enumerate(order):
        g = S[S.method == meth].set_index("rho").reindex(rhos)
        ax.bar(np.arange(len(rhos)) + (i - (len(order) - 1) / 2) * wbar, g.mean_error,
               width=wbar, color=cols[i % len(cols)], label=meth.replace("_", " "))
    ax.set(xticks=range(len(rhos)), xticklabels=[rf"$\rho$ = {r:g}" for r in rhos],
           ylabel="window error", title="stationary stream: online vs batch in hindsight")
    ax.legend(ncol=3, fontsize=7.5, loc="upper right")
    return fig


FIGURES = {
    "online_sequential": online_sequential,
    "online_birth_drift": online_birth_drift,
    "online_stream": online_stream,
}


if __name__ == "__main__":
    out = paths.ROOT / "figures"
    out.mkdir(exist_ok=True)
    for name, fn in FIGURES.items():
        f = fn()
        f.savefig(out / f"{name}.png")
        plt.close(f)
        print(f"  [fig] figures/{name}.png")
