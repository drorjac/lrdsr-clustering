"""The robustness figures, read from ``results/robustness/``.

    python -m analysis.figs_robustness   # write them to figures/
"""
from __future__ import annotations

import matplotlib as mpl

if __name__ == "__main__":
    mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.plots import OUT, RES, C, W, style

ARMS = (("oracle_gls", "oracle, knows the correlation (GLS)", C["oracle"], "--"),
        ("oracle_ols", "oracle, sums samples (the method's rule)", C["oracle"], "-"),
        ("lrdsr", "LR-DSR", C["mech"], "-"),
        ("soft_em", "soft EM", "#7b5ea7", "-"),
        ("mechanism_kmeans", "mechanism K-means", C["feat"], "-"))


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RES / "robustness" / name)


def _real_band(ax, acf: pd.DataFrame, top: bool = False) -> None:
    """Shade the lag-1 residual autocorrelation measured on the real days."""
    lag1 = acf[acf.lag == 1].set_index("dataset").acf_median
    ax.axvspan(lag1.min(), lag1.max(), color=C["ok"], alpha=0.16, lw=0)
    ax.text(lag1.mean(), 0.97 if top else 0.03, "real\ndays", transform=ax.get_xaxis_transform(),
            ha="center", va="top" if top else "bottom", fontsize=7.5, color=C["ok"])


def robustness_ar_noise():
    """Noise correlated in time. (a) Error against the correlation, mean over
    the zoo; (b) the exact formulas against 4000 windows; (c) where knowing
    the correlation helps; (d) how correlated real residuals are."""
    style()
    S, T = _csv("ar_noise_summary.csv"), _csv("ar_noise_theory.csv")
    A = _csv("real_residual_acf.csv")
    fig, ax = plt.subplots(1, 4, figsize=(W + 7.6, 3.4),
                           gridspec_kw={"width_ratios": [1.35, 1, 1.15, 1]})

    # (a) the sweep
    m = S.groupby("phi").mean(numeric_only=True)
    for key, lab, col, ls in ARMS:
        ax[0].plot(m.index, m[key], ls, marker="o", ms=3.5, color=col, label=lab)
    _real_band(ax[0], A)
    ax[0].set(xlabel=r"noise autocorrelation $\phi$ (lag 1)", ylabel="matched error",
              title="same laws, same noise level;\nonly the correlation changes")
    ax[0].legend(fontsize=6.8, loc="upper left", frameon=True, framealpha=0.92, edgecolor="none")

    # (b) theory against simulation
    lim = [0, max(T[["obs_ols", "obs_gls", "pred_ols", "pred_gls"]].max()) * 1.05]
    ax[1].plot(lim, lim, "-", color="#999999", lw=1)
    ax[1].scatter(T.pred_ols, T.obs_ols, s=9, color=C["oracle"], label="sum of samples (OLS)")
    ax[1].scatter(T.pred_gls, T.obs_gls, s=9, facecolors="none", edgecolors=C["mech"],
                  label="whitened (GLS)")
    ax[1].set(xlabel="predicted error (exact formula)", ylabel="observed, 4000 windows",
              title="the error is still a formula")
    ax[1].legend(fontsize=7, loc="upper left")

    # (c) what knowing the correlation is worth, per problem, at phi = 0.9
    g = S[S.phi == 0.9].groupby("problem")[["oracle_ols", "oracle_gls"]].mean()
    g = g.assign(gain=g.oracle_ols - g.oracle_gls).sort_values("gain")
    yp = np.arange(len(g))
    ax[2].barh(yp, g.gain, color=C["mech"])
    ax[2].set_yticks(yp)
    ax[2].set_yticklabels(g.index, fontsize=7.5)
    ax[2].set(xlabel="error saved by whitening", title=r"what knowing $\phi$ is worth ($\phi$ = 0.9)")

    # (d) real residuals against AR(1)
    lags = np.arange(0, A.lag.max() + 1)
    for (name, d), col in zip(A.groupby("dataset"), (C["feat"], C["mech"]), strict=True):
        d = d.sort_values("lag")
        ax[3].fill_between(d.lag, d.acf_q25, d.acf_q75, color=col, alpha=0.15, lw=0)
        ax[3].plot(np.r_[0, d.lag], np.r_[1, d.acf_median], "-o", ms=3.5, color=col,
                   label=f"{name} (median day)")
        phi1 = float(d.acf_median.iloc[0])
        ax[3].plot(lags, phi1 ** lags, ":", color=col, lw=1)
    ax[3].axhline(0, color="k", lw=0.7)
    ax[3].set(xlabel="lag (hours)", ylabel="residual autocorrelation",
              title="real days: short memory\n(dotted: AR(1), same lag 1)")
    ax[3].legend(fontsize=7)
    fig.tight_layout()
    return fig


def robustness_failure():
    """The failure in one picture: accuracy against the noise correlation,
    mean over the zoo. Red is what the method loses beyond its own rule; the
    dashed line is what a rule that knows the correlation reaches."""
    style()
    S, A = _csv("ar_noise_summary.csv"), _csv("real_residual_acf.csv")
    m = 1.0 - S.groupby("phi")[["oracle_ols", "oracle_gls", "lrdsr"]].mean()
    phi = m.index.to_numpy()

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.fill_between(phi, m.lrdsr, m.oracle_ols, color=C["bad"], alpha=0.18, lw=0,
                    label="LR-DSR's gap to the oracle\n(2 pts with independent noise; grows with $\\phi$)")
    ax.plot(phi, m.oracle_gls, "--", color="#777777", lw=1.4,
            label="oracle that knows the correlation\n(what a time-series version could reach)")
    ax.plot(phi, m.oracle_ols, "-o", ms=4, color=C["oracle"],
            label="oracle that knows the laws (the method's rule)")
    ax.plot(phi, m.lrdsr, "-o", ms=4, color=C["mech"], lw=2, label="LR-DSR")
    for p_, lo, hi in zip(phi, m.lrdsr, m.oracle_ols, strict=True):
        ax.annotate(f"-{100 * (hi - lo):.0f} pts", (p_, lo), textcoords="offset points",
                    xytext=(0, -13), ha="center", fontsize=8, color=C["bad"])
    _real_band(ax, A, top=True)
    ax.set(xlabel=r"noise autocorrelation $\phi$  (0 = independent, as the method assumes)",
           ylabel="accuracy (mean over 12 problems)", ylim=(0.5, 1.0),
           title="the method degrades as the noise becomes correlated in time")
    ax.legend(fontsize=7.6, loc="lower left", frameon=True, framealpha=0.92, edgecolor="none")
    fig.tight_layout()
    return fig


FIGURES = {"robustness_ar_noise": robustness_ar_noise,
           "robustness_failure": robustness_failure}


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
