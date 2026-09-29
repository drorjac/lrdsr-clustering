"""Plotting for LR-DSR fits: the pictures you want after any ``fit``.

Dataset-agnostic and label-optional: every function takes arrays the
estimators return (``DCSRResult``, ``SoftResult``, an ``OnlineLRDSR``
timeline) and an optional ``ax``, and returns the axes so calls compose into
a figure. ``analysis/plots.py`` builds the committed figures from
``results/``; this module is for looking at a fit you just made -- in a
notebook, or in :func:`animate_stream`, while it runs.

=========================  =================================================
``plot_windows``           the raw samples, coloured by regime
``plot_laws``              fitted laws over their input range, truth dashed
``plot_mechanism_space``   windows in law coordinates (2-D projection)
``plot_cost_matrix``       the (W, K) assignment cost, windows sorted
``plot_history``           learning curves of the hard loop or of EM
``plot_confusion``         aligned confusion matrix against a reference
``plot_responsibilities``  soft posteriors per window, as stacked bars
``plot_stream``            a real-time timeline: truth vs assignment
``plot_losses``            the loss family: rho(r) and psi(r)
``plot_error_curve``       window error against n*rho, with the ceiling
``animate_stream``         a live animation of online clustering
=========================  =================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: The project palette: regimes first, then the fixed roles.
PALETTE = ("#1f5fa8", "#d9822b", "#4a7a4a", "#7b5ea7", "#b03a2e", "#2a9d8f",
           "#8c6d31", "#e377c2", "#17becf", "#7f7f7f")
ROLE = {"oracle": "#222222", "mech": "#1f5fa8", "data": "#b03a2e",
        "geom": "#b0b0b0", "feat": "#d9822b", "ok": "#4a7a4a", "bad": "#b03a2e",
        "novel": "#9a9a9a"}


def _plt():
    import matplotlib.pyplot as plt
    return plt


def _ax(ax, figsize=(5.2, 3.4)):
    if ax is None:
        _, ax = _plt().subplots(figsize=figsize)
    return ax


def regime_color(k: int) -> str:
    """Colour of regime ``k``; ``-1`` (unassigned / novel) is grey."""
    return ROLE["novel"] if k < 0 else PALETTE[k % len(PALETTE)]


def style() -> None:
    """The project's matplotlib style (the one ``analysis/plots.py`` uses)."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif", "font.size": 9.5, "axes.titlesize": 10,
        "axes.labelsize": 9.5, "legend.fontsize": 8.5, "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.8,
        "lines.linewidth": 1.6, "lines.markersize": 5,
        "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 160,
        "savefig.bbox": "tight", "mathtext.fontset": "cm",
        # White whatever the viewer's theme: an IDE or Jupyter dark mode
        # otherwise shows transparent figures on a dark background.
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.transparent": False,
        "text.color": "black", "axes.labelcolor": "black",
        "axes.edgecolor": "black", "xtick.color": "black", "ytick.color": "black"})


# ==========================================================================
# data and laws
# ==========================================================================
def plot_windows(X_seq, y_seq, labels, ax=None, feature: int = 0,
                 max_windows: int = 80, seed: int = 0, s: float = 6,
                 alpha: float = 0.45, names=None):
    """Every sample of up to ``max_windows`` windows, coloured by ``labels``."""
    ax = _ax(ax)
    X_seq = np.asarray(X_seq, dtype=float)
    if X_seq.ndim == 2:
        X_seq = X_seq[:, :, None]
    y_seq = np.asarray(y_seq, dtype=float)
    labels = np.asarray(labels, int)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(labels))
    if len(idx) > max_windows:
        idx = np.sort(rng.choice(idx, max_windows, replace=False))
    for k in np.unique(labels):
        sel = idx[labels[idx] == k]
        if not len(sel):
            continue
        ax.scatter(X_seq[sel, :, feature].ravel(), y_seq[sel].ravel(), s=s,
                   lw=0, alpha=alpha, color=regime_color(int(k)),
                   label=(names[k] if names is not None and k >= 0
                          else ("unassigned" if k < 0 else f"regime {k}")))
    ax.set(xlabel="$x$", ylabel="$y$")
    ax.legend(loc="best", markerscale=2)
    return ax


def plot_laws(models=None, x_range=(-2, 2), ax=None, truth=None, n: int = 300,
              labels=None, coef=None, basis=None, feature_names=None,
              other=None):
    """Fitted laws over ``x_range`` (solid), and the true ones (dashed).

    Pass either ``models`` (objects with ``predict(X)``, as ``DCSRResult``
    holds) or ``coef`` rows over a ``basis`` (``SoftResult`` /
    ``OnlineLRDSR``; the default basis is the fast backend's library).
    ``truth`` is a list of callables ``x -> f(x)``. ``other`` fixes the
    remaining input columns for multi-input laws (a row of values, the
    plotted feature is column 0).
    """
    ax = _ax(ax)
    x = np.linspace(*x_range, n)
    X = x[:, None]
    if other is not None:
        X = np.column_stack([x] + [np.full(n, v) for v in other])
    curves = []
    if models is not None:
        curves = [np.asarray(m.predict(X)).ravel() for m in models]
    elif coef is not None:
        if basis is None:
            from .core.soft import library_terms
            Phi = library_terms(X, feature_names=feature_names)[0]
        else:
            Phi = np.asarray(basis(X), dtype=float)
        curves = [Phi @ b for b in np.atleast_2d(coef)]
    for k, c in enumerate(curves):
        ax.plot(x, c, "-", color=regime_color(k), lw=2.0,
                label=labels[k] if labels is not None else f"fitted {k}")
    for k, f in enumerate(truth or []):
        ax.plot(x, f(x), "--", color="#333333", lw=1.0,
                label="true laws" if k == 0 else None)
    ax.set(xlabel="$x$", ylabel="$f(x)$")
    ax.legend(loc="best")
    return ax


def plot_mechanism_space(X_seq, y_seq, labels, ax=None, S=None, title=None, **kwargs):
    """Windows in mechanism space, projected on its two leading directions.

    ``S`` may be passed if already computed; otherwise it is
    :func:`lrdsr.core.mechanism_space.mechanism_features` of the windows
    (``kwargs`` go there). Colour is ``labels`` -- the truth, if you have it,
    to see whether the space separates the regimes at all.
    """
    from .core.mechanism_space import mechanism_features
    ax = _ax(ax, figsize=(4.2, 3.6))
    if S is None:
        X_seq = np.asarray(X_seq, dtype=float)
        if X_seq.ndim == 2:
            X_seq = X_seq[:, :, None]
        S = mechanism_features(X_seq, np.asarray(y_seq, float), **kwargs)
    Sc = S - S.mean(0)
    _, _, vt = np.linalg.svd(Sc, full_matrices=False)
    P = Sc @ vt[:2].T
    labels = np.asarray(labels, int)
    for k in np.unique(labels):
        m = labels == k
        ax.scatter(P[m, 0], P[m, 1], s=10, lw=0, alpha=0.7,
                   color=regime_color(int(k)), label=f"regime {k}")
    ax.set(xlabel="mechanism axis 1", ylabel="mechanism axis 2",
           title=title or "mechanism space")
    ax.legend(loc="best", markerscale=1.5)
    return ax


# ==========================================================================
# the fit itself
# ==========================================================================
def plot_cost_matrix(total_cost, labels, ax=None, cmap="viridis"):
    """The assignment cost ``(W, K)``, windows grouped by assigned regime.

    A clean fit shows a dark stripe per block: each window is cheap under
    its own law and dear under the rest.
    """
    ax = _ax(ax, figsize=(3.4, 4.2))
    total_cost = np.asarray(total_cost, dtype=float)
    order = np.lexsort((total_cost.min(axis=1), np.asarray(labels)))
    im = ax.imshow(total_cost[order], aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set(xlabel="regime $k$", ylabel="window (sorted)", title="assignment cost $J_{wk}$",
           xticks=range(total_cost.shape[1]))
    ax.figure.colorbar(im, ax=ax, fraction=0.05)
    return ax


def plot_history(history: pd.DataFrame, ax=None, metrics=None):
    """Learning curves: one line per column of a fit's ``history``.

    For ``GroupedDCSR`` the default columns are the changed fraction and the
    mean assignment cost; for ``SoftLRDSR`` the log-likelihood (monotone at
    temperature 1). ARI is drawn on a twin axis when the history carries it.
    """
    ax = _ax(ax)
    h = history
    if metrics is None:
        metrics = [c for c in ("loglik", "mean_assignment_cost", "changed_fraction")
                   if c in h]
    x = h["iteration"] if "iteration" in h else np.arange(len(h))
    for i, c in enumerate(metrics):
        ax.plot(x, h[c], "-o", ms=3, color=PALETTE[i], label=c)
    ax.set(xlabel="iteration")
    handles, lab = ax.get_legend_handles_labels()
    if "ARI" in h:
        ax2 = ax.twinx()
        ax2.plot(x, h["ARI"], "--", color=ROLE["oracle"], lw=1.2, label="ARI (truth)")
        ax2.set(ylabel="ARI", ylim=(-0.05, 1.05))
        ax2.spines["right"].set_visible(True)
        h2, l2 = ax2.get_legend_handles_labels()
        handles, lab = handles + h2, lab + l2
    ax.legend(handles, lab, loc="best")
    return ax


def plot_confusion(true_labels, pred_labels, ax=None, normalize: bool = True,
                   names=None):
    """Confusion matrix with predicted clusters aligned to the reference."""
    from scipy.optimize import linear_sum_assignment
    ax = _ax(ax, figsize=(3.6, 3.2))
    t = np.asarray(true_labels, int)
    p = np.asarray(pred_labels, int)
    kt, kp = t.max() + 1, p.max() + 1
    C = np.zeros((kt, kp))
    for a, b in zip(t, p, strict=True):
        if b >= 0:
            C[a, b] += 1
    r, c = linear_sum_assignment(-C)
    order = list(c[np.argsort(r)]) + [j for j in range(kp) if j not in c]
    C = C[:, order]
    if normalize:
        C = C / np.maximum(C.sum(axis=1, keepdims=True), 1)
    ax.imshow(C, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            ax.text(j, i, f"{C[i, j]:.2f}" if normalize else f"{int(C[i, j])}",
                    ha="center", va="center", fontsize=8,
                    color="white" if C[i, j] > 0.6 * C.max() else "black")
    ax.set(xlabel="predicted (aligned)", ylabel="reference",
           xticks=range(C.shape[1]), yticks=range(C.shape[0]))
    if names is not None:
        ax.set_yticklabels(names)
    return ax


def plot_responsibilities(R, ax=None, order=None, max_windows: int = 200):
    """Soft posteriors per window as stacked bars: the uncertainty, visible.

    With more than ``max_windows`` windows an evenly spaced subsample is drawn
    *before* sorting, so every regime keeps its share of the bars.
    """
    ax = _ax(ax, figsize=(7, 2.4))
    R = np.asarray(R, dtype=float)
    if order is None:
        keep = (np.linspace(0, len(R) - 1, max_windows).round().astype(int)
                if len(R) > max_windows else np.arange(len(R)))
        sub = R[keep]
        order = keep[np.lexsort((-sub.max(axis=1), sub.argmax(axis=1)))]
    R = R[np.asarray(order)][:max_windows]
    bottom = np.zeros(len(R))
    for k in range(R.shape[1]):
        ax.bar(np.arange(len(R)), R[:, k], bottom=bottom, width=1.0,
               color=regime_color(k), lw=0, label=f"regime {k}")
        bottom += R[:, k]
    ax.set(xlabel="window (sorted)", ylabel="posterior", ylim=(0, 1),
           xlim=(-0.5, len(R) - 0.5))
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7.5,
              handlelength=1.0)
    return ax


# ==========================================================================
# real time
# ==========================================================================
def plot_stream(timeline: pd.DataFrame, ax=None, truth_col: str = "truth",
                label_col: str = "label", title=None):
    """A streaming run: truth ribbon on top, assignment below, certainty line.

    ``timeline`` is what ``OnlineLRDSR.fit_stream`` returns (or any frame with
    ``t`` and a label column). Unassigned (buffered, novel) windows are grey;
    a regime birth is marked with a vertical line.
    """
    ax = _ax(ax, figsize=(8, 2.6))
    tl = timeline
    t = tl["t"].to_numpy()
    rows = [(label_col, 0.0)]
    if truth_col in tl:
        rows = [(truth_col, 1.1), (label_col, 0.0)]
    for col, y0 in rows:
        v = tl[col].to_numpy()
        ax.bar(t, 1.0, bottom=y0, width=1.0, lw=0,
               color=[regime_color(int(k)) for k in v], align="center")
    if "max_posterior" in tl:
        ax.plot(t, tl["max_posterior"] - 1.15, color="#333333", lw=0.8)
    if "spawned" in tl:
        for tt in tl.loc[tl["spawned"].notna(), "t"]:
            ax.axvline(tt, color="#222222", ls=":", lw=1.2)
    ticks = [0.5, 1.6][: len(rows)] if len(rows) == 2 else [0.5]
    names = ["assigned", "truth"] if len(rows) == 2 else ["assigned"]
    ax.set_yticks(ticks + ([-0.65] if "max_posterior" in tl else []))
    ax.set_yticklabels(names + (["certainty"] if "max_posterior" in tl else []))
    ax.set(xlabel="window (arrival order)", xlim=(t.min() - 0.5, t.max() + 0.5),
           title=title or "online clustering")
    ax.spines["left"].set_visible(False)
    return ax


def animate_stream(model, X_seq, y_seq, truth=None, x_range=None, interval: int = 120,
                   feature_names=None, every: int = 1, tail: int = 40, dpi: int = 80):
    """Animate an :class:`~lrdsr.core.online.OnlineLRDSR` running on a stream.

    Left: the latest windows' samples, coloured by the regime each was
    assigned to, with the current laws drawn over them -- the laws move as
    windows are folded in, and a new one appears when a regime is born.
    Right: the timeline, growing. Returns a ``FuncAnimation``; in a notebook
    show it with ``IPython.display.HTML(anim.to_jshtml())``.

    The model is advanced **inside** the animation, so the animation is the
    run, not a replay of one.
    """
    plt = _plt()
    from matplotlib.animation import FuncAnimation

    X_seq = np.asarray(X_seq, dtype=float)
    if X_seq.ndim == 2:
        X_seq = X_seq[:, :, None]
    y_seq = np.asarray(y_seq, dtype=float)
    lo = X_seq[..., 0].min() if x_range is None else x_range[0]
    hi = X_seq[..., 0].max() if x_range is None else x_range[1]
    grid = np.linspace(lo, hi, 200)[:, None]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(9.5, 3.3), dpi=dpi,
                                 gridspec_kw={"width_ratios": [1.1, 1.4]})
    ylo, yhi = np.percentile(y_seq, [0.5, 99.5])
    state = {"labels": [], "t": 0}

    def frame(i):
        for _ in range(every):
            if state["t"] >= len(y_seq):
                break
            a = model.partial_fit(X_seq[state["t"]], y_seq[state["t"]])
            state["labels"].append(a.label)
            state["t"] += 1
        t = state["t"]
        a0.clear()
        a1.clear()
        start = max(0, t - tail)
        for w in range(start, t):
            a0.scatter(X_seq[w, :, 0], y_seq[w], s=5, lw=0, alpha=0.5,
                       color=regime_color(state["labels"][w]))
        Phi = model._phi(grid)
        for k in range(model.n_clusters):
            a0.plot(grid[:, 0], Phi @ model._coef(k), color=regime_color(k), lw=2.2)
        a0.set(xlim=(lo, hi), ylim=(ylo, yhi), xlabel="$x$", ylabel="$y$",
               title=f"window {t}: {model.n_clusters} regimes")
        labs = np.array(state["labels"])
        a1.bar(np.arange(t), 1.0, width=1.0, lw=0,
               color=[regime_color(int(k)) for k in labs])
        if truth is not None:
            a1.bar(np.arange(t), 1.0, bottom=1.1, width=1.0, lw=0,
                   color=[regime_color(int(k)) for k in np.asarray(truth)[:t]])
            a1.set_yticks([0.5, 1.6])
            a1.set_yticklabels(["assigned", "truth"])
        else:
            a1.set_yticks([0.5])
            a1.set_yticklabels(["assigned"])
        a1.set(xlim=(-0.5, len(y_seq)), xlabel="window", title="timeline")
        return []

    n_frames = int(np.ceil(len(y_seq) / every))
    anim = FuncAnimation(fig, frame, frames=n_frames, interval=interval, blit=False,
                         repeat=False)
    plt.close(fig)
    return anim


# ==========================================================================
# theory and losses
# ==========================================================================
def plot_losses(ax=None, losses=("squared", "huber", "cauchy", "tukey", "student_t"),
                r_max: float = 6.0, influence: bool = False, params=None):
    """The loss family on a scaled residual: ``rho(r)`` or ``psi(r)``."""
    from .core.losses import loss_psi, loss_value
    ax = _ax(ax)
    r = np.linspace(-r_max, r_max, 600)
    params = params or {}
    for i, name in enumerate(losses):
        p = params.get(name, {})
        v = loss_psi(r, name, **p) if influence else loss_value(r, name, **p)
        ax.plot(r, v, color=PALETTE[i], label=name)
    ax.set(xlabel="scaled residual $r$",
           ylabel=r"influence $\psi(r)$" if influence else r"loss $\rho(r)$")
    if not influence:
        ax.set_ylim(0, min(ax.get_ylim()[1], 0.5 * r_max ** 2 * 0.6))
    ax.axhline(0, color="#999999", lw=0.6)
    ax.legend(loc="best")
    return ax


def plot_error_curve(n_rho, error, ax=None, label="measured", theory: bool = True,
                     efficiency: float = 1.0, color=None, marker="o"):
    """Window error against ``n rho``, with the ceiling ``Q(sqrt(eta n rho)/2)``."""
    from scipy.stats import norm
    ax = _ax(ax)
    n_rho = np.asarray(n_rho, dtype=float)
    ax.plot(n_rho, error, marker, color=color or PALETTE[0], label=label, ms=4)
    if theory:
        g = np.geomspace(max(n_rho.min() * 0.7, 1e-3), n_rho.max() * 1.4, 200)
        ax.plot(g, norm.sf(np.sqrt(efficiency * g) / 2), "-", color=ROLE["oracle"],
                lw=1.2, label=(r"$Q(\sqrt{n\rho}/2)$" if efficiency == 1.0
                               else rf"$Q(\sqrt{{{efficiency:.2f}\,n\rho}}/2)$"))
    ax.set(xscale="log", yscale="log", xlabel=r"$n\rho$", ylabel="window error")
    ax.legend(loc="best")
    return ax
