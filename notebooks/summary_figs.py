"""Every figure of ``00_summary``, one function each.

The notebook reads as a story, so the plotting lives here: each notebook
cell is one call. Figures that come from committed results read
``results/``; the worked example, the bike calendar and the sine problems
compute live.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display

from lrdsr import GroupedDCSR, SoftLRDSR, paths, viz
from lrdsr.core.evaluation import aligned_accuracy

RES = paths.ROOT / "results"
DOCS = paths.ROOT / "docs" / "algorithm"
CFG = {"n_clusters": 3, "alpha_geom": 0.0, "backend": "fast", "backend_kwargs": {"max_terms": 5},
       "score_mode": "cross_fit", "residual_scale": "global", "random_state": 11}
TRUE_LAWS = [lambda x: 0.8 * x**2 + 1.5 * x + 0.5, lambda x: 2.5 * np.sin(2 * x) - 0.5,
             lambda x: 0.35 * x**3 - 1.2]


def show(name: str, width: int = 1050):
    """Display one of the diagrams in ``docs/algorithm``."""
    display(Image(filename=str(DOCS / name), width=width))


# ==========================================================================
# 1 · motivation
# ==========================================================================
def motivation():
    rng = np.random.default_rng(3)

    def law_a(x):
        return x**2 - 1

    def law_b(x):
        return 1.2 * x - 0.4

    wins = [("A", law_a, (-2.0, -0.6)), ("A", law_a, (0.6, 2.0)),
            ("B", law_b, (0.3, 1.9)), ("B", law_b, (-1.9, -0.5))]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6), gridspec_kw={"width_ratios": [1.5, 1.5, 1]})
    xg = np.linspace(-2.1, 2.1, 200)
    stats = []
    for i, (tag, f, (lo, hi)) in enumerate(wins):
        x = np.sort(rng.uniform(lo, hi, 14))
        y = f(x) + rng.normal(0, 0.18, x.size)
        ax[0].plot(x, y, "o", ms=5, color="#555555")
        ax[0].annotate(f"window {i + 1}", (x.mean(), y.min() - 0.55 if i == 2 else y.max() + 0.35),
                       ha="center", fontsize=9)
        ax[1].plot(x, y, "o", ms=5, color=viz.regime_color(0 if tag == "A" else 1))
        stats.append((np.polyfit(x, y, 1)[0], y.mean(), tag, i + 1))
    ax[0].set(title="what you get: four windows, no labels", xlabel="input x", ylabel="output y", ylim=(-3.2, 4))
    ax[1].plot(xg, law_a(xg), "--", color=viz.regime_color(0), lw=1.3, label="rule A: $y = x^2 - 1$")
    ax[1].plot(xg, law_b(xg), "--", color=viz.regime_color(1), lw=1.3, label="rule B: $y = 1.2x - 0.4$")
    ax[1].set(title="the truth: 1 and 2 share a rule, 3 and 4 share the other", xlabel="input x", ylim=(-3.2, 4))
    ax[1].legend(loc="upper center")
    for s, m, tag, i in stats:
        ax[2].scatter(s, m, s=70, color=viz.regime_color(0 if tag == "A" else 1))
        ax[2].annotate(f" {i}", (s, m), fontsize=10)
    ax[2].set(title="what a 'look-alike' clustering sees", xlabel="trend (slope)", ylabel="mean level")
    fig.tight_layout()


# ==========================================================================
# 3 · the worked example
# ==========================================================================
def example():
    """The worked example's data: three laws, 240 windows of 16 noisy samples."""
    from experiments.common.fitting import window_features
    from experiments.functions.data import simulate_function_windows

    X, y, _, z, equations = simulate_function_windows(n_windows=240, window_len=16, noise_std=1.5,
                                                      geometry_overlap=3.0, seed=11)
    Zf, _ = window_features(X, y)
    return {"X": X, "y": y, "z": z, "Zf": Zf, "equations": equations}


def example_data(d):
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
    ax[0].plot(d["X"][:60, :, 0].ravel(), d["y"][:60].ravel(), "o", ms=2.5, color="#777777")
    ax[0].set(title="what you get: 60 windows, pooled, unlabelled", xlabel="x", ylabel="y")
    viz.plot_windows(d["X"], d["y"], d["z"], ax=ax[1], max_windows=60)
    ax[1].set_title("the same, coloured by the hidden law")
    viz.plot_laws(truth=TRUE_LAWS, x_range=(-2.2, 2.2), ax=ax[2])
    ax[2].set_title("the three true laws")
    fig.tight_layout()


def looks_vs_laws(d):
    z, Zf = d["z"], d["Zf"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for k in range(3):
        ax[0].scatter(Zf[z == k, 0], Zf[z == k, 5], s=12, color=viz.regime_color(k), lw=0, label=f"law {k}")
    ax[0].set(title="how the windows LOOK: two summary statistics", xlabel="window mean of y",
              ylabel="window mean of x*y")
    ax[0].legend()
    viz.plot_mechanism_space(d["X"], d["y"], z, ax=ax[1], feature_names=["x"],
                             title="their LAW COEFFICIENTS: mechanism space")
    fig.tight_layout()


def loop_rounds(d):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    bad = KMeans(3, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(d["Zf"]))
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.4), sharey=True)
    for a, rounds in zip(ax, (0, 1, 2, 10), strict=True):
        r = GroupedDCSR(init=bad, max_iter=rounds, **CFG).fit(d["X"], d["y"], d["Zf"], feature_names=["x"])
        viz.plot_laws(r.models, x_range=(-2.2, 2.2), truth=TRUE_LAWS, ax=a)
        a.get_legend().remove()
        a.set_title(f"after {rounds} round{'s' * (rounds != 1)}" if rounds < 10 else "converged", fontsize=10)
        a.text(0.03, 0.95, f"accuracy {aligned_accuracy(d['z'], r.labels):.1%}", transform=a.transAxes,
               va="top", fontsize=10, weight="bold")
    ax[0].set_ylabel("f(x)")
    fig.suptitle("from a deliberately bad start: fit laws, reassign windows, repeat", y=1.03)
    fig.tight_layout()


def the_fit(d):
    """The default fit: printed laws, and four views of the result."""
    from lrdsr import FastSymbolicRegressor

    X, y, z = d["X"], d["y"], d["z"]
    res = GroupedDCSR(init="mechanism", **CFG).fit(X, y, d["Zf"], feature_names=["x"])
    d["res"] = res
    print(f"accuracy {aligned_accuracy(z, res.labels):.1%}, in {len(res.history)} rounds")
    for k, m in enumerate(res.models):
        print(f"   law {k}:  y = {m.expression()}")
    # the greedy BIC search, replayed for the group holding the parabola
    k_par = np.bincount(res.labels[z == 0]).argmax()
    Xk, yk = X[res.labels == k_par].reshape(-1, 1), y[res.labels == k_par].ravel()
    sr = FastSymbolicRegressor(max_terms=5, feature_names=["x"])
    lib, nk = sr._make_library(Xk), len(yk)
    chosen, best = [], sr._bic(yk, np.full(nk, yk.mean()), 1)
    trace = [(0, {"constant": best}, "constant")]
    for rnd in range(1, 6):
        cands = {}
        for j, term in enumerate(lib):
            if j not in chosen:
                Phi = np.column_stack([np.ones(nk)] + [lib[c].values for c in [*chosen, j]])
                coef, *_ = np.linalg.lstsq(Phi, yk, rcond=None)
                cands[term.name] = sr._bic(yk, Phi @ coef, Phi.shape[1])
        pick = min(cands, key=cands.get)
        trace.append((rnd, cands, pick if cands[pick] < best - 1e-6 else None))
        if cands[pick] >= best - 1e-6:
            break
        best = cands[pick]
        chosen.append(next(j for j, t in enumerate(lib) if t.name == pick))

    fig, ax = plt.subplots(1, 4, figsize=(16, 3.5), gridspec_kw={"width_ratios": [1.3, 1.1, 0.9, 0.9]})
    viz.plot_laws(res.models, x_range=(-2.2, 2.2), truth=TRUE_LAWS, ax=ax[0])
    ax[0].set_title("recovered laws (solid) vs truth (dashed)")
    path = []
    for rnd, cands, pick in trace:
        ax[1].scatter([rnd] * len(cands), list(cands.values()), s=14, color="#b5b5b5", zorder=2)
        if pick is not None:
            path.append((rnd, cands[pick]))
            ax[1].annotate(("+ " if rnd else "") + pick, (rnd, cands[pick]), textcoords="offset points",
                           xytext=(6, -3), fontsize=8.5, color=viz.PALETTE[1], weight="bold")
        else:
            ax[1].annotate("stop: no term\nhelps", (rnd, min(cands.values())), textcoords="offset points",
                           xytext=(-10, 12), fontsize=8, ha="center")
    ax[1].plot(*zip(*path, strict=True), "-o", color=viz.PALETTE[1], zorder=3)
    ax[1].set(title="building one law, term by term\n(grey: every term tried)", xlabel="round",
              ylabel="BIC (lower = better)", xticks=range(len(trace)))
    viz.plot_confusion(z, res.labels, ax=ax[2])
    ax[2].set_title("true law vs found group")
    viz.plot_cost_matrix(res.total_cost, res.labels, ax=ax[3])
    ax[3].set_title("cost of each window\nunder each law")
    fig.tight_layout()


def soft_version(d):
    soft = SoftLRDSR(n_clusters=3, feature_names=["x"], random_state=11).fit(d["X"], d["y"])
    unsure = np.mean(soft.responsibilities.max(1) < 0.9)
    print(f"accuracy {aligned_accuracy(d['z'], soft.labels):.1%};  {unsure:.1%} of windows are less than 90% sure")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={"width_ratios": [1, 1.6]})
    viz.plot_laws(coef=soft.coef, feature_names=["x"], x_range=(-2.2, 2.2), truth=TRUE_LAWS, ax=ax[0])
    ax[0].set_title("laws found by the soft version")
    viz.plot_responsibilities(soft.responsibilities, ax=ax[1])
    ax[1].set_title("probability of each law, one column per window")
    fig.tight_layout()


def noise_sweep():
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    from experiments.common.fitting import window_features
    from experiments.functions.data import simulate_function_windows

    rows = []
    for noise in (1.5, 3.0, 4.5, 6.0):
        Xn, yn, _, zn, _ = simulate_function_windows(n_windows=240, window_len=16, noise_std=noise,
                                                     geometry_overlap=3.0, seed=11)
        Zn, _ = window_features(Xn, yn)
        orc = np.column_stack([((yn - f(Xn[:, :, 0])) ** 2).sum(1) for f in TRUE_LAWS]).argmin(1)
        fit = GroupedDCSR(init="mechanism", **CFG).fit(Xn, yn, Zn, feature_names=["x"])
        km = KMeans(3, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(Zn))
        rows.append({"noise": noise, "oracle (knows the laws)": np.mean(orc == zn),
                     "LR-DSR (no labels)": aligned_accuracy(zn, fit.labels),
                     "K-means on look-alike statistics": aligned_accuracy(zn, km)})
    acc = pd.DataFrame(rows).set_index("noise")
    ax = acc.plot.bar(figsize=(9, 3.4), color=["#222222", viz.PALETTE[0], "#9a9a9a"], width=0.8, rot=0)
    ax.set(ylim=(0.3, 1.05), ylabel="accuracy", xlabel="noise level (standard deviation)",
           title="as the noise grows, the method stays with the best possible")
    ax.axhline(1 / 3, color="k", ls=":", lw=1)
    ax.text(3.42, 0.34, "chance", fontsize=8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8.5)


# ==========================================================================
# 4 · results read from results/
# ==========================================================================
def result(name: str):
    """A figure of the project's analysis, drawn from the committed results."""
    from analysis import plots

    plots.all_figures()[name]()


def sr_baseline():
    SB = RES / "srbaseline"
    A = pd.read_csv(SB / "estimation_assignment.csv")
    L = pd.read_csv(SB / "estimation_law.csv")
    arms = [("oracle", "oracle (knows the laws)", "#222222", "--"), ("lrdsr", "LR-DSR", viz.PALETTE[0], "-"),
            ("sr_per_window", "SR on each window, then cluster", viz.PALETTE[1], "-"),
            ("sr_pooled", "one SR law for everything", viz.PALETTE[4], "-")]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    for key, lab, c, ls in arms:
        ax[0].plot(A.rho, A[f"{key}_error"], ls, marker="o", color=c, label=lab)
        ax[1].plot(L.rho, L[f"{key}_law_error"], ls, marker="o", color=c, label=lab)
    ax[0].set(xscale="log", title="which window came from which law?", xlabel="how far apart the laws are  →",
              ylabel="share of windows wrong")
    ax[1].set(xscale="log", title="how close are the formulas found?", xlabel="how far apart the laws are  →",
              ylabel="law error (1 = as far as the two laws)")
    ax[0].legend(fontsize=8)
    fig.tight_layout()


# ==========================================================================
# 5 · real data: bike sharing
# ==========================================================================
def bike():
    from experiments.realdata.run import fourier
    from experiments.realdata.windows import load

    b = load("bike")
    sb = SoftLRDSR(2, basis=fourier, noise="student_t", random_state=11).fit(b.X_seq, b.y_seq)
    lab = sb.labels if np.mean(sb.labels == b.reference) > 0.5 else 1 - sb.labels   # 1 = working-day law
    print(f"{len(lab)} days;  agreement with the calendar: {np.mean(lab == b.reference):.1%}")
    return {"data": b, "labels": lab, "fit": sb}


def bike_days(bk):
    b, lab = bk["data"], bk["labels"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.5))
    hours = np.arange(24)
    for k, name in ((1, "commuting law (two peaks)"), (0, "leisure law (one hump)")):
        sel = np.flatnonzero(lab == k)
        ax[0].plot(hours, b.y_seq[sel[:200]].T, color=viz.regime_color(1 - k), lw=0.3, alpha=0.25)
        ax[0].plot(hours, b.y_seq[sel].mean(0), color=viz.regime_color(1 - k), lw=2.8, label=name)
    ax[0].set(title="every day, coloured by the law the method found", xlabel="hour of day",
              ylabel="rentals (log, day mean removed)", xticks=range(0, 25, 3))
    ax[0].legend(loc="lower center")
    ax[1].hist(bk["fit"].responsibilities.max(1), bins=30, color=viz.PALETTE[0])
    ax[1].set(title="how sure the method is, per day", xlabel="probability of the chosen law",
              ylabel="days", yscale="log")
    fig.tight_layout()


def bike_calendar(bk):
    from matplotlib.colors import ListedColormap

    b, lab = bk["data"], bk["labels"]
    dates = pd.to_datetime(pd.Series(b.dates))
    fig, axes = plt.subplots(2, 1, figsize=(15, 4.4))
    cmap = ListedColormap([viz.regime_color(1), viz.regime_color(0)])
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for a, year in zip(axes, sorted(dates.dt.year.unique()), strict=True):
        m = (dates.dt.year == year).to_numpy()
        d = dates[m]
        wk = d.dt.isocalendar().week.to_numpy().astype(int)
        dow = d.dt.dayofweek.to_numpy()
        wk = np.where((d.dt.month == 1).to_numpy() & (wk > 50), 0, wk)
        grid = np.full((7, 54), np.nan)
        grid[dow, wk] = lab[m]
        a.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="none")
        odd = lab[m] != b.reference[m]
        a.scatter(wk[odd], dow[odd], marker="x", color="k", s=28, lw=1.3)
        a.set_yticks(range(7))
        a.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], fontsize=8)
        starts = d.groupby(d.dt.month).apply(lambda s: s.dt.isocalendar().week.iloc[0]).to_numpy().astype(int)
        starts[0] = 0
        a.set_xticks(starts)
        a.set_xticklabels(months, fontsize=8)
        a.set_ylabel(str(year))
    axes[0].set_title("blue = commuting law, orange = leisure law;  x = the law disagrees with the calendar")
    fig.tight_layout()


# ==========================================================================
# 6 · what did not work
# ==========================================================================
def fast_oscillations():
    """Failure 1 live: every method on the same windows, noisy and clean."""
    from experiments.common.fitting import window_features
    from experiments.problems import zoo
    from experiments.problems.sinusoid_trial import (
        FREQ_GRID,
        _kmeans,
        _periodogram,
        _sr_per_window,
    )
    from experiments.problems.sinusoid_trial import fit as fit_hf
    from lrdsr.core.mechanism_space import mechanism_init

    prob = zoo.PROBLEM_BY_NAME["high_frequency"]
    truth = [lambda x: np.sin(4 * x), lambda x: np.sin(4.6 * x)]
    grid = zoo.eval_grid(prob, 200)
    methods = [("oracle", "oracle (knows the laws)", "#222222"),
               ("lrdsr_sin", "LR-DSR + fitted frequency", viz.PALETTE[0]),
               ("mech", "mechanism K-means", "#6f9fd8"), ("soft", "soft EM (library)", "#6f9fd8"),
               ("lrdsr", "LR-DSR, standard library", "#6f9fd8"),
               ("pgram", "periodogram + K-means", viz.PALETTE[1]),
               ("peak", "peak frequency + K-means", viz.PALETTE[1]),
               ("sr_sin", "SR per window (fitted freq.)", viz.PALETTE[2]),
               ("sr", "SR per window (library)", viz.PALETTE[2]),
               ("profile", "K-means, raw curve", "#9a9a9a"), ("summ", "K-means, window summaries", "#9a9a9a")]
    runs = {}
    for rho in (0.25, 1.0):
        X, y, z = zoo.make_windows(prob, zoo.sigma_for_rho(prob, rho), n_windows=240, window_len=32, seed=11)
        Z, _ = window_features(X, y)
        P = _periodogram(X, y)
        lib_fit, sin_fit = (fit_hf(be, X, y, Z, ["x"], 2, 11) for be in ("fast", "fast_sin"))
        labels = {"oracle": zoo.oracle_labels(prob, X, y), "lrdsr_sin": sin_fit.labels, "lrdsr": lib_fit.labels,
                  "soft": SoftLRDSR(2, feature_names=["x"], random_state=11).fit(X, y).labels,
                  "mech": mechanism_init(X, y, 2, seed=11, feature_names=["x"]),
                  "pgram": _kmeans(P, 2, 11), "peak": _kmeans(FREQ_GRID[P.argmax(1)], 2, 11),
                  "sr_sin": _sr_per_window("fast_sin", X, y, ["x"], grid, 2, 11),
                  "sr": _sr_per_window("fast", X, y, ["x"], grid, 2, 11),
                  "profile": _kmeans(np.take_along_axis(y, np.argsort(X[..., 0], 1), 1), 2, 11),
                  "summ": _kmeans(Z, 2, 11)}
        runs[rho] = {"acc": {k: aligned_accuracy(z, v) for k, v in labels.items()},
                     "fits": (lib_fit, sin_fit), "P": P, "z": z}
    for tag, m_ in zip(("standard library", "+ fitted frequency"), runs[0.25]["fits"], strict=True):
        print(f"LR-DSR, {tag}:  " + "   |   ".join(f"y = {m.expression()}" for m in m_.models))

    fig = plt.figure(figsize=(17, 13))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1.25, 1], hspace=0.45, wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(*prob.x_range, 400)
    for k, (f, lab) in enumerate(zip(prob.laws, prob.truths, strict=True)):
        ax.plot(xs, f(xs[:, None]), color=viz.regime_color(k), lw=1.8, label=lab)
    ax.set(title="the two true laws", xlabel="x")
    ax.legend(loc="lower left")
    for j, (tag, m_) in enumerate(zip(("standard library", "+ fitted frequency"), runs[0.25]["fits"],
                                      strict=True)):
        ax = fig.add_subplot(gs[0, j + 1])
        viz.plot_laws(m_.models, x_range=prob.x_range, truth=truth, ax=ax)
        ax.get_legend().remove()
        ax.set_ylim(-1.9, 1.9)
        ax.set_title(f"found by LR-DSR, {tag} (solid)")
    for j, (rho, ttl) in enumerate(((0.25, "noisy windows"), (1.0, "clean windows"))):
        ax = fig.add_subplot(gs[1, 0:2] if j == 0 else gs[1, 2])
        acc = runs[rho]["acc"]
        yp = np.arange(len(methods))[::-1]
        ax.barh(yp, [acc[k] for k, _, _ in methods], color=[c for _, _, c in methods], height=0.72)
        for y_i, (k, _, _) in zip(yp, methods, strict=True):
            ax.text(acc[k] + 0.005, y_i, f"{acc[k]:.1%}", va="center", fontsize=8.5)
        ax.axvline(0.5, color="k", ls=":", lw=0.8)
        ax.set_yticks(yp)
        ax.set_yticklabels([n for _, n, _ in methods] if j == 0 else [], fontsize=9)
        ax.set(xlim=(0.45, 1.06), xlabel="accuracy  (dotted: chance)", title=ttl)
    for j, rho in enumerate((0.25, 1.0)):
        ax = fig.add_subplot(gs[2, j])
        P, z = runs[rho]["P"], runs[rho]["z"]
        for k in range(2):
            for w in np.flatnonzero(z == k)[:6]:
                ax.plot(FREQ_GRID, P[w], color=viz.regime_color(k), lw=0.6, alpha=0.5)
            ax.plot(FREQ_GRID, P[z == k].mean(0), color=viz.regime_color(k), lw=2.4,
                    label=f"{prob.truths[k]}: average")
        for t in (4.0, 4.6):
            ax.axvline(t, color="#333333", ls="--", lw=0.9)
        ax.set(xlabel="frequency", ylabel="periodogram power", xlim=(2, 7),
               title=f"what a periodogram sees, {'noisy' if rho < 1 else 'clean'} (thin: single windows)")
        ax.legend(fontsize=8, loc="upper left")
    ax = fig.add_subplot(gs[2, 2])
    freqs = sorted(t[2] for m in runs[0.25]["fits"][1].models for t in m.terms_ if t[0] != "lib")
    peaks = FREQ_GRID[runs[0.25]["P"].argmax(1)]
    for k in range(2):
        ax.hist(peaks[runs[0.25]["z"] == k], bins=np.arange(2, 7.01, 0.1), color=viz.regime_color(k),
                alpha=0.6, label=f"one window's peak, {prob.truths[k]}")
    for f in freqs:
        ax.axvline(f, color=viz.PALETTE[0], lw=2.5)
    ax.set(xlabel="frequency", ylabel="windows", xlim=(2, 7),
           title="one window at a time vs all windows pooled (blue)")
    ax.legend(fontsize=7.5, loc="upper right")


def fitted_frequency_zoo():
    T = pd.read_csv(RES / "problems" / "sinusoid_trial_summary.csv").sort_values("gap_fast")
    yp = np.arange(len(T))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(yp + 0.2, T.gap_fast, 0.4, color="#9a9a9a", label="standard library")
    ax.barh(yp - 0.2, T.gap_fast_sin, 0.4, color=viz.PALETTE[0], label="+ fitted frequency")
    ax.set_yticks(yp)
    ax.set_yticklabels(T.problem, fontsize=8.5)
    ax.set(xlabel="distance from the best possible (lower = better)",
           title="all twelve problems: only the fast oscillation changes")
    ax.legend(loc="lower right")
    fig.tight_layout()


def correlated_noise_intuition():
    from experiments.robustness.run import ar1

    rng = np.random.default_rng(5)
    xs = np.linspace(-2, 2, 48)
    u = rng.normal(size=48)
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.1), sharey=True)
    for a, phi, ttl in zip(ax, (0.0, 0.6, 0.95),
                           ("independent noise (what the method assumes)", "noise that wanders a little",
                            "noise that wanders a lot"), strict=True):
        a.plot(xs, xs**2 - 1, "--", color="#333333", lw=1.2, label="the law")
        a.plot(xs, xs**2 - 1 + 0.8 * ar1(u, phi), "o-", ms=3, lw=0.8, color=viz.PALETTE[0], label="one window")
        a.set(title=ttl, xlabel="x  (samples in time order)")
    ax[0].set_ylabel("y")
    ax[0].legend(loc="upper center")
    fig.suptitle("same law, same noise size, same random numbers: only how the noise wanders differs", y=1.03)
    fig.tight_layout()


# ==========================================================================
# 7 · every task in one picture
# ==========================================================================
#: (question, verdict, what the bars measure, [(label, value key, role)])
def scorecard():
    from matplotlib.patches import Patch

    from analysis.report import numbers

    N = numbers()
    sa = pd.read_csv(RES / "srbaseline" / "estimation_assignment.csv").set_index("rho").loc[0.25]
    N.update({"_sr_oracle": sa.oracle_error, "_sr_lrdsr": sa.lrdsr_error,
              "_sr_window": sa.sr_per_window_error, "_sr_pooled": sa.sr_pooled_error,
              "_v1": N["v1_cells"] / 18, "_v4": N["v4_unchanged"] / N["v4_cells"],
              "_v10a": N["v10_fixed_in"] / N["v10_cells"], "_v10b": N["v10x_in"] / N["v10x_n"]})
    HOLD, PART = viz.PALETTE[2], viz.PALETTE[1]
    ME, BASE, REF = viz.PALETTE[0], "#9a9a9a", "#222222"
    panels = [
        ("1 · Is there a best possible error?", HOLD, "share where formula = measured",
         [("simple case", "_v1", "me"), ("shared part", "_v4", "me")]),
        ("2 · Reached without labels?", HOLD, "share of windows wrong",
         [("best possible", "bench_oracle", "ref"), ("our start", "bench_mechanism_kmeans", "me"),
          ("LR-DSR", "bench_lrdsr", "me"), ("look-alike", "bench_window_features", "base")]),
        ("3 · What does the loop add?", PART, "share wrong: start → after the loop",
         [("random", "loop_random_in", "base"), ("+loop", "loop_random_out", "me"),
          ("K-means", "loop_kmeans_in", "base"), ("+loop", "loop_kmeans_out", "me"),
          ("ours", "loop_mechanism_in", "base"), ("+loop", "loop_mechanism_out", "me")]),
        ("4 · Twelve hard problems?", HOLD, "distance from best possible",
         [("soft EM", "zoo_gap_soft_em", "me"), ("our start", "zoo_gap_mechanism_kmeans", "me"),
          ("raw curve", "zoo_gap_profile_kmeans", "base"), ("look-alike", "zoo_gap_geometry", "base")]),
        ("5 · Outliers in the noise?", HOLD, "share of windows wrong",
         [("best possible", "le_student_t3_oracle_lrt", "ref"), ("learned loss", "le_student_t3_hard_learned", "me"),
          ("Huber", "le_student_t3_hard_huber", "me"), ("squared", "le_student_t3_hard_squared", "base")]),
        ("6 · Switches seen in real time?", HOLD, "delay in samples",
         [("predicted", "cusum_delay_pred", "ref"), ("measured", "cusum_delay_learned", "me")]),
        ("7 · Real days: weekday vs weekend?", PART, "agreement (higher = better)",
         [("soft EM", "rd_traffic_soft_fourier_ari", "me"), ("LR-DSR", "rd_traffic_lrdsr_ari", "me"),
          ("raw curve", "rd_traffic_profile_ari", "base"), ("look-alike", "rd_traffic_geom_ari", "base")]),
        ("8 · Days with missing hours?", HOLD, "accuracy (higher = better)",
         [("by the law", "gap_lo_law", "me"), ("fill: mean", "gap_lo_mean", "base"),
          ("fill: line", "gap_lo_lin", "base")]),
        ("9 · Predict a partial day's error?", PART, "share decided by 5 am",
         [("predicted", "v11_traffic_5h_pred", "ref"), ("measured", "v11_traffic_5h_obs", "me"),
          ("err. pred.", "v11_cal_mid_pred", "ref"), ("err. meas.", "v11_cal_mid_obs", "me")]),
        ("10 · A law outside the library?", PART, "distance from best possible",
         [("library", "k_hf_soft_em_library", "base"), ("kernel basis", "k_hf_soft_em_kernel", "me")]),
        ("11 · Classification: error predicted?", HOLD, "share where formula = measured",
         [("fixed inputs", "_v10a", "me"), ("random inputs", "_v10b", "me")]),
        ("12 · 24 standard datasets?", PART, "share of series wrong",
         [("by law", "ucr_shape_law_vs_1nn_a", "me"), ("nearest raw", "ucr_shape_law_vs_1nn_b", "base"),
          ("best tuned", "ucr_shape_best_b", "base")]),
        ("13 · Repair: shifted in time", PART, "share wrong (GunPoint)",
         [("by law", "rp_GunPoint_law", "base"), ("+ shift", "rp_GunPoint_shift", "me"),
          ("DTW", "rp_GunPoint_dtw", "base")]),
        ("14 · Streaming days with gaps?", HOLD, "accuracy (higher = better)",
         [("partial days", "st_partial_acc", "me"), ("complete days", "st_complete_acc", "me")]),
        ("15 · Wind: cold vs warm air?", PART, "balanced error",
         [("best possible", "w_cold_ceiling", "ref"), ("by law", "w_cold_law", "me"),
          ("binned curve", "w_cold_bins_best", "base")]),
        ("16 · Better than plain SR?", HOLD, "share of windows wrong",
         [("best possible", "_sr_oracle", "ref"), ("LR-DSR", "_sr_lrdsr", "me"),
          ("SR per window", "_sr_window", "base"), ("one SR law", "_sr_pooled", "base")]),
    ]
    role = {"me": ME, "base": BASE, "ref": REF}
    fig, axes = plt.subplots(4, 4, figsize=(17, 13.5))
    for a, (title, verdict, ylab, bars) in zip(axes.flat, panels, strict=True):
        xp = np.arange(len(bars))
        vals = [float(N[k]) for _, k, _ in bars]
        a.bar(xp, vals, color=[role[r] for _, _, r in bars], width=0.66)
        for i, v in enumerate(vals):
            a.text(i, v, f"{v:.3f}" if v < 1 else f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        a.set_xticks(xp)
        a.set_xticklabels([lab for lab, _, _ in bars], fontsize=8, rotation=18 if len(bars) < 6 else 0)
        a.set_title(title, fontsize=10.5, weight="bold")
        a.set_ylabel(ylab, fontsize=8.5)
        a.set_ylim(0, max(vals) * 1.22)
        for s in a.spines.values():
            s.set_visible(True)
            s.set_color(verdict)
            s.set_linewidth(3)
    axes.flat[8].text(2.5, 0.35, "the timing is right;\nthe error level\nis off by ~3x", ha="center", fontsize=8.5)
    fig.legend(handles=[Patch(color=ME, label="this method"), Patch(color=BASE, label="alternative / before"),
                        Patch(color=REF, label="best possible / prediction"),
                        Patch(fc="white", ec=HOLD, lw=3, label="answer: yes"),
                        Patch(fc="white", ec=PART, lw=3, label="answer: partly")],
               loc="upper center", ncol=5, fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.98))
