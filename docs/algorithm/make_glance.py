"""One figure: every block of LR-DSR, its part of the objective, what it does to
the data, and its default and options.

    python docs/algorithm/make_glance.py      # writes 00_algorithm_at_a_glance.png

The data panels are computed live on the worked example of the summary
notebook (three laws, 240 windows of 16 samples). The block-5 panel uses a
law PySR actually returned in ``experiments.openlaws`` (damped, seed 23).
"""
# ruff: noqa: RUF001
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.common.fitting import window_features  # noqa: E402
from experiments.functions.data import simulate_function_windows  # noqa: E402
from lrdsr import GroupedDCSR, viz  # noqa: E402
from lrdsr.core.losses import noise_scale  # noqa: E402
from lrdsr.core.mechanism_space import mechanism_features, mechanism_init  # noqa: E402

OUT = Path(__file__).resolve().parent / "00_algorithm_at_a_glance.png"
BLUE, BLUE_FILL, ORANGE, ORANGE_FILL = "#1f5fa8", "#dce8f5", "#d9822b", "#fbe5d0"
GREY, INK = "#8a8a8a", "#222222"
viz.style()
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})

# ---------------------------------------------------------------- live data
X, y, _, z, _ = simulate_function_windows(n_windows=240, window_len=16, noise_std=1.5,
                                          geometry_overlap=3.0, seed=11)
Zf, _ = window_features(X, y)
S = mechanism_features(X, y, feature_names=["x"])
start = mechanism_init(X, y, 3, seed=11, feature_names=["x"])
res = GroupedDCSR(n_clusters=3, alpha_geom=0.0, init="mechanism", random_state=11).fit(
    X, y, Zf, feature_names=["x"], true_labels_for_eval=z)
resid = np.concatenate([y[w] - res.models[res.labels[w]].predict(X[w]) for w in range(len(y))])
s_hat = noise_scale(resid)
P = np.linalg.svd(S - S.mean(0), full_matrices=False)[2][:2]
S2 = (S - S.mean(0)) @ P.T

# ---------------------------------------------------------------- layout
BLOCKS = [
    ("0", "START", "mechanism space\n+ K-means", BLUE),
    ("1", "FIT", "one law per group", BLUE),
    ("2", "NOISE SCALE", "how big is the noise?", BLUE),
    ("3", "SCORE", "every window × every law", BLUE),
    ("4", "ASSIGN", "window → cheapest law", BLUE),
    ("5", "REFINE", "optional: deeper search", GREY),
]
PIECES = [
    r"$z^{(0)} = \mathrm{KMeans}(s_w)$",
    r"$\min_{f_k}\; \mathrm{BIC}(f_k \mid z)$",
    r"$\hat{s} = 1.48 \times \mathrm{median}\,|\,y - f_{z}(x)\,|$",
    r"$J_{wk} = \overline{\rho}\left(\frac{y_w - f_k}{\hat{s}}\right) + \beta\, C(f_k)$",
    r"$z_w = \arg\min_k\, J_{wk}$",
    r"$f_k \leftarrow g_k\ \ \mathrm{if}\ J(g_k) < J(f_k)$",
]
CHIPS = [  # (label, kind): kind = default / option / trial / off
    [("mechanism", "default"), ("kmeans", "option"), ("gmm", "option"), ("bgmm", "option"),
     ("fcm", "option"), ("given labels", "option"), ("basis: library", "default"),
     ("Fourier / kernel", "option")],
    [("fixed terms", "default"), ("+ fitted terms", "trial"), ("known law", "option"),
     ("known form", "option"), ("superposition", "option"), ("≤ 5 terms", "default")],
    [("global", "default"), ("per window", "option"), ("learned noise", "option")],
    [("Huber", "default"), ("squared", "option"), ("Cauchy", "option"), ("Tukey", "option"),
     ("Student-t", "option"), ("learned", "option"), ("cross-fit", "default"), ("in-sample", "option"),
     ("β = 0.002", "default"), ("α = 0", "default")],
    [("hard argmin", "default"), ("soft EM", "option"), ("online", "option"), ("CUSUM", "option"),
     ("classifier", "option"), ("stop < 1%", "default")],
    [("off", "default"), ("PySR", "option"), ("held-out check", "default")],
]
CHIP_STYLE = {"default": (BLUE_FILL, BLUE, "-"), "option": ("white", BLUE, "-"),
              "trial": (ORANGE_FILL, ORANGE, "-"), "off": ("#f1f1f1", GREY, "--")}

fig = plt.figure(figsize=(26, 13.2))
gs = fig.add_gridspec(4, 6, height_ratios=[0.55, 0.42, 1.55, 1.05], hspace=0.18, wspace=0.28,
                      left=0.075, right=0.99, top=0.84, bottom=0.03)
fig.text(0.075, 0.965, "LR-DSR at a glance", fontsize=26, weight="bold", color=BLUE)
fig.text(0.075, 0.925, r"objective:   $\min_{z,\,f}\ \sum_w \ \overline{\rho}\left(\frac{y_w - f_{z_w}(x_w)}"
         r"{\hat{s}}\right) + \beta\, C(f_{z_w})$        blocks 1-4 repeat until < 1% of windows move",
         fontsize=16, color=INK)
fig.text(0.075, 0.885, "the DETAILED view: each column, top to bottom = block · its piece of the objective · "
         "what it does to the data · default and options",
         fontsize=12, color="#555555")
ROW_LABELS = ["block", "objective", "data", "default\n& options"]

for c, (num, name, role, col) in enumerate(BLOCKS):
    # row 0: the block
    a = fig.add_subplot(gs[0, c])
    a.axis("off")
    a.set(xlim=(0, 1), ylim=(0, 1))
    optional = num == "5"
    a.add_patch(FancyBboxPatch((0.02, 0.05), 0.96, 0.9, boxstyle="round,pad=0.02,rounding_size=0.08",
                               fc="#f1f1f1" if optional else BLUE_FILL, ec=col, lw=2.2,
                               ls="--" if optional else "-"))
    a.text(0.5, 0.64, f"{num} · {name}", ha="center", va="center", fontsize=17, weight="bold", color=col)
    a.text(0.5, 0.27, role, ha="center", va="center", fontsize=11.5, color=INK)
    # row 1: the objective piece
    a = fig.add_subplot(gs[1, c])
    a.axis("off")
    a.text(0.5, 0.5, PIECES[c], ha="center", va="center", fontsize=15, color=INK)

# loop bracket over blocks 1-4
fig.patches.append(FancyBboxPatch((0.232, 0.705), 0.605, 0.155, transform=fig.transFigure,
                                  boxstyle="round,pad=0.004,rounding_size=0.01", fc="none",
                                  ec=ORANGE, lw=1.6, ls="-", zorder=0))
fig.text(0.535, 0.868, "THE LOOP", ha="center", fontsize=12, weight="bold", color=ORANGE)

# ---------------------------------------------------------------- row 2: the data
cols = [viz.regime_color(k) for k in range(3)]
# 0: mechanism space, starting labels
a = fig.add_subplot(gs[2, 0])
for k in range(3):
    m = start == k
    a.scatter(S2[m, 0], S2[m, 1], s=10, color=cols[k], lw=0, alpha=0.8)
a.set(title="windows as law coefficients", xticks=[], yticks=[], xlabel="mechanism axis 1",
      ylabel="mechanism axis 2")
# 1: one group's pooled data + its law
a = fig.add_subplot(gs[2, 1])
k1 = int(np.bincount(res.labels[z == 1]).argmax())
xs = X[res.labels == k1, :, 0].ravel()
a.scatter(xs, y[res.labels == k1].ravel(), s=4, color="#9a9a9a", lw=0, alpha=0.6)
xg = np.linspace(-2.2, 2.2, 200)
a.plot(xg, res.models[k1].predict(xg[:, None]), color=cols[k1], lw=3)
a.set(title="pooled windows → one formula", xticks=[], yticks=[], xlabel="x", ylabel="y")
# 2: residuals and the shared scale
a = fig.add_subplot(gs[2, 2])
a.hist(resid, bins=50, color="#9a9a9a")
for sgn in (-1, 1):
    a.axvline(sgn * s_hat, color=BLUE, lw=2.5)
a.text(s_hat * 1.08, a.get_ylim()[1] * 0.85, "ŝ", color=BLUE, fontsize=15, weight="bold")
a.set(title="misfit of every window under its current law\nŝ = its typical size (1.48 × median, = 1 sd)",
      yticks=[], xlabel="y − f(x)")
# 3: cost matrix
a = fig.add_subplot(gs[2, 3])
order = np.argsort(res.labels, kind="stable")
a.imshow(res.total_cost[order], aspect="auto", cmap="viridis", interpolation="nearest")
a.set(title="cost J[w, k]  (dark = cheap)", xticks=range(3), xticklabels=["law 0", "law 1", "law 2"],
      yticks=[], ylabel="windows (sorted)")
# 4: convergence
a = fig.add_subplot(gs[2, 4])
h = res.history
a.bar(h.iteration + 1, 100 * h.changed_fraction, color=ORANGE, width=0.5)
a.axhline(1, color="k", ls=":", lw=1.2)
a.text(0.55, 1.08, "stop line: 1%", fontsize=10)
a.set(title="windows that moved, per round", xlabel="round", ylabel="% of windows",
      xticks=range(1, len(h) + 1), xlim=(0.4, len(h) + 0.6),
      ylim=(0, max(2.0, 100 * h.changed_fraction.max() * 1.3)))
# 5: loop law vs a deeper law (a law PySR returned in experiments.openlaws)
a = fig.add_subplot(gs[2, 5])
rng = np.random.default_rng(0)
xd = rng.uniform(0, 4, 400)
truth = np.exp(-0.4 * xd) * np.sin(3 * xd)
a.scatter(xd, truth + rng.normal(0, 0.15, xd.size), s=4, color="#9a9a9a", lw=0, alpha=0.6)
xg2 = np.linspace(0, 4, 300)
from lrdsr.core.backends import make_symbolic_regressor  # noqa: E402

loop_law = make_symbolic_regressor("fast_sin", feature_names=["x"]).fit(
    xd[:, None], truth + rng.normal(0, 0.15, xd.size))
a.plot(xg2, loop_law.predict(xg2[:, None]), color=BLUE, lw=2, ls="--", label="loop engine: sum of sines")
a.plot(xg2, np.exp(-0.37539 * xg2) * np.sin(xg2 / 0.3348922), color=ORANGE, lw=2.6,
       label="PySR: e^(−0.38x)·sin(2.99x)")
a.plot(xg2, np.exp(-0.4 * xg2) * np.sin(3 * xg2), color="k", lw=1, ls=":", label="truth")
a.legend(fontsize=9, loc="upper right")
a.set(title="deeper search, kept only if cheaper", xticks=[], yticks=[], xlabel="x", ylabel="y")

# ---------------------------------------------------------------- row 3: defaults and options
for c, chips in enumerate(CHIPS):
    a = fig.add_subplot(gs[3, c])
    a.axis("off")
    a.set(xlim=(0, 1), ylim=(0, 1))
    per_row, cw, ch = 2, 0.47, 0.135
    for i, (lab, kind) in enumerate(chips):
        fc, ec, ls = CHIP_STYLE[kind]
        x0 = 0.02 + (i % per_row) * (cw + 0.03)
        y0 = 0.82 - (i // per_row) * (ch + 0.035)
        a.add_patch(FancyBboxPatch((x0, y0), cw, ch, boxstyle="round,pad=0.01,rounding_size=0.04",
                                   fc=fc, ec=ec, ls=ls, lw=1.5 if kind != "option" else 1.0))
        a.text(x0 + cw / 2, y0 + ch / 2, lab, ha="center", va="center", fontsize=11,
               weight="bold" if kind == "default" else "normal", color=INK)

# legend for the chips
lx = 0.755
for i, (lab, kind) in enumerate([("default", "default"), ("option", "option"),
                                 ("trial", "trial")]):
    fc, ec, ls = CHIP_STYLE[kind]
    fig.patches.append(FancyBboxPatch((lx + i * 0.08, 0.887), 0.018, 0.018, transform=fig.transFigure,
                                      boxstyle="round,pad=0.002", fc=fc, ec=ec, lw=1.4))
    fig.text(lx + i * 0.08 + 0.023, 0.896, lab, fontsize=11, va="center")

# row labels, aligned to the rows actually drawn
for r, lab in enumerate(ROW_LABELS):
    pos = gs[r, 0].get_position(fig)
    fig.text(0.012, (pos.y0 + pos.y1) / 2, lab, fontsize=13, weight="bold", color="#555555", va="center")

fig.savefig(OUT, dpi=130, facecolor="white")
print("wrote", OUT)
