"""Two high-level figures: the idea, and the vocabulary laws are built from.

    python docs/algorithm/make_overview.py
        -> 00_the_idea.png, 00_vocabulary.png

The idea figure is computed live on the worked example (three laws, 240
windows). The vocabulary figure draws the actual term library of the code.
"""
# ruff: noqa: RUF001
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.common.fitting import window_features  # noqa: E402
from experiments.functions.data import simulate_function_windows  # noqa: E402
from lrdsr import GroupedDCSR, viz  # noqa: E402
from lrdsr.core.soft import library_terms  # noqa: E402

HERE = Path(__file__).resolve().parent
BLUE, BLUE_FILL, ORANGE, ORANGE_FILL = "#1f5fa8", "#dce8f5", "#d9822b", "#fbe5d0"
GREEN_FILL, GREY, INK = "#dcefdc", "#8a8a8a", "#222222"
viz.style()
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})


def panel(fig, rect, fc="white", ec="#cccccc", lw=1.2, ls="-"):
    fig.patches.append(FancyBboxPatch(rect[:2], rect[2], rect[3], transform=fig.transFigure,
                                      boxstyle="round,pad=0.004,rounding_size=0.012",
                                      fc=fc, ec=ec, lw=lw, ls=ls, zorder=-5))


def farrow(fig, p, q, color=INK, rad=0.0, lw=2.2):
    fig.patches.append(FancyArrowPatch(p, q, transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=20, color=color, lw=lw,
                                       connectionstyle=f"arc3,rad={rad}"))


# ==========================================================================
# 00_the_idea
# ==========================================================================
def the_idea():
    X, y, _, _z, _ = simulate_function_windows(n_windows=240, window_len=16, noise_std=1.5,
                                              geometry_overlap=3.0, seed=11)
    Zf, _ = window_features(X, y)
    res = GroupedDCSR(n_clusters=3, alpha_geom=0.0, init="mechanism", random_state=11).fit(
        X, y, Zf, feature_names=["x"])
    lab = res.labels
    cols = [viz.regime_color(k) for k in range(3)]
    xg = np.linspace(-2.2, 2.2, 200)
    def tidy(expr):
        """The loop's own expression, rounded to 2 decimals and written plainly."""
        out = re.sub(r"(-?\d+\.\d+)", lambda m: f"{float(m.group(1)):.2f}", expr)
        for a, b in (("*(x)^3", "x³"), ("*(x)^2", "x²"), ("*x", "x"), ("*sin(2x)", "·sin 2x"),
                     ("*sin(x)", "·sin x"), ("*cos(x)", "·cos x"), ("+-", "− "), ("+", "+ "), (" -", " − ")):
            out = out.replace(a, b)
        return out.replace("-", "−")
    nice = [tidy(m.expression()) for m in res.models]

    fig = plt.figure(figsize=(22, 9.6))
    fig.text(0.02, 0.95, "LR-DSR: the idea", fontsize=26, weight="bold", color=BLUE)
    fig.text(0.02, 0.905, "Many short windows of data. Each was made by one of a few hidden laws. "
             "Find which law made each window, and write each law as a formula.", fontsize=14, color="#444444")

    # --- DATA
    panel(fig, (0.015, 0.08, 0.2, 0.76), fc="#fafafa")
    fig.text(0.115, 0.8, "1 · YOUR DATA", ha="center", fontsize=15, weight="bold")
    fig.text(0.115, 0.765, "short windows, no labels", ha="center", fontsize=11.5, color="#555555")
    for i, w in enumerate([0, 5, 9, 14, 21, 30]):
        a = fig.add_axes([0.03 + (i % 2) * 0.09, 0.56 - (i // 2) * 0.16, 0.08, 0.13])
        a.plot(X[w, :, 0], y[w], "o", ms=3, color="#555555")
        a.set(xticks=[], yticks=[])
        a.set_title(f"window {i + 1}", fontsize=9)
    fig.text(0.115, 0.11, "e.g. one day of traffic,\none turbine for 6 hours", ha="center", fontsize=10.5,
             color="#555555")

    # --- START
    farrow(fig, (0.218, 0.47), (0.262, 0.47))
    panel(fig, (0.265, 0.33, 0.13, 0.28), fc=BLUE_FILL, ec=BLUE)
    fig.text(0.33, 0.56, "2 · FIRST GUESS", ha="center", fontsize=13.5, weight="bold", color=BLUE)
    fig.text(0.33, 0.4, "describe each window\nby the coefficients of\nits best-fitting curve,\n"
             "group similar ones", ha="center", fontsize=10.5)

    # --- THE LOOP
    farrow(fig, (0.397, 0.47), (0.43, 0.47))
    panel(fig, (0.43, 0.08, 0.36, 0.76), fc="#fffaf3", ec=ORANGE, lw=2)
    fig.text(0.61, 0.8, "3 · REPEAT TWO QUESTIONS", ha="center", fontsize=15, weight="bold", color=ORANGE)
    fig.text(0.61, 0.765, "until the answers stop changing", ha="center", fontsize=11.5, color="#555555")
    # question A: what is each law?
    fig.text(0.525, 0.715, "A.  What is each law?", ha="center", fontsize=13, weight="bold")
    a = fig.add_axes([0.45, 0.43, 0.15, 0.26])
    for k in range(3):
        a.plot(X[lab == k, :, 0].ravel(), y[lab == k].ravel(), "o", ms=1.5, color=cols[k], alpha=0.25)
        a.plot(xg, res.models[k].predict(xg[:, None]), color=cols[k], lw=3)
    a.set(xticks=[], yticks=[])
    fig.text(0.525, 0.39, "labels fixed: pool each group's\nwindows, fit ONE formula to them", ha="center",
             fontsize=10.5)
    # question B: which law made each window?
    fig.text(0.695, 0.715, "B.  Which law made\n      each window?", ha="center", fontsize=13, weight="bold")
    a = fig.add_axes([0.62, 0.43, 0.15, 0.24])
    for i, w in enumerate([0, 5, 9, 14]):
        a.plot(X[w, :, 0], y[w] + 7 * i, "o", ms=2.5, color=cols[lab[w]])
        a.plot(xg, res.models[lab[w]].predict(xg[:, None]) + 7 * i, color=cols[lab[w]], lw=1.2, alpha=0.6)
    a.set(xticks=[], yticks=[])
    fig.text(0.695, 0.39, "formulas fixed: each window goes\nto the formula that fits it best", ha="center",
             fontsize=10.5)
    farrow(fig, (0.6, 0.62), (0.625, 0.62), color=ORANGE, rad=0.0)
    farrow(fig, (0.695, 0.34), (0.525, 0.34), color=ORANGE, rad=-0.35)
    fig.text(0.61, 0.2, "each step makes the fit better:\nthe same idea as K-means,\nwith a FORMULA "
             "instead of an average", ha="center", fontsize=11.5, color=ORANGE)

    # --- RESULT
    farrow(fig, (0.792, 0.47), (0.815, 0.47))
    panel(fig, (0.815, 0.08, 0.17, 0.76), fc=GREEN_FILL, ec="#4a7a4a")
    fig.text(0.9, 0.8, "4 · RESULT", ha="center", fontsize=15, weight="bold", color="#2f5f2f")
    fig.text(0.9, 0.715, "a label for every window", ha="center", fontsize=12)
    a = fig.add_axes([0.835, 0.53, 0.13, 0.15])
    a.imshow(lab[None, :80], aspect="auto", cmap=matplotlib.colors.ListedColormap(cols))
    a.set(xticks=[], yticks=[])
    a.set_xlabel("windows 1-80", fontsize=9)
    fig.text(0.9, 0.44, "a formula for every law", ha="center", fontsize=12)
    for k in range(3):
        fig.text(0.9, 0.38 - k * 0.06, nice[k], ha="center", fontsize=11, color=cols[k], weight="bold")
    fig.text(0.9, 0.11, "true laws:\n0.8x² + 1.5x + 0.5,  2.5 sin 2x − 0.5,\n0.35x³ − 1.2\n"
             "(the parabola comes back as a\nlook-alike with cos x: the same\ncurve on this range)",
             ha="center", fontsize=9.5, color="#555555")
    fig.text(0.02, 0.03, "The formulas are built from a list of simple terms (see 00_vocabulary.png). "
             "'Fits best' = smallest robust misfit, with a small penalty for long formulas.",
             fontsize=11, color="#555555")
    fig.savefig(HERE / "00_the_idea.png", dpi=130, facecolor="white")
    plt.close(fig)
    print("wrote 00_the_idea.png")


# ==========================================================================
# 00_vocabulary
# ==========================================================================
def vocabulary():
    fig = plt.figure(figsize=(22, 13.5))
    fig.text(0.02, 0.965, "What a law can be built from, and how much you already know", fontsize=24,
             weight="bold", color=BLUE)
    fig.text(0.02, 0.935, "Every law LR-DSR returns is a short sum of TERMS. Which terms are allowed "
             "decides which laws it can write down exactly.", fontsize=13.5, color="#444444")

    # --- A: fixed terms, straight from the code
    panel(fig, (0.015, 0.62, 0.97, 0.29), fc=BLUE_FILL, ec=BLUE, lw=1.8)
    fig.text(0.025, 0.885, "A · FIXED TERMS  (the default library)", fontsize=15, weight="bold", color=BLUE)
    fig.text(0.025, 0.86, "the search picks up to 5 of them and fits ONE number (a coefficient) in front of "
             "each.  Example:  0.5 + 1.5·x + 0.8·x²", fontsize=11.5)
    xg = np.linspace(-2.2, 2.2, 300)
    Phi, names = library_terms(xg[:, None], feature_names=["x"])
    shown = list(zip(names, Phi.T, strict=True))
    pretty = {"1": "1  (constant)", "x": "x", "(x)^2": "x²", "(x)^3": "x³",
              "log1p(abs(x))": "log(1 + |x|)", "sin(x)": "sin x", "cos(x)": "cos x", "sin(2*x)": "sin 2x"}
    for i, (nm, col) in enumerate(shown):
        a = fig.add_axes([0.03 + i * 0.105, 0.655, 0.085, 0.16])
        a.plot(xg, col, color=BLUE, lw=2.2)
        a.axhline(0, color="#bbbbbb", lw=0.7)
        a.set(xticks=[], yticks=[])
        a.set_title(pretty.get(nm, nm), fontsize=12.5, weight="bold")
    a = fig.add_axes([0.03 + len(shown) * 0.105, 0.655, 0.085, 0.16])
    a.axis("off")
    a.text(0.5, 0.55, "x₁·x₂\n(when there are\ntwo inputs)", ha="center", va="center", fontsize=11.5)

    # --- B: terms with a fitted inner parameter
    panel(fig, (0.015, 0.33, 0.97, 0.27), fc=ORANGE_FILL, ec=ORANGE, lw=1.8)
    fig.text(0.025, 0.575, "B · TERMS WITH A FITTED INNER NUMBER", fontsize=15, weight="bold", color=ORANGE)
    fig.text(0.025, 0.55, "the search also fits a number INSIDE the term, so one term covers a whole family of "
             "shapes.  Needed when the law is not a sum of fixed terms.", fontsize=11.5)
    fams = [("sin(a·x)", lambda x, p: np.sin(p * x), (1.0, 2.5, 4.6), "implemented (trial, backend 'fast_sin')"),
            ("cos(a·x)", lambda x, p: np.cos(p * x), (1.0, 2.5, 4.6), "implemented (trial, backend 'fast_sin')"),
            ("e^(b·x)", lambda x, p: np.exp(p * x), (-1.0, -0.3, 0.4), "same idea, not implemented"),
            ("x^c", lambda x, p: np.abs(x) ** p, (0.5, 1.5, 2.5), "same idea, not implemented"),
            ("1 / (1 + c·x²)", lambda x, p: 1 / (1 + p * x ** 2), (0.5, 2.0, 8.0), "same idea, not implemented")]
    for i, (nm, f, ps, status) in enumerate(fams):
        a = fig.add_axes([0.03 + i * 0.19, 0.37, 0.16, 0.15])
        done = status.startswith("implemented")
        for j, p in enumerate(ps):
            a.plot(xg, f(xg, p), color=ORANGE if done else GREY, lw=2, alpha=0.45 + 0.27 * j,
                   ls="-" if done else "--")
        a.set(xticks=[], yticks=[])
        a.set_title(nm, fontsize=13, weight="bold", color=INK if done else GREY)
        a.text(0.5, -0.16, status, transform=a.transAxes, ha="center", fontsize=10,
               color=ORANGE if done else GREY)

    # --- C: open-ended
    panel(fig, (0.015, 0.185, 0.97, 0.125), fc="#f4f4f4", ec=GREY, lw=1.5, ls="--")
    fig.text(0.025, 0.285, "C · OPEN-ENDED  (PySR, optional block 5, after the loop)", fontsize=15,
             weight="bold", color="#555555")
    fig.text(0.025, 0.215, "no list at all: formulas are BUILT from operators   +  −  ×  ÷  sin  cos  exp   "
             "with numbers fitted anywhere inside.\nFinds laws like  e^(−0.09x)·sin(3.0x)  or  sin(0.80x²)  — "
             "but takes minutes per law, so it runs once per group, never inside the loop.", fontsize=11.5)

    # --- D: the knowledge ladder
    fig.text(0.02, 0.145, "D · HOW MUCH YOU KNOW ABOUT THE LAWS  (more knowledge → fewer samples needed)",
             fontsize=15, weight="bold")
    rungs = [("open-ended search", "PySR (block 5)", "#f1f1f1", GREY),
             ("search the library", "A: DEFAULT   (+ B: trial)", BLUE_FILL, BLUE),
             ("known FORM", "constants fitted\n(mechanism_specs 'factory')", "white", BLUE),
             ("known law", "nothing fitted\n(mechanism_specs 'known')", "white", BLUE),
             ("ORACLE", "knows every law exactly:\nthe yardstick, not a method", "#333333", "#333333")]
    for i, (t, sub, fc, ec) in enumerate(rungs):
        x0 = 0.03 + i * 0.19
        panel(fig, (x0, 0.02, 0.165, 0.09), fc=fc, ec=ec, lw=1.6)
        fig.text(x0 + 0.0825, 0.08, t, ha="center", fontsize=12.5, weight="bold",
                 color="white" if fc == "#333333" else INK)
        fig.text(x0 + 0.0825, 0.035, sub, ha="center", fontsize=9.5,
                 color="white" if fc == "#333333" else "#444444")
        if i:
            farrow(fig, (x0 - 0.024, 0.065), (x0 - 0.002, 0.065), lw=1.6)
    fig.text(0.03, 0.118, "know nothing", fontsize=10, color="#555555")
    fig.text(0.955, 0.118, "know everything", fontsize=10, color="#555555", ha="right")
    fig.savefig(HERE / "00_vocabulary.png", dpi=130, facecolor="white")
    plt.close(fig)
    print("wrote 00_vocabulary.png")


if __name__ == "__main__":
    the_idea()
    vocabulary()
