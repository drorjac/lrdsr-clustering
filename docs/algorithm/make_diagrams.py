"""Block diagrams of the LR-DSR algorithm and the options at each block.

    python docs/algorithm/make_diagrams.py     # writes the PNGs next to this file

One visual language throughout:
    filled blue            the default (what every committed result uses)
    white, blue border     an available option
    orange                 the fitted-frequency trial (opt-in)
    grey, dashed           outside the algorithm, or parked (not used)
"""
# ruff: noqa: RUF001
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

OUT = Path(__file__).resolve().parent
BLUE, BLUE_FILL, ORANGE, ORANGE_FILL = "#1f5fa8", "#dce8f5", "#d9822b", "#fbe5d0"
GREY, GREY_FILL, GREEN_FILL, INK = "#8a8a8a", "#f1f1f1", "#dcefdc", "#222222"
STYLE = {
    "default": {"fc": BLUE_FILL, "ec": BLUE, "ls": "-", "lw": 1.6},
    "option": {"fc": "white", "ec": BLUE, "ls": "-", "lw": 1.1},
    "trial": {"fc": ORANGE_FILL, "ec": ORANGE, "ls": "-", "lw": 1.6},
    "parked": {"fc": GREY_FILL, "ec": GREY, "ls": "--", "lw": 1.1},
    "io": {"fc": GREEN_FILL, "ec": "#4a7a4a", "ls": "-", "lw": 1.3},
    "plain": {"fc": "white", "ec": "#bbbbbb", "ls": "-", "lw": 0.8},
}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def canvas(w, h, title, subtitle=None):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set(xlim=(0, w), ylim=(0, h))
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(0.3, h - 0.45, title, fontsize=17, weight="bold", color=BLUE, va="top")
    if subtitle:
        ax.text(0.3, h - 1.0, subtitle, fontsize=10.5, color="#555555", va="top")
    return fig, ax


def box(ax, x, y, w, h, text, kind="option", size=9.5, weight="normal", align="center"):
    s = STYLE[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=s["fc"], ec=s["ec"], ls=s["ls"], lw=s["lw"]))
    tx = x + w / 2 if align == "center" else x + 0.15
    ax.text(tx, y + h / 2, text, ha=align, va="center", fontsize=size, weight=weight,
            color=INK, linespacing=1.3)


def arrow(ax, p, q, color=INK, lw=1.4, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13, color=color, lw=lw,
                                 ls=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


def legend(ax, x, y, kinds=("default", "option", "trial", "parked")):
    names = {"default": "default", "option": "available option", "trial": "trial (fitted frequency)",
             "parked": "outside the algorithm / parked", "io": "input / output"}
    handles = [Patch(fc=STYLE[k]["fc"], ec=STYLE[k]["ec"], ls=STYLE[k]["ls"], lw=STYLE[k]["lw"],
                     label=names[k]) for k in kinds]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(x, y), ncol=len(kinds),
              frameon=False, fontsize=9, bbox_transform=ax.transData)


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


# ==========================================================================
# 01 overview
# ==========================================================================
def overview():
    fig, ax = canvas(20, 10.4, "LR-DSR: the pipeline",
                     "One objective, minimised by block coordinate descent: a start, then fit / score / assign until "
                     "fewer than 1% of windows move.")
    y, h = 4.3, 1.9
    box(ax, 0.3, y, 2.4, h, "INPUT\n\nwindows (X_w, y_w)\nw = 1..W\n+ number of laws K", "io", 9.5)
    box(ax, 3.3, y, 2.9, h, "0 · START\n\nmechanism space\n+ K-means\n→ starting labels", "default", 9.8)
    ax.add_patch(FancyBboxPatch((6.8, 3.55), 10.2, 3.5, boxstyle="round,pad=0.02,rounding_size=0.2",
                                fc="#fafafa", ec="#cccccc", lw=1, ls="-"))
    ax.text(11.9, 6.8, "THE LOOP  (≤ 10 rounds)", ha="center", fontsize=10.5, weight="bold", color="#555555")
    xs = [7.1, 9.6, 12.1, 14.6]
    labels = ["1 · FIT\n\none law per group\n(greedy BIC over\nthe term library)",
              "2 · NOISE SCALE\n\none shared robust\nspread of residuals",
              "3 · SCORE\n\nevery window under\nevery law (Huber,\nout-of-fold)",
              "4 · ASSIGN\n\nwindow → cheapest\nlaw; repair tiny\ngroups; stop < 1%"]
    for x, t in zip(xs, labels, strict=True):
        box(ax, x, y, 2.2, h, t, "default", 9.3)
    box(ax, 17.6, y, 2.1, h, "OUTPUT\n\nlabel z_w per window\n+ formula f_k\nper group", "io", 9.5)
    for a, b in [(2.7, 3.3), (6.2, 7.1), (9.3, 9.6), (11.8, 12.1), (14.3, 14.6), (16.8, 17.6)]:
        arrow(ax, (a, y + h / 2), (b, y + h / 2))
    arrow(ax, (15.7, y), (8.2, y), color=ORANGE, rad=-0.28, lw=1.6)
    ax.text(11.95, 2.9, "repeat: labels fixed → fit laws;  laws fixed → move windows", ha="center",
            color=ORANGE, fontsize=10)
    # objective
    ax.text(10, 8.55, "objective:   min over labels z and laws f   Σ_w  [ mean_i Huber( (y_wi − f_{z_w}(x_wi)) / s )"
            "  +  0.002 · size(f_{z_w}) ]", ha="center", fontsize=11, color=INK,
            bbox={"fc": "#f6f6f4", "ec": "#cccccc", "boxstyle": "round,pad=0.4"})
    # outside
    box(ax, 12.2, 0.35, 7.5, 1.75, "5 · REFINE  (optional, after the loop — lrdsr.core.refine)\n"
        "open-ended SR (PySR) once per final group;  its law REPLACES the loop's only if\n"
        "it lowers the group's cost on the SAME objective;  then one more assign step", "option", 9)
    arrow(ax, (18.6, y), (17.4, 2.1), color=BLUE, ls="--")
    # where options plug in
    ax.text(0.3, 2.65, "Where the options plug in:", fontsize=10, weight="bold")
    notes = ["block 0: initialiser, basis, nuisance  (diagram 02)",
             "block 1: law engine — library, + fitted frequency (trial), declared knowledge  (03)",
             "blocks 2-3: noise scale, loss, cross-fitting, extra cost terms  (04)",
             "block 4: hard / soft / online / sample-level / with labels  (05);  estimators map  (06)",
             "block 5 (optional): a deeper law search once per group, kept only if it lowers the cost  (07)"]
    for i, t in enumerate(notes):
        ax.text(0.4, 2.2 - i * 0.42, "• " + t, fontsize=9.5)
    legend(ax, 0.3, 7.35, ("io", "default", "option", "trial"))
    save(fig, "01_pipeline_overview.png")


# ==========================================================================
# 02 block 0
# ==========================================================================
def block0():
    fig, ax = canvas(18, 10.5, "Block 0 · the start: mechanism space",
                     "Compare windows by their LAW COEFFICIENTS, not by how they look; then plain K-means gives the "
                     "starting labels.")
    steps = [("1  expand", "B_w = library terms at the window's inputs"),
             ("2  profile nuisance", "remove per-window nuisance columns (e.g. a day's level)"),
             ("3  remove shared law", "r_w = y_w − B_w β_pooled   (what every window has in common)"),
             ("4  common footing", "G = mean_w B_w'B_w = V Λ V';   C_w = B_w V Λ^(-1/2)   (whiten)"),
             ("5  project", "s_w = C_w' r_w     one vector per window"),
             ("6  cluster", "K-means(s_1..s_W, K)  →  starting labels")]
    for i, (a, b) in enumerate(steps):
        yy = 7.9 - i * 1.15
        box(ax, 0.4, yy, 2.6, 0.85, a, "default", 10, "bold")
        box(ax, 3.2, yy, 6.4, 0.85, b, "plain", 9.3, align="left")
        if i:
            arrow(ax, (1.7, yy + 1.15), (1.7, yy + 0.85))
    ax.text(0.4, 0.9, "Why: if the laws lie in the span of the basis, the s_w are ROUND Gaussian clouds,\n"
            "one per regime, centres √(nρ) apart — exactly the case where K-means is the right rule.",
            fontsize=9.5, color="#333333")
    # options
    ax.text(10.4, 8.95, "OPTIONS", fontsize=11, weight="bold", color="#555555")
    ax.text(10.4, 8.45, "init =  (how the starting labels are made)", fontsize=9.5, weight="bold")
    opts = [("\"mechanism\"", "default"), ("\"kmeans\" on summaries", "option"), ("\"gmm\"", "option"),
            ("\"bgmm\"", "option"), ("\"fcm\" (fuzzy c-means)", "option"), ("an array of labels", "option")]
    for i, (t, k) in enumerate(opts):
        box(ax, 10.4 + (i % 3) * 2.5, 7.55 - (i // 3) * 0.75, 2.35, 0.6, t, k, 8.8)
    ax.text(10.4, 5.95, "mechanism_basis =  (the columns B)", fontsize=9.5, weight="bold")
    opts = [("term library", "default"), ("Fourier", "option"), ("cosine", "option"),
            ("Legendre", "option"), ("Nyström RBF", "option"), ("random Fourier", "option")]
    for i, (t, k) in enumerate(opts):
        box(ax, 10.4 + (i % 3) * 2.5, 5.05 - (i // 3) * 0.75, 2.35, 0.6, t, k, 8.8)
    ax.text(10.4, 3.45, "mechanism_nuisance / mechanism_groups", fontsize=9.5, weight="bold")
    box(ax, 10.4, 2.55, 3.6, 0.7, "none (default)", "default", 8.8)
    box(ax, 14.2, 2.55, 3.6, 0.7, "nuisance columns;  groups that\nshare a nuisance (e.g. a sensor)", "option", 8.5)
    box(ax, 10.4, 1.4, 7.4, 0.8, "trial note: the fitted-frequency term is NOT used here — the start stays on\n"
        "the library; the loop repairs it (measured on the sine problems)", "trial", 8.6)
    legend(ax, 0.4, 0.05, ("default", "option", "trial"))
    save(fig, "02_block0_start.png")


# ==========================================================================
# 03 block 1
# ==========================================================================
def block1():
    fig, ax = canvas(20, 13, "Block 1 · fit one law per group",
                     "Labels fixed. Pool all samples of all windows in group k and find a short formula. "
                     "Only this block's VOCABULARY differs between the options.")
    # greedy BIC mini-diagram
    ax.text(0.4, 11.4, "The default search: greedy forward BIC", fontsize=11, weight="bold")
    seq = ["constant", "+ best term", "+ best term", "…", "stop: no term\nlowers BIC, or 5 in"]
    for i, t in enumerate(seq):
        box(ax, 0.4 + i * 1.95, 10.2, 1.75, 0.85, t, "default" if i < 4 else "plain", 9)
        if i:
            arrow(ax, (0.4 + i * 1.95 - 0.2, 10.62), (0.4 + i * 1.95, 10.62))
    ax.text(0.4, 9.7, "each round: try every remaining term, least-squares coefficients, keep the one with the lowest\n"
            "BIC = N·log(RSS/N) + p·log N   →  complexity is chosen automatically", fontsize=9.2, color="#333333")
    # engines
    ax.text(0.4, 8.65, "LAW ENGINES  (backend = …  /  mechanism_specs per group)", fontsize=11, weight="bold")
    rows = [
        ("default", "\"fast\"  — fixed term library",
         "terms: 1, x, x², x³, log(1+|x|), sin x, cos x, sin 2x, pairwise products x_a·x_b\n"
         "deterministic, milliseconds;  every committed result uses it;  limit: laws outside the list come back as look-alikes"),
        ("trial", "\"fast_sin\"  — library + fitted frequency",
         "adds sin(a·x), cos(a·x) with a LEARNED (grid 0.5-8, then local refinement); BIC charges +1 for a\n"
         "safeguards: ≥ 1 period over the data · reject if 80% collinear · backward pruning · must beat plain library\n"
         "objective and loop unchanged;  ~8× slower (~2 s/problem);  fixes fast oscillations, not other families"),
        ("option", "declared knowledge  (mechanism_specs)",
         "\"known\": a fixed law, nothing fitted   ·   \"partial\": known part + searched residual\n"
         "\"factory\": known FORM with fitted constants (ParametricPriorRegressor)   ·   \"unknown\": search (default)"),
        ("option", "superposition = True",
         "a group's law may be a weighted mix of two other groups' laws (tested only for rare composites)"),
        ("option", "other bases (soft EM / online / classifier)",
         "Fourier, cosine, Legendre, Nyström RBF, random Fourier: flexible, dense coefficients, no term selection"),
        ("parked", "\"pysr\", \"dso\"  INSIDE the loop",
         "builds formulas from operators, fits constants inside;  minutes per fit × every round, group, fold\n"
         "→ hours: not used here.  PySR runs instead ONCE per group in the optional block 5 (diagram 07)"),
    ]
    yy = 8.25
    for kind, name, desc in rows:
        nl = desc.count("\n") + 1
        hh = 0.42 + 0.34 * nl
        box(ax, 0.4, yy - hh, 5.2, hh, name, kind, 9.4, "bold")
        box(ax, 5.8, yy - hh, 13.8, hh, desc, "plain", 8.9, align="left")
        yy -= hh + 0.18
    legend(ax, 0.4, yy - 0.55)
    save(fig, "03_block1_fit_law_engines.png")


# ==========================================================================
# 04 blocks 2-3
# ==========================================================================
def block23():
    fig, ax = canvas(19, 10.8, "Blocks 2-3 · noise scale and scoring",
                     "Laws fixed. Fill the cost matrix J[w, k]: what window w would pay to belong to law k.")
    ax.text(9.5, 9.05, "J[w,k] = mean_i  ρ( (y_wi − f_k(x_wi)) / s )   +  β·size(f_k)   +  α·D_geom(w,k)   +  λ·D_phys(w,k)",
            ha="center", fontsize=11.5, bbox={"fc": "#f6f6f4", "ec": "#cccccc", "boxstyle": "round,pad=0.4"})
    ax.text(9.5, 8.35, "each term is divided by its own median before weighting, so α, β, λ are real trade-offs, "
            "not unit conversions", ha="center", fontsize=9, color="#555555")
    cols = [
        ("2 · noise scale  s", "residual_scale =", [("\"global\": one robust spread\n(MAD in sd units) of all\nresiduals",
                                                   "default"),
                                                  ("\"per_window\": each window's\nown spread (older; blind to\nthe SIZE of a misfit)",
                                                   "option"),
                                                  ("with loss=\"learned\": the fitted\nnoise model's own scale", "option")]),
        ("3a · loss  ρ", "loss =", [("\"huber\"  δ = 1.345 sd", "default"), ("\"squared\"", "option"),
                                   ("\"absolute\"", "option"), ("\"cauchy\"", "option"), ("\"tukey\"", "option"),
                                   ("\"student_t\"", "option"),
                                   ("\"learned\": refit Gaussian /\nLaplace / Student-t each round", "option")]),
        ("3b · honest scoring", "score_mode =", [("\"cross_fit\": own-group entry\nscored by a law fitted\nwithout "
                                                  "that window's fold (3)", "default"),
                                                 ("\"in_sample\": law fitted on\nthe window scores it\n(flatters it)",
                                                  "option")]),
        ("3c · extra cost terms", "weights", [("β = 0.002 · complexity", "default"),
                                             ("α = 0 · geometry off\n(euclidean / mahalanobis\nwhen α > 0)", "default"),
                                             ("λ = 0 · physics anchor off", "default")]),
    ]
    for c, (title, key, opts) in enumerate(cols):
        x = 0.4 + c * 4.7
        ax.text(x, 7.55, title, fontsize=11, weight="bold")
        ax.text(x, 7.1, key, fontsize=9.3, color="#555555")
        yy = 6.8
        for t, k in opts:
            hh = 0.3 + 0.3 * (t.count("\n") + 1)
            box(ax, x, yy - hh, 4.3, hh, t, k, 8.8)
            yy -= hh + 0.15
    legend(ax, 0.4, 0.05, ("default", "option"))
    save(fig, "04_blocks2_3_noise_and_scoring.png")


# ==========================================================================
# 05 block 4 and the assignment family
# ==========================================================================
def block4():
    fig, ax = canvas(19, 10.2, "Block 4 · assignment, and the estimator family built around it",
                     "Same idea (a law per regime, windows go to the law that explains them); different ways to assign.")
    ax.text(0.4, 8.75, "The default: hard assignment (GroupedDCSR)", fontsize=11, weight="bold")
    seq = [("z_w = argmin_k J[w,k]", "default"), ("repair: groups < 4 windows\ntake their cheapest windows", "default"),
           ("changed < 1%?  (tol)", "default"), ("yes → refit laws, return\nno → back to block 1", "plain")]
    for i, (t, k) in enumerate(seq):
        box(ax, 0.4 + i * 4.6, 7.4, 4.2, 1.0, t, k, 9.4)
        if i:
            arrow(ax, (0.4 + i * 4.6 - 0.4, 7.9), (0.4 + i * 4.6, 7.9))
    ax.text(0.4, 6.75, "ALTERNATIVES", fontsize=11, weight="bold", color="#555555")
    rows = [
        ("SoftLRDSR — soft EM", "each window gets a PROBABILITY per regime (E-step); laws = weighted least squares over a\n"
         "fixed basis (M-step); noise \"gaussian\" or \"student_t\"; init \"mechanism\" / \"random\"; optional annealing",
         "uncertainty per window"),
        ("OnlineLRDSR — real time", "windows arrive one by one; each law updated in closed form (recursive least squares);\n"
         "forgetting < 1 tracks drift; a window no law explains is buffered → a consistent buffer is BORN as a new regime",
         "streams"),
        ("CusumSegmenter — switches", "sample by sample, CUSUM on the log-likelihood ratio between known/learned laws;\n"
         "alarm at threshold h; delay ≈ 2h/ρ samples (theory V9)", "one stream that switches"),
        ("LawClassifier — with labels", "labels known: fit L laws per class (the loop runs inside each class);\n"
         "a new window goes to the class whose laws explain it best; optional warp (phase) nuisance", "classification"),
    ]
    yy = 6.35
    for name, desc, use in rows:
        box(ax, 0.4, yy - 1.0, 4.4, 1.0, name, "option", 9.4, "bold")
        box(ax, 5.0, yy - 1.0, 10.9, 1.0, desc, "plain", 8.8, align="left")
        box(ax, 16.1, yy - 1.0, 2.6, 1.0, "use for:\n" + use, "plain", 8.8)
        yy -= 1.2
    legend(ax, 0.4, 0.05, ("default", "option"))
    save(fig, "05_block4_assign_and_family.png")


# ==========================================================================
# 06 estimators x blocks
# ==========================================================================
def estimators_map():
    fig, ax = canvas(20, 8.2, "Which estimator uses which block",
                     "Rows: the estimators. Columns: the blocks of the pipeline. Blue = what the estimator does there.")
    cols = ["labels\nneeded?", "0 · start", "1 · law", "2-3 · score", "4 · assign", "output"]
    rows = [
        ("GroupedDCSR\n(the default)", ["no", "mechanism\nspace", "term search\n(greedy BIC)", "Huber,\nout-of-fold",
                                        "hard argmin,\nloop", "labels + formulas"]),
        ("SoftLRDSR", ["no", "mechanism\nspace", "fixed basis,\nweighted LS", "likelihood\n(Gauss / t)",
                       "posteriors,\nEM", "probabilities + laws"]),
        ("OnlineLRDSR", ["warm start\nonly", "a short batch\nfit", "recursive LS,\nforgetting", "likelihood +\nnovelty test",
                         "one window\nat a time", "labels + new regimes"]),
        ("CusumSegmenter", ["laws known\nor learned", "—", "given", "log-likelihood\nratio", "per sample,\nthreshold h",
                            "switch times"]),
        ("LawClassifier", ["yes", "—", "L laws per class\n(loop inside)", "residual under\neach class", "argmin over\nclasses",
                           "class per window"]),
    ]
    x0, cw, rh = 3.3, 2.75, 1.05
    for j, c in enumerate(cols):
        box(ax, x0 + j * cw, 6.0, cw - 0.12, 0.8, c, "plain", 9.5, "bold")
    for i, (name, cells) in enumerate(rows):
        yy = 5.0 - i * (rh + 0.1)
        box(ax, 0.3, yy, 2.85, rh, name, "default" if i == 0 else "option", 9.5, "bold")
        for j, t in enumerate(cells):
            kind = "plain" if t in ("—", "no", "given") or j in (0, 5) else "default"
            box(ax, x0 + j * cw, yy, cw - 0.12, rh, t, kind, 8.8)
    ax.text(0.3, -0.35, "Law engines (diagram 03) plug into the '1 · law' column of GroupedDCSR; the fitted-frequency "
            "trial is one of them. Other bases plug into SoftLRDSR, OnlineLRDSR and LawClassifier.",
            fontsize=9.2, color="#555555")
    save(fig, "06_estimators_by_block.png")


# ==========================================================================
# 07 optional block 5
# ==========================================================================
def block5():
    fig, ax = canvas(19, 9.6, "Block 5 (optional) · refine: a deeper law search, once per group",
                     "Same job as block 1 (find a group's formula), searched far more deeply — too slow for the "
                     "loop, affordable once at the end.")
    seq = [("loop's result\nlabels + laws", "io"),
           ("for each group:\nPySR on the group's windows\n(held-out 1-SE formula choice)", "option"),
           ("split the group's windows:\nPySR and a copy of the loop's\nlaw fit on one half; compare\nSAME cost on the other half", "option"),
           ("keep the found law ONLY\nif its cost is lower;\notherwise keep the loop's", "option"),
           ("one more assign step:\nevery window → its\ncheapest kept law", "option")]
    for i, (t, k) in enumerate(seq):
        box(ax, 0.4 + i * 3.7, 5.6, 3.3, 1.7, t, k, 9.3)
        if i:
            arrow(ax, (0.4 + i * 3.7 - 0.4, 6.45), (0.4 + i * 3.7, 6.45))
    ax.text(0.4, 4.7, "Why it is part of the method and not a foreign layer", fontsize=11, weight="bold")
    for i, t in enumerate([
            "• a found law is kept only if it lowers the SAME cost as the loop's — it is one more 'fit laws, labels "
            "fixed' step of the same descent, with a larger vocabulary;",
            "• judged on held-out windows, like the loop's cross-fitting: a law never scores windows it was fitted on;  "
            "the final assign step is exactly block 4, run once more.",
            "Cost and behaviour (measured)",
            "• about 1 minute per group, against about 1 second for the whole loop;  needs  pip install \"pysr<2\"  "
            "(installs Julia on first use);",
            "• PySR often returns look-alikes of the true law (e.g. a rational stand-in for a decay); the guard then "
            "rejects them and nothing changes;",
            "• it can help labels only where the loop's laws were clearly wrong; it cannot fix a bad grouping on its "
            "own beyond one assign step."]):
        bold = not t.startswith("•")
        ax.text(0.4 if bold else 0.5, 4.2 - i * 0.55 - (0.25 if i >= 2 else 0), t,
                fontsize=11 if bold else 9.6, weight="bold" if bold else "normal")
    ax.text(0.4, 0.35, "code:  from lrdsr.core.refine import refine_laws;   out = refine_laws(result, X_seq, y_seq)",
            fontsize=9.6, family="DejaVu Sans Mono", color="#333333")
    legend(ax, 0.4, 7.7, ("io", "option"))
    save(fig, "07_block5_optional_refine.png")


if __name__ == "__main__":
    overview()
    block0()
    block1()
    block23()
    block4()
    estimators_map()
    block5()
