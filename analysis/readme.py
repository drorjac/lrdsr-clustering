"""README.md, generated from the numbers ``analysis.report`` computes.

The prose is here; the numbers are not. Anything in braces is filled from
``results/``, so the page cannot drift from the experiments.
"""
from __future__ import annotations

from lrdsr import paths

TEMPLATE = """# LR-DSR clustering

**Each window of data came from one of `K` unknown closed-form laws. Recover
the law each window came from.**

Not "cluster the windows and describe the clusters" -- the law *is* the
cluster identity, and two windows are similar when the same equation
produced them, however different they look. This project is the method for
that, and the analysis of when it works.

Every law here was written down by us, so every answer can be checked
against the truth rather than argued. Nothing in this repository reads a
real measurement.

```bash
pip install -e ".[dev]"
python -m experiments all        # the results every figure and number reads
python -m analysis.plots         # 16 figures -> figures/
python -m analysis.report        # every number, from results/
pytest -q
```

## The three claims, in order

### 1. There is a ceiling, and it is a formula

An **oracle** that knows both laws exactly and only has to decide which one
produced a window has error

```
P_err = Q( sqrt(n rho) / 2 )
```

with `rho` the separation between the laws relative to the noise and `n` the
window length. No estimator beats it. `experiments/theory` verifies the
identity is exact rather than asymptotic ({v1_cells} cells), prices the
Jensen step from the realised design to the population form, checks the
`K`-ary sandwich, and confirms the oracle is invariant to a component the
two laws **share** -- scaled to {v4_multiple:.0f}x the gap the error does not move in
{v4_unchanged}/{v4_cells} cells, because a term present in both carries no
information about which one you are looking at.

### 2. The coordinates are the method

The ceiling is stated between *laws*, so a method needs coordinates in which
distance between windows is distance between mechanisms. That is
`lrdsr/core/mechanism_space.py`: profile out the per-window nuisance,
subtract the law the group shares, whiten by the pooled Gram, project.

Under the model the result is an isotropic Gaussian mixture whose centres
are exactly `sqrt(n rho)` apart -- the easiest clustering problem there is.
V7 confirms plain K-means there attains the oracle: {v7_cells} cells, median
gap to the oracle {v7_gap_median:.4f}, median separation error
{v7_sep_rel_median_pct:.1f}%.

On the pair battery, where the geometry is built to carry nothing:

| method | matched error |
|---|---|
| oracle (knows both laws) | {bench_oracle:.3f} |
| LR-DSR | {bench_lrdsr:.3f} |
| K-means in mechanism space alone | {bench_mechanism_kmeans:.3f} |
| the same loop, initialised in data space | {bench_lrdsr_dataspace:.3f} |
| best clustering on window summaries | {bench_window_features:.3f} |
| best clustering on the planted geometry | {bench_geometry:.3f} |

### 3. The loop is a stabiliser, not an improvement

The alternating loop -- fit a law per cluster, rescore every window under
every law, reassign -- does **not** beat mechanism space on this battery.
That is easy to report as a flat negative and it would be the wrong reading,
because it is measured at one starting partition and that partition is
already at the ceiling.

`experiments/estimator/loop_value.py` characterises the loop as what it is,
an operator on a partition: hand it starts of controlled quality and measure
what comes out.

| what goes in | error in | error out |
|---|---|---|
| a perfect start | {loop_in_lo:.3f} | {loop_out_lo:.3f} |
| a start corrupted to chance | {loop_in_hi:.3f} | {loop_out_hi:.3f} |
| random partition | {loop_random_in:.3f} | {loop_random_out:.3f} |
| K-means on window summaries | {loop_kmeans_in:.3f} | {loop_kmeans_out:.3f} |
| data-space BGMM | {loop_bgmm_in:.3f} | {loop_bgmm_out:.3f} |
| mechanism space | {loop_mechanism_in:.3f} | {loop_mechanism_out:.3f} |

**The output does not depend on the input.** Error in spans
{loop_in_lo:.3f} to {loop_in_hi:.3f}; error out sits between
{loop_out_lo:.3f} and {loop_out_hi:.3f}. The loop is a contraction onto a
fixed point. It repairs above about {loop_crossover:.0%} corruption
(+{loop_gain_chance:.3f} from chance) and *damages* below it
({loop_gain_perfect:+.3f} at a perfect start), because mechanism space lands
past the fixed point.

So the two parts are complementary rather than redundant: **use the
coordinates when a library is available, and the loop when one is not.**

Two limits kept in rather than rounded away: the fixed point is not the
oracle, and at the lowest separation the loop makes mechanism space actively
worse.

## What the symbolic search actually does

The default backend is deterministic -- a fixed library, greedy forward
selection, BIC as the stopping rule -- because a stochastic search that
returns a different expression each run cannot support a claim that *this
law* was recovered. `experiments/estimator/search.py` shows the search step
by step and prices it against the floor the **true law itself** leaves on
held-out windows: the worst excess anywhere in the battery is
{search_worst_excess_pct:.2f}% of that floor, including the pair whose truth
is not in the library at all. A one-term surrogate for an inexpressible law
predicts as well as the law.

The assignment's complexity penalty is **inert**: matched error is
{beta_inert_lo:.3f} at every value of `beta` from 0 through 1 -- three orders
of magnitude either side of the default -- and only bites at
{beta_max:.0f}, where it makes things *worse* ({beta_err_at_max:.3f}). It is
reported as a constant sitting on a plateau, not as a tuned one.

## Layout

```text
lrdsr/core/        the method: model.py, mechanism_space.py, backends.py
lrdsr/theory/      verification.py -- the ceiling, V1-V4 and V7
experiments/       theory / estimator / functions, one block per question
results/<block>/   committed CSVs; every figure and number reads these
analysis/          plots.py (16 figures), report.py (every number)
tests/             the properties the method has to keep
```

## Protocol

- **True labels reach the final metric only, never a fit.** The one place a
  partition is constructed from something label-like is `loop_value`, whose
  starts come from the *oracle* -- a function of the data and the known laws
  -- and which says so where it builds them.
- **Tune on seeds {{3, 7, 19}}, report on {{11, 23, 42}}.** Never select on a
  reported seed.
- **No number is typed by hand.** This page is generated by
  `python -m analysis.report --write`.
- Negative results are results. The loop not beating its initialiser, the
  complexity penalty doing nothing, and the fixed point not being the oracle
  are all above, not in a footnote.
"""


def write(n: dict) -> None:
    (paths.ROOT / "README.md").write_text(TEMPLATE.format(**n), encoding="utf-8")
