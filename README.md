# LR-DSR clustering

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
identity is exact rather than asymptotic (18 cells), prices the
Jensen step from the realised design to the population form, checks the
`K`-ary sandwich, and confirms the oracle is invariant to a component the
two laws **share** -- scaled to 50x the gap the error does not move in
6/6 cells, because a term present in both carries no
information about which one you are looking at.

### 2. The coordinates are the method

The ceiling is stated between *laws*, so a method needs coordinates in which
distance between windows is distance between mechanisms. That is
`lrdsr/core/mechanism_space.py`: profile out the per-window nuisance,
subtract the law the group shares, whiten by the pooled Gram, project.

Under the model the result is an isotropic Gaussian mixture whose centres
are exactly `sqrt(n rho)` apart -- the easiest clustering problem there is.
V7 confirms plain K-means there attains the oracle: 72 cells, median
gap to the oracle 0.0022, median separation error
2.1%.

On the pair battery, where the geometry is built to carry nothing:

| method | matched error |
|---|---|
| oracle (knows both laws) | 0.033 |
| LR-DSR | 0.042 |
| K-means in mechanism space alone | 0.034 |
| the same loop, initialised in data space | 0.063 |
| best clustering on window summaries | 0.144 |
| best clustering on the planted geometry | 0.411 |

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
| a perfect start | 0.044 | 0.056 |
| a start corrupted to chance | 0.277 | 0.058 |
| random partition | 0.473 | 0.089 |
| K-means on window summaries | 0.181 | 0.078 |
| data-space BGMM | 0.175 | 0.084 |
| mechanism space | 0.046 | 0.056 |

**The output does not depend on the input.** Error in spans
0.044 to 0.277; error out sits between
0.056 and 0.058. The loop is a contraction onto a
fixed point. It repairs above about 10% corruption
(+0.219 from chance) and *damages* below it
(-0.013 at a perfect start), because mechanism space lands
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
0.15% of that floor, including the pair whose truth
is not in the library at all. A one-term surrogate for an inexpressible law
predicts as well as the law.

The assignment's complexity penalty is **inert**: matched error is
0.000 at every value of `beta` from 0 through 1 -- three orders
of magnitude either side of the default -- and only bites at
5, where it makes things *worse* (0.041). It is
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
- **Tune on seeds {3, 7, 19}, report on {11, 23, 42}.** Never select on a
  reported seed.
- **No number is typed by hand.** This page is generated by
  `python -m analysis.report --write`.
- Negative results are results. The loop not beating its initialiser, the
  complexity penalty doing nothing, and the fixed point not being the oracle
  are all above, not in a footnote.
