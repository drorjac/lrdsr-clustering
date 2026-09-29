# The LR-DSR algorithm, block by block

Diagrams of the pipeline and the options at each block. Regenerate with
`python docs/algorithm/make_diagrams.py`. The full written guide is
[`../lrdsr_algorithm.pdf`](../lrdsr_algorithm.pdf).

**Colour code, the same in every diagram**

| look | meaning |
|---|---|
| filled blue | the default: what every committed result uses |
| white, blue border | an available option |
| orange | the fitted-frequency trial (opt-in, `backend="fast_sin"`) |
| grey, dashed | not used: open-ended SR *inside* the loop (too slow) |

## Start here

**[00_algorithm_at_a_glance.png](00_algorithm_at_a_glance.png)** is the whole algorithm in one figure. There is one column per block, and each column shows four things, top to bottom:
1. **the block**;
2. **its piece of the objective**;
3. **what it does to the data**: a live plot from the worked example;
4. **its default and options**, as chips (blue = default, white = option, orange = the fitted-frequency trial).

Regenerate it with `python docs/algorithm/make_glance.py`. The diagrams below give more detail on each block.

**Hands-on:** `notebooks/09_algorithm_tutorial.ipynb` builds every block by hand in a few lines of numpy, checks it against the package, and demonstrates the options.

## The diagrams

1. **[01_pipeline_overview.png](01_pipeline_overview.png)**: the whole algorithm.
   - Input windows go to **0 · start** (mechanism space + K-means).
   - Then the loop runs **1 · fit**, **2 · noise scale**, **3 · score**, **4 · assign**, until under 1% of windows move.
   - Output: labels and one formula per group. The objective it minimises is shown on top.
   - The optional **5 · refine** block runs after the loop (diagram 07).
2. **[02_block0_start.png](02_block0_start.png)**: how a window becomes a vector of law coefficients (six steps), and the options for the initialiser, the basis and nuisance terms.
3. **[03_block1_fit_law_engines.png](03_block1_fit_law_engines.png)**: the greedy forward BIC search, and every law engine that can fill this block:
   - the fixed library (default);
   - library + fitted frequency (trial);
   - declared knowledge;
   - superposition;
   - other bases;
   - open-ended SR inside the loop (not used; too slow).
4. **[04_blocks2_3_noise_and_scoring.png](04_blocks2_3_noise_and_scoring.png)**: the cost matrix `J[w,k]` and its options:
   - noise scale;
   - loss (Huber by default, five others, or learned);
   - cross-fitting;
   - the extra cost weights β, α, λ.
5. **[05_block4_assign_and_family.png](05_block4_assign_and_family.png)**: hard assignment with repair and the stopping rule (default), and the estimators that assign differently: soft EM, real time, CUSUM, classifier.
6. **[06_estimators_by_block.png](06_estimators_by_block.png)**: a map of which estimator does what at each block, and what it outputs.
7. **[07_block5_optional_refine.png](07_block5_optional_refine.png)**: the optional block 5 (`lrdsr.core.refine.refine_laws`).
   - PySR searches each final group's law once, far more deeply than block 1.
   - The found law replaces the loop's law only if it lowers that group's cost on the same objective, judged on held-out windows (both laws fit on half the group and are compared on the other half), followed by one more assign step.
   - It takes about a minute per group and needs `pip install "pysr<2"`.

## In one sentence per block

- **0 · start:** compare windows by their law coefficients, not their looks, and K-means them.
- **1 · fit:** with labels fixed, build each group's formula term by term while BIC improves.
- **2 · noise scale:** one robust spread of all residuals, so a bad fit really costs more.
- **3 · score:** every window under every formula, and a window never judges a formula it helped fit.
- **4 · assign:** every window moves to its cheapest formula; repeat from 1 until nothing moves.
- **5 · refine (optional):** once, after the loop, a deeper search per group, kept only where it lowers the cost.

Only block 1's vocabulary changes between the law engines. The objective and the loop stay the same. PySR is too slow to run inside the loop, so it runs once in block 5, where the keep-only-if-better rule keeps it on the same objective.
