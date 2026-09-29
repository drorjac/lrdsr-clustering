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

Read these three in order:

1. **[00_the_idea.png](00_the_idea.png)**: the algorithm in pictures. Windows go in; the method alternates two questions ("what is each law?" and "which law made each window?"); labels and formulas come out.
2. **[00_vocabulary.png](00_vocabulary.png)**: what a law can be built from.
   - fixed terms (the default library, each one drawn);
   - terms with a fitted inner number (`sin(a·x)` is one implemented example);
   - open-ended PySR;
   - the knowledge ladder: from "know nothing" to the **oracle**, which knows every law exactly and is the yardstick, not a method.
3. **00_algorithm_at_a_glance.png**, the detailed view, described below.

Regenerate the first two with `python docs/algorithm/make_overview.py`.


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

## Why the refine step uses PySR, and not PhySO

Block 5 needs an **open-ended** symbolic-regression engine: one that builds
formulas from operators, rather than choosing from a list. Two established
packages do this:

| | **PySR** (chosen) | **PhySO** (tried, not adopted) |
|---|---|---|
| how it searches | genetic programming (Julia) | reinforcement learning: a neural network writes formulas (PyTorch) |
| particular strength | fast, clean formulas | can enforce physical units; "class SR" for one form with per-dataset constants |

**The trial.** This was a quick lab check (tuning seed 3), not a formal experiment.
- Setup: each engine got 800 noisy samples of one law, from the true group (no clustering).
- Laws: four that no fixed library can write.
- Budgets: PySR used 40 iterations; PhySO used 10 epochs, about 15 s each on 8 CPU cores.

| law | PhySO: distance to truth, time | PySR: distance to truth, time, formula |
|---|---|---|
| sin(0.8x²) | 0.005, 169 s | 0.003, 42 s, `sin(0.8005x²)` |
| 1/(1+x²) | 0.002, 194 s | 0.000, 28 s, `1.0005/(x² + 1.00006)` |
| x·sin 4x | 0.424, 270 s (**not found**) | 0.002, 22 s, `x·sin(3.999x)` |
| e^(−0.4x)·sin 3x | 0.372, 211 s (**not found**) | 0.064, 31 s (a close look-alike) |
| **total** | **2 of 4 found, 844 s** | **4 of 4 found, 123 s** |

Distance to truth is RMS(found − true) / RMS(true); 0 is exact, and below 0.1 counts as found.

**The reasoning.**
- **Accuracy:** PySR found every law, three of them in their own symbols. PhySO found two, and wrote even those as convoluted expressions.
- **Time:** PySR was about 7× faster. Block 5 runs once per group, so time matters.
- **PhySO's real advantage, physical units, does not apply here.** These laws, like most of the project's problems, have no units.
- **Fairness:** a larger epoch budget would help PhySO, but it would widen the time gap it already loses.

So PySR is the engine, and the PhySO adapter was removed from the code.
PhySO is worth revisiting only for data with real physical units (for
example wind speed → power), through a new adapter.

