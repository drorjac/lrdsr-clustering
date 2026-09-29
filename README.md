# LR-DSR clustering

[![tests](https://github.com/drorjac/lrdsr-clustering/actions/workflows/ci.yml/badge.svg)](https://github.com/drorjac/lrdsr-clustering/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Each window of data came from one of `K` unknown closed-form laws. Recover
the law each window came from.**

Not "cluster the windows and describe the clusters" -- the law *is* the
cluster identity, and two windows are similar when the same equation
produced them, however different they look. This project is the method for
that, and the analysis of when it works. **[RESULTS.md](RESULTS.md) is the
scorecard**: fifteen missions, a verdict and the figure for each.

**New here?** Read [the idea in one picture](docs/algorithm/00_the_idea.png)
and [what a law is built from](docs/algorithm/00_vocabulary.png), then open
`notebooks/00_summary.ipynb` (the project end to end) and
`notebooks/09_algorithm_tutorial.ipynb` (every block built by hand). A written
guide is in [docs/lrdsr_algorithm.pdf](docs/lrdsr_algorithm.pdf).

Almost every law here was written down by us, so almost every answer can be
checked against the truth rather than argued. The exceptions are the
real-data blocks: two public hourly series, whose reference labels are a
calendar proxy and are said to be one; 24 datasets of the UCR time-series
archive, whose labels are the archive's own; and a wind farm's SCADA,
scored against physical proxies because it has no operator labels.

## Use it on your data

```bash
pip install git+https://github.com/drorjac/lrdsr-clustering
```

```python
from lrdsr import GroupedDCSR

# X_seq: (windows, samples, inputs)   y_seq: (windows, samples)   Z: any window features
res = GroupedDCSR(n_clusters=3, alpha_geom=0.0).fit(X_seq, y_seq, Z, feature_names=["x"])
res.labels                                 # which law made each window
[m.expression() for m in res.models]       # each law, as a formula
```

`notebooks/09_algorithm_tutorial.ipynb` walks through every option.

## Reproduce everything

```bash
git clone https://github.com/drorjac/lrdsr-clustering && cd lrdsr-clustering
pip install -e ".[dev]"
python -m experiments all        # the results every figure and number reads
python -m analysis.plots         # every figure -> figures/
python -m analysis.report --write  # every number -> README.md, RESULTS.md
python notebooks/build.py        # build + execute the notebooks
pytest -q
```

| block | question | where |
|---|---|---|
| `theory` | is there a ceiling, and is it a formula? (V1-V4, V7) | [claim 1](#1-there-is-a-ceiling-and-it-is-a-formula) |
| `estimator` | does the method reach it; what is the loop for? | [claims 2-3](#2-the-coordinates-are-the-method) |
| `functions` | what is declared knowledge worth? | `results/functions/` |
| `problems` | which *shapes* of problem does it solve? | [the problem zoo](#the-problem-zoo) |
| `losses` | which loss, when, and can the loss be learned? (V8) | [losses](#which-loss-and-learning-it) |
| `online` | can it cluster in real time? (V9) | [real time](#clustering-in-real-time) |
| `realdata` | what does it find in real measurements, including days with gaps? | [real data](#real-data) |
| `kernel` | does a kernel basis repair a gap outside the library? | [kernels](#a-kernel-basis-instead-of-a-library) |
| `classify` | a class is a set of laws: theory (V10) and 24 real datasets | [classification](#classification-a-class-is-a-set-of-laws) |
| `wind` | power curves where every window has its own design (Kelmarsh SCADA) | [wind](#wind-power-curves-every-window-its-own-design) |

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
| LR-DSR | 0.038 |
| K-means in mechanism space alone | 0.034 |
| the same loop, initialised in data space | 0.052 |
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
| a perfect start | 0.044 | 0.051 |
| a start corrupted to chance | 0.277 | 0.052 |
| random partition | 0.473 | 0.084 |
| K-means on window summaries | 0.179 | 0.051 |
| data-space BGMM | 0.172 | 0.070 |
| mechanism space | 0.046 | 0.049 |

**The output does not depend on the input.** Error in spans
0.044 to 0.277; error out sits between
0.051 and 0.052. The loop is a contraction onto a
fixed point. It repairs above about 5% corruption
(+0.225 from chance) and *damages* below it
(-0.007 at a perfect start), because mechanism space lands
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
0.000 at every value of `beta` from 0 through 0.2 --
two orders of magnitude above the default of 0.002 -- and only bites
from 1 (0.004), and harder at 5
(0.041), where it only makes things *worse*. It is reported as a
constant sitting on a plateau, not as a tuned one.

## The problem zoo

`experiments/problems` is 12 problems, each built to break one
assumption: the law is not in the symbolic library, the *gap* between laws
is outside the library's span, `K` = 3 or 4, two inputs, a rescaled copy,
a response with matched mean and variance. Every regime shares one input
distribution, so only the equation can separate them. Mean excess error
over the oracle across all 36 cells:

| method | gap to oracle |
|---|---|
| LR-DSR (hard loop) | +0.025 |
| soft EM (`SoftLRDSR`) | +0.018 |
| K-means in mechanism space | +0.017 |
| K-means on the raw sorted profile | +0.130 |
| best clustering on window summaries | +0.221 |

The hard loop is within 0.02 of the oracle in 29/36
cells, soft EM in 28/36.

**"Out of the library" is usually harmless.** 5 problems have laws
no library term can write (`sin 1.3x`, `tanh x`, a moving hinge, `x^1.8`, a
damped sine) but a *gap* the library spans almost entirely; their mean gap
to the oracle is +0.006, and the laws come back as
surrogates that predict as well as the truth.

**The real failure is a gap outside the span.** `high_frequency` (`sin 4x` vs
`sin 4.6x`) leaves 37% of the gap outside the library.
There the hard loop is +0.237 from the oracle, soft EM
+0.096 and mechanism K-means +0.079: the
hard loop damages the start it was given, because the laws it fits cannot
represent what separates the regimes. Leave that problem out and the hard loop
and soft EM sit at +0.006 and +0.011. The
diagnostic, `gap_outside_library`, is computable before fitting anything.

## Which loss, and learning it

The equation term scores a window by a per-sample loss on its scaled
residual. `lrdsr.core.losses` has squared, absolute, Huber, Cauchy, Tukey and
Student-t, and `GroupedDCSR(loss="learned")` refits a noise model (Gaussian,
Laplace, or Student-t with fitted `nu`) to the residuals every iteration.

**V8: a loss costs exactly its efficiency.** For a decision made with loss
`rho` and influence `psi = rho'`, the window error is

```
P_err = Q( sqrt(n rho eta) / 2 ),   eta = Var(e) (E psi')^2 / (s^2 E psi^2)
```

the asymptotic relative efficiency of the M-estimator, in units where
squared loss has `eta = 1` under **every** noise and the likelihood-ratio
loss attains the maximum `Var(e) I(p)`. By quadrature: under Gaussian noise
Huber has `eta` = 0.950 and Cauchy 0.950; under
Student-t(3) Huber 1.89 and Cauchy 1.93 against a
ceiling of 2; under 10% contamination Huber 7.5
and Cauchy 8.3 against 9.0 -- the
squared loss needs nine times the samples. Simulated directly, the formula is
within the Monte-Carlo band in 56/56 cells for every
smooth loss when the per-sample gap is at most a quarter of the noise scale,
and 52/56 up to half. Above that it is optimistic (only
39% of cells in band), and the non-smooth absolute loss and squared
loss under heavy tails at small `n` break it earlier; `v8_verdict.csv` bins
it all by gap.

**The estimator, under four noise laws** (mean matched error; 3 pairs x 2
separations x 3 seeds):

| | Gaussian | Laplace | Student-t(3) | 10% contaminated |
|---|---|---|---|---|
| oracle, true-noise likelihood ratio | 0.064 | 0.024 | 0.018 | 0.000 |
| oracle, squared loss | 0.064 | 0.066 | 0.057 | 0.063 |
| hard loop, **learned loss** | 0.064 | 0.026 | 0.019 | 0.000 |
| hard loop, Huber (default) | 0.075 | 0.043 | 0.022 | 0.000 |
| hard loop, squared | 0.065 | 0.071 | 0.059 | 0.068 |
| soft EM, Student-t noise | 0.068 | 0.036 | 0.018 | 0.000 |
| soft EM, Gaussian noise | 0.069 | 0.091 | 0.362 | 0.457 |

**Learning the loss recovers the Bayes rule**: the learned loss is never more
than 0.002 above the true-noise oracle, under any noise.
Under Student-t(3) the fitted `nu` converges to 2.91 (hard loop) and
2.95 (soft EM).

**A bug this analysis found in the default, and its fix.** The Huber
default used to divide residuals by the raw MAD, which is 0.67 sigma, so
`delta = 1.5` was really about 1 sigma, where the V8 table puts Huber at only
~90% efficiency. Residuals are now divided by the normal-consistent MAD
(`lrdsr.core.losses.noise_scale`), and the constant was chosen on the
*tuning* seeds (`experiments/losses/huber_delta.py`): at 1 sigma -- the old
default -- the Gaussian error is 0.059, and at the textbook
`delta = 1.345` it is 0.049, level with the squared loss
(0.049), while Student-t(3) goes from 0.021 to
0.019. Laplace noise prefers the old, smaller constant
(0.034 against 0.044); that is the trade.
`GroupedDCSR(scale_convention="mad", robust_delta=1.5)` reproduces the old
behaviour. On the reporting seeds the default Huber is still
0.075 against 0.065 for squared under
Gaussian noise: the fix narrows the gap but does not close it there, and the
learned loss is the setting that does. Every number on this page was
regenerated after the fix.

One negative, kept in: a soft EM that *assumes* Gaussian noise collapses
under heavy tails.

## Clustering in real time

`lrdsr.core.online` has two real-time estimators. `OnlineLRDSR` keeps each
regime's law as sufficient statistics (recursive least squares), so a window
is scored, assigned and absorbed in one step; windows no law explains are
buffered, and a self-consistent buffer becomes a **new regime**; a forgetting
factor lets laws drift. `CusumSegmenter` segments a single sample stream.

**V9: the delay of sequential detection is a formula.** The per-sample KL
divergence between two laws of separation `rho` is `rho/2`, so CUSUM with
threshold `h` detects a switch after about `2h/rho` samples and false-alarms
no more than every `e^h`. Over 20 (`rho`, `h`) cells the first-order delay
is off by a median 11%; with Siegmund's overshoot correction in
the log-likelihood ratio's own tilt, by 2.5% (worst
8%). The `e^h` false-alarm bound holds in
20/20 cells.

**Streams** (window error, 3 seeds):

| | `rho` = 0.1 | 0.25 | 1 |
|---|---|---|---|
| oracle | 0.186 | 0.080 | 0.002 |
| batch soft EM, in hindsight | 0.191 | 0.083 | 0.002 |
| online, 160-window warm start | 0.188 | 0.089 | 0.002 |
| online, 40-window warm start | 0.338 | 0.163 | 0.002 |
| laws frozen at the warm start | 0.352 | 0.201 | 0.003 |

Online with enough history matches batch-in-hindsight and the oracle; with
too little at low separation it locks onto wrong laws (a negative, and the
reason warm-start size is a parameter). A new law is **born** in every run once
it is detectable, 3-9 windows after it first
appears, with error after the birth at most 0.021, and there
were 0 false births on stationary control streams. Under drift, error
is 0.285 with frozen laws, 0.126 without forgetting and
0.000 at `lambda = 0.95`. One window costs
342-646 us (at least 1,549 windows/s). On a Markov
switching sample stream, CUSUM with laws **learned** by warm start is as
accurate as with the true laws (0.947 against 0.947 at
`rho` = 1), with a mean delay of 9.5 samples against
`2h/rho` = 10.

`python -m experiments.online.live` runs the live demo, and
`notebooks/04_realtime_clustering.ipynb` animates it.

## Real data

`experiments/realdata` runs the method on two public UCI datasets, one window
per day: **Capital Bikeshare** hourly rentals (Washington DC, 2011-12;
723 of 731 days kept) and **I-94 Metro
Interstate** hourly traffic (Minneapolis, 2012-18; 1214 of
1860 days, the rest have sensor gaps). The input is the hour,
as `x = 2 pi h / 24`; the response is `log(1 + count)` minus the day's mean, so
seasonal level is removed and the regime has to be the **shape** of the
daily law. Archives are downloaded once into `.cache/`, and their sha256 is
checked against `experiments/realdata/checksums.json`.

The reference is the **calendar** day type (working day vs weekend or
holiday). It is a proxy, used for scoring only, and the results say exactly
where it is wrong.

| method (K = 2) | bike ARI | traffic ARI |
|---|---|---|
| LR-DSR hard loop | 0.915 | 0.921 |
| soft EM, Student-t noise | 0.955 | 0.924 |
| soft EM, Fourier basis | 0.955 | 0.927 |
| K-means in mechanism space | 0.917 | 0.924 |
| **K-means on the raw 24-hour profile** | 0.955 | 0.927 |
| best clustering on window summaries | 0.611 | 0.574 |

**The honest reading: a plain K-means on the raw profile ties the best
method.** It has to. Every day has the same 24-hour design, so mechanism
space is only a linear change of coordinates of the profile, and the
method's advantage on the simulator -- windows with *different* inputs --
does not exist here. What the method adds on this data is the laws, the
posteriors and the real-time version, not accuracy. The fast library, with
one full harmonic, is also slightly worse than a Fourier basis for a
two-peak commuting day (0.915 against 0.955 on bike).

**The remaining error is the calendar, not the method.** The label-free
separation is large (`rho` = 1.6 and 4.0 on the library
basis, 12 and 18 on Fourier), so the implied ceiling
is essentially zero error. Of the 8 bike and 22 traffic
days on which the law-based partition and the calendar disagree, all but three
are days the calendar mislabels: 6 and
15 holidays that are ordinary working days for the bikes and
the traffic (Columbus Day, Veterans Day, Washington's Birthday, the Minnesota
State Fair), 2 and 4 working days that behave
like days off (the Friday after Thanksgiving, Christmas Eve). The three
exceptions are July 2016 weekends that look like working days on the traffic
sensor, likely a detector anomaly (`realdata_disagreements.csv` lists every
day by date).

**Choosing `K` without labels does not work here, and we say so.** BIC picks
the largest `K` tried (8) on both datasets, because the likelihood
treats 24 correlated hours as independent and real days vary in many ways.
The first extra cluster is interpretable (rainy working days on bike; a
Christmas / New Year / blizzard anomaly cluster on traffic), but BIC is not a
usable stopping rule on this data.

**In real time**, warm-started on the first 60 days without labels and
then fed one day at a time, `OnlineLRDSR` matches the calendar on
97.6% of bike days and 97.2% of traffic days
(novelty `alpha` = 1e-6): error 2.4% and
2.8% against 2.1% and 2.0% for the batch
hard loop fitted in hindsight on every day. The
learned loss picks a Student-t on both: `nu` = 15 on bike (close to
Gaussian) and 6 on traffic (heavier tails), with the partition
essentially unchanged.

### Days with sensor gaps: the first real windows with different designs

Everything above keeps a traffic day only if all 24 hours are present,
because every window then shares one design. That drops a third of the
days, and on those days the raw-profile baseline that tied the method no
longer exists: a profile needs its hours. A law does not -- it is scored at
whichever hours were observed, with the day's level profiled out as a
nuisance (`experiments/realdata/gaps.py`). There are 616 such days with at
least 6 observed hours.

Label-free: the 1214 complete days are clustered as before, the two clusters
become two laws, and each partial day goes to the law that explains its own
hours. Against the calendar:

| hours observed | days | law at the observed hours | impute hour means, then profile | impute linearly, then profile |
|---|---|---|---|---|
| 6-11 | 38 | 1.000 | 0.868 | 0.711 |
| 12-17 | 203 | 0.966 | 0.980 | 0.921 |
| all | 616 | 0.976 | 0.971 | 0.943 |

The law wins where the day is most incomplete, and imputation catches up as
hours return; with 18 or more hours every route agrees. It is not a clean
sweep: at 12-17 hours, filling each gap with the hour's mean over complete
days (0.980) edges out the law (0.966), and the 6-11 bin is only
38 days. So the same question is asked where the answer is known: every
complete day gets the gap mask of a randomly drawn real gap day, and each
route is scored by agreement with **its own** full-day decision. At 6-11
hours (111 masked days over three seeds) the law keeps its decision
99.1% of the time, hour-mean imputation 88.3% and linear imputation
78.4%; at 12-17 hours, 99.5%, 96.6% and 94.9%. With the
calendar as training labels instead of clusters (a classifier, not a
clustering), the totals are 0.976 for the law and 0.974 for hour-mean
imputation.

### Predicting a partial day's error before scoring it (V11)

V1 prices a window by its realised design, but it assumes independent noise
and the likelihood-ratio rule. A real day breaks both: its hours are
correlated, and the classifier used is least squares with the day's level
profiled out. `lrdsr/theory/partial.py` prices *that* classifier exactly for
any set of observed hours `H`:

```
err(H) = Q( (|g~_H|^2 / 2 +/- s^2 log(pi_1/pi_0)) / sqrt(g~_H^T Sigma_H g~_H) )
```

with `g~_H` the level-profiled gap between the two day-laws at the observed
hours and `Sigma` the residual covariance of complete training days -- the
only estimated input, and it carries the correlation of neighbouring hours.
Under correlated Gaussian noise it matches the actual classifier within the
Monte-Carlo band (`tests/test_partial.py`).

On real days (`experiments/realdata/partial.py`; laws and `Sigma` from one half
of the complete days, every day of the other half re-scored under masks):

- **Which hours are missing matters, and V11 knows which.** A 6- or 12-hour
  block removed at each of the 24 start hours: the predicted error ranks the
  start hours like the observed one, Spearman 0.86 to 0.95 over both
  datasets and block lengths, and names the worst start hour in 3 of 4
  cases. Losing the night and early morning costs I-94 up to
  20%; losing the middle of the day costs nothing.
- **A day is decided by 5 am.** Read hour by hour from midnight, the share of
  I-94 days decided correctly at 99% confidence is predicted at
  0.990 after five hours and observed at 0.989; bike days by six hours,
  0.966 against 0.948. At the knee of the curve it is optimistic (four
  hours on I-94: 0.47 predicted, 0.28 observed).
- **The ranking is right; the level is within a factor of a few.** Binned
  by predicted error, it is conservative in the middle (traffic blocks
  predicted at 0.057, observed 0.018) and optimistic in the far tail
  (0.0003 against 0.0010), where a few days that are not Gaussian about
  their law set the rate.

### Streaming days with gaps, and a law the library cannot write

`OnlineLRDSR` now takes a per-window nuisance (the level, profiled out before
a window is scored or absorbed), windows of any length, and a kernel basis
fitted on its warm start (`experiments/online/ragged.py`).

**Every I-94 day, in calendar order.** The first real-time pass had to drop
the days with sensor gaps. Streamed with them -- 1727 days, 573 of them
partial -- the partial days are sorted as well as the complete ones:
96.3% against 96.3%, and the complete-days-only stream with the same
settings reaches 96.0%. (This stream uses a Fourier basis and the level
nuisance; the 97.2% above used the library on centred days, so the two
are not compared.)

**A newcomer only a kernel can see.** A stream of `high_frequency` windows:
warm start on `sin 4x` alone, `sin 4.6x` appears at window 40. With a Nystrom
basis the newcomer is born in 3/3 streams at `rho` = 1 and 3/3 at 2,
11.5 windows after it first appears, and every later window is sorted
correctly (0.000 error). With the term library it is born in 0/3 at `rho` = 1
and 1/3 at 2, although a known-law test would flag 75% of newcomer
windows at `rho` = 1. The library's law for `sin 4x` is a poor surrogate, its
misfit inflates the noise the novelty test measures, and the newcomer hides
in it. No false birth on the control streams (0 in total).

## A kernel basis instead of a library

The problem zoo's one real failure was a gap outside the library: `sin 4x`
against `sin 4.6x` leaves 37% of the difference outside the span of the
fast library's terms. `lrdsr/core/kernel.py` supplies bases that span a
function space instead -- Nystrom features of an RBF kernel, random Fourier
features, cosine, Legendre and Fourier bases -- as drop-in `basis=` arguments
for mechanism space, soft EM and the classifier. With a Nystrom basis the
mechanism-space distance between two windows is the RKHS distance between
their fitted functions, and `window_kernel` turns it into a kernel between
windows.

The rank is the one knob, and it trades bias (gap outside the span) against
variance (every basis direction is a noise direction in mechanism space).
The rank is chosen **without labels**. Five rules were compared on the
tuning seeds -- per-window leave-one-out (minimum, one standard error) and
the spectral SNR of mechanism space, the excess spread over the noise per
root dimension (its maximum, the smallest rank within one bootstrap standard
error of it, the maximum of the curve smoothed over neighbouring ranks) --
and `snr_1se` was kept, at mean error 0.0716 against 0.0762 for the plain
SNR maximum, 0.0754 smoothed and 0.0946 for leave-one-out (the rank chosen
against the truth, an optimistic bound, 0.0636). Leave-one-out asks how
rich *one* window's law is, and a noisy window cannot justify resolving
`sin 4x` (on `high_frequency` at `rho` = 0.25 it errs at 0.427).

**The rule the tuning seeds chose does not hold up on the reporting seeds.**
There `snr_1se` errs at 0.0722, against 0.0701 for the plain SNR maximum and
0.0689 smoothed: the three SNR variants are within seed-to-seed noise
of each other, and the tuning seeds picked one by chance. Its preference for
the smallest adequate rank gives part of the gap back on `high_frequency`.
The numbers below are the rule the protocol chose; switching now, after
seeing the reporting seeds, would be selecting on them.

On the reporting seeds, gap to the oracle:

| method | `high_frequency` | all 12 problems |
|---|---|---|
| soft EM, library | +0.096 | +0.018 |
| mechanism K-means, library | +0.079 | +0.017 |
| mechanism K-means, kernel | +0.055 | +0.015 |
| **soft EM, kernel** | +0.051 | +0.015 |
| K-means on the raw profile | | +0.130 |

The kernel basis takes the gap outside the span from 37% to
0.01% at rank 17 and cuts the failure by 30% for mechanism K-means and
47% for soft EM, without costing the other problems: soft EM in the kernel
basis is better than in the library on 6 of 12 and worse by more than
0.002 on 2. What is left on `high_frequency` is variance, not bias: at
`rho` = 0.1 even the best rank chosen against the truth errs at
0.153, and the chosen rule, which picks a different rank per seed, at
0.240. The price of the kernel is the name: the law is a dense RKHS
function, not an expression.

## Classification: a class is a set of laws

`lrdsr/core/classify.py` turns the clustering model into a classifier.
`LawClassifier` fits `L` laws per class with the hard LR-DSR loop inside each
class, and sends a window to the class whose mixture of laws explains it
best -- mixture discriminant analysis with laws for components. `L = 1` is
the linear rule, `L = "all"` makes every training window its own law (nearest
law). A window is scored at its own inputs, so series of different lengths,
with gaps or irregular sampling are scored as they are. `MechanismFeatures`
is mechanism space as a train/test feature map, the *feature domain* for any
scikit-learn classifier.

**V10: the learning curve is a formula.** With the laws known the error is the
ceiling `Q(sqrt(n rho)/2)`. Estimated from `m` labelled windows per class in
a basis of rank `p`, it is the expectation of a closed form over three
scalars (`lrdsr/theory/classification.py`), and to first order

```
P_err ~ Q( D^2 / (2 sqrt(D^2 + 2p/m)) ),   D^2 = n rho
```

**The price is the basis dimension `p`, not the window length `n`.** At a
fixed design the exact form is within the Monte-Carlo band in
58/60 cells (the first-order form in only 11: it is optimistic
at small `m`). At `n rho = 9` and one labelled window per class, the error
is 0.185 with `p` = 3 and 0.333 with `p` = 31, against a ceiling of
0.067. At a random design two corrections are needed and
both were derived, not fitted: a law estimated from `m n` randomly placed
samples costs `(p + 1)/n` of a window per class (the inverse-Wishart
factor), and the oracle itself sits above the ceiling by the Jensen gap of
V1. With both, 28/30 cells with `p/(m n) <= 0.05` are in band; without
them 18/60. They fail where `p/(m n) > 0.2` (0/9), where the
inverse-Wishart tail takes over -- at `p` = 31 and `m` = 1 the classifier is
at chance. **That tail can be computed rather than approximated**: sampling
the estimated laws from their exact least-squares law,
`beta^ ~ N(beta, sigma^2 (B^T B)^-1)`, over training and test designs, and
integrating the test noise analytically, puts 60/60 random-design cells in
band, the 9 with `p/(m n) > 0.2` included (9/9) -- so V10 is now
closed at every design and every `p/(m n)` tested.

**On 24 real datasets.** `experiments/classify` runs the classifiers on the UCR
archive's own train/test splits: 9 **daily-cycle** datasets (pedestrian
counts, power demand, freeway loops, household appliances -- a day or a
week as a window, the kind of data this method is for) and 15 **shape**
benchmarks (gestures, ECG, spectra, simulated patterns), the contrast, where
a class is a pattern with phase jitter rather than a law. The split into
groups was fixed before any run. Every hyperparameter -- basis size, `L`,
the level nuisance, regularisation -- is chosen by CV on the training split;
the test split is touched once. The comparisons that were declared before
the run, mean test error:

| comparison | daily: wins or ties | daily: mean error | shape: wins or ties | shape: mean error |
|---|---|---|---|---|
| law classifier vs raw 1-NN (the archive's reference) | 5/9 | 0.171 vs 0.203 | 11/15 | 0.140 vs 0.158 |
| law classifier vs raw nearest centroid | 8/9 | 0.171 vs 0.267 | 13/15 | 0.140 vs 0.253 |
| mechanism features vs raw profile, same logistic regression | 5/9 | 0.173 vs 0.196 | 11/15 | 0.154 vs 0.171 |
| mechanism features vs raw profile, same RBF SVM | 5/9 | 0.153 vs 0.187 | 10/15 | 0.126 vs 0.136 |
| best law classifier vs best raw-profile classifier | 1/9 | 0.162 vs 0.137 | 7/15 | 0.130 vs 0.107 |

Three readings, the last one a negative. **The law representation helps a
fixed decision rule**: the same 1-NN, centroid, logistic regression or SVM
does better on mechanism coordinates than on the raw profile, on a
majority of datasets in both groups (a bare one on the daily group, a clear
one on the shape group) -- the smoothing a basis does is worth more than
the detail it drops. The largest single gains are on shape data whose
classes *are* smooth functions: SyntheticControl 0.003 against 0.120
for raw 1-NN (and 0.017 for the archive's DTW), TwoPatterns
0.058 against 0.093. **Where a class is a time-warped pattern the
basis loses** -- Trace 0.260 against 0.010 for DTW -- and elastic methods
remain the right tool there. **And a generative law classifier does not beat
the best tuned discriminative classifier on a raw profile** when the training
set is full: the best law classifier wins on 1/9 daily datasets
and 7/15 shape ones (both sides are best-of on the test set, so
optimistic alike). A readable classifier has a price too: one *named* law per
class from the symbolic library errs at 0.284 on the daily group.

**Two repairs, and how far each goes** (`experiments/classify/fixes.py`; basis,
`L` and nuisance held at the benchmark's choice, so the difference is the
repair):

- **Phase as a profiled nuisance.** `LawClassifier(shift_grid=...)` scores a
  window against each law at its best shift and aligns each class's windows
  before fitting its laws -- what profiling the level does for *how much*,
  done for *when*. The shift range is chosen by CV from 0 to 20% of the
  series. On the shape group the mean error falls from 0.140 to
  0.128 (better on 4, worse on 2 of 15): GunPoint 0.087 to 0.027, below
  the archive's DTW (0.087); TwoPatterns 0.058 to 0.008; Trace
  0.260 to 0.120. Still behind DTW on average (0.094), and CV on a few
  dozen series picks wrongly both ways: on CBF it declines the warp
  (0.106), on ECG200 it takes one that hurts (0.150 to 0.190).
- **A discriminative head on laws.** `LawStack` feeds cross-fitted law
  evidence (class log-posteriors, best-law residuals) to a logistic
  regression. On the daily group it closes part of the gap to the best raw
  classifier -- 0.171 to 0.158 against 0.137 -- and beats it on
  1 of 9 datasets (on the shape group 6 of 15; Trace reaches 0.020).
  With a few dozen series and many classes the cross-fitting is too noisy
  and it backfires: FaceFour 0.159 to 0.307, Lightning7 0.356 to 0.493.

**Few labels: V10 on real data.** With `m` labelled series per class drawn
from the training split (5 draws x 3 seeds each), the basis size chosen from
the training series' own leave-one-out fit -- no label, so usable at `m` = 1:

| | daily, `m` = 1 | daily, `m` = 10 | shape, `m` = 1 | shape, `m` = 10 |
|---|---|---|---|---|
| nearest law (`L = all`) | 0.395 | 0.240 | 0.396 | 0.208 |
| raw 1-NN | 0.401 | 0.248 | 0.410 | 0.223 |
| one law per class (`L = 1`) | 0.395 | 0.290 | 0.396 | 0.304 |
| raw nearest centroid | 0.401 | 0.305 | 0.410 | 0.308 |

The direction V10 predicts holds on average -- each law rule is below its
raw counterpart at every `m` -- but the effect on real data is small, a point
or two, and not uniform: one law per class beats the raw centroid (the same
rule at dimension `p` instead of `n`) in only 46 of 90 (dataset, `m`)
cells. Real classes are not isotropic noise around one law, which is what
the formula assumes.

**Irregular sampling is close to a tie, and that sharpens the sensor-gap
result.**
Each series keeps a random share of its samples, a different subset per
series; the law classifier scores what was kept at its own times, the raw
rules see the series after linear interpolation. Mean test error:

| share kept | daily: law | daily: interpolate, 1-NN | shape: law | shape: interpolate, 1-NN |
|---|---|---|---|---|
| 100% | 0.172 | 0.203 | 0.140 | 0.158 |
| 25% | 0.335 | 0.356 | 0.240 | 0.222 |
| 10% | 0.405 | 0.419 | 0.396 | 0.392 |

On the daily group the law is one to three points better on average at every
share kept, but not systematically -- per dataset it wins about as often as
it loses -- and on the shape group there is no advantage at all. At 10% the
law is at or below interpolated 1-NN on 11 of 24 datasets. Scattered samples are exactly what linear
interpolation handles well -- every gap is short, so the interpolated
profile is close to the true one. The I-94 gaps above are the opposite
case, **contiguous** outages of six to eighteen hours, where interpolation
draws a straight line across the morning peak and the law does not need to
guess. The method's edge is at long gaps, not at sparse sampling as such.

**Clustering the same data** (train and test pooled, `K` the number of
classes, ARI against the archive's labels, 3 seeds): mechanism K-means in a
cosine basis, rank chosen by leave-one-out, against K-means on the raw
profile.

| | daily, all samples | daily, 25% kept | shape, all samples | shape, 25% kept |
|---|---|---|---|---|
| mechanism K-means | 0.302 | 0.232 | 0.326 | 0.187 |
| K-means on the (interpolated) raw profile | 0.300 | 0.245 | 0.328 | 0.235 |
| soft EM, cosine basis | 0.209 | | 0.300 | |

With every sample present the two tie, as they must: at a shared design
mechanism space is a linear change of coordinates of the profile -- the
same finding as the day windows above, now on 24 datasets. With a quarter
of the samples kept at random, mechanism K-means is **worse** than
interpolating first: a projection read off a few scattered samples is
noisier than the interpolated profile, the clustering analogue of the tie
in classification. Soft EM in the same basis is below both; a class of a
UCR dataset is not one Gaussian law in a cosine basis, and a likelihood
that assumes it is pays for it.

## Wind power curves: every window its own design

`experiments/wind` takes the method to the data it was built for. The
Kelmarsh wind farm (six 2050 kW turbines, 10-minute SCADA, CC-BY-4.0), cut
into six-hour turbine blocks -- 14626 of them, 2016-04 to 2017-12: `x` the
nacelle wind speed, `y` the power as a share of rated. Every block saw its
own wind speeds, so **no raw profile exists**: to use a profile method the
industry bins each block's power curve (the IEC 61400-12 method of bins,
0.5 m/s) and interpolates the empty bins -- most of them, as a median block
covers 7 of 24.

**There are no operator labels, and that was checked first.** The archive's
curtailment-by-cause columns are zero throughout, the power setpoint is
empty before 2021 and afterwards a controller reference that never caps
output, and the status log records one icing stop in two years. So the
blocks are scored against **physical proxies** computed from covariates the
fit never sees, and reported as proxies: colder air (block temperature
below the median), wake from a neighbouring turbine (wind direction against
the farm layout, within six rotor diameters), and night against day, whose
raw effect is at most 1.7% -- a negative control. Turbulence intensity
was planned and dropped: its column is 84% missing.

**Transfer across turbines** (train on 1-3, test on 4-6 and back; balanced
error, as the wake classes are 4:1; V1's ceiling is what the two proxy laws
would allow if the proxy were exactly a change of law):

| proxy | law classifier | mechanism features + logistic | method of bins, best | V1 ceiling |
|---|---|---|---|---|
| cold vs warm | **0.305** | 0.359 | 0.351 | 0.193 |
| waked vs free | **0.353** | 0.374 | 0.406 | 0.269 |
| night vs day (control) | 0.423 | 0.449 | 0.426 | 0.374 |

The laws beat the method of bins by about five points on both physical
proxies on turbines they never saw, and the control stays near chance for
everyone, as V1 said it would. The achieved error sits above the ceiling: a
proxy is not exactly one law per class.

The prediction declared before the run -- **laws gain most where a block saw
few wind-speed bins** -- holds for temperature (the law's edge over the best
bins method is 0.062 on the narrowest-coverage third and 0.011 on the
widest) and **fails for wake** (0.056 and 0.066).

**The physics.** Below rated power, a turbine's output scales with air
density, which at one pressure is `T_warm / T_cold`: 1.032 between the
cold and warm blocks (7.8 and 16.8 C). The measured cold/warm power ratio
starts at 1.20 at 4.25 m/s and falls to 1.026 at 11.75 m/s -- density near
rated, and something much larger at low wind that season carries with
temperature (where the turbulence column exists, high-turbulence samples make
1.18 times the power at 5 m/s, which fits; it is not proven). One
limitation this exposed: Nystrom centres at quantiles of a skewed design
starve its sparse end, and the law with them is erratic above 9 m/s
(off the binned ratio by up to 0.25); evenly spaced centres
(`NystromBasis(spacing="uniform")`, added after this was seen and used only
here) stay within 0.013 everywhere.

**Without labels** the dominant structure is none of the proxies: mechanism
K-means (K = 2) lines up with none of them or with the season (|ARI| at most
0.025), and K-means on binned curves splits calm from windy blocks (ARI
0.57), because the interpolated profile is mostly extrapolation. **In
real time**, each turbine streamed on its own gives birth to 23 regimes in
all; 5 of them are born within 24-26 December 2016, on 5 different turbines at
once -- the storms of that Christmas -- 4 of them in the windiest 3% of blocks
in the record (the fifth at the 86th percentile). The others fall in
every season.

## Notebooks

Generated from `notebooks/sources.py` and executed by `python notebooks/build.py`,
so every output in them is one the code produced.

| notebook | what it shows |
|---|---|
| `00_summary` | **start here**: the whole project end to end for a non-specialist -- motivation, objective and optimisation, a live worked example, the main results as plots, real days on a calendar, two failures, every task in one figure |
| `09_algorithm_tutorial` | the algorithm built by hand, block by block (start, fit, noise scale, score, assign, optional refine), each checked against the package, with every option |
| `01_quickstart` | simulate, fit the hard loop and soft EM, every `lrdsr.viz` plot, losses |
| `02_theory_and_losses` | the ceiling live, the efficiency table, V8 and V9 checked, the learned loss |
| `03_problem_zoo` | the twelve problems, why `high_frequency` fails, two-input laws, add your own |
| `04_realtime_clustering` | a stream with a new regime, a live animation, latency, drift, CUSUM |
| `05_real_data` | bike sharing and highway traffic, day by day, batch and real time |
| `06_kernels_and_classification` | the kernel basis live, V10, the law classifier, irregular sampling, days with sensor gaps |
| `07_partial_days_and_streams` | V11 live, which hours matter, early decision, streaming with gaps, the kernel birth |
| `08_wind_power_curves` | Kelmarsh blocks, the proxies, transfer across turbines, the physics check |

## Layout

```text
lrdsr/core/        the method: model.py (hard loop), soft.py (EM), online.py
                   (real time), mechanism_space.py, backends.py, losses.py,
                   kernel.py (RKHS and function bases), classify.py
                   (law classifier, mechanism features)
lrdsr/theory/      verification.py (the ceiling, V1-V4, V7), losses.py (V8),
                   sequential.py (V9), classification.py (V10), partial.py (V11)
lrdsr/viz.py       plots for any fit, and a live stream animation
experiments/       one block per question: theory, estimator, functions,
                   problems, losses, online, realdata, kernel, classify, wind,
                   srbaseline, robustness, openlaws
results/<block>/   committed CSVs; every figure and number reads these
analysis/          plots.py + figs_*.py (every figure), report.py (every number)
notebooks/         sources.py -> executed .ipynb
docs/              the algorithm guide (PDF) and diagrams (docs/algorithm/)
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

## Data

The real data are public and are downloaded on first use (not committed);
every archive's sha256 is checked on each load.

- **Capital Bikeshare**: UCI Bike Sharing Dataset (Fanaee-T & Gama, 2013), CC BY 4.0.
- **I-94 traffic**: UCI Metro Interstate Traffic Volume (Hogue, 2019), CC BY 4.0.
- **UCR archive**: the UCR Time Series Classification Archive (Dau et al., 2018), via timeseriesclassification.com.
- **Kelmarsh wind farm**: SCADA data, Plumley (2022), Zenodo, doi:10.5281/zenodo.5841834, CC BY 4.0.

## Contributing

Contributions are welcome, and especially **applications to new data**: if
your data come in short windows that may follow a few different laws (sensor
readings, daily or weekly profiles, machine operating curves, physiological
cycles), try the method on them and open a pull request or an issue with what
you found. New law engines, bases, problems for the zoo and corrections are
welcome too. `CONTRIBUTING.md` explains how to add a dataset and the house
rules that keep every number checkable.

## How to cite

```bibtex
@software{jacoby_lrdsr,
  author  = {Jacoby, Dror},
  title   = {LR-DSR: latent-regime clustering by symbolic law},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/drorjac/lrdsr-clustering}
}
```

GitHub's "Cite this repository" button gives the same from `CITATION.cff`.

## License and origins

MIT (see `LICENSE`); the code and results are provided as is. The method grew
out of earlier work on commercial microwave links (CML); some experiment
docstrings still refer to that setting, whose code is not part of this
repository.
