# LR-DSR clustering

**Each window of data came from one of `K` unknown closed-form laws. Recover
the law each window came from.**

Not "cluster the windows and describe the clusters" -- the law *is* the
cluster identity, and two windows are similar when the same equation
produced them, however different they look. This project is the method for
that, and the analysis of when it works.

Almost every law here was written down by us, so almost every answer can be
checked against the truth rather than argued. The exceptions are the
real-data blocks: two public hourly series, whose reference labels are a
calendar proxy and are said to be one, and 24 datasets of the UCR
time-series archive, whose labels are the archive's own.

```bash
pip install -e ".[dev]"
python -m experiments all        # the results every figure and number reads
python -m analysis.plots         # every figure -> figures/
python -m analysis.report        # every number, from results/
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
The rank is chosen **without labels**. Three rules were compared on the
tuning seeds -- per-window leave-one-out (minimum and one-standard-error)
and the spectral SNR of mechanism space (excess spread over the noise per
root dimension) -- and `snr` was kept (mean error 0.076 against
0.095 for leave-one-out; the rank chosen against the truth, an optimistic
bound, 0.064). Leave-one-out asks how rich *one* window's law is, and a
noisy window cannot justify resolving `sin 4x` (on `high_frequency` at
`rho` = 0.25 it errs at 0.427 against 0.042 for the SNR rule).

On the reporting seeds, gap to the oracle:

| method | `high_frequency` | all 12 problems |
|---|---|---|
| soft EM, library | +0.096 | +0.018 |
| mechanism K-means, library | +0.079 | +0.017 |
| mechanism K-means, kernel | +0.038 | +0.013 |
| **soft EM, kernel** | +0.032 | +0.011 |
| K-means on the raw profile | | +0.130 |

The kernel basis takes the gap outside the span from 37% to
0.01% at rank 17 and cuts the failure by about half for mechanism
K-means and two thirds for soft EM, without costing the other problems: soft EM in the kernel basis is better than in the library on
7 of 12 and worse by more than 0.002 on 2. What is left on
`high_frequency` is variance, not bias: at `rho` = 0.1 even the best rank
chosen against the truth errs at 0.153, and the SNR rule, which picks a
different rank per seed, at 0.229. The price of the kernel is the name:
the law is a dense RKHS function, not an expression.

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
at chance.

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

## Notebooks

Generated from `notebooks/sources.py` and executed by `python notebooks/build.py`,
so every output in them is one the code produced.

| notebook | what it shows |
|---|---|
| `01_quickstart` | simulate, fit the hard loop and soft EM, every `lrdsr.viz` plot, losses |
| `02_theory_and_losses` | the ceiling live, the efficiency table, V8 and V9 checked, the learned loss |
| `03_problem_zoo` | the twelve problems, why `high_frequency` fails, two-input laws, add your own |
| `04_realtime_clustering` | a stream with a new regime, a live animation, latency, drift, CUSUM |
| `05_real_data` | bike sharing and highway traffic, day by day, batch and real time |
| `06_kernels_and_classification` | the kernel basis live, V10, the law classifier, irregular sampling, days with sensor gaps |

## Layout

```text
lrdsr/core/        the method: model.py (hard loop), soft.py (EM), online.py
                   (real time), mechanism_space.py, backends.py, losses.py,
                   kernel.py (RKHS and function bases), classify.py
                   (law classifier, mechanism features)
lrdsr/theory/      verification.py (the ceiling, V1-V4, V7), losses.py (V8),
                   sequential.py (V9), classification.py (V10)
lrdsr/viz.py       plots for any fit, and a live stream animation
experiments/       one block per question: theory, estimator, functions,
                   problems, losses, online, realdata, kernel, classify
results/<block>/   committed CSVs; every figure and number reads these
analysis/          plots.py + figs_*.py (every figure), report.py (every number)
notebooks/         sources.py -> executed .ipynb
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
