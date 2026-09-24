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

Almost every law here was written down by us, so almost every answer can be
checked against the truth rather than argued. The one exception is the
real-data block, which runs the method on two public datasets and says
plainly that its reference labels are a calendar proxy, not the truth.

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
| `realdata` | what does it find in real measurements? | [real data](#real-data) |

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
{beta_inert_lo:.3f} at every value of `beta` from 0 through {beta_plateau_max:g} --
two orders of magnitude above the default of {beta_default:g} -- and only bites
from {beta_bite:g} ({beta_err_bite:.3f}), and harder at {beta_max:.0f}
({beta_err_at_max:.3f}), where it only makes things *worse*. It is reported as a
constant sitting on a plateau, not as a tuned one.

## The problem zoo

`experiments/problems` is {zoo_problems} problems, each built to break one
assumption: the law is not in the symbolic library, the *gap* between laws
is outside the library's span, `K` = 3 or 4, two inputs, a rescaled copy,
a response with matched mean and variance. Every regime shares one input
distribution, so only the equation can separate them. Mean excess error
over the oracle across all {zoo_cells} cells:

| method | gap to oracle |
|---|---|
| LR-DSR (hard loop) | {zoo_gap_lrdsr:+.3f} |
| soft EM (`SoftLRDSR`) | {zoo_gap_soft_em:+.3f} |
| K-means in mechanism space | {zoo_gap_mechanism_kmeans:+.3f} |
| K-means on the raw sorted profile | {zoo_gap_profile_kmeans:+.3f} |
| best clustering on window summaries | {zoo_gap_geometry:+.3f} |

The hard loop is within 0.02 of the oracle in {zoo_within_lrdsr}/{zoo_cells}
cells, soft EM in {zoo_within_soft}/{zoo_cells}.

**"Out of the library" is usually harmless.** {zoo_outlib_n} problems have laws
no library term can write (`sin 1.3x`, `tanh x`, a moving hinge, `x^1.8`, a
damped sine) but a *gap* the library spans almost entirely; their mean gap
to the oracle is {zoo_outlib_gap_lrdsr:+.3f}, and the laws come back as
surrogates that predict as well as the truth.

**The real failure is a gap outside the span.** `{zoo_worst}` (`sin 4x` vs
`sin 4.6x`) leaves {zoo_worst_outside_pct:.0f}% of the gap outside the library.
There the hard loop is {zoo_worst_gap_lrdsr:+.3f} from the oracle, soft EM
{zoo_worst_gap_soft:+.3f} and mechanism K-means {zoo_worst_gap_mech:+.3f}: the
hard loop damages the start it was given, because the laws it fits cannot
represent what separates the regimes. Leave that problem out and the hard loop
and soft EM sit at {zoo_rest_gap_lrdsr:+.3f} and {zoo_rest_gap_soft:+.3f}. The
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
Huber has `eta` = {eta_gaussian_huber:.3f} and Cauchy {eta_gaussian_cauchy:.3f}; under
Student-t(3) Huber {eta_student_t3_huber:.2f} and Cauchy {eta_student_t3_cauchy:.2f} against a
ceiling of {eta_student_t3_lrt:.0f}; under 10% contamination Huber {eta_contaminated_10_huber:.1f}
and Cauchy {eta_contaminated_10_cauchy:.1f} against {eta_contaminated_10_lrt:.1f} -- the
squared loss needs nine times the samples. Simulated directly, the formula is
within the Monte-Carlo band in {v8_small_within}/{v8_small_cells} cells for every
smooth loss when the per-sample gap is at most a quarter of the noise scale,
and {v8_mid_within}/{v8_mid_cells} up to half. Above that it is optimistic (only
{v8_big_share:.0%} of cells in band), and the non-smooth absolute loss and squared
loss under heavy tails at small `n` break it earlier; `v8_verdict.csv` bins
it all by gap.

**The estimator, under four noise laws** (mean matched error; 3 pairs x 2
separations x 3 seeds):

| | Gaussian | Laplace | Student-t(3) | 10% contaminated |
|---|---|---|---|---|
| oracle, true-noise likelihood ratio | {le_gaussian_oracle_lrt:.3f} | {le_laplace_oracle_lrt:.3f} | {le_student_t3_oracle_lrt:.3f} | {le_contaminated_10_oracle_lrt:.3f} |
| oracle, squared loss | {le_gaussian_oracle_squared:.3f} | {le_laplace_oracle_squared:.3f} | {le_student_t3_oracle_squared:.3f} | {le_contaminated_10_oracle_squared:.3f} |
| hard loop, **learned loss** | {le_gaussian_hard_learned:.3f} | {le_laplace_hard_learned:.3f} | {le_student_t3_hard_learned:.3f} | {le_contaminated_10_hard_learned:.3f} |
| hard loop, Huber (default) | {le_gaussian_hard_huber:.3f} | {le_laplace_hard_huber:.3f} | {le_student_t3_hard_huber:.3f} | {le_contaminated_10_hard_huber:.3f} |
| hard loop, squared | {le_gaussian_hard_squared:.3f} | {le_laplace_hard_squared:.3f} | {le_student_t3_hard_squared:.3f} | {le_contaminated_10_hard_squared:.3f} |
| soft EM, Student-t noise | {le_gaussian_soft_student_t:.3f} | {le_laplace_soft_student_t:.3f} | {le_student_t3_soft_student_t:.3f} | {le_contaminated_10_soft_student_t:.3f} |
| soft EM, Gaussian noise | {le_gaussian_soft_gaussian:.3f} | {le_laplace_soft_gaussian:.3f} | {le_student_t3_soft_gaussian:.3f} | {le_contaminated_10_soft_gaussian:.3f} |

**Learning the loss recovers the Bayes rule**: the learned loss is never more
than {le_learned_worst_excess:.3f} above the true-noise oracle, under any noise.
Under Student-t(3) the fitted `nu` converges to {nu_t3_hard:.2f} (hard loop) and
{nu_t3_soft:.2f} (soft EM).

**A bug this analysis found in the default, and its fix.** The Huber
default used to divide residuals by the raw MAD, which is 0.67 sigma, so
`delta = 1.5` was really about 1 sigma, where the V8 table puts Huber at only
~90% efficiency. Residuals are now divided by the normal-consistent MAD
(`lrdsr.core.losses.noise_scale`), and the constant was chosen on the
*tuning* seeds (`experiments/losses/huber_delta.py`): at 1 sigma -- the old
default -- the Gaussian error is {hd_gaussian_old:.3f}, and at the textbook
`delta = 1.345` it is {hd_gaussian_new:.3f}, level with the squared loss
({hd_gaussian_squared:.3f}), while Student-t(3) goes from {hd_student_t3_old:.3f} to
{hd_student_t3_new:.3f}. Laplace noise prefers the old, smaller constant
({hd_laplace_old:.3f} against {hd_laplace_new:.3f}); that is the trade.
`GroupedDCSR(scale_convention="mad", robust_delta=1.5)` reproduces the old
behaviour. On the reporting seeds the default Huber is still
{le_gaussian_hard_huber:.3f} against {le_gaussian_hard_squared:.3f} for squared under
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
no more than every `e^h`. Over {v9_cells} (`rho`, `h`) cells the first-order delay
is off by a median {v9_delay_first_med:.0f}%; with Siegmund's overshoot correction in
the log-likelihood ratio's own tilt, by {v9_delay_tilted_med:.1f}% (worst
{v9_delay_tilted_max:.0f}%). The `e^h` false-alarm bound holds in
{v9_arl_bound_ok}/{v9_arl_cells} cells.

**Streams** (window error, 3 seeds):

| | `rho` = 0.1 | 0.25 | 1 |
|---|---|---|---|
| oracle | {st_oracle_r010:.3f} | {st_oracle_r025:.3f} | {st_oracle_r100:.3f} |
| batch soft EM, in hindsight | {st_batch_soft_r010:.3f} | {st_batch_soft_r025:.3f} | {st_batch_soft_r100:.3f} |
| online, 160-window warm start | {st_online_history160_r010:.3f} | {st_online_history160_r025:.3f} | {st_online_history160_r100:.3f} |
| online, 40-window warm start | {st_online_r010:.3f} | {st_online_r025:.3f} | {st_online_r100:.3f} |
| laws frozen at the warm start | {st_online_frozen_r010:.3f} | {st_online_frozen_r025:.3f} | {st_online_frozen_r100:.3f} |

Online with enough history matches batch-in-hindsight and the oracle; with
too little at low separation it locks onto wrong laws (a negative, and the
reason warm-start size is a parameter). A new law is **born** in every run once
it is detectable, {birth_delay_lo:.0f}-{birth_delay_hi:.0f} windows after it first
appears, with error after the birth at most {birth_err_after_hi:.3f}, and there
were {birth_false:.0f} false births on stationary control streams. Under drift, error
is {drift_frozen:.3f} with frozen laws, {drift_l100:.3f} without forgetting and
{drift_l095:.3f} at `lambda = 0.95`. One window costs
{lat_lo_us:.0f}-{lat_hi_us:.0f} us (at least {lat_rate_lo:,.0f} windows/s). On a Markov
switching sample stream, CUSUM with laws **learned** by warm start is as
accurate as with the true laws ({cusum_acc_learned:.3f} against {cusum_acc_true:.3f} at
`rho` = 1), with a mean delay of {cusum_delay_learned:.1f} samples against
`2h/rho` = {cusum_delay_pred:.0f}.

`python -m experiments.online.live` runs the live demo, and
`notebooks/04_realtime_clustering.ipynb` animates it.

## Real data

`experiments/realdata` runs the method on two public UCI datasets, one window
per day: **Capital Bikeshare** hourly rentals (Washington DC, 2011-12;
{rd_bike_cov_days_kept:.0f} of {rd_bike_cov_days_raw:.0f} days kept) and **I-94 Metro
Interstate** hourly traffic (Minneapolis, 2012-18; {rd_traffic_cov_days_kept:.0f} of
{rd_traffic_cov_days_raw:.0f} days, the rest have sensor gaps). The input is the hour,
as `x = 2 pi h / 24`; the response is `log(1 + count)` minus the day's mean, so
seasonal level is removed and the regime has to be the **shape** of the
daily law. Archives are downloaded once into `.cache/`, and their sha256 is
checked against `experiments/realdata/checksums.json`.

The reference is the **calendar** day type (working day vs weekend or
holiday). It is a proxy, used for scoring only, and the results say exactly
where it is wrong.

| method (K = 2) | bike ARI | traffic ARI |
|---|---|---|
| LR-DSR hard loop | {rd_bike_lrdsr_ari:.3f} | {rd_traffic_lrdsr_ari:.3f} |
| soft EM, Student-t noise | {rd_bike_soft_t_ari:.3f} | {rd_traffic_soft_t_ari:.3f} |
| soft EM, Fourier basis | {rd_bike_soft_fourier_ari:.3f} | {rd_traffic_soft_fourier_ari:.3f} |
| K-means in mechanism space | {rd_bike_mech_ari:.3f} | {rd_traffic_mech_ari:.3f} |
| **K-means on the raw 24-hour profile** | {rd_bike_profile_ari:.3f} | {rd_traffic_profile_ari:.3f} |
| best clustering on window summaries | {rd_bike_geom_ari:.3f} | {rd_traffic_geom_ari:.3f} |

**The honest reading: a plain K-means on the raw profile ties the best
method.** It has to. Every day has the same 24-hour design, so mechanism
space is only a linear change of coordinates of the profile, and the
method's advantage on the simulator -- windows with *different* inputs --
does not exist here. What the method adds on this data is the laws, the
posteriors and the real-time version, not accuracy. The fast library, with
one full harmonic, is also slightly worse than a Fourier basis for a
two-peak commuting day ({rd_bike_lrdsr_ari:.3f} against {rd_bike_soft_fourier_ari:.3f} on bike).

**The remaining error is the calendar, not the method.** The label-free
separation is large (`rho` = {rd_bike_rho_lib:.1f} and {rd_traffic_rho_lib:.1f} on the library
basis, {rd_bike_rho_fourier:.0f} and {rd_traffic_rho_fourier:.0f} on Fourier), so the implied ceiling
is essentially zero error. Of the {rd_bike_disagree:.0f} bike and {rd_traffic_disagree:.0f} traffic
days on which the law-based partition and the calendar disagree, all but three
are days the calendar mislabels: {rd_bike_holiday_working:.0f} and
{rd_traffic_holiday_working:.0f} holidays that are ordinary working days for the bikes and
the traffic (Columbus Day, Veterans Day, Washington's Birthday, the Minnesota
State Fair), {rd_bike_working_off:.0f} and {rd_traffic_working_off:.0f} working days that behave
like days off (the Friday after Thanksgiving, Christmas Eve). The three
exceptions are July 2016 weekends that look like working days on the traffic
sensor, likely a detector anomaly (`realdata_disagreements.csv` lists every
day by date).

**Choosing `K` without labels does not work here, and we say so.** BIC picks
the largest `K` tried ({rd_bic_k:.0f}) on both datasets, because the likelihood
treats 24 correlated hours as independent and real days vary in many ways.
The first extra cluster is interpretable (rainy working days on bike; a
Christmas / New Year / blizzard anomaly cluster on traffic), but BIC is not a
usable stopping rule on this data.

**In real time**, warm-started on the first {rd_warmup_days:.0f} days without labels and
then fed one day at a time, `OnlineLRDSR` matches the calendar on
{rd_bike_online_acc_a6:.1%} of bike days and {rd_traffic_online_acc_a6:.1%} of traffic days
(novelty `alpha` = 1e-6): error {rd_bike_online_err_a6:.1%} and
{rd_traffic_online_err_a6:.1%} against {rd_bike_lrdsr_err:.1%} and {rd_traffic_lrdsr_err:.1%} for the batch
hard loop fitted in hindsight on every day. The
learned loss picks a Student-t on both: `nu` = {rd_bike_nu:.0f} on bike (close to
Gaussian) and {rd_traffic_nu:.0f} on traffic (heavier tails), with the partition
essentially unchanged.

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

## Layout

```text
lrdsr/core/        the method: model.py (hard loop), soft.py (EM), online.py
                   (real time), mechanism_space.py, backends.py, losses.py
lrdsr/theory/      verification.py (the ceiling, V1-V4, V7), losses.py (V8),
                   sequential.py (V9)
lrdsr/viz.py       plots for any fit, and a live stream animation
experiments/       one block per question: theory, estimator, functions,
                   problems, losses, online, realdata
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
