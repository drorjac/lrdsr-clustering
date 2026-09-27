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
checked against the truth rather than argued. The exceptions are the
real-data blocks: two public hourly series, whose reference labels are a
calendar proxy and are said to be one; 24 datasets of the UCR time-series
archive, whose labels are the archive's own; and a wind farm's SCADA,
scored against physical proxies because it has no operator labels.

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

### Days with sensor gaps: the first real windows with different designs

Everything above keeps a traffic day only if all 24 hours are present,
because every window then shares one design. That drops a third of the
days, and on those days the raw-profile baseline that tied the method no
longer exists: a profile needs its hours. A law does not -- it is scored at
whichever hours were observed, with the day's level profiled out as a
nuisance (`experiments/realdata/gaps.py`). There are {gap_days} such days with at
least {gap_min_hours} observed hours.

Label-free: the {gap_complete} complete days are clustered as before, the two clusters
become two laws, and each partial day goes to the law that explains its own
hours. Against the calendar:

| hours observed | days | law at the observed hours | impute hour means, then profile | impute linearly, then profile |
|---|---|---|---|---|
| 6-11 | {gap_lo_n} | {gap_lo_law:.3f} | {gap_lo_mean:.3f} | {gap_lo_lin:.3f} |
| 12-17 | {gap_mid_n} | {gap_mid_law:.3f} | {gap_mid_mean:.3f} | {gap_mid_lin:.3f} |
| all | {gap_all_n} | {gap_all_law:.3f} | {gap_all_mean:.3f} | {gap_all_lin:.3f} |

The law wins where the day is most incomplete, and imputation catches up as
hours return; with 18 or more hours every route agrees. It is not a clean
sweep: at 12-17 hours, filling each gap with the hour's mean over complete
days ({gap_mid_mean:.3f}) edges out the law ({gap_mid_law:.3f}), and the 6-11 bin is only
{gap_lo_n} days. So the same question is asked where the answer is known: every
complete day gets the gap mask of a randomly drawn real gap day, and each
route is scored by agreement with **its own** full-day decision. At 6-11
hours ({tp_lo_n} masked days over three seeds) the law keeps its decision
{tp_lo_law:.1%} of the time, hour-mean imputation {tp_lo_mean:.1%} and linear imputation
{tp_lo_lin:.1%}; at 12-17 hours, {tp_mid_law:.1%}, {tp_mid_mean:.1%} and {tp_mid_lin:.1%}. With the
calendar as training labels instead of clusters (a classifier, not a
clustering), the totals are {gap_sup_law:.3f} for the law and {gap_sup_mean:.3f} for hour-mean
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
  start hours like the observed one, Spearman {v11_rho_min:.2f} to {v11_rho_max:.2f} over both
  datasets and block lengths, and names the worst start hour in {v11_worst_hits} of {v11_worst_n}
  cases. Losing the night and early morning costs I-94 up to
  {v11_tr12_max:.0%}; losing the middle of the day costs nothing.
- **A day is decided by 5 am.** Read hour by hour from midnight, the share of
  I-94 days decided correctly at 99% confidence is predicted at
  {v11_traffic_5h_pred:.3f} after five hours and observed at {v11_traffic_5h_obs:.3f}; bike days by six hours,
  {v11_bike_6h_pred:.3f} against {v11_bike_6h_obs:.3f}. At the knee of the curve it is optimistic (four
  hours on I-94: {v11_traffic_4h_pred:.2f} predicted, {v11_traffic_4h_obs:.2f} observed).
- **The ranking is right; the level is within a factor of a few.** Binned
  by predicted error, it is conservative in the middle (traffic blocks
  predicted at {v11_cal_mid_pred:.3f}, observed {v11_cal_mid_obs:.3f}) and optimistic in the far tail
  ({v11_cal_low_pred:.4f} against {v11_cal_low_obs:.4f}), where a few days that are not Gaussian about
  their law set the rate.

### Streaming days with gaps, and a law the library cannot write

`OnlineLRDSR` now takes a per-window nuisance (the level, profiled out before
a window is scored or absorbed), windows of any length, and a kernel basis
fitted on its warm start (`experiments/online/ragged.py`).

**Every I-94 day, in calendar order.** The first real-time pass had to drop
the days with sensor gaps. Streamed with them -- {st_all_n} days, {st_partial_n} of them
partial -- the partial days are sorted as well as the complete ones:
{st_partial_acc:.1%} against {st_complete_acc:.1%}, and the complete-days-only stream with the same
settings reaches {st_only_acc:.1%}. (This stream uses a Fourier basis and the level
nuisance; the {rd_traffic_online_acc_a6:.1%} above used the library on centred days, so the two
are not compared.)

**A newcomer only a kernel can see.** A stream of `high_frequency` windows:
warm start on `sin 4x` alone, `sin 4.6x` appears at window 40. With a Nystrom
basis the newcomer is born in {cb_kernel_10}/{cb_seeds} streams at `rho` = 1 and {cb_kernel_20}/{cb_seeds} at 2,
{cb_kernel_delay:.1f} windows after it first appears, and every later window is sorted
correctly ({cb_kernel_err:.3f} error). With the term library it is born in {cb_library_10}/{cb_seeds} at `rho` = 1
and {cb_library_20}/{cb_seeds} at 2, although a known-law test would flag {cb_power_10:.0%} of newcomer
windows at `rho` = 1. The library's law for `sin 4x` is a poor surrogate, its
misfit inflates the noise the novelty test measures, and the newcomer hides
in it. No false birth on the control streams ({cb_false} in total).

## A kernel basis instead of a library

The problem zoo's one real failure was a gap outside the library: `sin 4x`
against `sin 4.6x` leaves {k_hf_out_lib:.0f}% of the difference outside the span of the
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
and `{k_rule}` was kept, at mean error {k_tune_snr_1se:.4f} against {k_tune_snr:.4f} for the plain
SNR maximum, {k_tune_snr_smooth:.4f} smoothed and {k_tune_loo_min:.4f} for leave-one-out (the rank chosen
against the truth, an optimistic bound, {k_tune_best:.4f}). Leave-one-out asks how
rich *one* window's law is, and a noisy window cannot justify resolving
`sin 4x` (on `high_frequency` at `rho` = 0.25 it errs at {k_hf_loo_r025:.3f}).

**The rule the tuning seeds chose does not hold up on the reporting seeds.**
There `{k_rule}` errs at {k_rep_chosen:.4f}, against {k_rep_snr:.4f} for the plain SNR maximum and
{k_rep_snr_smooth:.4f} smoothed: the three SNR variants are within seed-to-seed noise
of each other, and the tuning seeds picked one by chance. Its preference for
the smallest adequate rank gives part of the gap back on `high_frequency`.
The numbers below are the rule the protocol chose; switching now, after
seeing the reporting seeds, would be selecting on them.

On the reporting seeds, gap to the oracle:

| method | `high_frequency` | all {k_problems} problems |
|---|---|---|
| soft EM, library | {k_hf_soft_em_library:+.3f} | {k_gap_soft_em_library:+.3f} |
| mechanism K-means, library | {k_hf_mech_library:+.3f} | {k_gap_mech_library:+.3f} |
| mechanism K-means, kernel | {k_hf_mech_kernel:+.3f} | {k_gap_mech_kernel:+.3f} |
| **soft EM, kernel** | {k_hf_soft_em_kernel:+.3f} | {k_gap_soft_em_kernel:+.3f} |
| K-means on the raw profile | | {k_gap_profile_kmeans:+.3f} |

The kernel basis takes the gap outside the span from {k_hf_out_lib:.0f}% to
{k_hf_out_16:.2f}% at rank 17 and cuts the failure by {k_hf_cut_mech:.0f}% for mechanism K-means and
{k_hf_cut_soft:.0f}% for soft EM, without costing the other problems: soft EM in the kernel
basis is better than in the library on {k_soft_better} of {k_problems} and worse by more than
0.002 on {k_soft_worse_002}. What is left on `high_frequency` is variance, not bias: at
`rho` = 0.1 even the best rank chosen against the truth errs at
{k_hf_best_r010:.3f}, and the chosen rule, which picks a different rank per seed, at
{k_hf_snr_r010:.3f}. The price of the kernel is the name: the law is a dense RKHS
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
{v10_fixed_in}/{v10_cells} cells (the first-order form in only {v10_fixed_first_in}: it is optimistic
at small `m`). At `n rho = 9` and one labelled window per class, the error
is {v10_p3_m1:.3f} with `p` = 3 and {v10_p31_m1:.3f} with `p` = 31, against a ceiling of
{v10_ceiling9:.3f}. At a random design two corrections are needed and
both were derived, not fitted: a law estimated from `m n` randomly placed
samples costs `(p + 1)/n` of a window per class (the inverse-Wishart
factor), and the oracle itself sits above the ceiling by the Jensen gap of
V1. With both, {v10_rand_small_in}/{v10_rand_small_n} cells with `p/(m n) <= 0.05` are in band; without
them {v10_rand_uncorr_in}/60. They fail where `p/(m n) > 0.2` ({v10_rand_big_in}/{v10_rand_big_n}), where the
inverse-Wishart tail takes over -- at `p` = 31 and `m` = 1 the classifier is
at chance. **That tail can be computed rather than approximated**: sampling
the estimated laws from their exact least-squares law,
`beta^ ~ N(beta, sigma^2 (B^T B)^-1)`, over training and test designs, and
integrating the test noise analytically, puts {v10x_in}/{v10x_n} random-design cells in
band, the {v10x_big_n} with `p/(m n) > 0.2` included ({v10x_big_in}/{v10x_big_n}) -- so V10 is now
closed at every design and every `p/(m n)` tested.

**On 24 real datasets.** `experiments/classify` runs the classifiers on the UCR
archive's own train/test splits: {ucr_daily_n} **daily-cycle** datasets (pedestrian
counts, power demand, freeway loops, household appliances -- a day or a
week as a window, the kind of data this method is for) and {ucr_shape_n} **shape**
benchmarks (gestures, ECG, spectra, simulated patterns), the contrast, where
a class is a pattern with phase jitter rather than a law. The split into
groups was fixed before any run. Every hyperparameter -- basis size, `L`,
the level nuisance, regularisation -- is chosen by CV on the training split;
the test split is touched once. The comparisons that were declared before
the run, mean test error:

| comparison | daily: wins or ties | daily: mean error | shape: wins or ties | shape: mean error |
|---|---|---|---|---|
| law classifier vs raw 1-NN (the archive's reference) | {ucr_daily_law_vs_1nn_wins}/{ucr_daily_n} | {ucr_daily_law_vs_1nn_a:.3f} vs {ucr_daily_law_vs_1nn_b:.3f} | {ucr_shape_law_vs_1nn_wins}/{ucr_shape_n} | {ucr_shape_law_vs_1nn_a:.3f} vs {ucr_shape_law_vs_1nn_b:.3f} |
| law classifier vs raw nearest centroid | {ucr_daily_law_vs_cen_wins}/{ucr_daily_n} | {ucr_daily_law_vs_cen_a:.3f} vs {ucr_daily_law_vs_cen_b:.3f} | {ucr_shape_law_vs_cen_wins}/{ucr_shape_n} | {ucr_shape_law_vs_cen_a:.3f} vs {ucr_shape_law_vs_cen_b:.3f} |
| mechanism features vs raw profile, same logistic regression | {ucr_daily_mlog_wins}/{ucr_daily_n} | {ucr_daily_mlog_a:.3f} vs {ucr_daily_mlog_b:.3f} | {ucr_shape_mlog_wins}/{ucr_shape_n} | {ucr_shape_mlog_a:.3f} vs {ucr_shape_mlog_b:.3f} |
| mechanism features vs raw profile, same RBF SVM | {ucr_daily_msvm_wins}/{ucr_daily_n} | {ucr_daily_msvm_a:.3f} vs {ucr_daily_msvm_b:.3f} | {ucr_shape_msvm_wins}/{ucr_shape_n} | {ucr_shape_msvm_a:.3f} vs {ucr_shape_msvm_b:.3f} |
| best law classifier vs best raw-profile classifier | {ucr_daily_best_wins}/{ucr_daily_n} | {ucr_daily_best_a:.3f} vs {ucr_daily_best_b:.3f} | {ucr_shape_best_wins}/{ucr_shape_n} | {ucr_shape_best_a:.3f} vs {ucr_shape_best_b:.3f} |

Three readings, the last one a negative. **The law representation helps a
fixed decision rule**: the same 1-NN, centroid, logistic regression or SVM
does better on mechanism coordinates than on the raw profile, on a
majority of datasets in both groups (a bare one on the daily group, a clear
one on the shape group) -- the smoothing a basis does is worth more than
the detail it drops. The largest single gains are on shape data whose
classes *are* smooth functions: SyntheticControl {ucr_SyntheticControl_law:.3f} against {ucr_SyntheticControl_ed:.3f}
for raw 1-NN (and {ucr_SyntheticControl_dtw:.3f} for the archive's DTW), TwoPatterns
{ucr_TwoPatterns_law:.3f} against {ucr_TwoPatterns_ed:.3f}. **Where a class is a time-warped pattern the
basis loses** -- Trace {ucr_Trace_law:.3f} against {ucr_Trace_dtw:.3f} for DTW -- and elastic methods
remain the right tool there. **And a generative law classifier does not beat
the best tuned discriminative classifier on a raw profile** when the training
set is full: the best law classifier wins on {ucr_daily_best_wins}/{ucr_daily_n} daily datasets
and {ucr_shape_best_wins}/{ucr_shape_n} shape ones (both sides are best-of on the test set, so
optimistic alike). A readable classifier has a price too: one *named* law per
class from the symbolic library errs at {ucr_daily_symbolic:.3f} on the daily group.

**Two repairs, and how far each goes** (`experiments/classify/fixes.py`; basis,
`L` and nuisance held at the benchmark's choice, so the difference is the
repair):

- **Phase as a profiled nuisance.** `LawClassifier(shift_grid=...)` scores a
  window against each law at its best shift and aligns each class's windows
  before fitting its laws -- what profiling the level does for *how much*,
  done for *when*. The shift range is chosen by CV from 0 to 20% of the
  series. On the shape group the mean error falls from {rp_shape_law:.3f} to
  {rp_shape_shift:.3f} (better on {rp_shape_shift_better}, worse on {rp_shape_shift_worse} of 15): GunPoint {rp_GunPoint_law:.3f} to {rp_GunPoint_shift:.3f}, below
  the archive's DTW ({rp_GunPoint_dtw:.3f}); TwoPatterns {rp_TwoPatterns_law:.3f} to {rp_TwoPatterns_shift:.3f}; Trace
  {rp_Trace_law:.3f} to {rp_Trace_shift:.3f}. Still behind DTW on average ({rp_shape_dtw:.3f}), and CV on a few
  dozen series picks wrongly both ways: on CBF it declines the warp
  ({rp_CBF_shift:.3f}), on ECG200 it takes one that hurts ({rp_ECG200_law:.3f} to {rp_ECG200_shift:.3f}).
- **A discriminative head on laws.** `LawStack` feeds cross-fitted law
  evidence (class log-posteriors, best-law residuals) to a logistic
  regression. On the daily group it closes part of the gap to the best raw
  classifier -- {rp_daily_law:.3f} to {rp_daily_stack:.3f} against {rp_daily_raw:.3f} -- and beats it on
  {rp_daily_stack_beats_raw} of 9 datasets (on the shape group {rp_shape_stack_beats_raw} of 15; Trace reaches {rp_Trace_stack:.3f}).
  With a few dozen series and many classes the cross-fitting is too noisy
  and it backfires: FaceFour {rp_FaceFour_law:.3f} to {rp_FaceFour_stack:.3f}, Lightning7 {rp_Lightning7_law:.3f} to {rp_Lightning7_stack:.3f}.

**Few labels: V10 on real data.** With `m` labelled series per class drawn
from the training split (5 draws x 3 seeds each), the basis size chosen from
the training series' own leave-one-out fit -- no label, so usable at `m` = 1:

| | daily, `m` = 1 | daily, `m` = 10 | shape, `m` = 1 | shape, `m` = 10 |
|---|---|---|---|---|
| nearest law (`L = all`) | {fs_daily_m1_lall:.3f} | {fs_daily_m10_lall:.3f} | {fs_shape_m1_lall:.3f} | {fs_shape_m10_lall:.3f} |
| raw 1-NN | {fs_daily_m1_nn:.3f} | {fs_daily_m10_nn:.3f} | {fs_shape_m1_nn:.3f} | {fs_shape_m10_nn:.3f} |
| one law per class (`L = 1`) | {fs_daily_m1_l1:.3f} | {fs_daily_m10_l1:.3f} | {fs_shape_m1_l1:.3f} | {fs_shape_m10_l1:.3f} |
| raw nearest centroid | {fs_daily_m1_cen:.3f} | {fs_daily_m10_cen:.3f} | {fs_shape_m1_cen:.3f} | {fs_shape_m10_cen:.3f} |

The direction V10 predicts holds on average -- each law rule is below its
raw counterpart at every `m` -- but the effect on real data is small, a point
or two, and not uniform: one law per class beats the raw centroid (the same
rule at dimension `p` instead of `n`) in only {fs_l1_beats_cen} of {fs_cells} (dataset, `m`)
cells. Real classes are not isotropic noise around one law, which is what
the formula assumes.

**Irregular sampling is close to a tie, and that sharpens the sensor-gap
result.**
Each series keeps a random share of its samples, a different subset per
series; the law classifier scores what was kept at its own times, the raw
rules see the series after linear interpolation. Mean test error:

| share kept | daily: law | daily: interpolate, 1-NN | shape: law | shape: interpolate, 1-NN |
|---|---|---|---|---|
| 100% | {irr_daily_100_law:.3f} | {irr_daily_100_nn:.3f} | {irr_shape_100_law:.3f} | {irr_shape_100_nn:.3f} |
| 25% | {irr_daily_25_law:.3f} | {irr_daily_25_nn:.3f} | {irr_shape_25_law:.3f} | {irr_shape_25_nn:.3f} |
| 10% | {irr_daily_10_law:.3f} | {irr_daily_10_nn:.3f} | {irr_shape_10_law:.3f} | {irr_shape_10_nn:.3f} |

On the daily group the law is one to three points better on average at every
share kept, but not systematically -- per dataset it wins about as often as
it loses -- and on the shape group there is no advantage at all. At 10% the
law is at or below interpolated 1-NN on {irr10_law_wins} of {irr10_n} datasets. Scattered samples are exactly what linear
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
| mechanism K-means | {cl_daily_100_mproj:.3f} | {cl_daily_25_mproj:.3f} | {cl_shape_100_mproj:.3f} | {cl_shape_25_mproj:.3f} |
| K-means on the (interpolated) raw profile | {cl_daily_100_raw:.3f} | {cl_daily_25_raw:.3f} | {cl_shape_100_raw:.3f} | {cl_shape_25_raw:.3f} |
| soft EM, cosine basis | {cl_daily_100_soft:.3f} | | {cl_shape_100_soft:.3f} | |

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
into six-hour turbine blocks -- {w_blocks} of them, {w_first} to {w_last}: `x` the
nacelle wind speed, `y` the power as a share of rated. Every block saw its
own wind speeds, so **no raw profile exists**: to use a profile method the
industry bins each block's power curve (the IEC 61400-12 method of bins,
0.5 m/s) and interpolates the empty bins -- most of them, as a median block
covers {w_cov_med:.0f} of {w_cov_tot}.

**There are no operator labels, and that was checked first.** The archive's
curtailment-by-cause columns are zero throughout, the power setpoint is
empty before 2021 and afterwards a controller reference that never caps
output, and the status log records one icing stop in two years. So the
blocks are scored against **physical proxies** computed from covariates the
fit never sees, and reported as proxies: colder air (block temperature
below the median), wake from a neighbouring turbine (wind direction against
the farm layout, within six rotor diameters), and night against day, whose
raw effect is at most {w_night_dep:.1%} -- a negative control. Turbulence intensity
was planned and dropped: its column is {w_sd_missing:.0%} missing.

**Transfer across turbines** (train on 1-3, test on 4-6 and back; balanced
error, as the wake classes are {w_wake_ratio:.0f}:1; V1's ceiling is what the two proxy laws
would allow if the proxy were exactly a change of law):

| proxy | law classifier | mechanism features + logistic | method of bins, best | V1 ceiling |
|---|---|---|---|---|
| cold vs warm | **{w_cold_law:.3f}** | {w_cold_mech_logistic:.3f} | {w_cold_bins_best:.3f} | {w_cold_ceiling:.3f} |
| waked vs free | **{w_waked_law:.3f}** | {w_waked_mech_logistic:.3f} | {w_waked_bins_best:.3f} | {w_waked_ceiling:.3f} |
| night vs day (control) | {w_night_law:.3f} | {w_night_mech_logistic:.3f} | {w_night_bins_best:.3f} | {w_night_ceiling:.3f} |

The laws beat the method of bins by about five points on both physical
proxies on turbines they never saw, and the control stays near chance for
everyone, as V1 said it would. The achieved error sits above the ceiling: a
proxy is not exactly one law per class.

The prediction declared before the run -- **laws gain most where a block saw
few wind-speed bins** -- holds for temperature (the law's edge over the best
bins method is {w_cold_narrow_edge:.3f} on the narrowest-coverage third and {w_cold_wide_edge:.3f} on the
widest) and **fails for wake** ({w_waked_narrow_edge:.3f} and {w_waked_wide_edge:.3f}).

**The physics.** Below rated power, a turbine's output scales with air
density, which at one pressure is `T_warm / T_cold`: {w_density:.3f} between the
cold and warm blocks ({w_Tc:.1f} and {w_Tw:.1f} C). The measured cold/warm power ratio
starts at {w_ratio_lo:.2f} at {w_ws_lo:.2f} m/s and falls to {w_ratio_hi:.3f} at {w_ws_hi:.2f} m/s -- density near
rated, and something much larger at low wind that season carries with
temperature (where the turbulence column exists, high-turbulence samples make
{w_ti_ratio:.2f} times the power at 5 m/s, which fits; it is not proven). One
limitation this exposed: Nystrom centres at quantiles of a skewed design
starve its sparse end, and the law with them is erratic above 9 m/s
(off the binned ratio by up to {w_quantile_dev:.2f}); evenly spaced centres
(`NystromBasis(spacing="uniform")`, added after this was seen and used only
here) stay within {w_uniform_dev:.3f} everywhere.

**Without labels** the dominant structure is none of the proxies: mechanism
K-means (K = 2) lines up with none of them or with the season (|ARI| at most
{w_cl_mech_maxproxy:.3f}), and K-means on binned curves splits calm from windy blocks (ARI
{w_cl_bins_calm:.2f}), because the interpolated profile is mostly extrapolation. **In
real time**, each turbine streamed on its own gives birth to {w_births} regimes in
all; {w_births_xmas} of them are born within 24-26 December 2016, on {w_births_xmas_turbines} different turbines at
once -- the storms of that Christmas -- {w_xmas_top3} of them in the windiest 3% of blocks
in the record (the fifth at the {w_xmas_min_pct:.0f}th percentile). The others fall in
every season.

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
                   problems, losses, online, realdata, kernel, classify, wind
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
