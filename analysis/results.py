"""RESULTS.md: every mission, its verdict, its numbers and its figure.

The README explains the method; this is the scorecard. Like the README it is
generated -- every number from ``analysis.report.numbers()``, every figure
copied from ``figures/`` into ``docs/figures/`` (``figures/`` is a build
directory and is not committed; these are, so the page renders where the
repository is hosted).

    python -m analysis.results        # write RESULTS.md and docs/figures/
"""
from __future__ import annotations

import shutil

from lrdsr import paths

#: The figures RESULTS.md shows, in the order it shows them.
FIGURES = (
    "mechanism_vs_data_space", "attainability", "loop_value", "loss_estimator",
    "online_sequential", "realdata_laws", "realdata_gaps", "v11_which_hours",
    "v11_early_decision", "kernel_rank_sweep", "kernel_zoo", "v10_learning_curve",
    "classify_benchmark", "online_kernel_birth", "wind_transfer", "wind_physics",
)
DOCS = paths.ROOT / "docs" / "figures"

TEMPLATE = """# Results

**Each window of data came from one of `K` unknown closed-form laws; recover
the law each window came from.** This page is the scorecard: fifteen
missions, what each asked, what came out, and a verdict. The method and the
arguments are in the [README](README.md); every number here is generated from
`results/` by `python -m analysis.results`, so it cannot drift from the runs.

The protocol behind every number: tune on seeds {{3, 7, 19}}, report on
{{11, 23, 42}}; true labels reach the final metric only; a prediction is
declared before the run that tests it; negatives are reported.

## Scorecard

| # | mission | verdict | the result in one line |
|---|---|---|---|
| 1 | [Is there a ceiling?](#1-the-ceiling-is-a-formula) | **holds** | `Q(sqrt(n rho)/2)`, exact in {v1_cells} cells; unchanged by a shared component in {v4_unchanged}/{v4_cells} |
| 2 | [Can it be reached without labels?](#2-mechanism-space-reaches-it) | **holds** | K-means in mechanism space: median {v7_gap_median:.4f} from the oracle |
| 3 | [What is the loop for?](#3-the-loop-is-a-stabiliser) | **holds** (not as hoped) | a contraction: error out {loop_out_lo:.3f}-{loop_out_hi:.3f} whatever goes in |
| 4 | [Which problems does it solve?](#4-the-problem-zoo) | **holds, one failure** | {zoo_within_soft}/{zoo_cells} cells within 0.02 of the oracle; the gap outside the library fails |
| 5 | [Which loss?](#5-losses-v8) | **holds** | a loss costs its efficiency; the learned loss is within {le_learned_worst_excess:.3f} of Bayes |
| 6 | [Real time?](#6-real-time-v9) | **holds** | detection delay predicted to a median {v9_delay_tilted_med:.1f}% |
| 7 | [Real data, whole days](#7-real-data-whole-days) | **partly** | ARI {rd_traffic_soft_fourier_ari:.2f}, but a raw-profile K-means ties it |
| 8 | [Days with sensor gaps](#8-days-with-sensor-gaps) | **holds** (small sample) | {gap_lo_n} days with 6-11 hours: law {gap_lo_law:.3f}, imputation {gap_lo_mean:.3f} / {gap_lo_lin:.3f} |
| 9 | [A partial day's error (V11)](#9-a-partial-days-error-v11) | **partly** | ranks which hours matter; the level only within a factor of a few |
| 10 | [A kernel basis](#10-a-kernel-basis) | **partly** | the failure of mission 4 cut by {k_hf_cut_soft:.0f}%; the rank rule did not replicate |
| 11 | [Classification: V10](#11-classification-v10) | **holds** | the learning curve, exact: {v10_fixed_in}/{v10_cells} fixed-design and {v10x_in}/{v10x_n} random-design cells |
| 12 | [24 real classification tasks](#12-24-ucr-datasets) | **partly** | beats raw 1-NN on {ucr_shape_law_vs_1nn_wins}/15 shape and {ucr_daily_law_vs_1nn_wins}/9 daily sets; loses to tuned raw classifiers |
| 13 | [Repairs](#13-repairs) | **partly** | GunPoint {rp_GunPoint_law:.3f} -> {rp_GunPoint_shift:.3f} with a warp; a stack closes part of mission 12's gap |
| 14 | [Streaming with gaps](#14-streaming-with-gaps) | **holds** | {st_partial_n} partial days streamed at {st_partial_acc:.1%}; a kernel sees a newcomer the library cannot |
| 15 | [Wind power curves](#15-wind-power-curves) | **partly** | laws beat the method of bins by {w_cold_margin:.3f} and {w_waked_margin:.3f}; one declared prediction held, one failed |

---

## 1. The ceiling is a formula

An oracle that knows both laws errs at `P_err = Q(sqrt(n rho)/2)` (`rho` the
separation between the laws in noise units, `n` the window length). The
identity is exact, not asymptotic, in {v1_cells} cells, and a component both laws
share, scaled to {v4_multiple:.0f}x the gap, leaves the error unchanged in
{v4_unchanged}/{v4_cells}.

## 2. Mechanism space reaches it

Profile out the nuisance, subtract the shared law, whiten, project: windows
become an isotropic Gaussian mixture with centres `sqrt(n rho)` apart, where
plain K-means is the right rule. Over {v7_cells} cells its median gap to the
oracle is {v7_gap_median:.4f}. On the pair battery, where the geometry carries nothing:
oracle {bench_oracle:.3f}, LR-DSR {bench_lrdsr:.3f}, mechanism K-means {bench_mechanism_kmeans:.3f}, best clustering
of window summaries {bench_window_features:.3f}.

![](docs/figures/mechanism_vs_data_space.png)
![](docs/figures/attainability.png)

## 3. The loop is a stabiliser

The alternating fit-and-reassign loop does not beat its initialiser. Handed
starts of controlled quality -- error in from {loop_in_lo:.3f} to {loop_in_hi:.3f} -- it returns
{loop_out_lo:.3f} to {loop_out_hi:.3f}: a contraction onto a fixed point. It repairs a bad
start (+{loop_gain_chance:.3f} from chance) and slightly damages a perfect one
({loop_gain_perfect:+.3f}).

![](docs/figures/loop_value.png)

## 4. The problem zoo

Twelve problems, each built to break one assumption. Mean gap to the oracle:
soft EM {zoo_gap_soft_em:+.3f}, mechanism K-means {zoo_gap_mechanism_kmeans:+.3f}, raw-profile K-means {zoo_gap_profile_kmeans:+.3f},
window summaries {zoo_gap_geometry:+.3f}. Laws outside the symbolic library are harmless
when the *gap* is inside its span ({zoo_outlib_n} problems, {zoo_outlib_gap_lrdsr:+.3f}). The failure is
`{zoo_worst}`, whose gap is {zoo_worst_outside_pct:.0f}% outside the span ({zoo_worst_gap_soft:+.3f} for soft EM) --
mission 10's target.

## 5. Losses (V8)

A loss costs exactly its efficiency: `P_err = Q(sqrt(n rho eta)/2)`, in band in
{v8_small_within}/{v8_small_cells} cells at small per-sample gaps. Under 10% contamination the best
loss is {eta_contaminated_10_lrt:.1f} times as efficient as the squared loss -- the squared loss needs
that many times the samples -- and Huber recovers {eta_contaminated_10_huber:.1f} of it. A **learned**
loss -- the noise model refitted every iteration -- is within
{le_learned_worst_excess:.3f} of the true-noise Bayes rule under all four noise laws; a soft EM
that *assumes* Gaussian noise collapses under heavy tails
({le_student_t3_soft_gaussian:.3f} under Student-t(3)).

![](docs/figures/loss_estimator.png)

## 6. Real time (V9)

The per-sample KL divergence between two laws is `rho/2`, so CUSUM's delay is
a formula: predicted to a median {v9_delay_tilted_med:.1f}% (worst {v9_delay_tilted_max:.1f}%), and the
`e^h` false-alarm bound holds in {v9_arl_bound_ok}/{v9_arl_cells} cells. The online clusterer
matches batch-in-hindsight with enough history, gives birth to new regimes
{birth_delay_lo:.0f}-{birth_delay_hi:.0f} windows after they appear, with no false births, at
{lat_lo_us:.0f}-{lat_hi_us:.0f} us per window.

![](docs/figures/online_sequential.png)

## 7. Real data, whole days

Bike sharing (DC) and I-94 traffic (Minneapolis), one window per day, scored
against the calendar as a proxy. Against the calendar the laws reach ARI
{rd_bike_soft_fourier_ari:.3f} (bike) and {rd_traffic_soft_fourier_ari:.3f} (traffic), and the days they disagree on are
mostly days the calendar mislabels. **But K-means on the raw 24-hour profile
ties** ({rd_bike_profile_ari:.3f}, {rd_traffic_profile_ari:.3f}): with one shared design, mechanism space is a
linear change of coordinates of the profile. Choosing `K` by BIC does not
work on this data. In real time: {rd_traffic_online_acc_a6:.1%} of traffic days.

![](docs/figures/realdata_laws.png)

## 8. Days with sensor gaps

A third of the I-94 days ({gap_days}) had been dropped for sensor gaps. A law does
not need a complete day: it is scored at the observed hours with the level as
a nuisance. Against the calendar, days with **6-11 observed hours** ({gap_lo_n} days):
law {gap_lo_law:.3f}, hour-mean imputation {gap_lo_mean:.3f}, linear imputation {gap_lo_lin:.3f}. The
controlled version (real gap masks on complete days, {tp_lo_n} masked days) agrees:
the law keeps its full-day decision {tp_lo_law:.1%} of the time, imputation
{tp_lo_mean:.1%} and {tp_lo_lin:.1%}. Not a sweep: at 12-17 hours hour-mean imputation
edges it ({gap_mid_mean:.3f} vs {gap_mid_law:.3f}).

![](docs/figures/realdata_gaps.png)

## 9. A partial day's error (V11)

For the classifier actually used, with correlated hours, the error of a day
observed at hours `H` is exact:
`Q((|g~_H|^2/2 +/- s^2 log(pi_1/pi_0)) / sqrt(g~_H^T Sigma_H g~_H))`.

- **Which hours matter, predicted:** removing a 6- or 12-hour block at each of
  the 24 start hours, predicted and observed errors rank the starts alike
  (Spearman {v11_rho_min:.2f}-{v11_rho_max:.2f}); the worst start is named in {v11_worst_hits}/{v11_worst_n} cases.
- **When a day is decided:** I-94 days are decided at 99% confidence by 5 am
  ({v11_traffic_5h_pred:.3f} predicted, {v11_traffic_5h_obs:.3f} observed).
- **The level is not in band:** conservative mid-range ({v11_cal_mid_pred:.3f} vs {v11_cal_mid_obs:.3f}),
  optimistic in the far tail ({v11_cal_low_pred:.4f} vs {v11_cal_low_obs:.4f}).

![](docs/figures/v11_which_hours.png)
![](docs/figures/v11_early_decision.png)

## 10. A kernel basis

Nystrom (RBF), random-Fourier, cosine and Legendre bases as drop-in
replacements for the term library. On mission 4's failure the gap outside
the span falls from {k_hf_out_lib:.0f}% to {k_hf_out_16:.2f}%, and the gap to the oracle from
{k_hf_soft_em_library:+.3f} to {k_hf_soft_em_kernel:+.3f} (soft EM); over all twelve problems {k_gap_soft_em_library:+.3f} to
{k_gap_soft_em_kernel:+.3f}. **Negative:** the rank is chosen without labels, and the rule the
tuning seeds picked (`{k_rule}`, {k_tune_snr_1se:.4f} vs {k_tune_snr:.4f}) lost on the reporting seeds
({k_rep_chosen:.4f} vs {k_rep_snr:.4f}); the SNR variants are within seed noise of each other.
It is reported as chosen.

![](docs/figures/kernel_zoo.png)

## 11. Classification: V10

A class is a set of laws: `LawClassifier` fits `L` laws per class with the
LR-DSR loop inside each class and scores windows of any design. Its
learning curve is a formula whose price is the **basis dimension `p`, not the
window length**. The exact three-scalar form is in band in {v10_fixed_in}/{v10_cells}
fixed-design cells; at a random design the estimated laws' inverse-Wishart
tail is computed exactly and {v10x_in}/{v10x_n} cells are in band, the {v10x_big_n} with
`p/(m n) > 0.2` included.

![](docs/figures/v10_learning_curve.png)

## 12. 24 UCR datasets

Nine daily-cycle and fifteen shape datasets, the archive's own splits,
hyperparameters by CV on the training split.

| | daily | shape |
|---|---|---|
| law classifier vs raw 1-NN (wins or ties) | {ucr_daily_law_vs_1nn_wins}/9 ({ucr_daily_law_vs_1nn_a:.3f} vs {ucr_daily_law_vs_1nn_b:.3f}) | {ucr_shape_law_vs_1nn_wins}/15 ({ucr_shape_law_vs_1nn_a:.3f} vs {ucr_shape_law_vs_1nn_b:.3f}) |
| law features vs raw, same SVM | {ucr_daily_msvm_wins}/9 ({ucr_daily_msvm_a:.3f} vs {ucr_daily_msvm_b:.3f}) | {ucr_shape_msvm_wins}/15 ({ucr_shape_msvm_a:.3f} vs {ucr_shape_msvm_b:.3f}) |
| best law vs best tuned raw classifier | {ucr_daily_best_wins}/9 ({ucr_daily_best_a:.3f} vs {ucr_daily_best_b:.3f}) | {ucr_shape_best_wins}/15 ({ucr_shape_best_a:.3f} vs {ucr_shape_best_b:.3f}) |

The law representation helps a fixed rule; a generative law classifier does
**not** beat the best tuned discriminative classifier on a full training set.
With few labels the gain is a point or two; scattered irregular sampling and
clustering are close to ties (linear interpolation handles short gaps well --
the edge of mission 8 is at *long* gaps).

![](docs/figures/classify_benchmark.png)

## 13. Repairs

- **Phase as a nuisance** (a law played early or late): shape-group error
  {rp_shape_law:.3f} -> {rp_shape_shift:.3f}; GunPoint {rp_GunPoint_law:.3f} -> {rp_GunPoint_shift:.3f}, below DTW ({rp_GunPoint_dtw:.3f}); TwoPatterns
  {rp_TwoPatterns_law:.3f} -> {rp_TwoPatterns_shift:.3f}; Trace {rp_Trace_law:.3f} -> {rp_Trace_shift:.3f}. Still behind DTW on average ({rp_shape_dtw:.3f}),
  and CV on a few dozen series sometimes picks the wrong warp.
- **A discriminative head on laws:** daily-group error {rp_daily_law:.3f} -> {rp_daily_stack:.3f}
  against {rp_daily_raw:.3f} for the best raw classifier -- part of the gap, not all;
  unstable on tiny many-class sets (FaceFour {rp_FaceFour_law:.3f} -> {rp_FaceFour_stack:.3f}).

## 14. Streaming with gaps

The online clusterer now takes ragged windows, a level nuisance and a kernel
basis. Every I-94 day streams in calendar order -- {st_all_n} days, {st_partial_n} partial --
and the partial days are sorted as well as the complete ones ({st_partial_acc:.1%} vs
{st_complete_acc:.1%}). A newcomer law the term library cannot represent is born in
{cb_kernel_10}/{cb_seeds} streams at `rho` = 1 with a Nystrom basis and {cb_library_10}/{cb_seeds} with the library,
whose misfit inflates the noise the novelty test measures.

![](docs/figures/online_kernel_birth.png)

## 15. Wind power curves

Kelmarsh wind farm: {w_blocks} six-hour turbine blocks, each with its own wind
speeds -- the method's home ground, where no raw profile exists and industry
bins each block's curve instead (a median block covers {w_cov_med:.0f} of {w_cov_tot} bins).
The archive has **no operator labels** (checked), so blocks are scored
against physical proxies. On turbines never trained on (balanced error):

| proxy | law classifier | method of bins, best | V1 ceiling |
|---|---|---|---|
| cold vs warm | **{w_cold_law:.3f}** | {w_cold_bins_best:.3f} | {w_cold_ceiling:.3f} |
| waked vs free | **{w_waked_law:.3f}** | {w_waked_bins_best:.3f} | {w_waked_ceiling:.3f} |
| night vs day (negative control) | {w_night_law:.3f} | {w_night_bins_best:.3f} | {w_night_ceiling:.3f} |

The declared prediction that laws gain most on narrow wind-speed coverage
**held for temperature and failed for wake**. The cold/warm curve ratio
falls from {w_ratio_lo:.2f} at low wind to {w_ratio_hi:.3f} near rated, onto the air-density
prediction ({w_density:.3f}). Streamed turbine by turbine, {w_births_xmas} regimes are born
on 24-26 December 2016 on {w_births_xmas_turbines} turbines at once -- the storms of that Christmas.

![](docs/figures/wind_transfer.png)
![](docs/figures/wind_physics.png)

---

## What did not work

Kept here, not in footnotes:

- The loop does not improve on a good start (mission 3).
- On whole days with one shared design, a raw-profile K-means ties the method (7).
- Choosing `K` by BIC fails on real days (7).
- V11 ranks masks well but its absolute error level is off by a factor of a few (9).
- The rank rule chosen on the tuning seeds lost on the reporting seeds (10).
- A generative law classifier loses to the best tuned raw classifier (12); the
  repairs close part of the gap, not all, and misfire on tiny datasets (13).
- Sparse scattered sampling gives no systematic edge; only long gaps do (12).
- The declared coverage prediction failed for wake (15); unsupervised
  clusters of wind blocks match no physical proxy (15).

## Reproduce

```bash
pip install -e ".[dev]"
python -m experiments all          # every CSV in results/ (downloads ~0.5 GB of public data)
python -m analysis.plots           # every figure
python -m analysis.report --write  # README.md
python -m analysis.results         # this page
python notebooks/build.py          # eight executed notebooks
pytest -q
```
"""


def write(n: dict | None = None) -> None:
    from analysis.report import numbers

    n = dict(numbers() if n is None else n)
    n["w_cold_margin"] = n["w_cold_bins_best"] - n["w_cold_law"]
    n["w_waked_margin"] = n["w_waked_bins_best"] - n["w_waked_law"]
    DOCS.mkdir(parents=True, exist_ok=True)
    for name in FIGURES:
        src = paths.ROOT / "figures" / f"{name}.png"
        if not src.exists():
            raise SystemExit(f"figures/{name}.png is missing -- run `python -m analysis.plots`")
        shutil.copyfile(src, DOCS / f"{name}.png")
    (paths.ROOT / "RESULTS.md").write_text(TEMPLATE.format(**n), encoding="utf-8")


if __name__ == "__main__":
    write()
    print("wrote RESULTS.md and docs/figures/")
