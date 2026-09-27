"""The notebooks' source: one list of ``(kind, text)`` cells per notebook.

``notebooks/build.py`` turns each into an executed ``.ipynb``. Edit here,
never in the ``.ipynb``: the build overwrites it.
"""
from __future__ import annotations

SETUP = r'''
import sys, pathlib
ROOT = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
            if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from lrdsr import viz
viz.style()
pd.set_option("display.precision", 4)
RES = ROOT / "results"
'''

# ==========================================================================
# 01 -- quickstart
# ==========================================================================
QUICKSTART = [
    ("md", r"""
# 01 · Quickstart: recover the law each window came from

Every window of data was produced by one of `K` unknown closed-form laws
$y = f_{z_w}(x) + \varepsilon$. The label $z_w$ is never observed. LR-DSR
recovers **the partition and the laws together**.

This notebook runs the whole method on three laws in about ten seconds, and
draws every picture `lrdsr.viz` offers:

1. simulate windows whose *geometry* barely separates the regimes,
2. look at them in data space and in **mechanism space**,
3. fit the hard alternating loop (`GroupedDCSR`) and the soft EM (`SoftLRDSR`),
4. read off the recovered laws, the cost matrix, the learning curves, the
   posteriors,
5. change the loss, including a **learned** one.
"""),
    ("code", SETUP),
    ("md", r"""
## 1 · Simulate

Three laws, one shared input distribution $x \sim U[-2.2, 2.2]$, 16 noisy
samples a window. The window-level features `Z` a clustering would see are drawn with a
large spread (`geometry_overlap=3`), so they carry little about the regime.
The truth `z` is kept **for scoring only**.
"""),
    ("code", r'''
from experiments.functions.data import simulate_function_windows

X, y, Z, z, equations = simulate_function_windows(
    n_windows=240, window_len=16, noise_std=1.5, geometry_overlap=3.0, seed=11)
true_laws = [lambda x: 0.8 * x**2 + 1.5 * x + 0.5,
             lambda x: 2.5 * np.sin(2 * x) - 0.5,
             lambda x: 0.35 * x**3 - 1.2]
print(X.shape, y.shape, "windows x samples x inputs")
for k, e in enumerate(equations):
    print(f"regime {k}: y = {e}   ({np.sum(z == k)} windows)")
'''),
    ("code", r'''
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
viz.plot_windows(X, y, z, ax=ax[0], max_windows=60)
ax[0].set_title("the samples, coloured by the (hidden) regime")
viz.plot_laws(truth=true_laws, x_range=(-2.2, 2.2), ax=ax[1])
ax[1].set_title("the three true laws")
fig.tight_layout()
'''),
    ("md", r"""
## 2 · Data space against mechanism space

Left: two plain summaries of each window, what any clustering of windows
would see. Right: the same windows in **mechanism space** — whitened law
coefficients (`lrdsr.core.mechanism_space`). Distance there is distance
between *laws*, and the regimes separate.
"""),
    ("code", r'''
from experiments.common.fitting import window_features

Zf, names = window_features(X, y)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
for k in range(3):
    ax[0].scatter(Z[z == k, 0], Z[z == k, 1], s=9, color=viz.regime_color(k), lw=0)
    ax[1].scatter(Zf[z == k, 0], Zf[z == k, 1], s=9, color=viz.regime_color(k), lw=0)
ax[0].set(title="planted geometry Z", xlabel="$Z_1$", ylabel="$Z_2$")
ax[1].set(title="window summaries", xlabel=names[0], ylabel=names[1])
viz.plot_mechanism_space(X, y, z, ax=ax[2], feature_names=["x"])
fig.tight_layout()
'''),
    ("md", r"""
## 3 · Fit the hard loop

`GroupedDCSR` initialises in mechanism space, fits a symbolic law per
cluster (greedy BIC over a fixed library), scores **every window under every
law** out-of-fold, reassigns, and repeats. `alpha_geom=0` makes assignment
purely mechanism-based. `true_labels_for_eval` only adds ARI to the history.
"""),
    ("code", r'''
from lrdsr import GroupedDCSR
from lrdsr.core.evaluation import clustering_metrics

hard = GroupedDCSR(n_clusters=3, alpha_geom=0.0, init="mechanism", backend="fast",
                   backend_kwargs={"max_terms": 5}, score_mode="cross_fit",
                   residual_scale="global", random_state=11)
res = hard.fit(X, y, Z, feature_names=["x"], true_labels_for_eval=z)
print(clustering_metrics(z, res.labels))
for k, m in enumerate(res.models):
    print(f"law {k}: {m.expression()}")
'''),
    ("md", r"""
Read the expressions against the truth before trusting them. With 16 samples
a window at this noise, a law can come back as a **predictively equivalent**
surrogate rather than in its own symbols -- e.g. a quadratic as
$a + b\,x + c\cos x$, since $\cos x \approx 1 - x^2/2$ on this range. The
partition is exact either way; the plot below is the check that the curves
agree where the data live.
"""),
    ("code", r'''
fig, ax = plt.subplots(1, 4, figsize=(15, 3.4), gridspec_kw={"width_ratios": [1.3, 1, 0.8, 1.1]})
viz.plot_laws(res.models, x_range=(-2.2, 2.2), truth=true_laws, ax=ax[0])
ax[0].set_title("recovered (solid) vs true (dashed)")
viz.plot_confusion(z, res.labels, ax=ax[1])
viz.plot_cost_matrix(res.total_cost, res.labels, ax=ax[2])
viz.plot_history(res.history, ax=ax[3], metrics=["changed_fraction"])
ax[3].set_title("the loop, iteration by iteration")
fig.tight_layout()
'''),
    ("md", r"""
## 4 · Fit the soft EM

`SoftLRDSR` is the ordinary EM of the same mixture of library regressions:
every window gets a **posterior** over regimes rather than a hard label, and
the observed-data log-likelihood is a real, monotone learning curve.
"""),
    ("code", r'''
from lrdsr import SoftLRDSR

# started from a RANDOM partition, with deterministic annealing (T: 3 -> 1),
# so the learning curve has something to show
soft = SoftLRDSR(n_clusters=3, feature_names=["x"], init="random", temperature=3.0,
                 random_state=11).fit(X, y, true_labels_for_eval=z)
print(clustering_metrics(z, soft.labels), " sigma per regime:", soft.sigma.round(3))
fig, ax = plt.subplots(1, 3, figsize=(14, 3.2), gridspec_kw={"width_ratios": [1, 1, 1.6]})
viz.plot_history(soft.history, ax=ax[0], metrics=["loglik"])
viz.plot_laws(coef=soft.coef, feature_names=["x"], x_range=(-2.2, 2.2), truth=true_laws, ax=ax[1])
viz.plot_responsibilities(soft.responsibilities, ax=ax[2])
ax[2].set_title(f"posteriors; mean entropy {soft.entropy.mean():.3f} nats")
fig.tight_layout()
'''),
    ("md", r"""
## 5 · Change the loss, or learn it

The equation term scores a window by a per-sample loss on its scaled
residual. `huber` is the default; any of `lrdsr.core.losses.LOSSES` can be
used, and `loss="learned"` refits a noise model (Gaussian / Laplace /
Student-t with fitted $\nu$) to the residuals every iteration, so the loss
is **learned** from the data. Here the noise is made heavy-tailed on purpose.
"""),
    ("code", r'''
rng = np.random.default_rng(0)
clean = np.stack([true_laws[k](X[w, :, 0]) for w, k in enumerate(z)])
y_heavy = clean + 1.0 * rng.standard_t(2.0, size=y.shape)     # Student-t, nu = 2
rows = []
for loss in ("squared", "huber", "cauchy", "learned"):
    m = GroupedDCSR(n_clusters=3, alpha_geom=0.0, init="mechanism", loss=loss,
                    residual_scale="global", random_state=11)
    r = m.fit(X, y_heavy, Z, feature_names=["x"])
    rows.append({"loss": loss, **clustering_metrics(z, r.labels),
                 "learned": getattr(m, "learned_loss_", None) and
                            f"{m.learned_loss_.loss} {m.learned_loss_.params}"})
pd.DataFrame(rows)
'''),
    ("code", r'''
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
viz.plot_losses(ax=ax[0])
viz.plot_losses(ax=ax[1], influence=True)
ax[0].set_title(r"$\rho(r)$: what a residual costs")
ax[1].set_title(r"$\psi(r)$: how hard one sample pulls")
fig.tight_layout()
'''),
    ("md", r"""
## Your own data

Everything above needs three arrays:

```python
X_seq  # (W, n, d)  the inputs of each window
y_seq  # (W, n)     the response
Z      # (W, p)     any window-level features (only the geometry term reads them;
       #            with alpha_geom=0 they can be anything, e.g. zeros)
```

Then `GroupedDCSR(n_clusters=K, alpha_geom=0, init="mechanism").fit(X_seq, y_seq, Z)`
for named laws, `SoftLRDSR(K).fit(X_seq, y_seq)` for posteriors, and
`OnlineLRDSR().warm_start(...)` to keep going in real time (notebook 04).
"""),
]


# ==========================================================================
# 04 -- real time
# ==========================================================================
REALTIME = [
    ("md", r"""
# 04 · Clustering in real time

A batch estimator sees every window before deciding anything. A stream does
not wait. `lrdsr.core.online` has two real-time estimators:

* **`OnlineLRDSR`** — window by window. Each regime keeps the sufficient
  statistics of its law (recursive least squares), so a window is scored,
  assigned and absorbed in microseconds. A window **no** law explains is
  buffered, and a buffer of mutually consistent ones becomes a **new regime**.
  A forgetting factor lets laws drift.
* **`CusumSegmenter`** — sample by sample. With the laws known or learned, a
  single stream that switches regime at unknown times is segmented by CUSUM
  on the log-likelihood ratio, with a delay the theory predicts:
  $\approx h / \mathrm{KL} = 2h/\rho$ samples.
"""),
    ("code", SETUP),
    ("md", r"""
## 1 · A stream in which a new law appears

Two laws for the first 100 windows; from then on a third one, never seen in
the history, arrives too. The model is warm-started on 40 windows of history
**without labels** (a 2-regime `SoftLRDSR` fit), then run.
"""),
    ("code", r'''
from lrdsr import OnlineLRDSR
from lrdsr.core.evaluation import aligned_accuracy

rng = np.random.default_rng(11)
laws = [lambda x: x**2 + x, lambda x: x**2 - x, lambda x: 2 * np.sin(3 * x)]
n, sigma = 32, 1.0

def windows(ks):
    x = rng.uniform(-2, 2, (len(ks), n))
    y = np.stack([laws[k](x[i]) for i, k in enumerate(ks)]) + rng.normal(0, sigma, x.shape)
    return x[:, :, None], y

z_hist = rng.integers(0, 2, 40)
X_hist, y_hist = windows(z_hist)
z = np.r_[rng.integers(0, 2, 100), rng.choice(3, 140)]
X, y = windows(z)

online = OnlineLRDSR(feature_names=["x"], novelty_alpha=1e-3, novelty_patience=6)
online.warm_start(X_hist, y_hist, n_clusters=2)
timeline = online.fit_stream(X, y, true_labels_for_eval=z)
born = timeline.loc[timeline.spawned.notna(), "t"].tolist()
first = int(np.argmax(z == 2))
ok = timeline.label >= 0
print(f"regimes at the end: {online.n_clusters}; third law first seen at window {first}, "
      f"born at {born}")
print(f"accuracy on assigned windows: {aligned_accuracy(z[ok], timeline.label[ok]):.3f}; "
      f"windows held in the novelty buffer: {(~ok).sum()}")
'''),
    ("code", r'''
fig, ax = plt.subplots(1, 2, figsize=(13, 3.0), gridspec_kw={"width_ratios": [2.2, 1]})
viz.plot_stream(timeline, ax=ax[0], title="truth (top), assignment (middle), certainty (bottom)")
viz.plot_laws(coef=online.coef_, feature_names=["x"], truth=laws, ax=ax[1])
ax[1].set_title("laws at the end of the stream")
fig.tight_layout()
'''),
    ("md", r"""
## 2 · Watch it run

The animation below **is** the run: each frame feeds the next windows to a
fresh model, the laws are redrawn from its current statistics, and the
timeline grows. The third law appears when its regime is born.
"""),
    ("code", r'''
from IPython.display import HTML
import matplotlib as mpl
mpl.rcParams["animation.embed_limit"] = 40

live = OnlineLRDSR(feature_names=["x"], novelty_alpha=1e-3, novelty_patience=6)
live.warm_start(X_hist, y_hist, n_clusters=2)
anim = viz.animate_stream(live, X, y, truth=z, every=8, interval=200, x_range=(-2, 2), dpi=60)
HTML(anim.to_jshtml())
'''),
    ("md", r"""
## 3 · How fast is "real time"?
"""),
    ("code", r'''
import time
bench = OnlineLRDSR(feature_names=["x"]).warm_start(X_hist, y_hist, n_clusters=2)
t0 = time.perf_counter()
for w in range(len(y)):
    bench.partial_fit(X[w], y[w])
dt = (time.perf_counter() - t0) / len(y)
print(f"{dt * 1e6:.0f} µs per window of {n} samples  ->  {1 / dt:,.0f} windows / s")
'''),
    ("md", r"""
## 4 · Laws that drift: forgetting

One law's slope drifts from $+1$ to $-0.5$ over the stream. With
$\lambda = 1$ the regime's statistics average the whole past; with
$\lambda < 1$ they remember $\sim 1/(1-\lambda)$ windows and track it.
"""),
    ("code", r'''
T = 300
slope = np.linspace(1.0, -0.5, T)
xs = rng.uniform(-2, 2, (T, n))
ys = xs**2 + slope[:, None] * xs + rng.normal(0, 0.5, xs.shape)
Xh = rng.uniform(-2, 2, (30, n)); yh = Xh**2 + Xh + rng.normal(0, 0.5, Xh.shape)
# The library is collinear (x, x^3, sin x, ...), so no single coefficient is
# identifiable; the LAW is. Track its odd part at x = 1, (f(1) - f(-1)) / 2,
# which for x^2 + s x is exactly the slope s.
from lrdsr.core.soft import library_terms
probe = library_terms(np.array([[1.0], [-1.0]]), feature_names=["x"])[0]
fig, ax = plt.subplots(figsize=(7, 3))
for lam, c in ((1.0, viz.PALETTE[4]), (0.97, viz.PALETTE[1]), (0.9, viz.PALETTE[0])):
    m = OnlineLRDSR(feature_names=["x"], forgetting=lam, novelty_alpha=0.0)
    m.warm_start(Xh[:, :, None], yh, labels=np.zeros(30, int))
    est = []
    for t in range(T):
        m.partial_fit(xs[t][:, None], ys[t])
        f = probe @ m.coef_[0]
        est.append((f[0] - f[1]) / 2)
    ax.plot(est, color=c, label=rf"$\lambda$ = {lam}  (memory ~{1 / (1 - lam) if lam < 1 else np.inf:.0f})")
ax.plot(slope, "k--", lw=1, label="true slope")
ax.set(xlabel="window", ylabel="$(f(1) - f(-1))/2$",
       title="tracking a drifting law")
ax.legend();
'''),
    ("md", r"""
## 5 · Sample-level switches: CUSUM, against its theory

One long stream, one sample at a time, switching between two laws every 400
samples. The per-sample KL divergence between the laws is
$\mathrm{KL} = \rho/2$, so a switch should be detected
$\approx 2h/\rho$ samples after it happens.
"""),
    ("code", r'''
from lrdsr import CusumSegmenter

seg = np.repeat(np.arange(10) % 2, 400)
x = rng.uniform(-2, 2, seg.size)
yy = np.where(seg == 0, laws[0](x), laws[1](x)) + rng.normal(0, 2.0, seg.size)
rho = np.mean((2 * x) ** 2) / 2.0 ** 2
out = CusumSegmenter(laws[:2], sigma=2.0, threshold=6.0).run(x, yy)
switches = np.flatnonzero(np.diff(seg)) + 1
delays = [int(np.argmax(out.state.values[s:] == seg[s])) for s in switches]
print(f"rho = {rho:.3f}; theory delay 2h/rho = {2 * 6 / rho:.1f} samples; "
      f"measured mean delay {np.mean(delays):.1f}; accuracy {np.mean(out.state == seg):.3f}")

fig, ax = plt.subplots(2, 1, figsize=(11, 3.6), sharex=True,
                       gridspec_kw={"height_ratios": [1, 1.4]})
ax[0].step(np.arange(seg.size), seg, color="#999999", lw=2.5, label="truth")
ax[0].step(np.arange(seg.size), out.state, color=viz.PALETTE[0], lw=1.0, label="CUSUM")
ax[0].set(yticks=[0, 1], ylabel="regime"); ax[0].legend(ncol=2, loc="upper right")
ax[1].plot(out.cusum, color=viz.PALETTE[1], lw=0.8)
ax[1].axhline(6.0, color="k", ls=":", lw=1)
ax[1].set(xlabel="sample", ylabel="CUSUM statistic")
fig.tight_layout()
'''),
]


# ==========================================================================
# 02 -- theory and losses
# ==========================================================================
THEORY = [
    ("md", r"""
# 02 · Theory, losses, and learning the loss

Three results, each a formula checked against simulation:

1. **The ceiling.** An oracle that knows both laws errs with probability
   $P_\mathrm{err} = Q\big(\sqrt{n\rho}/2\big)$ per window, with
   $\rho = \mathbb{E}[(f_1-f_0)^2]/\sigma^2$ (V1-V7, `lrdsr.theory.verification`).
2. **What a loss costs (V8).** A decision made with loss $\rho(\cdot)$ and
   influence $\psi = \rho'$ errs with
   $Q\big(\sqrt{n\rho\,\eta}/2\big)$, where
   $\eta = \mathrm{Var}(e)\,(\mathbb{E}\psi')^2 / (s^2\,\mathbb{E}\psi^2)$ is the
   loss's efficiency under the noise (`lrdsr.theory.losses`). Squared loss has
   $\eta = 1$ under *every* noise; the likelihood-ratio loss reaches the
   maximum, $\mathrm{Var}(e)\,I(p)$.
3. **How fast a switch is seen (V9).** CUSUM detects a change of law after
   $\approx h/\mathrm{KL} = 2h/\rho$ samples and false-alarms at most every
   $\approx e^h$ (`lrdsr.theory.sequential`).
"""),
    ("code", SETUP),
    ("md", r"""
## 1 · The ceiling, live

Simulate the oracle on the polynomial pair $x^2 \pm x$ (gap $2x$) at many
$n\rho$ and lay the result on $Q(\sqrt{n\rho}/2)$. The small upward bias at
large $n\rho$ is the Jensen step from the realised design to the population
form, which V2 prices exactly.
"""),
    ("code", r'''
rng = np.random.default_rng(11)
rows = []
for n in (8, 32, 128):
    for rho in np.geomspace(0.01, 1.5, 9):
        x = rng.uniform(-2, 2, (20000, n))
        sigma = np.sqrt((16 / 3) / rho)
        y = x**2 + x + rng.normal(0, sigma, x.shape)                 # truth: f0
        err = np.mean(np.sum((y - (x**2 - x))**2, 1) < np.sum((y - (x**2 + x))**2, 1))
        rows.append((n, rho, n * rho, err))
sim = pd.DataFrame(rows, columns=["n", "rho", "n_rho", "error"])
ax = None
for i, (n, g) in enumerate(sim.groupby("n")):
    ax = viz.plot_error_curve(g.n_rho, g.error, ax=ax, label=f"oracle, n = {n}",
                              theory=(i == 0), color=viz.PALETTE[i])
ax.set_ylim(1e-4, 0.6); ax.set_title("the ceiling depends on n and rho only through n*rho");
'''),
    ("md", r"""
## 2 · The loss family

Each loss is a penalty on the scaled residual; its influence $\psi$ says how
hard one sample can pull. Squared loss has unbounded influence, Huber caps it,
Cauchy / Student-t / Tukey let it fall back towards zero (*redescending*).
"""),
    ("code", r'''
fig, ax = plt.subplots(1, 2, figsize=(11, 3.3))
viz.plot_losses(ax=ax[0]); viz.plot_losses(ax=ax[1], influence=True)
fig.tight_layout()
'''),
    ("md", r"""
## 3 · Efficiency: which loss, under which noise

$\eta$ by quadrature for every loss and noise law. Read each row against its
`lrt` column, the best any loss can do.
"""),
    ("code", r'''
from lrdsr.theory.losses import LOSS_GRID, efficiency, lrt_efficiency, noise_families

fams = noise_families()
eta = pd.DataFrame({name: {**{loss: efficiency(loss, nz, **p) for loss, p in LOSS_GRID.items()},
                           "lrt": lrt_efficiency(nz)} for name, nz in fams.items()}).T
eta.round(3)
'''),
    ("md", r"""
Under Gaussian noise nothing beats squared loss, and the robust losses give up
4-5%. Under 10% contamination the squared loss is **nine times** less
efficient than the likelihood ratio: it needs nine times as many samples a
window for the same error. The table is the whole argument for choosing the
loss by the noise, and the next cell checks it against simulation.
"""),
    ("code", r'''
from lrdsr.theory.losses import simulate_error

nz = fams["contaminated_10"]
rows = []
for n_rho in (0.5, 1, 2, 4, 8):
    out = simulate_error(nz, n=64, rho=n_rho / 64, seed=11, n_windows=20000)
    for loss, (sim_err, pred, e) in out.items():
        rows.append({"loss": loss, "n_rho": n_rho, "simulated": sim_err, "predicted": pred})
df = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(6.5, 3.6))
for i, (loss, g) in enumerate(df.groupby("loss")):
    ax.plot(g.n_rho, g.simulated, "o", color=viz.PALETTE[i], ms=4)
    ax.plot(g.n_rho, g.predicted, "-", color=viz.PALETTE[i], lw=1.2, label=loss)
ax.set(xscale="log", yscale="log", xlabel=r"$n\rho$", ylabel="window error",
       title="10% contamination: dots simulated, lines $Q(\\sqrt{n\\rho\\eta}/2)$")
ax.legend(ncol=2, fontsize=7.5);
'''),
    ("md", r"""
The committed verification sweeps five noise laws, seven losses and two
window lengths (`results/losses/v8_*.csv`). The formula holds within the
Monte-Carlo band wherever the per-sample gap is small against the noise
scale, and the simulation shows exactly where it stops holding: large gaps,
the non-smooth absolute loss, and squared loss under heavy tails at small
$n$, where the statistic is not yet Gaussian.
"""),
    ("code", r'''
from analysis.figs_losses import FIGURES as LF
LF["loss_efficiency"]();
'''),
    ("code", r'''
pd.read_csv(RES / "losses" / "v8_verdict.csv").pivot_table(
    index="loss", columns="gap_bin", values="share_within").round(2)
'''),
    ("md", r"""
## 4 · The estimator under each loss, and a learned loss

The same comparison with the **estimator** instead of an oracle: every loss
in the hard loop, the soft EM with Gaussian and with Student-t noise, and
`loss="learned"`, which refits the noise model to the residuals each
iteration. Mean matched error over three pairs, two separations, three seeds.
"""),
    ("code", r'''
S = pd.read_csv(RES / "losses" / "loss_estimator_summary.csv")
S.pivot_table(index="arm", columns="noise", values="mean_error").round(3)
'''),
    ("code", r'''
LF["loss_estimator"]();
'''),
    ("md", r"""
Two honest readings of that table:

* the **learned** loss is within a fraction of a percent of the true-noise
  oracle under every noise — learning the loss recovers the Bayes rule;
* the default **Huber** loss still trails the squared loss a little under
  Gaussian noise. It used to trail it by ~30%: residuals were divided by the
  raw MAD (0.67 sigma), so `delta = 1.5` was really ~1 sigma, where Huber is
  only ~90% efficient. The scale is now in sigma units and `delta = 1.345`,
  chosen on the tuning seeds -- the sweep is below. And a soft EM that
  *assumes* Gaussian noise collapses under heavy tails, where its Student-t
  twin sits on the oracle.
"""),
    ("code", r'''
# the sweep that chose the default Huber constant (tuning seeds {3, 7, 19});
# delta = 1.0 is about where the old default sat, 100 is the squared loss
H = pd.read_csv(RES / "losses" / "loss_huber_delta_summary.csv")
H.pivot_table(index="noise", columns="delta", values="matched_error").round(4)
'''),
    ("md", r"""
Learning $\nu$: the Student-t tail parameter each learner converges to.
"""),
    ("code", r'''
LF["loss_learning"]();
'''),
    ("code", r'''
from lrdsr.core.losses import learn_loss
rng = np.random.default_rng(3)
samples = {"gaussian": rng.normal(size=5000), "laplace": rng.laplace(size=5000),
           "student_t3": rng.standard_t(3, 5000),
           "contaminated_10": np.where(rng.random(5000) < 0.1, 10, 1) * rng.normal(size=5000)}
pd.DataFrame([{"noise": k, "learned loss": (fit := learn_loss(v)).loss,
               "nu": fit.params.get("nu"), "scale": fit.scale} for k, v in samples.items()])
'''),
    ("md", r"""
## 5 · Sequential detection: the real-time ceiling

For a stream that switches law, the per-sample Kullback-Leibler divergence is
$\rho/2$. CUSUM with threshold $h$ detects the switch after
$\approx 2h/\rho$ samples (first order); Siegmund's correction for the
overshoot makes it $\approx (h' + e^{-h'} - 1)/\mathrm{KL}$ with $h'$ the
threshold plus the expected overshoot.
"""),
    ("code", r'''
from lrdsr.theory.sequential import predicted_arl, predicted_delay
tab = pd.DataFrame([{"rho": r, "h": h,
                     "delay 2h/rho": predicted_delay(h, r, method="first_order"),
                     "delay (Siegmund)": predicted_delay(h, r, kappa=0.8),
                     "ARL >= e^h": predicted_arl(h, r, method="first_order")}
                    for r in (0.25, 1.0) for h in (3, 6, 9)])
tab.round({"delay 2h/rho": 1, "delay (Siegmund)": 1, "ARL >= e^h": 0})
'''),
    ("code", r'''
from analysis.figs_online import FIGURES as OF
OF["online_sequential"]();
'''),
]


# ==========================================================================
# 03 -- the problem zoo
# ==========================================================================
ZOO = [
    ("md", r"""
# 03 · The problem zoo: twelve problems, each hard for its own reason

`experiments/problems/zoo.py` is a registry of regime-recovery problems, each
built to break **one** assumption: the law is not in the symbolic library;
the *gap* between laws is outside the library's span; there are 3 or 4
regimes; the law has two inputs; one law is a rescaled copy of the other;
the response has the same mean and variance under both laws. In every
problem all regimes share **one** input distribution, so only the equation
can tell them apart, and the noise is set so the closest pair sits at a
chosen separation $\rho$.

This notebook reads the committed results (`results/problems/`), re-runs
two problems live, and shows how to add your own.
"""),
    ("code", SETUP),
    ("code", r'''
cat = pd.read_csv(RES / "problems" / "problems_catalog.csv")
cat[["problem", "K", "d", "truths", "in_library", "gap_outside_library", "why"]]
'''),
    ("code", r'''
from analysis.figs_problems import FIGURES as PF
PF["problem_atlas"]();
'''),
    ("md", r"""
## The results, against the oracle

The oracle knows the true laws and picks the one with the smaller residual:
the Bayes ceiling. `gap` columns are excess error over it, averaged over the
reporting seeds {11, 23, 42}.
"""),
    ("code", r'''
S = pd.read_csv(RES / "problems" / "problems_summary.csv")
cols = ["oracle", "lrdsr", "soft_em", "mechanism_kmeans", "profile_kmeans", "geometry"]
S.pivot_table(index="problem", columns="rho", values=["lrdsr_gap", "soft_em_gap"]).round(3)
'''),
    ("code", r'''
PF["problem_error_vs_rho"]();
'''),
    ("code", r'''
PF["problem_gap_heatmap"]();
'''),
    ("md", r"""
## Why `high_frequency` fails and `frequency_shift` does not

Both pairs are "out of the library" — no `sin(1.3x)` or `sin(4.6x)` term
exists. What matters for **assignment** is whether the library can represent
the *gap* between the laws. `gap_outside_library` measures that share: tiny
for a 30% frequency shift, a third of the gap for $\sin 4x$ vs $\sin 4.6x$.
Run both live at $\rho = 0.25$:
"""),
    ("code", r'''
from experiments.problems.zoo import (PROBLEM_BY_NAME, gap_outside_library, make_windows,
                                      oracle_labels, sigma_for_rho)
from lrdsr import SoftLRDSR
from lrdsr.core.evaluation import aligned_accuracy
from experiments.common.fitting import fit_lrdsr, window_features

fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
for a, name in zip(ax, ("frequency_shift", "high_frequency")):
    P = PROBLEM_BY_NAME[name]
    X, y, z = make_windows(P, sigma_for_rho(P, 0.25), 150, 48, seed=11)
    Zf, fn = window_features(X, y)
    hard = fit_lrdsr(Zf, fn, X, y, P.feature_names, P.K, 11, alpha_geom=0.0)
    soft = SoftLRDSR(P.K, feature_names=P.feature_names, random_state=11).fit(X, y)
    def err(lab, z=z):
        return 1 - aligned_accuracy(z, lab)
    viz.plot_laws(hard.models, x_range=P.x_range, truth=[lambda x, f=f: f(x[:, None]) for f in P.laws], ax=a)
    a.set_title(f"{name}: gap outside span {gap_outside_library(P):.1%}\n"
                f"oracle {err(oracle_labels(P, X, y)):.3f} | hard {err(hard.labels):.3f} | "
                f"soft {err(soft.labels):.3f}", fontsize=9)
fig.tight_layout()
'''),
    ("md", r"""
## Two inputs: the regimes differ only in how they combine

`interaction` is $x_1 x_2$ against $x_1 + x_2$ — same inputs, same ranges,
and the laws are recovered in their own symbols.
"""),
    ("code", r'''
P = PROBLEM_BY_NAME["interaction"]
X, y, z = make_windows(P, sigma_for_rho(P, 0.25), 150, 48, seed=11)
Zf, fn = window_features(X, y)
hard = fit_lrdsr(Zf, fn, X, y, P.feature_names, P.K, 11, alpha_geom=0.0)
print("error:", round(1 - aligned_accuracy(z, hard.labels), 3))
for m in hard.models:
    print("  ", m.expression())
g = np.linspace(-2, 2, 60)
G1, G2 = np.meshgrid(g, g)
grid = np.column_stack([G1.ravel(), G2.ravel()])
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for k, m in enumerate(hard.models):
    im = ax[k].contourf(G1, G2, m.predict(grid).reshape(G1.shape), 20, cmap="RdBu_r")
    ax[k].set(title=f"recovered law {k}", xlabel="$x_1$", ylabel="$x_2$")
    fig.colorbar(im, ax=ax[k])
fig.tight_layout()
'''),
    ("md", r"""
## Add your own problem

A `Problem` is its laws, their names, a shared input sampler, and one line
saying why it is hard. `run_cell` runs every method on it.
"""),
    ("code", r'''
from experiments.problems.zoo import Problem, _u
from experiments.problems.run import run_cell

mine = Problem(
    "my_pair",
    (lambda X: np.abs(X[..., 0]), lambda X: 0.5 * X[..., 0] ** 2 + 0.3),
    ("|x|", "0.5*x^2 + 0.3"), _u(-2.0, 2.0), 1, (-2.0, 2.0), False,
    "a V against a U: equal at two points, |x| is not a library term")
rows, laws = run_cell(mine, seed=11, rho_grid=(0.1, 0.5), n_windows=120, window_len=32)
pd.DataFrame(rows).pivot_table(index="method", columns="rho", values="matched_error").round(3)
'''),
]


# ==========================================================================
# 05 -- real data
# ==========================================================================
REALDATA = [
    ("md", r"""
# 05 · Real data: bike sharing and highway traffic, day by day

Two public UCI datasets, one **window per day**:

* **Capital Bikeshare**, hourly rentals, Washington DC, 2011-12;
* **I-94 Metro Interstate**, hourly westbound traffic, Minneapolis, 2012-18.

The input is the hour ($x = 2\pi h/24$); the response is $\log(1+\text{count})$
minus that day's mean, so the level of the season is gone and a regime can
only be the **shape** of the daily law. The reference labels are the
calendar day type — a *proxy*, used for scoring only. The first run
downloads the two archives (~0.7 MB) into `.cache/` and checks their sha256.
"""),
    ("code", SETUP),
    ("code", r'''
from experiments.realdata.windows import load
from experiments.realdata.run import fourier

data = {name: load(name) for name in ("bike", "traffic")}
for name, d in data.items():
    print(f"{name:8s} {d.X_seq.shape[0]:5d} days x {d.X_seq.shape[1]} hours; "
          f"working-day share {d.reference.mean():.2f}; coverage {d.coverage}")
'''),
    ("md", r"""
## 1 · What the days look like

Every day, drawn as a thin line and coloured by the calendar. The two daily
laws are visible by eye: a commuting double peak and a single midday hump.
"""),
    ("code", r'''
fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
hours = np.arange(24)
for a, (name, d) in zip(ax, data.items()):
    for k, lab in ((1, "working day"), (0, "weekend / holiday")):
        sel = np.flatnonzero(d.reference == k)[:150]
        a.plot(hours, d.y_seq[sel].T, color=viz.regime_color(1 - k), lw=0.4, alpha=0.25)
        a.plot(hours, d.y_seq[d.reference == k].mean(0), color=viz.regime_color(1 - k),
               lw=2.5, label=lab)
    a.set(title=name, xlabel="hour", ylabel="log(1+count) - day mean", xticks=range(0, 25, 3))
    a.legend(loc="lower center")
fig.tight_layout()
'''),
    ("md", r"""
## 2 · Fit, live: hard loop and soft EM

`K = 2`, no labels. The soft EM uses a Fourier basis of four harmonics — the
basis a practitioner would reach for with a daily cycle — and the hard loop
the fast symbolic library.
"""),
    ("code", r'''
from lrdsr import GroupedDCSR, SoftLRDSR
from lrdsr.core.evaluation import clustering_metrics
from experiments.common.fitting import window_features

fits = {}
for name, d in data.items():
    Zf, _ = window_features(d.X_seq, d.y_seq)
    hard = GroupedDCSR(n_clusters=2, alpha_geom=0.0, init="mechanism", residual_scale="global",
                       random_state=11).fit(d.X_seq, d.y_seq, Zf, feature_names=["x"])
    soft = SoftLRDSR(2, basis=fourier, noise="student_t", random_state=11).fit(d.X_seq, d.y_seq)
    fits[name] = (hard, soft)
    print(f"{name:8s} hard ARI {clustering_metrics(d.reference, hard.labels)['ARI']:.3f} | "
          f"soft ARI {clustering_metrics(d.reference, soft.labels)['ARI']:.3f} | "
          f"learned tail nu per regime {soft.nu.round(1)}")
    for m in hard.models:
        print("     law:", m.expression())
'''),
    ("code", r'''
fig, ax = plt.subplots(1, 4, figsize=(15, 3.2))
xg = np.linspace(0, 2 * np.pi, 200)
for i, (name, d) in enumerate(data.items()):
    hard, soft = fits[name]
    viz.plot_laws(coef=soft.coef, basis=fourier, x_range=(0, 2 * np.pi), ax=ax[2 * i],
                  labels=[f"regime {k} (soft EM)" for k in range(2)])
    ax[2 * i].set(title=f"{name}: the two daily laws", xlabel="x = 2 pi hour / 24")
    viz.plot_responsibilities(soft.responsibilities, ax=ax[2 * i + 1])
    ax[2 * i + 1].set_title(f"{name}: posteriors, {np.mean(soft.entropy > 0.1):.1%} uncertain")
fig.tight_layout()
'''),
    ("md", r"""
## 3 · The committed comparison, every method

Mean over seeds {11, 23, 42}, from `results/realdata/realdata_summary.csv`.
"""),
    ("code", r'''
S = pd.read_csv(RES / "realdata" / "realdata_summary.csv")
S.pivot_table(index=["family", "method"], columns="dataset", values="ARI_mean").round(3)
'''),
    ("code", r'''
from analysis.figs_realdata import FIGURES as RF
RF["realdata_methods"]();
'''),
    ("md", r"""
**A K-means on the raw 24-hour profile ties the best method on both
datasets.** That is the expected result, not a surprise: every day has the
same 24-hour design, so mechanism space is only a linear change of
coordinates of the profile. The method's advantage on the simulator — windows
whose *inputs differ* — does not exist here. What it adds on this data is the
laws, the posteriors, the label-free separation, and the real-time version.

## 4 · Where the law and the calendar disagree

The separation between the two laws is large, so the implied ceiling is
essentially zero error. The disagreements are therefore worth reading one by
one — they are almost all days the **calendar** gets wrong.
"""),
    ("code", r'''
pd.read_csv(RES / "realdata" / "realdata_rho.csv")[
    ["dataset", "basis", "sigma", "rho_trace", "ceiling_trace", "best_error"]]
'''),
    ("code", r'''
pd.read_csv(RES / "realdata" / "realdata_disagreements.csv")
'''),
    ("code", r'''
RF["realdata_laws"]();
'''),
    ("md", r"""
## 5 · Choosing K, honestly

BIC over `K = 1…8` keeps improving to the largest `K`: the likelihood treats
24 autocorrelated hours as independent, and real days vary in many ways. The
table shows it, and the contingency tables show what the first extra
cluster is.
"""),
    ("code", r'''
K = pd.read_csv(RES / "realdata" / "realdata_k_selection.csv")
K.pivot_table(index="K", columns=["dataset", "basis"], values="ARI_vs_daytype").round(3)
'''),
    ("code", r'''
C = pd.read_csv(RES / "realdata" / "realdata_contingency.csv")
C.head(20)
'''),
    ("md", r"""
## 6 · In real time, over the calendar

Warm-started on the first 60 days without labels, then fed one day at a time
in date order: each day is assigned the moment it ends, and the laws are
updated with it.
"""),
    ("code", r'''
from lrdsr import OnlineLRDSR
from lrdsr.core.evaluation import aligned_accuracy

d = data["bike"]
on = OnlineLRDSR(basis=fourier, novelty_alpha=1e-6).warm_start(d.X_seq[:60], d.y_seq[:60],
                                                                n_clusters=2)
tl = on.fit_stream(d.X_seq[60:], d.y_seq[60:], true_labels_for_eval=d.reference[60:])
ok = tl.label >= 0
print(f"accuracy against the calendar: {aligned_accuracy(tl.truth[ok], tl.label[ok]):.3f} "
      f"on {ok.sum()} days; {(~ok).sum()} held as novel")
fig, ax = plt.subplots(figsize=(13, 2.6))
viz.plot_stream(tl, ax=ax, title="bike, day by day: calendar (top) vs online assignment")
'''),
    ("code", r'''
RF["realdata_online"]();
'''),
    ("code", r'''
pd.read_csv(RES / "realdata" / "realdata_online_summary.csv")
'''),
]


# ==========================================================================
# 06 -- beyond the library: kernels, classification, windows with gaps
# ==========================================================================
EXTENSIONS = [
    ("md", r"""
# 06 · Kernels, classification, and windows with gaps

The method's claim is about **laws**, not about any one term library, any one
task or any one sampling grid. This notebook takes each of those away:

1. **A kernel basis instead of the library** -- and the zoo's one failure,
   `sin 4x` against `sin 4.6x`, comes back.
2. **Classification instead of clustering** -- a class is a *set* of laws,
   and the plug-in learning curve is a formula (V10).
3. **Real data where windows have different designs** -- UCR series sampled
   irregularly, and I-94 traffic days with sensor gaps, scored at the hours
   that were observed.

Every live cell runs in seconds; the full results are read from `results/`.
"""),
    ("code", SETUP),
    ("md", r"""
## 1 · A kernel basis repairs the gap outside the span

`high_frequency` from the problem zoo: two laws the fast library cannot tell
apart well, because 37% of their *difference* is outside its span. A Nystrom
basis of an RBF kernel spans a function space instead; its rank is the one
knob.
"""),
    ("code", r'''
from experiments.problems.zoo import PROBLEMS, make_windows, sigma_for_rho, oracle_labels
from experiments.kernel.run import gap_outside_basis, _snr
from lrdsr.core.kernel import NystromBasis
from lrdsr.core.mechanism_space import mechanism_features, mechanism_init, mechanism_noise
from lrdsr.core.evaluation import aligned_accuracy
from sklearn.cluster import KMeans

prob = next(p for p in PROBLEMS if p.name == "high_frequency")
X, y, z = make_windows(prob, sigma_for_rho(prob, 0.25), 150, 48, seed=11)
def err(lab):
    return 1 - aligned_accuracy(z, lab)


print(f"oracle            {err(oracle_labels(prob, X, y)):.3f}")
print(f"library           {err(mechanism_init(X, y, 2, seed=11, feature_names=['x'])):.3f}")
for r in (4, 8, 12, 16, 24):
    b = NystromBasis(r).fit(X.reshape(-1, 1))
    S = mechanism_features(X, y, basis=b)
    lab = KMeans(2, n_init=30, random_state=11).fit_predict(S)
    print(f"Nystrom rank {b.rank:2d}  {err(lab):.3f}   gap outside {gap_outside_basis(prob, b):6.1%}"
          f"   SNR {_snr(S, mechanism_noise(X, y, basis=b)):.2f}")
'''),
    ("md", r"""
The rank is chosen **without labels**: the rank that maximises the spectral
SNR of mechanism space (excess spread over the noise, per root dimension).
That rule was picked over per-window leave-one-out on the *tuning* seeds; the
sweep over every problem is in `results/kernel/`.
"""),
    ("code", r'''
from analysis.figs_kernel import kernel_rank_sweep, kernel_zoo
kernel_rank_sweep(); kernel_zoo()
pd.read_csv(RES / "kernel" / "kernel_zoo_summary.csv").set_index("problem").round(3)
'''),
    ("md", r"""
## 2 · Classification: the learning curve is a formula (V10)

With labels, fit the laws per class and send a window to the class whose law
explains it. With a basis of rank $p$ and $m$ labelled windows per class,
the plug-in error is

$$P_{err} \approx Q\left(\frac{D^2}{2\sqrt{D^2 + 2p/m}}\right),\qquad D^2 = n\rho,$$

and exactly a three-scalar expectation. **The price is $p/m$**: the basis
dimension, not the window length.
"""),
    ("code", r'''
from lrdsr.theory.classification import plugin_error, plugin_error_first_order, _simulate, Q
for p, m in ((3, 1), (15, 2), (31, 8)):
    sims = [_simulate("fixed", 9.0, p, m, s, n_test=500)[0] for s in range(30)]
    print(f"p={p:2d} m={m:2d}  simulated {np.mean(sims):.3f}  exact {plugin_error(9.0, p, m):.3f}"
          f"  first order {plugin_error_first_order(9.0, p, m):.3f}  ceiling {Q(1.5):.3f}")
from analysis.figs_classify import v10_learning_curve
v10_learning_curve();
'''),
    ("md", r"""
## 3 · A class is a set of laws

`LawClassifier` fits `L` laws per class with the hard LR-DSR loop inside
each class. `L = 1` is the linear rule; `L = "all"` makes every training
series its own law (nearest law). Chinatown: pedestrian counts, weekday or
weekend, 20 training days.
"""),
    ("code", r'''
from experiments.classify import ucr
from lrdsr.core.classify import LawClassifier
from lrdsr.core.kernel import CosineBasis

ds = ucr.load("Chinatown")
Xtr, ytr = ucr.to_windows(ds.train); Xte, yte = ucr.to_windows(ds.test)
for L in (1, 2, 4, "all"):
    clf = LawClassifier(basis=CosineBasis(8), nuisance=None, laws_per_class=L).fit(Xtr, ytr, ds.y_train)
    print(f"L = {L!s:4} laws {clf.n_laws:3d}  test error {np.mean(clf.predict(Xte, yte) != ds.y_test):.3f}")

clf = LawClassifier(basis=CosineBasis(8), nuisance=None, laws_per_class=2).fit(Xtr, ytr, ds.y_train)
grid = np.linspace(0, 1, 200)[:, None]
fig, ax = plt.subplots(figsize=(7, 3))
for j, c in enumerate(clf.law_class_):
    ax.plot(grid[:, 0] * 23, clf.design_.B(grid) @ clf.coef_[j], color=viz.regime_color(c),
            label=f"{('weekend', 'weekday')[c]}, law {j}")
ax.set(xlabel="hour", ylabel="count", title="Chinatown: two laws per class"); ax.legend(fontsize=8);
'''),
    ("md", r"""
**Irregular sampling.** Keep a random quarter of each series' hours -- a
different quarter per series. The law classifier scores the kept samples at
their own times; a profile method has to interpolate first.
"""),
    ("code", r'''
from sklearn.neighbors import KNeighborsClassifier
for keep in (1.0, 0.5, 0.25):
    tr = ucr.subsample(ds.train, keep, 11) if keep < 1 else ds.train
    te = ucr.subsample(ds.test, keep, 12) if keep < 1 else ds.test
    Xa, ya = ucr.to_windows(tr); Xb, yb = ucr.to_windows(te)
    law = LawClassifier(basis=CosineBasis(8), nuisance=None, laws_per_class=2).fit(Xa, ya, ds.y_train)
    nn = KNeighborsClassifier(1).fit(ucr.to_grid(tr, 24), ds.y_train)
    print(f"keep {keep:4.0%}  law {np.mean(law.predict(Xb, yb) != ds.y_test):.3f}   "
          f"interpolated 1-NN {np.mean(nn.predict(ucr.to_grid(te, 24)) != ds.y_test):.3f}")
'''),
    ("md", r"""
### The 24-dataset benchmark

Nine daily-cycle datasets (the kind of data this method is for) and fifteen
shape benchmarks (the contrast). Hyperparameters by CV on the training split,
test split touched once.
"""),
    ("code", r'''
b = pd.read_csv(RES / "classify" / "ucr_benchmark.csv")
b.pivot_table(index=["group", "dataset"], columns="method", values="error").round(3)
'''),
    ("code", r'''
from analysis.figs_classify import classify_benchmark, classify_fewshot, classify_irregular
classify_benchmark(); classify_fewshot(); classify_irregular();
'''),
    ("md", r"""
## 4 · I-94 days with sensor gaps

The first pass at the real data dropped every traffic day with a missing hour
-- a third of the days. Score each partial day at the hours it has, with its
level as a nuisance, and compare with imputing the gaps first.
"""),
    ("code", r'''
from experiments.realdata import gaps
days = gaps.traffic_days()
part = days[~days.complete]
print(f"{days.complete.sum()} complete days, {len(part)} partial days with >= {gaps.MIN_HOURS} hours")
c, lab = gaps._cluster_complete(days, seed=11)
Xc, yc = gaps.windows(c["hours"], c["count"])
law = LawClassifier(basis=gaps._fourier(), nuisance="intercept").fit(Xc, yc, lab)

few = part[part.n_hours <= 9].head(3)
fig, axes = plt.subplots(1, len(few), figsize=(11, 2.8), sharey=True)
hh = np.arange(24); xg = (2 * np.pi * hh / 24)[:, None]
for ax, (_, d) in zip(axes, few.iterrows()):
    Xd, yd = gaps.windows([d.hours], [d["count"]])
    p = law.predict_proba(Xd, yd)[0]
    for k in range(2):
        f = law.design_.B(xg) @ law.coef_[k]
        off = np.mean(yd[0] - (law.design_.B(Xd[0]) @ law.coef_[k]))
        ax.plot(hh, f + off, color=viz.regime_color(k), label=f"law {k}: p = {p[k]:.2f}")
    ax.plot(d.hours, yd[0], "ko", ms=4)
    ax.set(title=f"{d.date:%Y-%m-%d} ({'working' if d.reference else 'off'}), {d.n_hours} h",
           xlabel="hour")
    ax.legend(fontsize=7)
axes[0].set_ylabel("log(1 + count)"); fig.tight_layout()
'''),
    ("code", r'''
from analysis.figs_realdata import realdata_gaps
realdata_gaps()
pd.read_csv(RES / "realdata" / "realdata_gaps_summary.csv").pivot_table(
    index=["arm", "hours_bin"], columns="route", values="accuracy").round(3)
'''),
]


NOTEBOOKS: dict[str, list[tuple[str, str]]] = {
    "01_quickstart": QUICKSTART,
    "02_theory_and_losses": THEORY,
    "03_problem_zoo": ZOO,
    "04_realtime_clustering": REALTIME,
    "05_real_data": REALDATA,
    "06_kernels_and_classification": EXTENSIONS,
}
