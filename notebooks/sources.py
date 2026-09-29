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


# ==========================================================================
# 07 -- partial days: V11, early decision, streams with gaps
# ==========================================================================
PARTIAL = [
    ("md", r"""
# 07 · A partial day: its error before it is scored, and the hour it is decided

A day of I-94 traffic observed at hours $H$ is classified by least squares
with its level profiled out. With the gap $\tilde g_H$ between the two
day-laws at those hours (level removed) and the residual covariance
$\Sigma$ of complete days, V11 gives its error exactly:

$$\mathrm{err}(H) = Q\left(\frac{|\tilde g_H|^2/2 \pm s^2 \log(\pi_1/\pi_0)}
{\sqrt{\tilde g_H^\top \Sigma_H \tilde g_H}}\right).$$

The only estimated input is $\Sigma$, and it carries the correlation of
neighbouring hours that V1's independent noise leaves out.
"""),
    ("code", SETUP),
    ("code", r'''
from experiments.realdata import partial
from lrdsr.theory.partial import masked_error

Y = partial.complete_days("traffic")
clf, F, Sigma, lab = partial.fit_half(Y[: len(Y) // 2], seed=11)
g = F[:, 1] - F[:, 0]
s2 = float(clf.sigma_[0] ** 2); L = float(clf.log_prior_[1] - clf.log_prior_[0])
print(f"{len(Y)} complete days; laws and Sigma from the first half")
errs = {}
for name, H in {"all 24 hours": np.arange(24), "no night (06-23)": np.arange(6, 24),
                "no morning (12-23)": np.arange(12, 24), "no evening (00-11)": np.arange(12),
                "six scattered hours": np.array([1, 5, 9, 13, 17, 21])}.items():
    e = [masked_error(g[H], Sigma[k][np.ix_(H, H)], s2, L, truth=k) for k in (0, 1)]
    errs[name] = np.mean(e)
    print(f"{name:22s} predicted error: working-day law {e[1]:.4f}, day-off law {e[0]:.4f}")
fig, ax = plt.subplots(figsize=(7, 2.4))
ax.barh(list(errs), list(errs.values()), color=viz.regime_color(0))
ax.set(xscale="log", xlabel="V11 predicted error (mean over the two laws)"); fig.tight_layout()
'''),
    ("md", r"""
The same number of hours can cost nothing or a great deal. The experiment
removes a 6- or 12-hour block at every start hour from every held-out day and
compares the prediction with what happens:
"""),
    ("code", r'''
from analysis.figs_partial import v11_which_hours, v11_early_decision, online_kernel_birth
v11_which_hours(); v11_early_decision()
pd.read_csv(RES / "realdata" / "realdata_partial_ranking.csv")
'''),
    ("md", r"""
## Streams with gaps, and a law the library cannot write

`OnlineLRDSR` takes a level nuisance, windows of any length and a kernel basis.
Every I-94 day in calendar order, the partial ones included:
"""),
    ("code", r'''
pd.read_csv(RES / "online" / "online_ragged_traffic_summary.csv")
'''),
    ("md", r"""
And a newcomer the term library cannot represent (`sin 4.6x` after `sin 4x`),
live for one stream, then over every seed:
"""),
    ("code", r'''
from experiments.online.ragged import birth_cell
for basis in ("library", "kernel"):
    print(basis, birth_cell(1.0, 11, basis, control=False))
online_kernel_birth();
'''),
]


# ==========================================================================
# 08 -- wind power curves
# ==========================================================================
WIND = [
    ("md", r"""
# 08 · Wind power curves: every window its own design

Kelmarsh wind farm, six turbines, 10-minute SCADA (CC-BY-4.0; the first run
downloads ~270 MB into `.cache/wind/` and checks its sha256). One window is a
turbine over six hours: $x$ the wind speed it happened to see, $y$ the power
as a share of rated. No two windows share a design, so there is no raw
profile -- a profile method has to bin each window's curve first.

The archive has **no operator labels** (checked: curtailment columns empty,
setpoint never caps output), so the blocks are scored against physical
proxies: colder air, wake from a neighbour, and night vs day as a negative
control.
"""),
    ("code", SETUP),
    ("code", r'''
from experiments.wind.windows import build
b = build()
m = b.meta
print(f"{len(b.X)} blocks; median wind-speed bins covered: {m.coverage.median():.0f} of 24")
for p in ("cold", "waked", "night"):
    print(f"{p:6s}", m[p].value_counts().to_dict())
'''),
    ("code", r'''
fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
rng = np.random.default_rng(0)
for ax, (proxy, names) in zip(axes, (("cold", ("warm", "cold")), ("waked", ("free", "waked")),
                                     ("night", ("day", "night")))):
    for k in (0, 1):
        idx = rng.choice(np.flatnonzero(m[proxy].to_numpy() == k), 40, replace=False)
        for i in idx:
            ax.plot(b.X[i][:, 0], b.y[i], ".", ms=2, color=viz.regime_color(k), alpha=0.5)
        ax.plot([], [], "o", color=viz.regime_color(k), label=names[k])
    ax.set(title=proxy, xlabel="wind speed (m/s)", xlim=(2, 14)); ax.legend()
axes[0].set_ylabel("power / rated"); fig.tight_layout()
'''),
    ("md", r"""
## Transfer across turbines

Trained on turbines 1-3, tested on 4-6 and back; balanced error. The law
classifier against the method of bins, and V1's ceiling for a clean
two-law proxy:
"""),
    ("code", r'''
from analysis.figs_wind import wind_transfer, wind_physics
wind_transfer()
t = pd.read_csv(RES / "wind" / "wind_transfer.csv")
t.pivot_table(index="proxy", columns="method", values="balanced_error").round(3)
'''),
    ("code", r'''
pd.read_csv(RES / "wind" / "wind_coverage_terciles.csv").pivot_table(
    index=["proxy", "coverage"], columns="method", values="balanced_error").round(3)
'''),
    ("md", r"""
## The physics check

Below rated, power scales with air density. The cold/warm ratio of the
recovered curves against the density ratio the temperatures imply:
"""),
    ("code", r'''
wind_physics()
pd.read_csv(RES / "wind" / "wind_physics.csv")[["ws", "bins_ratio", "law_ratio",
                                                "law_uniform_ratio", "density_ratio"]].round(3)
'''),
    ("md", r"""
## In real time

Each turbine streamed on its own. The regimes born, and when:
"""),
    ("code", r'''
on = pd.read_csv(RES / "wind" / "wind_online.csv")
on[on.regime >= 0].merge(m.assign(born_at=m.block.astype(str))[
    ["turbine", "born_at", "ws_mean", "temp"]], on=["turbine", "born_at"], how="left").round(1)
'''),
]


# ==========================================================================
# 00 -- the whole project, end to end
# ==========================================================================
SUMMARY = [
    ("md", r"""
# 00 · The whole project in one notebook

**The question.** You have many short pieces of data -- *windows*: one day of
traffic, six hours of a wind turbine, one heartbeat. Each window was produced
by one of a few hidden **rules** (closed-form laws, e.g. $y = x^2 + x$). Nobody
tells you which rule produced which window. Can you recover **both** the rules
**and** which window came from which rule?

This notebook is the whole project from the beginning, written for a reader
who is not a specialist:

| part | what you get |
|---|---|
| 1 · Motivation | why "cluster by the law" and not by how the data look |
| 2 · The method | the model, the **objective function**, the **optimisation**, the pipeline |
| 3 · A worked example | the method run live, step by step, on simulated data |
| 4 · What works | the main results, as plots |
| 5 · Real data, live | bike-sharing days on a calendar, coloured by the law found |
| 6 · What did not work | two failures, shown, and the list of the others |
| 7 · Every task in one picture | the fifteen missions + the symbolic-regression baseline |

**How to run.** Top to bottom, about two minutes. Parts 3, 5 and 6 compute
live; the other plots read the committed results in `results/` (produced by
`python -m experiments all`), so nothing here needs a long run. Notebooks
01-08 go deeper into each part.
"""),
    ("code", SETUP),
    ("md", r"""
---
# 1 · Motivation: similar-looking is not the same mechanism

The usual recipe is *cluster the windows, then describe each cluster*: compute
a few statistics of each window (mean, spread, trend, ...), group windows that
are close, then fit a model per group. That recipe groups windows that **look**
alike. But two windows can look very different and still come from the same
rule -- because they saw different inputs -- and two windows can look alike
and come from different rules.

Below: four windows. Which ones belong together?
"""),
    ("code", r'''
rng = np.random.default_rng(3)
def law_A(x): return x**2 - 1          # a parabola
def law_B(x): return 1.2 * x - 0.4     # a straight line
wins = [("A", law_A, (-2.0, -0.6)), ("A", law_A, (0.6, 2.0)),
        ("B", law_B, (0.3, 1.9)),   ("B", law_B, (-1.9, -0.5))]
fig, ax = plt.subplots(1, 3, figsize=(14, 3.6), gridspec_kw={"width_ratios": [1.5, 1.5, 1]})
xg = np.linspace(-2.1, 2.1, 200)
stats = []
for i, (tag, f, (lo, hi)) in enumerate(wins):
    x = np.sort(rng.uniform(lo, hi, 14)); y = f(x) + rng.normal(0, 0.18, x.size)
    ax[0].plot(x, y, "o", ms=5, color="#555555")
    ty = y.min() - 0.55 if i == 2 else y.max() + 0.35
    ax[0].annotate(f"window {i + 1}", (x.mean(), ty), ha="center", fontsize=9)
    c = viz.regime_color(0 if tag == "A" else 1)
    ax[1].plot(x, y, "o", ms=5, color=c)
    stats.append((np.polyfit(x, y, 1)[0], y.mean(), tag, i + 1))
ax[0].set(title="what you get: four windows, no labels", xlabel="input x", ylabel="output y", ylim=(-3.2, 4))
ax[1].plot(xg, law_A(xg), "--", color=viz.regime_color(0), lw=1.3, label="rule A: $y = x^2 - 1$")
ax[1].plot(xg, law_B(xg), "--", color=viz.regime_color(1), lw=1.3, label="rule B: $y = 1.2x - 0.4$")
ax[1].set(title="the truth: 1 and 2 share a rule, 3 and 4 share the other", xlabel="input x", ylim=(-3.2, 4))
ax[1].legend(loc="upper center")
for s, m, tag, i in stats:
    ax[2].scatter(s, m, s=70, color=viz.regime_color(0 if tag == "A" else 1))
    ax[2].annotate(f" {i}", (s, m), fontsize=10)
ax[2].set(title="what a 'look-alike' clustering sees", xlabel="trend (slope)", ylabel="mean level")
fig.tight_layout()
'''),
    ("md", r"""
In the right panel (the statistics a look-alike clustering would use),
window 2 sits next to window 3 -- a different rule -- while window 1 is as far
from its true partner 2 as anything on the plot. The only thing the two windows of rule A have in common is **the
equation that produced them**. So this project makes the equation the cluster
identity:

> two windows are similar when **the same law** produced them, however
> different they look.

Where this matters in practice: the same machine seen at different operating
points (a wind turbine at low vs. high wind), days observed at different hours
(a sensor that dropped out), any setting where the *inputs* of a window vary.
The output is also more useful than a cluster number: it is a **readable
equation per group**.
"""),
    ("md", r"""
---
# 2 · The method (LR-DSR: *latent-regime symbolic regression*)

## 2.1 The model

We observe $W$ windows. Window $w$ has $n$ input/output pairs
$(x_{w1}, y_{w1}), \dots, (x_{wn}, y_{wn})$. The model is

$$
y_{wi} \;=\; f_{z_w}(x_{wi}) + \varepsilon_{wi}, \qquad z_w \in \{1,\dots,K\},
\qquad \varepsilon_{wi} \sim \text{noise of scale } \sigma .
$$

* $f_1, \dots, f_K$ -- the $K$ unknown laws (the *mechanisms*);
* $z_w$ -- the hidden label: which law produced window $w$ (never observed);
* each law is a short sum of terms from a **library** of simple functions,
  $f_k(x) = \beta_{k0} + \sum_{j \in S_k} \beta_{kj}\,\phi_j(x)$,
  with $\phi_j \in \{x, x^2, x^3, \log(1+|x|), \sin x, \cos x, \sin 2x, x_a x_b\}$
  and at most 5 terms. That is what makes the result a readable formula.

## 2.2 The objective function

Recover the labels and the laws together by minimising one cost:

$$
\min_{z,\; f_1..f_K}\;\; \sum_{w=1}^{W} J_w(z_w),
\qquad
J_w(k) \;=\; \underbrace{\frac{1}{n}\sum_{i=1}^{n}\rho\!\left(\frac{y_{wi} - f_k(x_{wi})}{\hat s}\right)}_{\text{how badly law } k \text{ explains window } w}
\;+\; \beta\,C(f_k) \;+\; \alpha\,D_\text{geom}(w,k)
$$

* $\rho$ -- the per-sample **loss**. Default: **Huber** (quadratic for small
  residuals, linear for large ones, so one outlier cannot dominate). Squared,
  Cauchy, Tukey, Student-t, or a **learned** loss are options.
* $\hat s$ -- **one shared noise scale** (robust spread of all residuals), so a
  window a law explains badly really costs more.
* $C(f_k)$ -- the number of terms (a small complexity penalty, $\beta = 0.002$).
* $D_\text{geom}$ -- an optional "where the window sits" term. The default is
  $\alpha = 0$: assignment is **by the law only**.

## 2.3 The optimisation: alternate two easy problems

The objective is hard jointly (labels are discrete, the law's *form* is
discrete too), but each half is easy with the other fixed. This is **block
coordinate descent** -- the same idea as K-means or hard EM, with "centroid"
replaced by "law":

| step | fixed | solved | how |
|---|---|---|---|
| 0 · start | -- | labels $z$ | K-means in **mechanism space** (2.4) |
| 1 · fit | labels $z$ | laws $f_k$ | per cluster, **greedy forward selection with BIC** over the library; coefficients by least squares |
| 2 · score | laws | cost matrix $J_w(k)$ | every window under **every** law, **out-of-fold** (cross-fit: a window is never scored by a law fitted on itself) |
| 3 · assign | cost matrix | labels $z$ | $z_w = \arg\min_k J_w(k)$; tiny clusters repaired |
| 4 · repeat | | | until fewer than 1% of windows move (max 10 rounds), then refit the laws |

*Greedy forward BIC* (step 1): start from a constant; at each round try adding
every library term, keep the one that lowers
$\mathrm{BIC} = N\log(\mathrm{RSS}/N) + p\log N$ the most; stop when nothing
lowers it or 5 terms are in. BIC trades fit ($\mathrm{RSS}$) against size ($p$
coefficients), so the law stays short. The engine is swappable (PySR, PhySO
are supported); this deterministic one is the default.

## 2.4 The start: mechanism space

The start matters, so it is not done on look-alike statistics either. Each
window is mapped to its **law coefficients**: regress the window on the
library, remove what every window shares, and whiten so that noise is equally
large in every direction ($s_w = C_w^\top r_w$, `lrdsr.core.mechanism_space`).
In that space each regime is a round blob, centres $\sqrt{n\rho}$ apart --
the setting where **plain K-means is the right rule**.

## 2.5 How well can anyone do? The ceiling

Even someone who **knows** both laws misclassifies a window sometimes, because
of noise. That best-possible error is a formula:

$$
P_\text{err} = Q\!\left(\tfrac{\sqrt{n\,\rho}}{2}\right), \qquad
\rho = \frac{\mathbb{E}\big[(f_1(x) - f_0(x))^2\big]}{\sigma^2}
\quad(\text{separation of the laws in noise units}),
$$

$Q$ the Gaussian tail. Every result below is measured **against this
ceiling** (the "oracle"), so "good" has a precise meaning: *close to what is
possible at all*.

## 2.6 Variants of the same idea

* **Soft EM** (`SoftLRDSR`): the same mixture, but each window gets a
  *probability* per law instead of a hard label; Gaussian or Student-t noise.
* **Real time** (`OnlineLRDSR`): window by window, microseconds each; a window
  no law explains is buffered and a consistent buffer becomes a **new law**.
* **Classification** (`LawClassifier`): with labels, fit $L$ laws per class and
  assign a new window to the class whose laws explain it best.

The pipeline end to end:
"""),
    ("code", r'''
from matplotlib.patches import FancyBboxPatch

steps = [("windows\n$(x_w, y_w)$", "#e8e8e8"),
         ("mechanism space\nlaw coefficients\n$s_w$", "#dce8f5"),
         ("K-means\nstart labels", "#dce8f5"),
         ("fit one law per cluster\ngreedy forward BIC\nover the library", "#fbe5d0"),
         ("score every window\nunder every law\n(out-of-fold, Huber)", "#fbe5d0"),
         ("reassign\n$z_w = \\arg\\min_k J_w(k)$", "#fbe5d0"),
         ("laws $f_1..f_K$\n+ labels\n(+ posteriors)", "#d9ecd9")]
fig, ax = plt.subplots(figsize=(15, 2.9))
xs = np.linspace(0.07, 0.93, len(steps)); wbox = 0.118
for i, ((txt, col), x) in enumerate(zip(steps, xs)):
    ax.add_patch(FancyBboxPatch((x - wbox / 2, 0.36), wbox, 0.42, boxstyle="round,pad=0.012",
                                fc=col, ec="#555555", lw=1))
    ax.text(x, 0.57, txt, ha="center", va="center", fontsize=8.6)
    if i < len(steps) - 1:
        ax.annotate("", (xs[i + 1] - wbox / 2 - 0.004, 0.57), (x + wbox / 2 + 0.004, 0.57),
                    arrowprops={"arrowstyle": "->", "lw": 1.4})
ax.annotate("", (xs[3], 0.34), (xs[5], 0.34),
            arrowprops={"arrowstyle": "->", "lw": 1.4, "color": viz.PALETTE[1],
                        "connectionstyle": "arc3,rad=-0.35"})
ax.text(xs[4], 0.06, "repeat until < 1% of windows move  (block coordinate descent)",
        ha="center", color=viz.PALETTE[1], fontsize=9.5)
ax.text(xs[1] + 0.06, 0.9, "step 0: the start", ha="center", color=viz.PALETTE[0], fontsize=9.5)
ax.text(xs[4], 0.9, "steps 1-3: the loop", ha="center", color=viz.PALETTE[1], fontsize=9.5)
ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off");
'''),
    ("md", r"""
The library the laws are built from, and the three losses most used for the
equation term. The right panel is the *influence*: how hard one sample can
pull the fit. Squared loss lets an outlier pull without limit; Huber caps it.
"""),
    ("code", r'''
from lrdsr.core.soft import library_terms

xg = np.linspace(-2.2, 2.2, 300)
Phi, names = library_terms(xg[:, None], feature_names=["x"])
fig, ax = plt.subplots(1, 3, figsize=(15, 3.3))
for j in range(1, Phi.shape[1]):
    ax[0].plot(xg, Phi[:, j], lw=1.6, color=viz.PALETTE[(j - 1) % len(viz.PALETTE)], label=names[j])
ax[0].set(title="the term library (+ a constant)", xlabel="x", ylim=(-4, 5))
ax[0].legend(fontsize=7.5, ncol=2, loc="upper center")
viz.plot_losses(ax=ax[1], losses=("squared", "huber", "cauchy"))
viz.plot_losses(ax=ax[2], losses=("squared", "huber", "cauchy"), influence=True)
ax[1].set_title("loss: what a residual costs"); ax[2].set_title("influence: how hard one sample pulls")
fig.tight_layout()
'''),
    ("md", r"""
---
# 3 · A worked example, live

Three laws, 240 windows of 16 noisy samples each. The inputs of every window
are drawn from the same range, so *where* a window sits says nothing about
its law. The true labels are kept aside **for scoring only**.
"""),
    ("code", r'''
from experiments.functions.data import simulate_function_windows
from experiments.common.fitting import window_features
from lrdsr import GroupedDCSR, SoftLRDSR, FastSymbolicRegressor
from lrdsr.core.evaluation import aligned_accuracy

X, y, Zgeo, z, equations = simulate_function_windows(
    n_windows=240, window_len=16, noise_std=1.5, geometry_overlap=3.0, seed=11)
true_laws = [lambda x: 0.8 * x**2 + 1.5 * x + 0.5,
             lambda x: 2.5 * np.sin(2 * x) - 0.5,
             lambda x: 0.35 * x**3 - 1.2]
for k, e in enumerate(equations):
    print(f"true law {k}:  y = {e}   ({np.sum(z == k)} windows)")

fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
ax[0].plot(X[:60, :, 0].ravel(), y[:60].ravel(), "o", ms=2.5, color="#777777")
ax[0].set(title="what you get: 60 windows, pooled, unlabelled", xlabel="x", ylabel="y")
viz.plot_windows(X, y, z, ax=ax[1], max_windows=60)
ax[1].set_title("the same, coloured by the hidden law")
viz.plot_laws(truth=true_laws, x_range=(-2.2, 2.2), ax=ax[2]); ax[2].set_title("the three true laws")
fig.tight_layout()
'''),
    ("md", r"""
**Look-alike space vs. mechanism space.** Left: two window statistics (what a
standard clustering would use) -- the colours are mixed. Right: the same
windows in mechanism space -- three separate groups.
"""),
    ("code", r'''
Zf, fnames = window_features(X, y)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for k in range(3):
    ax[0].scatter(Zf[z == k, 0], Zf[z == k, 5], s=12, color=viz.regime_color(k), lw=0, label=f"law {k}")
ax[0].set(title="look-alike space: window statistics", xlabel="window mean of y", ylabel="window mean of x*y")
ax[0].legend()
viz.plot_mechanism_space(X, y, z, ax=ax[1], feature_names=["x"])
fig.tight_layout()
'''),
    ("md", r"""
**The loop, iteration by iteration.** To *see* the loop work, start it from a
bad partition on purpose (K-means on the look-alike statistics) and stop it
after 0, 1, 2 and all rounds. Solid: the laws fitted at that point; dashed:
the truth.
"""),
    ("code", r'''
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

bad_start = KMeans(3, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(Zf))
cfg = {"n_clusters": 3, "alpha_geom": 0.0, "backend": "fast", "backend_kwargs": {"max_terms": 5},
       "score_mode": "cross_fit", "residual_scale": "global", "random_state": 11}
fig, ax = plt.subplots(1, 4, figsize=(16, 3.4), sharey=True)
for a, rounds in zip(ax, (0, 1, 2, 10)):
    r = GroupedDCSR(init=bad_start, max_iter=rounds, **cfg).fit(X, y, Zf, feature_names=["x"])
    viz.plot_laws(r.models, x_range=(-2.2, 2.2), truth=true_laws, ax=a)
    a.get_legend().remove()
    a.set_title(f"after {rounds} round{'s' * (rounds != 1)}" if rounds < 10 else "converged",
                fontsize=10)
    a.text(0.03, 0.95, f"accuracy {aligned_accuracy(z, r.labels):.1%}", transform=a.transAxes,
           va="top", fontsize=10, weight="bold")
ax[0].set_ylabel("f(x)")
fig.suptitle("the loop repairs a bad start: fit laws, reassign windows, repeat", y=1.03)
fig.tight_layout()
'''),
    ("md", r"""
**The real fit** (the project default: start in mechanism space, Huber loss,
out-of-fold scoring, assignment by the law only). And, inside it, how the law
of one cluster is *built*: greedy forward selection adds one library term at a
time while BIC keeps falling.
"""),
    ("code", r'''
res = GroupedDCSR(init="mechanism", **cfg).fit(X, y, Zf, feature_names=["x"], true_labels_for_eval=z)
print(f"accuracy {aligned_accuracy(z, res.labels):.1%}  in {len(res.history)} rounds\n")
for k, m in enumerate(res.models):
    print(f"recovered law {k}:  y = {m.expression()}")

# Greedy forward BIC, replayed step by step for the cluster holding the
# parabola: every round tries every remaining term and keeps the best one.
k_par = np.bincount(res.labels[z == 0]).argmax()
Xk = X[res.labels == k_par].reshape(-1, 1); yk = y[res.labels == k_par].ravel()
sr = FastSymbolicRegressor(max_terms=5, feature_names=["x"])
lib, nk = sr._make_library(Xk), len(yk)
chosen, best = [], sr._bic(yk, np.full(nk, yk.mean()), 1)
trace = [(0, {"constant": best}, "constant")]
for rnd in range(1, 6):
    cands = {}
    for j, term in enumerate(lib):
        if j not in chosen:
            Phi = np.column_stack([np.ones(nk)] + [lib[c].values for c in [*chosen, j]])
            coef, *_ = np.linalg.lstsq(Phi, yk, rcond=None)
            cands[term.name] = sr._bic(yk, Phi @ coef, Phi.shape[1])
    pick = min(cands, key=cands.get)
    trace.append((rnd, cands, pick if cands[pick] < best - 1e-6 else None))
    if cands[pick] >= best - 1e-6:
        break
    best = cands[pick]; chosen.append(next(j for j, t in enumerate(lib) if t.name == pick))
'''),
    ("code", r'''
fig, ax = plt.subplots(1, 4, figsize=(16, 3.5), gridspec_kw={"width_ratios": [1.3, 1.1, 0.9, 0.9]})
viz.plot_laws(res.models, x_range=(-2.2, 2.2), truth=true_laws, ax=ax[0])
ax[0].set_title("recovered laws (solid) vs truth (dashed)")
path = []
for rnd, cands, pick in trace:
    ax[1].scatter([rnd] * len(cands), list(cands.values()), s=14, color="#b5b5b5", zorder=2)
    if pick is not None:
        path.append((rnd, cands[pick]))
        ax[1].annotate(("+ " if rnd else "") + pick, (rnd, cands[pick]), textcoords="offset points",
                       xytext=(6, -3), fontsize=8.5, color=viz.PALETTE[1], weight="bold")
    else:
        ax[1].annotate("stop: no term\nlowers BIC", (rnd, min(cands.values())), textcoords="offset points",
                       xytext=(-10, 12), fontsize=8, ha="center")
ax[1].plot(*zip(*path), "-o", color=viz.PALETTE[1], zorder=3)
ax[1].set(title=f"building law {k_par}: grey = every candidate term", xlabel="round",
          ylabel="BIC (lower = better)", xticks=range(len(trace)))
viz.plot_confusion(z, res.labels, ax=ax[2])
viz.plot_cost_matrix(res.total_cost, res.labels, ax=ax[3])
ax[3].set_title("cost of each window under each law")
fig.tight_layout()
'''),
    ("md", r"""
With 16 noisy points a window, a law can come back as an *equivalent*
formula rather than in its own symbols (e.g. $\cos x \approx 1 - x^2/2$ standing
in for part of a parabola). The curves agree where the data are, which is
what the left panel checks.

**The soft version** gives each window a probability per law, so you see
*how sure* each assignment is (right: one row per window, sorted).
"""),
    ("code", r'''
soft = SoftLRDSR(n_clusters=3, feature_names=["x"], random_state=11).fit(X, y)
unsure = np.mean(soft.responsibilities.max(1) < 0.9)
print(f"soft EM accuracy {aligned_accuracy(z, soft.labels):.1%}; {unsure:.1%} of windows are < 90% sure")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={"width_ratios": [1, 1.6]})
viz.plot_laws(coef=soft.coef, feature_names=["x"], x_range=(-2.2, 2.2), truth=true_laws, ax=ax[0])
ax[0].set_title("soft EM laws")
viz.plot_responsibilities(soft.responsibilities, ax=ax[1])
ax[1].set_title("posterior probability of each law, per window")
fig.tight_layout()
'''),
    ("md", r"""
**How close to the ceiling is that?** At this noise level even the oracle --
who knows the true laws and labels each window by the smallest residual -- is
perfect, so turn the noise up. The method knows nothing and should stay near
the oracle; the look-alike clustering should not. (On one sample of 240
windows the method can land a hair *above* the oracle by chance; averaged over
seeds it does not -- the oracle is the ceiling.)
"""),
    ("code", r'''
rows = []
for noise in (1.5, 3.0, 4.5, 6.0):
    Xn, yn, _, zn, _ = simulate_function_windows(n_windows=240, window_len=16, noise_std=noise,
                                                 geometry_overlap=3.0, seed=11)
    Zn, _ = window_features(Xn, yn)
    orc = np.column_stack([((yn - f(Xn[:, :, 0])) ** 2).sum(1) for f in true_laws]).argmin(1)
    fit = GroupedDCSR(init="mechanism", **cfg).fit(Xn, yn, Zn, feature_names=["x"])
    km = KMeans(3, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(Zn))
    rows.append({"noise": noise, "oracle (knows the laws)": np.mean(orc == zn),
                 "LR-DSR (no labels)": aligned_accuracy(zn, fit.labels),
                 "K-means on look-alike statistics": aligned_accuracy(zn, km)})
acc = pd.DataFrame(rows).set_index("noise")
ax = acc.plot.bar(figsize=(9, 3.4), color=["#222222", viz.PALETTE[0], "#9a9a9a"], width=0.8, rot=0)
ax.set(ylim=(0.3, 1.05), ylabel="accuracy", xlabel="noise level (sigma)",
       title="the method tracks the best possible accuracy as the noise grows")
ax.axhline(1 / 3, color="k", ls=":", lw=1); ax.text(3.42, 0.34, "chance", fontsize=8)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8.5);
'''),
    ("md", r"""
### An honest note on the loop

The worked example started the loop from a bad partition to show it moving.
Measured systematically (mission 3), the loop turns out to be a
**stabiliser**: whatever quality of start it is given, it returns roughly the
same answer. It repairs a bad start a lot, but on the good mechanism-space
start it changes almost nothing (it can even cost a fraction of a percent).
The *start* does most of the work; the loop makes the method robust to a bad
one and is what produces the named laws.
"""),
    ("code", r'''
from analysis import plots as P
P.loop_value();
'''),
    ("md", r"""
---
# 4 · What works: the main results

Each plot is read from the committed results (6 random seeds: tuned on 3,
reported on 3 others; true labels used only for scoring).

**The ceiling is reachable without labels.** Left: the measured error of the
oracle lies on the formula $Q(\sqrt{n\rho}/2)$. Right: K-means in mechanism
space (no labels) against that oracle, over 72 settings -- on the diagonal.
"""),
    ("code", r'''
P.attainability();
'''),
    ("md", r"""
**Robust and learned losses.** Under heavy-tailed noise the squared loss wastes
data; Huber and a *learned* loss (the method fits the noise distribution while
it clusters) reach the best possible error.
"""),
    ("code", r'''
from analysis.figs_losses import FIGURES as LF
LF["loss_estimator"]();
'''),
    ("md", r"""
**Real time.** A stream switches laws sample by sample; the detector finds each
switch after a delay the theory predicts (to a median 2.5%).
"""),
    ("code", r'''
from analysis.figs_online import FIGURES as OF
OF["online_sequential"]();
'''),
    ("md", r"""
**Days with sensor gaps.** Hourly highway traffic (I-94, Minneapolis). A law
does not need a complete day: it is evaluated at the hours that were observed.
On days with only 6-11 hours observed the law labels every day correctly; the
standard fix -- fill in the missing hours, then classify -- does not.
"""),
    ("code", r'''
from analysis.figs_realdata import FIGURES as RF
RF["realdata_gaps"]();
'''),
    ("md", r"""
**A new law appears in a stream.** With a flexible (kernel) basis the online
method *discovers* a law it had never seen and gives it its own group.
"""),
    ("code", r'''
from analysis.figs_partial import FIGURES as PF
PF["online_kernel_birth"]();
'''),
    ("md", r"""
**Classification, with a formula for its learning curve.** When labels exist,
the error as a function of the number of training windows is predicted
exactly by the theory (V10).
"""),
    ("code", r'''
from analysis.figs_classify import FIGURES as CF
CF["v10_learning_curve"]();
'''),
    ("md", r"""
**Wind turbines: every window has its own inputs.** Six-hour blocks of a real
wind farm (Kelmarsh, UK); each block sees different wind speeds, so there is no
common "profile" to compare. Laws separate cold from warm air and waked from
free flow better than the industry method of bins, on turbines never trained
on -- and the cold/warm power ratio lands on the air-density prediction.
"""),
    ("code", r'''
from analysis.figs_wind import FIGURES as WF
WF["wind_transfer"](); WF["wind_physics"]();
'''),
    ("md", r"""
**Is this better than plain symbolic regression?** The natural competitor: run
the *same* symbolic-regression engine without the latent-regime structure --
one law for all windows ("pooled SR"), or one law per window and then cluster
the curves ("SR per window"). *These are the newest runs (2026-09-29) and are
not yet in `RESULTS.md`.*
"""),
    ("code", r'''
SB = RES / "srbaseline"
A = pd.read_csv(SB / "estimation_assignment.csv"); L = pd.read_csv(SB / "estimation_law.csv")
F = pd.read_csv(SB / "families_error.csv"); C = pd.read_csv(SB / "classification_summary.csv")
arms = [("oracle", "oracle (knows the laws)", "#222222", "--"), ("lrdsr", "LR-DSR", viz.PALETTE[0], "-"),
        ("sr_per_window", "SR per window, then cluster", viz.PALETTE[1], "-"),
        ("sr_pooled", "SR pooled (one law)", viz.PALETTE[4], "-")]
fig, ax = plt.subplots(1, 4, figsize=(17, 3.5))
for key, lab, c, ls in arms:
    ax[0].plot(A.rho, A[f"{key}_error"], ls, marker="o", color=c, label=lab)
    ax[1].plot(L.rho, L[f"{key}_law_error"], ls, marker="o", color=c, label=lab)
ax[0].set(xscale="log", title="which window came from which law", xlabel=r"separation $\rho$", ylabel="assignment error")
ax[1].set(xscale="log", title="how far the recovered laws are", xlabel=r"separation $\rho$", ylabel="law error (1 = gap between laws)")
ax[0].legend(fontsize=7.5)
f = F[F.suite == "families"].set_index("rho")
cols = [("oracle", "#222222"), ("lrdsr", viz.PALETTE[0]), ("mech_sr", viz.PALETTE[5]),
        ("soft_em", viz.PALETTE[3]), ("poly_mix", viz.PALETTE[2]), ("sr_per_window", viz.PALETTE[1]),
        ("summary_sr", "#999999")]
xp = np.arange(len(cols))
ax[2].bar(xp, [f.loc[0.25, c_] for c_, _ in cols], color=[c for _, c in cols])
ax[2].set_xticks(xp); ax[2].set_xticklabels([c_ for c_, _ in cols], rotation=35, ha="right", fontsize=8)
ax[2].set(title=r"regimes of different families, $\rho = 0.25$", ylabel="assignment error")
for s, mk in (("one_law", "o"), ("two_laws", "s")):
    g = C[C.structure == s]
    ax[3].plot(g.m, g.law_L1 if s == "one_law" else g.law_L2, "-" + mk, color=viz.PALETTE[0],
               label=f"law classifier ({s})")
    ax[3].plot(g.m, g.sr_per_class, "--" + mk, color=viz.PALETTE[1], label=f"SR per class ({s})")
ax[3].set(xscale="log", title="classification", xlabel="training windows per class", ylabel="test error")
ax[3].legend(fontsize=7)
fig.tight_layout()
'''),
    ("md", r"""
Pooled SR is at chance (one law cannot describe two). Fitting a law to each
window first and clustering afterwards makes about **twice the assignment
error** of LR-DSR at low separation, because 16 noisy points are too few to
pin down a law -- pooling the windows of a cluster is the whole point. When
a class is made of *two* laws, the classifier that fits two laws per class
(right, squares) beats one SR law per class by a wide margin. One declared
prediction failed narrowly: at the largest separation SR-per-window's laws are
as good as LR-DSR's (0.135 vs 0.137).

---
# 5 · Real data, live: a year of bike sharing, on a calendar

Capital Bikeshare (Washington DC), hourly rentals 2011-12, **one window per
day**; the input is the hour of the day. The method is given $K=2$ and no
labels. Afterwards we compare with the calendar (working day or not) -- a
*proxy* for the truth, not the truth.
"""),
    ("code", r'''
from experiments.realdata.windows import load
from experiments.realdata.run import fourier

bike = load("bike")
sb = SoftLRDSR(2, basis=fourier, noise="student_t", random_state=11).fit(bike.X_seq, bike.y_seq)
lab = sb.labels if np.mean(sb.labels == bike.reference) > 0.5 else 1 - sb.labels   # 1 = working-day law
print(f"{len(lab)} days; agreement with the calendar: {np.mean(lab == bike.reference):.1%}")

fig, ax = plt.subplots(1, 2, figsize=(13, 3.5))
hours = np.arange(24)
for k, name in ((1, "law 1: commuting (two peaks)"), (0, "law 0: leisure (one hump)")):
    sel = np.flatnonzero(lab == k)
    ax[0].plot(hours, bike.y_seq[sel[:200]].T, color=viz.regime_color(1 - k), lw=0.3, alpha=0.25)
    ax[0].plot(hours, bike.y_seq[sel].mean(0), color=viz.regime_color(1 - k), lw=2.8, label=name)
ax[0].set(title="every day, coloured by the law the method found", xlabel="hour of day",
          ylabel="log(1 + rentals) - day mean", xticks=range(0, 25, 3)); ax[0].legend(loc="lower center")
ent = sb.entropy
ax[1].hist(sb.responsibilities.max(1), bins=30, color=viz.PALETTE[0])
ax[1].set(title="how sure the method is, per day", xlabel="probability of the chosen law", ylabel="days", yscale="log")
fig.tight_layout()
'''),
    ("code", r'''
dates = pd.to_datetime(pd.Series(bike.dates))
fig, axes = plt.subplots(2, 1, figsize=(15, 4.4))
from matplotlib.colors import ListedColormap
cmap = ListedColormap([viz.regime_color(1), viz.regime_color(0)])
for a, year in zip(axes, sorted(dates.dt.year.unique())):
    m = (dates.dt.year == year).to_numpy()
    d = dates[m]; wk = d.dt.isocalendar().week.to_numpy().astype(int); dow = d.dt.dayofweek.to_numpy()
    wk = np.where((d.dt.month == 1).to_numpy() & (wk > 50), 0, wk)
    grid = np.full((7, 54), np.nan); grid[dow, wk] = lab[m]
    a.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="none")
    odd = lab[m] != bike.reference[m]
    a.scatter(wk[odd], dow[odd], marker="x", color="k", s=28, lw=1.3)
    a.set_yticks(range(7)); a.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], fontsize=8)
    starts = d.groupby(d.dt.month).apply(lambda s: s.dt.isocalendar().week.iloc[0]).to_numpy().astype(int)
    starts[0] = 0
    a.set_xticks(starts); a.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug",
                                             "Sep", "Oct", "Nov", "Dec"], fontsize=8)
    a.set_ylabel(str(year))
axes[0].set_title("blue = commuting law, orange = leisure law;  x = the law disagrees with the calendar")
fig.tight_layout()
'''),
    ("md", r"""
Weekends and the big holidays (Memorial Day, July 4th, Labor Day,
Thanksgiving, ...) are found without being told what a weekend or a holiday
is. The crosses are where law and calendar disagree, and they are of two
kinds: official holidays most people still work (Columbus Day, Veterans Day,
DC Emancipation Day -- the law says *commuting*), and days off the calendar
does not know about (the Friday after Thanksgiving, Christmas Eve -- the law
says *leisure*). In both the law is arguably right and the calendar wrong. On this data every day has the same 24 inputs, so here a
simpler clustering of the raw 24-hour profile ties the method (mission 7); the
method's edge on real data appears when the inputs differ -- days with
missing hours (above) and wind turbines.

---
# 6 · What did not work

### Failure 1: a law the library cannot write -- and a fix (trial)

Two fast oscillations, $\sin 4x$ vs. $\sin 4.6x$. The library's fastest term
is $\sin 2x$, so no combination of its terms can follow them: about 37% of the
difference between the two laws is invisible to it.

**The trial fix** keeps the problem, the objective and the loop exactly as
they are, and only gives the per-cluster fit a richer vocabulary: two new term
types, $\sin(a\,x)$ and $\cos(a\,x)$, whose frequency $a$ is **fitted**
(`SinusoidSymbolicRegressor`, backend `"fast_sin"`). For every $a$ on a grid
the coefficient is ordinary least squares, the best $a$ is refined locally,
and BIC charges the term one extra parameter for $a$. It competes with the
fixed terms, so it is used only when a sinusoid explains the data better.

Every method on the **same windows** (240 windows of 32 samples), at two
noise levels -- noisy ($\rho = 0.25$) and clean ($\rho = 1$):

| family | methods |
|---|---|
| the ceiling | **oracle** -- knows the true laws |
| this project | **LR-DSR + fitted frequency** (the trial), LR-DSR with the standard library, soft EM, mechanism K-means |
| signal processing | **periodogram + K-means** (Lomb-Scargle, the standard tool for oscillations), peak frequency + K-means |
| plain symbolic regression | a law per window, then K-means on the fitted curves -- with the standard library, and with the same fitted-frequency term |
| clustering the data | K-means on the raw curve, K-means on window summaries |
"""),
    ("code", r'''
from experiments.problems import zoo
from experiments.problems.sinusoid_trial import FREQ_GRID, _kmeans, _periodogram, _sr_per_window, fit as fit_hf
from lrdsr.core.mechanism_space import mechanism_init

prob = zoo.PROBLEM_BY_NAME["high_frequency"]
truth_hf = [lambda x: np.sin(4 * x), lambda x: np.sin(4.6 * x)]
grid_hf = zoo.eval_grid(prob, 200)
METHODS_HF = [  # (key, label, colour)
    ("oracle", "oracle (knows the laws)", "#222222"),
    ("lrdsr_sin", "LR-DSR + fitted frequency", viz.PALETTE[0]),
    ("mech", "mechanism K-means", "#6f9fd8"), ("soft", "soft EM (library)", "#6f9fd8"),
    ("lrdsr", "LR-DSR, standard library", "#6f9fd8"),
    ("pgram", "periodogram + K-means", viz.PALETTE[1]), ("peak", "peak frequency + K-means", viz.PALETTE[1]),
    ("sr_sin", "SR per window (fitted freq.)", viz.PALETTE[2]), ("sr", "SR per window (library)", viz.PALETTE[2]),
    ("profile", "K-means, raw curve", "#9a9a9a"), ("summ", "K-means, window summaries", "#9a9a9a")]

runs = {}
for rho in (0.25, 1.0):
    X_, y_, z_ = zoo.make_windows(prob, zoo.sigma_for_rho(prob, rho), n_windows=240, window_len=32, seed=11)
    Z_, _ = window_features(X_, y_)
    P_ = _periodogram(X_, y_)
    lib_fit, sin_fit = (fit_hf(be, X_, y_, Z_, ["x"], 2, 11) for be in ("fast", "fast_sin"))
    lab = {"oracle": zoo.oracle_labels(prob, X_, y_), "lrdsr_sin": sin_fit.labels, "lrdsr": lib_fit.labels,
           "soft": SoftLRDSR(2, feature_names=["x"], random_state=11).fit(X_, y_).labels,
           "mech": mechanism_init(X_, y_, 2, seed=11, feature_names=["x"]),
           "pgram": _kmeans(P_, 2, 11), "peak": _kmeans(FREQ_GRID[P_.argmax(1)], 2, 11),
           "sr_sin": _sr_per_window("fast_sin", X_, y_, ["x"], grid_hf, 2, 11),
           "sr": _sr_per_window("fast", X_, y_, ["x"], grid_hf, 2, 11),
           "profile": _kmeans(np.take_along_axis(y_, np.argsort(X_[..., 0], 1), 1), 2, 11),
           "summ": _kmeans(Z_, 2, 11)}
    runs[rho] = {"acc": {k: aligned_accuracy(z_, v) for k, v in lab.items()},
                 "fits": (lib_fit, sin_fit), "P": P_, "z": z_}

print(f"{'accuracy':34s} {'noisy (rho=0.25)':>17s} {'clean (rho=1)':>14s}")
for key, name, _ in METHODS_HF:
    print(f"{name:34s} {runs[0.25]['acc'][key]:17.1%} {runs[1.0]['acc'][key]:14.1%}")
for tag, m_ in zip(("standard library", "+ fitted frequency"), runs[0.25]["fits"]):
    print(f"\nLR-DSR, {tag} (noisy):"); [print("   y =", m.expression()) for m in m_.models]
'''),
    ("code", r'''
fig = plt.figure(figsize=(17, 13))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1.25, 1], hspace=0.45, wspace=0.25)

# row 1: the laws
ax = fig.add_subplot(gs[0, 0])
xs_ = np.linspace(*prob.x_range, 400)
for k, (f, lab_) in enumerate(zip(prob.laws, prob.truths)):
    ax.plot(xs_, f(xs_[:, None]), color=viz.regime_color(k), lw=1.8, label=lab_)
ax.set(title="the two true laws", xlabel="x"); ax.legend(loc="lower left")
for j, (tag, m_) in enumerate(zip(("standard library", "+ fitted frequency"), runs[0.25]["fits"])):
    ax = fig.add_subplot(gs[0, j + 1])
    viz.plot_laws(m_.models, x_range=prob.x_range, truth=truth_hf, ax=ax)
    ax.get_legend().remove(); ax.set_ylim(-1.9, 1.9)
    ax.set_title(f"recovered by LR-DSR, {tag} (solid)")

# row 2: every method, noisy and clean
for j, (rho, ttl) in enumerate(((0.25, r"noisy windows ($\rho$ = 0.25)"), (1.0, r"clean windows ($\rho$ = 1)"))):
    ax = fig.add_subplot(gs[1, 0:2] if j == 0 else gs[1, 2])
    acc = runs[rho]["acc"]
    yp = np.arange(len(METHODS_HF))[::-1]
    ax.barh(yp, [acc[k] for k, _, _ in METHODS_HF], color=[c for _, _, c in METHODS_HF], height=0.72)
    for y_i, (k, _, _) in zip(yp, METHODS_HF):
        ax.text(acc[k] + 0.005, y_i, f"{acc[k]:.1%}", va="center", fontsize=8.5)
    ax.axvline(0.5, color="k", ls=":", lw=0.8)
    ax.set_yticks(yp); ax.set_yticklabels([n for _, n, _ in METHODS_HF] if j == 0 else [], fontsize=9)
    ax.set(xlim=(0.45, 1.06), xlabel="accuracy  (dotted: chance)", title=ttl)

# row 3: what a periodogram sees
for j, rho in enumerate((0.25, 1.0)):
    ax = fig.add_subplot(gs[2, j])
    P_, z_ = runs[rho]["P"], runs[rho]["z"]
    for k in range(2):
        for w in np.flatnonzero(z_ == k)[:6]:
            ax.plot(FREQ_GRID, P_[w], color=viz.regime_color(k), lw=0.6, alpha=0.5)
        ax.plot(FREQ_GRID, P_[z_ == k].mean(0), color=viz.regime_color(k), lw=2.4, label=f"{prob.truths[k]}: mean")
    for t in (4.0, 4.6):
        ax.axvline(t, color="#333333", ls="--", lw=0.9)
    ax.set(xlabel="frequency a", ylabel="periodogram power", xlim=(2, 7),
           title=rf"what a periodogram sees, $\rho$ = {rho} (thin: single windows)")
    ax.legend(fontsize=8, loc="upper left")
ax = fig.add_subplot(gs[2, 2])
freqs = sorted(t[2] for m in runs[0.25]["fits"][1].models for t in m.terms_ if t[0] != "lib")
peaks = FREQ_GRID[runs[0.25]["P"].argmax(1)]
for k in range(2):
    ax.hist(peaks[runs[0.25]["z"] == k], bins=np.arange(2, 7.01, 0.1), color=viz.regime_color(k),
            alpha=0.6, label=f"single-window peak, {prob.truths[k]}")
for f_ in freqs:
    ax.axvline(f_, color=viz.PALETTE[0], lw=2.5)
ax.set(xlabel="frequency a", ylabel="windows", xlim=(2, 7),
       title="one window's peak vs LR-DSR's pooled fit (blue)\n" + r"noisy, $\rho$ = 0.25")
ax.legend(fontsize=7.5, loc="upper right")
'''),
    ("md", r"""
**Reading the three rows.**

1. **The laws.** The standard library can only return surrogates (a cubic plus
   slow sines); with the fitted frequency LR-DSR returns the laws in their own
   symbols, frequencies within ~0.02 of 4 and 4.6.
2. **Every method.** In noisy windows only the methods that **pool** windows
   get near the oracle -- LR-DSR with the fitted frequency is at the oracle (on
   one sample it can land a hair above it by chance), the library versions of
   this project's methods follow, and the standard methods are far behind:
   the periodogram at ~67%, plain symbolic regression and clustering the raw
   data near chance. On clean windows the standard spectral methods and
   per-window SR with the fitted term catch up; only per-window SR with the
   plain library, and clustering the data, stay at chance.
3. **Why the spectral methods lose in noise.** A single window has 32 samples
   over a range that holds less than one period of the *difference* between
   4 and 4.6, so one window's periodogram peak is wide (left: thin lines) and
   the two regimes' peaks overlap. Most single-window peaks land near the
   right frequency, but the two groups overlap around 4.2-4.5 and a tail of
   windows peaks far off (2-3.5, 5-6.6) -- enough to break a clustering of
   them (right). LR-DSR fits one frequency to all the windows of a regime --
   thousands of samples -- and lands on the truth (blue lines).

"""),
    ("md", r"""
**Does the extra freedom hurt the problems that did not need it?** The same
trial on all twelve zoo problems (`python -m experiments.problems.sinusoid_trial`;
150 windows, three separations, three seeds; four predictions declared
before the run, all four held):
"""),
    ("code", r'''
T = pd.read_csv(RES / "problems" / "sinusoid_trial_summary.csv").sort_values("gap_fast")
yp = np.arange(len(T))
fig, ax = plt.subplots(1, 2, figsize=(14, 3.8), gridspec_kw={"width_ratios": [1.4, 1]})
ax[0].barh(yp + 0.2, T.gap_fast, 0.4, color="#9a9a9a", label="standard library")
ax[0].barh(yp - 0.2, T.gap_fast_sin, 0.4, color=viz.PALETTE[0], label="+ fitted frequency")
ax[0].set_yticks(yp); ax[0].set_yticklabels(T.problem, fontsize=8.5)
ax[0].set(xlabel="gap to the oracle (lower = better)", title="every zoo problem: only high_frequency changes")
ax[0].legend(loc="lower right")
raw = pd.read_csv(RES / "problems" / "sinusoid_trial_raw.csv")
hf = raw[(raw.problem == "high_frequency") & (raw.backend == "fast_sin")]
for _, r in hf.iterrows():
    f = [float(v) for v in r.fitted_freqs.split()]
    ax[1].scatter([r.rho * (0.93 + 0.07 * (r.seed % 3))] * len(f), f, s=22, color=viz.PALETTE[0])
for t in (4.0, 4.6):
    ax[1].axhline(t, color="#333333", ls="--", lw=1)
ax[1].set(xscale="log", xlabel=r"separation $\rho$ (3 seeds each)", ylabel="fitted frequency a",
          title="the frequencies found (dashed: truth 4 and 4.6)", ylim=(3.7, 4.9))
fig.tight_layout()
'''),
    ("md", r"""
The gap on `high_frequency` falls from 0.237 to 0.013; the other eleven
problems are unchanged (mean change -0.0002, none worse by more than 0.003);
every seed at every separation finds both frequencies to within 0.06. The
price is speed -- the fit is about 8x slower, still ~2 s per problem. The
earlier kernel basis (mission 10) roughly halved this failure without
readable laws; the fitted frequency closes it *with* them. It is a trial: not
yet the default, and not yet in `RESULTS.md`.
"""),    ("md", r"""
**And the standard methods?** On the two sine problems, every standard way to
group these windows, on the same windows (`python -m experiments.problems.sinusoid_trial --baselines`):
clustering window summaries, K-means on the raw curve, plain symbolic
regression per window -- with the library *and* with the same fitted-frequency
vocabulary, so the new term is not credited to LR-DSR -- and the
signal-processing standard for oscillations, a **periodogram** per window
(Lomb-Scargle, which allows irregular sampling), then K-means.
"""),
    ("code", r'''
B = pd.read_csv(RES / "problems" / "sinusoid_baselines_raw.csv")
B = B.groupby(["problem", "rho", "method"]).error.mean().reset_index()
arms = [("oracle", "oracle (knows the laws)", "#222222", "--", 2.0),
        ("lrdsr_sin", "LR-DSR + fitted frequency", viz.PALETTE[0], "-", 2.6),
        ("lrdsr", "LR-DSR, standard library", viz.PALETTE[0], ":", 1.6),
        ("soft_em", "soft EM (library)", viz.PALETTE[3], "-", 1.2),
        ("periodogram_kmeans", "periodogram + K-means", viz.PALETTE[1], "-", 1.8),
        ("sr_per_window_sin", "SR per window (fitted freq.), then cluster", viz.PALETTE[2], "-", 1.4),
        ("sr_per_window", "SR per window (library), then cluster", viz.PALETTE[2], ":", 1.4),
        ("profile_kmeans", "K-means on the raw curve", "#9a9a9a", "-", 1.2),
        ("geometry", "clustering window summaries (best of 7)", "#9a9a9a", ":", 1.2)]
fig, ax = plt.subplots(1, 2, figsize=(15, 4.0), sharey=True)
for a, prob_name, ttl in zip(ax, ("high_frequency", "frequency_shift"),
                             (r"high_frequency: $\sin 4x$ vs $\sin 4.6x$", r"frequency_shift: $\sin x$ vs $\sin 1.3x$")):
    g = B[B.problem == prob_name]
    for key, lab_, col, ls, lw in arms:
        h = g[g.method == key].sort_values("rho")
        a.plot(h.rho, h.error, ls, marker="o", ms=4, color=col, lw=lw, label=lab_)
    a.axhline(0.5, color="k", lw=0.7, ls=":"); a.text(0.105, 0.51, "chance", fontsize=8)
    a.set(xscale="log", xlabel=r"separation $\rho$  (left: noisy, right: clean)", title=ttl)
ax[0].set_ylabel("error (lower = better)")
ax[1].legend(fontsize=7.8, loc="upper left", bbox_to_anchor=(1.01, 1))
fig.tight_layout()
'''),
    ("md", r"""
**On clean data (right end) the standard spectral methods solve it too** -- a
periodogram is at 0.000 once $\rho = 1$. **The method's edge is in noise**: at
$\rho = 0.25$ on `high_frequency` it errs at 0.049 against 0.26-0.48 for every
standard method, close to the oracle's 0.038. The reason is pooling: a
periodogram or a per-window fit has to find the frequency from one window's
48 noisy samples, while LR-DSR fits one frequency to *all* the windows of a
regime. Giving plain symbolic regression the same fitted-frequency vocabulary
helps it only on clean data -- the vocabulary is not the advantage, the
pooling is. Of four predictions declared before this comparison, three held;
the fourth (that periodograms lose by 0.1 on `frequency_shift` at *every*
separation) failed at $\rho = 1$, where they are perfect as well.
"""),
    ("md", r"""
### Failure 2: against strong classifiers on 24 standard datasets

On 24 public classification datasets (UCR archive: 9 with a daily cycle, 15
"shape" datasets), classifying by law beats a plain nearest-neighbour on the
raw series most of the time, but **loses to the best tuned standard
classifier** (and to time-warping, DTW, on shapes). When every series has the
same complete inputs, a discriminative classifier on the raw values is hard to
beat; the repairs (a time-shift nuisance, a stacked head) close part of the
gap, not all.
"""),
    ("code", r'''
CF["classify_benchmark"]();
'''),
    ("md", r"""
### Failure 3: noise that is correlated in time

The model is $y = f(x) + \varepsilon$, and the method assumes the **noise**
$\varepsilon$ is independent from sample to sample -- a fresh coin flip each
time. Time correlation in the *data* is fine: a daily curve that rises and
falls smoothly is the law $f$ itself. But real noise often *wanders*: once a
day runs above its curve it tends to stay above for a while. Wandering noise
carries less information per sample, and the method -- which sums samples as
if each were new evidence -- does not know that.

This check (`python -m experiments robustness`) keeps the twelve zoo problems,
their laws and their noise level exactly, and changes only one thing: the
noise becomes an AR(1) process with lag-1 correlation $\phi$. Left: what that
looks like.
"""),
    ("code", r'''
from experiments.robustness.run import ar1
from analysis.figs_robustness import FIGURES as RB

rng = np.random.default_rng(5)
xs = np.linspace(-2, 2, 48); u = rng.normal(size=48)
fig, ax = plt.subplots(1, 3, figsize=(15, 3.1), sharey=True)
for a, phi in zip(ax, (0.0, 0.6, 0.95)):
    a.plot(xs, xs**2 - 1, "--", color="#333333", lw=1.2, label="the law")
    a.plot(xs, xs**2 - 1 + 0.8 * ar1(u, phi), "o-", ms=3, lw=0.8, color=viz.PALETTE[0], label="one window")
    a.set(title=rf"$\phi$ = {phi}" + ("  (independent: what the method assumes)" if phi == 0 else ""),
          xlabel="x (samples in time order)")
ax[0].set_ylabel("y"); ax[0].legend(loc="upper center")
fig.suptitle("same law, same noise level, same random numbers -- only the correlation differs", y=1.03)
fig.tight_layout()
'''),
    ("md", r"""
**The failure in one picture.** As the correlation grows, the method falls
further behind the oracle -- from 2 points with independent noise to 19 at
$\phi = 0.95$. The green band is how correlated the residuals of the real days
are: there the extra cost is about one point.
"""),
    ("code", r'''
RB["robustness_failure"]();
'''),
    ("md", r"""
In more detail. **(a)** Every method gets worse as $\phi$ grows -- including the
oracle that knows the laws but sums samples (solid black), because correlated
samples are fewer *effective* samples. The method's extra loss on top of that
stays small up to $\phi \approx 0.3$ (0.029 vs 0.023 with independent noise) but
grows beyond (0.084 at $\phi = 0.8$): the start in mechanism space assumes round
noise clouds, and correlated noise makes them elongated. **(b)** The error is
still an exact formula, for both rules. **(c)** An oracle that *knows* the
correlation (dashed in (a)) whitens it away and even gets **better** at very
high $\phi$: smooth noise is easy to separate from a law gap that changes
quickly (a moving step, an interaction). **(d)** On the real bike and traffic
days the residual correlation is moderate (lag 1: 0.35 and 0.43) and dies out
after one hour -- shorter memory than AR(1) -- so real days sit in the green
band of (a), where the cost is small.
"""),
    ("code", r'''
RB["robustness_ar_noise"]();
'''),
    ("md", r"""
**Verdict: partly robust.** On data like the real days the independence
assumption costs little; with strongly wandering noise ($\phi \ge 0.6$) the
method loses clearly more than its own rule does, and a time-series version
(whitening the residuals inside the loop, with $\phi$ estimated from them) is
the natural fix -- the dashed curve shows how much it could gain. Of the five
predictions declared before the run, two held (the exact formulas at $\phi = 0$;
knowing $\phi$ helps more as it grows, though mostly on `moving_step` rather
than `high_frequency`) and three failed: the formula band was set tighter
than the Monte-Carlo noise of 150 windows (at 4000 windows every cell is
within 3 standard errors -- a check added after the run), the method's extra
loss passes 0.05 at $\phi = 0.6$, and the real residual correlation is below the
predicted 0.5.

### The other negatives (kept, not hidden)

* Choosing the number of laws $K$ by BIC fails on real days (it keeps adding).
* The partial-day error formula (V11) ranks which hours matter correctly, but
  its absolute level is off by a factor of a few.
* Scattered, irregular sampling gives no edge -- only *long* gaps do.
* Wind: one declared prediction failed (the wake effect), and unsupervised
  clusters of wind blocks match no physical proxy.

---
# 7 · Every task in one picture

Fifteen missions and the new symbolic-regression baseline. Each panel is the
headline comparison of one mission; the frame colour is the verdict
(**green** holds, **amber** partly). Unless the panel says otherwise, lower is
better. All numbers are read from `results/` by `analysis.report.numbers()`.
"""),
    ("code", r'''
from analysis.report import numbers
N = numbers()
sa = pd.read_csv(RES / "srbaseline" / "estimation_assignment.csv").set_index("rho").loc[0.25]
HOLD, PART = viz.PALETTE[2], viz.PALETTE[1]
ME, BASE, REF = viz.PALETTE[0], "#9a9a9a", "#222222"

panels = [  # (title, verdict, y-label, [(bar label, value, role)])
 ("1 · the ceiling is a formula", HOLD, "share of cells in band",
  [("V1 exact", N["v1_cells"] / 18, "me"), ("shared comp.", N["v4_unchanged"] / N["v4_cells"], "me")]),
 ("2 · reached without labels", HOLD, "error",
  [("oracle", N["bench_oracle"], "ref"), ("mech. K-means", N["bench_mechanism_kmeans"], "me"),
   ("LR-DSR", N["bench_lrdsr"], "me"), ("look-alike", N["bench_window_features"], "base")]),
 ("3 · the loop: a stabiliser", PART, "error: start -> after loop",
  [("random", N["loop_random_in"], "base"), ("+loop", N["loop_random_out"], "me"),
   ("K-means", N["loop_kmeans_in"], "base"), ("+loop", N["loop_kmeans_out"], "me"),
   ("mech.", N["loop_mechanism_in"], "base"), ("+loop", N["loop_mechanism_out"], "me")]),
 ("4 · twelve-problem zoo", HOLD, "gap to oracle",
  [("soft EM", N["zoo_gap_soft_em"], "me"), ("mech.", N["zoo_gap_mechanism_kmeans"], "me"),
   ("raw profile", N["zoo_gap_profile_kmeans"], "base"), ("look-alike", N["zoo_gap_geometry"], "base")]),
 ("5 · losses (t3 noise)", HOLD, "error",
  [("oracle", N["le_student_t3_oracle_lrt"], "ref"), ("learned", N["le_student_t3_hard_learned"], "me"),
   ("Huber", N["le_student_t3_hard_huber"], "me"), ("squared", N["le_student_t3_hard_squared"], "base")]),
 ("6 · real-time detection", HOLD, "delay (samples)",
  [("predicted", N["cusum_delay_pred"], "ref"), ("measured", N["cusum_delay_learned"], "me")]),
 ("7 · real days (traffic)", PART, "ARI (higher better)",
  [("soft EM", N["rd_traffic_soft_fourier_ari"], "me"), ("LR-DSR", N["rd_traffic_lrdsr_ari"], "me"),
   ("raw profile", N["rd_traffic_profile_ari"], "base"), ("look-alike", N["rd_traffic_geom_ari"], "base")]),
 ("8 · days with 6-11 hours", HOLD, "accuracy (higher better)",
  [("law", N["gap_lo_law"], "me"), ("mean-fill", N["gap_lo_mean"], "base"), ("linear-fill", N["gap_lo_lin"], "base")]),
 ("9 · partial-day error (V11)", PART, "share decided by 5 am",
  [("predicted", N["v11_traffic_5h_pred"], "ref"), ("observed", N["v11_traffic_5h_obs"], "me"),
   ("err. pred.", N["v11_cal_mid_pred"], "ref"), ("err. obs.", N["v11_cal_mid_obs"], "me")]),
 ("10 · kernel basis", PART, "gap to oracle, high freq.",
  [("library", N["k_hf_soft_em_library"], "base"), ("kernel", N["k_hf_soft_em_kernel"], "me")]),
 ("11 · classification (V10)", HOLD, "share of cells in band",
  [("fixed design", N["v10_fixed_in"] / N["v10_cells"], "me"), ("random design", N["v10x_in"] / N["v10x_n"], "me")]),
 ("12 · 24 UCR datasets", PART, "error (shape sets)",
  [("law", N["ucr_shape_law_vs_1nn_a"], "me"), ("raw 1-NN", N["ucr_shape_law_vs_1nn_b"], "base"),
   ("best tuned", N["ucr_shape_best_b"], "base")]),
 ("13 · repairs (GunPoint)", PART, "error",
  [("law", N["rp_GunPoint_law"], "base"), ("+ warp", N["rp_GunPoint_shift"], "me"), ("DTW", N["rp_GunPoint_dtw"], "base")]),
 ("14 · streaming with gaps", HOLD, "accuracy (higher better)",
  [("partial days", N["st_partial_acc"], "me"), ("complete days", N["st_complete_acc"], "me")]),
 ("15 · wind: cold vs warm", PART, "balanced error",
  [("ceiling", N["w_cold_ceiling"], "ref"), ("law", N["w_cold_law"], "me"), ("bins, best", N["w_cold_bins_best"], "base")]),
 ("16 · vs plain SR (new)", HOLD, r"error, $\rho = 0.25$",
  [("oracle", sa.oracle_error, "ref"), ("LR-DSR", sa.lrdsr_error, "me"),
   ("SR / window", sa.sr_per_window_error, "base"), ("SR pooled", sa.sr_pooled_error, "base")]),
]
role = {"me": ME, "base": BASE, "ref": REF}
fig, axes = plt.subplots(4, 4, figsize=(17, 13))
for a, (title, verdict, ylab, bars) in zip(axes.flat, panels):
    xp = np.arange(len(bars)); vals = [b[1] for b in bars]
    a.bar(xp, vals, color=[role[b[2]] for b in bars], width=0.66)
    for i, v in enumerate(vals):
        a.text(i, v, f"{v:.3f}" if v < 1 else f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    a.set_xticks(xp); a.set_xticklabels([b[0] for b in bars], fontsize=8, rotation=18 if len(bars) < 6 else 0)
    a.set_title(title, fontsize=10.5, weight="bold"); a.set_ylabel(ylab, fontsize=8.5)
    a.set_ylim(0, max(vals) * 1.22)
    for s in a.spines.values():
        s.set_visible(True); s.set_color(verdict); s.set_linewidth(3)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=ME, label="this method"), Patch(color=BASE, label="baseline / before"),
                    Patch(color=REF, label="oracle / theory"),
                    Patch(fc="white", ec=HOLD, lw=3, label="verdict: holds"),
                    Patch(fc="white", ec=PART, lw=3, label="verdict: partly")],
           loc="upper center", ncol=5, fontsize=10, bbox_to_anchor=(0.5, 1.02))
axes.flat[8].text(2.5, 0.35, "the timing is right;\nthe error level is\noff by ~3x", ha="center", fontsize=8.5)
fig.tight_layout(rect=(0, 0, 1, 0.98))
'''),
    ("md", r"""
## Take-aways

1. **Clustering by the law is a well-posed problem with a known ceiling**,
   $Q(\sqrt{n\rho}/2)$, and the method reaches it without labels: the key is
   the *start* in mechanism space; the alternating loop makes it robust and
   produces readable laws.
2. **It wins where the inputs differ from window to window** -- days with
   long sensor gaps, wind turbines at different wind speeds, a new law in a
   stream. When every window has the same complete inputs, a good raw-data
   method ties it (clustering) or beats it (tuned classifiers).
3. **It beats plain symbolic regression**: fitting a law per window is too
   noisy, one law for everything is wrong; pooling the windows of a regime is
   what makes the laws accurate.
4. **Its known limit is the library**: a law the library cannot express costs
   accuracy. For fast oscillations a term with a fitted frequency (trial)
   closes the gap (0.237 → 0.013) and keeps the laws readable.
5. **It assumes independent noise.** At the correlation real days show, that
   costs little; strongly wandering noise costs more, and whitening is the
   fix to build next.

**Where next:** notebook 01 (the API on your own data), 02 (theory and
losses), 03 (the problem zoo), 04 (real time), 05 (real days), 06 (kernels
and classification), 07 (partial days), 08 (wind); `RESULTS.md` has every
number.
"""),
]


# ==========================================================================
# 09 -- the algorithm, block by block (tutorial)
# ==========================================================================
TUTORIAL = [
    ("md", r"""
# 09 · Tutorial: the algorithm, block by block

This notebook builds LR-DSR **by hand**, one block at a time, in a few lines
of numpy each, and then shows the one-line package call and the options for
that block. At the end the hand-built loop is checked against the package.

The map we follow (`docs/algorithm/00_algorithm_at_a_glance.png`):
"""),
    ("code", SETUP),
    ("code", r'''
from IPython.display import Image
Image(filename=str(ROOT / "docs" / "algorithm" / "00_algorithm_at_a_glance.png"), width=1100)
'''),
    ("md", r"""
**The objective** every block serves:

$$
\min_{z,\;f}\;\sum_{w} \underbrace{\frac{1}{n}\sum_i \rho\!\left(\frac{y_{wi}-f_{z_w}(x_{wi})}{\hat s}\right)}_{\text{misfit of window } w \text{ under its law}} \;+\; \beta\, C(f_{z_w})
$$

$z_w$ is the label of window $w$, $f_k$ the law of group $k$, $\rho$ the Huber
loss, $\hat s$ one shared noise scale, $C$ the size of a formula.

## The data

Three laws, 240 windows of 16 noisy samples. The labels `z` are kept aside
**for checking only**; no block ever reads them.
"""),
    ("code", r'''
from experiments.functions.data import simulate_function_windows
from lrdsr.core.evaluation import aligned_accuracy

X, y, _, z, equations = simulate_function_windows(
    n_windows=240, window_len=16, noise_std=1.5, geometry_overlap=3.0, seed=11)
W, n, _ = X.shape
K = 3
true_laws = [lambda x: 0.8 * x**2 + 1.5 * x + 0.5, lambda x: 2.5 * np.sin(2 * x) - 0.5,
             lambda x: 0.35 * x**3 - 1.2]
print(f"{W} windows x {n} samples;  true laws: {equations}")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
ax[0].plot(X[:40, :, 0].ravel(), y[:40].ravel(), "o", ms=2.5, color="#777777")
ax[0].set(title="what we get (40 windows, no labels)", xlabel="x", ylabel="y")
viz.plot_laws(truth=true_laws, x_range=(-2.2, 2.2), ax=ax[1]); ax[1].set_title("what we must find")
fig.tight_layout()
'''),
    ("md", r"""
---
## Block 0 · START: windows as law coefficients, then K-means

**Role in the objective:** gives the loop its first labels $z^{(0)}$.
**Role on the data:** turns each window into a vector of *law coefficients*,
so windows are compared by the law that made them, not by how they look.

Four steps, straight from `lrdsr/core/mechanism_space.py`:
1. evaluate the term library at every window's inputs: $B_w$;
2. remove the law all windows share: $r_w = y_w - B_w\beta_{\text{pooled}}$;
3. whiten with the pooled Gram matrix, so noise is equal in every direction;
4. project: $s_w = C_w^\top r_w$, one vector per window, then K-means.
"""),
    ("code", r'''
from sklearn.cluster import KMeans
from lrdsr.core.mechanism_space import library_basis

B = library_basis(X.reshape(-1, 1), feature_names=["x"]).reshape(W, n, -1)   # 1. terms per window
beta_pooled, *_ = np.linalg.lstsq(B.reshape(W * n, -1), y.reshape(-1), rcond=None)
r = y - B @ beta_pooled                                                        # 2. remove the shared law
G = np.einsum("wnp,wnq->pq", B, B) / W                                         # 3. pooled Gram ...
lam, V = np.linalg.eigh(G)
keep = lam > 1e-6 * lam.max()
M = V[:, keep] / np.sqrt(lam[keep])                                            #    ... and whitening
s = np.einsum("wnp,wn->wp", B @ M, r)                                          # 4. one vector per window
s -= s.mean(0)
z0 = KMeans(K, n_init=30, random_state=11).fit_predict(s)
print(f"mechanism space: {s.shape[1]} coordinates per window;  start accuracy {aligned_accuracy(z, z0):.1%}")
'''),
    ("code", r'''
from experiments.common.fitting import window_features
Zf, fnames = window_features(X, y)
P2 = s @ np.linalg.svd(s, full_matrices=False)[2][:2].T
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for k in range(K):
    ax[0].scatter(Zf[z == k, 0], Zf[z == k, 5], s=10, color=viz.regime_color(k), lw=0)
    ax[1].scatter(P2[z == k, 0], P2[z == k, 1], s=10, color=viz.regime_color(k), lw=0)
ax[0].set(title="how the windows LOOK (two summary statistics)", xlabel="mean of y", ylabel="mean of x*y")
ax[1].set(title="their LAW COEFFICIENTS (mechanism space)", xlabel="axis 1", ylabel="axis 2")
fig.tight_layout()
'''),
    ("md", r"""
Coloured by the true law (for checking): mixed on the left, separate on the
right. That is why K-means in mechanism space is a good start.

**In the package:** `GroupedDCSR(init="mechanism")` (the default). The other
starts, compared on these windows, before and after the loop:
"""),
    ("code", r'''
from lrdsr import GroupedDCSR
base = {"n_clusters": K, "alpha_geom": 0.0, "random_state": 11}
rows = []
for init in ("mechanism", "kmeans", "gmm", "fcm"):
    start_only = GroupedDCSR(init=init, max_iter=0, **base).fit(X, y, Zf, feature_names=["x"])
    full = GroupedDCSR(init=init, **base).fit(X, y, Zf, feature_names=["x"])
    rows.append((init, aligned_accuracy(z, start_only.labels), aligned_accuracy(z, full.labels)))
T = pd.DataFrame(rows, columns=["init", "start", "after the loop"]).set_index("init")
ax = T.plot.bar(figsize=(7, 3), rot=0, color=["#b0b0b0", viz.PALETTE[0]])
ax.set(ylim=(0.3, 1.02), ylabel="accuracy", title="block 0 options: the start, and what the loop makes of it");
'''),
    ("md", r"""
The look-alike starts (`kmeans`, `gmm`, `fcm` on window summaries) begin much
worse, and here the loop repairs them completely; on harder data it repairs
most, not all (notebook 00, "an honest note on the loop"). The mechanism
start is already right.

*Try it:* set `geometry_overlap=0.5` in the data cell (summaries now carry
information) and rerun: the look-alike starts improve.

---
## Block 1 · FIT: one law per group

**Role in the objective:** with the labels fixed, choose each $f_k$.
**Role on the data:** pool all samples of all windows in a group and find a
short formula: greedy forward selection over the term library, stopping when
BIC stops falling ($\mathrm{BIC} = N\log(\mathrm{RSS}/N) + p\log N$).
"""),
    ("code", r'''
from lrdsr import FastSymbolicRegressor

def fit_laws(labels):
    """Block 1: one law per group, from that group's pooled samples."""
    return [FastSymbolicRegressor(max_terms=5, feature_names=["x"])
            .fit(X[labels == k].reshape(-1, 1), y[labels == k].ravel()) for k in range(K)]

laws = fit_laws(z0)
for k, f in enumerate(laws):
    print(f"group {k}  ({np.sum(z0 == k)} windows):  y = {f.expression()}")
fig, ax = plt.subplots(1, 3, figsize=(14, 3.2), sharey=True)
xg = np.linspace(-2.2, 2.2, 200)
for k, a in enumerate(ax):
    a.plot(X[z0 == k, :, 0].ravel(), y[z0 == k].ravel(), "o", ms=2, color="#aaaaaa")
    a.plot(xg, laws[k].predict(xg[:, None]), color=viz.regime_color(k), lw=3)
    a.set(title=f"group {k}: pooled windows → one formula", xlabel="x")
fig.tight_layout()
'''),
    ("md", r"""
**In the package:** `backend="fast"` (the default), `backend_kwargs={"max_terms": 5}`.
Options for this block:

| option | what it changes |
|---|---|
| `backend="fast_sin"` (trial) | adds $\sin(a x)$, $\cos(a x)$ with the frequency $a$ **fitted** |
| `mechanism_specs=[...]` | declare a known law, a known part + searched rest, or a known form with fitted constants, per group |
| `superposition=True` | a group's law may be a mix of two other groups' laws |

The vocabulary matters when a law is outside the library. Two fast
oscillations, $\sin 4x$ vs $\sin 4.6x$:
"""),
    ("code", r'''
from experiments.problems import zoo
prob = zoo.PROBLEM_BY_NAME["high_frequency"]
Xh, yh, zh = zoo.make_windows(prob, zoo.sigma_for_rho(prob, 0.5), n_windows=200, window_len=32, seed=11)
Zh, _ = window_features(Xh, yh)
fig, ax = plt.subplots(1, 2, figsize=(12, 3.2), sharey=True)
for a, be in zip(ax, ("fast", "fast_sin")):
    r_ = GroupedDCSR(init="mechanism", backend=be, **{**base, "n_clusters": 2}).fit(Xh, yh, Zh, feature_names=["x"])
    viz.plot_laws(r_.models, x_range=prob.x_range, truth=[lambda x: np.sin(4 * x), lambda x: np.sin(4.6 * x)], ax=a)
    a.get_legend().remove()
    a.set_title(f'backend="{be}":  accuracy {aligned_accuracy(zh, r_.labels):.1%}')
    print(be, [m.expression() for m in r_.models])
fig.tight_layout()
'''),
    ("md", r"""
---
## Block 2 · NOISE SCALE: one shared spread

**Role in the objective:** $\hat s$, the unit residuals are measured in.
**Role on the data:** the robust spread of *all* residuals under the current
labels (median absolute deviation times 1.4826, i.e. in standard-deviation units).
One shared scale, so a window a law explains badly really costs more.
"""),
    ("code", r'''
from lrdsr.core.losses import noise_scale

def shared_scale(laws, labels):
    """Block 2: robust spread of every window's residual under its own law."""
    res = np.concatenate([y[w] - laws[labels[w]].predict(X[w]) for w in range(W)])
    return noise_scale(res), res

s_hat, res0 = shared_scale(laws, z0)
plt.figure(figsize=(6, 2.8))
plt.hist(res0, bins=60, color="#9a9a9a")
for sgn in (-1, 1):
    plt.axvline(sgn * s_hat, color=viz.PALETTE[0], lw=2.5)
plt.title(f"residuals; shared scale s = {s_hat:.2f}  (true noise sd = 1.5)"); plt.xlabel("y - f(x)");
'''),
    ("md", r"""
**In the package:** `residual_scale="global"` (default) or `"per_window"`
(each window's own spread, which hides the *size* of a misfit). With
`loss="learned"` the noise model and its scale are refitted here.

---
## Block 3 · SCORE: every window under every law

**Role in the objective:** the cost matrix
$J_{wk} = \frac1n\sum_i \rho\big((y_{wi}-f_k(x_{wi}))/\hat s\big)$, what window $w$
would pay to belong to law $k$.
**Role on the data:** one detail matters: a law fitted *on* a window flatters
it. So a window's cost under its **own** group's law is computed with a law
fitted **without** it (3-fold cross-fitting), as `score_mode="cross_fit"` does.
"""),
    ("code", r'''
from lrdsr.core.losses import aggregate_window_residual

def cost_matrix(laws, labels, s_hat, folds=3, seed=11):
    """Block 3: Huber cost of every window under every law; own group out-of-fold."""
    J = np.array([[aggregate_window_residual(y[w], f.predict(X[w]), scale=s_hat) for f in laws]
                  for w in range(W)])
    rng = np.random.default_rng(seed)
    for k in range(K):
        members = np.flatnonzero(labels == k)
        for fold in np.array_split(rng.permutation(members), folds):
            rest = np.setdiff1d(members, fold)
            f = FastSymbolicRegressor(max_terms=5).fit(X[rest].reshape(-1, 1), y[rest].ravel())
            for w in fold:
                J[w, k] = aggregate_window_residual(y[w], f.predict(X[w]), scale=s_hat)
    return J

J = cost_matrix(laws, z0, s_hat)
order = np.argsort(z0, kind="stable")
plt.figure(figsize=(4.5, 3.6))
plt.imshow(J[order], aspect="auto", cmap="viridis", interpolation="nearest")
plt.colorbar(label="cost (dark = cheap)")
plt.xticks(range(K), [f"law {k}" for k in range(K)]); plt.ylabel("windows, sorted by group")
plt.title("the cost matrix J[w, k]");
'''),
    ("md", r"""
**In the package:** `loss="huber"` (default, δ = 1.345 noise sd), or
`"squared"`, `"absolute"`, `"cauchy"`, `"tukey"`, `"student_t"`, `"learned"`;
`score_mode="cross_fit"` (default) or `"in_sample"`; plus a complexity weight
`beta_complexity` and an optional geometry weight `alpha_geom` (0 by default).

The loss matters when the noise has outliers. The same windows with
heavy-tailed noise (Student-t, 2 degrees of freedom):
"""),
    ("code", r'''
rng = np.random.default_rng(0)
clean = np.stack([true_laws[k](X[w, :, 0]) for w, k in enumerate(z)])
y_heavy = clean + rng.standard_t(2.0, size=y.shape)
acc = {loss: aligned_accuracy(z, GroupedDCSR(init="mechanism", loss=loss, **base)
                              .fit(X, y_heavy, Zf, feature_names=["x"]).labels)
       for loss in ("squared", "huber", "cauchy", "learned")}
ax = pd.Series(acc).plot.bar(figsize=(6, 2.8), rot=0, color=["#b0b0b0"] + [viz.PALETTE[0]] * 3)
ax.set(ylim=(0.5, 1.02), ylabel="accuracy", title="block 3 option: the loss, under heavy-tailed noise");
'''),
    ("md", r"""
---
## Block 4 · ASSIGN, and the loop

**Role in the objective:** with the laws fixed, $z_w = \arg\min_k J_{wk}$:
every window moves to its cheapest law. Then blocks 1-4 repeat until fewer
than 1% of windows move. That is the whole algorithm: here it is in full,
built from the four functions above.
"""),
    ("code", r'''
def lrdsr_by_hand(z_start, max_rounds=10, tol=0.01):
    labels, history = z_start.copy(), []
    for rnd in range(max_rounds):
        laws = fit_laws(labels)                      # block 1
        s_hat, _ = shared_scale(laws, labels)        # block 2
        J = cost_matrix(laws, labels, s_hat)         # block 3
        new = J.argmin(axis=1)                       # block 4
        moved = np.mean(new != labels)
        history.append((rnd + 1, moved, aligned_accuracy(z, new)))
        labels = new
        if moved < tol:
            break
    return labels, fit_laws(labels), pd.DataFrame(history, columns=["round", "moved", "accuracy"])

# start from a BAD partition on purpose, to watch the loop work
bad = KMeans(K, n_init=10, random_state=0).fit_predict((Zf - Zf.mean(0)) / Zf.std(0))
lab_hand, laws_hand, hist = lrdsr_by_hand(bad)
hist
'''),
    ("code", r'''
fig, ax = plt.subplots(1, 2, figsize=(12, 3.2))
ax[0].plot(hist["round"], 100 * hist.moved, "-o", color=viz.PALETTE[1], label="% of windows moved")
ax[0].plot(hist["round"], 100 * hist.accuracy, "-o", color=viz.PALETTE[0], label="accuracy %")
ax[0].axhline(1, color="k", ls=":", lw=1)
ax[0].set(xlabel="round", title="the loop, from a bad start", xticks=hist["round"])
ax[0].legend()
viz.plot_laws(laws_hand, x_range=(-2.2, 2.2), truth=true_laws, ax=ax[1]); ax[1].set_title("laws found by hand")
fig.tight_layout()
'''),
    ("md", r"""
**The same thing, in the package**, from the same bad start and from the
default mechanism-space start:
"""),
    ("code", r'''
pkg_bad = GroupedDCSR(init=bad, **base).fit(X, y, Zf, feature_names=["x"])
pkg = GroupedDCSR(init="mechanism", **base).fit(X, y, Zf, feature_names=["x"])
print(f"by hand, bad start:        {aligned_accuracy(z, lab_hand):.1%}")
print(f"package, bad start:        {aligned_accuracy(z, pkg_bad.labels):.1%}")
print(f"package, mechanism start:  {aligned_accuracy(z, pkg.labels):.1%}")
print("\nthe package's laws:")
for m in pkg.models:
    print("   y =", m.expression())
'''),
    ("md", r"""
The package does the same four steps and adds two safeguards the hand version
skips: tiny groups are repaired, and each cost term is divided by its median
before weighting, so `alpha_geom` and `beta_complexity` are real trade-offs.

**Other ways to assign** (same idea, different machinery):

| estimator | assignment |
|---|---|
| `SoftLRDSR` | a *probability* per law for every window (EM) |
| `OnlineLRDSR` | one window at a time, in real time; can create new laws |
| `CusumSegmenter` | sample by sample, detects switches in one stream |
| `LawClassifier` | labels known: laws per class, new windows to the best class |

The soft version on our windows: each row is a window, and the colours are
its probabilities.
"""),
    ("code", r'''
from lrdsr import SoftLRDSR
soft = SoftLRDSR(n_clusters=K, feature_names=["x"], random_state=11).fit(X, y)
print(f"soft EM accuracy {aligned_accuracy(z, soft.labels):.1%};  "
      f"{np.mean(soft.responsibilities.max(1) < 0.9):.1%} of windows less than 90% sure")
fig, ax = plt.subplots(figsize=(9, 2.6))
viz.plot_responsibilities(soft.responsibilities, ax=ax);
'''),
    ("md", r"""
---
## Block 5 · REFINE (optional): a deeper law search, once per group

**Role in the objective:** one more "fit laws, labels fixed" step, with a far
larger vocabulary. **Role on the data:** PySR builds formulas from operators
and fits the constants inside them. Too slow for the loop, so it runs once
per final group, and its law **replaces the loop's only if it is cheaper on
held-out windows**, judged on the same cost.

It needs `pip install "pysr<2"` (Julia installs on first use). The cell
below skips itself when PySR is missing.
"""),
    ("code", r'''
import importlib.util
if importlib.util.find_spec("pysr") is None:
    print('PySR not installed: pip install "pysr<2" to run block 5.')
else:
    from lrdsr.core.refine import refine_laws
    from experiments.openlaws.run import PROBLEMS as OPEN
    dp = OPEN[1]                                                  # damped: exp(-0.4x) sin 3x vs exp(-0.1x) sin 3x
    Xd, yd, zd = zoo.make_windows(dp, zoo.sigma_for_rho(dp, 1.0), n_windows=150, window_len=48, seed=23)
    Zd, _ = window_features(Xd, yd)
    loop = GroupedDCSR(init="mechanism", backend="fast_sin", **{**base, "n_clusters": 2}).fit(
        Xd, yd, Zd, feature_names=["x"])
    out = refine_laws(loop, Xd, yd, seed=23)
    for k in range(2):
        print(f"group {k}:  loop law   {loop.models[k].expression()}")
        print(f"          PySR found {out.expressions_found[k]}")
        print(f"          held-out cost  loop {out.cost_before[k]:.4f}   PySR {out.cost_found[k]:.4f}"
              f"   -> {'REPLACED' if out.replaced[k] else 'kept the loop law'}")
    print(f"\naccuracy: loop {aligned_accuracy(zd, loop.labels):.1%} -> after block 5 {aligned_accuracy(zd, out.labels):.1%}")
'''),
    ("md", r"""
---
## The whole thing in three lines

```python
from lrdsr import GroupedDCSR
res = GroupedDCSR(n_clusters=K, alpha_geom=0.0).fit(X_seq, y_seq, Z, feature_names=["x"])
res.labels, [m.expression() for m in res.models]
```

`X_seq` is `(windows, samples, inputs)`, `y_seq` is `(windows, samples)`,
`Z` any window-level features (unused when `alpha_geom=0`; zeros are fine).

| block | parameter | default | options |
|---|---|---|---|
| 0 start | `init` | `"mechanism"` | `"kmeans"`, `"gmm"`, `"bgmm"`, `"fcm"`, an array |
| 1 fit | `backend`, `backend_kwargs` | `"fast"`, 5 terms | `"fast_sin"` (trial), `mechanism_specs`, `superposition` |
| 2 noise | `residual_scale` | `"global"` | `"per_window"` |
| 3 score | `loss`, `score_mode` | `"huber"`, `"cross_fit"` | 5 other losses, `"learned"`; `"in_sample"` |
| 3 score | `beta_complexity`, `alpha_geom` | 0.002 (project setting), 0 | any weight |
| 4 assign | the estimator | `GroupedDCSR` | `SoftLRDSR`, `OnlineLRDSR`, `CusumSegmenter`, `LawClassifier` |
| 4 stop | `max_iter`, `tol` | 10, 0.01 | |
| 5 refine | `refine_laws(res, X, y)` | off | PySR (or any engine) |

**Try it:** change the noise (`noise_std`), the window length (`window_len`)
or the number of laws, and watch which block starts to struggle. Notebook
`00_summary` shows how the whole method compares with the alternatives.
"""),
]


NOTEBOOKS: dict[str, list[tuple[str, str]]] = {
    "00_summary": SUMMARY,
    "01_quickstart": QUICKSTART,
    "02_theory_and_losses": THEORY,
    "03_problem_zoo": ZOO,
    "04_realtime_clustering": REALTIME,
    "05_real_data": REALDATA,
    "06_kernels_and_classification": EXTENSIONS,
    "07_partial_days_and_streams": PARTIAL,
    "08_wind_power_curves": WIND,
    "09_algorithm_tutorial": TUTORIAL,
}
