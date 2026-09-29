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

**The question.** You have many short pieces of data (*windows*): a day of
traffic, six hours of a wind turbine, one heartbeat. Each window was made by
one of a few hidden **rules**, closed-form laws like $y = x^2 + x$. Nobody tells
you which rule made which window. Can you find **both**: the rules, and which
window came from which rule?

**How to read this notebook.** Every figure comes with three short notes:
- **The task**: the question the figure answers;
- **What you see**: how to read it;
- **Take-away**: what it means.

The plotting code lives in `notebooks/summary_figs.py`, so each cell here is
one line. Numbers, seeds and statistical details are in `RESULTS.md`; the
algorithm is built by hand in `09_algorithm_tutorial`.

| part | what you get |
|---|---|
| 1 · Motivation | why group windows by their law, not by how they look |
| 2 · The method | the idea, what laws are built from, the objective |
| 3 · A worked example | the method run live, step by step |
| 4 · What works | the main results |
| 5 · Real data | two years of bike-sharing days, on a calendar |
| 6 · What did not work | three failures, and what they teach |
| 7 · Every task in one picture | all sixteen questions, answered |
"""),
    ("code", r'''
import sys, pathlib
ROOT = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
            if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
%matplotlib inline
from lrdsr import viz
viz.style()
from notebooks import summary_figs as S
'''),
    ("md", r"""
---
# 1 · Motivation: looking alike is not the same law

**The task.** The usual way to group windows is to compute a few statistics
of each one (mean, trend, …) and cluster those. Does that find windows made by
the same rule?

**What you see.**
- **Left:** four windows, as you would get them.
- **Middle:** the truth. Windows 1 and 2 come from a parabola, windows 3 and 4 from a straight line.
- **Right:** the statistics a usual clustering would compare.
"""),
    ("code", "S.motivation()\n"),
    ("md", r"""
**Take-away.** On the right, window 2 sits next to window 3, a *different*
rule, and far from its true partner, window 1. Windows 1 and 2 look nothing
alike because they saw different inputs. The only thing they share is the
**equation** that made them. So this project groups windows by their law, and
returns that law as a readable formula.

---
# 2 · The method

**The idea in one picture.** Windows go in. The method alternates two
questions until the answers stop changing: *what is each law?* and *which
law made each window?* Labels and formulas come out.
"""),
    ("code", 'S.show("00_the_idea.png")\n'),
    ("md", r"""
**What a law is built from.** A law is a short sum of simple **terms**.
- **A · Fixed terms**: the default library.
- **B · Terms with a number fitted inside**, like $\sin(a\,x)$: a trial option.
- **C · Open-ended search** (PySR): optional, once at the end.

The ladder at the bottom goes from "know nothing" to the **oracle**, which
knows every law exactly. The oracle is not a method; it is the yardstick for
the best possible result.
"""),
    ("code", 'S.show("00_vocabulary.png")\n'),
    ("md", r"""
**The objective, in one line.** Choose the labels $z$ and the laws $f$ so that
every window is explained well by its own law, with short formulas preferred:

$$
\min_{z,\,f}\;\sum_{\text{windows } w}\; \text{misfit}\big(\text{window } w,\ \text{law } f_{z_w}\big) \;+\; \text{a small cost per term}
$$

The misfit is measured in units of the noise. It uses a robust (Huber) loss,
so one outlier cannot dominate.

**How it is minimised.** Like K-means, with a *formula* in place of an average:
1. **Start:** describe each window by the coefficients of its best-fitting
   curve, and group those with K-means.
2. **Fit:** for each group, fit one short formula to all its windows together.
3. **Assign:** move every window to the formula that fits it best.
4. **Repeat** 2-3 until almost nothing moves.

**How good can anyone do?** Even the oracle mislabels some windows, because
of noise. Its error is an exact formula,
$P_{\text{err}} = Q\!\left(\sqrt{n\rho}/2\right)$: it depends only on the window
length $n$ and on how far apart the laws are, in noise units ($\rho$). Every
result below is measured against it.

---
# 3 · A worked example, live

**The task.** Three laws, 240 windows of 16 noisy points each. The true
labels are kept aside, for checking only.

**What you see.**
- **Left:** the data as you get it.
- **Middle:** the same, coloured by the hidden law.
- **Right:** the three laws to be found.
"""),
    ("code", "d = S.example()\nS.example_data(d)\n"),
    ("md", r"""
**Take-away.** Pooled together, the windows are a cloud; nothing in it says
"three parabola-like groups".

**The start: looks vs laws.**
- **What you see:** the same windows, coloured by their true law.
  - **Left:** placed by how they look (two summary statistics).
  - **Right:** placed by their law coefficients (*mechanism space*).
- **Take-away:** on the left the colours are mixed, and no clustering can
  separate them. On the right they form three separate groups, which K-means
  finds easily. This is why the method starts in mechanism space.
"""),
    ("code", "S.looks_vs_laws(d)\n"),
    ("md", r"""
**The loop at work.**
- **The task:** to watch the loop, start it from a deliberately bad grouping.
- **What you see:** the laws fitted after 0, 1 and 2 rounds and at the end
  (solid), against the truth (dashed), with the accuracy of the grouping.
- **Take-away:** each round, the laws improve because the groups improve, and
  the groups improve because the laws improve. Here two rounds are enough.
"""),
    ("code", "S.loop_rounds(d)\n"),
    ("md", r"""
**The real fit** (the default settings: good start, then the loop).

**What you see.**
- **The printed laws.**
- **Left:** the laws found against the truth.
- **Second:** how one law is built. Each round tries every term (grey dots)
  and keeps the one that helps most, until none helps.
- **Third:** true law against found group. A clean diagonal means every
  window is labelled right.
- **Right:** the cost of every window under every law. The dark diagonal
  blocks mean each window is cheap only under its own law.

**Take-away.** Every window is labelled correctly. One law comes back as a
*look-alike*: the parabola is written with $\cos x$. On this range it is the
same curve, but it is not the same formula; that is the price of a fixed
term library.
"""),
    ("code", "S.the_fit(d)\n"),
    ("md", r"""
**The soft version** gives each window a *probability* for each law, rather
than one label.

**What you see.**
- **Left:** the laws it finds.
- **Right:** one column per window, coloured by its probabilities. A column of
  one solid colour means the method is sure.

**Take-away.** Almost every window is certain; the few mixed columns are the
genuinely ambiguous windows.
"""),
    ("code", "S.soft_version(d)\n"),
    ("md", r"""
**How close to the best possible?**
- **The task:** make the problem harder by adding noise, and compare three
  things: the oracle (knows the laws), the method (knows nothing), and the
  usual look-alike clustering.
- **What you see:** accuracy at four noise levels.
- **Take-away:** the method stays with the oracle at every noise level. The
  look-alike clustering falls towards chance.
"""),
    ("code", "S.noise_sweep()\n"),
    ("md", r"""
**What is the loop for? An honest note.**
- **The task:** hand the loop starting groupings of different quality, and
  measure what it returns.
- **What you see:**
  - **Left:** how much the loop repairs, against how broken the start was.
  - **Middle:** error in against error out.
  - **Right:** the usual starting methods, before and after the loop.
- **Take-away:** the loop returns about the same answer whatever it is given.
  It **repairs bad starts** a lot, but it does not improve a good one (right,
  last pair). The *start* does most of the work; the loop makes the method
  robust, and produces the formulas.
"""),
    ("code", 'S.result("loop_value")\n'),
    ("md", r"""
---
# 4 · What works

All figures in this part are read from the committed results.

### The best possible error is reachable without labels

- **The task:** is the formula for the best possible error right, and can the
  method reach it without labels?
- **What you see:**
  - **Left:** the method's start (K-means in mechanism space) against the exact
    best possible error, in many settings. The points lie on the diagonal.
  - **Right:** the same, when all laws share a large common part. The result
    does not change.
- **Take-away:** without any labels, the method reaches the best possible
  error, and a part that all laws share does not disturb it.
"""),
    ("code", 'S.result("attainability")\n'),
    ("md", r"""
### Outliers in the noise

- **The task:** real noise has outliers. Which way of measuring misfit keeps
  the method near the best possible?
- **What you see:** the error under four kinds of noise, from normal (left) to
  10% gross outliers (right), for each loss. Black is the best possible.
- **Take-away:** with normal noise, every loss is fine. With heavy tails, the
  squared loss (and the normal-noise soft version, pink) fail badly, while
  Huber and the **learned** loss (which fits the noise shape as it goes) stay
  at the best possible.
"""),
    ("code", 'S.result("loss_estimator")\n'),
    ("md", r"""
### Real time: when does a switch get noticed?

- **The task:** in one long stream that switches between laws, how long after
  a switch does the method notice it, and how often does it raise false alarms?
- **What you see:**
  - **Left:** delay after a switch, measured (dots) against the prediction (lines).
  - **Right:** time between false alarms, which always stays above the
    guaranteed bound (dashed).
- **Take-away:** the delay is predicted by a formula before running anything,
  and false alarms are as rare as promised.
"""),
    ("code", 'S.result("online_sequential")\n'),
    ("md", r"""
### Days with missing hours

- **The task:** many real traffic days have sensor gaps. The usual fix is to
  fill in the missing hours, then classify. Can the law do better, using only
  the hours observed?
- **What you see:** the share of days labelled correctly, against how many
  hours were observed.
  - **Left:** real days with gaps.
  - **Right:** complete days with real gap patterns cut out.
  - **Blue:** the law; **orange and black:** two ways of filling in.
- **Take-away:** with long gaps (6-11 of 24 hours observed), the law labels
  every real day correctly (99% in the controlled version), while filling in
  gets many wrong. With few hours missing, all
  methods agree.
"""),
    ("code", 'S.result("realdata_gaps")\n'),
    ("md", r"""
### A new law appears in a stream

- **The task:** a law that was never seen before starts appearing in a stream.
  Does the real-time version notice, and create a new group for it?
- **What you see:** the share of streams in which the new law got its own group.
  - **Blue:** with a flexible (kernel) basis.
  - **Orange:** with the fixed term library.
  - **Dotted:** what an ideal test could do.
- **Take-away:** with a flexible basis the new law is found; with the fixed
  library it is mostly missed, because the library cannot describe it.
"""),
    ("code", 'S.result("online_kernel_birth")\n'),
    ("md", r"""
### Classification: how many labelled examples are enough?

- **The task:** when some labelled windows exist, how does the error fall as
  more are added, and can that be predicted?
- **What you see:** classification error against the number of labelled
  windows per class. Markers are measured; lines are the formula; grey is the
  best possible. Each line style is a different law size.
- **Take-away:** the formula predicts the whole learning curve. It depends on
  how many coefficients a law has, not on the window length.
"""),
    ("code", 'S.result("v10_learning_curve")\n'),
    ("md", r"""
### Wind turbines: every window sees different inputs

This is the method's home ground. Each 6-hour block of a turbine sees
different wind speeds, so there is no common "profile" to compare.

- **The task:** can the law tell cold air from warm, and a turbine in another
  turbine's wake from one in free wind, on turbines never trained on?
- **What you see:**
  - **Top:** error per question (lower is better). Blue is the law; the
    others are the industry's binned-curve method; red dashes are the best
    possible. "Night vs day" is a control that should show nothing.
  - **Bottom:** how much more power a turbine makes in cold air than in warm
    air, at each wind speed. The grey line is the physics prediction (air
    density).
- **Take-away:** the law beats the industry method on both real questions,
  while on the control every method stays near chance, as it should. The measured power ratio settles onto the
  physics prediction, so the laws found are physically meaningful.
"""),
    ("code", 'S.result("wind_transfer")\nS.result("wind_physics")\n'),
    ("md", r"""
### Is this better than plain symbolic regression?

- **The task:** compare with the same formula search used the usual way:
  either one formula for all windows, or one formula per window, clustered
  afterwards.
- **What you see:**
  - **Left:** share of windows labelled wrong.
  - **Right:** how far the formulas found are from the true ones.
  - Both are shown against how far apart the laws are.
- **Take-away:** one formula for everything is at chance. One formula per
  window is much worse than LR-DSR when the data are noisy, because 16 noisy
  points are too few to pin down a formula. Pooling the windows of a group is
  what makes the laws accurate.
"""),
    ("code", "S.sr_baseline()\n"),
    ("md", r"""
---
# 5 · Real data: two years of bike sharing

**The task.** Washington DC bike rentals, one window per day, with the hour
of the day as input. The method gets no labels and is asked for two laws.
Afterwards we compare with the calendar (working day or not). The calendar is
a proxy, not the truth.
"""),
    ("code", "bk = S.bike()\nS.bike_days(bk)\n"),
    ("md", r"""
**What you see.**
- **Left:** every day, coloured by the law the method found. The two shapes
  are a commuting day (morning and evening peaks) and a leisure day (one
  afternoon hump).
- **Right:** how sure the method is; almost every day is certain.

**Take-away.** The two laws are exactly the two kinds of days, found without
being told what a weekend is.
"""),
    ("code", "S.bike_calendar(bk)\n"),
    ("md", r"""
**What you see.** The same labels on a calendar: weeks across, weekdays down.
Crosses mark days where the law and the calendar disagree.

**Take-away.** Weekends and big holidays are found. The disagreements are of
two kinds:
- **official holidays most people still work** (Columbus Day, Veterans Day):
  the law says *commuting*;
- **days off the calendar does not know about** (the Friday after
  Thanksgiving, Christmas Eve): the law says *leisure*.

In both cases the law is arguably right, and the calendar wrong.

---
# 6 · What did not work

### Failure 1: a law the library cannot write, and a trial fix

**The task.** Two fast oscillations, $\sin 4x$ and $\sin 4.6x$. The library's
fastest term is $\sin 2x$, so it cannot write them. The trial fix adds terms
$\sin(a\,x)$, $\cos(a\,x)$ with the number $a$ fitted. Every method runs on the
same windows, at two noise levels.

**What you see.**
- **Top:** the true laws; what LR-DSR finds with the standard library
  (look-alikes); what it finds with the fitted frequency (the true laws).
- **Middle:** accuracy of every method, on noisy and on clean windows. Black
  is the best possible; blue is this project's methods; orange is the
  signal-processing standard (periodogram); green is plain symbolic regression
  per window; grey is clustering the raw data.
- **Bottom:** why the standard methods fail in noise. A single window's
  spectrum is broad and its peak wanders; the method fits one frequency to all
  of a group's windows at once (blue lines).
"""),
    ("code", "S.fast_oscillations()\n"),
    ("md", r"""
**Take-away.**
- With the standard library, the method is clearly behind the best possible.
- With the fitted frequency, it reaches the best possible, even in noise.
- On clean data the standard methods also solve it; in noise only the methods
  that **pool windows** get close.

**Does the extra freedom hurt the problems that did not need it?**
- **What you see:** all twelve test problems, standard library against
  fitted frequency.
- **Take-away:** only the fast oscillation changes; nothing else gets worse.
"""),
    ("code", "S.fitted_frequency_zoo()\n"),
    ("md", r"""
### Failure 2: against strong classifiers on 24 standard datasets

- **The task:** on 24 public classification benchmarks, compare the best law
  classifier with the best standard classifier on the raw series.
- **What you see:** one point per dataset. Horizontal: the law's error;
  vertical: the raw classifier's error. Points **below** the diagonal are
  datasets where the raw classifier wins.
- **Take-away:** most points are below or on the diagonal. When every series
  has the same complete inputs, a well-tuned standard classifier is hard to
  beat. The method's edge is where inputs differ between windows (gaps, wind),
  not here.
"""),
    ("code", 'S.result("classify_benchmark")\n'),
    ("md", r"""
### Failure 3: noise that wanders over time

**The task.** The method assumes the noise is independent from one sample to
the next, like fresh coin flips. Real noise often wanders: once it is above
the curve, it stays above for a while. How much does the method lose when
that assumption breaks?

**What you see.** The same law and the same noise size, with the noise
wandering more from left to right.
"""),
    ("code", "S.correlated_noise_intuition()\n"),
    ("md", r"""
**What you see next.** Accuracy as the noise wanders more.
- **Black:** the best possible *for this kind of rule*.
- **Blue:** the method.
- **Red area:** the gap between them.
- **Dashed:** a rule that knows how the noise wanders.
- **Green band:** how much the noise wanders on real days.
"""),
    ("code", 'S.result("robustness_failure")\n'),
    ("md", r"""
**Take-away.**
- At the level of wandering seen on real days, the method loses very little.
- With strongly wandering noise it falls clearly behind.
- The dashed curve shows that a rule aware of the wandering would do much
  better. That is the natural next improvement: "whiten" the noise inside the
  loop.

### The other negatives, kept in view

- The number of laws must be given. Choosing it automatically (by BIC) fails
  on real days.
- The formula for a partial day's error ranks which hours matter correctly,
  but its absolute level is off by a factor of a few.
- Scattered missing samples give no advantage; only *long* gaps do.
- Wind: one of two predictions made before the run failed. Clustering wind
  blocks without labels matches no physical cause.

---
# 7 · Every task in one picture

Sixteen questions, one panel each. **The frame colour is the answer**:
**green** = yes, **amber** = partly. Blue bars are this method, grey bars the
alternative (or the situation before), and black bars the best possible or a
prediction. Lower is better unless the panel says otherwise.

| # | the question | what the bars compare |
|---|---|---|
| 1 | Is there a best possible error, as a formula? | share of settings where formula and measurement agree |
| 2 | Can it be reached without labels? | best possible vs our start vs LR-DSR vs look-alike clustering |
| 3 | What does the loop add? | error of a starting grouping, before and after the loop |
| 4 | Does it work on twelve hard problems? | average distance from the best possible, per method |
| 5 | Does it survive outliers? | best possible vs robust losses vs the squared loss |
| 6 | Are switches seen in real time, as predicted? | predicted vs measured delay |
| 7 | Real days: weekday vs weekend? | agreement with the calendar |
| 8 | Days with long sensor gaps? | the law vs filling in the gaps |
| 9 | Can a partial day's error be predicted? | predicted vs measured (timing right, level off) |
| 10 | A law the library cannot write? | fixed library vs flexible kernel basis |
| 11 | Can the classification error be predicted? | share of settings where formula and measurement agree |
| 12 | 24 standard classification datasets? | law vs nearest raw series vs best tuned classifier |
| 13 | Repair: series shifted in time? | law, law with a time shift, the time-warping standard |
| 14 | Streaming days with gaps? | accuracy on partial vs complete days |
| 15 | Wind: cold vs warm air? | best possible vs law vs the industry method |
| 16 | Better than plain symbolic regression? | best possible vs LR-DSR vs SR per window vs one SR law |
"""),
    ("code", "S.scorecard()\n"),
    ("md", r"""
## Take-aways

1. **Grouping windows by their law is a well-posed problem, with a known best
   possible error, and the method reaches it without labels.** The start does
   most of the work; the loop makes it robust and produces readable laws.
2. **It wins where the inputs differ from window to window:** days with long
   sensor gaps, wind turbines at different wind speeds, a new law in a stream,
   noisy oscillations. When every window has the same complete inputs, a good
   standard method ties it or beats it.
3. **It beats plain symbolic regression.** One formula per window is too noisy
   and one formula for everything is wrong; pooling a group's windows is what
   makes the laws accurate.
4. **Its limits are known:**
   - the library (a law it cannot write costs accuracy, and a fitted-number
     term or a kernel basis fixes that);
   - noise that wanders over time;
   - the number of laws must be given.

**Where next:**
- `09_algorithm_tutorial`: every block by hand, with all options.
- `01_quickstart`: the API on your own data.
- Notebooks 02-08 go deeper into each part.
- `RESULTS.md` has every number.
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

**The idea in one picture** (`docs/algorithm/00_the_idea.png`): windows go
in; two questions alternate ("what is each law?" and "which law made each
window?"); labels and formulas come out.
"""),
    ("code", SETUP),
    ("code", r'''
from IPython.display import Image
DOCS = ROOT / "docs" / "algorithm"
Image(filename=str(DOCS / "00_the_idea.png"), width=1100)
'''),
    ("md", r"""
---
## The vocabulary: what a law can be built from

A law is never searched among *all* functions. It is a short sum of
**terms**, and the list of allowed terms decides which laws can be written
down exactly. Three levels of vocabulary:
"""),
    ("code", r'''
Image(filename=str(DOCS / "00_vocabulary.png"), width=1100)
'''),
    ("md", r"""
**A · Fixed terms (the default).** This is the actual list, read from the
code. The search picks up to 5 terms and fits one number in front of each:
"""),
    ("code", r'''
from lrdsr.core.soft import library_terms
xg = np.linspace(-2.2, 2.2, 300)
Phi, names = library_terms(xg[:, None], feature_names=["x"])
print("the default library:", ", ".join(names))
print("with two inputs x1, x2: the same terms for each, plus the product x1*x2")
fig, ax = plt.subplots(1, len(names), figsize=(15, 2.0), sharex=True)
for a, nm, col in zip(ax, names, Phi.T):
    a.plot(xg, col, color=viz.PALETTE[0], lw=2); a.set_title(nm, fontsize=9); a.set(xticks=[], yticks=[])
fig.tight_layout()
'''),
    ("md", r"""
**B · Terms with a fitted inner number.** A term like $\sin(a\,x)$ has a
number *inside* it, so one term stands for a whole family of shapes, and the
search fits $a$ as well. `sin(a·x)` and `cos(a·x)` are the only ones
implemented (the `fast_sin` trial). They are **one example** of the idea,
chosen because oscillations were the zoo's failure; $e^{b x}$, $x^c$ or
$1/(1+c x^2)$ would work the same way, but are not implemented.

**C · Open-ended.** PySR builds formulas from operators, with no list at all.
It runs only in the optional block 5, once per group (end of this notebook).

**How much you know about the laws** is the same ladder seen from the other
side. The more you declare, the less the data must tell you:

| you know… | how you say it | what is fitted |
|---|---|---|
| nothing beyond the vocabulary | default (`mechanism_specs=None`) | which terms + their numbers |
| the **form** of a law, e.g. $a\,e^{-b x}$ | `{"mode": "factory", "factory": lambda: ParametricPriorRegressor(form, p0)}` | only $a$, $b$ |
| part of a law | `{"mode": "partial", "prior_function": h}` | the rest, from the library |
| the law exactly | `{"mode": "known", "function": f}` | nothing |
| **every** law exactly | the **oracle** | nothing: it is the yardstick, not a method |

The **oracle** used throughout the project is just the top of this ladder:
it knows every law exactly and labels each window by the smallest residual.
No method can beat it, so every result is measured against it.

---
## The detailed map

Every block, its piece of the objective, what it does to the data, and its
options (`docs/algorithm/00_algorithm_at_a_glance.png`). The rest of this
notebook builds each column by hand.
"""),
    ("code", r'''
Image(filename=str(DOCS / "00_algorithm_at_a_glance.png"), width=1100)
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
## Block 2 · NOISE SCALE: how big is the noise?

**Role in the objective:** $\hat s$, the ruler misfits are measured with:
the objective scores $(y - f(x))/\hat s$, a misfit *in units of the noise*.
**Role on the data:** take every window's misfit under its current law,
$y - f_{z}(x)$, and measure its typical size robustly:
$\hat s = 1.48 \times \mathrm{median}\,|y - f_z(x)|$. The median ignores
outliers; the factor 1.48 turns it into a standard deviation (for Gaussian
noise, the median absolute value is 0.674 standard deviations, and
1/0.674 = 1.48). One shared ruler for all windows, so a window a law explains
badly really costs more.
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
