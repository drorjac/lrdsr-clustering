"""the functions block -- three classification *shapes*, in pure function space.

The CML sections pose three problems on a physical simulator and on real
links. When one of them fails, the failure has two possible causes and the
CML setting cannot tell them apart: the **method** cannot solve a problem of
that shape, or the **instance** is unsolvable because the two laws are too
close at the noise the physics imposes.

This rebuilds the same three shapes out of clean functions, where the
separation is a **knob** rather than a consequence of ITU coefficients.
Nothing here imports ``lrdsr.cml``: there is no rain, no attenuation and no
link length, only a covariate ``s`` that plays the structural role length
plays. Each experiment sweeps its knob, so the answer is not "it works" or
"it fails" but a curve -- *here is where the method stops working*.

====  =====================================================================
F1    **two competing parametric families**, one nested in the other:
      ``y = a x^b s`` against ``y = a x^b s + c x^b``. The extra term does
      not scale with ``s``, so its share falls like ``1/s`` -- structurally
      a wet antenna against a path term. Knob: ``c/a``. Carries its own
      control, the best possible threshold on ``s``, which needs no model.
F2    **null against signal, with a heterogeneous null.** One regime is
      pure noise, one carries the target, and up to three more carry
      *different* functions that still count as null. Does a
      mechanism-based objective lose when asked to merge unlike mechanisms
      into one class? Knob: how many disturbances.
F3    **multi-class with the two hard pairs explicit.** A base law, a
      rescaled copy (same shape, different amplitude -- the rain/wet-snow
      relationship), a genuinely different shape, and a lagged variant
      (same driver, delayed response -- the wet-antenna relationship).
      Knob: the rescaling factor.
====  =====================================================================

Carried back from the tag ``exploration-archive`` and brought up to the
project's current protocol: reporting seeds {11, 23, 42} (the archived run
used the tuning seeds), mechanism-space initialisation, and a
mechanism-space K-means arm so the scenarios speak to \\S3 as well.

Writes ``results/functions/f{1,2,3}_*.csv``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from experiments.common.fitting import window_features
from lrdsr import paths
from lrdsr.core.backends import ParametricPriorRegressor
from lrdsr.core.baselines import geometry_baselines
from lrdsr.core.evaluation import clustering_metrics
from lrdsr.core.mechanism_space import mechanism_init
from lrdsr.core.model import GroupedDCSR

RESULTS = paths.results_dir("functions", figs=False)
REPORT_SEEDS = (11, 23, 42)
WINDOW_LEN = 32


# ==========================================================================
# shared
# ==========================================================================
def _fit(X_seq, y_seq, Z, K, seed, specs=None, alpha_geom=0.25,
         order_key=None):
    """One LR-DSR fit with the project's current defaults."""
    model = GroupedDCSR(
        n_clusters=K, alpha_geom=alpha_geom, beta_complexity=0.002,
        max_iter=10, tol=0.01, backend="fast",
        backend_kwargs={"max_terms": 5}, random_state=seed,
        min_windows_per_cluster=4, init="mechanism",
        geom_metric="mahalanobis", score_mode="cross_fit",
        residual_scale="global", mechanism_specs=specs,
    )
    return model.fit(X_seq, y_seq, Z,
                     feature_names=["x", "s"] if X_seq.shape[-1] > 1 else ["x"],
                     slot_order_key=(order_key if specs is not None else None))


def _rho(delta: np.ndarray, noise_std: float) -> float:
    """rho = E[(f_j - f_k)^2] / sigma^2 for a realised difference series."""
    return float(np.mean(np.asarray(delta, float) ** 2)) / max(noise_std ** 2, 1e-12)


def _binary(truth: np.ndarray, pred: np.ndarray) -> dict:
    """Accuracy, precision, recall and MCC for a binary prediction."""
    t, p = np.asarray(truth, bool), np.asarray(pred, bool)
    tp, tn = int((p & t).sum()), int((~p & ~t).sum())
    fp, fn = int((p & ~t).sum()), int((~p & t).sum())
    n = tp + tn + fp + fn
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return {"accuracy": (tp + tn) / max(n, 1),
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "mcc": (tp * tn - fp * fn) / den if den > 0 else 0.0}


def _to_binary(labels: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Orient a 2-cluster partition so cluster 1 is the signal-richer one."""
    labels = np.asarray(labels, int)
    a = truth[labels == 1].mean() if (labels == 1).any() else 0.0
    b = truth[labels == 0].mean() if (labels == 0).any() else 0.0
    return (labels == (1 if a >= b else 0))


def _align(true_labels, pred_labels) -> dict:
    """Hungarian mapping pred -> true."""
    k = max(int(np.max(true_labels)), int(np.max(pred_labels))) + 1
    conf = np.zeros((k, k), int)
    for t, p in zip(true_labels, pred_labels, strict=True):
        conf[int(t), int(p)] += 1
    row, col = linear_sum_assignment(-conf)
    return {int(c): int(r) for r, c in zip(row, col, strict=True)}


# ==========================================================================
# F1 -- two competing parametric families
# ==========================================================================
F1_A, F1_B = 0.4, 1.1
F1_RATIOS = (0.25, 0.5, 1.0, 2.0, 4.0)
F1_S = (0.5, 1, 2, 4, 8, 16)


def simulate_f1(n_per_group=120, window_len=WINDOW_LEN, s_values=F1_S,
                a=F1_A, b=F1_B, c_over_a=1.0, noise_std=0.15,
                share_threshold=0.15, x_range=(0.2, 5.0),
                assignment="per_group", seed=7):
    """Windows from the two nested families, with the covariate ``s``.

    ``assignment`` decides *who* carries the extra term, and it changes what
    the experiment measures:

    ``per_group``
        the extra term belongs to a whole ``s`` group, as a wet antenna
        belongs to a whole link. The truth is then largely determined by
        ``s``, so a covariate threshold is a strong competitor -- which is
        exactly the situation the CML classification finds itself in.
    ``per_window``
        the extra term is in a random half of the windows, independently of
        ``s``. No threshold on ``s`` can help, so the only way to recover the
        truth is to compare the two laws. The clean test of whether the
        method does parametric model selection at all.

    ``truth`` is 1 where the extra term is present *and* carries at least
    ``share_threshold`` of the window's signal.
    """
    rng = np.random.default_rng(seed)
    c = c_over_a * a
    if assignment not in ("per_group", "per_window"):
        raise ValueError("assignment must be 'per_group' or 'per_window'")
    group_has = {s: bool(rng.uniform() < 0.5) for s in s_values}

    S, Xs, Ys, truth, rhos = [], [], [], [], []
    for s in s_values:
        for _ in range(n_per_group):
            x = rng.uniform(*x_range, size=window_len)
            base, extra = a * x ** b * s, c * x ** b
            share = float(np.mean(np.abs(extra))
                          / max(np.mean(np.abs(base) + np.abs(extra)), 1e-12))
            has = (group_has[s] if assignment == "per_group"
                   else bool(rng.uniform() < 0.5))
            Ys.append(base + (extra if has else 0.0)
                      + rng.normal(0.0, noise_std, window_len))
            Xs.append(np.column_stack([x, np.full(window_len, s)]))
            S.append(s)
            truth.append(int(has and share > share_threshold))
            rhos.append(_rho(extra, noise_std))
    o = rng.permutation(len(S))
    return (np.asarray(Xs)[o], np.asarray(Ys)[o],
            np.asarray(S, float)[o], np.asarray(truth)[o], np.asarray(rhos)[o])


def f1_specs(a: float = F1_A, b: float = F1_B):
    """The two competing hypotheses as declared mechanism slots.

    This is the ``known_form`` framing of the knowledge ladder: the FORM of
    each law is declared, its coefficients are not, and the bounds keep each
    fit inside its own family so one slot cannot absorb the other.
    """
    def pure(X, theta):
        return theta[0] * np.maximum(X[:, 0], 1e-9) ** theta[1] * X[:, 1]

    def modulated(X, theta):
        x, s = np.maximum(X[:, 0], 1e-9), X[:, 1]
        return theta[0] * x ** theta[1] * s * (1.0 + theta[2] / np.maximum(s, 1e-9))

    return [
        {"mode": "factory", "expression": "a x^b s",
         "factory": lambda: ParametricPriorRegressor(
             prior_form=pure, p0=[a, b],
             bounds=([0.25 * a, 0.6 * b], [4 * a, 1.6 * b]),
             expression_template="{0:.4g}*x^{1:.3f}*s")},
        {"mode": "factory", "expression": "a x^b s (1 + c/s)",
         "factory": lambda: ParametricPriorRegressor(
             prior_form=modulated, p0=[a, b, 0.5],
             bounds=([0.25 * a, 0.6 * b, 0.01], [4 * a, 1.6 * b, 20.0]),
             expression_template="{0:.4g}*x^{1:.3f}*s*(1+{2:.3f}/s)")},
    ]


def run_f1(seeds=REPORT_SEEDS) -> pd.DataFrame:
    """Does the method beat the covariate, when it has to compare two laws?"""
    rows = []
    for assignment in ("per_group", "per_window"):
        for c_over_a in F1_RATIOS:
            for seed in seeds:
                X, y, s, truth, rho = simulate_f1(
                    c_over_a=c_over_a, seed=seed, assignment=assignment)
                if len(np.unique(truth)) < 2:
                    continue
                Z, _ = window_features(X, y)
                base = {"assignment": assignment, "c_over_a": c_over_a,
                        "seed": seed, "median_rho": float(np.median(rho))}

                res = _fit(X, y, Z, 2, seed, specs=f1_specs(),
                           alpha_geom=0.0, order_key=y.mean(axis=1))
                f1m = clustering_metrics(truth, res.labels)
                rows.append({**base, "method": "LR-DSR (form declared)", **f1m})

                lab = mechanism_init(X, y, 2, seed=seed)
                rows.append({**base, "method": "K-means in mechanism space",
                             **clustering_metrics(truth, lab)})

                for bname, lab in geometry_baselines(Z, 2, seed=seed).items():
                    rows.append({**base, "method": bname,
                                 **clustering_metrics(truth, lab)})

                # the control the CML version carries: the best achievable
                # threshold on the covariate the modulation depends on.
                best, best_t = -np.inf, None
                for t in np.unique(s):
                    v = clustering_metrics(truth, (s < t).astype(int))["ARI"]
                    if v > best:
                        best, best_t = v, t
                rows.append({**base, "method": "best threshold on s",
                             "threshold": float(best_t),
                             **clustering_metrics(truth, (s < best_t).astype(int))})
                print(f"  [F1] {assignment:11s} c/a {c_over_a:4.2f} seed {seed:3d}"
                      f"  LR-DSR {f1m['ARI']:.3f}  threshold {best:.3f}",
                      flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "f1_raw.csv", index=False)
    summary = (df.groupby(["assignment", "c_over_a", "method"])
                 [["ARI", "NMI", "median_rho"]].mean().reset_index())
    summary.to_csv(RESULTS / "f1_summary.csv", index=False)

    piv = summary.pivot_table(index=["assignment", "c_over_a"],
                              columns="method", values="ARI").reset_index()
    piv["gain_over_threshold"] = (piv["LR-DSR (form declared)"]
                                  - piv["best threshold on s"])
    piv["verdict"] = np.where(
        piv.gain_over_threshold > 0.05,
        "the law comparison adds information beyond the covariate",
        "no better than thresholding the covariate")
    piv.to_csv(RESULTS / "f1_verdict.csv", index=False)
    print("\n[F1]"); print(piv.round(3).to_string(index=False))
    return df


# ==========================================================================
# F2 -- null against signal, with a heterogeneous null
# ==========================================================================
#: The F2 regimes: name, a factory taking the amplitude, whether it counts
#: as signal, how it prints, and why it is in the pool. The simulator and
#: the generated documentation read the same tuple, so they cannot drift.
F2_REGIMES = (
    ("null", lambda a: (lambda x: np.zeros_like(x)), 0, "0",
     "nothing happening"),
    ("signal", lambda a: (lambda x: a * (0.8 * x ** 2 + 1.5 * x)), 1,
     "a (0.8 x^2 + 1.5 x)", "the thing to be detected"),
    ("step", lambda a: (lambda x: a * 1.2 * np.sign(x)), 0, "1.2 a sign(x)",
     "a level change: unlike the null and unlike the signal"),
    ("oscillation", lambda a: (lambda x: a * 1.4 * np.sin(3.0 * x)), 0,
     "1.4 a sin(3x)", "structure at a frequency neither class has"),
    ("drift", lambda a: (lambda x: a * 0.9 * x), 0, "0.9 a x",
     "a linear trend, the most common real contaminant"),
)


def simulate_f2(n_per_regime=120, window_len=WINDOW_LEN, n_disturbances=2,
                amplitude=1.0, noise_std=0.25, x_range=(-2.0, 2.0), seed=7):
    """Null / signal / disturbances, where disturbances count as null.

    The disturbance functions are deliberately *unlike* both the null and
    the signal: that is the point, because it forces a K=2 solution to merge
    mechanisms a mechanism-based objective wants to keep apart.
    """
    rng = np.random.default_rng(seed)
    laws = [(n, f(amplitude), c) for n, f, c, _, _ in F2_REGIMES[:2]]
    laws += [(n, f(amplitude), c)
             for n, f, c, _, _ in F2_REGIMES[2:2 + n_disturbances]]

    Xs, Ys, truth = [], [], []
    for _, f, is_signal in laws:
        for _ in range(n_per_regime):
            x = np.sort(rng.uniform(*x_range, size=window_len))
            Ys.append(f(x) + rng.normal(0.0, noise_std, window_len))
            Xs.append(x[:, None])
            truth.append(is_signal)
    o = rng.permutation(len(truth))
    return (np.asarray(Xs)[o], np.asarray(Ys)[o], np.asarray(truth)[o],
            [n for n, _, _ in laws])


#: The disturbance count alone leaves LR-DSR at MCC 1.0 everywhere -- the
#: framing question answers "no" too easily at one noise level. Noise is
#: swept with it so the scenario produces a curve rather than a ceiling.
F2_NOISE = (0.25, 1.0, 2.0)


def run_f2(seeds=REPORT_SEEDS) -> pd.DataFrame:
    """Is the failure the method, or the K=2 framing it was given?"""
    rows = []
    for noise in F2_NOISE:
      for n_dist in (0, 1, 2, 3):
        for seed in seeds:
            X, y, truth, names = simulate_f2(n_disturbances=n_dist, seed=seed,
                                             noise_std=noise)
            Z, _ = window_features(X, y)
            K_true = len(names)
            base = {"noise_std": noise, "n_disturbances": n_dist, "seed": seed}

            res = _fit(X, y, Z, 2, seed)
            k2 = _binary(truth, _to_binary(res.labels, truth))
            rows.append({**base, "method": "LR-DSR (K=2)", **k2})

            if K_true > 2:
                rk = _fit(X, y, Z, K_true, seed)
                merged = np.zeros(len(truth), bool)
                for k in np.unique(rk.labels):
                    m = rk.labels == k
                    if m.any() and truth[m].mean() > 0.5:
                        merged |= m
                rows.append({**base,
                             "method": f"LR-DSR (K={K_true}, then merged)",
                             **_binary(truth, merged)})

            lab = mechanism_init(X, y, 2, seed=seed)
            rows.append({**base, "method": "K-means in mechanism space",
                         **_binary(truth, _to_binary(lab, truth))})

            for bname, lab in geometry_baselines(Z, 2, seed=seed).items():
                rows.append({**base, "method": bname,
                             **_binary(truth, _to_binary(lab, truth))})
            print(f"  [F2] noise {noise:4.2f} disturbances {n_dist} "
                  f"seed {seed:3d}  LR-DSR(K=2) MCC {k2['mcc']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "f2_raw.csv", index=False)
    summary = (df.groupby(["noise_std", "n_disturbances", "method"])
                 [["accuracy", "precision", "recall", "mcc"]].mean()
                 .reset_index()
                 .sort_values(["noise_std", "n_disturbances", "mcc"],
                              ascending=[True, True, False]))
    summary.to_csv(RESULTS / "f2_summary.csv", index=False)
    print("\n[F2]")
    print(summary.round(3).to_string(index=False))
    return df


# ==========================================================================
# F3 -- multi-class, with the two hard pairs made explicit
# ==========================================================================
F3_SCALES = (0.2, 0.4, 0.6, 0.8)


#: The F3 regimes, in the order the simulator builds them: how each prints
#: and what it stands for. `base`/`scaled` are the hard pair.
F3_REGIMES = (
    ("base", "0.8 x^2 + 1.5 x", "the reference law"),
    ("scaled", "s (0.8 x^2 + 1.5 x)",
     "the same shape at a different amplitude -- the rain / wet-snow "
     "relationship, and the pair that closes as s -> 1"),
    ("other", "2.2 sin(2x)", "a genuinely different function: the easy class"),
    ("lagged", "3 * EWMA_0.85(0.8 x^2 + 1.5 x)",
     "the same input, answered with memory -- the wet-antenna relationship"),
)


def simulate_f3(n_per_regime=100, window_len=WINDOW_LEN, scale=0.5,
                lag_decay=0.85, noise_std=0.25, x_range=(-2.0, 2.0), seed=7):
    """Base law, a rescaled copy, a different shape, and a lagged variant.

    ``base`` and ``scaled`` share a shape and differ only in amplitude -- the
    rain / wet-snow relationship. ``lagged`` is driven by the same input but
    responds with memory -- the wet-antenna relationship. ``other`` is a
    genuinely different function, the easy class. ``scale`` is the knob: at
    1.0 the hard pair is identical and unsolvable by construction, at 0.2 it
    is easy.
    """
    rng = np.random.default_rng(seed)

    def base_f(x):
        return 0.8 * x ** 2 + 1.5 * x

    def lagged(x):
        out, acc = np.zeros_like(x), 0.0
        for i, v in enumerate(base_f(x)):
            acc = lag_decay * acc + (1 - lag_decay) * v
            out[i] = acc
        return out * 3.0                      # comparable amplitude

    laws = [("base", base_f), ("scaled", lambda x: scale * base_f(x)),
            ("other", lambda x: 2.2 * np.sin(2.0 * x)), ("lagged", lagged)]
    assert [n for n, _ in laws] == [n for n, _, _ in F3_REGIMES]

    Xs, Ys, labels = [], [], []
    for ri, (_, f) in enumerate(laws):
        for _ in range(n_per_regime):
            x = np.sort(rng.uniform(*x_range, size=window_len))
            Ys.append(f(x) + rng.normal(0.0, noise_std, window_len))
            Xs.append(x[:, None])
            labels.append(ri)
    o = rng.permutation(len(labels))
    return (np.asarray(Xs)[o], np.asarray(Ys)[o], np.asarray(labels)[o],
            [n for n, _ in laws])


def run_f3(seeds=REPORT_SEEDS) -> pd.DataFrame:
    """Which pairs confuse, as the hard pair is brought together?"""
    rows, pairs = [], []
    for scale in F3_SCALES:
        for seed in seeds:
            X, y, labels, names = simulate_f3(scale=scale, seed=seed)
            Z, _ = window_features(X, y)
            K = len(names)
            base = {"scale": scale, "seed": seed}

            res = _fit(X, y, Z, K, seed)
            lr = clustering_metrics(labels, res.labels)
            rows.append({**base, "method": "LR-DSR", **lr})
            lab = mechanism_init(X, y, K, seed=seed)
            rows.append({**base, "method": "K-means in mechanism space",
                         **clustering_metrics(labels, lab)})
            for bname, lb in geometry_baselines(Z, K, seed=seed).items():
                rows.append({**base, "method": bname,
                             **clustering_metrics(labels, lb)})

            inv = _align(labels, np.asarray(res.labels))
            for i, ni in enumerate(names):
                m = labels == i
                got = np.array([inv.get(int(c), -1) for c in np.asarray(res.labels)[m]])
                worst_j, worst_v = -1, 0.0
                for j in range(K):
                    if j != i and (v := float(np.mean(got == j))) > worst_v:
                        worst_j, worst_v = j, v
                pairs.append({**base, "regime": ni,
                              "recovered_fraction": float(np.mean(got == i)),
                              "most_confused_with":
                                  names[worst_j] if worst_j >= 0 else "",
                              "confusion_fraction": worst_v})
            print(f"  [F3] scale {scale:.1f} seed {seed:3d}  "
                  f"ARI {lr['ARI']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "f3_raw.csv", index=False)
    summary = (df.groupby(["scale", "method"])
                 [["ARI", "NMI", "aligned_accuracy"]].mean().reset_index()
                 .sort_values(["scale", "ARI"], ascending=[True, False]))
    summary.to_csv(RESULTS / "f3_summary.csv", index=False)
    conf = (pd.DataFrame(pairs)
              .groupby(["scale", "regime", "most_confused_with"])
              [["recovered_fraction", "confusion_fraction"]].mean().reset_index())
    conf.to_csv(RESULTS / "f3_confusions.csv", index=False)
    print("\n[F3]")
    print(summary.round(3).to_string(index=False))
    return df


def run(args=None) -> None:
    print("\n=== F1: two competing parametric families ===", flush=True)
    run_f1()
    print("\n=== F2: null vs signal, heterogeneous null ===", flush=True)
    run_f2()
    print("\n=== F3: multi-class, the hard pairs ===", flush=True)
    run_f3()
    print(f"\n[functions/scenarios] CSVs -> {RESULTS}")


if __name__ == "__main__":
    run()
