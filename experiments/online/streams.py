"""Real-time clustering by law, measured.

Five questions about :mod:`lrdsr.core.online`, each answered against a
reference that knows more than the streaming estimator does.

==============  ===========================================================
``stationary``  Warm-start on 40 unlabelled windows (and, as a second arm,
                on 160), then assign 400 more one at a time. Against: the
                same laws frozen at warm start
                (what updating buys), the batch ``SoftLRDSR`` and
                ``GroupedDCSR`` fitted on the *whole* stream in hindsight
                (what waiting would buy), and the oracle.
``birth``       A third law ``x^2 + A sin 2x`` starts appearing after 100
                windows. How long until a regime is born for it, how many
                false births, and how well the windows after the birth are
                assigned -- swept over its amplitude ``A`` against the
                chi-square power the novelty test has for it, and over
                ``novelty_alpha`` and ``novelty_patience``, with a
                stationary control stream for the false births.
``drift``       One law's intercept drifts by 6 over the stream until a
                *stale* copy of it explains its windows worse than the other
                law does. Forgetting ``lambda`` in {1, 0.98, 0.95} against
                frozen laws: tracking error of the law and accuracy.
``latency``     Microseconds per ``partial_fit`` against window length and
                the number of regimes -- the claim "real time" is a number.
``cusum``       One *sample* stream with Markov switches, segmented by CUSUM
                with the laws learned from unlabelled history, against the
                same CUSUM with the true laws.
==============  ===========================================================

The base pair is the polynomial pair ``x^2 + x`` / ``x^2 - x`` on U[-2, 2]
(gap ``2x``, ``E g^2 = 16/3``); ``sigma`` is set from ``rho``, windows are 32
samples. True labels reach the metrics only: every estimator here is
warm-started without them, and its slots are matched to the truth after the
fact (Hungarian, over the whole stream, so a regime cannot change meaning
half way).

Writes ``results/online/stream_*.csv``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from experiments.common.fitting import fit_lrdsr, window_features
from lrdsr import paths
from lrdsr.core.online import CusumSegmenter, OnlineLRDSR
from lrdsr.core.soft import SoftLRDSR, library_terms

RESULTS = paths.results_dir("online", figs=False)
REPORT_SEEDS = (11, 23, 42)

WINDOW_LEN = 32
X_RANGE = (-2.0, 2.0)
GAP_MS = 16.0 / 3.0
N_HISTORY = 40
N_STREAM = 400

LAWS = (lambda x: x ** 2 + x,
        lambda x: x ** 2 - x,
        # the newcomer in ``birth``: shares the x^2, differs in shape
        lambda x: x ** 2 + 2.0 * np.sin(2.0 * x))
LAW_NAMES = ("x^2 + x", "x^2 - x", "x^2 + 2 sin(2x)")


def sigma_for(rho: float) -> float:
    """Noise level giving separation ``rho`` to the base pair."""
    return float(np.sqrt(GAP_MS / rho))


def make_windows(z, sigma, rng, laws=LAWS, n=WINDOW_LEN):
    """One window per entry of ``z``, drawn from ``laws[z]``."""
    z = np.asarray(z, int)
    x = rng.uniform(*X_RANGE, size=(len(z), n))
    clean = np.empty_like(x)
    for k in np.unique(z):
        clean[z == k] = laws[k](x[z == k])
    return x[:, :, None], clean + rng.normal(0.0, sigma, size=x.shape)


def matched_error(truth, pred) -> float:
    """Window error after one Hungarian matching over the whole stream.

    ``pred = -1`` (a window held in the novelty buffer, not assigned) counts
    as an error: an unassigned window is not a correct one.
    """
    truth = np.asarray(truth, int)
    pred = np.asarray(pred, int)
    kp = max(int(pred.max()) + 1, 1)
    kt = int(truth.max()) + 1
    C = np.zeros((kt, kp))
    for t, p in zip(truth, pred, strict=True):
        if p >= 0:
            C[t, p] += 1
    r, c = linear_sum_assignment(-C)
    return 1.0 - C[r, c].sum() / len(truth)


def _mapping(truth, pred) -> dict:
    """Hungarian map pred -> truth (pred slots without a partner map to -2)."""
    truth, pred = np.asarray(truth, int), np.asarray(pred, int)
    kp, kt = max(int(pred.max()) + 1, 1), int(truth.max()) + 1
    C = np.zeros((kt, kp))
    for t, p in zip(truth, pred, strict=True):
        if p >= 0:
            C[t, p] += 1
    r, c = linear_sum_assignment(-C)
    m = {int(cc): int(rr) for rr, cc in zip(r, c, strict=True)}
    return {k: m.get(k, -2) for k in range(kp)}


def oracle_labels(X, y, laws):
    res = np.column_stack([np.sum((y - f(X[:, :, 0])) ** 2, axis=1) for f in laws])
    return res.argmin(axis=1)


# ==========================================================================
# (a) stationary
# ==========================================================================
RHO_STATIONARY = (0.1, 0.25, 1.0)
#: A second, longer warm start: at low rho, 40 windows may not be enough to
#: find the laws at all, and a stream assigned by wrong laws reinforces them.
N_HISTORY_LONG = 160
TIME_BLOCK = 50


def run_stationary(seeds=REPORT_SEEDS, verbose=True):
    rows, trows = [], []
    for seed in seeds:
        for rho in RHO_STATIONARY:
            rng = np.random.default_rng(seed * 100 + int(rho * 100))
            sigma = sigma_for(rho)
            zh = rng.integers(0, 2, N_HISTORY_LONG)
            Xl, yl = make_windows(zh, sigma, rng)
            Xh, yh = Xl[:N_HISTORY], yl[:N_HISTORY]
            z = rng.integers(0, 2, N_STREAM)
            X, y = make_windows(z, sigma, rng)

            preds = {}
            longer = OnlineLRDSR(feature_names=["x"]).warm_start(
                Xl, yl, n_clusters=2, random_state=seed)
            preds[f"online_history{N_HISTORY_LONG}"] = longer.fit_stream(X, y)["label"].to_numpy()
            online = OnlineLRDSR(feature_names=["x"]).warm_start(
                Xh, yh, n_clusters=2, random_state=seed)
            preds["online"] = online.fit_stream(X, y)["label"].to_numpy()
            frozen = OnlineLRDSR(feature_names=["x"]).warm_start(
                Xh, yh, n_clusters=2, random_state=seed)
            preds["online_frozen"] = frozen.fit_stream(X, y, update=False)["label"].to_numpy()
            nov_off = OnlineLRDSR(feature_names=["x"], novelty_alpha=0.0).warm_start(
                Xh, yh, n_clusters=2, random_state=seed)
            preds["online_no_novelty"] = nov_off.fit_stream(X, y)["label"].to_numpy()

            # batch, in hindsight: every window of history and stream at once
            Xa, ya = np.concatenate([Xh, X]), np.concatenate([yh, y])
            soft = SoftLRDSR(2, feature_names=["x"], random_state=seed).fit(Xa, ya)
            preds["batch_soft"] = soft.labels[N_HISTORY:]
            Z, names = window_features(Xa, ya)
            hard = fit_lrdsr(Z, names, Xa, ya, ["x"], 2, seed, alpha_geom=0.0)
            preds["batch_lrdsr"] = hard.labels[N_HISTORY:]
            preds["oracle"] = oracle_labels(X, y, LAWS[:2])

            for method, p in preds.items():
                rows.append({"seed": seed, "rho": rho, "method": method,
                             "error": matched_error(z, p),
                             "unassigned": int(np.sum(p < 0))})
                m = _mapping(z, p)
                mapped = np.array([m.get(int(v), -2) if v >= 0 else -1 for v in p])
                for b0 in range(0, N_STREAM, TIME_BLOCK):
                    sl = slice(b0, b0 + TIME_BLOCK)
                    trows.append({"seed": seed, "rho": rho, "method": method,
                                  "block_start": b0,
                                  "error": float(np.mean(mapped[sl] != z[sl]))})
            if verbose:
                r = {d["method"]: d["error"] for d in rows[-len(preds):]}
                print(f"  stationary seed={seed} rho={rho:<5} "
                      + "  ".join(f"{k} {v:.3f}" for k, v in r.items()), flush=True)
    df, tdf = pd.DataFrame(rows), pd.DataFrame(trows)
    df.to_csv(RESULTS / "stream_stationary.csv", index=False)
    tdf.to_csv(RESULTS / "stream_stationary_time.csv", index=False)
    summ = df.groupby(["rho", "method"], as_index=False).agg(
        mean_error=("error", "mean"), mean_unassigned=("unassigned", "mean"))
    summ.to_csv(RESULTS / "stream_stationary_summary.csv", index=False)
    return df


# ==========================================================================
# (b) regime birth
# ==========================================================================
#: The newcomer is ``x^2 + A sin(2x)``; ``A`` is the knob. Its separation from
#: the nearer existing law, in the units the base pair is set in, is computed
#: by :func:`newcomer_rho`, and with it the chi-square power of the novelty
#: test -- so birth is measured against what the test *can* see.
AMPLITUDES = (1.0, 2.0, 4.0, 6.0)
RHO_BIRTH = 1.0
ALPHAS = (1e-2, 1e-3, 1e-4)
PATIENCES = (3, 6)
BIRTH_AT = 100
N_BIRTH_STREAM = 300


def newcomer_law(A):
    return lambda x: x ** 2 + A * np.sin(2.0 * x)


def newcomer_rho(A, sigma, n_mc=400_000) -> float:
    """``min_k E[(f_new - f_k)^2] / sigma^2`` over the two existing laws."""
    x = np.random.default_rng(0).uniform(*X_RANGE, n_mc)
    f = newcomer_law(A)(x)
    return float(min(np.mean((f - LAWS[k](x)) ** 2) for k in (0, 1)) / sigma ** 2)


def novelty_power(rho_new, alpha, n=WINDOW_LEN) -> float:
    """P(a newcomer window is flagged): non-central chi-square with ``n rho``.

    Ignores that the existing laws are estimates and that the noise level is
    too; it is the power the test would have with both known.
    """
    from scipy.stats import chi2, ncx2
    return float(ncx2.sf(chi2.isf(alpha, n), n, n * rho_new))


def run_birth(seeds=REPORT_SEEDS, verbose=True):
    rows = []
    sigma = sigma_for(RHO_BIRTH)
    for seed in seeds:
        for A in (None, *AMPLITUDES):            # None: the control stream
            rng = np.random.default_rng(seed * 1000 + (0 if A is None else int(10 * A)))
            laws = LAWS[:2] + ((newcomer_law(A),) if A is not None else ())
            zh = rng.integers(0, 2, N_HISTORY_LONG)
            Xh, yh = make_windows(zh, sigma, rng, laws=laws)
            z = np.r_[rng.integers(0, 2, BIRTH_AT),
                      rng.integers(0, len(laws), N_BIRTH_STREAM - BIRTH_AT)]
            X, y = make_windows(z, sigma, rng, laws=laws)
            first = int(np.argmax(z == 2)) if (z == 2).any() else -1
            rho_new = newcomer_rho(A, sigma) if A is not None else np.nan
            for alpha in ALPHAS:
                for pat in PATIENCES:
                    on = OnlineLRDSR(feature_names=["x"], novelty_alpha=alpha,
                                     novelty_patience=pat).warm_start(
                        Xh, yh, n_clusters=2, random_state=seed)
                    tl = on.fit_stream(X, y)
                    births = tl.loc[tl["spawned"].notna(), "t"].to_numpy()
                    lab = tl["label"].to_numpy()
                    row = {"seed": seed, "stream": "control" if A is None else "newcomer",
                           "amplitude": np.nan if A is None else A,
                           "rho_newcomer": rho_new,
                           "novelty_alpha": alpha, "novelty_patience": pat,
                           "n_births": len(births),
                           "false_births": (len(births) if A is None
                                            else int(np.sum(births < first))),
                           "error": matched_error(z, lab),
                           "unassigned": int(np.sum(lab < 0))}
                    if A is not None:
                        power = novelty_power(rho_new, alpha)
                        row["predicted_power"] = power
                        # patience flagged newcomers needed; one in three stream
                        # windows is a newcomer after BIRTH_AT
                        row["predicted_delay_newcomer_windows"] = pat / max(power, 1e-9)
                        after = births[births >= first]
                        t_b = int(after[0]) if len(after) else -1
                        row["born"] = t_b >= 0
                        row["birth_delay_windows"] = (t_b - first) if t_b >= 0 else np.nan
                        row["birth_delay_newcomer_windows"] = (
                            int(np.sum(z[first:t_b + 1] == 2)) if t_b >= 0 else np.nan)
                        m = _mapping(z, lab)
                        row["born_regime_is_newcomer"] = bool(
                            t_b >= 0 and m.get(int(lab[t_b]), -2) == 2)
                        if t_b >= 0 and t_b + 1 < len(z):
                            row["error_after_birth"] = matched_error(z[t_b + 1:], lab[t_b + 1:])
                        else:
                            row["error_after_birth"] = np.nan
                    rows.append(row)
            if verbose:
                sub = pd.DataFrame(rows[-len(ALPHAS) * len(PATIENCES):])
                tag = "control" if A is None else f"A={A:g} rho_new={rho_new:.2f}"
                print(f"  birth seed={seed} {tag:<22} births {sub.n_births.tolist()}",
                      flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "stream_birth.csv", index=False)
    summ = df.groupby(["stream", "amplitude", "novelty_alpha", "novelty_patience"],
                      as_index=False, dropna=False).agg(
        rho_newcomer=("rho_newcomer", "first"),
        mean_births=("n_births", "mean"),
        mean_false_births=("false_births", "mean"),
        frac_born=("born", "mean"),
        frac_born_is_newcomer=("born_regime_is_newcomer", "mean"),
        predicted_power=("predicted_power", "first"),
        predicted_delay_newcomer_windows=("predicted_delay_newcomer_windows", "first"),
        mean_birth_delay_newcomer_windows=("birth_delay_newcomer_windows", "mean"),
        mean_birth_delay_windows=("birth_delay_windows", "mean"),
        mean_error=("error", "mean"),
        mean_error_after_birth=("error_after_birth", "mean"),
        mean_unassigned=("unassigned", "mean"))
    summ.to_csv(RESULTS / "stream_birth_summary.csv", index=False)
    return df


# ==========================================================================
# (c) drift
# ==========================================================================
DRIFT_SIGMA = 2.0
DRIFT_SHIFT = 6.0
FORGETTING = (1.0, 0.98, 0.95)
N_DRIFT = 400


def run_drift(seeds=REPORT_SEEDS, verbose=True):
    """``f0 = x^2 + x + c(t)`` with ``c`` 0 -> 6; ``f1 = x^2 - x + 3`` fixed.

    At ``c = 0`` a window of ``f0`` is closer to the warm-start ``f0``; by
    ``c = 6`` it is closer to ``f1`` than to the stale ``f0``, so frozen laws
    must misassign the late ``f0`` windows. Novelty is off: this is about
    tracking a law, not about birthing a new one for its later self.
    """
    grid = np.linspace(*X_RANGE, 101)[:, None]
    Phi_g = library_terms(grid, feature_names=["x"])[0]
    rows, trows = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed + 5000)
        zh = rng.integers(0, 2, N_HISTORY)
        c_t = DRIFT_SHIFT * np.arange(N_DRIFT) / (N_DRIFT - 1)
        z = rng.integers(0, 2, N_DRIFT)
        x = rng.uniform(*X_RANGE, size=(N_DRIFT, WINDOW_LEN))
        y = np.where(z[:, None] == 0, x ** 2 + x + c_t[:, None], x ** 2 - x + 3.0)
        y = y + rng.normal(0, DRIFT_SIGMA, size=x.shape)
        xh = rng.uniform(*X_RANGE, size=(N_HISTORY, WINDOW_LEN))
        yh = np.where(zh[:, None] == 0, xh ** 2 + xh, xh ** 2 - xh + 3.0)
        yh = yh + rng.normal(0, DRIFT_SIGMA, size=xh.shape)
        X, Xh = x[:, :, None], xh[:, :, None]
        arms = [("frozen", 1.0, False)] + [(f"lambda={lam:g}", lam, True)
                                           for lam in FORGETTING]
        for arm, lam, update in arms:
            on = OnlineLRDSR(feature_names=["x"], forgetting=lam, novelty_alpha=0.0
                             ).warm_start(Xh, yh, n_clusters=2, random_state=seed)
            # which slot is f0: the one whose law is closer to it at c = 0
            f0_0 = grid[:, 0] ** 2 + grid[:, 0]
            slot0 = int(np.argmin([np.mean((Phi_g @ on._coef(k) - f0_0) ** 2)
                                   for k in range(2)]))
            labels, track = [], []
            for w in range(N_DRIFT):
                a = on.partial_fit(X[w], y[w], update=update)
                labels.append(a.label)
                truth_f0 = grid[:, 0] ** 2 + grid[:, 0] + c_t[w]
                track.append(float(np.sqrt(np.mean((Phi_g @ on._coef(slot0) - truth_f0) ** 2))))
            labels, track = np.array(labels), np.array(track)
            pred_is_f0 = labels == slot0
            correct = pred_is_f0 == (z == 0)
            half = N_DRIFT // 2
            rows.append({"seed": seed, "arm": arm, "forgetting": lam,
                         "updates": update,
                         "error": float(1 - correct.mean()),
                         "error_second_half": float(1 - correct[half:].mean()),
                         "error_f0_second_half": float(
                             1 - correct[half:][z[half:] == 0].mean()),
                         "tracking_rmse": float(track.mean()),
                         "tracking_rmse_second_half": float(track[half:].mean())})
            for b0 in range(0, N_DRIFT, TIME_BLOCK):
                sl = slice(b0, b0 + TIME_BLOCK)
                trows.append({"seed": seed, "arm": arm, "block_start": b0,
                              "drift_offset": float(c_t[sl].mean()),
                              "error": float(1 - correct[sl].mean()),
                              "tracking_rmse": float(track[sl].mean())})
            if verbose:
                r = rows[-1]
                print(f"  drift seed={seed} {arm:<12} error {r['error']:.3f} "
                      f"(2nd half {r['error_second_half']:.3f})  law RMSE "
                      f"{r['tracking_rmse']:.3f} (2nd half "
                      f"{r['tracking_rmse_second_half']:.3f})", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "stream_drift.csv", index=False)
    pd.DataFrame(trows).to_csv(RESULTS / "stream_drift_time.csv", index=False)
    df.groupby("arm", as_index=False, sort=False).mean(numeric_only=True).drop(
        columns="seed").to_csv(RESULTS / "stream_drift_summary.csv", index=False)
    return df


# ==========================================================================
# (d) latency
# ==========================================================================
def run_latency(verbose=True, reps: int = 300):
    rng = np.random.default_rng(REPORT_SEEDS[0])
    rows = []
    for n in (16, 32, 64, 128):
        for K in (2, 4, 8):
            laws = [lambda x, a=a: x ** 2 + a * x for a in np.linspace(-2, 2, K)]
            zh = np.repeat(np.arange(K), 10)
            Xh, yh = make_windows(zh, 1.0, rng, laws=laws, n=n)
            on = OnlineLRDSR(feature_names=["x"]).warm_start(Xh, yh, labels=zh)
            X, y = make_windows(rng.integers(0, K, reps), 1.0, rng, laws=laws, n=n)
            for w in range(10):                          # warm the caches
                on.partial_fit(X[w], y[w])
            t = np.empty(reps)
            for w in range(reps):
                t0 = time.perf_counter()
                on.partial_fit(X[w], y[w])
                t[w] = time.perf_counter() - t0
            rows.append({"window_len": n, "n_regimes": K,
                         "median_us": float(np.median(t) * 1e6),
                         "p95_us": float(np.percentile(t, 95) * 1e6),
                         "windows_per_s": float(1.0 / np.median(t))})
            if verbose:
                r = rows[-1]
                print(f"  latency n={n:<4} K={K}  median {r['median_us']:7.0f} us  "
                      f"p95 {r['p95_us']:7.0f} us", flush=True)
    # CUSUM, per sample
    x = rng.uniform(-2, 2, 20_000)
    yv = x ** 2 + x + rng.normal(0, 1, x.size)
    seg = CusumSegmenter([lambda v: v ** 2 + v, lambda v: v ** 2 - v], 1.0, 5.0)
    t0 = time.perf_counter()
    seg.run(x, yv)
    per = (time.perf_counter() - t0) / x.size
    rows.append({"window_len": 1, "n_regimes": 2, "median_us": per * 1e6,
                 "p95_us": np.nan, "windows_per_s": 1.0 / per,
                 })
    df = pd.DataFrame(rows)
    df["estimator"] = ["OnlineLRDSR.partial_fit"] * (len(df) - 1) + ["CusumSegmenter (per sample)"]
    df.to_csv(RESULTS / "stream_latency.csv", index=False)
    if verbose:
        print(f"  latency CUSUM {per * 1e6:.1f} us per sample", flush=True)
    return df


# ==========================================================================
# (e) CUSUM segmentation of one sample stream
# ==========================================================================
RHO_CUSUM = (0.25, 1.0)
H_CUSUM = (3.0, 5.0, 8.0)
MEAN_SEGMENT = 200
N_SAMPLES = 20_000


def _markov_states(rng, T, mean_len):
    s = np.empty(T, dtype=int)
    cur = int(rng.integers(0, 2))
    for t in range(T):
        if rng.random() < 1.0 / mean_len:
            cur = 1 - cur
        s[t] = cur
    return s


def _delays(truth, state):
    """Samples from each true switch to the detector entering the new regime."""
    sw = np.flatnonzero(np.diff(truth)) + 1
    out = []
    for i, t0 in enumerate(sw):
        end = sw[i + 1] if i + 1 < len(sw) else len(truth)
        hit = np.flatnonzero(state[t0:end] == truth[t0])
        out.append(int(hit[0]) + 1 if len(hit) else np.nan)
    return np.array(out, dtype=float)


def run_cusum(seeds=REPORT_SEEDS, verbose=True):
    rows = []
    for seed in seeds:
        for rho in RHO_CUSUM:
            rng = np.random.default_rng(seed * 7 + int(rho * 100))
            sigma = sigma_for(rho)
            truth = _markov_states(rng, N_SAMPLES, MEAN_SEGMENT)
            x = rng.uniform(*X_RANGE, N_SAMPLES)
            y = np.where(truth == 0, LAWS[0](x), LAWS[1](x)) + rng.normal(0, sigma, N_SAMPLES)
            # learned laws: SoftLRDSR on unlabelled history windows
            zh = rng.integers(0, 2, 60)
            Xh, yh = make_windows(zh, sigma, rng)
            soft = SoftLRDSR(2, feature_names=["x"], random_state=seed).fit(Xh, yh)

            def learned(k, soft=soft):
                return lambda v: library_terms(np.asarray(v)[:, None],
                                               feature_names=["x"])[0] @ soft.coef[k]
            law_sets = {"true_laws": ([LAWS[0], LAWS[1]], sigma),
                        "learned_laws": ([learned(0), learned(1)], soft.sigma)}
            for name, (laws, sig) in law_sets.items():
                for h in H_CUSUM:
                    out = CusumSegmenter(laws, sig, threshold=h, start=0).run(x, y)
                    st = out["state"].to_numpy()
                    # learned slots carry no names: match them to the truth once
                    m = _mapping(truth, st)
                    st_m = np.array([m[int(v)] for v in st])
                    d = _delays(truth, st_m)
                    n_sw = int(np.sum(np.diff(truth) != 0))
                    rows.append({
                        "seed": seed, "rho": rho, "laws": name, "h": h,
                        "accuracy": float(np.mean(st_m == truth)),
                        "true_switches": n_sw,
                        "alarms": int(out["alarm"].sum()),
                        "missed": int(np.isnan(d).sum()),
                        "mean_delay": float(np.nanmean(d)),
                        "median_delay": float(np.nanmedian(d)),
                        "predicted_delay_first_order": 2 * h / rho,
                    })
                    if verbose:
                        r = rows[-1]
                        print(f"  cusum seed={seed} rho={rho:<5} {name:<12} h={h:<4} "
                              f"acc {r['accuracy']:.3f} alarms {r['alarms']:4d}/"
                              f"{n_sw} switches, delay {r['mean_delay']:6.1f}",
                              flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "stream_cusum.csv", index=False)
    df.groupby(["rho", "laws", "h"], as_index=False).mean(numeric_only=True).drop(
        columns="seed").to_csv(RESULTS / "stream_cusum_summary.csv", index=False)
    return df


def run(verbose=True):
    t = {}
    for name, fn in (("stationary", run_stationary), ("birth", run_birth),
                     ("drift", run_drift), ("latency", run_latency),
                     ("cusum", run_cusum)):
        t0 = time.time()
        print(f"\n--- {name} ---", flush=True)
        fn(verbose=verbose)
        t[name] = time.time() - t0
        print(f"  [{name}] {t[name]:.0f} s", flush=True)
    return t
