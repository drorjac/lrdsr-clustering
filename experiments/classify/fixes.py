"""Two repairs to the law classifier, on the same 24 UCR datasets.

    python -m experiments.classify.fixes

The benchmark (``run.py``) named two losses. On shape data whose classes are
patterns played early or late the basis loses to DTW (Trace 0.26 against
0.01); and a generative law classifier loses to the best tuned
discriminative classifier on the raw profile. Two repairs, each measured on
every dataset:

``law_shift``  phase as a profiled nuisance (``LawClassifier(shift_grid=...)``):
               the shift range is chosen by stratified CV on the training
               split from {0, 5%, 10%, 20%} of the series (0 = no warp, so a
               dataset where phase carries the class keeps it)
``law_stack``  ``LawStack``: a logistic head on cross-fitted law evidence,
               on top of the law ``law_shift`` chose

Everything else is held at the benchmark's choice for the dataset (basis
size, ``L``, nuisance: ``ucr_configs.csv``), so the difference is the repair.
The shift CV uses at most ``CV_CAP`` training series (a stratified
subsample), a rule fixed before the run for the large datasets.

Writes ``results/classify/ucr_fixes.csv``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, train_test_split

from experiments.classify import ucr
from lrdsr import paths
from lrdsr.core.classify import LawClassifier, LawStack
from lrdsr.core.kernel import CosineBasis

RESULTS = paths.results_dir("classify", figs=False)
SHIFTS = (0.0, 0.05, 0.1, 0.2)
STEP = 0.025
CV_CAP = 1000


def grid(width: float):
    return None if width == 0 else np.arange(-width, width + 1e-9, STEP)


def _law(cfg, width):
    return lambda: LawClassifier(basis=CosineBasis(cfg["size"]), nuisance=cfg["nuisance"],
                                 laws_per_class=cfg["L"], shift_grid=grid(width))


def cv_shift(Xs, ys, y, cfg, seed=0) -> tuple[float, float]:
    idx = np.arange(len(y))
    if len(y) > CV_CAP:
        idx, _ = train_test_split(idx, train_size=CV_CAP, stratify=y, random_state=seed)
    k = int(min(5, np.bincount(y[idx]).min()))
    if k < 2:
        return 0.0, np.nan
    folds = list(StratifiedKFold(k, shuffle=True, random_state=seed).split(idx, y[idx]))
    best, best_acc = 0.0, -1.0
    for w in SHIFTS:
        acc = []
        for tr, va in folds:
            tr, va = idx[tr], idx[va]
            m = _law(cfg, w)().fit([Xs[i] for i in tr], [ys[i] for i in tr], y[tr])
            acc.append(np.mean(m.predict([Xs[i] for i in va], [ys[i] for i in va]) == y[va]))
        if np.mean(acc) > best_acc + 1e-9:                # ties keep the smaller warp
            best, best_acc = w, float(np.mean(acc))
    return best, best_acc


def one(name: str, cfg: dict) -> list[dict]:
    s = ucr.load(name)
    Xs, ys = ucr.to_windows(s.train)
    Xt, yt = ucr.to_windows(s.test)
    t0 = time.time()
    width, cv_acc = cv_shift(Xs, ys, s.y_train, cfg)
    law = _law(cfg, width)().fit(Xs, ys, s.y_train)
    e_shift = float(np.mean(law.predict(Xt, yt) != s.y_test))
    t1 = time.time()
    stack = LawStack(_law(cfg, width)).fit(Xs, ys, s.y_train)
    e_stack = float(np.mean(stack.predict(Xt, yt) != s.y_test))
    print(f"  [{name}] shift {width:g}: {e_shift:.3f}, stack {e_stack:.3f} "
          f"({t1 - t0:.0f}s + {time.time() - t1:.0f}s)", flush=True)
    base = {"dataset": name, "group": s.group, "shift_width": width,
            "shift_cv_accuracy": cv_acc}
    return [{**base, "method": "law_shift", "error": e_shift},
            {**base, "method": "law_stack", "error": e_stack}]


def run(args=None) -> None:
    t0 = time.time()
    cfg = pd.read_csv(RESULTS / "ucr_configs.csv", keep_default_na=False).set_index("dataset")
    todo = []
    for name in ucr.DATASETS:
        c = cfg.loc[name]
        todo.append((name, {"size": int(c["cosine_size"]),
                            "L": c["cosine_L"] if c["cosine_L"] == "all" else int(c["cosine_L"]),
                            "nuisance": None if c["cosine_nuisance"] == "None" else "intercept"}))
    # the largest dataset first, so it is not the straggler
    todo.sort(key=lambda t: -len(ucr.load(t[0]).train))
    out = Parallel(n_jobs=8)(delayed(one)(n, c) for n, c in todo)
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv(RESULTS / "ucr_fixes.csv", index=False)
    print(df.pivot_table(index=["group", "dataset"], columns="method", values="error")
          .round(3).to_string())
    print(f"\n[classify.fixes] {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    run()
