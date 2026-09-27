"""Every classifier and clusterer the ``classify`` block compares, in one place.

Two families, given the same series:

*Law methods* see ragged windows -- only the observed samples, at their own
times -- and never an interpolated grid:

==================  =======================================================
``law``             ``LawClassifier``: ``L`` laws per class in a basis;
                    basis size, ``L`` and the level nuisance chosen by
                    stratified CV on the training split
``law_symbolic``    one *named* law per class from the fast backend's
                    library (``x = 2 pi t / n``): what a readable classifier
                    costs
``mech_*``          ``MechanismFeatures`` (the feature domain) into a
                    logistic regression or an RBF SVM, the SVM being a
                    kernel between windows' laws
==================  =======================================================

*Raw-profile methods* see every series on the common grid, gaps linearly
interpolated: 1-NN Euclidean (the archive's own reference), nearest
centroid, logistic regression, RBF SVM and a random forest.
"""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from experiments.classify.ucr import to_grid, to_windows
from lrdsr.core.classify import LawClassifier, MechanismFeatures
from lrdsr.core.kernel import CosineBasis, NystromBasis, select_rank

SIZES = (4, 8, 16, 32)
LAWS = (1, 2, 4, "all")
NUISANCE = (None, "intercept")
BASES = {"cosine": CosineBasis, "nystrom": NystromBasis}


def _folds(y, max_folds=5):
    k = int(min(max_folds, np.bincount(y).min()))
    return StratifiedKFold(k, shuffle=True, random_state=0) if k >= 2 else None


def _sizes(length: int):
    return [s for s in SIZES if s <= max(length - 2, 2)]


def _law(kind, size, L, nuis, seed=0):
    return LawClassifier(basis=BASES[kind](size), nuisance=nuis, laws_per_class=L,
                         random_state=seed)


def cv_law(Xs, ys, y, length, kind="cosine", laws=LAWS, seed=0) -> tuple[dict, float]:
    """The ``(size, L, nuisance)`` with the best stratified-CV accuracy on training
    windows. Ties go to the simpler model (smaller size, then smaller ``L``)."""
    # three folds on a large training split: the grid is 32 fits per fold
    folds = _folds(y, 3 if len(y) > 500 else 5)
    grid = list(itertools.product(_sizes(length), laws, NUISANCE))
    if folds is None:
        return {"size": _sizes(length)[1] if len(_sizes(length)) > 1 else 4,
                "L": 1, "nuisance": None}, np.nan
    best, best_acc = None, -1.0
    splits = list(folds.split(np.zeros(len(y)), y))
    for size, L, nuis in grid:
        accs = []
        for tr, va in splits:
            m = _law(kind, size, L, nuis, seed).fit([Xs[i] for i in tr],
                                                    [ys[i] for i in tr], y[tr])
            accs.append(np.mean(m.predict([Xs[i] for i in va], [ys[i] for i in va])
                                == y[va]))
        acc = float(np.mean(accs))
        if acc > best_acc + 1e-9:
            best, best_acc = {"size": size, "L": L, "nuisance": nuis}, acc
    return best, best_acc


def law_predict(train, y_train, test, cfg, kind="cosine", seed=0):
    Xs, ys = to_windows(train)
    Xt, yt = to_windows(test)
    m = _law(kind, cfg["size"], cfg["L"], cfg["nuisance"], seed).fit(Xs, ys, y_train)
    return m.predict(Xt, yt), m


def law_symbolic_predict(train, y_train, test):
    Xs, ys = to_windows(train, angle=True)
    Xt, yt = to_windows(test, angle=True)
    m = LawClassifier(basis=None, feature_names=["x"], nuisance="intercept",
                      law="symbolic").fit(Xs, ys, y_train)
    return m.predict(Xt, yt), m


def mech_predict(train, y_train, test, size, nuisance, mode="project", seed=0,
                 which=("logistic", "svm")):
    """Mechanism-space features into logistic regression and an RBF SVM (CV on train)."""
    Xs, ys = to_windows(train)
    Xt, yt = to_windows(test)
    fm = MechanismFeatures(basis=CosineBasis(size), nuisance=nuisance, mode=mode)
    S = fm.fit_transform(Xs, ys)
    St = fm.transform(Xt, yt)
    return _feature_classifiers(S, y_train, St, seed, which)


def _feature_classifiers(S, y, St, seed=0, which=("logistic", "svm")) -> dict:
    folds = _folds(y, 5)
    out = {}
    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
    if folds is not None:
        logit = GridSearchCV(logit, {"logisticregression__C": [0.01, 0.1, 1, 10, 100]},
                             cv=folds)
        svm = GridSearchCV(svm, {"svc__C": [0.1, 1, 10, 100],
                                 "svc__gamma": ["scale", 0.1, 0.01]}, cv=folds)
    if "logistic" in which:
        out["logistic"] = logit.fit(S, y).predict(St)
    if "svm" in which:
        out["svm"] = svm.fit(S, y).predict(St)
    return out


def raw_predict(train, y_train, test, length=None, seed=0, which=None) -> dict:
    """Raw-profile baselines on the interpolated common grid."""
    G, Gt = to_grid(train, length), to_grid(test, length)
    out = {"1nn_ed": KNeighborsClassifier(1).fit(G, y_train).predict(Gt),
           "centroid": NearestCentroid().fit(G, y_train).predict(Gt)}
    if which == "fast":
        return out
    out.update(_feature_classifiers(G, y_train, Gt, seed))
    out["random_forest"] = RandomForestClassifier(
        500, random_state=seed, n_jobs=1).fit(G, y_train).predict(Gt)
    return out


def loo_size(train, kind="cosine", nuisance=None) -> int:
    """Basis size from per-window leave-one-out on training inputs alone --
    label-free, so usable at one labelled window per class."""
    Xs, ys = to_windows(train)
    length = max(len(v) for v in ys)
    size, _ = select_rank(Xs, ys, kind, _sizes(length) + [s for s in (48, 64)
                                                          if s <= length // 2],
                          nuisance=None if nuisance is None else _const)
    return size


def _const(X):
    return np.ones((len(X), 1))
