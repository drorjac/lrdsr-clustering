from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import StandardScaler

from .backends import (
    KnownMechanismRegressor,
    ResidualSymbolicRegressor,
    SuperpositionRegressor,
    make_symbolic_regressor,
)
from .losses import (
    HUBER_DELTA,
    LOSSES,
    aggregate_window_residual,
    joint_cost,
    learn_loss,
    noise_scale,
    robust_scale,
)
from .mechanism_space import mechanism_features

# Mechanism-knowledge levels a cluster slot may declare.
#   unknown  -- full symbolic search
#   partial  -- fixed known prior h_k plus a discovered residual g_k
#   known    -- fixed complete law, nothing is fitted
#   factory  -- the spec supplies a zero-argument callable returning a fresh
#               regressor. This covers knowledge that is neither fully fixed
#               nor absent: a prior whose FORM is known but whose coefficients
#               must be fitted (see ParametricPriorRegressor). Treated as
#               declared knowledge everywhere 'partial' is.
_MECHANISM_MODES = ("unknown", "partial", "known", "factory")


def _validate_mechanism_specs(specs, n_clusters: int):
    if specs is None:
        return None
    if len(specs) != n_clusters:
        raise ValueError(
            f"mechanism_specs must contain one spec per cluster: "
            f"got {len(specs)} for n_clusters={n_clusters}"
        )
    for k, spec in enumerate(specs):
        mode = spec.get("mode", "unknown")
        if mode not in _MECHANISM_MODES:
            raise ValueError(f"mechanism_specs[{k}]: invalid mode {mode!r}, "
                             f"expected one of {_MECHANISM_MODES}")
        if mode == "partial" and not callable(spec.get("prior_function")):
            raise ValueError(f"mechanism_specs[{k}]: mode 'partial' requires a "
                             "callable 'prior_function'")
        if mode == "known" and not callable(spec.get("function")):
            raise ValueError(f"mechanism_specs[{k}]: mode 'known' requires a "
                             "callable 'function'")
        if mode == "factory" and not callable(spec.get("factory")):
            raise ValueError(f"mechanism_specs[{k}]: mode 'factory' requires a "
                             "callable 'factory' returning a fresh regressor")
    return list(specs)


@dataclass
class DCSRResult:
    labels: np.ndarray
    models: list
    centroids: np.ndarray
    scaler: StandardScaler
    history: pd.DataFrame
    total_cost: np.ndarray


class GroupedDCSR:
    """Alternating grouped/window-level latent-regime symbolic regression.

    Core LR-DSR model: y_i = f_{z_i}(x_i) with latent assignments z. Knowledge
    about each mechanism f_k can be *unknown* (full symbolic search),
    *partial* (known prior h_k plus discovered residual g_k), or *known*
    (fixed h_k, no search), declared per cluster via ``mechanism_specs``.
    ``alpha_geom=0`` gives pure mechanism-based assignment (K-means remains
    only the initializer).
    """

    def __init__(
        self,
        n_clusters: int,
        alpha_geom: float = 0.25,
        beta_complexity: float = 0.0,
        lambda_phys: float = 0.0,
        max_iter: int = 10,
        tol: float = 0.01,
        robust_delta: float = HUBER_DELTA,
        backend: str = "fast",
        backend_kwargs: dict | None = None,
        random_state: int = 7,
        min_windows_per_cluster: int = 3,
        mechanism_specs: list[dict] | None = None,
        score_mode: str = "cross_fit",
        crossfit_folds: int = 3,
        init: str = "kmeans",
        mechanism_basis=None,
        mechanism_nuisance=None,
        mechanism_groups=None,
        geom_metric: str = "euclidean",
        superposition: bool = False,
        superposition_margin: float = 0.0,
        max_composites: int | None = None,
        residual_scale: str = "global",
        loss: str = "huber",
        loss_params: dict | None = None,
        scale_convention: str = "sd",
    ):
        self.n_clusters = n_clusters
        self.alpha_geom = alpha_geom
        self.beta_complexity = beta_complexity
        self.lambda_phys = lambda_phys
        self.max_iter = max_iter
        self.tol = tol
        self.robust_delta = robust_delta
        self.backend = backend
        self.backend_kwargs = backend_kwargs or {}
        self.random_state = random_state
        self.min_windows_per_cluster = min_windows_per_cluster
        self.mechanism_specs = _validate_mechanism_specs(mechanism_specs, n_clusters)
        if score_mode not in ("in_sample", "cross_fit"):
            raise ValueError("score_mode must be 'in_sample' or 'cross_fit'")
        self.score_mode = score_mode
        self.crossfit_folds = crossfit_folds
        if not (isinstance(init, np.ndarray)
                or init in ("kmeans", "gmm", "bgmm", "fcm", "mechanism")):
            raise ValueError(
                "init must be 'kmeans', 'gmm', 'bgmm', 'fcm', 'mechanism', "
                "or an array of initial labels")
        # 'mechanism' clusters the windows by their LAW COEFFICIENTS rather
        # than by summaries of the data (lrdsr.core.mechanism_space). It reads
        # (X_seq, y_seq), never Z, so the geometry term is untouched.
        self.init = init
        self.mechanism_basis = mechanism_basis
        self.mechanism_nuisance = mechanism_nuisance
        # What a nuisance is constant over -- the link, on real data. Only
        # used by the 'mechanism' initialiser.
        self.mechanism_groups = mechanism_groups
        if geom_metric not in ("euclidean", "mahalanobis"):
            raise ValueError("geom_metric must be 'euclidean' or 'mahalanobis'")
        self.geom_metric = geom_metric
        self.superposition = bool(superposition)
        if superposition_margin < 0.0:
            raise ValueError("superposition_margin must be >= 0")
        self.superposition_margin = float(superposition_margin)
        if max_composites is not None and max_composites < 0:
            raise ValueError("max_composites must be >= 0 or None")
        # None = uncapped; set a small cap when composites are assumed rare.
        self.max_composites = n_clusters if max_composites is None else int(max_composites)
        if residual_scale not in ("per_window", "global"):
            raise ValueError("residual_scale must be 'per_window' or 'global'")
        # 'global' (the default since the 2026-09 revision) standardizes
        # every window by ONE shared scale -- the robust spread of the
        # residuals under the current assignment, i.e. an estimate of the noise
        # level -- so a window a mechanism explains badly actually costs more.
        # 'per_window' is the former default, kept reachable because every
        # window-level number produced before that revision used it: it
        # standardizes each window's residual by its OWN robust scale, which
        # makes the equation term blind to the SIZE of the misfit.
        # See lrdsr.core.losses.aggregate_window_residual.
        self.residual_scale = residual_scale
        # The per-sample penalty of the equation term. "huber" (with
        # robust_delta) is the historical score; any name in
        # lrdsr.core.losses.LOSSES is accepted, and "learned" re-fits the loss
        # every iteration from the residuals of the current assignment
        # (lrdsr.core.losses.learn_loss): a Gaussian, Laplace or Student-t
        # noise model, whichever explains them best, with its own scale. The
        # learned loss needs a shared scale, so it implies residual_scale
        # 'global'.
        if loss != "learned" and loss not in LOSSES:
            raise ValueError(f"loss must be 'learned' or one of {LOSSES}")
        if loss == "learned" and residual_scale != "global":
            raise ValueError("loss='learned' needs residual_scale='global'")
        self.loss = loss
        self.loss_params = dict(loss_params or {})
        # What a residual is divided by before the loss. "sd" (the default
        # since 2026-09-24) is the normal-consistent MAD, so robust_delta and
        # every other loss constant is in units of the noise sd. "mad" is the
        # raw MAD (0.6745 sd), the former default, under which delta = 1.5 was
        # really ~1 sd and the Huber score lost ~10% efficiency -- kept only to
        # reproduce numbers made before the fix.
        if scale_convention not in ("sd", "mad"):
            raise ValueError("scale_convention must be 'sd' or 'mad'")
        self.scale_convention = scale_convention
        self._active_loss = ("huber" if loss == "learned" else loss,
                             dict(self.loss_params))

    def _window_cost(self, y, pred, scale) -> float:
        """One window's equation cost under the loss currently in force."""
        name, params = self._active_loss
        return aggregate_window_residual(y, pred, robust_delta=self.robust_delta,
                                         scale=scale, loss=name, loss_params=params,
                                         scale_convention=self.scale_convention)

    def _make_model_for_cluster(self, k: int, feature_names=None):
        spec = self.mechanism_specs[k] if self.mechanism_specs else {"mode": "unknown"}
        mode = spec.get("mode", "unknown")
        if mode == "unknown":
            return make_symbolic_regressor(
                self.backend, feature_names=feature_names, **self.backend_kwargs
            )
        if mode == "factory":
            return spec["factory"]()
        if mode == "partial":
            return ResidualSymbolicRegressor(
                prior_function=spec["prior_function"],
                residual_model=make_symbolic_regressor(
                    self.backend, feature_names=feature_names, **self.backend_kwargs
                ),
                prior_expression=spec.get("prior_expression"),
            )
        return KnownMechanismRegressor(
            spec["function"], expression_name=spec.get("expression")
        )

    def _geometry_cost(self, Zs, labels, centroids):
        """Per-window geometric assignment cost [W, K].

        "euclidean": squared distance to the cluster mean (spherical).
        "mahalanobis": full-covariance negative log Gaussian density up to a
        constant (0.5*maha + 0.5*logdet - log pi_k) — a hard-EM GMM E-step,
        so elliptical, unequal-spread regimes are costed correctly. Shifted
        to be non-negative; row-wise ordering is unaffected.
        """
        W, d = Zs.shape
        if self.geom_metric == "euclidean":
            return np.column_stack([
                np.sum((Zs - centroids[k]) ** 2, axis=1)
                for k in range(self.n_clusters)
            ])

        geometry = np.empty((W, self.n_clusters), dtype=float)
        for k in range(self.n_clusters):
            members = Zs[labels == k]
            diff = Zs - centroids[k]
            if members.shape[0] > d + 1:
                cov = np.cov(members, rowvar=False) + 1e-6 * np.eye(d)
            elif members.shape[0] > 1:
                cov = np.diag(members.var(axis=0) + 1e-3)
            else:
                cov = np.eye(d)
            _, logdet = np.linalg.slogdet(cov)
            maha = np.einsum("ij,ij->i", diff, np.linalg.solve(cov, diff.T).T)
            pi_k = max(members.shape[0] / W, 1e-12)
            geometry[:, k] = 0.5 * maha + 0.5 * logdet - np.log(pi_k)
        return geometry - geometry.min()

    def _cluster_score(self, model, X_seq, y_seq, windows, scale=None) -> float:
        """Mean per-window aggregated residual of ``model`` on ``windows``."""
        return float(np.mean([
            self._window_cost(y_seq[w], model.predict(X_seq[w]), scale)
            for w in windows
        ]))

    def _shared_residual_scale(self, X_seq, y_seq, labels, models):
        """Robust spread of the residuals under the CURRENT assignment.

        Estimates the noise level from the model each window is currently
        assigned to, so the equation term keeps magnitude information without
        being rescaled by how badly the wrong mechanisms fit. Returns None in
        'per_window' mode, which leaves the historical behaviour untouched.
        """
        if self.residual_scale != "global":
            return None
        residuals = np.concatenate([
            np.asarray(y_seq[w], dtype=float).ravel()
            - np.asarray(models[labels[w]].predict(X_seq[w])).ravel()
            for w in range(X_seq.shape[0])])
        if self.loss == "learned":
            # Learn the noise model, and with it the loss, from the residuals
            # of the current assignment. Its scale replaces the MAD: it is the
            # scale that loss is defined against.
            learned = learn_loss(residuals)
            self._active_loss = (learned.loss, dict(learned.params))
            self.learned_loss_ = learned
            return learned.scale
        if self.scale_convention == "mad":
            return robust_scale(residuals)
        return noise_scale(residuals)

    def _fit_models(self, X_seq, y_seq, labels, feature_names=None, scale=None):
        models = []
        for k in range(self.n_clusters):
            windows = np.where(labels == k)[0]
            if len(windows) < self.min_windows_per_cluster:
                raise RuntimeError(f"Cluster {k} has only {len(windows)} windows")
            Xk = X_seq[windows].reshape(-1, X_seq.shape[-1])
            yk = y_seq[windows].reshape(-1)
            model = self._make_model_for_cluster(k, feature_names=feature_names)
            model.fit(Xk, yk)
            models.append(model)

        if self.superposition:
            models = self._apply_superposition(X_seq, y_seq, labels, models, scale=scale)
        return models

    def _apply_superposition(self, X_seq, y_seq, labels, base_models, scale=None):
        """Replace a cluster's law by a weighted sum of other clusters' laws.

        Second pass over freshly fitted base models: for every 'unknown'
        cluster, test each pair of *other* base models as a superposition
        hypothesis f_k = w_i f_i + w_j f_j + c (weights by least squares) and
        keep it when it explains the cluster's own windows better than its
        free symbolic fit. Components are always base models, so composites
        are never built from composites.
        """
        models = list(base_models)
        candidates = []  # (relative_gain, k, model)
        for k in range(self.n_clusters):
            spec = self.mechanism_specs[k] if self.mechanism_specs else {}
            if spec.get("mode") in ("known", "partial", "factory"):
                continue  # never override declared mechanism knowledge
            windows = np.where(labels == k)[0]
            if len(windows) < self.min_windows_per_cluster:
                continue
            Xk = X_seq[windows].reshape(-1, X_seq.shape[-1])
            yk = y_seq[windows].reshape(-1)
            base_score = self._cluster_score(base_models[k], X_seq, y_seq, windows,
                                             scale=scale)
            if not np.isfinite(base_score) or base_score <= 0:
                continue

            best_model, best_score = None, base_score * (1.0 + self.superposition_margin)
            others = [j for j in range(self.n_clusters) if j != k]
            for a in range(len(others)):
                for b in range(a + 1, len(others)):
                    i, j = others[a], others[b]
                    candidate = SuperpositionRegressor(
                        [base_models[i], base_models[j]],
                        component_labels=[f"cluster {i}", f"cluster {j}"],
                    )
                    try:
                        candidate.fit(Xk, yk)
                        if not candidate.is_valid:
                            continue  # degenerate: a rescaled single law
                        score = self._cluster_score(candidate, X_seq, y_seq, windows,
                                                    scale=scale)
                    except (ValueError, np.linalg.LinAlgError):
                        continue
                    if np.isfinite(score) and score < best_score:
                        best_model, best_score = candidate, score
            if best_model is not None:
                # Rank by how much better the composite explains this cluster
                # than its own free symbolic fit.
                candidates.append(((base_score - best_score) / base_score, k, best_model))

        # Composites are rare by assumption: accept only the strongest few.
        candidates.sort(key=lambda c: -c[0])
        for _, k, model in candidates[: self.max_composites]:
            models[k] = model
        return models

    def _crossfit_equation_scores(self, X_seq, y_seq, labels, equation, feature_names,
                                  scale=None):
        """Replace self-fit equation scores with out-of-fold scores.

        Only the diagonal (window scored under its own current cluster) is
        vulnerable to self-fit bias, so only those entries are recomputed:
        windows of cluster k are split into folds and each fold is scored by
        a temporary model trained on the remaining windows of k. Clusters too
        small to split, and 'known' mechanisms (which fit nothing), keep
        their in-sample scores.
        """
        rng = np.random.default_rng(self.random_state)
        for k in range(self.n_clusters):
            spec = self.mechanism_specs[k] if self.mechanism_specs else {}
            if spec.get("mode") == "known":
                continue
            windows = np.where(labels == k)[0]
            if len(windows) < 2 * self.crossfit_folds:
                continue  # graceful fallback to in-sample for small clusters
            folds = np.array_split(rng.permutation(windows), self.crossfit_folds)
            for fold in folds:
                train = np.setdiff1d(windows, fold)
                model = self._make_model_for_cluster(k, feature_names=feature_names)
                model.fit(
                    X_seq[train].reshape(-1, X_seq.shape[-1]),
                    y_seq[train].reshape(-1),
                )
                for w in fold:
                    equation[w, k] = self._window_cost(
                        y_seq[w], model.predict(X_seq[w]), scale)
        return equation

    def _repair_initial_labels(self, Zs, labels):
        """Make the geometric initialisation satisfy ``min_windows_per_cluster``.

        K-means/GMM/Birch can return a cluster with almost nothing in it,
        and the first symbolic fit then has too little data to fit anything.
        Repairing the initial partition on plain distance-to-centroid keeps
        the loop's own repair (which uses the full assignment cost) for later
        iterations, where it has a cost matrix to work with.
        """
        labels = np.asarray(labels).copy()
        counts = np.bincount(labels, minlength=self.n_clusters)
        if counts.min() >= self.min_windows_per_cluster:
            return labels

        for k in np.where(counts < self.min_windows_per_cluster)[0]:
            members = Zs[labels == k]
            if members.shape[0] == 0:
                # Empty cluster: seed it with the point furthest from all
                # current centroids, so it starts somewhere meaningful.
                present = [j for j in range(self.n_clusters)
                           if np.any(labels == j)]
                centres = np.vstack([Zs[labels == j].mean(axis=0)
                                     for j in present])
                d = ((Zs[:, None, :] - centres[None, :, :]) ** 2).sum(-1).min(1)
                centre = Zs[int(np.argmax(d))]
            else:
                centre = members.mean(axis=0)
            dist = ((Zs - centre) ** 2).sum(axis=1)
            for w in np.argsort(dist):
                if np.sum(labels == k) >= self.min_windows_per_cluster:
                    break
                donor = labels[w]
                if donor == k or np.sum(labels == donor) <= self.min_windows_per_cluster:
                    continue
                labels[w] = k
        return labels

    def _repair_small_clusters(self, labels, total_cost):
        labels = labels.copy()
        counts = np.bincount(labels, minlength=self.n_clusters)
        for k in np.where(counts < self.min_windows_per_cluster)[0]:
            need = self.min_windows_per_cluster - counts[k]
            # Move points for which k is relatively cheap, while avoiding donors
            # that would themselves fall below the minimum size.
            ranking = np.argsort(total_cost[:, k])
            moved = 0
            for w in ranking:
                donor = labels[w]
                donor_count = np.sum(labels == donor)
                if donor == k or donor_count <= self.min_windows_per_cluster:
                    continue
                labels[w] = k
                moved += 1
                if moved >= need:
                    break
        return labels

    def _order_slots(self, labels, slot_order_key):
        """Renumber the initial clusters by an OBSERVABLE per-window statistic.

        ``mechanism_specs`` binds knowledge to a cluster *slot*, but a
        geometric initialiser numbers its clusters arbitrarily, so "slot 1 is
        the rain law" is meaningless unless the slots are given a canonical
        order first. Passing e.g. the mean attenuation of each window as
        ``slot_order_key`` sorts the slots from quietest to loudest, which is
        what makes a declared prior land on the regime it describes.

        The key must be computable from the observations alone -- it is
        applied before any fitting and would otherwise smuggle supervision in.
        Without it the historical (arbitrary) slot numbering is kept.
        """
        key = np.asarray(slot_order_key, dtype=float).reshape(-1)
        if key.shape[0] != labels.shape[0]:
            raise ValueError("slot_order_key must have one value per window")
        means = np.array([
            key[labels == k].mean() if np.any(labels == k) else np.inf
            for k in range(self.n_clusters)
        ])
        order = np.argsort(means)              # old id at position new id
        remap = np.empty(self.n_clusters, dtype=int)
        remap[order] = np.arange(self.n_clusters)
        return remap[labels]

    def fit(
        self,
        X_seq: np.ndarray,
        y_seq: np.ndarray,
        Z: np.ndarray,
        feature_names: list[str] | None = None,
        physics_cost_builder=None,
        true_labels_for_eval: np.ndarray | None = None,
        slot_order_key: np.ndarray | None = None,
    ) -> DCSRResult:
        """Recover the partition and one symbolic law per regime, together.

        The loop is: initialise geometrically, fit a law per cluster, score
        **every** window under **every** law, reassign each window to the law
        that explains it best, repeat until almost nothing moves.

        Step three is what separates this from cluster-then-fit. A window can
        migrate to the mechanism that explains it even when its features say
        it belongs elsewhere, which is the only way to recover regimes that
        overlap in feature space but differ in the law that generated them.

        Parameters
        ----------
        X_seq : (W, n, d)
            The symbolic design, one sequence per window. Laws are fitted and
            evaluated on these columns.
        y_seq : (W, n)
            The response for each window.
        Z : (W, p)
            **Window-level** features, used to initialise the partition and to
            compute the geometry term of the assignment cost. They are
            standardised here, so they need not be scaled by the caller. With
            ``init='mechanism'`` the initial partition comes from the law
            coefficients instead and ``Z`` only feeds the geometry term, which
            ``alpha_geom=0`` switches off.
        feature_names : list[str], optional
            Names for the `d` columns of ``X_seq``; they appear in the
            recovered expressions.
        physics_cost_builder : callable, optional
            ``f(labels, models) -> (W, K)``, an optional weak physical anchor.
            Only consulted when ``lambda_phys > 0``.
        true_labels_for_eval : (W,), optional
            **Scoring only.** Passed to `clustering_metrics` so the iteration
            history can record ARI against the truth. It never reaches the
            cost, the fit or the assignment — on real data it does not exist,
            which is why it is a separate argument rather than a column.
        slot_order_key : (W,), optional
            A per-window quantity used to order the cluster slots after
            initialisation. **Required whenever `mechanism_specs` declares
            knowledge**, because knowledge binds to a slot: if slot 1 is meant
            to be the wetter regime and the initialiser happens to label it 0,
            a declared law lands on the wrong cluster and actively hurts. The
            key must be informative about the thing that distinguishes the
            regimes; an uninformative key silently reintroduces the bug.

        Returns
        -------
        DCSRResult
            ``labels`` (W,), ``models`` (one fitted law per slot),
            ``centroids``, the ``scaler`` fitted on ``Z``, a ``history`` frame
            with one row per iteration, and the final ``total_cost`` (W, K).

        Notes
        -----
        With ``alpha_geom = 0`` assignment is purely mechanism-based. With
        ``alpha_geom = 1`` and Mahalanobis geometry the reassignment step *is*
        a hard-EM GMM E-step, which is the sanity anchor at the other end.
        ``alpha_geom`` is problem-dependent and should be tuned per problem —
        measured optima across this project range from 0.0 to 0.8.
        """
        X_seq = np.asarray(X_seq, dtype=float)
        y_seq = np.asarray(y_seq, dtype=float)
        Z = np.asarray(Z, dtype=float)
        W = X_seq.shape[0]
        if y_seq.shape[0] != W or Z.shape[0] != W:
            raise ValueError("X_seq, y_seq and Z must have the same number of windows")

        scaler = StandardScaler()
        Zs = scaler.fit_transform(Z)
        # Geometric initialization only — labels are refined by the
        # mechanism-aware reassignment loop below. "gmm"/"bgmm" (full
        # covariance) handle elliptical, unequal-spread regimes that K-means
        # splits badly; "fcm" starts from soft centroid memberships.
        if isinstance(self.init, np.ndarray):
            # An explicit starting partition. The loop is an operator on a
            # partition, and this is the only way to ask what it does to one
            # of known quality -- ``experiments.estimator.loop_value`` uses
            # it and nothing else does. It is never a way to feed labels to a
            # fit: what is passed there is derived from the data, and the
            # module says so where it builds it.
            labels = np.asarray(self.init, int).copy()
            if labels.shape != (len(y_seq),):
                raise ValueError(
                    f"init array must have one label per window: "
                    f"got {labels.shape}, expected {(len(y_seq),)}")
        elif self.init == "mechanism":
            from sklearn.cluster import KMeans as _KMeans
            # No standardisation: the whitening already made the within-regime
            # covariance isotropic, and rescaling the columns would undo it.
            S = mechanism_features(
                X_seq, y_seq, basis=self.mechanism_basis,
                nuisance=self.mechanism_nuisance,
                groups=self.mechanism_groups, feature_names=feature_names,
            )
            labels = _KMeans(n_clusters=self.n_clusters, n_init=30,
                             random_state=self.random_state).fit_predict(S)
        elif self.init == "gmm":
            labels = GaussianMixture(
                n_components=self.n_clusters, covariance_type="full",
                n_init=5, random_state=self.random_state,
            ).fit_predict(Zs)
        elif self.init == "bgmm":
            labels = BayesianGaussianMixture(
                n_components=self.n_clusters, covariance_type="full",
                n_init=3, max_iter=300, random_state=self.random_state,
            ).fit_predict(Zs)
        elif self.init == "fcm":
            from .baselines import fuzzy_cmeans
            labels, _ = fuzzy_cmeans(Zs, self.n_clusters, seed=self.random_state)
        else:
            labels = KMeans(
                n_clusters=self.n_clusters, n_init=30, random_state=self.random_state
            ).fit_predict(Zs)

        labels = self._repair_initial_labels(Zs, labels)
        if slot_order_key is not None:
            labels = self._order_slots(labels, slot_order_key)

        history = []
        total = None
        models = None
        centroids = None
        scale = None  # shared residual scale; only used in 'global' mode

        for iteration in range(self.max_iter):
            centroids = np.vstack([Zs[labels == k].mean(axis=0) for k in range(self.n_clusters)])
            models = self._fit_models(X_seq, y_seq, labels,
                                      feature_names=feature_names, scale=scale)
            scale = self._shared_residual_scale(X_seq, y_seq, labels, models)

            geometry = self._geometry_cost(Zs, labels, centroids)

            equation = np.zeros((W, self.n_clusters), dtype=float)
            for w in range(W):
                for k, model in enumerate(models):
                    equation[w, k] = self._window_cost(
                        y_seq[w], model.predict(X_seq[w]), scale)

            if self.score_mode == "cross_fit":
                equation = self._crossfit_equation_scores(
                    X_seq, y_seq, labels, equation, feature_names, scale=scale
                )

            physics = None
            if physics_cost_builder is not None and self.lambda_phys > 0:
                physics = physics_cost_builder(labels=labels, models=models)

            complexity = np.tile(
                np.array([m.complexity() for m in models])[None, :],
                (W, 1),
            )

            total = joint_cost(
                geometry,
                equation,
                alpha_geom=self.alpha_geom,
                physics_cost=physics,
                lambda_phys=self.lambda_phys,
                complexity_cost=complexity,
                beta_complexity=self.beta_complexity,
            )

            new_labels = np.argmin(total, axis=1)
            new_labels = self._repair_small_clusters(new_labels, total)
            changed = float(np.mean(new_labels != labels))

            row = {
                "iteration": iteration,
                "changed_fraction": changed,
                "mean_assignment_cost": float(np.mean(np.min(total, axis=1))),
                "min_cluster_size": int(np.min(np.bincount(new_labels, minlength=self.n_clusters))),
                "loss": self._active_loss[0],
            }
            if self.loss == "learned" and "nu" in self._active_loss[1]:
                row["loss_nu"] = float(self._active_loss[1]["nu"])
            if true_labels_for_eval is not None:
                from .evaluation import clustering_metrics
                row.update(clustering_metrics(true_labels_for_eval, new_labels))
            history.append(row)
            labels = new_labels
            if changed < self.tol:
                break

        centroids = np.vstack([Zs[labels == k].mean(axis=0) for k in range(self.n_clusters)])
        models = self._fit_models(X_seq, y_seq, labels, feature_names=feature_names,
                                  scale=scale)
        return DCSRResult(
            labels=labels,
            models=models,
            centroids=centroids,
            scaler=scaler,
            history=pd.DataFrame(history),
            total_cost=total,
        )


class RowDCSR:
    """Convenience wrapper: each row is treated as a one-point group."""
    def __init__(self, **kwargs):
        self.grouped = GroupedDCSR(**kwargs)

    def fit(self, X, y, Z, feature_names=None, true_labels_for_eval=None,
            slot_order_key=None):
        X_seq = np.asarray(X)[:, None, :]
        y_seq = np.asarray(y)[:, None]
        return self.grouped.fit(
            X_seq,
            y_seq,
            Z,
            feature_names=feature_names,
            true_labels_for_eval=true_labels_for_eval,
            slot_order_key=slot_order_key,
        )
