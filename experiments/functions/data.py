"""The three-law function mixture: the benchmark this project started from.

``f1`` quadratic, ``f2`` sine, ``f3`` cubic -- three regimes whose geometry
overlaps (the ``Z`` blobs sit 1.2 apart with spread ``geometry_overlap``) and
whose laws are unrelated. A clustering that reads only ``Z`` cannot do better
than that overlap allows; the laws are the only thing that separates them
cleanly, which is the whole point.

``simulate_function_mixture`` is the original row-level generator, one sample
per regime draw. ``simulate_function_windows`` is the same three laws in the
WINDOW setting the rest of the paper uses: ``n`` samples share a label. The
two together are what section 4.5 sweeps, because how much declared knowledge is
worth depends almost entirely on ``n``.

Regimes share overlapping geometry but obey different equations. A regime is
specified as a list of base-function terms and generates the **linear
combination of its components** — the mix-phenomena layer: a composite regime
"0.5*quadratic+2*sine" obeys f(x) = 0.5*f_quadratic(x) + 2*f_sine(x), and
"quadratic+sine" is the unweighted sum. The default three single-component
regimes reproduce the original benchmark exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The three laws, ``f1``--``f3`` in the text.
# Base-function library: name -> (printable equation, callable).
# Composite regimes are element-wise sums of these components.
BASE_FUNCTIONS = {
    "quadratic": ("0.8*x^2 + 1.5*x + 0.5", lambda x: 0.8 * x**2 + 1.5 * x + 0.5),
    "sine": ("2.5*sin(2*x) - 0.5", lambda x: 2.5 * np.sin(2.0 * x) - 0.5),
    "cubic": ("0.35*x^3 - 1.2", lambda x: 0.35 * x**3 - 1.2),
    # A second trig function, close in kind to "sine" (same family, different
    # frequency/phase) rather than a different shape entirely -- deliberately
    # added so "sine" and "cosine" make a HARD pair, and "sine+cosine" a
    # composite regime that shares structure with both of its parents. Used
    # by the notebook's harder-scenario stages; not part of DEFAULT_COMPONENTS
    # so the frozen 3-regime benchmark is untouched.
    "cosine": ("1.8*cos(1.5*x) + 0.3", lambda x: 1.8 * np.cos(1.5 * x) + 0.3),
}

DEFAULT_COMPONENTS = [["quadratic"], ["sine"], ["cubic"]]


def _parse_term(term: str) -> tuple[str, float]:
    """Parse "name" or "weight*name" into (name, weight)."""
    term = term.strip()
    if "*" in term:
        weight_str, name = term.split("*", 1)
        try:
            weight = float(weight_str)
        except ValueError:
            raise ValueError(f"Invalid weight in term {term!r}") from None
        if weight == 0.0:
            raise ValueError(f"Zero weight in term {term!r}")
        return name.strip(), weight
    return term, 1.0


def _normalize_components(components) -> list[list[tuple[str, float]]]:
    """Accept ["0.5*quadratic+2*sine", ...] or [["quadratic","sine"], ...].

    Each regime becomes a list of (base name, weight) terms; the regime's
    law is the linear combination sum(w * f_name).
    """
    out = []
    for spec in components:
        raw = spec.split("+") if isinstance(spec, str) else list(spec)
        terms = [_parse_term(p) for p in raw]
        names = [name for name, _ in terms]
        unknown = [n for n in names if n not in BASE_FUNCTIONS]
        if unknown:
            raise ValueError(f"Unknown base functions: {unknown} "
                             f"(available: {sorted(BASE_FUNCTIONS)})")
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate components in regime spec {spec!r}")
        out.append(terms)
    return out


def _regime_centers(n_regimes: int) -> np.ndarray:
    if n_regimes == 3:  # legacy layout, preserved exactly
        return np.array([[-1.2, 0.0], [0.0, 0.8], [1.2, 0.0]])
    angles = 2.0 * np.pi * np.arange(n_regimes) / n_regimes
    return 1.2 * np.column_stack([np.cos(angles), np.sin(angles)])


@dataclass
class FunctionMixtureData:
    X: np.ndarray
    Z: np.ndarray
    y: np.ndarray
    labels: np.ndarray
    true_equations: list[str]


def simulate_function_mixture(
    n_per_regime: int = 300,
    noise_std: float = 0.12,
    geometry_overlap: float = 1.0,
    irrelevant_dims: int = 0,
    imbalance: tuple[float, ...] | None = None,
    components: list | None = None,
    seed: int = 7,
) -> FunctionMixtureData:
    rng = np.random.default_rng(seed)
    components = _normalize_components(components or DEFAULT_COMPONENTS)
    n_regimes = len(components)
    centers = _regime_centers(n_regimes)
    imbalance = imbalance or (1.0,) * n_regimes
    if len(imbalance) != n_regimes:
        raise ValueError("imbalance must have one entry per regime")

    equations = [
        " + ".join(
            BASE_FUNCTIONS[name][0] if weight == 1.0
            else f"{weight:g}*({BASE_FUNCTIONS[name][0]})"
            for name, weight in terms
        )
        for terms in components
    ]

    Xs, Zs, ys, labels = [], [], [], []
    for k, terms in enumerate(components):
        n = max(30, int(round(n_per_regime * imbalance[k])))
        x = rng.uniform(-2.2, 2.2, n)
        z = rng.normal(centers[k], geometry_overlap, size=(n, 2))
        if irrelevant_dims:
            nuisance = rng.normal(0, 1, size=(n, irrelevant_dims))
            z = np.column_stack([z, nuisance])

        clean = np.sum([weight * BASE_FUNCTIONS[name][1](x)
                        for name, weight in terms], axis=0)

        y = clean + rng.normal(0, noise_std, n)
        Xs.append(x[:, None])
        Zs.append(z)
        ys.append(y)
        labels.append(np.full(n, k, dtype=int))

    return FunctionMixtureData(
        X=np.vstack(Xs),
        Z=np.vstack(Zs),
        y=np.concatenate(ys),
        labels=np.concatenate(labels),
        true_equations=equations,
    )


#: The names the paper uses for the three default regimes.
REGIME_LABEL = {"quadratic": "$f_1$ quadratic", "sine": "$f_2$ sine",
                "cubic": "$f_3$ cubic"}


def simulate_function_windows(
    n_windows: int = 150,
    window_len: int = 8,
    noise_std: float = 0.6,
    geometry_overlap: float = 1.0,
    components: list | None = None,
    seed: int = 7,
):
    """The same three laws, with ``window_len`` samples sharing each label.

    ``window_len = 1`` reproduces the row-level benchmark exactly, which is
    what makes the sweep over ``n`` a like-for-like comparison rather than
    two different experiments.

    Returns ``(X_seq, y_seq, Z, labels, equations)`` with ``X_seq``
    ``(W, n, 1)`` and ``Z`` the window mean of the row-level geometry.
    """
    rng = np.random.default_rng(seed)
    terms_per_regime = _normalize_components(components or DEFAULT_COMPONENTS)
    k_regimes = len(terms_per_regime)
    centers = _regime_centers(k_regimes)
    equations = [
        " + ".join(BASE_FUNCTIONS[name][0] if w == 1.0
                   else f"{w:g}*({BASE_FUNCTIONS[name][0]})" for name, w in terms)
        for terms in terms_per_regime
    ]

    labels = rng.integers(0, k_regimes, size=n_windows)
    x = rng.uniform(-2.2, 2.2, size=(n_windows, window_len))
    clean = np.zeros_like(x)
    for k, terms in enumerate(terms_per_regime):
        m = labels == k
        if not np.any(m):
            continue
        clean[m] = sum(w * BASE_FUNCTIONS[name][1](x[m]) for name, w in terms)
    y = clean + rng.normal(0.0, noise_std, size=x.shape)
    # The geometry a clustering would see. The spread is per WINDOW and does
    # NOT shrink with ``window_len``: windowing the row-level draw would give
    # ``overlap / sqrt(n)``, which makes the geometry easier exactly as the
    # law evidence grows and confounds the two. Holding it fixed keeps the
    # geometric baseline constant across n (~0.28 ARI here), so a sweep over
    # n measures what n does to the EVIDENCE ABOUT THE LAW and nothing else.
    Z = centers[labels] + rng.normal(0.0, geometry_overlap,
                                     size=(n_windows, 2))
    return x[:, :, None], y, Z, labels, equations
