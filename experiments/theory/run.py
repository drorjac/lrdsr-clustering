"""Is the separation theory right?

    python -m experiments theory

Pure simulation, no LR-DSR: every quantity here is closed-form or a
Monte-Carlo **oracle** that knows the true laws. This block establishes the
ceiling the rest of the project is measured against, so nothing in it may
depend on an estimator.

====  ===================================================================
V1    the exact random-design error ``E_x[Q(sqrt(D_n)/2 sigma)]`` matches
      simulation, for three gap shapes
V2    the population formula ``Q(sqrt(n rho)/2)`` is always optimistic,
      and the extra samples needed grow linearly in ``kappa``
V2.4  the one point that looked off the kappa line, rerun on 20 seeds
V3    with ``K > 2`` laws the error sits inside the union/pairwise sandwich
V4    a common component 50x the gap changes nothing
V6    the headline figure: identical ``n rho``, different difficulty
V7    mechanism space attains the bound, and the separation is measurable
      on unlabelled windows
====  ===================================================================

The block order is ``lrdsr.theory.verification.BLOCKS``, so adding one there
adds it here. The parent project's ``V5`` instantiated ``rho(L)`` for a
particular physics and is not part of this one.

Writes ``results/theory/``. The implementation is ``lrdsr.theory.verification``.
"""
from __future__ import annotations

from lrdsr.theory import verification as V


def run(args=None) -> None:
    V.run_all(args)
    print(f"\n[theory] CSVs -> {V.RESULTS}")


if __name__ == "__main__":
    run()
