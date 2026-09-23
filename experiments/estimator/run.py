"""Does the method reach the ceiling, and what is the loop for?

    python -m experiments estimator

====================  =====================================================
``benchmark``         four law pairs with identical input distributions:
                      LR-DSR against the oracle, against the data-space
                      initialiser it replaced, and against every geometric
                      clustering
``shared_component``  the invariance V4 proves for the oracle, tested on the
                      estimator: a shared term swept to 381x the gap, inside
                      and outside the symbolic library
``search``            the process rather than the score: the greedy search
                      term by term, what complexity buys against the true
                      law's own held-out error, the assignment's complexity
                      penalty swept, and the loop iteration by iteration
``loop_value``        the alternating loop characterised as an operator:
                      partitions of controlled quality in, and what
                      comes out
====================  =====================================================

Writes ``results/estimator/``. Reported on seeds {11, 23, 42}; nothing is
tuned here (``alpha_geom = 0`` is the principled value when the geometry
carries nothing about the regime).
"""
from __future__ import annotations

from experiments.estimator import (
    benchmark,
    loop_value,
    search,
    shared_component,
)


def run(args=None) -> None:
    print("\n=== the pair battery ===", flush=True)
    benchmark.run_benchmark()
    benchmark.recovered_laws()
    print("\n=== the shared-component sweep ===", flush=True)
    shared_component.run_sweep()
    print("\n=== the search itself ===", flush=True)
    search.run()
    print("\n=== what the alternating loop is for ===", flush=True)
    loop_value.run()
    print(f"\n[estimator] CSVs -> {benchmark.RESULTS}")


if __name__ == "__main__":
    run()
