"""The synthetic-law battery: what knowledge is worth, and three shapes.

    python -m experiments functions
"""
from __future__ import annotations

from experiments.functions import knowledge, scenarios


def run(args=None) -> None:
    print(f"\n{'=' * 74}\n  the knowledge ladder\n{'=' * 74}", flush=True)
    knowledge.run()
    print(f"\n{'=' * 74}\n  the three scenario shapes\n{'=' * 74}", flush=True)
    scenarios.run()
    print(f"\n[functions] CSVs -> {knowledge.RESULTS}")


if __name__ == "__main__":
    run()
