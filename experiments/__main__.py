"""Run the experiments, block by block.

    python -m experiments list        # what each block answers
    python -m experiments theory      # the ceiling: V1-V4, V7. No estimator.
    python -m experiments estimator   # does the method reach it, and what is
                                      #   the loop for
    python -m experiments functions   # what declared knowledge is worth, and
                                      #   three classification shapes
    python -m experiments all

Each block writes CSVs to ``results/<block>/``. The figures and every number
in the README are built from those CSVs by ``python -m analysis.report``.
"""
from __future__ import annotations

import importlib
import sys
import time

SECTIONS = {
    "theory": ("experiments.theory.run",
               "the ceiling: is the separation theory right? (V1-V4, V7)"),
    "estimator": ("experiments.estimator.run",
                  "does LR-DSR reach the ceiling when geometry cannot help, "
                  "what does the symbolic search actually do, and what is "
                  "the alternating loop for"),
    "functions": ("experiments.functions.run",
                  "what is declared knowledge worth (known/known-form/"
                  "unknown), and the three classification shapes F1/F2/F3"),
}


def main(argv: list[str]) -> int:
    name = argv[0] if argv else "list"
    if name in ("list", "-h", "--help"):
        print(__doc__.strip() + "\n")
        for key, (_, what) in SECTIONS.items():
            print(f"  {key:12s} {what}")
        return 0
    todo = list(SECTIONS) if name == "all" else [name]
    for key in todo:
        if key not in SECTIONS:
            print(f"unknown block: {key}. Try 'list'.")
            return 2
        t0 = time.time()
        print(f"\n{'=' * 74}\n  {key}\n{'=' * 74}", flush=True)
        importlib.import_module(SECTIONS[key][0]).run()
        print(f"\n[{key}] done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
