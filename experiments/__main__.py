"""Run the experiments, block by block.

    python -m experiments list        # what each block answers
    python -m experiments theory      # the ceiling: V1-V4, V7. No estimator.
    python -m experiments estimator   # does the method reach it, and what is
                                      #   the loop for
    python -m experiments functions   # what declared knowledge is worth, and
                                      #   three classification shapes
    python -m experiments problems    # a zoo of twelve problems, each hard for
                                      #   its own reason
    python -m experiments losses      # which loss, when: the efficiency theory
                                      #   (V8) and a learned loss
    python -m experiments online      # clustering in real time, and the delay
                                      #   theory of sequential detection (V9)
    python -m experiments realdata    # two real datasets: bike sharing and
                                      #   highway traffic, day by day, and the
                                      #   traffic days with sensor gaps
    python -m experiments kernel      # mechanism space over an RKHS: the zoo's
                                      #   one failure, and a label-free rank
    python -m experiments classify    # classification: the learning curve
                                      #   (V10) and 24 UCR datasets
    python -m experiments wind        # wind power curves, every window its
                                      #   own design (Kelmarsh SCADA)
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
    "problems": ("experiments.problems.run",
                 "twelve problems, each breaking one assumption: out-of-library "
                 "laws, gaps outside the span, K = 3 and 4, two inputs, a "
                 "rescaled copy, matched moments"),
    "losses": ("experiments.losses.run",
               "which loss, when: the efficiency of a robust loss (V8), every "
               "loss under four noise laws, and a loss learned from residuals"),
    "online": ("experiments.online.run",
               "clustering in real time: sequential-detection theory (V9), "
               "streams with regime birth and drift, latency"),
    "realdata": ("experiments.realdata.run",
                 "two public datasets, one window per day: UCI bike sharing and "
                 "I-94 traffic -- batch, soft, real time, what K to use, and the "
                 "days with sensor gaps"),
    "kernel": ("experiments.kernel.run",
               "mechanism space over an RKHS instead of a term library: does it "
               "repair the gap outside the span, and a label-free rank rule"),
    "classify": ("experiments.classify.run",
                 "classification: the plug-in learning curve of a law classifier "
                 "(V10), and 24 UCR datasets -- full, few-shot, irregular, clustering"),
    "wind": ("experiments.wind.run",
             "wind power curves: six-hour turbine blocks, each with its own wind "
             "speeds, against the method of bins and physical proxies"),
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
