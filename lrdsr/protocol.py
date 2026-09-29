"""The experimental protocol every run in this project follows.

These are not tuning knobs -- they are the rules a result has to be produced
under for it to count.

The seed rule is the important one. Selecting on the seed you report is how a
sweep becomes a result that will not reproduce, so the two sets are named
separately and never mixed: tune, sweep and choose on ``TUNE_SEEDS``, then
report on ``REPORT_SEEDS`` and report whatever comes out.
"""
from __future__ import annotations

#: Seeds to tune, sweep and select on.
TUNE_SEEDS = (3, 7, 19)

#: Seeds to report. Never select on these, and never quietly widen the set --
#: ``REPORT_SEEDS[0]`` is the single-seed default where one is needed.
REPORT_SEEDS = (11, 23, 42)
