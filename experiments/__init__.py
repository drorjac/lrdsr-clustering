"""The experiments, one sub-package per question.

``theory``      the ceiling: what any method can do, before any method
                exists. V1-V4 and V7, oracle only.
``estimator``   does LR-DSR reach it when the geometry carries nothing;
                what the symbolic search actually does; and what the
                alternating loop is for.
``functions``   synthetic laws with a knob each: what declared knowledge is
                worth, and three classification shapes (F1 competing
                families, F2 a heterogeneous null, F3 a converging pair).

Each writes ``results/<block>/``. Nothing here reads a real measurement:
every law in this project was written down, so every answer can be checked
against the truth rather than argued.
"""
