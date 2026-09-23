"""the functions block -- synthetic laws: what is knowable, and what is worth knowing.

    python -m experiments functions

====================  =====================================================
``knowledge``         three laws (f1 quadratic, f2 sine, f3 cubic), fitted
                      under three framings of declared knowledge -- known,
                      known form, unknown -- swept over noise and over
                      window length
``scenarios``         three classification SHAPES with a knob each: F1 two
                      competing parametric families, F2 null against a
                      heterogeneous null, F3 multi-class with the hard
                      pairs explicit
====================  =====================================================

Simulation only: nothing here imports ``lrdsr.cml``, and the separation is
a knob rather than a consequence of the physics. Seeds {11, 23, 42}.
"""
