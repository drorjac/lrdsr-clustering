"""The losses block: what the per-sample loss costs, and whether it can be learned.

``lrdsr.theory.losses``  V8 -- the ceiling for a decision made with a loss
                         other than the likelihood ratio: ``Q(sqrt(n rho eta)/2)``.
``estimator``            the same question on the estimator: GroupedDCSR with
                         six losses and SoftLRDSR with two noise models, under
                         four noise laws, against two oracles.
``learning``             learning the loss: what ``loss='learned'`` and the
                         Student-t EM converge to, and whether ``learn_loss``
                         names the noise family from residuals alone.
"""
