"""What does the loss cost, and can it be learned?

    python -m experiments losses

====================  =====================================================
V8                    the ceiling for a loss that is not the likelihood
                      ratio: ``Q(sqrt(n rho eta)/2)``, ``eta`` the loss's
                      efficiency under the noise, checked by simulation
                      across 7 losses x 5 noise laws x (n, n rho)
``huber_delta``       where the default Huber constant comes from: a sweep
                      on the TUNING seeds {3, 7, 19}
``estimator``         GroupedDCSR with six losses and SoftLRDSR with two
                      noise models, on the pair battery under four noise laws
``learning``          learning the loss: what ``loss='learned'`` and the
                      Student-t EM converge to, and whether the family can be
                      named from residuals alone
====================  =====================================================

Writes ``results/losses/``. ``huber_delta`` tunes on {3, 7, 19}; everything
else is reported on {11, 23, 42}.
"""
from __future__ import annotations

import time
import warnings

from experiments.losses import estimator, huber_delta, learning
from lrdsr.theory import losses as V8


def run(args=None) -> None:
    t0 = time.time()
    with warnings.catch_warnings():
        # quad's roundoff notices on the heavy-tailed integrands: the
        # integrals are checked against the closed forms in the tests
        warnings.simplefilter("ignore")
        print("\n=== V8: the efficiency of a loss ===", flush=True)
        V8.run_v8()
        print(f"  [{(time.time() - t0) / 60:.1f} min]", flush=True)
        print("\n=== the default Huber constant (tuning seeds) ===", flush=True)
        huber_delta.run()
        print(f"  [{(time.time() - t0) / 60:.1f} min]", flush=True)
        print("\n=== the estimator under non-Gaussian noise ===", flush=True)
        estimator.run()
        print(f"  [{(time.time() - t0) / 60:.1f} min]", flush=True)
        print("\n=== learning the loss ===", flush=True)
        learning.run()
    print(f"\n[losses] CSVs -> {V8.RESULTS}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    run()
