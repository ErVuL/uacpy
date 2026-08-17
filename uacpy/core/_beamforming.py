"""The single Bartlett/MVDR numerical core.

Three public surfaces run the same two operations over a Hermitian
covariance ``R`` and a bank of weight vectors ``W``:

* :func:`uacpy.acoustic_signal.bartlett_spectrum` / ``mvdr_spectrum`` —
  plane-wave steering vectors over one covariance;
* :func:`uacpy.sonar.bartlett` / ``mvdr`` — unit-norm replica banks over a
  measured CSDM, trace-normalised / max-scaled;
* :meth:`uacpy.core.results.Covariance.bartlett` / ``mvdr`` — per-frequency
  replica grids over OASN's ``.xsm`` covariance, unnormalised.

Each keeps its own policy (weight normalisation, loading default, NaN/inf
conventions, output scaling) — the physics they share lives here, so the
three cannot drift apart numerically the way three hand-rolled einsums did.

This module sits in ``core`` because ``acoustic_signal`` and ``sonar`` both
depend on ``core`` and ``core`` must depend on neither.
"""

import numpy as np


def quadratic_form(R: np.ndarray, W: np.ndarray) -> np.ndarray:
    """``Re( w_p^H · R · w_p )`` for every row ``w_p`` of ``W``.

    The Bartlett power (with ``R`` the covariance) and the MVDR denominator
    (with ``R`` the loaded inverse) are both this contraction. ``W`` is
    ``(n_points, n_elements)``; returns ``(n_points,)`` float. The imaginary
    part of a Hermitian quadratic form is round-off, hence the ``Re``.
    """
    return np.real(np.einsum('pi,ij,pj->p', np.conj(W), R, W))


def loaded_inverse(R: np.ndarray, loading: float) -> np.ndarray:
    """``inv( R + loading · (tr R / N) · I )`` — the MVDR-stabilised inverse.

    ``loading`` is a *fraction of the average eigenvalue* ``tr(R)/N``, so it
    vanishes with the trace: callers must reject a powerless ``R``
    (``tr R <= 0``) before calling, each with its own diagnostic — that
    policy is deliberately not here.
    """
    R = np.asarray(R, dtype=complex)
    n = R.shape[0]
    if loading:
        R = R + loading * (np.real(np.trace(R)) / n) * np.eye(n)
    return np.linalg.inv(R)
