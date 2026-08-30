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

The covariance those surfaces consume is built here as well:
:func:`snapshot_covariance` is the ``(d dH)/L`` average behind both
:func:`uacpy.acoustic_signal.sample_covariance` and :func:`uacpy.sonar.csdm`,
so the estimate and the checks that make it meaningful are one implementation
under two public names.

This module sits in ``core`` because ``acoustic_signal`` and ``sonar`` both
depend on ``core`` and ``core`` must depend on neither.
"""

import numpy as np

from uacpy.core.exceptions import ConfigurationError


def snapshot_covariance(snapshots, caller: str) -> np.ndarray:
    """``K = (d dH) / L`` for an ``(N, L)`` snapshot matrix, with its guards.

    Backs :func:`uacpy.acoustic_signal.sample_covariance` and
    :func:`uacpy.sonar.csdm`: one average, one set of checks, two public
    names that add their own policy on top (``sample_covariance`` offers
    diagonal loading). The three checks travel with the arithmetic because
    each of them is a property of the average itself:

    * 2-D shape - a flat array is not a snapshot matrix;
    * ``N >= 1`` - a zero-sensor matrix averages to an empty ``(0, 0)``
      covariance, which every beamformer then scores as a silent all-zero
      surface - the same shape of failure as the NaN one below;
    * ``L >= 1`` - the average over ``L`` snapshots is undefined at ``L = 0``
      and every entry would be NaN;
    * finiteness - one NaN sample poisons every entry of ``K``, and the
      Bartlett/MVDR surface built from it comes back all-NaN, which
      :func:`uacpy.sonar.bartlett` reports with no diagnostic at all.

    ``caller`` names the public function in every message.
    """
    try:
        d = np.asarray(snapshots, dtype=complex)
    except (TypeError, ValueError) as exc:
        # A Field (or any object numpy cannot make a numeric array of) reaches
        # the cast as "must be real number, not Field", which names nothing the
        # caller can act on. The estimators here are array-in by design and
        # ``.data`` is the documented bridge.
        what = type(snapshots).__name__
        bridge = ("Pass {}.data.".format(what)
                  if hasattr(snapshots, "data") and hasattr(snapshots, "coords")
                  else "Pass a numeric array.")
        raise ConfigurationError(
            f"{caller}: snapshots must be a numeric array; got {what}. "
            f"{bridge}") from exc
    if d.ndim != 2:
        raise ConfigurationError(
            f"{caller}: snapshots must be 2-D (n_sensors, n_snapshots); got "
            f"shape {d.shape}. A single snapshot is column-shaped: "
            f"d[:, None].")
    if d.shape[0] == 0:
        raise ConfigurationError(
            f"{caller}: snapshots has zero sensor rows (shape {d.shape}); "
            f"the covariance would be an empty (0, 0) matrix and every "
            f"beamformer surface built from it silently all-zero. Pass at "
            f"least one sensor row.")
    if d.shape[1] == 0:
        raise ConfigurationError(
            f"{caller}: snapshots has zero snapshot columns (shape {d.shape}); "
            f"the average over L snapshots is undefined at L=0 and every "
            f"matrix entry would be NaN. Pass at least one snapshot column.")
    bad = ~np.isfinite(d)
    if bad.any():
        raise ConfigurationError(
            f"{caller}: snapshots contain NaN or Inf (a dead hydrophone, or a "
            f"shadow-zone column of a modelled field), which would silently "
            f"contaminate every entry of the covariance and leave the "
            f"Bartlett/MVDR surface all-NaN; clean the snapshots first. Got "
            f"{int(np.count_nonzero(bad))} non-finite value(s) of {d.size}, "
            f"first at flat index {int(np.argmax(bad))}.")
    return (d @ d.conj().T) / d.shape[1]


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
