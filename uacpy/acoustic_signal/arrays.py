"""Array processing: steering vectors and conventional/adaptive beamforming.

Steering-vector construction, the conventional plane-wave beamformer, and the
data-adaptive MVDR (Capon) / subspace MUSIC estimators that operate on a spatial
covariance matrix, plus array-shading tapers.

References
----------
Van Trees, H.L. *Optimum Array Processing* (Part IV) — MVDR, MUSIC.
Capon, J. (1969). High-resolution frequency-wavenumber spectrum analysis.
Schmidt, R.O. (1986). Multiple emitter location and signal parameter estimation.
"""

from __future__ import annotations

import warnings
from collections import namedtuple
from typing import Optional

import numpy as np
from scipy.signal import get_window

from uacpy.core.constants import DEFAULT_SOUND_SPEED
from uacpy.core.exceptions import ConfigurationError
from uacpy._log import log_message

BeamformResult = namedtuple("BeamformResult", "snr angles peak_snr")


def steering_vectors(positions_m, angles_deg, frequency: float,
                     c: float = DEFAULT_SOUND_SPEED):
    """Unit plane-wave steering vectors for a line array.

    ``e_n(theta) = exp(j*k*z_n*sin(theta)) / sqrt(N)`` with ``k = 2*pi*f/c``,
    ``theta`` measured from broadside. ``positions_m`` are element coordinates
    along the array axis (m). The ``+j`` sign convention matches the
    Acoustics-Toolbox ``planewave_rep.m`` reference.

    Returns
    -------
    ndarray
        Shape ``(n_angles, n_elements)``, unit-norm per row.
    """
    z = np.asarray(positions_m, dtype=float)
    angles = np.atleast_1d(np.asarray(angles_deg, dtype=float))
    k = 2.0 * np.pi * float(frequency) / float(c)
    phase = np.outer(np.sin(np.deg2rad(angles)), z)
    e = np.exp(1j * k * phase)
    return e / np.sqrt(z.size)


def sample_covariance(snapshots, *, diagonal_loading: float = 0.0):
    """Spatial covariance ``R = <x x^H>`` from snapshots.

    Parameters
    ----------
    snapshots : ndarray
        Shape ``(n_elements, n_snapshots)`` complex array data.
    diagonal_loading : float
        Fraction of ``trace(R)/N`` added to the diagonal for robustness.

    Returns
    -------
    ndarray
        Hermitian covariance matrix ``(n_elements, n_elements)``.
    """
    x = np.asarray(snapshots, dtype=complex)
    if x.ndim != 2:
        raise ConfigurationError("sample_covariance: snapshots must be 2-D (N, snaps)")
    n, m = x.shape
    r = (x @ x.conj().T) / m
    if diagonal_loading > 0.0:
        r = r + diagonal_loading * (np.trace(r).real / n) * np.eye(n)
    return r


def bartlett_spectrum(R, steering):
    """Conventional (Bartlett) beamformer power vs angle: ``e^H R e``.

    ``steering`` is the ``(n_angles, n_elements)`` matrix from
    :func:`steering_vectors` — one steering vector per scan angle.
    """
    e = np.asarray(steering, dtype=complex)
    return np.real(np.einsum("ai,ij,aj->a", e.conj(), R, e))


def _powerless_covariance(R, caller: str) -> bool:
    """True when ``R`` carries no power, leaving a normalised spectrum undefined.

    ``diagonal_loading`` is a *fraction of* ``trace(R)/N``, so it vanishes with
    the trace: it stabilises a rank-deficient covariance that still carries
    power, but cannot rescue an all-zero one. That case is ordinary data, not a
    contrived input — ``sample_covariance`` of a silent segment (a dead
    element, a stretch of digital silence) returns exactly it. Left alone, MVDR
    raises a bare ``numpy`` ``LinAlgError`` and MUSIC returns a finite *uniform*
    pseudospectrum, which is worse: it looks like an answer.
    """
    trace = np.trace(R).real
    scale = trace / R.shape[0]
    if np.isfinite(scale) and scale > 0.0:
        return False
    warnings.warn(
        f"{caller}: the covariance carries no power (trace={trace:g}), so the "
        f"spectrum is undefined; returning NaN.",
        UserWarning, stacklevel=3,
    )
    return True


def mvdr_spectrum(R, steering, *, diagonal_loading: float = 1e-6):
    """MVDR / Capon power vs angle: ``1 / (e^H R^-1 e)``.

    ``steering`` is the ``(n_angles, n_elements)`` matrix from
    :func:`steering_vectors`. ``diagonal_loading`` (fraction of
    ``trace(R)/N``) stabilises the inverse for rank-deficient or
    snapshot-starved covariances — but only while ``R`` still carries power;
    see :func:`_powerless_covariance`.
    """
    R = np.asarray(R, dtype=complex)
    n = R.shape[0]
    e = np.asarray(steering, dtype=complex)
    if _powerless_covariance(R, "mvdr_spectrum"):
        return np.full(e.shape[0], np.nan)
    if diagonal_loading > 0.0:
        R = R + diagonal_loading * (np.trace(R).real / n) * np.eye(n)
    Rinv = np.linalg.inv(R)
    denom = np.real(np.einsum("ai,ij,aj->a", e.conj(), Rinv, e))
    # R carries power and diagonal_loading > 0, so R is positive-definite and
    # denom > 0. denom -> 0 only for a singular/rank-deficient R with no
    # loading, where 1/denom -> +inf is the honest degenerate answer (a steering
    # direction in R's null space); silence the spurious divide warning.
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / denom


def music_spectrum(R, steering, n_sources: int):
    """MUSIC pseudospectrum vs angle.

    ``P(theta) = 1 / (e^H E_n E_n^H e)`` where ``E_n`` spans the noise subspace
    (eigenvectors of ``R`` beyond the ``n_sources`` largest eigenvalues).
    ``steering`` is the ``(n_angles, n_elements)`` matrix from
    :func:`steering_vectors`.
    """
    R = np.asarray(R, dtype=complex)
    n = R.shape[0]
    if not 1 <= n_sources < n:
        raise ConfigurationError(
            f"music_spectrum: n_sources must be in [1, {n - 1}], got {n_sources}"
        )
    e = np.asarray(steering, dtype=complex)
    if _powerless_covariance(R, "music_spectrum"):
        return np.full(e.shape[0], np.nan)
    evals, evecs = np.linalg.eigh(R)
    noise = evecs[:, : n - n_sources]
    proj = e.conj() @ noise
    denom = np.sum(np.abs(proj) ** 2, axis=1)
    # denom -> 0 at a true source direction (steering orthogonal to the noise
    # subspace) is the *intended* sharp MUSIC peak -> 1/denom -> +inf; do NOT
    # clamp it, just silence the spurious divide warning.
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / denom


def shading_taper(n_elements: int, window: str = "hann"):
    """Array-shading taper (amplitude weights), normalised to unit mean.

    ``window`` is any ``scipy.signal.get_window`` name (e.g. ``'hann'``,
    ``'hamming'``, ``('chebwin', 30)``, ``('taylor', ...)``).
    """
    w = get_window(window, int(n_elements), fftbins=False)
    return w / np.mean(w)


def beamform(
    pressure: np.ndarray,
    phone_coords: np.ndarray,
    frequency: float,
    angles: Optional[np.ndarray] = None,
    SL: float = 150.0,
    NL: float = 0.0,
    c: float = DEFAULT_SOUND_SPEED
) -> BeamformResult:
    """
    Plane-wave beamformer — returns signal-to-noise ratio per look angle.

    Performs conventional plane-wave beamforming on ``pressure`` interpreted
    as a transfer function from a 0-dB source. The returned value is the
    receive level minus the ambient noise level, i.e. SNR in dB.

    Parameters
    ----------
    pressure : ndarray
        Pressure transfer-function data with shape (n_phones, n_ranges).
        Can be complex-valued.
    phone_coords : ndarray
        Hydrophone depth coordinates (m).
    frequency : float
        Frequency in Hz.
    angles : ndarray, optional
        Beam angles in degrees relative to broadside (default: -90 to 90 in
        1° steps).
    SL : float, optional
        Source level in dB re 1 µPa @ 1 m. Default 150.0.
    NL : float, optional
        **Per-element, wideband** noise level in dB at the receiver
        (i.e. dB re 1 µPa², already integrated over the signal
        bandwidth). The unit-normalised steering vector already folds
        the array gain ``10·log10(N)`` into ``|e.conj() @ pressure|``, so do
        not pre-correct ``NL`` for the number of elements. For a PSD
        in dB re 1 µPa²/Hz, multiply by the integration bandwidth in
        Hz before passing. Default 0.0.
    c : float, optional
        Reference sound speed for steering vectors in m/s.

    Returns
    -------
    snr : ndarray
        Signal-to-noise ratio in dB with shape (n_angles, n_ranges).
    angles_out : ndarray
        Angles used for beamforming (degrees from broadside).
    peak_snr : float
        Maximum value of ``snr``.

    Notes
    -----
    The beamformer computes::

        snr = 20·log10(|e.conj() @ pressure|) + SL - NL

    where ``e`` is the unit-normalised steering-vector matrix from
    :func:`steering_vectors`. Pass ``NL=0`` to recover the receive
    level alone.

    References
    ----------
    Original MATLAB code by mbp, 2 March 2001
    """
    if angles is None:
        angles = np.arange(-90, 91, 1)
    e = steering_vectors(phone_coords, angles, frequency, c)
    # Conjugate the steering vector (matched filter), consistent with
    # bartlett/mvdr/music_spectrum; ``e @ pressure`` (no conjugate) resolves
    # the mirror angle -theta for a source at +theta.
    beamformed = e.conj() @ pressure
    mag = np.abs(beamformed)
    if not np.any(mag):
        log_message(
            "beamform",
            "all beam outputs are zero (all-zero pressure?); SNR is -inf at "
            "every angle — check the input pressure.",
            level="warning",
        )
    with np.errstate(divide='ignore'):
        snr = 20 * np.log10(mag) + SL - NL
    peak_snr = np.max(snr)

    return BeamformResult(snr, angles, peak_snr)
