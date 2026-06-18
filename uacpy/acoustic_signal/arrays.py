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

from typing import Optional, Tuple

import numpy as np
from scipy.signal import get_window

from uacpy.core.constants import DEFAULT_SOUND_SPEED
from uacpy.core.exceptions import ConfigurationError


def steering_vectors(positions_m, angles_deg, freq: float,
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
    k = 2.0 * np.pi * float(freq) / float(c)
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


def mvdr_spectrum(R, steering, *, diagonal_loading: float = 1e-6):
    """MVDR / Capon power vs angle: ``1 / (e^H R^-1 e)``.

    ``steering`` is the ``(n_angles, n_elements)`` matrix from
    :func:`steering_vectors`. ``diagonal_loading`` (fraction of
    ``trace(R)/N``) stabilises the inverse for rank-deficient or
    snapshot-starved covariances.
    """
    R = np.asarray(R, dtype=complex)
    n = R.shape[0]
    if diagonal_loading > 0.0:
        R = R + diagonal_loading * (np.trace(R).real / n) * np.eye(n)
    Rinv = np.linalg.inv(R)
    e = np.asarray(steering, dtype=complex)
    denom = np.real(np.einsum("ai,ij,aj->a", e.conj(), Rinv, e))
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
    evals, evecs = np.linalg.eigh(R)
    noise = evecs[:, : n - n_sources]
    e = np.asarray(steering, dtype=complex)
    proj = e.conj() @ noise
    denom = np.sum(np.abs(proj) ** 2, axis=1)
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
    freq: float,
    angles: Optional[np.ndarray] = None,
    SL: float = 150.0,
    NL: float = 0.0,
    c: float = DEFAULT_SOUND_SPEED
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    freq : float
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
        the array gain ``10·log10(N)`` into ``|e @ pressure|``, so do
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

        snr = 20·log10(|e @ pressure|) + SL - NL

    where ``e`` is the unit-normalised steering-vector matrix from
    :func:`steering_vectors`. Pass ``NL=0`` to recover the receive
    level alone.

    References
    ----------
    Original MATLAB code by mbp, 2 March 2001
    """
    if angles is None:
        angles = np.arange(-90, 91, 1)
    e = steering_vectors(phone_coords, angles, freq, c)
    beamformed = e @ pressure
    snr = 20 * np.log10(np.abs(beamformed)) + SL - NL
    peak_snr = np.max(snr)

    return snr, angles, peak_snr


