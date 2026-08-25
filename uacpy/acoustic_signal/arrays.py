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
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core._beamforming import (
    loaded_inverse, quadratic_form, snapshot_covariance)
from uacpy._log import log_message
from uacpy.acoustic_signal._signal_validate import (
    require_finite_signal,
    require_positive_finite_scalar,
)

BeamformResult = namedtuple("BeamformResult", "snr angles peak_snr")


def steering_vectors(positions_m, angles_deg, frequency: float,
                     c: float = DEFAULT_SOUND_SPEED):
    """Unit plane-wave steering vectors for a line array.

    ``e_n(theta) = exp(-j*k*z_n*sin(theta)) / sqrt(N)`` with ``k = 2*pi*f/c``,
    ``theta`` measured from broadside and positive downward (the declination
    sign of ``Bellhop/bellhop.f90:453``). ``positions_m`` are element
    coordinates along the array axis (m).

    The ``-j`` is the conjugate of Acoustics-Toolbox ``planewave_rep.m:32``
    because every consumer here applies the vector in Hermitian form —
    ``e**H p`` for :func:`beamform`, ``e**H R e`` for the spectra — whereas
    ``Matlab/beamform.m:20`` applies its ``+j`` vector as ``e p``. Under AT's
    ``exp(+i*omega*t)`` convention (``KrakenField/EvaluateMod.f90:42``) an
    arrival at declination ``+theta`` carries depth phase
    ``exp(-i*k*z*sin(theta))``, so ``e**H p`` peaks at ``+theta`` only with
    this sign.

    Returns
    -------
    ndarray
        Shape ``(n_angles, n_elements)``, unit-norm per row.
    """
    z = np.atleast_1d(np.asarray(positions_m, dtype=float))
    if z.ndim != 1:
        # np.outer flattens, so an (N, 2) coordinate array would come back as
        # a unit-norm (n_angles, 2N) manifold — the right shape for a 2N-element
        # array that does not exist. sample_covariance checks ndim for the same
        # reason.
        raise ConfigurationError(
            f"steering_vectors: positions_m must be 1-D element coordinates "
            f"along the array axis (m); got shape {z.shape}. For a planar or "
            f"volumetric array, project the coordinates onto the array axis.")
    angles = np.atleast_1d(np.asarray(angles_deg, dtype=float))
    frequency = require_positive_finite_scalar(
        frequency, "steering_vectors", "frequency", " Hz")
    c = require_positive_finite_scalar(c, "steering_vectors", "c", " m/s")
    k = 2.0 * np.pi * frequency / c
    phase = np.outer(np.sin(np.deg2rad(angles)), z)
    e = np.exp(-1j * k * phase)
    return e / np.sqrt(z.size)


def sample_covariance(snapshots, *, diagonal_loading: float = 0.0):
    """Spatial covariance ``R = <x x^H>`` from snapshots.

    A non-finite snapshot (dead hydrophone) is refused here — one NaN
    poisons every covariance entry and the downstream Bartlett surface
    would come back all-NaN with no diagnostic.

    :func:`uacpy.sonar.csdm` is the same estimate under the matched-field
    name; both call ``core._beamforming.snapshot_covariance``, so they carry
    the same guards. This one adds ``diagonal_loading``.

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
    if not diagonal_loading >= 0.0:
        raise ConfigurationError(
            f"sample_covariance: diagonal_loading must be >= 0 (got "
            f"{diagonal_loading!r}); it scales the trace(R)/N ridge added "
            f"to the diagonal, and only a non-negative ridge regularises R.")
    # The shared covariance core (core/_beamforming): the same average and the
    # same shape/L/finiteness checks as uacpy.sonar.csdm.
    r = snapshot_covariance(snapshots, "sample_covariance")
    n = r.shape[0]
    if diagonal_loading > 0.0:
        r = r + diagonal_loading * (np.trace(r).real / n) * np.eye(n)
    return r


def bartlett_spectrum(R, steering):
    """Conventional (Bartlett) beamformer power vs angle: ``e^H R e``.

    ``steering`` is the ``(n_angles, n_elements)`` matrix from
    :func:`steering_vectors` — one steering vector per scan angle. The power
    is unnormalised, in the units of ``R``.

    :func:`uacpy.sonar.bartlett` is the same processor over a matched-field
    replica bank: column-major weights and a surface divided by ``tr K``. The
    ``uacpy.sonar.matched_field`` module docstring tabulates the four
    differences.
    """
    e = np.asarray(steering, dtype=complex)
    # The shared Bartlett/MVDR core (core/_beamforming): one einsum for all
    # three beamforming surfaces in the package.
    return quadratic_form(R, e)


def _powerless_covariance(R, caller: str) -> bool:
    """True when ``R`` carries no power, leaving a normalised spectrum undefined.

    ``diagonal_loading`` is a *fraction of* ``trace(R)/N``, so it vanishes with
    the trace: it stabilises a rank-deficient covariance that still carries
    power, but cannot rescue an all-zero one. That case is ordinary data, not a
    contrived input — ``sample_covariance`` of a silent segment (a dead
    element, a stretch of digital silence) returns exactly it. With no power
    MVDR's inverse is singular and MUSIC's noise subspace is arbitrary, so both
    decline rather than return a finite *uniform* pseudospectrum that looks
    like an answer.

    :func:`bartlett_spectrum` needs no such guard: ``e^H R e`` inverts nothing,
    so a zero covariance simply gives zero power at every angle.
    """
    trace = np.trace(R).real
    scale = trace / R.shape[0]
    if np.isfinite(scale) and scale > 0.0:
        return False
    warnings.warn(
        f"{caller}: the covariance carries no power (trace={trace:g}), so the "
        f"spectrum is undefined; returning NaN.",
        UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
    )
    return True


def mvdr_spectrum(R, steering, *, diagonal_loading: float = 1e-6):
    """MVDR / Capon power vs angle: ``1 / (e^H R^-1 e)``.

    ``steering`` is the ``(n_angles, n_elements)`` matrix from
    :func:`steering_vectors`. ``diagonal_loading`` (fraction of
    ``trace(R)/N``) stabilises the inverse for rank-deficient or
    snapshot-starved covariances — but only while ``R`` still carries power;
    see :func:`_powerless_covariance`. The output is unscaled.

    :func:`uacpy.sonar.mvdr` is the same processor over a matched-field
    replica bank, with a max-scaled surface and a 10 000x larger loading
    default (``1e-2`` against this ``1e-6``) — a deliberate per-surface
    policy, not drift. The ``uacpy.sonar.matched_field`` module docstring
    tabulates the four differences.
    """
    R = np.asarray(R, dtype=complex)
    e = np.asarray(steering, dtype=complex)
    if _powerless_covariance(R, "mvdr_spectrum"):
        return np.full(e.shape[0], np.nan)
    denom = quadratic_form(loaded_inverse(R, diagonal_loading), e)
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
    # np.linalg.eigh returns eigenvalues in ascending order, so the leading
    # n - n_sources columns are the smallest eigenvalues: the noise subspace.
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
    """Array-shading taper (amplitude weights), RMS-normalised: ``mean(w**2) = 1``.

    So ``||w|| = sqrt(n_elements)`` and a ``'boxcar'`` taper is all-ones — the
    normalisation that leaves a :func:`steering_vectors` row unit-norm after
    ``steering_vectors(...) * shading_taper(N)``, and leaves ``trace(R)``
    alone when the taper is applied to element data. Normalising to unit
    *mean* instead scaled every power the taper touched by ``mean(w**2)``,
    which is +2.04 dB for a Hann window on 16 elements — in the direction
    that makes an array look better than it is.

    ``window`` is any ``scipy.signal.get_window`` name (e.g. ``'hann'``,
    ``'hamming'``, ``('chebwin', 30)``, ``('taylor', ...)``).
    """
    w = get_window(window, int(n_elements), fftbins=False)
    return w / np.sqrt(np.mean(w ** 2))


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
    pressure = require_finite_signal(pressure, "beamform")
    frequency = require_positive_finite_scalar(
        frequency, "beamform", "frequency", " Hz")
    c = require_positive_finite_scalar(c, "beamform", "c", " m/s")
    e = steering_vectors(phone_coords, angles, frequency, c)
    # Matched filter, the same Hermitian form bartlett/mvdr/music_spectrum use.
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
