"""Many channels at once: beamforming, and the gather transforms.

One question — *what does this array of receivers see* — answered by steering
and beamforming (:func:`steering_vectors`, :func:`beamform`,
:func:`mvdr_spectrum`, …) and by the transforms that take a whole gather into
another domain (:func:`fk_transform`, :func:`taup_transform`,
:func:`radon_transform`, each with an inverse). Every one of them takes the
receiver spacing ``dx``, which is what separates them from the single-channel
estimators in :mod:`uacpy.acoustic_signal.estimate`.
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


# ──────────────────────────────────────────────────────────────────────
# Beamforming
#
# Steering vectors, the sample covariance a beamformer
# needs, and the conventional and adaptive spectra over them.
# ──────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────
# Gather transforms
#
# A whole panel into another domain and back:
# f-k, tau-p and Radon. Each takes the receiver spacing ``dx``, which is what
# separates them from the single-channel estimators.
# ──────────────────────────────────────────────────────────────────────

_RADON_KINDS = ("linear", "parabolic", "hyperbolic")

RadonResult = namedtuple("RadonResult", "moveout taus panel")
TauPResult = namedtuple("TauPResult", "slownesses taus panel")
FKResult = namedtuple("FKResult", "frequencies wavenumbers power spectrum")


def _taper(spec, n):
    """Length-``n`` taper for a ``scipy.signal.get_window`` spec.

    ``spec`` is ``None`` (rectangular ``ones``) or any get_window argument —
    a name (``'hann'``) or a ``(name, *params)`` tuple (``('kaiser', 8)``).
    Periodic (``fftbins=True``) form, the correct convention for spectra.
    """
    if spec is None:
        return np.ones(int(n))
    return get_window(spec, int(n), fftbins=True).astype(float)


def _fk_tapers(window, nt, nx):
    """Separable (time, space) tapers for the 2-D f-k window.

    ``window`` applies one spec to both axes; a 2-element ``list``
    ``[time_spec, space_spec]`` tapers the axes independently.
    """
    if isinstance(window, list):
        if len(window) != 2:
            raise ConfigurationError(
                "fk_transform: window list must be [time_window, space_window]"
                f"; got {len(window)} entries")
        t_spec, x_spec = window
    else:
        t_spec = x_spec = window
    return _taper(t_spec, nt), _taper(x_spec, nx)


def _warn_spatial_aliasing(D, freqs, dx, slownesses, caller):
    """Warn when the requested slowness range outruns the trace spacing.

    A slant stack reads the moveout only at the sensors, so between adjacent
    traces it sees the phase ``2*pi*f*p*dx`` modulo a turn. Past half a turn
    the stack can no longer tell ``p`` from ``p -/+ 1/(f*dx)``: measured on a
    900 Hz plane wave at ``p = +4e-4`` s/m with ``dx = 2`` m, the panel peaks
    at ``-1.550e-4`` against ``-1.556e-4`` predicted — the right event, the
    wrong slowness, and the wrong direction of travel.

    Nothing recovers it. The wavenumber was undersampled by the array before
    the transform ran, which is why there is no spatial zero-padding knob
    here: padding a sum over sensors with zero traces adds zero terms to that
    sum and is a bit-exact no-op (unlike :func:`fk_transform`, whose spatial
    FFT length sets the wavenumber grid and so does interpolate it). The
    remedies are all upstream of the panel — a narrower slowness range, a
    low-passed gather, or a finer ``dx``.

    The bound is taken against the frequency below which 99% of the record's
    energy lies, not the Nyquist rate, so a narrowband gather is judged on the
    band it occupies. The quantile is deliberately generous: window leakage
    already lifts it (a clean 200 Hz tone reads 218.8 Hz, 9% high), which
    makes the guard fire slightly early rather than slightly late, and at
    ``p_max``'s 1e-3 default a 200 Hz tone still passes quietly.
    """
    p_max = float(np.max(np.abs(slownesses))) if slownesses.size else 0.0
    if p_max <= 0.0 or dx <= 0.0:
        return
    power = (np.abs(D) ** 2).sum(axis=1)
    total = power.sum()
    if not np.isfinite(total) or total <= 0.0:
        return              # a silent gather aliases nothing
    f_edge = float(freqs[np.searchsorted(np.cumsum(power) / total, 0.99)])
    if f_edge <= 0.0:
        return              # all energy at DC: every slowness is unaliased
    p_alias = 1.0 / (2.0 * f_edge * dx)
    if p_max <= p_alias:
        return
    # The three remedies are quoted rounded, and a reader types the quoted
    # number back. Round each one the SAFE way — down for a ceiling on |p| and
    # on dx, down for the low-pass corner — so that following the message
    # literally clears the guard instead of tripping it again on the third
    # significant figure.

    def _floor_sig(v, n=3):
        if not np.isfinite(v) or v <= 0.0:
            return v
        scale = 10.0 ** (np.floor(np.log10(v)) - (n - 1))
        return float(np.floor(v / scale) * scale)

    p_safe = _floor_sig(p_alias)
    f_safe = _floor_sig(1.0 / (2.0 * p_max * dx))
    dx_safe = _floor_sig(1.0 / (2.0 * f_edge * p_max))
    warnings.warn(
        f"{caller}: the slowness axis reaches |p| = {p_max:.2e} s/m, but "
        f"with dx = {dx:g} m and 99% of the record's energy below "
        f"{f_edge:.0f} Hz the stack aliases beyond |p| = {p_safe:.3g} s/m — "
        f"an event steeper than that is indistinguishable from p -/+ "
        f"1/(f*dx) and can surface at the wrong slowness, or the wrong sign. "
        f"Cap the slowness range at {p_safe:.3g} s/m, low-pass the gather "
        f"below {f_safe:.3g} Hz, or sample the array at "
        f"dx = {dx_safe:.3g} m or finer. Zero-padding cannot help: the "
        f"wavenumber was undersampled before the transform ran.",
        UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
    )


def _fk_nfft(nfft, nt, nx):
    """Resolve the zero-padded f-k transform shape ``(NT, NX) >= (nt, nx)``."""
    if nfft is None:
        return nt, nx
    if np.isscalar(nfft):
        NT = NX = int(nfft)
    else:
        nfft = tuple(nfft)
        if len(nfft) != 2:
            raise ConfigurationError(
                "fk_transform: nfft must be an int or (n_time, n_space)"
                f"; got {len(nfft)} entries")
        NT, NX = int(nfft[0]), int(nfft[1])
    if NT < nt or NX < nx:
        raise ConfigurationError(
            f"fk_transform: nfft {(NT, NX)} must be >= data shape {(nt, nx)} "
            "(zero-pad only, no truncation)")
    return NT, NX


def _require_scalar_geometry(caller, sample_rate, dx, signature):
    """Reject an array where a scalar ``sample_rate``/``dx`` belongs.

    ``inverse_taup`` and ``inverse_radon`` both take a panel, a parameter axis
    and the scalar geometry, but in different positions, so passing one's
    argument order to the other lands an array on a scalar slot. Catching it
    here names the expected signature instead of failing later inside numpy.
    """
    for name, value in (("sample_rate", sample_rate), ("dx", dx)):
        if np.ndim(value) != 0:
            raise ConfigurationError(
                f"{caller}: {name} must be a scalar, got an array of shape "
                f"{np.shape(value)} — check the argument order, which is "
                f"{signature}.")


def _flip_wavenumber_axis(F):
    """Negate the spatial-frequency convention of an unshifted 2-D DFT.

    numpy's ``fft2`` carries ``exp(-i2π(ft + νx))``, which puts a wave
    travelling towards +x on ``ω = -c·k``. Reversing the (unshifted) spatial
    axis re-indexes column ``ν`` to hold ``-ν``, so the wave lands on
    ``ω = +c·k`` — the package-wide ``k = ω/c`` convention that
    :func:`taup_transform` and :func:`radon_transform` already use. The map is
    its own inverse, so :func:`inverse_fk` applies the same reversal.
    """
    return np.roll(F[:, ::-1], 1, axis=1)


def _moveout_times(kind, taus, x, m):
    """Moveout time ``t(tau, x; m)`` for the requested Radon kind."""
    if kind == "linear":
        return taus + m * x
    if kind == "parabolic":
        return taus + m * x ** 2
    if kind == "hyperbolic":
        return np.sqrt(taus ** 2 + (x / m) ** 2)
    raise ConfigurationError(
        f"radon: kind must be one of {_RADON_KINDS}, got {kind!r}"
    )


def radon_transform(data, sample_rate, dx, moveout, kind="linear", x0=0.0):
    """Forward Radon transform (slant stack) of a ``(nt, nx)`` gather.

    Sums the data along moveout curves ``t = t(tau, x)``:

    * ``linear``      ``t = tau + p*x``        (``moveout`` = slowness p, s/m)
    * ``parabolic``   ``t = tau + q*x**2``     (``moveout`` = curvature q, s/m^2)
    * ``hyperbolic``  ``t = sqrt(tau^2+(x/v)^2)`` (``moveout`` = velocity v, m/s)

    Parameters
    ----------
    data : ndarray
        Gather, shape ``(nt, nx)`` (time down columns, offset across rows).
    sample_rate : float
        Temporal sample rate (Hz).
    dx : float
        Sensor spacing (m).
    moveout : array
        Moveout parameters to scan (units per ``kind`` above).
    kind : {'linear', 'parabolic', 'hyperbolic'}
        Moveout family. ``'linear'`` is the tau-p slant stack.
    x0 : float
        Reference offset (m) subtracted from sensor positions.

    Returns
    -------
    RadonResult
        Namedtuple ``(moveout, taus, panel)``: the scanned moveout axis, the
        intercept-time axis (s), and the Radon panel ``(len(moveout), nt)``.
    """
    d = np.asarray(data, dtype=float)
    if d.ndim != 2:
        raise ConfigurationError(
            "radon_transform: data must be 2-D (nt, nx)"
            f"; got shape {d.shape}")
    nt, nx = d.shape
    sample_rate = require_positive_finite_scalar(
        sample_rate, "radon_transform", "sample_rate", " Hz")
    dx = require_positive_finite_scalar(dx, "radon_transform", "dx", " m")
    x = np.arange(nx) * dx - float(x0)
    moveout = np.atleast_1d(np.asarray(moveout, dtype=float))
    if kind == "hyperbolic" and np.any(moveout <= 0):
        raise ConfigurationError(
            "radon_transform: hyperbolic moveout is a velocity (m/s) and the "
            "moveout curve sqrt(tau^2 + (x/v)^2) divides by it, so every "
            f"value must be > 0; got min {moveout.min()}.")
    taus = np.arange(nt) / float(sample_rate)
    R = np.zeros((moveout.size, nt))
    for i, m in enumerate(moveout):
        for ix in range(nx):
            tt = _moveout_times(kind, taus, x[ix], m)
            R[i] += np.interp(tt, taus, d[:, ix], left=0.0, right=0.0)
    return RadonResult(moveout, taus, R)


def inverse_radon(R, sample_rate, dx, moveout, nx, kind="linear", x0=0.0):
    """Adjoint (back-projection) Radon transform: ``(n_moveout, nt) -> (nt, nx)``.

    Spreads each Radon sample back along its moveout curve. This is the matched
    adjoint, not a least-squares inverse, so a forward-then-adjoint round trip is
    band-limited, not exact. It is the exact transpose of
    :func:`radon_transform` for every ``kind`` — ``<L x, y> == <x, A y>`` to
    machine precision — which is what an iterative least-squares (sparse Radon)
    solver needs.

    ``sample_rate``/``dx`` are the geometry and ``moveout`` the scanned
    parameter axis; note the order differs from :func:`inverse_taup`, whose
    slowness axis comes second.
    """
    R = np.asarray(R, dtype=float)
    if R.ndim != 2:
        raise ConfigurationError(
            "inverse_radon: R must be 2-D (n_moveout, nt)"
            f"; got shape {R.shape}")
    nm, nt = R.shape
    _require_scalar_geometry(
        "inverse_radon", sample_rate, dx,
        "inverse_radon(R, sample_rate, dx, moveout, nx)")
    moveout = np.atleast_1d(np.asarray(moveout, dtype=float))
    if moveout.size != nm:
        raise ConfigurationError(
            f"inverse_radon: moveout length ({moveout.size}) must match R rows "
            f"({nm}); the signature is inverse_radon(R, sample_rate, dx, "
            "moveout, nx) — the moveout axis comes fourth, unlike "
            "inverse_taup where the slowness axis comes second.")
    if kind == "hyperbolic" and np.any(moveout <= 0):
        raise ConfigurationError(
            "inverse_radon: hyperbolic moveout is a velocity (m/s) and the "
            "moveout curve sqrt(tau^2 + (x/v)^2) divides by it, so every "
            f"value must be > 0; got min {moveout.min()}.")
    fs = float(sample_rate)
    taus = np.arange(nt) / fs
    x = np.arange(int(nx)) * float(dx) - float(x0)
    out = np.zeros((nt, int(nx)))
    for i, m in enumerate(moveout):
        for ix in range(int(nx)):
            # Scatter: each Radon sample is split between the two grid samples
            # straddling t(tau, x) with the same weights the forward
            # `np.interp` gives them, which is the transpose of that
            # interpolation. Gathering instead — reading the curve back with
            # `np.interp(taus, tt, ...)` — only coincides with the transpose
            # when the moveout is a pure time shift (linear, parabolic); a
            # hyperbolic curve compresses near tau=0 and the gather returns
            # early samples at a fraction of their forward weight.
            idx = _moveout_times(kind, taus, x[ix], m) * fs
            inside = (idx >= 0.0) & (idx <= nt - 1)
            j = np.floor(idx[inside]).astype(int)
            w = idx[inside] - j
            r = R[i][inside]
            out[:, ix] += np.bincount(j, weights=(1.0 - w) * r,
                                      minlength=nt)[:nt]
            out[:, ix] += np.bincount(j + 1, weights=w * r,
                                      minlength=nt + 1)[:nt]
    return out


def taup_transform(data, sample_rate, dx, slownesses=None, n_slowness=201,
                   p_max=None, *, x0=0.0, window=None, nfft=None):
    """Forward linear tau-p (slant stack), frequency-domain.

    Returns a :class:`TauPResult` namedtuple ``(slownesses, taus, panel)``:
    slowness axis (s/m), intercept-time axis (s), and the panel ``(n_slowness,
    NT)``.

    ``x0`` is the reference offset (m) subtracted from the sensor positions, as
    in :func:`radon_transform` — it walks the same moveout curve
    ``t = tau + p*(x - x0)``.

    Built from FFT products, the transform is **circular in tau**: each trace
    is read at ``(tau + p*(x - x0)) mod (NT/fs)``, ``NT`` being the time-FFT
    length. An intercept ``t0 - p*(x - x0)`` outside the ``[0, NT/fs)`` window
    (either sign) therefore stacks at full amplitude at that intercept
    **modulo** ``NT/fs`` — a tau where :func:`radon_transform`
    (``kind='linear'``, same ``x0``), which drops out-of-window samples, is
    exactly zero.

    An in-window intercept is necessary for the two to agree but not
    sufficient: the shift here is a spectral phase ramp (band-limited) while
    the Radon panel interpolates each trace linearly, so a moveout of a
    non-integer number of samples per trace separates them even in-window —
    measured 9.1% of the peak at ``p*dx*fs = 2.5``. Expect exact agreement
    only for a whole-sample moveout.

    ``window`` is a temporal :func:`scipy.signal.get_window` spec (name or
    ``(name, *params)`` tuple) applied down each trace before the time FFT to
    curb leakage; ``None`` is rectangular. ``nfft`` zero-pads the time FFT to
    ``NT >= nt`` samples; ``None`` keeps ``nt``. The ``tau`` spacing is
    ``1/fs`` either way — zero-padding does not refine it, it **lengthens**
    the tau axis into a guard band: an intercept up to ``(NT - nt)/fs`` s
    outside the record lands in the padded ``[nt/fs, NT/fs)`` rows instead of
    aliasing in among the physical taus.

    The spatial axis has no such knob, and deliberately so
    -----------------------------------------------------
    The slowness axis is yours to set outright — ``slownesses`` takes any
    array, uniform or not, and ``n_slowness``/``p_max`` build one for you — so
    resolution in ``p`` is never limited by a transform length the way
    :func:`fk_transform`'s wavenumber axis is. There is correspondingly no
    spatial ``nfft``: this transform SUMS over sensors rather than
    transforming across them, so appending zero traces adds zero terms to that
    sum and changes nothing, bit for bit. (In ``fk_transform`` the spatial FFT
    length sets the wavenumber grid ``dk = 2*pi/(NX*dx)``, which is why the
    knob is real there and meaningless here.)

    What the array spacing DOES limit is aliasing. Between adjacent traces the
    stack sees ``2*pi*f*p*dx`` modulo a turn, so past ``|p| = 1/(2*f*dx)`` it
    cannot separate ``p`` from ``p -/+ 1/(f*dx)``: a 900 Hz plane wave at
    ``p = +4e-4`` s/m on a ``dx = 2`` m array surfaces at ``-1.55e-4`` — the
    wrong slowness AND the wrong direction of travel. A warning fires when the
    requested range crosses that bound, judged against the frequency holding
    99% of the record's energy rather than the Nyquist rate so a narrowband
    gather is measured on the band it actually occupies. Heed it upstream: no
    padding recovers a wavenumber the array never sampled.
    """
    d = np.asarray(data, dtype=float)
    if d.ndim != 2:
        raise ConfigurationError(
            "taup_transform: data must be 2-D (nt, nx)"
            f"; got shape {d.shape}")
    nt, nx = d.shape
    fs = require_positive_finite_scalar(sample_rate, "taup_transform",
                                        "sample_rate", " Hz")
    dx = require_positive_finite_scalar(dx, "taup_transform", "dx", " m")
    NT = nt if nfft is None else int(nfft)
    if NT < nt:
        raise ConfigurationError(
            f"taup_transform: nfft ({NT}) must be >= nt ({nt}) (zero-pad only)")
    d = d * _taper(window, nt)[:, None]
    x = np.arange(nx) * dx - float(x0)
    if slownesses is None:
        if p_max is None:
            # +/- 1e-3 s/m: everything with an apparent velocity above
            # 1000 m/s, which spans the water column and most sediments.
            p_max = 1.0 / 1000.0
        slownesses = np.linspace(-p_max, p_max, int(n_slowness))
    slownesses = np.atleast_1d(np.asarray(slownesses, dtype=float))
    D = np.fft.rfft(d, n=NT, axis=0)
    freqs = np.fft.rfftfreq(NT, 1.0 / fs)                # Hz
    omega = 2.0 * np.pi * freqs                          # rad/s
    _warn_spatial_aliasing(D, freqs, dx, slownesses, "taup_transform")
    taup = np.empty((slownesses.size, NT))
    for i, p in enumerate(slownesses):
        # Sign: numpy's forward transform carries exp(-j*omega*t), so a
        # +exp(j*omega*p*x) factor advances trace x by p*x. Summing over x is
        # then u(tau, p) = sum_x d(tau + p*x, x) — the same moveout curve
        # `radon_transform(kind='linear')` interpolates in the time domain.
        phase = np.exp(1j * omega[:, None] * (p * x)[None, :])
        taup[i] = np.fft.irfft(np.sum(D * phase, axis=1), n=NT)
    return TauPResult(slownesses, np.arange(NT) / fs, taup)


def inverse_taup(taup, slownesses, sample_rate, dx, nx, *, x0=0.0):
    """Adjoint slant stack ``(n_slowness, nt) -> (nt, nx)``.

    Standalone inverse — pass a tau-p panel you already have (e.g. a filtered
    one) plus its slowness axis and geometry; no prior :func:`taup_transform`
    call needed. ``x0`` is the reference offset (m) the forward transform used;
    pass the same value back.

    The slowness axis comes **second** here, while :func:`inverse_radon` takes
    its moveout axis **fourth** (after ``sample_rate`` and ``dx``). Both are
    positional arrays, so the two orders are not interchangeable; the scalar
    geometry arguments are type-checked below to catch the swap.
    """
    u = np.asarray(taup, dtype=float)
    if u.ndim != 2:
        raise ConfigurationError(
            "inverse_taup: taup must be 2-D (n_slowness, nt)"
            f"; got shape {u.shape}")
    n_p, nt = u.shape
    _require_scalar_geometry("inverse_taup", sample_rate, dx,
                             "inverse_taup(taup, slownesses, sample_rate, dx, nx)")
    slownesses = np.atleast_1d(np.asarray(slownesses, dtype=float))
    if slownesses.size != n_p:
        raise ConfigurationError(
            f"inverse_taup: slownesses length ({slownesses.size}) must match "
            f"taup rows ({n_p}); the signature is inverse_taup(taup, "
            "slownesses, sample_rate, dx, nx) — the slowness axis comes "
            "second, unlike inverse_radon where it comes fourth.")
    x = np.arange(int(nx)) * float(dx) - float(x0)
    U = np.fft.rfft(u, axis=1)
    omega = 2.0 * np.pi * np.fft.rfftfreq(nt, 1.0 / float(sample_rate))
    D = np.zeros((omega.size, int(nx)), dtype=complex)
    for i, p in enumerate(slownesses):
        # Conjugate phase of the forward transform (the adjoint): each slowness
        # is spread back along its own moveout, delayed by p*x.
        D += U[i][:, None] * np.exp(-1j * omega[:, None] * (p * x)[None, :])
    return np.fft.irfft(D, n=nt, axis=0)


def inverse_fk(FK):
    """Inverse f-k transform: complex (fftshifted) spectrum -> real gather.

    Pass the (possibly filtered/muted) complex spectrum — i.e. the ``spectrum``
    returned by :func:`fk_transform` (single-segment), after any f-k mask. It
    must be in the ``fftshift``ed layout that :func:`fk_transform` produces.

    The output has the **spectrum's** shape ``(NT, NX)`` — the zero-padded
    ``nfft`` shape when the forward transform was padded, with the original
    ``(nt, nx)`` gather in its top-left corner followed by the padding.
    The forward ``window`` taper is **not** undone: a windowed forward
    transform inverts to the *tapered* gather, and recovering the original
    data requires dividing the tapers back out (undefined where they are
    zero). For an exact round trip run ``fk_transform`` with ``window=None``.
    """
    if FK is None:
        raise ConfigurationError(
            "inverse_fk: spectrum is None — an f-k panel averaged over more "
            "than one segment has no phase and cannot be inverted. Re-run "
            "fk_transform with nperseg=None for an invertible spectrum.")
    if isinstance(FK, tuple):
        raise ConfigurationError(
            "inverse_fk: pass the complex spectrum (the .spectrum field / 4th "
            "element of fk_transform's result), not the whole FKResult tuple.")
    fk = np.asarray(FK)
    if fk.ndim != 2:
        raise ConfigurationError(
            "inverse_fk: FK must be 2-D (nt, nx)"
            f"; got shape {fk.shape}")
    return np.real(np.fft.ifft2(
        _flip_wavenumber_axis(np.fft.ifftshift(fk, axes=(0, 1)))))


def fk_transform(data, sample_rate, dx, *, nperseg=None, noverlap=None,
                 window=None, nfft=None, normalize=False):
    """Frequency-wavenumber transform with optional Welch time-averaging.

    Returns an :class:`FKResult` namedtuple ``(frequencies, wavenumbers, power,
    spectrum)``. ``frequencies`` are in Hz; ``wavenumbers`` is the **angular**
    wavenumber ``k = 2π·ν`` in **rad/m** (the package-wide ``k = ω/c``
    convention), so a wave travelling towards +x at speed ``c`` sits on the
    line ``ω = +c·k`` — the same sign as the apparent slowness ``p = +1/c``
    that :func:`taup_transform` and :func:`radon_transform` report for it.
    The spatial axis is therefore the negative of raw ``np.fft.fft2``
    indexing, whose ``exp(-i2πνx)`` kernel would place that wave on
    ``ω = -c·k``; directional f-k muting must use this sign.
    ``power`` is the real ``|FK|^2`` panel (fftshifted); when ``normalize=True``
    it is a PSD density per ``Hz·rad/m`` with ``ΣP·Δf·Δk = ⟨x²⟩``. Whenever the
    settings yield a single segment (``nperseg=None``, i.e. the whole record, or
    an ``nperseg``/``noverlap`` pair that fits only one block) ``spectrum`` is
    that segment's complex (fftshifted) panel for :func:`inverse_fk`. With
    several segments the time axis is split into overlapping blocks, ``|FK|^2``
    is averaged across them (variance ~1/sqrt(N); a single-snapshot f-k panel is
    an inconsistent estimator), and ``spectrum`` is ``None`` — an averaged power
    panel has no single phase and is not invertible.

    Parameters
    ----------
    data : ndarray
        Gather ``(nt, nx)``.
    sample_rate : float
        Temporal sample rate (Hz).
    dx : float
        Sensor spacing (m).
    nperseg : int, optional
        Time-segment length for Welch averaging. ``None`` (default) uses the
        whole record (one segment, invertible).
    noverlap : int, optional
        Overlap between segments (samples). Defaults to ``nperseg // 2`` when
        ``nperseg`` is set; ``0`` otherwise. Must satisfy ``0 <= noverlap < nperseg``.
    window, nfft, normalize
        As in the single-segment transform, applied per segment.
    """
    d = np.asarray(data)
    if d.ndim != 2:
        raise ConfigurationError(
            "fk_transform: data must be 2-D (nt, nx)"
            f"; got shape {d.shape}")
    nt, nx = d.shape
    fs = require_positive_finite_scalar(
        sample_rate, "fk_transform", "sample_rate", " Hz")
    dx = require_positive_finite_scalar(dx, "fk_transform", "dx", " m")
    seg = nt if nperseg is None else int(nperseg)
    if seg > nt or seg < 1:
        raise ConfigurationError(
            f"fk_transform: nperseg ({seg}) must be in [1, nt={nt}]")
    ov = (seg // 2) if (noverlap is None and nperseg is not None) else int(noverlap or 0)
    if not (0 <= ov < seg):
        raise ConfigurationError(
            f"fk_transform: noverlap ({ov}) must be in [0, nperseg={seg})")
    wt, wx = _fk_tapers(window, seg, nx)
    NF, NX = _fk_nfft(nfft, seg, nx)

    # Whole-segment starts only (trailing samples that don't fill a segment are
    # dropped, as in scipy's Welch); each block is therefore exactly `seg` long.
    starts = range(0, nt - seg + 1, seg - ov)
    power = np.zeros((NF, NX))
    last_spectrum = None
    n_seg = 0
    for s0 in starts:
        block = d[s0:s0 + seg]
        bw = block * wt[:, None] * wx[None, :]
        FKc = np.fft.fftshift(
            _flip_wavenumber_axis(np.fft.fft2(bw, s=(NF, NX))), axes=(0, 1))
        last_spectrum = FKc
        FKp = np.abs(FKc) ** 2
        if normalize:
            s2 = float(np.sum(wt ** 2) * np.sum(wx ** 2))
            # Density per (Hz · rad/m): the extra 2π converts the per-bin spatial
            # width to rad/m so that ΣP·Δf·Δk = ⟨x²⟩ still holds with k in rad/m.
            FKp = FKp * (float(dx) / (fs * s2 * 2.0 * np.pi))
        power += FKp
        n_seg += 1
    power /= n_seg

    freqs = np.fft.fftshift(np.fft.fftfreq(NF, d=1.0 / fs))
    # Angular wavenumber k = 2π·ν in rad/m (ν = fftfreq is cycles/m), matching
    # the package-wide convention k = ω/c used by the models: a wave of speed c
    # lies on the line ω = c·k (i.e. f = c·k/2π — the acoustic "sound cone").
    # The axis needs no negation here because `_flip_wavenumber_axis` already
    # re-indexed the panel columns onto it.
    wavenumbers = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(NX, d=dx))
    spectrum = last_spectrum if n_seg == 1 else None
    return FKResult(freqs, wavenumbers, power, spectrum)
