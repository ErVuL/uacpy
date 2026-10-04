"""What an array of receivers sees: steering, covariance and
beamforming.

:func:`steering_vectors`, :func:`snapshots` / :func:`sample_covariance`, the
Bartlett and MVDR processors (:func:`bartlett`, :func:`mvdr`, over any bank
of weight rows) and the MUSIC spectrum, :func:`beamform` along one range
line and :func:`beamform_field` over a whole modelled field, with the
array-gain references (:func:`plane_wave_array_gain`,
:func:`matched_replica_gain`).
"""

from __future__ import annotations

import warnings
from collections import namedtuple
from typing import Optional
import numpy as np
from scipy.signal import get_window
from uacpy.core.constants import DEFAULT_SOUND_SPEED
from uacpy.core.exceptions import ConfigurationError, ValidityWarning
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core._beamforming import (
    loaded_inverse, quadratic_form, snapshot_covariance,
)
from uacpy._log import log_message
from uacpy.acoustic_signal._results import PlottedResult, ResultTuple
from uacpy.core.acoustics.arrays import steering_vectors
from uacpy.core._validate import (
    require_finite_signal, require_positive_finite_scalar,
)
from uacpy.acoustic_signal.windows import _default_noverlap, _fk_tapers


# ──────────────────────────────────────────────────────────────────────
# Beamforming
#
# Steering vectors, the sample covariance a beamformer
# needs, and the conventional and adaptive spectra over them.
# ──────────────────────────────────────────────────────────────────────

class BeamformResult(ResultTuple,
                     namedtuple("BeamformResult", "snr angles peak_snr")):
    """Conventional beamformer scan along one range line: ``snr`` (dB) per
    look angle and range, the look ``angles`` (deg) and the ``peak_snr``
    (dB).

    The tuple is the measurement, so ``snr, angles, peak_snr = ...``
    keeps working. ``ranges`` (m), the axis of ``snr`` after the angle one,
    rides as an attribute when :func:`beamform` was given it, ``None``
    otherwise; it survives pickling and copying and takes no part in
    equality.
    """

    _attrs = ("ranges",)

    def __new__(cls, snr, angles, peak_snr, *, ranges=None):
        self = super().__new__(cls, snr, angles, peak_snr)
        self.ranges = ranges
        return self

    def _field_units(self):
        return {"snr": "dB", "angles": "deg", "peak_snr": "dB"}


class Snapshots(ResultTuple, namedtuple("Snapshots", "frequency data")):
    """Complex single-frequency snapshots, and the bin they were read at.

    The tuple is the measurement, so ``frequency, data = snapshots(...)``
    keeps working. ``data`` is ``(n_elements, n_snapshots)``, the shape
    :func:`sample_covariance` and :func:`uacpy.sonar.csdm` take.

    ``frequency`` is the DFT bin actually used, which is not the frequency
    asked for: a 1024-point segment at 2 kHz resolves 1.95 Hz, so a request
    for 200 Hz is answered at 199.22 Hz. It comes back first because the
    replicas must be built at the same frequency as the data — steering at
    the requested value instead of the bin mismatches the replica against
    the snapshots, and nothing downstream can detect that.
    """

    __slots__ = ()

    def covariance(self, *, diagonal_loading: float = 0.0):
        """``sample_covariance`` of these snapshots, for convenience."""
        return sample_covariance(self.data,
                                 diagonal_loading=diagonal_loading)

    def _field_units(self):
        # The snapshots are unscaled DFT bins of the segments.
        return {"frequency": "Hz", "data": None}


def snapshots(data, sample_rate, frequency, *, nperseg, noverlap=None,
              window=None):
    """Single-frequency snapshots from an array time record.

    The bridge between what an array records and what the covariance
    estimators take. :func:`sample_covariance` averages over snapshots, but
    a hydrophone array delivers a real time series per element; getting from
    one to the other means segmenting the record, windowing, transforming
    each segment and reading one bin — with a transpose at the end, because
    the record is ``(n_samples, n_elements)`` (what :func:`fk_transform`
    takes) and the covariance wants ``(n_elements, n_snapshots)``.

    Parameters
    ----------
    data : ndarray, shape ``(n_samples, n_elements)``
        The array record, real or complex.
    sample_rate : float
        Sampling rate (Hz).
    frequency : float
        Frequency of interest (Hz). Answered at the nearest DFT bin, whose
        value is returned — see :class:`Snapshots`.
    nperseg : int
        Samples per segment. Required, and the only parameter that matters
        twice: it sets the frequency resolution (hence which bin answers
        ``frequency``) and the snapshot count (hence whether an adaptive
        estimator has enough). There is no defensible default for both.
    noverlap : int, optional
        Overlap between segments. Defaults to ``nperseg // 2``, matching
        :func:`fk_transform`. Overlapped segments are correlated, so they
        buy fewer independent snapshots than their count suggests.
    window : array_like or str, optional
        Time taper, as :func:`fk_transform` takes it. Default ``None``, a
        rectangular window (as on :func:`fk_transform`): an on-bin tone keeps
        its full amplitude. Pass ``'hann'`` to trade that (a Hann halves a
        snapshot's amplitude) for lower leakage from strong neighbouring
        bins.

    Returns
    -------
    Snapshots
        ``(frequency, data)`` — the bin frequency and the
        ``(n_elements, n_snapshots)`` complex array.

    Notes
    -----
    One bin, deliberately. Pooling a band of bins into one covariance and
    beamforming it with a single replica steers the off-centre bins at the
    wrong frequency: measured over 170-229 Hz on a 12-element array, that
    costs 2.4 dB of MVDR contrast against steering each bin at its own
    frequency. Broadband processing is a sum of per-bin surfaces, each with
    its own replica — a different operation, not a wider window here.
    """
    d = np.asarray(data)
    if d.ndim != 2:
        raise ConfigurationError(
            f"snapshots: data must be 2-D (n_samples, n_elements); got "
            f"shape {d.shape}.")
    nt, nx = d.shape
    fs = float(sample_rate)
    if not fs > 0 or not np.isfinite(fs):
        raise ConfigurationError(
            f"snapshots: sample_rate must be a positive finite number of Hz; "
            f"got {sample_rate!r}.")
    seg = int(nperseg)
    if seg < 1 or seg > nt:
        raise ConfigurationError(
            f"snapshots: nperseg ({seg}) must be in [1, n_samples={nt}].")
    ov = _default_noverlap(window, seg) if noverlap is None else int(noverlap)
    if not 0 <= ov < seg:
        raise ConfigurationError(
            f"snapshots: noverlap ({ov}) must be in [0, nperseg={seg}).")
    f0 = float(frequency)
    nyquist = fs / 2.0
    if not 0.0 <= f0 <= nyquist:
        raise ConfigurationError(
            f"snapshots: frequency ({f0:g} Hz) must lie in [0, Nyquist="
            f"{nyquist:g} Hz] for a {fs:g} Hz record.")
    wt, _ = _fk_tapers(window, seg, nx)

    # Clamped to the last non-negative bin. ``round`` is half-to-even, so a
    # request at exactly Nyquist with an odd ``nperseg`` of the form 4m+3
    # rounds *up* past seg//2 and lands on the first NEGATIVE-frequency bin —
    # whose frequency is not ``bin_index * fs / seg`` at all. Unclamped,
    # nperseg=255 at fs=2000 reported 1003.92 Hz for a 1000 Hz Nyquist,
    # reading the -996.08 Hz bin: neither the frequency asked for nor the one
    # supplied, which is the single thing this return value exists to be.
    bin_index = min(int(round(f0 * seg / fs)), seg // 2)
    bin_frequency = bin_index * fs / seg
    # Whole segments only, as fk_transform does: a trailing partial block
    # would be transformed at a different resolution and land in a different
    # bin, which is not the same measurement.
    starts = range(0, nt - seg + 1, seg - ov)
    columns = [np.fft.fft(d[s0:s0 + seg] * wt[:, None], axis=0)[bin_index]
               for s0 in starts]
    if not columns:
        raise ConfigurationError(
            f"snapshots: no whole segment fits — nperseg={seg} with "
            f"n_samples={nt}.")
    return Snapshots(bin_frequency, np.asarray(columns).T)


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


def _unit_rows(weights):
    """``weights`` with every row (last axis) scaled to unit norm. A row of
    zeros stays zero: it is a candidate a forward model put no energy at (a
    shadow-zone cell of a ray or PE replica bank, a source depth on a
    pressure-release boundary), and dividing it by 1 lets :func:`bartlett`
    score it as a genuine zero and :func:`mvdr` as undefined, instead of the
    0/0 that would put a NaN and a RuntimeWarning in either surface."""
    norm = np.linalg.norm(weights, axis=-1, keepdims=True)
    return weights / np.where(norm > 0, norm, 1.0)


def _weight_slices(covariance, weights, who):
    """``(R, W, stacked)``: the covariance as a complex ``(F, N, N)`` stack and
    the weights as a complex ``(F, P, N)`` stack of unit rows, with
    ``stacked`` saying whether the caller passed a stack. Refuses shapes that
    do not pair: a covariance that is not square, weights whose last axis is
    not the element count, a stack whose leading axis differs from the
    weights'."""
    R = np.asarray(covariance, dtype=complex)
    W = np.asarray(weights, dtype=complex)
    if R.ndim not in (2, 3) or R.shape[-1] != R.shape[-2]:
        raise ConfigurationError(
            f"{who}: covariance must be one (N, N) matrix or an (F, N, N) "
            f"stack of them; got shape {R.shape}.")
    n = R.shape[-1]
    if W.ndim < 1 or W.shape[-1] != n:
        raise ConfigurationError(
            f"{who}: weights must carry one row of {n} element weights per "
            f"candidate, shape (..., {n}), row-major; got shape {W.shape}. A "
            f"column-major bank (N, *grid) is np.moveaxis(bank, 0, -1).")
    stacked = R.ndim == 3
    if stacked:
        if W.ndim < 2 or W.shape[0] != R.shape[0]:
            raise ConfigurationError(
                f"{who}: a covariance stack of {R.shape[0]} matrices needs "
                f"weights of shape ({R.shape[0]}, ..., {n}), one bank per "
                f"matrix; got shape {W.shape}.")
        return R, _unit_rows(W.reshape(W.shape[0], -1, n)), True
    return R[None], _unit_rows(W.reshape(1, -1, n)), False


def bartlett(covariance, weights, *, normalize: Optional[str] = None):
    """Conventional (Bartlett) beamformer power: ``w^H R w`` for every weight
    row ``w``, each scaled to unit norm, with ``R`` the element
    ``covariance``.

    One processor for every weight bank in the package: plane-wave
    :func:`steering_vectors` (a power against look angle), a matched-field
    replica bank (a power against candidate source position), or any row of
    element weights. ``weights`` is row-major, ``(..., N)``, one row per
    candidate; the output has shape ``weights.shape[:-1]``. A row of zeros
    scores zero.

    ``covariance`` is one ``(N, N)`` matrix, or an ``(F, N, N)`` stack (one
    per frequency) with ``weights`` of shape ``(F, ..., N)``, one bank per
    matrix.

    ``normalize``:

    * ``None`` — the power in the units of ``covariance``;
    * ``'trace'`` — divided by ``tr R`` (each matrix's own), so the output is
      ``1`` where a weight row matches a rank-one covariance exactly and in
      ``[0, 1]`` everywhere. A covariance with zero trace is a matrix of
      zeros and so is the power: it is returned as it is, zero.

    :func:`uacpy.sonar.bartlett` and
    :meth:`uacpy.core.results.Covariance.bartlett` are this function over a
    :class:`~uacpy.core.results.Replicas` set, returning the surface as an
    ambiguity Field.

    Parameters
    ----------
    covariance : ndarray
        ``(N, N)`` element covariance, or an ``(F, N, N)`` stack.
    weights : ndarray
        Weight rows ``(..., N)``, or ``(F, ..., N)`` against a stack.
    normalize : {None, 'trace'}, optional
        Scale of the output (see above).
    """
    if normalize not in (None, 'trace'):
        raise ConfigurationError(
            f"bartlett: normalize must be None or 'trace'; got "
            f"{normalize!r}.")
    R, W, stacked = _weight_slices(covariance, weights, 'bartlett')
    out = np.empty(W.shape[:2], dtype=float)
    for f in range(R.shape[0]):
        # The shared Bartlett/MVDR core (core/_beamforming).
        power = quadratic_form(R[f], W[f])
        if normalize == 'trace':
            trace = np.real(np.trace(R[f]))
            # A zero-trace covariance is a positive-semidefinite matrix of
            # zeros, so the numerator is zero too: divide by 1 rather than
            # 0/0.
            power = power / (trace if trace != 0 else 1.0)
        out[f] = power
    shape = np.shape(weights)[:-1]
    return out.reshape(shape) if stacked else out[0].reshape(shape)


def _powerless_covariance(R, who: str) -> bool:
    """True when ``R`` carries no power, leaving a normalised spectrum undefined.

    ``diagonal_loading`` is a *fraction of* ``trace(R)/N``, so it vanishes with
    the trace: it stabilises a rank-deficient covariance that still carries
    power, but cannot rescue an all-zero one. That case is ordinary data, not a
    contrived input — ``sample_covariance`` of a silent segment (a dead
    element, a stretch of digital silence) returns exactly it. With no power
    MVDR's inverse is singular and MUSIC's noise subspace is arbitrary, so both
    decline rather than return a finite *uniform* pseudospectrum that looks
    like an answer.

    :func:`bartlett` needs no such guard: ``w^H R w`` inverts nothing, so a
    zero covariance simply gives zero power at every candidate.
    """
    trace = np.trace(R).real
    scale = trace / R.shape[0]
    if np.isfinite(scale) and scale > 0.0:
        return False
    warnings.warn(
        f"{who}: the covariance carries no power (trace={trace:g}), so the "
        f"output is undefined; returning NaN.",
        ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP,
    )
    return True


def mvdr(covariance, weights, *, diagonal_loading: float = 1e-6,
         normalize: Optional[str] = None):
    """MVDR / Capon power: ``1 / (w^H (R + δ I)^-1 w)`` for every weight row
    ``w``, each scaled to unit norm, with ``R`` the element ``covariance``
    and ``δ = diagonal_loading · tr(R)/N``.

    Laid out as :func:`bartlett`: ``weights`` row-major ``(..., N)`` (or
    ``(F, ..., N)`` against an ``(F, N, N)`` covariance stack), output of
    shape ``weights.shape[:-1]``.

    ``diagonal_loading`` (a fraction of ``tr(R)/N``) stabilises the inverse
    of a rank-deficient or snapshot-starved covariance: small values give
    sharp Capon peaks but are sensitive to mismatch, larger ones flatten the
    output toward Bartlett. It vanishes with the trace, so a covariance (or
    one matrix of a stack) carrying no power leaves its output undefined:
    NaN, with a warning. A candidate whose ``w^H (R + δ I)^-1 w`` is not
    positive — a zero weight row, or a covariance that is not
    positive-definite — is NaN too: neither is a power, and a finite value
    there would sit in the output as a genuine peak. This is *not* the
    Cox/Zeskind/Owen white-noise-constrained processor.

    ``normalize``: ``None`` leaves the power unscaled; ``'max'`` divides
    the whole output by its largest finite value, so it peaks at 1.

    :func:`uacpy.sonar.mvdr` and :meth:`uacpy.core.results.Covariance.mvdr`
    are this function over a :class:`~uacpy.core.results.Replicas` set.

    Parameters
    ----------
    covariance : ndarray
        ``(N, N)`` element covariance, or an ``(F, N, N)`` stack.
    weights : ndarray
        Weight rows ``(..., N)``, or ``(F, ..., N)`` against a stack.
    diagonal_loading : float, optional
        Loading as a fraction of ``tr(R)/N``. Default 1e-6.
    normalize : {None, 'max'}, optional
        Scale of the output (see above).
    """
    if normalize not in (None, 'max'):
        raise ConfigurationError(
            f"mvdr: normalize must be None or 'max'; got {normalize!r}.")
    R, W, stacked = _weight_slices(covariance, weights, 'mvdr')
    out = np.empty(W.shape[:2], dtype=float)
    for f in range(R.shape[0]):
        who = f"mvdr (frequency bin {f})" if stacked else "mvdr"
        if _powerless_covariance(R[f], who):
            out[f] = np.nan
            continue
        denom = quadratic_form(loaded_inverse(R[f], diagonal_loading), W[f])
        with np.errstate(divide='ignore', invalid='ignore'):
            out[f] = np.where(denom > 0, 1.0 / denom, np.nan)
    if normalize == 'max' and not np.all(np.isnan(out)):
        peak = np.nanmax(out)
        if peak and np.isfinite(peak):
            out = out / peak
    shape = np.shape(weights)[:-1]
    return out.reshape(shape) if stacked else out[0].reshape(shape)


def music_spectrum(covariance, steering, n_sources: int):
    """MUSIC pseudospectrum vs angle.

    ``P(theta) = 1 / (e^H E_n E_n^H e)`` where ``E_n`` spans the noise subspace
    (eigenvectors of the element ``covariance`` beyond the ``n_sources``
    largest eigenvalues).
    ``steering`` is the ``(n_angles, n_elements)`` matrix from
    :func:`steering_vectors`.

    Parameters
    ----------
    covariance : ndarray
        ``(N, N)`` element covariance.
    steering : ndarray
        ``(n_angles, N)`` steering vectors.
    n_sources : int
        Signal-subspace dimension.
    """
    R = np.asarray(covariance, dtype=complex)
    n = R.shape[0]
    if not 1 <= n_sources < n:
        raise ConfigurationError(
            f"music_spectrum: n_sources must be in [1, {n - 1}], got {n_sources}."
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

    Parameters
    ----------
    n_elements : int
        Elements in the array.
    window : str or tuple, optional
        A :func:`scipy.signal.get_window` spec. Default ``'hann'``.
    """
    w = get_window(window, int(n_elements), fftbins=False)
    return w / np.sqrt(np.mean(w ** 2))


def beamform(
    pressure: np.ndarray,
    positions_m: np.ndarray,
    angles_deg: np.ndarray,
    frequency: float,
    *,
    source_level_dB: float,
    noise_level_dB: float,
    sound_speed: float = DEFAULT_SOUND_SPEED,
    weights: Optional[np.ndarray] = None,
    ranges=None,
) -> BeamformResult:
    """
    Plane-wave beamformer — returns signal-to-noise ratio per look angle.

    Performs conventional plane-wave beamforming on ``pressure`` interpreted
    as a transfer function from a 0-dB source. The returned value is the
    receive level minus the ambient noise level, i.e. SNR in dB. The first
    four arguments are in :func:`beamform_field`'s order.

    Parameters
    ----------
    pressure : ndarray
        Pressure transfer-function data with shape (n_phones, n_ranges).
        Can be complex-valued.
    positions_m : ndarray
        Hydrophone positions along the array axis (m), one per row of
        ``pressure``.
    angles_deg : ndarray
        Beam angles in degrees relative to broadside, e.g.
        ``np.arange(-90, 91)``.
    frequency : float
        Frequency in Hz.
    source_level_dB : float
        Source level in dB re 1 µPa @ 1 m. Required: the SNR is only as
        right as this number, so there is no default to fall back on.
    noise_level_dB : float
        **Per-element, wideband** noise level in dB at the receiver
        (i.e. dB re 1 µPa², already integrated over the signal
        bandwidth). The unit-normalised steering vector already folds
        the array gain ``10·log10(N)`` into ``|e.conj() @ pressure|``, so do
        not pre-correct it for the number of elements. For a PSD
        in dB re 1 µPa²/Hz, multiply by the integration bandwidth in
        Hz before passing. Pass ``0`` (with ``source_level_dB=0``) to read
        the receive level of a 0-dB source.
    sound_speed : float, optional
        Reference sound speed for steering vectors in m/s.
    weights : ndarray, optional
        Per-element shading, one value per hydrophone — e.g. from
        :func:`shading_taper`, but any real or complex vector will do.
        Applied to the steering bank and
        re-normalised to unit row norm, so the noise gain stays 1 and the
        peak of a matched plane wave is the array gain the taper leaves,
        ``10*log10(|sum w|**2 / ||w||**2)``. Default: unshaded.
    ranges : array_like, optional
        The range (m) of each column of ``pressure``, carried on the result
        as ``BeamformResult.ranges`` so the ``snr`` axis keeps its values.

    Returns
    -------
    BeamformResult
        ``(snr, angles, peak_snr)``, with ``ranges`` as an attribute:
    snr : ndarray
        Signal-to-noise ratio in dB with shape (n_angles, n_ranges).
    angles_out : ndarray
        Angles used for beamforming (degrees from broadside).
    peak_snr : float
        Maximum value of ``snr``.

    Notes
    -----
    The beamformer computes::

        snr = 20·log10(|e.conj() @ pressure|) + source_level_dB - noise_level_dB

    where ``e`` is the unit-normalised steering-vector matrix from
    :func:`steering_vectors`.

    References
    ----------
    Original MATLAB code by mbp, 2 March 2001
    """
    angles = np.atleast_1d(np.asarray(angles_deg, dtype=float))
    if angles.ndim != 1:
        raise ConfigurationError(
            f"beamform: angles_deg must be a 1-D array of look angles "
            f"(degrees); got shape {angles.shape}.")
    pressure = require_finite_signal(pressure, "beamform", "pressure")
    frequency = require_positive_finite_scalar(
        frequency, "beamform", "frequency", " Hz")
    sound_speed = require_positive_finite_scalar(
        sound_speed, "beamform", "sound_speed", " m/s")
    for name, level in (("source_level_dB", source_level_dB),
                        ("noise_level_dB", noise_level_dB)):
        if not (np.ndim(level) == 0 and np.isfinite(level)):
            raise ConfigurationError(
                f"beamform: {name} must be a finite scalar in dB; got "
                f"{level!r}.")
    if ranges is not None:
        ranges = np.array(ranges, dtype=float)
        if pressure.ndim != 2 or ranges.shape != (pressure.shape[1],):
            raise ConfigurationError(
                f"beamform: ranges gives the range of each column of a "
                f"(n_phones, n_ranges) pressure, so it must be 1-D with "
                f"{pressure.shape[1] if pressure.ndim == 2 else 'n_ranges'} "
                f"values; got shape {ranges.shape} for pressure of shape "
                f"{pressure.shape}.")
    e = _shaded_steering(positions_m, angles, frequency, sound_speed, weights,
                         'beamform')
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
        snr = 20 * np.log10(mag) + float(source_level_dB) - float(noise_level_dB)
    peak_snr = np.max(snr)

    return BeamformResult(snr, angles, peak_snr, ranges=ranges)


def _shaded_steering(positions_m, angles_deg, frequency, sound_speed, weights,
                     who):
    """Unit-norm steering bank, optionally shaded and re-normalised.

    The unit row norm is the convention that keeps the NOISE gain at 1, so a
    beam output stays readable as an SNR: without it a taper would rescale
    the noise along with the signal, and the peak of a matched plane wave
    would no longer be the array gain the taper leaves.
    """
    e = steering_vectors(positions_m, angles_deg, frequency, sound_speed)
    if weights is None:
        return e
    # Complex is kept, not cast away: a shading may carry phase — a fixed
    # taper steered off the scan grid, a null placed by design, an adaptive
    # weight vector. ``dtype=float`` on those discards the imaginary part
    # with only a numpy warning and returns a plausible wrong number.
    w = np.asarray(weights)
    if not np.issubdtype(w.dtype, np.number):
        raise ConfigurationError(
            f"{who}: weights must be numeric, one per element; got dtype "
            f"{w.dtype}."
        )
    w = w.astype(complex if np.iscomplexobj(w) else float)
    if w.shape != (e.shape[1],):
        raise ConfigurationError(
            f"{who}: weights must be one per element, shape "
            f"({e.shape[1]},); got {w.shape}. These are element shadings "
            f"(see shading_taper), not per-angle weights."
        )
    e = e * w[None, :]
    norm = np.linalg.norm(e, axis=1, keepdims=True)
    if not np.all(norm > 0.0):
        # 0/0 would hand back a bank of NaNs and every beam power downstream
        # would be NaN with nothing to say why.
        # The shaded row norm is sqrt(sum|w|^2 / N), which does not depend
        # on the look angle, so this is all-or-nothing: it fires only when
        # every weight is zero.
        raise ConfigurationError(
            f"{who}: weights carry no power, so the shaded steering "
            f"vector has no norm to divide by; got weights summing to "
            f"{float(np.sum(np.abs(w))):.3g} in modulus."
        )
    return e / norm


_BeamformedFields = namedtuple("BeamformedField",
                               "response angles element_power frequencies")


class BeamformedField(PlottedResult, _BeamformedFields):
    """Beam output over a whole field grid, with the reference to read it against.

    ``response`` is the COMPLEX beam output, shape ``(n_angles, *grid)`` —
    every axis of the input after the element axis is carried through
    untouched, so a ``(n_elements, n_depths, n_ranges)`` plane gives a beam
    output per look angle at every point of that plane. ``power`` is its
    modulus squared, which is what an energy detector sees; the phase is
    kept because a time-domain reception cannot be rebuilt without it.

    ``element_power`` is the mean intensity across the elements at each
    point — the single-sensor reference an array gain is measured against.

    ``frequencies`` is ``None`` for a single-frequency beamformer. When it
    is an array, the LAST axis of ``response`` is frequency, every bin was
    steered at its own frequency, and :meth:`to_time_trace` can synthesise
    what the beam actually receives.

    ``grid_coords`` names the grid: a dict holding one 1-D coordinate array
    per grid axis, in axis order (the frequency axis of a band is
    ``frequencies``), when :func:`beamform_field` was given it, ``None``
    otherwise. It rides as an attribute, survives pickling and copying, and
    takes no part in equality.
    """

    _attrs = ("grid_coords",)

    def __new__(cls, response, angles, element_power, frequencies, *,
                grid_coords=None):
        self = super().__new__(cls, response, angles, element_power,
                               frequencies)
        self.grid_coords = grid_coords
        return self

    @property
    def power(self):
        """Beam power, ``|response|**2`` — what an energy detector sees."""
        return np.abs(self.response) ** 2

    @property
    def best(self):
        """Power of the winning beam at each point — a scanning detector's output."""
        return self.power.max(axis=0)

    @property
    def best_angle(self):
        """Look angle (deg) that won at each point.

        On a multi-mode arrival this hops between +/- theta rather than
        migrating smoothly, because the modes arrive in up- and down-going
        pairs and the interference decides which one wins.
        """
        return np.asarray(self.angles, dtype=float)[
            np.argmax(self.power, axis=0)]

    def array_gain(self):
        """Realised array gain (dB) at each point:
        :func:`realised_array_gain` of this field's ``response`` over its
        ``element_power`` (what the ratio is, and where it is the array
        gain, is set out there)."""
        return realised_array_gain(self.response, self.element_power)

    _plotter = "plot_beam_power"

    def plot(self, ax=None, **kwargs):
        """Draw the beam power against look angle.

        Dispatches to :func:`uacpy.plot.plot_beam_power`. Spelled
        ``plot()`` like the other carriers, because a beamformed field has
        only the one rendering — unlike a :class:`~uacpy.Source`, whose
        ``plot_beam_pattern`` is named for its attribute because its other
        rendering is the marker an environment draws.
        """
        return super().plot(ax=ax, **kwargs)

    def _field_units(self):
        # response and element_power carry the input field's samples,
        # steered and summed, unscaled.
        return {"response": None, "angles": "deg", "element_power": None,
                "frequencies": "Hz"}

    def to_time_trace(self, angle_deg, *, range, source_spectrum=None,
                      source_waveform=None, sample_rate=None, window='auto',
                      nfft=None, t_start=None):
        """What one beam actually receives, as a time series.

        Only defined for a broadband beamformer — a single frequency has no
        band to transform. Selects the look angle nearest ``angle_deg`` and
        synthesises its complex response through the package's one IFFT
        synthesis, the one :meth:`uacpy.Field.to_time_trace` runs, so the
        window, the time base and the source convolution are not a second
        implementation of them. Returns a :class:`~uacpy.Field` over
        ``time`` with the separation pinned as its ``range``; a beam has no
        depth, so none is pinned.

        ``range`` is the source-to-array separation in metres, keyword-only
        like :meth:`Field.to_time_trace`'s, and it is required
        rather than defaulted: it sets where the time window starts, and
        there is no range at which a default would be right. Left at zero
        the window would open at t = 0 — before anything can have arrived —
        and the wrap warning below is itself suppressed at t_start = 0, so
        the one certainly-wrong time base would be the one that said
        nothing.

        ``source_spectrum`` is the transmitted signal's spectrum on this
        object's ``frequencies`` (or ``source_waveform`` with its
        ``sample_rate``, whose exact spectrum is evaluated there); without
        either the result is the channel's band-limited impulse response
        through the beam.

        ``window``, ``nfft`` and ``t_start`` are :meth:`Field.to_time_trace`'s:
        ``window='auto'`` (the default) picks ``'hann'`` for the bare impulse
        response and no window with a source; ``None`` is always rectangular.
        ``t_start`` is worth setting: the default window is anchored on a
        nominal sound speed, and a faster path arrives before the window and
        wraps to the end of the record.
        """
        # Local because this is the only method that needs them, and
        # beamforming.py is imported on nearly every uacpy path.
        from uacpy.core.results import Field, PhaseReference
        from uacpy.acoustic_signal._synthesis import (
            check_source_spectrum, synthesis_grid, synthesize_trace,
            warn_band_edge_cuts_spectrum, waveform_spectrum_on)
        who = "BeamformedField.to_time_trace"
        if self.frequencies is None:
            raise ConfigurationError(
                "BeamformedField.to_time_trace: this beamformer ran at a "
                "single frequency, so there is no band to transform. Pass a "
                "frequency array to beamform_field (the shape a BROADBAND "
                "run returns) to synthesise a reception."
            )
        if self.response.ndim != 2:
            raise ConfigurationError(
                f"BeamformedField.to_time_trace: needs the beam at one "
                f"point in space, i.e. response of shape (n_angles, "
                f"n_frequencies); got {self.response.shape}. Select the "
                f"grid point first — a whole plane of receptions is a set "
                f"of traces, not one."
            )
        freqs = np.asarray(self.frequencies, dtype=float)
        spectrum = waveform_spectrum_on(freqs, source_waveform, sample_rate,
                                        source_spectrum, who)
        if spectrum is not None:
            check_source_spectrum(spectrum, freqs.size, who)
        if window == 'auto':
            window = 'hann' if spectrum is None else None
        # The transmitted signal's duration, for the record-edge notice.
        if source_waveform is not None:
            pulse_s = np.size(source_waveform) / float(sample_rate)
        elif spectrum is None:
            pulse_s = 0.0
        else:
            pulse_s = None
        idx = int(np.argmin(np.abs(np.asarray(self.angles, dtype=float)
                                   - float(angle_deg))))
        separation = float(range)
        plan = synthesis_grid(freqs, window=window, nfft=nfft,
                              sample_rate=None, n_samples_floor=0,
                              who=who)
        warn_band_edge_cuts_spectrum(spectrum, window, who)
        time, trace = synthesize_trace(
            np.asarray(self.response)[idx], spectrum, plan, range=separation,
            t_start=t_start, c_max=0.0, c0=0.0, c_slow=0.0,
            cell_label=f"range {separation:g} m", who=who,
            pulse_s=pulse_s)
        return Field(
            data=trace, coords={'time': time}, pinned={'range': separation},
            phase_reference=PhaseReference.TIME_DOMAIN_NATIVE,
            synthesis_window=window,
            # Without a source spectrum the trace is the band-limited
            # impulse response, per second.
            **({'kind': 'impulse_response', 'unit': '1/s'}
               if spectrum is None else {}))


def beamform_field(pressure, positions_m, angles_deg, frequency, *,
                   sound_speed: float = DEFAULT_SOUND_SPEED,
                   weights=None, grid_coords=None) -> BeamformedField:
    """Conventional beamformer over every point of a field.

    :func:`beamform` answers "what does this array hear along one range
    line", in dB SNR. This answers "what does it hear at every point of a
    modelled field", in power, and keeps the look angle that won — which is
    what a coverage map, a realised array gain or a re-steering study needs.

    Parameters
    ----------
    pressure : ndarray
        Complex pressure, element axis FIRST: ``(n_elements, *grid)``. The
        grid may be anything — ``(n_ranges,)``, ``(n_depths, n_ranges)`` —
        and comes back unchanged behind the angle axis.
    positions_m : array_like
        Element positions (m), one per element of ``pressure``'s first axis.
    angles_deg : array_like
        Look angles to scan, in degrees from broadside.
    frequency : float or array_like
        One frequency, or a whole band. Given a band, the LAST axis of
        ``pressure`` is taken as the frequency axis — the shape a
        ``RunMode.BROADBAND`` run returns — and **each bin is steered at its
        own frequency**. That is not a refinement: a beam delay is a
        frequency-dependent phase, so one steering vector at the band centre
        mis-steers both edges. A band also makes
        :meth:`BeamformedField.to_time_trace` available.
    sound_speed : float, optional
        Reference sound speed (m/s) for the steering vectors.
    weights : array_like, optional
        Per-element shading, one value per element — any real or **complex**
        vector, not only a :func:`shading_taper` output. Applied to the
        steering bank and re-normalised to unit row norm, so the noise gain
        stays 1 whatever scale the weights arrive at. Complex entries carry
        phase, which is how a fixed taper steers off the scan grid or places
        a null by design. What this is *not* is a per-angle weight **bank**:
        an adaptive design with a different vector per look angle is a
        different object, and a ``(n_angles, n_elements)`` array is refused
        rather than broadcast.
    grid_coords : dict, optional
        One 1-D coordinate array per grid axis, in axis order, e.g.
        ``{'depth': z, 'range': r}`` for ``(n_elements, n_depths,
        n_ranges)``; the frequency axis of a band is not a grid axis. Carried
        on the result as ``BeamformedField.grid_coords``.

    Returns
    -------
    BeamformedField
        ``response`` (complex) ``(n_angles, *grid)``, ``angles``,
        ``element_power`` ``(*grid,)`` and ``frequencies``, with
        ``grid_coords`` as an attribute; with ``.power``, ``.best``,
        ``.best_angle``, ``.array_gain()`` and — for a band —
        ``.to_time_trace()``.

    Examples
    --------
    A plane wave from 20° on a 16-element λ/2 line (λ = 1 m) is found at
    20°, with the matched gain ``10·log10(16)``:

    >>> import numpy as np
    >>> positions = 0.5 * np.arange(16)
    >>> p = np.exp(-2j * np.pi * positions * np.sin(np.radians(20.0)))
    >>> beams = beamform_field(p[:, None], positions,
    ...                        np.linspace(-60.0, 60.0, 121), 1500.0)
    >>> float(beams.best_angle[0]), round(float(beams.array_gain()[0]), 2)
    (20.0, 12.04)
    """
    p = np.asarray(pressure)
    pos = np.asarray(positions_m, dtype=float).ravel()
    if p.ndim < 1 or p.shape[0] != pos.size:
        raise ConfigurationError(
            f"beamform_field: the FIRST axis of pressure is the element "
            f"axis, so it must be {pos.size} long to match positions_m; got "
            f"shape {p.shape}. Grid axes follow it — e.g. "
            f"(n_elements, n_depths, n_ranges)."
        )
    angles = np.asarray(angles_deg, dtype=float).ravel()
    # A frequency AXIS (more than one value) makes the last axis of
    # ``pressure`` the frequency axis and turns this into a broadband
    # beamformer; a scalar keeps the narrowband shape exactly as it was.
    # An ARRAY means broadband, however short. Switching on size instead
    # would make a one-bin band silently narrowband, and to_time_trace
    # would then tell the caller to pass a frequency array — which they did.
    freq_in = np.asarray(frequency, dtype=float)
    freqs = np.atleast_1d(freq_in) if freq_in.ndim > 0 else None
    freq_arr = np.atleast_1d(freq_in)
    if freqs is not None:
        if p.ndim < 2 or p.shape[-1] != freqs.size:
            raise ConfigurationError(
                f"beamform_field: {freqs.size} frequencies were given, so "
                f"the LAST axis of pressure is the frequency axis and must "
                f"be {freqs.size} long; got shape {p.shape}. A broadband "
                f"field is (n_elements, *grid, n_frequencies) — the shape "
                f"a BROADBAND run returns."
            )
        e = None
    else:
        e = _shaded_steering(pos, angles, float(freq_arr[0]), sound_speed,
                             weights,
                             "beamform_field")
    if grid_coords is not None:
        grid_coords = _checked_grid_coords(
            grid_coords, p.shape[1:-1] if freqs is not None else p.shape[1:],
            freqs is not None)
    # One matrix product over the element axis, with every grid axis folded
    # into a single column axis and restored afterwards, so the same code
    # serves a range line and a depth-range plane.
    if freqs is None:
        # One matrix product over the element axis, with every grid axis
        # folded into a single column axis and restored afterwards, so the
        # same code serves a range line and a depth-range plane.
        flat = np.reshape(p, (p.shape[0], -1))
        resp = np.reshape(e.conj() @ flat, (angles.size,) + p.shape[1:])
    else:
        # A beam delay is a frequency-dependent phase, so each bin gets its
        # own steering bank. Steering the band once at its centre mis-steers
        # both edges, by the fractional bandwidth times the steer angle.
        flat = np.reshape(p, (p.shape[0], -1, freqs.size))
        resp = np.empty((angles.size, flat.shape[1], freqs.size),
                        dtype=complex)
        for i, f_i in enumerate(freqs):
            e_i = _shaded_steering(pos, angles, f_i, sound_speed, weights,
                                   "beamform_field")
            resp[:, :, i] = e_i.conj() @ flat[:, :, i]
        resp = np.reshape(resp, (angles.size,) + p.shape[1:])
    return BeamformedField(resp, angles, np.mean(np.abs(p) ** 2, axis=0),
                           freqs, grid_coords=grid_coords)


def _checked_grid_coords(grid_coords, grid_shape, band: bool) -> dict:
    """``grid_coords`` as ``{name: float copy}``, one 1-D array per axis of
    ``grid_shape`` and as long as it; refused otherwise."""
    coords = dict(grid_coords)
    if len(coords) != len(grid_shape):
        raise ConfigurationError(
            f"beamform_field: grid_coords names {len(coords)} axes "
            f"({list(coords)}), but pressure has {len(grid_shape)} grid "
            f"axes {tuple(grid_shape)} after the element axis"
            + (" (its last axis is the frequency band, which frequencies "
               "names)" if band else "") + ".")
    checked = {}
    for (name, values), n in zip(coords.items(), grid_shape):
        arr = np.array(values, dtype=float)
        if arr.shape != (n,):
            raise ConfigurationError(
                f"beamform_field: grid_coords[{name!r}] must be 1-D with "
                f"{n} values, one per point of that grid axis; got shape "
                f"{arr.shape}.")
        checked[str(name)] = arr
    return checked


def plane_wave_array_gain(weights) -> float:
    """Array gain (dB) a weight vector realises on a BROADSIDE plane wave.

    ``AG = |sum w|^2 / ||w||^2`` — the coherent sum of the weights against
    the noise power a unit-norm version of them passes. For an unshaded
    array this is ``10log10(N)``; a Hann taper spends about 1.95 dB of it on
    sidelobes.

    **Broadside, not "matched".** For a real taper the two are the same
    thing: the weights are in phase, so the wave they are matched to is the
    one arriving broadside. A *complex* taper steers, and then they part
    company — its matched wave arrives at the steered angle, and this
    function still reports the gain at broadside. With a phase ramp of 8
    radians across 16 elements that is -0.33 dB here against 10.0 dB for
    the wave the taper is actually matched to, found by
    :meth:`BeamformedField.array_gain`. Both are right; they answer
    different questions, and which one a budget wants depends on where the
    signal is coming from.

    The broadside value is the one that keeps a superdirective design
    honest: Butler & Sherman's alternating shading gives a genuinely
    NEGATIVE broadside gain, which a magnitudes-only form would erase.

    This is Abraham's shaded-line-array directivity index at the DESIGN
    frequency: ``DI ~ 10log10[(f_c/f_d)(sum w)^2 / sum w^2]`` with
    ``f_c = f_d``. Away from the design frequency that leading factor is
    real — see :meth:`BeamformedField.array_gain`, which measures it.

    It is the **white-noise gain** — the gain against spatially
    uncorrelated noise. That equals the directivity index against isotropic
    noise only at half-wavelength spacing; against any other noise field the
    sonar equation's AG needs that field's coherence (isotropic noise has a
    ``sinc(k d)`` cross-spectral matrix), which this function does not model.

    It is **not** ``-10log10(sum |w|^4)``, which is the inverse of a
    normalised effective element count: the two agree for a boxcar and
    differ by about 1.1 dB for a Hann taper, which is small enough to read
    as plausible and wrong enough to matter.

    Scale-invariant, so an unnormalised taper may be passed directly, and
    defined for complex weights too — ``|sum w|`` is then a coherent sum, so
    a phase ramp across the elements legitimately lowers the gain. Weights
    that sum to zero null the steered direction, which Butler & Sherman's
    alternating superdirective shading does on purpose; the answer is
    ``-inf`` dB rather than an error.

    Parameters
    ----------
    weights : array_like
        Element weights, real or complex, any scale.
    """
    w = np.asarray(weights)
    if not np.issubdtype(w.dtype, np.number):
        raise ConfigurationError(
            f"plane_wave_array_gain: weights must be numeric; got dtype "
            f"{w.dtype}."
        )
    w = w.astype(complex if np.iscomplexobj(w) else float).ravel()
    if w.size == 0:
        raise ConfigurationError(
            "plane_wave_array_gain: weights is empty; expected one shading "
            "per element."
        )
    denom = float(np.sum(np.abs(w) ** 2))
    if denom <= 0.0:
        raise ConfigurationError(
            "plane_wave_array_gain: weights carry no power, so the gain is "
            "undefined; got all zeros."
        )
    # A weight vector summing to zero nulls the steered direction outright
    # — Butler & Sherman's alternating superdirective shading does exactly
    # that — so -inf is the answer, not an error and not a numpy warning.
    with np.errstate(divide='ignore'):
        return float(10.0 * np.log10(np.abs(np.sum(w)) ** 2 / denom))


def realised_array_gain(response, element_power):
    """Realised array gain (dB) at each point: best beam over mean element.

    ``response`` is a complex beam output ``(n_angles, *grid)`` and
    ``element_power`` the mean element intensity ``(*grid,)`` it is
    measured against: what :func:`beamform_field` returns as
    ``BeamformedField.response`` and ``.element_power``, whose
    :meth:`~BeamformedField.array_gain` is this function. The pair to
    :func:`matched_replica_gain`, the same ratio for the matched replica.

    This is Ainslie's Equation (6.70) prescription — the signal-to-noise
    ratio "calculated not just once, but twice, with and without the
    effects of the beamformer" — evaluated on a modelled field, and it
    is what a scalar AG misses when the signal is not one plane wave.

    The reference is the MEAN element power, not any single element's: a
    lone hydrophone can sit in an interference null, which reads as
    enormous "gain" from an array whose ceiling is ``10log10(N)``.

    **What this ratio is, exactly.** It is a ratio of SIGNAL powers.
    Array gain is a ratio of signal-to-NOISE ratios — Abraham gives it
    for a general noise covariance ``Q`` and weight vector ``w`` as
    ``G_a = v|w^H d|^2 / (w^H Q w)`` (*Underwater Acoustic Signal
    Processing*, 8.4.4) — so the two agree only when the beamformer
    passes the noise unchanged:

        AG = signal gain - 10log10(w^H R_n w)

    The weights are unit-norm, so the noise term vanishes when the noise
    is spatially uncorrelated across the elements — which for isotropic
    3-D noise, coherence ``sinc(k*d)``, holds at every integer multiple
    of ``lambda/2`` and at no spacing between them. Only the first is
    useful: the rest are grating-lobed. At the design frequency of a
    half-wavelength array this method IS the array gain. Away from it
    the noise term is real and is NOT included here. For a 24-element
    lambda/2-at-200-Hz array with a Hann taper (the case example 42
    runs), the correction is

        150 Hz  +1.25 dB      225 Hz  -0.51 dB
        175 Hz  +0.58 dB      250 Hz  -0.97 dB
        200 Hz   0.00 dB      300 Hz  -1.76 dB

    so a broadband call returns per-bin signal gains that are array
    gains only in the bin where the spacing is half a wavelength. A
    positive correction means this method OVERSTATES the array gain.

    Those entries are the exact quadratic form ``w^H R_n w`` for
    ``R_n = sinc(k*d_ij)``, and they land on the ``f_c/f_d`` factor in
    Abraham's shaded-line-array directivity index,
    ``DI ~ 10log10[(f_c/f_d)(sum w)^2 / sum w^2]`` — his "10-dB-per-
    decade reduction when operating the array below the design
    frequency". Abraham writes that with ``~``, and the size of the
    approximation is exactly statable. Writing the quadratic form as an
    integral of the array factor over ``s = sin(theta)``,

        w^H R_n w = (1/2u) * integral_{-u}^{u} |W(pi*s)|^2 ds,
        u = f_c / f_d

    and using ``integral_{-1}^{1} |W|^2 ds = 2||w||^2``, the departure
    from Abraham's factor IS the array-factor energy lying outside
    ``|sin(theta)| <= f_c/f_d`` — an identity, verified here to 1e-14.

    So the condition is about where a window puts its energy, not about
    which window it is: one tapering to zero at the edges leaves almost
    nothing outside (~1e-6 dB), while an unshaded array leaves 1.3 % of
    it out at half the design frequency, worth 0.058 dB. Predict your
    own case from that rather than from this table, which is one
    taper's arithmetic. Above the design frequency ``u > 1``, the
    containment window exceeds a period, and the agreement genuinely
    breaks — which is also where the array is spatially aliased and
    Abraham's approximation is not intended at all.
    Real ocean noise is not isotropic either (Butler & Sherman 8.3.1:
    "sea noise is probably never isotropic"), which moves it again.

    Parameters
    ----------
    response : ndarray
        Complex beam output ``(n_angles, *grid)``.
    element_power : ndarray
        Mean element intensity ``(*grid,)``.
    """
    r = np.asarray(response)
    if r.ndim < 1 or r.shape[0] < 1:
        raise ConfigurationError(
            "realised_array_gain: the first axis of response is the "
            f"look-angle axis and must be non-empty; got shape {r.shape}.")
    best = (np.abs(r) ** 2).max(axis=0)
    element_power = np.asarray(element_power)
    if element_power.shape != best.shape:
        raise ConfigurationError(
            f"realised_array_gain: element_power must have the grid shape "
            f"{best.shape} of response after its angle axis; got "
            f"{element_power.shape}.")
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(best / element_power)


def matched_replica_gain(pressure):
    """Array gain (dB) of the replica matched to the field itself.

    The ceiling a conventional scan is measured against. Correlating the
    field with ``p / ||p||`` gives ``||p||^2`` against a mean element power
    of ``||p||^2 / N``, so the gain is ``10log10(N)`` exactly — whatever
    shape the field has, with no plane wave anywhere in it. That is the
    point: the shortfall a plane-wave scan shows in a waveguide belongs to
    the *replica*, not to the channel, and the right replica is the
    channel's own Green's function (matched-field processing — see
    :mod:`uacpy.sonar.matched_field`).

    Computed from ``pressure`` rather than returned as a constant, so a
    point with no signal comes back NaN instead of a confident number.

    Parameters
    ----------
    pressure : array_like
        Complex pressure, elements on the first axis.
    """
    p = np.asarray(pressure)
    if p.ndim < 1 or p.shape[0] < 1:
        raise ConfigurationError(
            "matched_replica_gain: the first axis of pressure is the element "
            f"axis and must be non-empty; got shape {p.shape}."
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(np.linalg.norm(p, axis=0) ** 2
                               / np.mean(np.abs(p) ** 2, axis=0))


def _decorrelation_spacing(positions_m, weights, frequency, sound_speed,
                           unshaded):
    """Spacing in ``sin(theta)`` at which two beams' NOISE outputs decorrelate.

    Two beams a distance ``delta`` apart in ``sin(theta)`` have outputs
    whose correlation in spatially white noise is

        w(theta)^H w(theta+delta) = sum_n |w_n|^2 exp(-j k z_n delta)

    — the array factor of the POWER weights, not of the weights. Two
    consequences fall out of that form: a phase ramp on the taper cancels
    exactly (steering a beam moves it without widening it), and the
    relevant width is that of ``|w|^2``, which is broader than the beam
    pattern's. For a Hann taper the pattern's first null sits at two DFT
    bins while ``hann^2`` reaches its first null at three, so the looks a
    scan really has are fewer than the *resolution* argument suggests.

    This is deliberately NOT the resolution criterion — "half the
    first-null beamwidth (FNBW/2)" (Balanis, *Antenna Theory*, 2) — which
    answers a different question: whether two SOURCES can be told apart.
    A false-alarm count is about when two beams' NOISE stops being shared,
    and those coincide only for an unshaded array, where both reduce to
    ``lambda/(N*d)``.
    """
    z = np.asarray(positions_m, dtype=float).ravel()
    w = np.abs(np.asarray(weights)) ** 2
    k = 2.0 * np.pi * float(frequency) / float(sound_speed)
    # Search out to 8 unshaded cells: Blackman needs ~3, and anything
    # broader than 8 is a weighting with essentially no main lobe.
    grid = np.linspace(unshaded / 512.0, 8.0 * unshaded, 8193)
    af = np.abs(np.exp(-1j * k * np.outer(grid, z)) @ w)
    rising = np.flatnonzero(np.diff(np.sign(np.diff(af))) > 0)
    if rising.size == 0:
        log_message(
            "independent_beams",
            "these weights give a beam with no null within 8 unshaded "
            "cells, so the resolution cell cannot be measured; falling back "
            "to the unshaded spacing, which OVER-counts the looks and so "
            "sets a stricter threshold than needed.",
            level="warning",
        )
        return unshaded
    return float(grid[rising[0] + 1])


def independent_beams(positions_m, angles_deg, frequency,
                      sound_speed: float = DEFAULT_SOUND_SPEED, *,
                      weights=None) -> float:
    """How many INDEPENDENT looks a scan over ``angles_deg`` really holds.

    Beams spaced ``lambda / (N*d)`` apart in ``sin(theta)`` are mutually
    orthogonal for a uniform line array — the DFT spacing — so a sector
    spanning ``span`` in ``sin(theta)`` holds ``span * N*d / lambda``
    resolution cells however finely it is sampled. Scanning 361 angles does
    not buy 361 looks. Over the whole visible region a half-wavelength array
    of N sensors gives exactly N, which is Stergiopoulos's count: "with any
    array of (2N + 1) sensors, we may produce a beamforming network with
    (2N + 1) orthogonal beam ports ... [they] represent 2N independent look
    directions, one per beam" (*Advanced Signal Processing Handbook*, 2).

    Pass ``weights`` and the count becomes taper-aware. Shading widens the
    cell, so a shaded scan holds FEWER independent looks than the DFT count
    — measured here, a Hann taper costs a factor 3.2 at 16 elements (3.0 in
    the limit) and Blackman 5.3. Omitting ``weights`` on a shaded array
    over-counts the looks, which sets a stricter threshold than necessary:
    safe, but it spends detection performance.

    Feed the result to :func:`uacpy.sonar.per_look_false_alarm`: a detector
    that keeps the largest of these looks false-alarms at the scan's rate,
    not at one beam's.

    ``N*d`` is the element span plus one mean spacing, which is exactly
    ``N*d`` for a uniform array and degrades sensibly for a ragged one. Note
    this is NOT the aperture ``(N-1)*d``: beams at the aperture spacing are
    still correlated: |corr| = 1/N at that spacing, 0.062 for 16
    elements and 0.042 for 24, against 1e-16 here. Being the wider spacing
    it also yields FEWER cells — for a +/-45 deg scan, 10.61 against 11.31
    at N=16 and 16.26 against 16.97 at N=24 — so a threshold set from it is
    set for fewer looks than the detector really takes, and is optimistic.

    Parameters
    ----------
    positions_m : array_like
        Element coordinates along the array axis (m).
    angles_deg : array_like
        Look angles from broadside (deg), positive downward.
    frequency : float
        Frequency (Hz).
    sound_speed : float, optional
        Sound speed (m/s). Default
        :data:`~uacpy.core.constants.DEFAULT_SOUND_SPEED`.
    weights : array_like, optional
        The array shading; given, the count is taper-aware.
    """
    frequency = require_positive_finite_scalar(
        frequency, "independent_beams", "frequency", " Hz")
    sound_speed = require_positive_finite_scalar(
        sound_speed, "independent_beams", "sound_speed", " m/s")
    pos = np.asarray(positions_m, dtype=float).ravel()
    if pos.size < 2:
        raise ConfigurationError(
            f"independent_beams: needs at least 2 elements to have a beam "
            f"width at all; got {pos.size}."
        )
    if weights is not None:
        w_arr = np.asarray(weights)
        if not np.issubdtype(w_arr.dtype, np.number):
            raise ConfigurationError(
                f"independent_beams: weights must be numeric, one per "
                f"element; got dtype {w_arr.dtype}."
            )
        if w_arr.shape != (pos.size,):
            raise ConfigurationError(
                f"independent_beams: weights must be one per element, shape "
                f"({pos.size},); got {w_arr.shape}."
            )
    span_m = float(np.ptp(pos))
    if span_m <= 0.0:
        raise ConfigurationError(
            "independent_beams: every element is at the same position, so "
            "the array has no aperture and forms one beam."
        )
    effective_m = span_m * pos.size / (pos.size - 1)      # == N*d if uniform
    span_sin = float(np.ptp(np.sin(np.deg2rad(
        np.asarray(angles_deg, dtype=float)))))
    wavelength = sound_speed / frequency
    unshaded = wavelength / effective_m
    if weights is None:
        return span_sin / unshaded
    return span_sin / _decorrelation_spacing(pos, weights, frequency,
                                             sound_speed,
                                             unshaded)
