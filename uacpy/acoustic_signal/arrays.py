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
    c: float = DEFAULT_SOUND_SPEED,
    *,
    weights: Optional[np.ndarray] = None,
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
    weights : ndarray, optional
        Per-element shading, one value per hydrophone — e.g. from
        :func:`shading_taper`, but any real or complex vector will do.
        Applied to the steering bank and
        re-normalised to unit row norm, so the noise gain stays 1 and the
        peak of a matched plane wave is the array gain the taper leaves,
        ``10*log10(|sum w|**2 / ||w||**2)``. Default: unshaded.

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
    e = _shaded_steering(phone_coords, angles, frequency, c, weights,
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
        snr = 20 * np.log10(mag) + SL - NL
    peak_snr = np.max(snr)

    return BeamformResult(snr, angles, peak_snr)


def _shaded_steering(positions_m, angles_deg, frequency, c, weights, caller):
    """Unit-norm steering bank, optionally shaded and re-normalised.

    The unit row norm is the convention that keeps the NOISE gain at 1, so a
    beam output stays readable as an SNR: without it a taper would rescale
    the noise along with the signal, and the peak of a matched plane wave
    would no longer be the array gain the taper leaves.
    """
    e = steering_vectors(positions_m, angles_deg, frequency, c)
    if weights is None:
        return e
    # Complex is kept, not cast away: a shading may carry phase — a fixed
    # taper steered off the scan grid, a null placed by design, an adaptive
    # weight vector. ``dtype=float`` on those discards the imaginary part
    # with only a numpy warning and returns a plausible wrong number.
    w = np.asarray(weights)
    if not np.issubdtype(w.dtype, np.number):
        raise ConfigurationError(
            f"{caller}: weights must be numeric, one per element; got dtype "
            f"{w.dtype}."
        )
    w = w.astype(complex if np.iscomplexobj(w) else float)
    if w.shape != (e.shape[1],):
        raise ConfigurationError(
            f"{caller}: weights must be one per element, shape "
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
            f"{caller}: weights carry no power, so the shaded steering "
            f"vector has no norm to divide by; got weights summing to "
            f"{float(np.sum(np.abs(w))):.3g} in modulus."
        )
    return e / norm


_BeamformedFields = namedtuple("BeamformedField",
                               "response angles element_power frequencies")


class BeamformedField(_BeamformedFields):
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
    """

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
        """Realised array gain (dB) at each point: best beam over mean element.

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
        3-D noise, coherence ``sinc(k*d)``, holds at ``d = lambda/2`` and
        nowhere else. At the design frequency of a half-wavelength array
        this method IS the array gain. Away from it the noise term is real
        and is NOT included here; for a 24-element lambda/2-at-200-Hz array
        the correction runs

            150 Hz  +1.25 dB      225 Hz  -0.51 dB
            175 Hz  +0.58 dB      250 Hz  -0.97 dB
            200 Hz   0.00 dB      300 Hz  -1.76 dB

        so a broadband call returns per-bin signal gains that are array
        gains only in the bin where the spacing is half a wavelength. A
        positive correction means this method OVERSTATES the array gain.

        That correction is not a private derivation: it reproduces, to
        machine precision at every frequency tried, the ``f_c/f_d`` factor
        in Abraham's shaded-line-array directivity index,
        ``DI ~ 10log10[(f_c/f_d)(sum w)^2 / sum w^2]`` — his "10-dB-per-
        decade reduction when operating the array below the design
        frequency". Above the design frequency the array is spatially
        aliased, where that approximation is not intended and the entries
        below should be read as the isotropic-noise arithmetic only.
        Real ocean noise is not isotropic either (Butler & Sherman 8.3.1:
        "sea noise is probably never isotropic"), which moves it again.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return 10.0 * np.log10(self.best / self.element_power)

    def to_time_trace(self, angle_deg, *, range_m, source_spectrum=None,
                      **kwargs):
        """What one beam actually receives, as a time series.

        Only defined for a broadband beamformer — a single frequency has no
        band to transform. Selects the look angle nearest ``angle_deg`` and
        hands its complex response to
        :meth:`uacpy.core.results.Field.to_time_trace`, so the window, the
        time base and the ``source_spectrum`` convolution are the package's
        own and not a second implementation of them.

        ``range_m`` is the source-to-array separation, and it is required
        rather than defaulted: it sets where the time window starts, and
        there is no range at which a default would be right. Left at zero
        the window would open at t = 0 — before anything can have arrived —
        and the wrap warning below is itself suppressed at t_start = 0, so
        the one certainly-wrong time base would be the one that said
        nothing.

        ``source_spectrum`` is the transmitted signal's spectrum on this
        object's ``frequencies``; without it the result is the channel's
        band-limited impulse response through the beam.

        Remaining keyword arguments (``t_start``, ``window``, ``nfft``) pass
        straight through. ``t_start`` is worth setting: the default window
        is anchored on a nominal sound speed, and a faster path arrives
        before the window and wraps to the end of the record.
        """
        # Local because this is the only method that needs it, and
        # arrays.py is imported on nearly every uacpy path.
        from uacpy.core.results import Field
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
        idx = int(np.argmin(np.abs(np.asarray(self.angles, dtype=float)
                                   - float(angle_deg))))
        beam = np.asarray(self.response)[idx][None, None, :]
        holder = Field(data=beam,
                       coords={'depth': np.array([0.0]),
                               'range': np.array([float(range_m)]),
                               'frequency': np.asarray(self.frequencies,
                                                       dtype=float)},
                       model='beamform_field')
        return holder.to_time_trace(depth=0.0, range=float(range_m),
                                    source_spectrum=source_spectrum, **kwargs)


def beamform_field(pressure, positions_m, angles_deg, frequency, *,
                   c: float = DEFAULT_SOUND_SPEED,
                   weights=None) -> BeamformedField:
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
    c : float, optional
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

    Returns
    -------
    BeamformedField
        ``response`` (complex) ``(n_angles, *grid)``, ``angles``,
        ``element_power`` ``(*grid,)`` and ``frequencies``; with ``.power``,
        ``.best``, ``.best_angle``, ``.array_gain()`` and — for a band —
        ``.to_time_trace()``.
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
        e = _shaded_steering(pos, angles, float(freq_arr[0]), c, weights,
                             "beamform_field")
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
            e_i = _shaded_steering(pos, angles, f_i, c, weights,
                                   "beamform_field")
            resp[:, :, i] = e_i.conj() @ flat[:, :, i]
        resp = np.reshape(resp, (angles.size,) + p.shape[1:])
    return BeamformedField(resp, angles, np.mean(np.abs(p) ** 2, axis=0),
                           freqs)


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


def _decorrelation_spacing(positions_m, weights, frequency, c, unshaded):
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
    k = 2.0 * np.pi * float(frequency) / float(c)
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
                      c: float = DEFAULT_SOUND_SPEED, *,
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
    """
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
    wavelength = float(c) / float(frequency)
    unshaded = wavelength / effective_m
    if weights is None:
        return span_sin / unshaded
    return span_sin / _decorrelation_spacing(pos, weights, frequency, c,
                                             unshaded)


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
#: What :attr:`FKResult.scaling` may hold: ``'density'`` for the
#: calibrated panel of ``fk_transform(..., normalize=True)`` (x² per
#: Hz·rad/m, two-sided in f and k, ``ΣP·Δf·Δk = ⟨x²⟩``) and ``'power'`` for
#: the raw ``|FK|²`` of the windowed, zero-padded FFT, which carries no
#: physical unit.
FK_SCALINGS = ("density", "power")


_FKFields = namedtuple("FKResult", "frequencies wavenumbers power spectrum")


class FKResult(_FKFields):
    """The ``(frequencies, wavenumbers, power, spectrum)`` 4-tuple of
    :func:`fk_transform`, carrying the panel's ``scaling`` as an attribute.

    Unpacking stays four-wide (``f, k, power, spectrum = fk_transform(...)``)
    and :func:`inverse_fk` keeps taking the fourth element. ``scaling`` is one
    of :data:`FK_SCALINGS` and tells :func:`~uacpy.visualization.plot_fk`
    which unit the panel is in, so it is not a fifth tuple element: a fifth
    element would change every unpack site for one flag the plotter reads.
    """

    def __new__(cls, frequencies, wavenumbers, power, spectrum, *, scaling):
        if scaling not in FK_SCALINGS:
            raise ConfigurationError(
                f"FKResult: scaling must be one of {FK_SCALINGS}; got "
                f"{scaling!r}")
        self = super().__new__(cls, frequencies, wavenumbers, power, spectrum)
        self.scaling = scaling
        return self

    def _replace(self, **kwargs):
        scaling = kwargs.pop("scaling", self.scaling)
        fields = [kwargs.pop(name, value)
                  for name, value in zip(self._fields, self)]
        if kwargs:
            raise ValueError(f"Got unexpected field names: {list(kwargs)!r}")
        return FKResult(*fields, scaling=scaling)

    def __getnewargs_ex__(self):
        return tuple(self), {"scaling": self.scaling}

    def __repr__(self):
        return f"{super().__repr__()[:-1]}, scaling={self.scaling!r})"


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
    it is a PSD density per ``Hz·rad/m`` with ``ΣP·Δf·Δk = ⟨x²⟩``, and the
    result's ``scaling`` attribute reads ``'density'``; with ``normalize=False``
    (the default) it is the raw squared magnitude of the windowed, zero-padded
    FFT, which grows with the record size and carries no physical unit, and
    ``scaling`` reads ``'power'``. :func:`~uacpy.visualization.plot_fk` labels
    the panel from that attribute. Whenever the
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
    return FKResult(freqs, wavenumbers, power, spectrum,
                    scaling="density" if normalize else "power")
