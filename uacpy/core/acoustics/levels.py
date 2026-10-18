"""From volts to pascals to decibels.

The three steps a recording takes to become a level: :func:`pressure_from_recording` turns
the recorded voltage (or ADC counts) into pascals given the hydrophone
sensitivity and preamplifier gain, :func:`spl` turns a pressure waveform into
a sound pressure level, and :func:`power_to_dB` turns a spectral estimate —
already a power — into decibels against the same reference.

Two seams to watch. :func:`pressure_from_recording` returns **pascals**, and the sensitivity
it takes is dB re 1 V/µPa, so the µPa→Pa factor lives inside it and not in the
caller. The two dB doors then default to ``ref=1e-6`` Pa
(:data:`~uacpy.core.constants.REFERENCE_PRESSURE_WATER`, the underwater
convention) and take ``ref`` for the others — quote the same waveform against
1 Pa and the level reads 120 dB lower.

A complex record is two different things, and the numbers cannot say which,
so :func:`spl`, :func:`peak_level` and :func:`sound_exposure_level` refuse one
unless ``complex_record=`` names it:

* ``'analytic'`` — ``x = p + i·H{p}``, the record itself plus its Hilbert
  quadrature. The pressure is the real part; ``|x|²`` is twice ``p²`` and
  would read 3.01 dB high.
* ``'baseband'`` — the complex envelope, ``p(t) = Re{x(t)·e^{iω_c t}}``. Its
  real part depends on the carrier phase, so the level comes from ``|x|``:
  mean square ``|x|²/2`` (``10·log10(mean|x|²/2 / ref²)``) and peak ``|x|``.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from typing import Optional, Tuple

from uacpy.core.exceptions import ConfigurationError, ValidityWarning
from uacpy.core._export import ExportRecord
from uacpy.core._plotting import plotter
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.constants import (NO_ENERGY_DB, PRESSURE_FLOOR,
                                  REFERENCE_PRESSURE_WATER)
from uacpy.core.acoustics.seawater import _sequence_as_array
from uacpy.core.acoustics.weighting import _real_pressure, _weighted_blocks
from uacpy.core._weighting_inputs import warn_in_air_group_in_water
from uacpy.core._validate import (
    normalize_axis, refuse_non_finite_samples, require_positive_finite_scalar)


#: Relative slack on a band edge's comparison with a grid end, which
#: a band edge computed in floating point may miss by rounding.
BAND_EDGE_RTOL = 1e-12

__all__ = [
    'BAND_EDGE_RTOL',
    'pressure_from_recording',
    'spl',
    'power_to_dB',
    'peak_level',
    'sound_exposure_level',
    'cumulative_sound_exposure_level',
    'CumulativeExposure',
    'track_exposure',
    'transmission_loss_dB',
    'received_level_dB',
    'no_energy_mask',
    'sum_levels_dB',
    'integrate_psd',
    'band_level',
]


def pressure_from_recording(
    x: np.ndarray,
    sensitivity: float,
    gain: float,
    volt_params: Optional[Tuple[int, float]] = None,
    x_scale: str = 'counts',
) -> np.ndarray:
    """
    Convert a recorded signal to acoustic pressure in **pascals**.

    ``p = 1e-6 · x / (10**(SH/20) · 10**(G/20))`` — the hydrophone sensitivity
    is quoted against a micropascal, so the division lands in µPa and the
    factor carries it to Pa. Pascals because that is what the rest of uacpy
    reads: every level helper and every plotter defaults to
    ``REFERENCE_PRESSURE_WATER`` (1e-6, one µPa written in Pa), so a chain
    that starts here needs no reference argument anywhere downstream. Handing
    µPa to those defaults reads 120 dB high and nothing in the numbers says
    so.

    Parameters
    ----------
    x : ndarray
        Signal in voltage or bit depth
    sensitivity : float
        Receiving sensitivity SH in dB re 1 V/µPa (hydrophone data sheets
        quote it this way; a typical value is -180). A positive value (more
        than 1 V per µPa) gives a ``ValidityWarning``: it is usually a sign
        slip.
    gain : float
        Preamplifier gain in dB
    volt_params : tuple of (int, float), optional
        If provided, (nbits, v_ref) where nbits is number of bits per sample
        and v_ref is the ADC's full-scale (0-peak) voltage, the voltage of
        count ``2**(nbits-1)``. Used to convert bits to voltage: counts are
        divided by ``2**(nbits-1)``, the ADC convention.
    x_scale : {'counts', 'read_wav'}, optional
        What ``x`` holds when ``volt_params`` is given. ``'counts'``
        (default): ADC counts, the signed integers of a PCM WAV.
        ``'read_wav'``: the output of :func:`uacpy.io.read_wav`, which
        divides those integers by the symmetric full scale
        ``2**(nbits-1) - 1`` (so ±1.0 round-trips through
        :func:`~uacpy.io.write_wav`); they are multiplied back before the
        ADC conversion, so the pressure is exact. Pass ``nbits =
        uacpy.io.read_wav_info(path).bits``. The two full scales differ by
        ``20·log10(2**(b-1) / (2**(b-1) - 1))``, 0.00027 dB at 16 bits.

    Returns
    -------
    ndarray
        Acoustic pressure signal in pascals

    Examples
    --------
    With ``sensitivity=0`` and ``gain=0`` both scale factors are unity, so the
    voltage is carried across by the µPa-to-Pa factor alone:

    >>> x_volt = np.array([0.0, 0.5, -0.5])
    >>> pressure_from_recording(x_volt, sensitivity=0, gain=0)
    array([ 0.e+00,  5.e-07, -5.e-07])

    A bit-depth input is divided by the full-scale count first, so half of
    full scale on a signed 16-bit sample (2**15) against a 1 V reference lands
    on the same 0.5 V:

    >>> x_bits = np.array([0, 16384, -16384])
    >>> pressure_from_recording(x_bits, sensitivity=0, gain=0, volt_params=(16, 1.0))
    array([ 0.e+00,  5.e-07, -5.e-07])

    A 16-bit WAV as :func:`uacpy.io.read_wav` returns it, straight in:

    >>> x_wav = x_bits / (2**15 - 1)
    >>> pressure_from_recording(x_wav, sensitivity=0, gain=0,
    ...                         volt_params=(16, 1.0), x_scale='read_wav')
    array([ 0.e+00,  5.e-07, -5.e-07])
    """
    if x_scale not in ('counts', 'read_wav'):
        raise ConfigurationError(
            f"pressure_from_recording: x_scale={x_scale!r} is not one of "
            f"'counts' (ADC counts) or 'read_wav' (uacpy.io.read_wav's "
            f"output).",
            remediation="Pass x_scale='counts' or x_scale='read_wav'.")
    if x_scale == 'read_wav' and volt_params is None:
        raise ConfigurationError(
            "pressure_from_recording: x_scale='read_wav' converts read_wav's "
            "fractions of full scale back to counts, which needs the bit "
            "depth, and no volt_params was given.",
            remediation="Pass volt_params=(uacpy.io.read_wav_info(path).bits, "
                        "v_ref).")
    x = _sequence_as_array(x)
    if sensitivity > 0:
        # RVS = 20 log10 |M|, M in V/µPa (Butler & Sherman, *Transducers
        # and Arrays for Underwater Sound*, §6.1.1): a positive value is
        # more than 1 V per µPa, a million volts for a 1 Pa wave. A sign
        # slip (+165 for -165) reads the pressure 330 dB low.
        warnings.warn(
            f"pressure_from_recording: sensitivity={sensitivity:g} dB re "
            f"1 V/µPa means {10 ** (sensitivity / 20):.3g} V per µPa; "
            f"hydrophone sensitivities are negative (e.g. -201 dB re 1 "
            f"V/µPa, Istepanian & Stojanovic, *Underwater Acoustic Digital "
            f"Signal Processing and Communication Systems*, §3.2). Check its "
            f"sign: the pressure returned is {2 * sensitivity:g} dB below "
            f"what -{sensitivity:g} gives.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP)
    nu = 10 ** (sensitivity / 20)
    G = 10 ** (gain / 20)

    if volt_params is not None:
        nbits, v_ref = volt_params
        if x_scale == 'read_wav':
            x = x * (2 ** (nbits - 1) - 1)
        x = x * v_ref / (2 ** (nbits - 1))

    # 1e-6: SH is quoted per micropascal, so the division lands in µPa and
    # this carries the result to the pascals everything downstream defaults to.
    return 1e-6 * x / (nu * G)


_COMPLEX_RECORDS = ('analytic', 'baseband')


def _complex_rule(x, complex_record, who):
    """``x`` as an array, with the complex-record reading checked: ``None``
    for a real record, else the named reading (see the module docstring)."""
    x = np.asarray(x)
    if not np.iscomplexobj(x):
        return x, None
    if complex_record not in _COMPLEX_RECORDS:
        raise ConfigurationError(
            f"{who}: a complex record is either an analytic signal or a "
            f"baseband (complex-envelope) one, and they give different "
            f"levels; got complex_record={complex_record!r}.",
            remediation="Pass complex_record='analytic' (the pressure is the "
                        "real part) or complex_record='baseband' (the level "
                        "comes from |x|, mean square |x|²/2).")
    return x, complex_record


def _squared_pressure(x, complex_record, who):
    """``p²`` per sample, under the complex-record reading."""
    x, rule = _complex_rule(x, complex_record, who)
    if rule is None:
        return x ** 2
    if rule == 'analytic':
        return x.real ** 2
    return np.abs(x) ** 2 / 2.0


def _peak_amplitude(x, complex_record, who):
    """``|p|`` per sample whose maximum is the peak pressure."""
    x, rule = _complex_rule(x, complex_record, who)
    if rule == 'analytic':
        return np.abs(x.real)
    # A real record's |p|, or a baseband envelope, whose passband peak over a
    # carrier cycle is |x| itself.
    return np.abs(x)


def spl(x: np.ndarray, ref: float = REFERENCE_PRESSURE_WATER,
        *, axis=None, floor: float = PRESSURE_FLOOR ** 2,
        complex_record: Optional[str] = None):
    """
    Calculate Sound Pressure Level (SPL) of acoustic pressure signal.

    Parameters
    ----------
    x : ndarray
        Acoustic pressure signal in pascals, as :func:`pressure_from_recording` returns. A
        complex record needs ``complex_record=``.
    ref : float, optional
        Reference pressure in the same unit as ``x`` (default:
        ``REFERENCE_PRESSURE_WATER``, one µPa written in Pa). For air in Pa,
        ``20e-6``.
    axis : int or tuple of int, optional
        Axes to average over. ``None`` (the default) averages everything and
        returns a scalar; ``-1`` gives one level per row of a block of
        records, which is what a gridded result needs.
    floor : float, optional
        Lower bound on the ratio of the MEAN-SQUARE pressure to ``ref**2``
        before the log, read the same way :func:`power_to_dB` and
        :func:`peak_level` read it. Default ``PRESSURE_FLOOR**2`` (1e-60):
        a silent record reads -600 dB, the no-energy marker.
    complex_record : {'analytic', 'baseband'}, optional
        Required for a complex record, ignored for a real one: what the
        complex numbers are (see the module docstring). A complex record
        without it is refused, since the two readings give different
        levels.

    Returns
    -------
    float or ndarray
        SPL in dB re reference pressure — a scalar for ``axis=None``, and
        ``x``'s shape without ``axis`` otherwise.

    Raises
    ------
    ConfigurationError
        The record holds a NaN or inf: the first is named by its sample
        (and channel, in a block of records). A level of a record with a
        missing sample is undefined.

    Examples
    --------
    A 100 µPa-rms white signal — 1e-4 Pa — sits at ``20*log10(100) = 40`` dB
    re 1 µPa; the seed makes the sampling scatter around that reproducible.

    >>> rng = np.random.default_rng(0)
    >>> pressure_signal = rng.standard_normal(1000) * 100e-6
    >>> spl_dB = spl(pressure_signal)
    >>> print(f"SPL: {spl_dB:.2f} dB re 1 µPa")
    SPL: 39.81 dB re 1 µPa

    The ratio ``mean(x²) / ref²`` is floored at ``floor`` (default
    ``PRESSURE_FLOOR**2``) before the log, so a silent (all-zero) signal
    returns -600 dB for every ``ref`` instead of ``-inf``: the level view of
    the no-energy marker (:func:`no_energy_mask`), the level
    :func:`power_to_dB`, :func:`peak_level` and :func:`sound_exposure_level`
    give a silent record too.

    >>> float(spl(np.zeros(16)))
    -600.0
    """
    rmsx = np.sqrt(np.mean(_squared_pressure(x, complex_record, 'spl'),
                           axis=axis))
    _refuse_a_non_finite_record(rmsx, x, 'spl', axis)
    return 20 * np.log10(np.maximum(rmsx / ref, np.sqrt(floor)))


def power_to_dB(power, ref: float = REFERENCE_PRESSURE_WATER, *,
                floor: float = PRESSURE_FLOOR ** 2):
    """Mean-square / power-like pressure quantity → level in dB re ``ref``.

    For a *squared* quantity (PSD in Pa²/Hz, SEL in Pa²·s, mean-square
    pressure, an f-k spectrum, …) the level is ``10·log10(power / ref²)``. The
    single conversion every spectral estimator should use: ``power / ref²`` is
    floored at ``floor`` before the log, so a silent (zero) sample yields
    -600 dB, the level view of the no-energy marker that
    :func:`no_energy_mask` finds, instead of ``-inf`` (which would otherwise
    poison a subsequent ``mean`` / ``histogram``).

    Parameters
    ----------
    power : array_like
        Squared-pressure quantity (e.g. PSD, SEL, |p|²); same units as
        ``ref**2``.
    ref : float, optional
        Reference pressure (default: 1 µPa, water). Use
        ``REFERENCE_PRESSURE_AIR`` for air.
    floor : float, optional
        Lower bound applied to the ratio ``power / ref²`` before the log,
        guarding ``log10(0)``. The default ``PRESSURE_FLOOR**2`` (1e-60) puts
        a silent input at ``10*log10(1e-60) = -600`` dB whatever ``ref`` is:
        :data:`~uacpy.core.constants.NO_ENERGY_DB` with a level's sign, the
        same level :func:`spl`, :func:`peak_level` and
        :func:`sound_exposure_level` give a silent record
        (``spl(0) == power_to_dB(0) == -600.0``).

    Returns
    -------
    numpy.ndarray
        Level in dB re ``ref``.
    """
    power = np.asarray(power, dtype=float)
    return 10.0 * np.log10(np.maximum(power / (ref ** 2), floor))


def peak_level(x, ref: float = REFERENCE_PRESSURE_WATER, *, axis=None,
               floor: float = PRESSURE_FLOOR ** 2,
               complex_record: Optional[str] = None):
    """Peak pressure level of a record — ``20·log10(max|x| / ref)``.

    The other half of the dual metric: a level that a *peak* sets, where
    :func:`spl` is set by the rms and :func:`power_to_dB` by an energy.
    Southall et al. (2019) state their impulsive criteria as a pair, peak
    SPL beside an exposure, because a short transient can reach a damaging
    peak while carrying little energy and a long one can do the reverse —
    neither number implies the other. Their non-impulsive criteria are the
    exposure alone: "For impulsive exposures, both criteria are defined
    for all marine mammal groups. However, for non-impulsive exposures,
    only frequency-weighted SEL criteria are given here".

    Parameters
    ----------
    x : array_like
        Pressure record (Pa). A complex record needs ``complex_record=``:
        an analytic one peaks at ``max|Re x|``, a baseband one at
        ``max|x|``.
    ref : float, optional
        Reference pressure (default 1 µPa in water).
    axis : int or tuple of int, optional
        Axes to take the maximum over. ``None`` reduces everything to a
        scalar; ``-1`` gives one level per record of a block.
    floor : float, optional
        Lower bound on the ratio of the SQUARED peak to ``ref**2`` before the
        log, read the same way :func:`power_to_dB` reads it, so a silent
        record returns -600 dB (the no-energy marker) rather than ``-inf`` —
        which would poison any mean taken over a map of them.
    complex_record : {'analytic', 'baseband'}, optional
        Required for a complex record, ignored for a real one; as on
        :func:`spl`.

    Returns
    -------
    float or ndarray
        dB re ``ref``.

    Raises
    ------
    ConfigurationError
        The record holds a NaN or inf: the first is named by its sample
        (and channel, in a block of records). A level of a record with a
        missing sample is undefined.
    """
    peak = np.max(_peak_amplitude(x, complex_record, 'peak_level'),
                  axis=axis)
    _refuse_a_non_finite_record(peak, x, 'peak_level', axis)
    return 20 * np.log10(np.maximum(peak / ref, np.sqrt(floor)))


def sound_exposure_level(pressure, sample_rate: float,
                         ref: float = REFERENCE_PRESSURE_WATER, *,
                         axis: int = -1, floor: float = PRESSURE_FLOOR ** 2,
                         complex_record: Optional[str] = None,
                         weighting=None):
    """Sound exposure level of a pressure record —
    ``10·log10( Σ p² / sample_rate / ref² )``, in dB re ``ref²``·s.

    The energy flux density of the record (Abraham, *Underwater Acoustic
    Signal Processing*, sect. 3.2.1.5; ISO 18405). Unlike an rms level it
    does not divide by the duration, so it is the quantity that accumulates
    over a transient and the one exposure criteria are written in: doubling
    the duration of a steady signal adds 3 dB here and nothing to
    :func:`spl`.

    For the same energy split into standard bands and left linear (Pa²·s),
    see :func:`~uacpy.acoustic_signal.sound_exposure`. This one is the
    total, in dB. Its build-up over a long record is
    :func:`cumulative_sound_exposure_level`, and the exposure several events
    deliver together is the energy sum of theirs, :func:`sum_levels_dB`
    (N identical events add ``10·log10(N)``).

    ``weighting=`` gives the frequency-weighted SEL: the integral of the
    squared frequency-weighted pressure
    (:func:`~uacpy.acoustics.weighted_pressure`), the metric of Southall et
    al. (2019). It runs block by block and never holds the weighted record.
    The integral includes the little the weighting spreads past the record's
    ends, so an event's weighted SEL does not depend on how much silence
    surrounds it.

    Parameters
    ----------
    pressure : array_like
        Pressure record (Pa). A complex record needs ``complex_record=``
        (an analytic one contributes its real part, a baseband one
        ``|x|²/2``).
    sample_rate : float
        Sample rate (Hz), as everywhere in the package —
        :meth:`Field.sound_exposure_level`, the estimators, the generators.
        A value below 1 Hz is refused as a sample interval passed by
        mistake: ``1/fs`` read as a rate puts the level ``20·log10(fs)``
        high (93.6 dB at 48 kHz).
    ref : float, optional
        Reference pressure (default 1 µPa in water).
    axis : int, default -1
        Time axis. Every other axis is carried through, so a
        ``(depth, range, time)`` block returns a map.
    floor : float, optional
        As :func:`power_to_dB`.
    complex_record : {'analytic', 'baseband'}, optional
        Required for a complex record, ignored for a real one; as on
        :func:`spl`. A weighted level refuses a baseband record, whose
        frequencies are offsets from a carrier it does not carry.
    weighting : str, callable or (array_like, array_like), optional
        Frequency weighting, unweighted by default: a hearing group of
        :data:`~uacpy.noise.HEARING_GROUPS` (Southall et al. 2019), your own
        function ``W(frequency_Hz) -> dB``, or a ``(frequencies_Hz, W_dB)``
        table (see :func:`~uacpy.acoustics.weighted_pressure`).

    Returns
    -------
    float or ndarray
        dB re ``ref²``·s.

    Raises
    ------
    ConfigurationError
        The record holds a NaN or inf: the first is named by its sample
        (and channel, in a block of records). A level of a record with a
        missing sample is undefined.
    """
    who = 'sound_exposure_level'
    fs = _exposure_rate(sample_rate, who)
    warn_in_air_group_in_water(weighting, ref, who)
    if weighting is None:
        p2 = _squared_pressure(pressure, complex_record, who)
        energy = np.sum(p2, axis=axis)
        _refuse_a_non_finite_record(energy, pressure, who, axis)
        return power_to_dB(energy / fs, ref, floor=floor)
    x = np.moveaxis(_real_pressure(np.asarray(pressure), complex_record, who),
                    axis, -1)
    rows = x.reshape(-1, x.shape[-1])
    energy = np.zeros(rows.shape[0])
    for channels, _start, y in _weighted_blocks(rows, fs, weighting, who=who):
        energy[channels] += np.sum(y ** 2, axis=-1)
    return power_to_dB(energy.reshape(x.shape[:-1]) / fs, ref, floor=floor)


def _refuse_a_non_finite_record(level, pressure, who, axis):
    """Refuse ``pressure`` at its first NaN or inf once ``level``, the
    quantity reduced from it, came out non-finite
    (:func:`~uacpy.core._validate.refuse_non_finite_samples`).

    The time axis is ``axis``; ``None`` or a tuple reads the record
    flattened. The record is scanned only on that path, a block of
    :data:`_UNWEIGHTED_BLOCK` samples at a time, so a finite record pays
    nothing and a ``np.memmap`` is never loaded whole. A level that
    overflowed from finite samples finds none and passes."""
    if np.all(np.isfinite(level)):
        return
    x = np.asarray(pressure)
    if axis is None or isinstance(axis, tuple) or x.ndim == 0:
        rows = x.reshape(1, -1)
        label = 'sample' if x.ndim <= 1 else 'flat index'
    else:
        x = np.moveaxis(x, axis, -1)
        rows = x.reshape(-1, x.shape[-1])
        label = 'sample'
    for start in range(0, rows.shape[-1], _UNWEIGHTED_BLOCK):
        refuse_non_finite_samples(rows[:, start:start + _UNWEIGHTED_BLOCK],
                                  who, start=start, n_channels=rows.shape[0],
                                  label=label)


def _exposure_rate(sample_rate, who):
    """``sample_rate`` as a float rate in Hz, refusing a sample interval
    passed by mistake."""
    fs = require_positive_finite_scalar(sample_rate, who, "sample_rate",
                                        " Hz")
    if fs < 1.0:
        raise ConfigurationError(
            f"{who}: sample_rate={fs:g} Hz reads as a sample "
            f"INTERVAL; the argument is the rate.",
            remediation=f"Pass sample_rate={1.0 / fs:g} (1/dt).")
    return fs


def cumulative_sound_exposure_level(pressure, sample_rate: float,
                                    ref: float = REFERENCE_PRESSURE_WATER, *,
                                    step: float = 1.0,
                                    reset: Optional[float] = None,
                                    weighting=None, axis: int = -1,
                                    floor: float = PRESSURE_FLOOR ** 2,
                                    complex_record: Optional[str] = None):
    """The sound exposure level accumulated over a record, at the end of
    every ``step`` seconds.

    ``SEL(t_k) = 10·log10( ∫₀^{t_k} p(t)² dt / ref² )``. The curve rises as
    energy arrives, and its last value is
    :func:`sound_exposure_level` of the whole record. It answers when an
    exposure criterion is crossed, not only whether.

    ``reset`` restarts the sum every ``reset`` seconds: an accumulation period,
    such as the 24 h after which Southall et al. (2007) proposed to "reset"
    the accumulation. Southall et al. (2019) keep that period as a
    precautionary choice and note that a shorter one may suit some cases.
    The period is therefore yours to set, and unset means no reset.

    The record is read one block at a time, weighted or not, so a
    ``np.memmap`` of a multi-hour record is never loaded whole. Only the
    curve, one value per step and channel, is held. A weighting's spread
    past the record's ends counts in the first and last step, so the last
    value is the weighted :func:`sound_exposure_level` exactly.

    Parameters
    ----------
    pressure : array_like
        Pressure record (Pa), time along ``axis``; a ``np.memmap`` streams.
    sample_rate : float
        Sample rate (Hz).
    ref : float, optional
        Reference pressure (default 1 µPa in water).
    step : float, default 1.0
        Seconds between curve values; the last step may be partial.
    reset : float, optional
        Accumulation period in seconds, a whole number of ``step``.
    weighting : str, callable or (array_like, array_like), optional
        As on :func:`sound_exposure_level`.
    axis : int, default -1
        Time axis; the curve replaces it, one value per step.
    floor : float, optional
        As :func:`power_to_dB`.
    complex_record : {'analytic', 'baseband'}, optional
        As on :func:`sound_exposure_level`.

    Returns
    -------
    CumulativeExposure
        ``(times, levels)`` — it unpacks as the two arrays: the end of each
        step (s from the first sample) and the accumulated SEL at each time,
        dB re ``ref²``·s, the curve on ``axis``; ``.plot()`` draws it.

    Raises
    ------
    ConfigurationError
        The record holds a NaN or inf: the first is named by its sample
        (and channel, in a block of records). A level of a record with a
        missing sample is undefined.

    Examples
    --------
    One second of a 1 Pa rms tone, accumulated over 0.25 s steps: each
    quarter second adds the same energy, so the curve rises by 3.01 dB,
    then 1.76 dB, then 1.25 dB, and ends at the whole record's 120 dB:

    >>> import numpy as np
    >>> p = np.sqrt(2) * np.sin(2 * np.pi * 100.0 * np.arange(8000) / 8000)
    >>> t, sel = cumulative_sound_exposure_level(p, 8000.0, step=0.25)
    >>> [round(float(v), 2) for v in sel]
    [113.98, 116.99, 118.75, 120.0]
    """
    who = 'cumulative_sound_exposure_level'
    fs = _exposure_rate(sample_rate, who)
    warn_in_air_group_in_water(weighting, ref, who)
    step = require_positive_finite_scalar(step, who, "step", " s")
    x = np.moveaxis(np.asarray(pressure), axis, -1)
    n = x.shape[-1]
    if n == 0:
        raise ConfigurationError(f"{who}: the record is empty.")
    per_step = step * fs
    if per_step < 1.0:
        raise ConfigurationError(
            f"{who}: step={step:g} s is shorter than one sample at "
            f"{fs:g} Hz.")
    n_steps = int((n - 1) // per_step) + 1
    if weighting is None:
        rows = x.reshape(-1, n)
        blocks = ((slice(None), start, _squared_pressure(
            rows[:, start:start + _UNWEIGHTED_BLOCK], complex_record, who))
                  for start in range(0, n, _UNWEIGHTED_BLOCK))
    else:
        rows = _real_pressure(x, complex_record, who).reshape(-1, n)
        blocks = ((channels, start, y ** 2) for channels, start, y
                  in _weighted_blocks(rows, fs, weighting, who=who))
    energy = np.zeros((rows.shape[0], n_steps))
    for channels, start, p2 in blocks:
        k = np.clip((start + np.arange(p2.shape[-1])) // per_step, 0,
                    n_steps - 1).astype(int)
        first = np.flatnonzero(np.diff(k, prepend=-1))
        energy[channels, k[first]] += np.add.reduceat(p2, first, axis=-1)
    _refuse_a_non_finite_record(energy, x, who, -1)
    running = np.cumsum(energy.reshape(x.shape[:-1] + (n_steps,)) / fs,
                        axis=-1)
    if reset is not None:
        reset = require_positive_finite_scalar(reset, who, "reset", " s")
        per_period = round(reset / step)
        if per_period < 1 or abs(reset / step - per_period) > 1e-9 * per_period:
            raise ConfigurationError(
                f"{who}: reset={reset:g} s must be a whole number of "
                f"steps (step={step:g} s), so each period ends on a curve "
                f"value.")
        k = np.arange(n_steps)
        before = np.concatenate((np.zeros(running.shape[:-1] + (1,)),
                                 running[..., :-1]), axis=-1)
        running = running - before[..., (k // per_period) * per_period]
    times = np.minimum((np.arange(n_steps) + 1) * step, n / fs)
    return CumulativeExposure(
        times=times,
        levels=np.moveaxis(power_to_dB(running, ref, floor=floor), -1, axis),
        ref=float(ref), axis=int(axis))


@dataclass(frozen=True, eq=False)
class CumulativeExposure(ExportRecord):
    """The sound exposure level accumulated over a record, what
    :func:`cumulative_sound_exposure_level` returns: ``levels`` (dB re
    ``ref²``·s) at each of ``times`` (s), the curve on ``axis`` of
    ``levels``. ``times, levels = ...`` unpacks it; ``.plot()`` draws it
    with :func:`uacpy.plot.plot_cumulative_exposure`.
    """

    times: np.ndarray
    levels: np.ndarray
    ref: float
    axis: int = -1

    _ARRAY_FIELDS = ('times', 'levels')
    _REPR_UNITS = {'levels': 'dB'}

    def __iter__(self):
        return iter((self.times, self.levels))

    def _payload(self):
        dims = ['channel_dim%d' % i for i in range(self.levels.ndim)]
        dims[self.axis] = 'time'
        return {'levels': (self.levels, tuple(dims), 'dB')}

    def _coords(self):
        return {'time': (self.times, 's')}

    def plot(self, **kwargs):
        """Draw the running SEL with :func:`uacpy.plot.plot_cumulative_exposure`;
        ``kwargs`` reach it. Returns ``(fig, ax)``."""
        return plotter('plot_cumulative_exposure')(self, **kwargs)


#: Samples per channel the unweighted cumulative sum reads at a time.
_UNWEIGHTED_BLOCK = 2 ** 20


def track_exposure(sel_dB, ranges, positions, *, closest_approach,
                   along=None, axis: int = -1, event_duration=None):
    """Cumulative SEL of a source moving along a straight track: the energy
    sum, over every event on the track, of one event's SEL at that event's
    distance.

    ``sel_dB`` is the SEL one event delivers as a function of horizontal
    range — a model's single-shot exposure map, for instance. Events sit on
    a straight line at along-track positions ``x_i``. A receiver at
    along-track coordinate ``a`` and distance ``d`` from the line (its
    closest approach) is ``r_i = √(d² + (x_i − a)²)`` from event ``i``,
    so::

        SEL_cum(a, d) = 10·log10 Σ_i 10^(SEL(r_i)/10)

    with ``SEL(r_i)`` interpolated linearly in range. This assumes the
    environment is the same along the track (range-independent, or one
    bearing standing for all).

    This is the computation impact studies of survey lines make. Etter
    (*Underwater Acoustic Modeling and Simulation*, sect. 6.16.3) summarises
    Breitzke and Bohlen (2010), who model "cumulative sound-exposure levels
    of multiple shots fired along a seismic line" with marine mammals as
    "static receivers", and Lurton (2016), who integrates the received
    intensity of an echo sounder moving along a survey line and reports the
    ranges at which impact thresholds are reached. Breitzke and Bohlen's
    two scalings follow from the sum. A threshold's distance from the line
    grows as ``10·log10 r`` (cylindrical) for many shots, against
    ``20·log10 r`` for one. Doubling the shot interval costs 3 dB.

    Pulses fired every ``Δt`` from a platform at speed ``v`` sit at
    ``positions = np.arange(-L, L, v * Δt)``. A continuous source (a vessel,
    a vibrator) is the same sum with one event per ``Δt``, positions ``v·Δt``
    apart, whose SEL is the received level held over it plus
    ``10·log10(Δt / 1 s)`` — the energy of a constant-power interval
    (Ainslie, *Principles of Sonar Performance Modelling*, sect. 3.3,
    Eq. 3.155): pass that level map as ``sel_dB`` with
    ``event_duration=Δt``. ``Δt`` short against the passage keeps each
    event's range nearly constant over its interval. Several distinct events
    at one place are :func:`sum_levels_dB`.

    Parameters
    ----------
    sel_dB : array_like
        Single-event SEL (dB re 1 µPa²·s) along ``axis``, one value per
        range; any other axes (depth, …) are carried through. With
        ``event_duration``, the received level (dB re 1 µPa) the source
        holds over each event instead.
    ranges : array_like
        The strictly increasing horizontal ranges (m) of ``sel_dB``.
    positions : array_like
        Along-track position of each event (m), measured from the point of
        closest approach.
    closest_approach : float or array_like
        Horizontal distance(s) (m) of the receiver from the track.
    along : float or array_like, optional
        Along-track coordinate(s) (m) of the receiver, in the frame of
        ``positions``; 0 by default, so ``positions`` are measured from the
        receiver's point of closest approach. An array gives the plan view
        ``SEL_cum(along, closest_approach)``: a receiver abeam of the
        track's end gets less than one abeam of its middle.
    axis : int, default -1
        The range axis of ``sel_dB``. It is replaced by an ``along`` axis
        (when ``along`` is an array) followed by a ``closest_approach`` axis
        (when that is an array); a scalar adds no axis.
    event_duration : float, optional
        Seconds each event lasts, for a continuous source: ``sel_dB`` is then
        a received level, and each event's SEL is that level plus
        ``10·log10(event_duration / 1 s)``. ``None`` (default) reads
        ``sel_dB`` as single-event SELs.

    Returns
    -------
    float or ndarray
        Cumulative SEL, dB re 1 µPa²·s.

    Raises
    ------
    ConfigurationError
        An event lies outside ``[ranges[0], ranges[-1]]`` of a receiver. Its
        exposure is unknown, and dropping it would understate the sum.

    Examples
    --------
    Spherical spreading from a 200 dB re 1 µPa²·s·m² event, one every 10 m
    along a 20 km track, receiver 100 m off it. The sum approaches the
    closed form ``200 + 10·log10(π / (Δx·d))`` = 174.97 dB:

    >>> import numpy as np
    >>> r = np.geomspace(1.0, 20e3, 4000)
    >>> x = np.arange(-10e3, 10e3 + 1, 10.0)
    >>> round(float(track_exposure(200 - 20 * np.log10(r), r, x,
    ...                            closest_approach=100.0)), 2)
    174.94
    """
    who = 'track_exposure'
    sel = np.moveaxis(np.asarray(sel_dB, dtype=float), axis, -1)
    r = np.asarray(ranges, dtype=float)
    if r.ndim != 1 or r.size < 2 or r.size != sel.shape[-1]:
        raise ConfigurationError(
            f"{who}: ranges must be 1-D with one value per sel_dB sample "
            f"along axis={axis} (at least two); got {r.size} range(s) for "
            f"{sel.shape[-1]} sample(s).")
    if not np.all(np.diff(r) > 0):
        raise ConfigurationError(f"{who}: ranges must strictly increase.")
    x = np.atleast_1d(np.asarray(positions, dtype=float))
    d = np.asarray(closest_approach, dtype=float)
    a = np.asarray(0.0 if along is None else along, dtype=float)
    if (x.ndim != 1 or d.ndim > 1 or a.ndim > 1
            or not all(np.all(np.isfinite(v)) for v in (x, d, a))):
        raise ConfigurationError(
            f"{who}: positions is a 1-D array, and closest_approach and "
            f"along are scalars or 1-D arrays, all finite.")
    if np.any(d < 0):
        raise ConfigurationError(
            f"{who}: closest_approach is a distance, >= 0.")
    if event_duration is not None:
        held = require_positive_finite_scalar(event_duration, who,
                                              "event_duration", " s")
        # SEL of a constant level held over the interval (Ainslie Eq. 3.155).
        sel = sel + 10.0 * np.log10(held)
    # One receiver row of the plan view at a time: the whole
    # (..., along, closest_approach, event) block can outgrow memory.
    rows = []
    for a_k in np.atleast_1d(a):
        dist = np.hypot(np.atleast_1d(d)[:, None], x[None, :] - a_k)
        if dist.min() < r[0] * (1 - BAND_EDGE_RTOL):
            raise ConfigurationError(
                f"{who}: an event passes {dist.min():g} m from a receiver, "
                f"closer than the {r[0]:g} m the exposure starts at.",
                remediation=f"Start closest_approach at {r[0]:g} m or more, "
                            f"or give the exposure from a shorter range.")
        if dist.max() > r[-1] * (1 + BAND_EDGE_RTOL):
            raise ConfigurationError(
                f"{who}: an event lies {dist.max():g} m from a receiver, "
                f"past the {r[-1]:g} m the exposure reaches; its share of "
                f"the sum is unknown, and dropping it would understate it.",
                remediation="Give the exposure out to the farthest event "
                            "(run the model further), or shorten the track "
                            "or the along-track span.")
        j = np.clip(np.searchsorted(r, dist, side='right') - 1, 0,
                    r.size - 2)
        t = np.clip((dist - r[j]) / (r[j + 1] - r[j]), 0.0, 1.0)
        levels = sel[..., j] * (1.0 - t) + sel[..., j + 1] * t
        rows.append(np.asarray(sum_levels_dB(levels, axis=-1)))
    total = np.stack(rows, axis=-2)                  # (..., along, approach)
    if d.ndim == 0:
        total = total[..., 0]
    if a.ndim == 0:
        total = total[..., 0, :] if d.ndim else total[..., 0]
    if total.ndim == 0:
        return float(total)
    added = a.ndim + d.ndim
    position = axis % sel.ndim
    return np.moveaxis(total, list(range(total.ndim - added, total.ndim)),
                       list(range(position, position + added)))


def transmission_loss_dB(pressure, *, floor: float = PRESSURE_FLOOR):
    """Complex pressure → transmission loss in dB — ``-20·log10(|p|)``.

    The canonical conversion every uacpy result uses for a TL view
    (:attr:`~uacpy.Field.dB`, :meth:`~uacpy.Field.to_dB`, the metrics in
    :mod:`uacpy.core.metrics`, and the RAM and ray plotters), exposed so a
    complex field from anywhere else converts the same way.

    Note the **minus**: this is a loss, so a quiet cell is a LARGE number,
    the opposite sign to :func:`spl` and :func:`power_to_dB`. It also takes
    the pressure itself, not its square.

    ``|p|`` is clamped at :data:`~uacpy.core.constants.PRESSURE_FLOOR`
    first, which caps an exactly-zero sample — a cell no energy reached —
    at 600 dB rather than ``+inf``, keeping the array finite for plotting
    and reductions. Shape is preserved; nothing is squeezed.

    Parameters
    ----------
    pressure : array_like
        Complex (or real) pressure, in whatever unit the 1 m reference is
        expressed in. There is no ``ref``: a loss is a ratio to the 1 m
        reference the pressure already carries, so there is nothing to
        divide by.
    floor : float, optional
        Lower bound on ``|p|`` ITSELF before the log — not on the squared
        ratio ``|p|²/ref²`` as the ``floor`` of :func:`spl`,
        :func:`peak_level` and :func:`power_to_dB` is — so the default
        :data:`PRESSURE_FLOOR` caps a loss at 600 dB (:data:`NO_ENERGY_DB`,
        the no-energy marker), the loss view of the -600 dB the level
        helpers give a silent record. Pass ``0.0`` for the unclamped
        ``+inf``.

    Returns
    -------
    ndarray
        Loss in dB, same shape as ``pressure``.
    """
    return -20.0 * np.log10(np.maximum(np.abs(pressure), floor))


def sum_levels_dB(*levels_dB, axis=None):
    """Incoherent (power) sum of dB levels — ``10·log10(Σ 10^(Lᵢ/10))``.

    The sum of uncorrelated contributions to one level: ambient-noise
    components, reverberation from several scatterers, the paths of an
    incoherent TL. Evaluated through ``np.logaddexp`` so a very loud term
    cannot overflow ``10**(L/10)`` (float64 overflows past ~3080 dB), and a
    ``-inf`` level (a switched-off source) contributes nothing. Every
    argument must carry the same reference; the result carries it too.

    The levels are the separate arguments, ``sum_levels_dB(L1, L2, L3)``:
    one array argument is one level (each element its own cell), so
    ``sum_levels_dB([100, 100, 100])`` returns the three levels unchanged.
    To sum the entries of one array, name the axis they run along:
    ``sum_levels_dB([100, 100, 100], axis=0)`` is 104.77 dB, or pass
    ``*levels``.

    Parameters
    ----------
    *levels_dB : float or array_like
        One or more levels, broadcast together.
    axis : int, optional
        With exactly one argument, sum its levels along this axis.

    Returns
    -------
    float or ndarray
        The summed level, in the broadcast shape; a float for scalars.

    Examples
    --------
    Two equal levels add 3.01 dB:

    >>> round(float(sum_levels_dB(60.0, 60.0)), 2)
    63.01
    >>> round(float(sum_levels_dB([100.0, 100.0, 100.0], axis=0)), 2)
    104.77
    """
    if not levels_dB:
        raise ConfigurationError("sum_levels_dB: need at least one level.")
    if axis is not None:
        if len(levels_dB) != 1:
            raise ConfigurationError(
                f"sum_levels_dB: axis= sums the levels of ONE array along an "
                f"axis; got {len(levels_dB)} arguments.",
                remediation="Pass the levels stacked in one array with "
                            "axis=, or as separate arguments without it.")
        levels = np.asarray(levels_dB[0], dtype=float)
        try:
            stack = np.moveaxis(levels, axis, 0)
        except (np.exceptions.AxisError, TypeError) as exc:
            raise ConfigurationError(
                f"sum_levels_dB: axis={axis!r} is not an axis of levels "
                f"shaped {levels.shape}.") from exc
    else:
        arrs = [np.asarray(x, dtype=float) for x in levels_dB]
        try:
            arrs = np.broadcast_arrays(*arrs)
        except ValueError:
            raise ConfigurationError(
                f"sum_levels_dB: the levels do not broadcast together; got "
                f"shapes {[a.shape for a in arrs]}.",
                remediation="Pass levels of one shape, or scalars with "
                            "arrays.") from None
        stack = np.stack(arrs, axis=0)
    ln10 = np.log(10.0)
    # A NaN level is a cell with no value; it stays NaN, and numpy's reduce
    # would warn about it on every call that carries one.
    with np.errstate(invalid='ignore'):
        total = (10.0 / ln10) * np.logaddexp.reduce(stack * (ln10 / 10.0),
                                                     axis=0)
    # The sum of no-energy markers is the marker: two -600 dB cells sum to
    # -596.99, which no_energy_mask would read as a level.
    total = np.where(np.all(no_energy_mask(stack), axis=0)
                     & ~no_energy_mask(total), -NO_ENERGY_DB, total)
    return float(total) if np.ndim(total) == 0 else total


def integrate_psd(psd, frequencies, freq_min=None, freq_max=None, *,
                  axis: int = -1):
    """The power a one-sided power spectral density carries in the band
    ``[freq_min, freq_max]`` Hz: ``∫ psd df``, in ``psd`` units × Hz (Pa²/Hz →
    Pa²).

    The band edges are spliced into the grid points inside the band and
    ``psd`` is interpolated linearly onto them, so the edge intervals carry
    their true width; the trapezoid rule then runs over those nodes. An unset
    edge is the grid's own end. ``frequencies`` is strictly increasing and
    the band lies inside it: a band reaching past the grid would integrate
    only the covered part, which is not that band's power, so it is refused.
    ``psd`` is non-negative.

    Parameters
    ----------
    psd : array_like
        One-sided power spectral density, non-negative: one spectrum, or
        one per cell of the other axes (a spectrogram, a PSD per depth).
    frequencies : array_like
        Strictly increasing frequencies (Hz), one per sample of ``psd``
        along ``axis``.
    freq_min, freq_max : float, optional
        The band (Hz); an unset edge is the grid's own end.
    axis : int, default -1
        The frequency axis of ``psd``.

    Returns
    -------
    float or ndarray
        The band power: a float for one spectrum, ``psd``'s shape without
        ``axis`` otherwise.
    """
    psd = np.asarray(psd, dtype=float)
    f = np.asarray(frequencies, dtype=float)
    if psd.ndim == 0 or f.ndim != 1:
        raise ConfigurationError(
            f"integrate_psd: psd holds spectra along axis= and frequencies "
            f"is one 1-D axis; got shapes {psd.shape} and {f.shape}.")
    axis = normalize_axis(psd, axis, "integrate_psd")
    if psd.shape[axis] != f.size:
        raise ConfigurationError(
            f"integrate_psd: psd has {psd.shape[axis]} samples along "
            f"axis={axis} (shape {psd.shape}) but frequencies has {f.size}; "
            f"one value of psd per frequency along axis=.")
    if f.size < 2 or np.any(np.diff(f) <= 0):
        raise ConfigurationError(
            "integrate_psd: frequencies must be at least two strictly "
            "increasing samples to span a band.")
    if np.any(psd < 0):
        raise ConfigurationError(
            f"integrate_psd: psd holds {int(np.count_nonzero(psd < 0))} "
            f"negative value(s); a power spectral density is non-negative.")
    lo = f[0] if freq_min is None else float(freq_min)
    hi = f[-1] if freq_max is None else float(freq_max)
    if not lo < hi:
        raise ConfigurationError(
            f"integrate_psd: need freq_min < freq_max; got [{lo:g}, {hi:g}] Hz.")
    if (lo < f[0] * (1.0 - BAND_EDGE_RTOL)
            or hi > f[-1] * (1.0 + BAND_EDGE_RTOL)):
        raise ConfigurationError(
            f"integrate_psd: the band [{lo:g}, {hi:g}] Hz reaches past the "
            f"spectrum's [{f[0]:g}, {f[-1]:g}] Hz; the covered part alone is "
            f"not the band's power.",
            remediation="Pass a spectrum that spans the band, or a band "
                        "inside it.")
    interior = f[(f > lo) & (f < hi)]
    nodes = np.unique(np.concatenate(([lo], interior, [hi])))
    # Linear interpolation of every spectrum onto the nodes, along axis.
    right = np.clip(np.searchsorted(f, nodes, side='right'), 1, f.size - 1)
    left = right - 1
    weight = np.clip((nodes - f[left]) / (f[right] - f[left]), 0.0, 1.0)
    x = np.moveaxis(psd, axis, -1)
    on_nodes = x[..., left] * (1.0 - weight) + x[..., right] * weight
    power = np.trapezoid(on_nodes, nodes, axis=-1)
    return float(power) if power.ndim == 0 else power


def band_level(level_density_dB, frequencies, freq_min=None,
               freq_max=None, *, axis: int = -1):
    """Band level (dB re ref²) of a level density (dB re ref²/Hz) over
    ``[freq_min, freq_max]`` Hz: ``10·log10 ∫ 10^(L/10) df``, integrated by
    :func:`integrate_psd`.

    A band that carries no power (every level ``-inf``) is ``-inf``, the
    level of an empty band throughout uacpy.

    Parameters
    ----------
    level_density_dB : array_like
        Level density (dB re ref²/Hz): one spectrum, or one per cell of the
        other axes.
    frequencies : array_like
        Strictly increasing frequencies (Hz), one per sample along ``axis``.
    freq_min, freq_max : float, optional
        The band (Hz); an unset edge is the grid's own end.
    axis : int, default -1
        The frequency axis of ``level_density_dB``.

    Returns
    -------
    float or ndarray
        The band level: a float for one spectrum, the input's shape without
        ``axis`` otherwise.
    """
    power = integrate_psd(
        10.0 ** (np.asarray(level_density_dB, dtype=float) / 10.0),
        frequencies, freq_min, freq_max, axis=axis)
    with np.errstate(divide='ignore'):
        level = 10.0 * np.log10(power)
    return float(level) if np.ndim(level) == 0 else level


def received_level_dB(source_level_dB, tl_dB):
    """Received level ``SL - TL`` (dB re 1 µPa), keeping the no-energy
    marker.

    The propagation term of the sonar equation: a transmission loss is
    referenced to a unit source at 1 m, and subtracting it from the level
    the source is driven at gives what a hydrophone there reads. A cell no
    energy reached carries the loss marker (``|TL| >= NO_ENERGY_DB``, see
    :func:`no_energy_mask`); ``SL - 600`` would be a finite level hundreds
    of dB down that no mask recognises, so such a cell is returned as
    ``-NO_ENERGY_DB``, the level view of the same marker.

    Parameters
    ----------
    source_level_dB : float or array_like
        Source level, dB re 1 µPa at 1 m.
    tl_dB : float or array_like
        Transmission loss, dB re 1 m; broadcast against the source level.

    Returns
    -------
    float or ndarray
        The received level, in the broadcast shape; a float for scalars.

    Examples
    --------
    >>> float(received_level_dB(180.0, 60.0))
    120.0
    >>> float(received_level_dB(180.0, 600.0))
    -600.0
    """
    sl = np.asarray(source_level_dB, dtype=float)
    tl = np.asarray(tl_dB, dtype=float)
    level = np.where(no_energy_mask(tl), -NO_ENERGY_DB, sl - tl)
    return float(level) if level.ndim == 0 else level


def no_energy_mask(levels_dB):
    """``True`` where a dB level is the no-energy marker, not a level.

    uacpy writes a cell no energy reached as an exact zero and lets
    :data:`~uacpy.core.constants.PRESSURE_FLOOR` speak for it, so its loss is
    600 dB (:data:`~uacpy.core.constants.NO_ENERGY_DB`) and its level -600
    dB; a silent record's level is -600 dB too, through :func:`spl`,
    :func:`peak_level`, :func:`sound_exposure_level` and
    :func:`power_to_dB`. That number is finite, so a reduction that masks only NaN/inf would
    average it in as a real 600 dB loss — one such cell in 50 turns a 0.8 dB
    model agreement into 75 dB. Every metric and plotter excludes cells with
    this one predicate.

    The test is on MAGNITUDE, ``|level| >= NO_ENERGY_DB``, which covers a
    loss view (+600) and a level view (-600) of the same cell. A genuine deep
    null — a real 76 dB or even a round-off 325 dB — is kept: only the marker
    reaches 600.

    Anything that turns a loss into another level has to carry the marker
    across, or this predicate stops seeing it: ``SL - TL`` puts a
    no-energy cell at ``SL - 600``. :func:`received_level_dB` does, and
    :meth:`~uacpy.Field.at_source_level` goes through it; the sonar budgets
    (:func:`uacpy.sonar.echo_level`, :func:`uacpy.sonar.passive_signal_excess`,
    :func:`uacpy.sonar.active_signal_excess` and the two signal-excess
    ``*_field`` forms) mask on the one-way TL.

    Parameters
    ----------
    levels_dB : array_like
        Levels or losses in dB.

    Returns
    -------
    ndarray of bool
        Same shape as ``levels_dB``; NaN is not the marker (it is no data).
    """
    x = np.asarray(levels_dB, dtype=float)
    with np.errstate(invalid='ignore'):
        return np.abs(x) >= NO_ENERGY_DB
