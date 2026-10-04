"""From volts to pascals to decibels.

The three steps a recording takes to become a level: :func:`pressure` turns
the recorded voltage (or ADC counts) into pascals given the hydrophone
sensitivity and preamplifier gain, :func:`spl` turns a pressure waveform into
a sound pressure level, and :func:`power_to_dB` turns a spectral estimate —
already a power — into decibels against the same reference.

Two seams to watch. :func:`pressure` returns **pascals**, and the sensitivity
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

import numpy as np
from typing import Optional, Tuple

from uacpy.core.exceptions import ConfigurationError
from uacpy.core.constants import (NO_ENERGY_DB, PRESSURE_FLOOR,
                                  REFERENCE_PRESSURE_WATER)
from uacpy.core.acoustics.seawater import _sequence_as_array


#: Relative slack on a band edge's comparison with a grid end, which
#: a band edge computed in floating point may miss by rounding.
BAND_EDGE_RTOL = 1e-12

__all__ = [
    'BAND_EDGE_RTOL',
    'pressure',
    'spl',
    'power_to_dB',
    'peak_level',
    'sound_exposure_level',
    'transmission_loss_dB',
    'received_level_dB',
    'no_energy_mask',
    'sum_levels_dB',
    'integrate_psd',
    'band_level',
]


def pressure(
    x: np.ndarray,
    sensitivity: float,
    gain: float,
    volt_params: Optional[Tuple[int, float]] = None,
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
        quote it this way; a typical value is -180)
    gain : float
        Preamplifier gain in dB
    volt_params : tuple of (int, float), optional
        If provided, (nbits, v_ref) where nbits is number of bits per sample
        and v_ref is reference voltage. Used to convert bits to voltage —
        a WAV of signed integers goes straight through.

    Returns
    -------
    ndarray
        Acoustic pressure signal in pascals

    Examples
    --------
    With ``sensitivity=0`` and ``gain=0`` both scale factors are unity, so the
    voltage is carried across by the µPa-to-Pa factor alone:

    >>> x_volt = np.array([0.0, 0.5, -0.5])
    >>> pressure(x_volt, sensitivity=0, gain=0)
    array([ 0.e+00,  5.e-07, -5.e-07])

    A bit-depth input is divided by the full-scale count first, so half of
    full scale on a signed 16-bit sample (2**15) against a 1 V reference lands
    on the same 0.5 V:

    >>> x_bits = np.array([0, 16384, -16384])
    >>> pressure(x_bits, sensitivity=0, gain=0, volt_params=(16, 1.0))
    array([ 0.e+00,  5.e-07, -5.e-07])
    """
    x = _sequence_as_array(x)
    nu = 10 ** (sensitivity / 20)
    G = 10 ** (gain / 20)

    if volt_params is not None:
        nbits, v_ref = volt_params
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
        *, axis=None, floor: float = PRESSURE_FLOOR,
        complex_record: Optional[str] = None):
    """
    Calculate Sound Pressure Level (SPL) of acoustic pressure signal.

    Parameters
    ----------
    x : ndarray
        Acoustic pressure signal in pascals, as :func:`pressure` returns. A
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
        Lower bound on the MEAN-SQUARE pressure before the log, read the
        same way :func:`power_to_dB` and :func:`peak_level` read it (in
        ``ref**2`` units). Default :data:`PRESSURE_FLOOR`.
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

    Examples
    --------
    A 100 µPa-rms white signal — 1e-4 Pa — sits at ``20*log10(100) = 40`` dB
    re 1 µPa; the seed makes the sampling scatter around that reproducible.

    >>> rng = np.random.default_rng(0)
    >>> pressure_signal = rng.standard_normal(1000) * 100e-6
    >>> spl_dB = spl(pressure_signal)
    >>> print(f"SPL: {spl_dB:.2f} dB re 1 µPa")
    SPL: 39.81 dB re 1 µPa

    The mean-square pressure is floored at ``floor`` (default
    ``PRESSURE_FLOOR``) before the log, so a silent (all-zero) signal returns
    a finite ``20*log10(sqrt(PRESSURE_FLOOR)/ref)`` instead of ``-inf`` — -180 dB re
    1 µPa at the default reference, the same level :func:`power_to_dB` floors
    a silent signal at, because both read the same constant against the same
    reference.
    """
    rmsx = np.sqrt(np.mean(_squared_pressure(x, complex_record, 'spl'),
                           axis=axis))
    return 20 * np.log10(np.maximum(rmsx, np.sqrt(floor)) / ref)


def power_to_dB(power, ref: float = REFERENCE_PRESSURE_WATER, *,
                floor: float = PRESSURE_FLOOR):
    """Mean-square / power-like pressure quantity → level in dB re ``ref``.

    For a *squared* quantity (PSD in Pa²/Hz, SEL in Pa²·s, mean-square
    pressure, an f-k spectrum, …) the level is ``10·log10(power / ref²)``. The
    single conversion every spectral estimator should use: ``power`` is floored
    at ``floor`` before the log so a silent (zero) sample yields a finite, very
    negative level instead of ``-inf`` (which would otherwise poison a
    subsequent ``mean`` / ``histogram``).

    Parameters
    ----------
    power : array_like
        Squared-pressure quantity (e.g. PSD, SEL, |p|²); same units as
        ``ref**2``.
    ref : float, optional
        Reference pressure (default: 1 µPa, water). Use
        ``REFERENCE_PRESSURE_AIR`` for air.
    floor : float, optional
        Lower bound applied to ``power`` before the log (default
        :data:`PRESSURE_FLOOR`), guarding ``log10(0)``. Read in ``power``'s
        own units (``ref**2``), so the level a fully-silent input floors at
        depends on ``ref``: ``10*log10(1e-30 / 1e-12) = -180`` dB re 1 µPa
        at the default ``ref`` — the same level :func:`spl` floors a silent
        signal at, since it floors the rms pressure at
        ``sqrt(PRESSURE_FLOOR)`` = 1e-15 Pa and ``20*log10(1e-15 / 1e-6)``
        is -180 dB as well (``spl(0) == power_to_dB(0) == -180.0``).

    Returns
    -------
    numpy.ndarray
        Level in dB re ``ref``.
    """
    power = np.asarray(power, dtype=float)
    return 10.0 * np.log10(np.maximum(power, floor) / (ref ** 2))


def peak_level(x, ref: float = REFERENCE_PRESSURE_WATER, *, axis=None,
               floor: float = PRESSURE_FLOOR,
               complex_record: Optional[str] = None):
    """Peak pressure level of a record — ``20·log10(max|x| / ref)``.

    The other half of the dual metric: a level that a *peak* sets, where
    :func:`spl` is set by the rms and :func:`power_to_dB` by an energy.
    Southall et al. (2019) state injury criteria as a pair, peak SPL beside
    an exposure, because a short transient can reach a damaging peak while
    carrying little energy and a long one can do the reverse — neither
    number implies the other.

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
        Lower bound on the SQUARED peak before the log, read the same way
        :func:`power_to_dB` reads it, so a silent record returns
        ``-180`` dB re 1 µPa rather than ``-inf`` — which would poison any
        mean taken over a map of them.
    complex_record : {'analytic', 'baseband'}, optional
        Required for a complex record, ignored for a real one; as on
        :func:`spl`.

    Returns
    -------
    float or ndarray
        dB re ``ref``.
    """
    peak = np.max(_peak_amplitude(x, complex_record, 'peak_level'),
                  axis=axis)
    return 20 * np.log10(np.maximum(peak, np.sqrt(floor)) / ref)


def sound_exposure_level(pressure, sample_rate: float,
                         ref: float = REFERENCE_PRESSURE_WATER, *,
                         axis: int = -1, floor: float = PRESSURE_FLOOR,
                         complex_record: Optional[str] = None):
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
    total, in dB.

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
        :func:`spl`.

    Returns
    -------
    float or ndarray
        dB re ``ref²``·s.
    """
    p2 = _squared_pressure(pressure, complex_record, 'sound_exposure_level')
    fs = float(sample_rate)
    if not (np.isfinite(fs) and fs > 0.0):
        raise ConfigurationError(
            f"sound_exposure_level: sample_rate must be a positive rate in "
            f"Hz; got {sample_rate!r}.")
    if fs < 1.0:
        raise ConfigurationError(
            f"sound_exposure_level: sample_rate={fs:g} Hz reads as a sample "
            f"INTERVAL; the argument is the rate.",
            remediation=f"Pass sample_rate={1.0 / fs:g} (1/dt).")
    return power_to_dB(np.sum(p2, axis=axis) / fs, ref, floor=floor)


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
        Lower bound on ``|p|`` ITSELF before the log — not on ``|p|²`` as
        the ``floor`` of :func:`spl`, :func:`peak_level` and
        :func:`power_to_dB` is — so the default :data:`PRESSURE_FLOOR`
        caps a loss at 600 dB (:data:`NO_ENERGY_DB`, the no-energy marker)
        where the level helpers floor at -180 dB re 1 µPa. Pass ``0.0`` for
        the unclamped ``+inf``.

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
        arrs = np.broadcast_arrays(*[np.asarray(x, dtype=float)
                                     for x in levels_dB])
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


def integrate_psd(psd, frequencies, freq_min=None, freq_max=None) -> float:
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
        One-sided power spectral density, non-negative.
    frequencies : array_like
        Strictly increasing frequencies (Hz).
    freq_min, freq_max : float, optional
        The band (Hz); an unset edge is the grid's own end.
    """
    psd = np.asarray(psd, dtype=float)
    f = np.asarray(frequencies, dtype=float)
    if psd.ndim != 1 or psd.shape != f.shape:
        raise ConfigurationError(
            f"integrate_psd: psd and frequencies are one spectrum of equal "
            f"length; got shapes {psd.shape} and {f.shape}.")
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
    return float(np.trapezoid(np.interp(nodes, f, psd), nodes))


def band_level(level_density_dB, frequencies, freq_min=None,
               freq_max=None) -> float:
    """Band level (dB re ref²) of a level density (dB re ref²/Hz) over
    ``[freq_min, freq_max]`` Hz: ``10·log10 ∫ 10^(L/10) df``, integrated by
    :func:`integrate_psd`.

    A band that carries no power (every level ``-inf``) is ``-inf``, the
    level of an empty band throughout uacpy.

    Parameters
    ----------
    level_density_dB : array_like
        Level density (dB re ref²/Hz).
    frequencies : array_like
        Strictly increasing frequencies (Hz).
    freq_min, freq_max : float, optional
        The band (Hz); an unset edge is the grid's own end.
    """
    power = integrate_psd(
        10.0 ** (np.asarray(level_density_dB, dtype=float) / 10.0),
        frequencies, freq_min, freq_max)
    with np.errstate(divide='ignore'):
        return float(10.0 * np.log10(power))


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
    dB. That number is finite, so a reduction that masks only NaN/inf would
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
    :meth:`~uacpy.Field.at_source_level` goes through it; the signal-excess
    budgets (:func:`uacpy.sonar.passive_signal_excess_field`,
    :func:`uacpy.sonar.active_signal_excess_field`) mask on the one-way TL.

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
