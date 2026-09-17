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
"""

import numpy as np
from typing import Optional, Tuple

from uacpy.core.constants import PRESSURE_FLOOR, REFERENCE_PRESSURE_WATER

__all__ = [
    'pressure',
    'spl',
    'power_to_dB',
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
    nu = 10 ** (sensitivity / 20)
    G = 10 ** (gain / 20)

    if volt_params is not None:
        nbits, v_ref = volt_params
        x = x * v_ref / (2 ** (nbits - 1))

    # 1e-6: SH is quoted per micropascal, so the division lands in µPa and
    # this carries the result to the pascals everything downstream defaults to.
    return 1e-6 * x / (nu * G)


def spl(x: np.ndarray, ref: float = REFERENCE_PRESSURE_WATER) -> float:
    """
    Calculate Sound Pressure Level (SPL) of acoustic pressure signal.

    Parameters
    ----------
    x : ndarray
        Acoustic pressure signal in pascals, as :func:`pressure` returns
    ref : float, optional
        Reference pressure in the same unit as ``x`` (default:
        ``REFERENCE_PRESSURE_WATER``, one µPa written in Pa). For air in Pa,
        ``20e-6``.

    Returns
    -------
    float
        Average SPL in dB re reference pressure

    Examples
    --------
    A 100 µPa-rms white signal — 1e-4 Pa — sits at ``20*log10(100) = 40`` dB
    re 1 µPa; the seed makes the sampling scatter around that reproducible.

    >>> rng = np.random.default_rng(0)
    >>> pressure_signal = rng.standard_normal(1000) * 100e-6
    >>> spl_dB = spl(pressure_signal)
    >>> print(f"SPL: {spl_dB:.2f} dB re 1 µPa")
    SPL: 39.81 dB re 1 µPa

    The rms pressure is floored at ``sqrt(PRESSURE_FLOOR)`` before the log, so
    a silent (all-zero) signal returns a finite
    ``20*log10(sqrt(PRESSURE_FLOOR)/ref)`` instead of ``-inf`` — -180 dB re
    1 µPa at the default reference, the same level :func:`power_to_dB` floors
    a silent signal at, because both read the same constant against the same
    reference.
    """
    rmsx = np.sqrt(np.mean(np.abs(x) ** 2))
    return 20 * np.log10(np.maximum(rmsx, np.sqrt(PRESSURE_FLOOR)) / ref)


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
        at the default ``ref``, against the -300 dB re 1 µPa :func:`spl`
        floors a silent signal at under its µPa default.

    Returns
    -------
    numpy.ndarray
        Level in dB re ``ref``.
    """
    power = np.asarray(power, dtype=float)
    return 10.0 * np.log10(np.maximum(power, floor) / (ref ** 2))
