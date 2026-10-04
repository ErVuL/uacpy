"""The passband PHY: the raised-cosine and root-raised-cosine pulses,
pulse shaping and its matched filter, and the up- and down-conversion
between baseband and a carrier."""

from __future__ import annotations

import numpy as np

from uacpy.core.exceptions import ConfigurationError


#: Symbol periods within which a pulse time counts as ON a removable
#: singularity of the raised-cosine pulses and takes its limit.
_SINGULARITY_ATOL = 1e-8


def _require_integer_sps(who, sps):
    """Return ``sps`` as an ``int``, accepting exactly-integral floats.

    ``fs / baud`` naturally produces a float (``4.0``, ``np.float64(4.0)``)
    that names the same sample grid as the integer, so it coerces. A
    fractional value names no grid at all: the upsampler's ``up[::sps]``
    stride and the RRC time axis ``arange(span*sps + 1) / sps`` are defined
    only for a whole number of samples per symbol.
    """
    try:
        i = int(sps)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(
            f"{who}: sps must be a whole number of samples per symbol; "
            f"got {sps!r} ({exc}).") from exc
    if i != sps:
        raise ConfigurationError(
            f"{who}: sps must be a whole number of samples per symbol; "
            f"got {sps!r}. If fs/baud is fractional, pick a sample rate the "
            f"baud rate divides evenly.")
    return i


def _require_rolloff(who, rolloff):
    """Refuse a raised-cosine roll-off outside ``[0, 1]``."""
    if not 0.0 <= rolloff <= 1.0:
        raise ConfigurationError(
            f"{who}: rolloff must be in [0, 1]; got {rolloff!r}.")


def rrc_pulse(t_symbols, rolloff):
    """Root-raised-cosine pulse ``g(t)`` at arbitrary times, unnormalised.

    ``t_symbols`` is time in symbol periods, any shape; the value at
    ``t = 0`` is ``1 - beta + 4 beta / pi``. This is the continuous pulse
    :func:`rrc_filter` samples on its ``sps`` grid, exposed so a tap can be
    placed at a delay that is not a whole number of samples
    (:meth:`uacpy.core.results.Arrivals.channel_taps` evaluates it at
    ``k T - tau_i``). Scale by ``sqrt(sum(rrc_filter(...) ** 2))`` of the
    same ``sps``/``span`` grid to put it on ``rrc_filter``'s unit-energy
    footing; here the peak is the textbook value and the energy is not one.

    Parameters
    ----------
    t_symbols : float or array_like
        Times in symbol periods.
    rolloff : float
        Excess bandwidth (roll-off factor), in ``[0, 1]``.
    """
    _require_rolloff("rrc_pulse", rolloff)
    t = np.asarray(t_symbols, dtype=float)
    b = float(rolloff)
    # The general expression divides by ``pi*t*(1 - (4*b*t)**2)``, which
    # vanishes at t = 0 and at |t| = 1/(4b). Both are removable singularities
    # of the RRC impulse response, so those points take their analytic
    # limits; _SINGULARITY_ATOL catches a grid landing on (or numerically
    # next to) either.
    at_zero = np.abs(t) < _SINGULARITY_ATOL
    at_pole = (np.abs(np.abs(t) - 1 / (4 * b)) < _SINGULARITY_ATOL if b > 0
               else np.zeros(t.shape, dtype=bool))
    safe = np.where(at_zero | at_pole, 0.5, t)
    with np.errstate(divide='ignore', invalid='ignore'):
        num = (np.sin(np.pi * safe * (1 - b))
               + 4 * b * safe * np.cos(np.pi * safe * (1 + b)))
        den = np.pi * safe * (1 - (4 * b * safe) ** 2)
        g = num / den
    g = np.where(at_zero, 1 - b + 4 * b / np.pi, g)
    if b > 0:
        pole = (b / np.sqrt(2)) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * b))
                                   + (1 - 2 / np.pi) * np.cos(np.pi / (4 * b)))
        g = np.where(at_pole, pole, g)
    return g


def rc_pulse(t_symbols, rolloff):
    """Raised-cosine pulse at arbitrary times, unit peak.

    ``sinc(t) cos(pi beta t) / (1 - (2 beta t)^2)`` with ``t_symbols`` in
    symbol periods, any shape; the removable singularity at
    ``|t| = 1/(2 beta)`` takes its limit ``(pi/4) sinc(1/(2 beta))``. It is
    the transmit root-raised-cosine convolved with its matched filter, so
    it is what a receiver sees at its decision instants: zero at every
    non-zero integer ``t`` (Nyquist) and, between integers, the
    inter-symbol interference a path delayed by a fraction of a symbol
    leaves. :meth:`uacpy.core.results.Arrivals.channel_taps` samples it at
    ``k T - tau_i`` for its symbol-spaced (``sps=1``) taps.

    Parameters
    ----------
    t_symbols : float or array_like
        Times in symbol periods.
    rolloff : float
        Excess bandwidth (roll-off factor), in ``[0, 1]``.
    """
    _require_rolloff("rc_pulse", rolloff)
    t = np.asarray(t_symbols, dtype=float)
    b = float(rolloff)
    at_pole = (np.abs(np.abs(t) - 1 / (2 * b)) < _SINGULARITY_ATOL if b > 0
               else np.zeros(t.shape, dtype=bool))
    safe = np.where(at_pole, 0.0, t)
    g = np.sinc(safe) * np.cos(np.pi * b * safe) / (1 - (2 * b * safe) ** 2)
    if b > 0:
        g = np.where(at_pole, (np.pi / 4) * np.sinc(1 / (2 * b)), g)
    return g


def rrc_filter(sps, rolloff, span):
    """Root-raised-cosine taps: ``span`` symbols, ``sps`` samples/symbol,
    unit energy.

    :func:`rrc_pulse` sampled at ``(arange(span*sps + 1) - span*sps/2) / sps``
    symbol periods and scaled to unit energy.

    Parameters
    ----------
    sps : int
        Samples per symbol, >= 1.
    rolloff : float
        Excess bandwidth (roll-off factor), in ``[0, 1]``.
    span : int
        Filter length in symbols, >= 1.
    """
    _require_rolloff("rrc_filter", rolloff)
    sps = _require_integer_sps("rrc_filter", sps)
    # sps is the divisor of the symbol-period axis: 0 makes every tap NaN and
    # returns a length-1 filter, a negative value makes `span*sps` negative
    # and returns an empty one — both convolve without complaint.
    if sps < 1:
        raise ConfigurationError(
            f"rrc_filter: sps must be >= 1 (samples per symbol); got "
            f"{sps!r}. It sets the taps' time axis (arange(span*sps + 1) "
            f"scaled by 1/sps), so the filter is undefined below 1.")
    if int(span) < 1:
        raise ConfigurationError(
            f"rrc_filter: span must be >= 1 (symbols); got {span!r}.")
    n = span * sps
    t = (np.arange(n + 1) - n / 2) / sps      # time in symbol periods
    h = rrc_pulse(t, rolloff)
    return h / np.sqrt(np.sum(h ** 2))


def pulse_shape(symbols, sps, rolloff=0.25, span=8):
    """Upsample symbols by ``sps`` and root-raised-cosine filter -> baseband samples.

    Parameters
    ----------
    symbols : array_like
        Symbols to shape.
    sps : int
        Samples per symbol, >= 1.
    rolloff : float, optional
        Excess bandwidth, in ``[0, 1]``. Default 0.25.
    span : int, optional
        Filter length in symbols. Default 8.
    """
    sps = _require_integer_sps("pulse_shape", sps)
    if sps < 1:
        raise ConfigurationError(
            f"pulse_shape: need sps >= 1 (samples per symbol); got {sps!r}.")
    s = np.asarray(symbols, dtype=complex).ravel()
    up = np.zeros(s.size * sps, dtype=complex)
    up[::sps] = s
    return np.convolve(up, rrc_filter(sps, rolloff, span))


def rrc_matched_filter(samples, sps, rolloff=0.25, span=8):
    """Matched root-raised-cosine filter (completes the Nyquist response).

    Named for the filter, not for the operation: "matched filter" alone means
    replica correlation everywhere else in the package
    (:func:`uacpy.acoustic_signal.matched_filter`, which takes a replica
    where this takes a samples-per-symbol count), and reading this module a
    bare ``matched_filter`` looked like that one. ``uacpy.comms`` exports it
    under this same name.

    Parameters
    ----------
    samples : array_like
        The received baseband samples.
    sps : int
        Samples per symbol, >= 1.
    rolloff : float, optional
        The transmit filter's roll-off. Default 0.25.
    span : int, optional
        The transmit filter's length in symbols. Default 8.
    """
    return np.convolve(np.asarray(samples, dtype=complex),
                       rrc_filter(sps, rolloff, span))


def upconvert(baseband, sample_rate, fc):
    """Mix complex baseband up to a real passband signal at carrier ``fc``.

    Parameters
    ----------
    baseband : array_like
        Complex baseband samples.
    sample_rate : float
        Sample rate (Hz).
    fc : float
        Carrier frequency (Hz).
    """
    x = np.asarray(baseband, dtype=complex)
    n = np.arange(x.size)
    return np.real(x * np.exp(2j * np.pi * fc * n / sample_rate))


def downconvert(passband, sample_rate, fc):
    """Mix a real passband signal down to complex baseband (image left for the LPF/MF).

    The factor 2 makes the ``upconvert``/``downconvert`` pair unity-gain:
    ``upconvert`` emits ``Re{b·e^{jwn}} = (b·e^{jwn} + b*·e^{-jwn})/2``, so
    ``2·x·e^{-jwn} = b + b*·e^{-2jwn}`` and the low-pass part is ``b`` itself.

    Parameters
    ----------
    passband : array_like
        Real passband samples.
    sample_rate : float
        Sample rate (Hz).
    fc : float
        Carrier frequency (Hz).
    """
    x = np.asarray(passband, dtype=float)
    n = np.arange(x.size)
    return 2.0 * x * np.exp(-2j * np.pi * fc * n / sample_rate)
