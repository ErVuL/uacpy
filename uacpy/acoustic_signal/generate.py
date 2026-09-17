"""Signals to transmit or to test with: chirps, pulses, coded sequences
and noise realisations.

One question — *give me a signal* — answered three ways: parametric waveforms
(:func:`lfm_chirp`, :func:`tone_burst`, :func:`ricker_wavelet`, …), coded
sequences (:func:`mseq`, :func:`bpsk_modulate`), and noise built to a target
spectrum (:func:`synthesize_noise_from_psd`, :func:`add_noise`). Every one is
a pure function returning arrays; pass ``rng=`` for a reproducible draw.
"""

from typing import Optional, Tuple, Literal
import numpy as np
from uacpy.core.exceptions import ConfigurationError
from uacpy.acoustic_signal._signal_validate import (
    require_below_nyquist,
    require_positive_finite_scalar,
)
import math
import warnings


# ──────────────────────────────────────────────────────────────────────
# Parametric waveforms
#
# Deterministic pulses and sweeps: chirps,
# tone bursts, Ricker, Gaussian, N-wave, the SPARC pulse library.
# ──────────────────────────────────────────────────────────────────────

def sparc_pulse(
    t: np.ndarray,
    omega: float,
    pulse_type: Literal["P", "R", "A", "S", "H", "N", "M", "G", "T", "C", "E"],
) -> Tuple[np.ndarray, str]:
    """
    Compute source time series for various pulse shapes.

    Based on the original SPARC (1988) pulse library. Generates analytical
    pulse shapes commonly used in underwater acoustics.

    Parameters
    ----------
    t : ndarray
        Time vector (can be scalar or array)
    omega : float
        Angular frequency characterizing the pulse (rad/s)
        F = omega / (2*pi) is the characteristic frequency
    pulse_type : str
        Single letter code indicating pulse type. Each entry gives AT's own
        **spectral** label — the frequency of the spectral peak and the band
        occupied — then the time interval the code gates the pulse to:
        - 'P': Pseudo gaussian; peak at 0, band [0, 3F]; t in (0, 1/F]
        - 'R': Ricker wavelet; peak at F, band [0, 2F]; t > 0
        - 'A': Approximate Ricker wavelet; peak at F, band [0, 2.5F];
          t in (0, 1.55/F]
        - 'S': Single sine; peak at F, band [0, inf) with nulls at nF;
          t in (0, 1/F]
        - 'H': Hanning weighted four sine; peak at F, first null near 1.5F;
          t in (0, 4/F]
        - 'N': N-wave; peak at F, band [0, 4F] ([0, 3F] also OK); t in (0, 1/F]
        - 'M': Miracle wave; peak at 0, band [0, inf); t > 0
        - 'G': Gaussian; peak at 0, band [0, inf); t > 0
        - 'T': Tone burst / gated sinewave; peak at F, band [0, inf);
          t in (0, 0.4 s]
        - 'C': Sinc function; uniform spectrum [0, F]; every t != 0
        - 'E': One-sided exponential (AT gives no label); t > 0

    Returns
    -------
    s : ndarray
        Time series (same shape as t)
    pulse_title : str
        Descriptive name of the pulse

    Notes
    -----
    If forming a Hilbert transform, evaluate pulses at negative time to
    avoid artifacts.

    Most pulses have finite support (zero outside a time interval), making
    them suitable for transient analysis.

    Examples
    --------
    >>> # Generate a Ricker wavelet at 100 Hz
    >>> t = np.linspace(-0.1, 0.1, 1000)
    >>> f = 100.0  # Hz
    >>> omega = 2 * np.pi * f
    >>> s, title = sparc_pulse(t, omega, 'R')
    >>> print(title)
    Ricker wavelet

    >>> # Generate pseudo-Gaussian pulse
    >>> s_gauss, _ = sparc_pulse(t, omega, 'P')

    References
    ----------
    Transcribed from the shipped SPARC pulse library,
    ``Acoustics-Toolbox/Matlab/waveforms/cans.m`` (mbp's port of the 1988 SPARC
    Fortran). The Fortran copy ``Acoustics-Toolbox/tslib/cans.f90:24-93``
    carries identical coefficients but has no ``'E'`` case. Every coefficient
    here is that source's, including the ``omega*T - 5`` Ricker centring and
    the ``0.48829 / 0.14128 / 0.01168`` approximate-Ricker window. The
    "peak / band" labels above are AT's own (``cans.f90:26-88``) and are
    spectral, not time intervals: ``cans.f90:47`` labels the single sine
    "support [0, infinity], nulls at nF" while gating it to ``t <= 1/F``.

    Both AT copies gate every pulse to ``T > 0``; the sinc here is evaluated at
    negative time too, so it is the full two-sided sinc.
    """
    omega = require_positive_finite_scalar(omega, "sparc_pulse", "omega",
                                           " rad/s")
    t = np.asarray(t, dtype=float)
    s = np.zeros(t.shape)
    F = omega / (2.0 * np.pi)

    pulse_key = pulse_type[0].upper()

    if pulse_key == "P":  # Pseudo gaussian
        ii = (t > 0) & (t <= 1 / F)
        T = t[ii]
        s[ii] = 0.75 - np.cos(omega * T) + 0.25 * np.cos(2.0 * omega * T)
        pulse_title = "Pseudo gaussian"

    elif pulse_key == "R":  # Ricker wavelet
        ii = t > 0
        T = t[ii]
        U = omega * T - 5.0
        s[ii] = 0.5 * (0.25 * U * U - 0.5) * np.sqrt(np.pi) * np.exp(-0.25 * U * U)
        pulse_title = "Ricker wavelet"

    elif pulse_key == "A":  # Approximate Ricker wavelet
        # (TC/2pi)^2 times the second derivative of a 4-term Blackman-Harris
        # window over [0, TC]: 0.48829/0.14128/0.01168 are that window's
        # a1/a2/a3 (a0 = 0.35875 differentiates away) and the 1/4/9 are the n^2
        # from differentiating cos(2*pi*n*T/TC) twice. The true Ricker is the
        # second derivative of a Gaussian; this is its compact-support analogue.
        TC = 1.55 / F
        ii = (t > 0) & (t <= TC)
        T = t[ii]
        s[ii] = (
            +0.48829 * np.cos(2.0 * np.pi * T / TC)
            - 0.14128 * 4 * np.cos(4.0 * np.pi * T / TC)
            + 0.01168 * 9 * np.cos(6.0 * np.pi * T / TC)
        )
        pulse_title = "Approximate Ricker wavelet"

    elif pulse_key == "S":  # Single sine
        ii = (t > 0) & (t <= 1 / F)
        T = t[ii]
        s[ii] = np.sin(omega * T)
        pulse_title = "Single sine"

    elif pulse_key == "H":  # Hanning weighted four sine
        ii = (t > 0) & (t <= 4 / F)
        T = t[ii]
        s[ii] = 0.5 * np.sin(omega * T) * (1 - np.cos(omega * T / 4.0))
        pulse_title = "Hanning weighted four sine"

    elif pulse_key == "N":  # N-wave
        ii = (t > 0) & (t <= 1 / F)
        T = t[ii]
        s[ii] = np.sin(omega * T) - 0.5 * np.sin(2.0 * omega * T)
        pulse_title = "N-wave"

    elif pulse_key == "M":  # Miracle wave
        ii = t > 0
        T = t[ii]
        A = 1.0 / (6.0 * F)
        T0 = 6.0 * A
        TS = (T - T0) / A
        s[ii] = 1.0 / (1.0 + TS * TS)
        pulse_title = "Miracle wave"

    elif pulse_key == "G":  # Gaussian
        ii = t > 0
        T = t[ii]
        NSIG = 3
        A = 1.0 / F / (2.0 * NSIG)
        T0 = NSIG * A
        s[ii] = np.exp(-(((T - T0) / A) ** 2))
        pulse_title = "Gaussian"

    elif pulse_key == "T":  # Tone burst
        ii = (t > 0) & (t <= 0.4)
        T = t[ii]
        s[ii] = np.sin(omega * T)
        pulse_title = "Tone"

    elif pulse_key == "C":  # Sinc
        ii = t != 0  # Avoid division by zero
        T = t[ii]
        s[ii] = np.sin(omega * T) / (omega * T)
        s[t == 0] = 1.0  # Limit as t->0
        pulse_title = "Sinc"

    elif pulse_key == "E":  # One-sided exponential
        ii = t > 0
        T = t[ii]
        s[ii] = np.exp(-omega * T)
        pulse_title = "One-sided exponential"

    else:
        raise ConfigurationError(
            f"Unknown pulse type: '{pulse_type}'. "
            "Valid types: P, R, A, S, H, N, M, G, T, C, E"
        )

    return s, pulse_title


def ricker_wavelet(time: np.ndarray, frequency: float,
                   delay: Optional[float] = None) -> np.ndarray:
    """
    Generate a Ricker wavelet (Mexican hat wavelet).

    The Ricker wavelet is the second derivative of a Gaussian and is
    commonly used in seismic and acoustic applications. Uses the AT
    ``Ricker.m`` centring ``u = 2πFt − 8``; SPARC's internal Ricker
    (``cans.f90``, documented in ``models/sparc.py``) centres at
    ``ωT − 5`` — the two "Ricker" pulses are offset in time.

    ``delay`` overrides that fixed centring and places the wavelet's centre at
    a time you choose, which is what a gather needs: one pulse per trace at a
    moveout-dependent time. It broadcasts against ``time``, so a whole gather
    is one call. Omitting it keeps AT's centring exactly, and
    ``delay=4/(pi*frequency)`` reproduces that same wavelet to within
    floating-point round-off (~1e-15 of the lobe amplitude: the two
    dimensionless-time expressions are algebraically equal but not
    bit-identical).

    Parameters
    ----------
    time : ndarray
        Time vector
    frequency : float
        Nominal source frequency in Hz
    delay : float or ndarray, optional
        Centre the wavelet here instead of at AT's ``4/(pi*frequency)``.
        Broadcast against ``time``. Note that AT's offset exists to make
        truncation at ``time = 0`` free (``s(0)`` is 3e-6 of the lobe
        amplitude); a delay smaller than that truncates the leading flank,
        which is the caller's call to make.

    Returns
    -------
    s : ndarray
        Ricker wavelet time series

    Notes
    -----
    ``Ricker.m``'s label "peak at F, support [0, 2F]" is spectral, as AT's
    labels are throughout the family (``cans.f90:47`` labels the one-period
    single sine "support [0, infinity], nulls at nF"): the amplitude spectrum
    peaks at ``frequency`` and is ~14 dB down by twice it.

    Substituting ``tau = time - 4/(pi*frequency)`` turns the expression into
    ``0.25*sqrt(pi) * (2*pi^2*f^2*tau^2 - 1) * exp(-pi^2*f^2*tau^2)``, i.e. the
    standard Ricker parameterised by *peak* frequency, scaled by
    ``0.25*sqrt(pi)`` and **negated** — the central lobe at
    ``time = 4/(pi*frequency)`` is a trough of -0.443, not a peak. The ``-8``
    offset places that centre far enough from the origin that truncating at
    ``time = 0`` costs nothing: ``s(0)`` is 3e-6 of the lobe amplitude, against
    2e-2 for SPARC's ``omega*T - 5`` centring.

    Examples
    --------
    >>> time = np.linspace(0, 0.1, 1000)
    >>> s = ricker_wavelet(time, 50.0)

    References
    ----------
    Original MATLAB code: Ricker.m
    """
    frequency = require_positive_finite_scalar(frequency, "ricker_wavelet",
                                               "frequency", " Hz")
    time = np.asarray(time, dtype=float)
    if delay is None:
        u = 2 * np.pi * frequency * time - 8  # Dimensionless time
    else:
        centre = np.asarray(delay, dtype=float)
        if not np.isfinite(centre).all():
            raise ConfigurationError(
                "ricker_wavelet: delay must be finite.",
                remediation="A non-finite centre puts the whole wavelet at "
                            "NaN. Omit delay for the Acoustics-Toolbox "
                            "centring at 4/(pi*frequency).")
        # Same dimensionless time, centred where the caller asked: at
        # delay = 4/(pi*frequency) this is identical to the branch above.
        u = 2 * np.pi * frequency * (time - centre)
    s = 0.5 * (0.25 * u**2 - 0.5) * np.sqrt(np.pi) * np.exp(-0.25 * u**2)
    return s


def gaussian_pulse(time: np.ndarray, delay: float, duration: float) -> np.ndarray:
    """
    Generate a Gaussian pulse.

    Parameters
    ----------
    time : ndarray
        Vector of sample times
    delay : float
        Time of the pulse peak location
    duration : float
        Pulse duration (width parameter)

    Returns
    -------
    y : ndarray
        Gaussian pulse

    Notes
    -----
    The pulse has form: exp(-((t - delay) / duration)^2)

    Time, delay, and duration should all be in the same units (e.g., seconds).

    Examples
    --------
    >>> time = np.linspace(0, 1, 1000)
    >>> pulse = gaussian_pulse(time, delay=0.5, duration=0.1)

    References
    ----------
    Original MATLAB code by mbp, 2001
    """
    duration = require_positive_finite_scalar(duration, "gaussian_pulse",
                                              "duration", " s")
    time = np.asarray(time, dtype=float)
    y = np.exp(-(((time - delay) / duration) ** 2))
    return y


def lfm_chirp(
    fmin: float, fmax: float, duration: float, sample_rate: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a Linear Frequency Modulated (LFM) pulse (chirp).

    Creates a signal that sweeps linearly from fmin to fmax over ``duration``.

    Parameters
    ----------
    fmin : float
        Sweep start frequency in Hz, ``>= 0``
    fmax : float
        Sweep end frequency in Hz, ``>= 0`` (may be below ``fmin`` for a
        downsweep)
    duration : float
        Duration of time-series in seconds
    sample_rate : float
        Samples per second (Hz)

    Returns
    -------
    time : ndarray
        Time vector
    s : ndarray
        LFM signal

    Notes
    -----
    The signal is the conventional linear sweep ``s(t) = sin(2π φ(t))``
    with quadratic phase ``φ(t) = fmin·t + (fmax - fmin)·t² / (2·T)``;
    the instantaneous frequency ``dφ/dt = fmin + (fmax - fmin)·t / T``
    therefore ramps linearly from ``fmin`` at ``t = 0`` to ``fmax`` at
    ``t = T``. This is a standard chirp used in sonar and radar
    applications.

    ``fmin == fmax`` is accepted and yields a constant-frequency tone (the
    sweep rate is simply zero); ``duration`` and ``sample_rate`` must be
    positive and long enough for at least one sample, and the sweep must stay
    below the Nyquist frequency ``sample_rate/2``, otherwise a
    :class:`~uacpy.core.exceptions.ConfigurationError` is raised.

    Examples
    --------
    >>> # Generate 1-second chirp from 100 to 1000 Hz
    >>> t, s = lfm_chirp(100, 1000, 1.0, 10000)

    >>> # Can also use scipy.signal.chirp for similar functionality
    >>> from scipy.signal import chirp
    >>> t = np.linspace(0, 1, 10000)
    >>> s_scipy = chirp(t, 100, 1, 1000)

    References
    ----------
    Original MATLAB code: lfm.m
    """
    duration = require_positive_finite_scalar(duration, "lfm_chirp",
                                              "duration", " s")
    sample_rate = require_positive_finite_scalar(sample_rate, "lfm_chirp",
                                                 "sample_rate", " Hz")
    # >= 0 rather than > 0: fmin == fmax == 0 is a DC "sweep", degenerate but
    # harmless. A negative bound is not — the sweep crosses DC and folds about
    # it, returning a waveform whose instantaneous frequency never matches what
    # was asked for. hfm_chirp and tone_burst already refuse it via
    # require_positive_finite_scalar; written as the negated condition so NaN
    # is refused here too.
    for _name, _value in (("fmin", fmin), ("fmax", fmax)):
        if not (_value >= 0.0):
            raise ConfigurationError(
                f"lfm_chirp: {_name} must be a finite frequency >= 0 Hz; got "
                f"{_value}. A negative bound folds the sweep about DC.")
    f_top = max(fmin, fmax)
    require_below_nyquist(f_top, sample_rate, "lfm_chirp",
                          "the top of the sweep",
                          "the sampled waveform aliases")
    T = duration  # local alias for the sweep-duration symbol in the phase law
    N = int(round(T * sample_rate))  # so dt == 1/sample_rate exactly
    if N <= 0:
        raise ConfigurationError(
            "lfm_chirp: duration * sample_rate must cover at least one sample; "
            f"got duration={duration}, sample_rate={sample_rate}.")
    time = np.arange(N) / sample_rate

    # Time-averaged frequency over [0, t]; 2*pi*f_avg*t is the chirp phase
    # (instantaneous frequency is fmin + (fmax-fmin)*t/T, twice the slope).
    f_avg = fmin + (fmax - fmin) * time / (2 * T)
    s = np.sin(2.0 * np.pi * f_avg * time)

    return time, s


def tone_burst(
    frequency: float, n_cycles: int, sample_rate: float, window: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a tone burst (windowed sinusoid).

    Parameters
    ----------
    frequency : float
        Tone frequency in Hz
    n_cycles : int
        Number of cycles
    sample_rate : float
        Sample rate in Hz
    window : bool, optional
        If True, apply Hanning window (default: True)

    Returns
    -------
    time : ndarray
        Time vector
    s : ndarray
        Tone burst signal

    Notes
    -----
    All parameters must be positive, the tone below the Nyquist frequency
    ``sample_rate/2``, and the burst long enough for at least one sample
    (``round(n_cycles / frequency * sample_rate) >= 1``), otherwise a
    :class:`~uacpy.core.exceptions.ConfigurationError` is raised.

    Examples
    --------
    >>> # Generate 5-cycle 1000 Hz tone burst
    >>> t, s = tone_burst(1000.0, 5, 48000)

    >>> # Without windowing
    >>> t, s_rect = tone_burst(1000.0, 5, 48000, window=False)
    """
    from scipy.signal.windows import hann

    # ``T`` is the requested burst duration in seconds; the sample count
    # ``N`` is the *nearest* integer that keeps ``n_cycles`` faithful at
    # the given ``sample_rate``. ``time`` is built as
    # ``np.arange(N) / sample_rate`` so ``dt == 1 / sample_rate`` exactly.
    frequency = require_positive_finite_scalar(frequency, "tone_burst",
                                               "frequency", " Hz")
    n_cycles = require_positive_finite_scalar(n_cycles, "tone_burst",
                                              "n_cycles", " cycles")
    sample_rate = require_positive_finite_scalar(sample_rate, "tone_burst",
                                                 "sample_rate", " Hz")
    require_below_nyquist(frequency, sample_rate, "tone_burst", "frequency",
                          "the sampled tone aliases")
    T = n_cycles / frequency
    N = int(round(T * sample_rate))
    if N <= 0:
        raise ConfigurationError(
            "tone_burst: n_cycles / frequency * sample_rate must cover at "
            f"least one sample; got frequency={frequency}, "
            f"n_cycles={n_cycles}, sample_rate={sample_rate}.")
    time = np.arange(N) / float(sample_rate)

    s = np.sin(2 * np.pi * frequency * time)

    if window:
        s = s * hann(N)

    return time, s


def hfm_chirp(
    fmin: float, fmax: float, duration: float, sample_rate: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a Hyperbolic Frequency Modulated (HFM) pulse.

    Also known as Linear Period Modulation (LPM). The instantaneous
    frequency varies hyperbolically with time.

    Parameters
    ----------
    fmin : float
        Minimum frequency in Hz
    fmax : float
        Maximum frequency in Hz
    duration : float
        Duration in seconds
    sample_rate : float
        Sample rate in Hz

    Returns
    -------
    time : ndarray
        Time vector
    s : ndarray
        HFM signal

    Notes
    -----
    HFM chirps have constant period change rate rather than constant
    frequency change rate (like LFM). This makes them more Doppler-tolerant.

    ``fmin`` and ``fmax`` must both be positive and distinct (the phase law
    divides by ``fmin - fmax`` and by ``fmin``), and the sweep must stay below
    the Nyquist frequency ``sample_rate/2``; ``fmin > fmax`` is accepted
    and gives a down-sweep. Degenerate parameters raise
    :class:`~uacpy.core.exceptions.ConfigurationError`.

    The phase is: φ(t) = (2π/b) * log(1 + b*t/P1)
    where b = (fmin - fmax)/(fmin*fmax*T) and P1 = 1/fmin

    Sign convention: ``b`` here is the slope of the period, ``Period(t) =
    1/fmin + b*t``, running from ``1/fmin`` at ``t = 0`` to ``1/fmax`` at
    ``t = T`` — so ``b`` is **negative** for an up-sweep. Abraham,
    *Underwater Acoustic Signal Processing*, §8.3.6 defines the opposite sign,
    ``b_A = (f1 - f0)/(f0*f1*Tp)`` with ``φ = -(2π/b_A)·log(1 - b_A*f0*t)``.
    The two sign flips cancel identically, so this is Abraham's pulse, not a
    down-sweep: do not "fix" either sign in isolation.

    Examples
    --------
    >>> t, s = hfm_chirp(1000, 5000, 0.1, 48000)

    References
    ----------
    Original MATLAB: ``third_party/Acoustics-Toolbox/Matlab/waveforms/hfm.m``
    """
    fmin = require_positive_finite_scalar(fmin, "hfm_chirp", "fmin", " Hz")
    fmax = require_positive_finite_scalar(fmax, "hfm_chirp", "fmax", " Hz")
    duration = require_positive_finite_scalar(duration, "hfm_chirp",
                                              "duration", " s")
    sample_rate = require_positive_finite_scalar(sample_rate, "hfm_chirp",
                                                 "sample_rate", " Hz")
    if fmin == fmax:
        raise ConfigurationError(
            "hfm_chirp: fmin and fmax must differ (the hyperbolic phase law "
            f"divides by fmin - fmax); got fmin == fmax == {fmin}. For a "
            "constant-frequency signal use tone_burst or lfm_chirp with "
            "fmin == fmax.")
    f_top = max(fmin, fmax)
    require_below_nyquist(f_top, sample_rate, "hfm_chirp",
                          "the top of the sweep",
                          "the sampled waveform aliases")
    T = duration  # local alias for the sweep-duration symbol in the phase law
    N = int(round(T * sample_rate))  # so dt == 1/sample_rate exactly
    if N <= 0:
        raise ConfigurationError(
            "hfm_chirp: duration * sample_rate must cover at least one sample; "
            f"got duration={duration}, sample_rate={sample_rate}.")
    time = np.arange(N) / sample_rate

    # b < 0 for an up-sweep; see the sign convention in the docstring.
    b = (fmin - fmax) / (fmin * fmax * T)
    P1 = 1 / fmin
    s = np.sin((2 * np.pi / b) * np.log(1 + b * time / P1))

    return time, s


def nwave(time: np.ndarray, frequency: float) -> np.ndarray:
    """
    Generate an N-wave pulse.

    An N-wave is a characteristic waveform shape consisting of a sin wave
    minus half of its second harmonic, creating an N-shaped pulse.

    Parameters
    ----------
    time : ndarray
        Time vector
    frequency : float
        Nominal source frequency in Hz

    Returns
    -------
    s : ndarray
        N-wave signal, zero outside [0, 1/frequency]

    Notes
    -----
    The N-wave formula:
        s(t) = sin(ωt) - 0.5*sin(2ωt)  for 0 ≤ t ≤ 1/F
        s(t) = 0                         otherwise

    ``Nwave.m``'s own label "peak at F, support [0, 4F], [0,3F] also OK" is
    spectral (see :func:`ricker_wavelet`): the spectrum peaks near ``frequency``
    and is essentially spent by 3-4 times it. The *time* extent is the gate
    below, ``[0, 1/frequency]``.

    Translated from ``third_party/Acoustics-Toolbox/Matlab/waveforms/Nwave.m``

    Examples
    --------
    >>> # Generate 100 Hz N-wave
    >>> t = np.linspace(-0.01, 0.02, 1000)
    >>> s = nwave(t, 100.0)
    >>> print(f"Non-zero samples: {np.sum(s != 0)}")
    Non-zero samples: 332
    """
    frequency = require_positive_finite_scalar(frequency, "nwave",
                                               "frequency", " Hz")
    time = np.asarray(time, dtype=float)
    omega = 2 * np.pi * frequency
    s = np.sin(omega * time) - 0.5 * np.sin(2 * omega * time)

    # Zero outside [0, 1/frequency]. np.where rather than mask assignment: a
    # scalar `time` makes `s` 0-d, which does not support item assignment, and
    # ricker_wavelet/sparc_pulse both take scalars.
    return np.where((time > 1 / frequency) | (time < 0), 0.0, s)


# ──────────────────────────────────────────────────────────────────────
# Coded sequences
#
# M-sequences, BPSK modulation and the
# m-sequence channel probe.
# ──────────────────────────────────────────────────────────────────────

def bpsk_modulate(
    s_bipolar: np.ndarray, fc: float, sample_rate: float, chips_per_sec: float
) -> np.ndarray:
    """
    Encode binary sequence as Binary Phase Shift Keying (BPSK) signal.

    Parameters
    ----------
    s_bipolar : ndarray
        Binary source sequence (+1/-1 values)
    fc : float
        Carrier frequency in Hz
    sample_rate : float
        Sample frequency in Hz
    chips_per_sec : float
        Chip rate (symbols per second)

    Returns
    -------
    s : ndarray
        BPSK modulated signal

    Notes
    -----
    Each binary symbol (chip) is represented by a sinusoid of length
    samples_per_chip. The phase is 0 for +1, π for -1.

    One chip-length carrier is built once and replicated, so the carrier phase
    **restarts at zero in every chip** (as in the ``bpsk.m`` original). The
    result is phase-continuous across chip boundaries only when ``fc`` is an
    integer multiple of ``chips_per_sec``; otherwise each boundary carries a
    phase step that widens the transmitted spectrum.

    Chip values outside {-1, +1} (a 0-valued chip emits silence — on-off
    keying, not BPSK) and a carrier at or above ``sample_rate/2`` raise
    :class:`~uacpy.core.exceptions.ConfigurationError`.

    Examples
    --------
    >>> # Binary sequence
    >>> bits = np.array([1, -1, 1, 1, -1, 1])
    >>>
    >>> # BPSK modulation
    >>> fc = 12000  # 12 kHz carrier
    >>> sample_rate = 48000  # 48 kHz sample rate
    >>> chips_per_sec = 3000  # 3k chips/sec
    >>> s = bpsk_modulate(bits, fc, sample_rate, chips_per_sec)

    References
    ----------
    Original MATLAB code by Michael B. Porter, April 2000
    """
    sample_rate = require_positive_finite_scalar(
        sample_rate, "bpsk_modulate", "sample_rate", " Hz")
    chips_per_sec = require_positive_finite_scalar(
        chips_per_sec, "bpsk_modulate", "chips_per_sec", " chips/s")
    samples_per_chip = int(sample_rate / chips_per_sec)

    if sample_rate / chips_per_sec != samples_per_chip:
        raise ConfigurationError(
            "bpsk_modulate: samples_per_chip must be an integer; got "
            f"sample_rate/chips_per_sec = {sample_rate:g}/{chips_per_sec:g} "
            f"= {sample_rate / chips_per_sec:g}")

    require_below_nyquist(fc, sample_rate, "bpsk_modulate", "fc",
                          "the sampled carrier aliases")

    chips = np.asarray(s_bipolar)
    invalid = chips[~np.isin(chips, (-1, 1))]
    if invalid.size:
        raise ConfigurationError(
            f"bpsk_modulate: s_bipolar must contain only +1/-1 chips; got "
            f"{np.unique(invalid)[:5]}. A 0-valued chip emits silence, "
            f"turning BPSK into on-off keying — map bits first with "
            f"s = 1 - 2*bits.")

    deltat = 1 / sample_rate
    t_chip = np.arange(samples_per_chip) * deltat
    sinwave = np.sin(2 * np.pi * fc * t_chip)

    # Outer product: each column is one chip, so a column-major (Fortran-order)
    # flatten concatenates the chips in sequence order.
    s_matrix = np.outer(sinwave, s_bipolar)
    s = s_matrix.flatten(order="F")

    return s


def mseq(m: int) -> np.ndarray:
    """
    Generate an m-sequence (maximum-length sequence).

    M-sequences are pseudorandom binary sequences with excellent
    autocorrelation properties, useful for coded waveforms in sonar.

    Parameters
    ----------
    m : int
        Sequence order (2 ≤ m ≤ 15).
        Generates sequence of length 2^m - 1.

    Returns
    -------
    s : ndarray
        M-sequence as +1/-1 values. Length = 2^m - 1

    Notes
    -----
    Uses shift register with feedback based on primitive polynomials.
    The resulting sequence has:
    - Length N = 2^m - 1
    - Two-valued periodic autocorrelation (N at zero lag, -1 at every other
      lag) — ideal for matched filtering
    - Balanced to within one symbol: 2^(m-1) chips of -1 and 2^(m-1)-1 of +1,
      so the sequence sums to -1 rather than 0

    Chips use the standard BPSK mapping ``s = 1 - 2*bit`` (bit 0 → +1,
    bit 1 → -1), the same polarity as :func:`uacpy.comms.modulate.m_sequence`.
    The two are **not interchangeable across a spread/despread pair**: they
    start from different register seeds (``[1, 0, 0, 0, 0]`` here,
    ``(1,) * n`` there), so even the tap sets that generate the same cycle
    produce a shift of it. Despreading one function's output with the other's
    sequence lands on the m-sequence's off-peak correlation ``-1/N`` — sign
    inverted and collapsed by a factor of ``N`` (measured ``-0.032`` against
    ``1.0`` at ``n = 5``). Use the same generator at both ends.

    Translated from ``third_party/Acoustics-Toolbox/Matlab/waveforms/mseq.m``
    (Michael B. Porter, April 2000); the feedback-coefficient table and the
    shift recursion below are that file's, which credits Proakis, *Digital
    Communications*, p. 433. The MATLAB original maps the opposite way
    (``s(s == 0) = -1``, i.e. bit 1 → +1), so this sequence is the negative
    of ``mseq.m``'s — the autocorrelation is unaffected.

    Examples
    --------
    >>> # Generate m-sequence of order 5
    >>> s = mseq(5)
    >>> print(f"Length: {len(s)} (should be 2^5-1 = 31)")
    Length: 31 (should be 2^5-1 = 31)

    >>> # Check autocorrelation
    >>> shat = np.fft.fft(s)
    >>> scorr = np.real(np.fft.ifft(shat * np.conj(shat)))
    """
    if m < 2 or m > 15 or m != int(m):
        raise ConfigurationError(
            f"mseq: m must be an integer between 2 and 15; got {m!r}")

    m = int(m)

    # Feedback coefficients for primitive polynomials
    coefficients = {
        2: [1, 1],
        3: [1, 0, 1],
        4: [1, 0, 0, 1],
        5: [1, 0, 0, 1, 0],
        6: [1, 0, 0, 0, 0, 1],
        7: [1, 0, 0, 0, 0, 0, 1],
        8: [1, 0, 0, 0, 1, 1, 1, 0],
        9: [1, 0, 0, 0, 0, 1, 0, 0, 0],
        10: [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        11: [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        12: [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
        13: [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
        14: [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        15: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    }

    c = np.array(coefficients[m])
    length = 2**m - 1

    # Successive shifts with feedback. Any non-zero seed traverses the same
    # cycle (differing only by a shift); all-zero is the LFSR's absorbing state
    # and would emit zeros forever, so it is the one seed that is excluded.
    seed = np.zeros(m)
    seed[0] = 1
    s = np.zeros(length)

    for ii in range(length):
        out = np.zeros(m)
        out[: m - 1] = seed[1:m]
        out[m - 1] = np.mod(np.dot(c, seed), 2)  # Addition mod 2
        seed = out
        s[ii] = out[0]

    # Standard BPSK mapping: bit 0 -> +1, bit 1 -> -1.
    s = 1.0 - 2.0 * s

    return s


def make_mseq_probe(fmin: float, fmax: float, sample_rate: float, T_tot: float) -> np.ndarray:
    """
    Generate an m-sequence probe signal with BPSK modulation.

    Creates a repeated m-sequence probe with a leader, suitable for
    channel sounding experiments.

    Parameters
    ----------
    fmin : float
        Minimum frequency in Hz
    fmax : float
        Maximum frequency in Hz
    sample_rate : float
        Sampling rate in Hz
    T_tot : float
        Total duration in seconds

    Returns
    -------
    probe : ndarray
        BPSK-modulated m-sequence probe signal

    Notes
    -----
    The probe consists of:

    1. Leader (0.2 s of zeros)
    2. Repeated m-sequence (order 10, length 1023)
    3. BPSK modulation at center frequency fc = (fmin + fmax) / 2
    4. Zero-padding to T_tot

    Chip rate is (fmax - fmin) / 2: a rectangular chip of duration ``T_chip``
    has a sinc spectrum whose first nulls sit at ``+/- 1/T_chip``, so a chip
    rate of half the requested width fills ``[fmin, fmax]`` between those
    nulls. Output is normalized to 0.95 of full scale and is exactly
    ``round(T_tot * sample_rate)`` samples long.

    Raises :class:`~uacpy.core.exceptions.ConfigurationError` if ``T_tot`` is
    too short to hold the leader plus one full m-sequence period — a partial
    period would lose the two-valued autocorrelation the probe exists for, so
    increase ``T_tot`` or widen ``fmax - fmin`` to raise the chip rate.

    Translated from OALIB makemseq.m by mbp.

    Examples
    --------
    >>> # Generate 10-second probe, 1-2 kHz
    >>> probe = make_mseq_probe(1000, 2000, 10000, 10.0)
    >>> print(f"Probe length: {len(probe)} samples")
    Probe length: 100000 samples
    """
    lead_time = 0.2  # seconds

    # M-sequence parameters
    fc = 0.5 * (fmin + fmax)  # center frequency
    chips_per_sec = 0.5 * (fmax - fmin)

    # The BPSK main lobe spans fc +/- chips_per_sec, i.e. exactly
    # [fmin, fmax], so its upper edge fc + chips_per_sec = fmax must sit
    # below Nyquist.
    if fc + chips_per_sec >= sample_rate / 2:
        raise ConfigurationError(
            f"make_mseq_probe: the carrier (fmin + fmax)/2 = {fc:g} Hz plus "
            f"the chip-rate bandwidth (fmax - fmin)/2 = {chips_per_sec:g} Hz "
            f"reaches {fc + chips_per_sec:g} Hz (= fmax), at or above the "
            f"Nyquist frequency sample_rate/2 = {sample_rate / 2:g} Hz, so "
            f"the sampled probe aliases.")

    # Generate base m-sequence (order 10 → length 1023)
    s_m = mseq(10)
    s = bpsk_modulate(s_m, fc, sample_rate, chips_per_sec)

    # Whole m-sequence periods that fit after the leader, counted in samples so
    # the probe lands at exactly target_n. Counting the leader is what keeps the
    # probe inside T_tot; a period is never truncated, since a partial
    # m-sequence loses the two-valued autocorrelation the probe exists for.
    leader = np.zeros(int(lead_time * sample_rate))
    target_n = int(round(T_tot * sample_rate))
    Nreps = (target_n - leader.size) // len(s)
    if Nreps < 1:
        raise ConfigurationError(
            f"make_mseq_probe: T_tot={T_tot:g} s is too short for the "
            f"{lead_time:g} s leader plus one m-sequence period "
            f"({len(s) / sample_rate:.3f} s at chip rate {chips_per_sec:g} chips/s). "
            f"Increase T_tot, or widen (fmax - fmin) to raise the chip rate."
        )
    probe = np.tile(s, Nreps)
    probe_max = np.max(np.abs(probe))
    if probe_max > 0:
        probe = np.concatenate([leader, 0.95 * probe / probe_max])
    else:
        probe = np.concatenate([leader, probe])

    # Zero-fill to the exact total duration (leader + Nreps periods <= target_n).
    if probe.size < target_n:
        probe = np.concatenate([probe, np.zeros(target_n - probe.size)])

    return probe


# ──────────────────────────────────────────────────────────────────────
# Noise synthesis
#
# A realisation matching a target spectrum,
# band-limited noise, and mixing a signal with noise at a stated SNR.
# ──────────────────────────────────────────────────────────────────────

def synthesize_noise_from_psd(Pxx, Fxx, duration=1, scale=1, *,
                              n_fft=65536, sample_rate=None, interp='linear',
                              rng=None):
    """
    Spectral Synthesis of Random Processes.

    Generate a time-domain noise realisation whose one-sided PSD matches a
    user-supplied target ``Pxx(Fxx)``. The target is resampled onto the
    FFT-native frequency grid ``f_k = k * sample_rate / n_fft`` before synthesis,
    so ``Fxx`` may be uniform, log-spaced, or coarse (e.g. Wenz curves).

    Parameters
    ----------
    Pxx : array_like
        One-sided power spectral density in (U/scale)**2/Hz. Length ≥ 2.
    Fxx : array_like
        Frequency array in Hz, strictly increasing. Need not be uniform.
    duration : float
        Duration of the generated signal in seconds.
    scale : float
        Scale factor applied to the output signal.
    n_fft : int, optional
        IFFT chunk size. Defaults to 65536; must be a power of two in
        [16, 262144] — values below 16 are reset to 65536, values above
        262144 are clamped, and non-powers of two are rounded to the
        geometrically closest power of two (nearest in log2), each with a
        warning.
    sample_rate : float, optional
        Output sample rate in Hz. Defaults to 2*Fxx[-1].
    interp : {'linear', 'log', 'pchip', 'nearest'}, optional
        How to resample ``Pxx(Fxx)`` onto the FFT-native grid. ``'log'``
        interpolates ``log10(Pxx)`` vs ``log10(f)`` — recommended for
        broadband PSDs spanning many decades. Frequencies outside
        ``[Fxx[0], Fxx[-1]]`` are set to zero.
    rng : numpy.random.Generator, optional
        Random generator for the spectral draw. Pass a seeded generator for a
        reproducible realisation.

    Returns
    -------
    t : ndarray
        Time array in seconds.
    x : ndarray
        Generated signal array.
    sample_rate : float
        Sampling frequency in Hz — the rate the time axis is built from
        (equal to the ``sample_rate`` argument, or ``2*Fxx[-1]`` when that
        was omitted).

    Examples
    --------
    >>> import numpy as np
    >>> f = np.logspace(0, 4, 64)
    >>> Pxx = 1e-6 / (1 + (f / 100) ** 2)
    >>> t, x, sample_rate = synthesize_noise_from_psd(Pxx, f, duration=10,
    ...                 n_fft=2**16, sample_rate=40_000, interp='log')
    """
    MAX_NFFT = 262144

    Pxx = np.asarray(Pxx, dtype=float)
    Fxx = np.asarray(Fxx, dtype=float)
    if Pxx.ndim != 1 or Fxx.shape != Pxx.shape:
        raise ConfigurationError(
            f"synthesize_noise_from_psd: Pxx and Fxx must be 1-D arrays of equal length; "
            f"got Pxx.shape={Pxx.shape} and Fxx.shape={Fxx.shape}"
        )
    if Pxx.size < 2:
        raise ConfigurationError(
            f"synthesize_noise_from_psd: Pxx must have at least 2 points (got {Pxx.size})"
        )
    if not np.all(np.diff(Fxx) > 0):
        raise ConfigurationError(
            "synthesize_noise_from_psd: Fxx must be strictly increasing; got "
            f"{int(np.count_nonzero(np.diff(Fxx) <= 0))} non-increasing "
            f"step(s), first at index {int(np.argmax(np.diff(Fxx) <= 0))}")

    if sample_rate is None:
        sample_rate = 2 * Fxx[-1]

    if n_fft is None:
        n_fft = 65536
    elif n_fft < 16:
        warnings.warn(
            f"synthesize_noise_from_psd: n_fft={n_fft} is below the minimum "
            f"16; using the default 65536 instead.",
            UserWarning, stacklevel=2,
        )
        n_fft = 65536
    elif n_fft > MAX_NFFT:
        warnings.warn(
            f"synthesize_noise_from_psd: n_fft={n_fft} above MAX_NFFT={MAX_NFFT}; "
            f"clamping to {MAX_NFFT}.",
            UserWarning, stacklevel=2,
        )
        n_fft = MAX_NFFT
    if not _is_power_of_two(n_fft):
        rounded = _closest_power_of_two(n_fft)
        warnings.warn(
            f"synthesize_noise_from_psd: n_fft={n_fft} is not a power of two; "
            f"rounding to {rounded}.",
            UserWarning, stacklevel=2,
        )
        n_fft = rounded

    # Interior bins of the one-sided grid: an even n_fft has n_fft//2 + 1 rfft
    # bins, of which DC and Nyquist must be real for a real signal. Drawing a
    # complex value there would be wrong, so both are pinned to zero below and
    # only the N = n_fft//2 - 1 interior bins are synthesised.
    N = n_fft // 2 - 1
    dF = sample_rate / n_fft
    f_grid = np.arange(1, N + 1) * dF
    Pxx_grid = _resample_psd(Pxx, Fxx, f_grid, interp)
    # Per-bin variance of each complex draw below. The band power of a
    # one-sided PSD is Pxx*dF, split evenly between +f and -f of the real
    # signal, and w = (vi + i*vq)*sqrt(v) carries E|w|^2 = 2v — hence v =
    # Pxx*dF/4. Synthesising a flat PSD and re-estimating it round-trips to
    # within 0.3 %.
    v = Pxx_grid * dF / 4

    chunk_size = n_fft
    overlap_size = chunk_size // 4
    samples_needed = int(duration * sample_rate)
    if samples_needed < 1:
        raise ConfigurationError(
            "synthesize_noise_from_psd: duration * sample_rate must cover at "
            f"least one sample; got duration={duration}, "
            f"sample_rate={sample_rate}.")
    num_chunks = int(np.ceil(samples_needed / (chunk_size - overlap_size)))

    x_total = np.zeros(samples_needed)
    t_total = np.arange(samples_needed) / sample_rate
    rng = np.random.default_rng() if rng is None else rng

    for i in range(num_chunks):
        vi = rng.standard_normal(N)
        vq = rng.standard_normal(N)
        w = (vi + 1j * vq) * np.sqrt(v)
        spectrum = np.concatenate(([0.0], w, [0.0]))   # DC, interior, Nyquist
        # numpy's irfft carries a 1/n_fft; the *chunk_size undoes it, giving
        # the unnormalised inverse DFT the v = Pxx*dF/4 calibration assumes.
        chunk = np.fft.irfft(spectrum, chunk_size) * chunk_size

        # Sine/cosine crossfade rather than linear: the fade-in sin(pi/2 * u)
        # and fade-out cos(pi/2 * u) are power-complementary (sin^2 + cos^2 =
        # 1), so summing two independent chunks over the overlap reproduces the
        # target variance. A linear fade would dip to half power mid-overlap.
        fade = np.ones(chunk_size)
        if i > 0:
            fade[:overlap_size] = np.sin(
                np.pi / 2 * np.linspace(0, 1, overlap_size))
        if i < num_chunks - 1:
            fade[-overlap_size:] = np.sin(
                np.pi / 2 * np.linspace(1, 0, overlap_size))
        chunk = chunk * fade

        start_idx = i * (chunk_size - overlap_size)
        end_idx = start_idx + chunk_size
        if end_idx > samples_needed:
            chunk = chunk[: samples_needed - start_idx]
            end_idx = samples_needed

        x_total[start_idx:end_idx] += chunk[: end_idx - start_idx] * scale

    return t_total, x_total, float(sample_rate)


def _resample_psd(Pxx, Fxx, f_target, method):
    """Resample a one-sided PSD onto ``f_target``; out-of-range bins → 0."""
    if method == 'linear':
        return np.maximum(
            np.interp(f_target, Fxx, Pxx, left=0.0, right=0.0), 0.0)

    in_range = (f_target >= Fxx[0]) & (f_target <= Fxx[-1])
    out = np.zeros_like(f_target)

    if method == 'log':
        if Fxx[0] <= 0 or np.any(Pxx <= 0):
            raise ConfigurationError(
                "synthesize_noise_from_psd: interp='log' requires strictly "
                f"positive Pxx and Fxx; got Fxx[0]={Fxx[0]:g} and "
                f"{int(np.count_nonzero(Pxx <= 0))} non-positive Pxx value(s) "
                f"(minimum {Pxx.min():g})"
            )
        out[in_range] = 10.0 ** np.interp(
            np.log10(f_target[in_range]),
            np.log10(Fxx), np.log10(Pxx))
        return out

    if method == 'pchip':
        from scipy.interpolate import PchipInterpolator
        out[in_range] = PchipInterpolator(
            Fxx, Pxx, extrapolate=False)(f_target[in_range])
        return np.maximum(np.where(np.isnan(out), 0.0, out), 0.0)

    if method == 'nearest':
        from scipy.interpolate import interp1d
        out[in_range] = interp1d(
            Fxx, Pxx, kind='nearest',
            bounds_error=False, fill_value=0.0)(f_target[in_range])
        return np.maximum(out, 0.0)

    raise ConfigurationError(
        f"synthesize_noise_from_psd: unknown interp={method!r}; "
        "valid: 'linear', 'log', 'pchip', 'nearest'."
    )


def _is_power_of_two(x):
    return x > 0 and x.is_integer() and ((int(x) & (int(x) - 1)) == 0)


def _closest_power_of_two(x):
    """Power of two nearest to ``x`` in log2 (geometric) distance.

    ``round(log2(x))`` splits at the geometric midpoint ``sqrt(2)*2**n``, not
    the arithmetic one ``1.5*2**n``, so values between the two (e.g. 183)
    round *up* even though the lower power is arithmetically closer.
    """
    n = round(math.log2(x))
    return 2 ** n


def make_noise_waveform(
    fc: float,
    bandwidth: float,
    duration: float,
    sample_rate: float,
    *,
    rng=None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate bandpass-filtered Gaussian random noise waveform.

    Creates a noise time series centered at a specified frequency with
    a given bandwidth, useful for noise probes or testing.

    Parameters
    ----------
    fc : float
        Center frequency in Hz
    bandwidth : float
        Bandwidth in Hz
    duration : float
        Duration in seconds
    sample_rate : float
        Sample rate in Hz
    rng : numpy.random.Generator, optional
        Random generator for the white-noise draw. Pass a seeded generator for
        a reproducible realisation.

    Returns
    -------
    time : ndarray
        Sample times (s), ``np.arange(N)/sample_rate``.
    nts : ndarray
        Noise time series, 1-D of length ``int(duration*sample_rate)``.

    The ``(time, signal)`` order is the package-wide convention, shared with
    the tonal generators (``tone_burst``, ``lfm_chirp``, ``hfm_chirp``) and
    the channel/synthesis helpers.

    Notes
    -----
    The algorithm:
    1. Generate Gaussian white noise at bandwidth rate
    2. Resample to sampling rate sample_rate
    3. Heterodyne with carrier frequency fc

    This creates bandpass noise centered at fc with the given bandwidth.

    Translated from OALIB makenoise.m by mbp (27 Sept 2007)

    Examples
    --------
    >>> # Generate 1 kHz noise, 200 Hz bandwidth, 1 second
    >>> t, nts = make_noise_waveform(1000, 200, 1.0, 10000)
    >>> print(f"Noise signal: {len(nts)} samples")
    Noise signal: 10000 samples
    """
    bandwidth = require_positive_finite_scalar(
        bandwidth, "make_noise_waveform", "bandwidth", " Hz")
    duration = require_positive_finite_scalar(
        duration, "make_noise_waveform", "duration", " s")
    sample_rate = require_positive_finite_scalar(
        sample_rate, "make_noise_waveform", "sample_rate", " Hz")
    N = int(duration * sample_rate)  # number of samples
    # Build the time axis from the same N used for resample so the carrier
    # and the resampled noise always have matching length (np.arange(0,
    # duration, 1/sample_rate) can yield N±1 samples from float accumulation).
    time = np.arange(N) / sample_rate
    N2 = int(duration * bandwidth)
    if N < 1 or N2 < 1:
        raise ConfigurationError(
            f"make_noise_waveform: duration*sample_rate ({N}) and "
            f"duration*bandwidth ({N2}) must each resolve to at least one "
            f"sample; got duration={duration}, sample_rate={sample_rate}, "
            f"bandwidth={bandwidth}")

    # The generator side of the Nyquist split: the heterodyne below places
    # the band at fc +/- bandwidth/2, and an edge at or above fs/2 folds
    # back to fs - f, which returns a plausible-looking waveform centred
    # somewhere else entirely. ``make_bandlimited_noise`` in this file refuses
    # the same band through ``_bandpass_design``.
    require_below_nyquist(fc + bandwidth / 2.0, sample_rate,
                          "make_noise_waveform",
                          "the upper band edge fc + bandwidth/2",
                          "the band folds back to sample_rate - f and the "
                          "noise comes out centred somewhere else")
    if not (fc - bandwidth / 2.0 > 0.0):
        raise ConfigurationError(
            f"make_noise_waveform: the lower band edge fc - bandwidth/2 "
            f"({fc - bandwidth / 2.0:g} Hz) must be > 0 Hz; a band "
            f"straddling DC is the mirror of the same fold. Got fc={fc!r}, "
            f"bandwidth={bandwidth!r}.")

    rng = np.random.default_rng() if rng is None else rng
    nts = rng.standard_normal(N2)  # Gaussian white noise

    # Resample to sample_rate rate
    from scipy.signal import resample

    nts = resample(nts, N)

    # Heterodyne with carrier
    nts = np.sin(2 * np.pi * fc * time) * nts
    return time, nts


def _bandpass_design(fc: float, bandwidth: float, sample_rate: float):
    """4th-order Butterworth bandpass, as second-order sections, for
    ``fc +/- bandwidth/2``.

    ``scipy.signal.butter`` requires only ``0 < Wn < 1``, so that is the sole
    constraint applied. A band whose edges fall outside the sample rate is
    refused rather than moved: substituting a different band silently returns
    noise centred somewhere the caller did not ask for, while
    ``_noise_equivalent_bandwidth`` scales the level from the *same* substituted
    design — so the result is internally consistent and nothing downstream
    notices.
    """
    from scipy.signal import butter
    nyquist = sample_rate / 2.0
    flow = fc - bandwidth / 2.0
    fhigh = fc + bandwidth / 2.0
    if not (0.0 < flow < fhigh < nyquist):
        raise ConfigurationError(
            f"band {flow:g}-{fhigh:g} Hz (fc={fc:g}, bandwidth={bandwidth:g}) "
            f"is not realisable at sample_rate={sample_rate:g} Hz; it must sit "
            f"strictly inside 0-{nyquist:g} Hz.",
            remediation="Raise sample_rate, or move fc / narrow bandwidth so "
                        "the whole band fits below Nyquist.",
        )
    # Second-order sections, not transfer-function coefficients: a narrow band
    # at a high sample rate sits at a normalised frequency of order 1e-3, where
    # the (b, a) form loses so much precision that the response collapses. That
    # numerical failure is what the old 0.01/0.02 normalised-frequency clamps
    # were hiding — they kept the design away from the unstable region by
    # silently moving the band.
    return butter(4, [flow / nyquist, fhigh / nyquist], btype='band',
                  output='sos')


def _noise_equivalent_bandwidth(sos, sample_rate: float, fc: float,
                                bandwidth: float, n_freq: int = 8192):
    """One-sided noise-equivalent bandwidth (Hz) of ``sosfiltfilt(sos, ...)``.

    ``sosfiltfilt`` applies the cascade twice, so the power response is ``|H|**4``
    and the equivalent rectangular bandwidth is ``∫|H|**4 df / max|H|**4``. This
    is what a unit-RMS band-limited realisation actually spreads its power over
    — narrower than the nominal -3 dB ``bandwidth``.

    The integration grid is centred on ``fc ± bandwidth/2`` rather than spread
    over the whole ``[0, Nyquist]``: a uniform grid of ``n_freq`` points over
    the full band gives a spacing of ``sample_rate / (2 n_freq)`` (5.9 Hz at
    96 kHz), so a narrow band is sampled by a handful of points and the
    quadrature error is unbounded — once the passband falls between two grid
    points the integral collapses to a single grid step. Measured at
    ``sample_rate=96 kHz``: +5.14 dB at ``bandwidth=2`` Hz, +1.16 dB at 5 Hz,
    -1.45 dB at 10 Hz, and (because the error is set by where the band happens
    to land between grid points) +3.65 dB at 5 Hz for ``fc=12002.9`` Hz where
    the same band at ``fc=12000`` Hz reads +1.16 dB. Ten bandwidths out the
    double-pass 4th-order Butterworth is below 1e-18 of its peak, so nothing
    outside the focused window contributes; bands already wide enough for the
    old grid move by < 1e-13 relative.
    """
    from scipy.signal import sosfreqz
    nyquist = float(sample_rate) / 2.0
    # Ten bandwidths of skirt on each side, clipped to the real axis: for a
    # wide band this reduces to the full [0, Nyquist] the old grid used.
    worN = np.linspace(max(0.0, fc - bandwidth / 2.0 - 10.0 * bandwidth),
                       min(nyquist, fc + bandwidth / 2.0 + 10.0 * bandwidth),
                       int(n_freq))
    f, h = sosfreqz(sos, worN=worN, fs=float(sample_rate))
    p = np.abs(h) ** 4
    return float(np.trapezoid(p, f) / p.max())


def add_noise(
    timeseries: np.ndarray,
    sample_rate: float,
    source_level: float,
    noise_level: float,
    fc: float,
    bandwidth: float,
    *,
    rng=None
) -> np.ndarray:
    """
    Incorporate source level and noise into existing time series.

    The receiver timeseries is assumed to be based on a 0 dB source.
    This function scales it by the source level and adds band-limited noise.

    Parameters
    ----------
    timeseries : ndarray
        Clean receiver time series (normalized to 0 dB source)
        Shape: (n_samples,) or (n_samples, n_receivers)
    sample_rate : float
        Sample rate in Hz
    source_level : float
        Source level in dB (total power)
    noise_level : float
        Noise amplitude in dB (power spectral density, not total power)
    fc : float
        Center frequency for band-limited noise in Hz
    bandwidth : float
        Bandwidth for band-limited noise in Hz
    rng : numpy.random.Generator, optional
        Random generator for the noise realisation(s). Pass a seeded generator
        for a reproducible result.

    Returns
    -------
    ndarray
        Time series with source level and noise incorporated
        Same shape as input timeseries

    Notes
    -----
    The noise is generated as filtered Gaussian random noise with:
    - Center frequency fc
    - Bandwidth BW
    - Power spectral density specified by noise_level

    Total noise power = PSD + 10*log10(BW)

    Examples
    --------
    >>> # Clean signal (0 dB reference); seeded so the example repeats
    >>> rng = np.random.default_rng(0)
    >>> clean_signal = rng.standard_normal(48000)
    >>> clean_signal = clean_signal / np.max(np.abs(clean_signal))
    >>>
    >>> # Add 185 dB source level and 40 dB noise
    >>> noisy = add_noise(clean_signal, 48000, 185.0, 40.0, 10000.0, 10000.0)

    References
    ----------
    Original MATLAB code by mbp, 4/09
    """
    timeseries = np.asarray(timeseries, dtype=float)
    SL = 10.0 ** (source_level / 20.0)

    # Target noise RMS for a one-sided in-band PSD level ``noise_level`` (dB
    # re Pa²/Hz):
    #     P_total = S_target · NEB = 10^(L/10) · NEB   (Pa²)
    #     RMS     = √P_total                            (Pa)
    # ``make_bandlimited_noise`` returns unit-RMS noise spread over the
    # zero-phase filter's noise-equivalent bandwidth NEB (not the nominal -3 dB
    # ``bandwidth``), so multiplying by ``A = RMS`` puts the in-band density at
    # exactly the requested level.
    neb = _noise_equivalent_bandwidth(
        _bandpass_design(fc, bandwidth, sample_rate), sample_rate,
        fc, bandwidth)
    A = np.sqrt(neb * 10.0 ** (noise_level / 10.0))

    # Generate band-limited noise — independent realisation per receiver
    # so cross-channel correlation is zero (required for beamforming and
    # array-gain assertions).
    # The noise is drawn by sample count, not by a duration: routing through
    # seconds and back (`int((n/fs)*fs)`) returns n-1 samples for 4-7 % of
    # lengths at every rate that is not a power of two, and the sum below
    # would then raise a raw numpy broadcast error naming neither argument.
    rng = np.random.default_rng() if rng is None else rng
    n_samples = timeseries.shape[0]

    if timeseries.ndim == 1:
        noise_ts = _bandlimited_noise_samples(
            fc, bandwidth, n_samples, sample_rate, rng=rng,
            caller="add_noise")[1] * A
        rts = timeseries * SL + noise_ts
    else:
        n_rcv = timeseries.shape[1]
        noise_block = np.column_stack([
            _bandlimited_noise_samples(fc, bandwidth, n_samples, sample_rate,
                                       rng=rng, caller="add_noise")[1]
            for _ in range(n_rcv)
        ]) * A
        rts = timeseries * SL + noise_block

    return rts


def make_bandlimited_noise(
    fc: float,
    bandwidth: float,
    duration: float,
    sample_rate: float,
    *,
    rng=None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate band-limited Gaussian noise.

    Creates filtered Gaussian random noise centered at fc with specified bandwidth.

    Parameters
    ----------
    fc : float
        Center frequency in Hz
    bandwidth : float
        Bandwidth in Hz
    duration : float
        Duration in seconds
    sample_rate : float
        Sample rate in Hz
    rng : numpy.random.Generator, optional
        Random generator for the white-noise draw. Pass a seeded generator for
        a reproducible realisation.

    Returns
    -------
    time : ndarray
        Sample times (s), ``np.arange(N)/sample_rate``.
    noise : ndarray
        Band-limited, unit-RMS noise time series, 1-D of length
        ``int(duration*sample_rate)``.

    Notes
    -----
    The noise is generated in the frequency domain and transformed to time domain.
    This ensures precise control over the frequency content.

    Examples
    --------
    >>> t, noise = make_bandlimited_noise(10000.0, 5000.0, 1.0, 48000.0)
    >>> print(f"Generated {len(noise)} samples")
    Generated 48000 samples
    """
    return _bandlimited_noise_samples(
        fc, bandwidth, int(duration * sample_rate), sample_rate,
        rng=rng, caller="make_bandlimited_noise")


def _bandlimited_noise_samples(fc, bandwidth, n_samples, sample_rate, *,
                               rng=None, caller):
    """``(time, noise)`` of exactly ``n_samples`` band-limited unit-RMS samples.

    The sample count is the argument rather than a duration because
    ``int(duration*sample_rate)`` is one short of ``round(duration*sample_rate)``
    for 4-7 % of record lengths at every rate that is not a power of two
    (``int((n/fs)*fs) == n - 1``), and a caller adding this noise to a record
    of its own then hits a raw numpy broadcast error.
    """
    from scipy.signal import sosfiltfilt
    sos = _bandpass_design(fc, bandwidth, sample_rate)
    # sosfiltfilt pads by 3*(2*n_sections+1) - 1 and refuses a shorter signal
    # with a bare ValueError naming only `padlen`.
    padlen = 3 * (2 * len(sos) + 1) - 1
    if n_samples <= padlen:
        raise ConfigurationError(
            f"{caller}: {n_samples} sample(s) is too short for the "
            f"zero-phase bandpass, which pads by {padlen} samples on each "
            f"end; it needs more than {padlen}. Lengthen the record or "
            f"raise the sample rate.")
    time = np.arange(n_samples) / sample_rate
    rng = np.random.default_rng() if rng is None else rng
    noise = rng.standard_normal(n_samples)
    filtered_noise = sosfiltfilt(sos, noise)

    # Normalise to unit RMS so callers can scale by the target RMS
    # directly (e.g. RMS = √(BW · 10^(PSD_dB/10)) for a target one-sided
    # PSD level). Filtfilt's double-pass + Butterworth rolloff make the
    # post-filter variance filter-shape-dependent; unit-RMS removes that.
    rms = float(np.std(filtered_noise))
    if rms > 0:
        filtered_noise = filtered_noise / rms

    return time, filtered_noise


def fourier_synthesis(
    pressure_freq: np.ndarray,
    frequencies: np.ndarray,
    source_spectrum: Optional[np.ndarray] = None,
    Tstart: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fourier synthesis to make time series from frequency-domain transfer function.

    Converts frequency-domain pressure field to time domain using inverse FFT,
    optionally weighted by a source spectrum. Direct translation of AT's
    ``stack.m`` (raw-DFT scaling, output grid = the input frequency grid)
    for working with externally produced spectra; model results should use
    ``Field.synthesize_time_series`` / ``Field.to_time_trace``, which
    handle bin placement, windowing and grid-independent amplitude.

    Parameters
    ----------
    pressure_freq : ndarray
        Frequency-domain pressure field
        Shape: (n_freq, n_depths, n_ranges) or (n_freq, n_receivers)
    frequencies : ndarray
        Frequency vector in Hz
    source_spectrum : ndarray, optional
        Source spectrum (complex) at frequencies in frequencies
        If None, assumes unit spectrum (impulse response)
    Tstart : float, optional
        Starting time offset in seconds (default: 0.0)

    Returns
    -------
    time : ndarray
        Time vector in seconds
    rmod : ndarray
        Time-domain received signal
        Shape matches input pressure_freq with frequency dim converted to time

    Notes
    -----
    The process:
    1. Apply time-shift via phase rotation: exp(i * 2*pi * Tstart * f)
    2. Weight by source spectrum if provided
    3. Inverse FFT to convert to time domain
    4. Scale by 2 and take real part (conjugate symmetry)

    The time sampling is determined by the frequency spacing:
    - deltaf = frequencies[1] - frequencies[0]
    - Tmax = 1 / deltaf
    - deltat = Tmax / Nfreq

    The output is a *baseband* trace when the grid starts above DC. The IFFT
    runs over the supplied bins with bin 0 at DC, so a band starting at
    ``frequencies[0]`` comes back demodulated by ``frequencies[0]`` at a rate
    of ``Nfreq * deltaf``: a Gaussian centred at 150 Hz on a 100-200 Hz grid
    peaks at 50 Hz in the output spectrum. ``stack.m`` leaves its heterodyne
    back to the base frequency commented out, and this translation follows it.
    ``Tstart`` is a time-origin shift only — it rotates the phase and relabels
    the axis, and does not move the spectrum — so for a passband waveform use
    ``Field.synthesize_time_series`` / ``Field.to_time_trace`` instead.

    Examples
    --------
    >>> # Generate frequency-domain transfer function
    >>> freqs = np.linspace(10, 1000, 100)
    >>> rng = np.random.default_rng(0)
    >>> H_freq = (rng.standard_normal((100, 50, 20))
    ...           + 1j * rng.standard_normal((100, 50, 20)))
    >>>
    >>> # Convert to time domain (impulse response)
    >>> t, h_time = fourier_synthesis(H_freq, freqs)
    >>>
    >>> # With source spectrum
    >>> s_hat = np.exp(-(freqs - 500)**2 / (2*100**2))  # Gaussian spectrum
    >>> t, r_time = fourier_synthesis(H_freq, freqs, source_spectrum=s_hat)

    References
    ----------
    Original MATLAB code: stack.m by mbp, 9/96
    Updated 2014 for compatibility with current file formats
    """
    Nfreq = len(frequencies)
    original_shape = pressure_freq.shape

    # Reshape to (Nfreq, -1) for processing. ``.copy()`` so the in-place
    # multiplications below do not mutate the caller's input through the
    # reshape view.
    # astype(complex) rather than .copy(): the in-place phase rotations
    # below assign complex values, and on a real-dtype array numpy keeps
    # only the real part (a ComplexWarning, not an error), silently
    # discarding the rotation.
    if pressure_freq.ndim == 1:
        pressure_work = pressure_freq.reshape(-1, 1).astype(complex)
    else:
        n_receivers = np.prod(original_shape[1:])
        pressure_work = pressure_freq.reshape(
            Nfreq, n_receivers).astype(complex)
    if Tstart != 0.0:
        for irec in range(pressure_work.shape[1]):
            pressure_work[:, irec] = (pressure_work[:, irec] *
                                      np.exp(1j * 2 * np.pi * Tstart * frequencies))
    if len(frequencies) > 0 and frequencies[0] > 0:
        warnings.warn(
            f"fourier_synthesis: frequencies[0]={frequencies[0]:.3g} Hz > 0. "
            "The IFFT places bin 0 at DC, so the returned trace is the "
            "complex envelope demodulated by frequencies[0] — a "
            f"{frequencies[0]:.3g} Hz band start puts a component at f back "
            f"at f-{frequencies[0]:.3g} Hz — sampled at Nfreq*df, not the "
            "passband waveform. This is stack.m's behaviour (its heterodyne "
            "back to the base frequency is commented out) and is what the "
            "raw-DFT route returns; Tstart only moves the time origin and "
            "does not re-modulate. For a passband trace use "
            "Field.synthesize_time_series / Field.to_time_trace.",
            UserWarning, stacklevel=2,
        )

    if source_spectrum is not None:
        for irec in range(pressure_work.shape[1]):
            pressure_work[:, irec] = pressure_work[:, irec] * source_spectrum

    rmod_work = np.fft.ifft(pressure_work, n=Nfreq, axis=0)

    # stack.m zeroes the negative half of the spectrum before this transform,
    # so the conjugate-symmetric partner of every bin is missing: doubling the
    # real part restores the full real signal.
    rmod_work = 2 * np.real(rmod_work)

    if pressure_freq.ndim == 1:
        rmod = rmod_work.flatten()
    else:
        new_shape = (Nfreq,) + original_shape[1:]
        rmod = rmod_work.reshape(new_shape)
    deltaf = frequencies[1] - frequencies[0] if len(frequencies) > 1 else 1.0
    Tmax = 1 / deltaf
    deltat = Tmax / Nfreq
    # Anchor the output time axis at ``Tstart`` so the IFFT trace lines
    # up with absolute travel time when the caller passes r/c0 (or any
    # other origin). Tstart=0.0 (default) reproduces the original
    # source-local axis.
    time = Tstart + np.linspace(0.0, Tmax - deltat, Nfreq)

    return time, rmod
