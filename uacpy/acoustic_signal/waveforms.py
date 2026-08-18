"""Deterministic source waveforms: SPARC pulses, Ricker/Gaussian pulses, LFM/HFM chirps, tone bursts and N-waves."""

from typing import Tuple, Literal

import numpy as np

from uacpy.core.exceptions import ConfigurationError


def _require_positive(caller: str, **params: float) -> None:
    """Raise :class:`ConfigurationError` naming each parameter that is not
    strictly positive (NaN counts as invalid)."""
    for name, value in params.items():
        if not value > 0:
            raise ConfigurationError(
                f"{caller}: {name} must be > 0; got {value}.")


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
    _require_positive("sparc_pulse", omega=omega)
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


def ricker_wavelet(time: np.ndarray, frequency: float) -> np.ndarray:
    """
    Generate a Ricker wavelet (Mexican hat wavelet).

    The Ricker wavelet is the second derivative of a Gaussian and is
    commonly used in seismic and acoustic applications. Uses the AT
    ``Ricker.m`` centring ``u = 2πFt − 8``; SPARC's internal Ricker
    (``cans.f90``, documented in ``models/sparc.py``) centres at
    ``ωT − 5`` — the two "Ricker" pulses are offset in time.

    Parameters
    ----------
    time : ndarray
        Time vector
    frequency : float
        Nominal source frequency in Hz

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
    _require_positive("ricker_wavelet", frequency=frequency)
    time = np.asarray(time, dtype=float)
    u = 2 * np.pi * frequency * time - 8  # Dimensionless time
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
    _require_positive("gaussian_pulse", duration=duration)
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
        Minimum frequency in Hz
    fmax : float
        Maximum frequency in Hz
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
    positive and long enough for at least one sample, otherwise a
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
    _require_positive("lfm_chirp", duration=duration, sample_rate=sample_rate)
    T = duration  # local alias for the sweep-duration symbol in the phase law
    N = int(T * sample_rate)
    if N <= 0:
        raise ConfigurationError(
            "lfm_chirp: duration * sample_rate must cover at least one sample; "
            f"got duration={duration}, sample_rate={sample_rate}.")
    deltat = T / N
    time = np.linspace(0.0, T - deltat, N)

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
    All parameters must be positive and the burst long enough for at least
    one sample (``round(n_cycles / frequency * sample_rate) >= 1``),
    otherwise a :class:`~uacpy.core.exceptions.ConfigurationError` is raised.

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
    _require_positive("tone_burst", frequency=frequency, n_cycles=n_cycles,
                      sample_rate=sample_rate)
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
    divides by ``fmin - fmax`` and by ``fmin``); ``fmin > fmax`` is accepted
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
    _require_positive("hfm_chirp", fmin=fmin, fmax=fmax, duration=duration,
                      sample_rate=sample_rate)
    if fmin == fmax:
        raise ConfigurationError(
            "hfm_chirp: fmin and fmax must differ (the hyperbolic phase law "
            f"divides by fmin - fmax); got fmin == fmax == {fmin}. For a "
            "constant-frequency signal use tone_burst or lfm_chirp with "
            "fmin == fmax.")
    T = duration  # local alias for the sweep-duration symbol in the phase law
    N = int(T * sample_rate)
    if N <= 0:
        raise ConfigurationError(
            "hfm_chirp: duration * sample_rate must cover at least one sample; "
            f"got duration={duration}, sample_rate={sample_rate}.")
    deltat = T / N
    time = np.linspace(0.0, T - deltat, N)

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
    """
    _require_positive("nwave", frequency=frequency)
    time = np.asarray(time, dtype=float)
    omega = 2 * np.pi * frequency
    s = np.sin(omega * time) - 0.5 * np.sin(2 * omega * time)

    # Zero outside [0, 1/frequency]
    s[(time > 1 / frequency) | (time < 0)] = 0

    return s
