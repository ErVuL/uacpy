"""Deterministic source waveforms: SPARC pulses, Ricker/Gaussian pulses, LFM/HFM chirps, tone bursts and N-waves."""

from typing import Tuple, Literal

import numpy as np

from uacpy.core.exceptions import ConfigurationError


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
        Single letter code indicating pulse type:
        - 'P': Pseudo gaussian (peak at 0, support [0, 3F])
        - 'R': Ricker wavelet (peak at F, support [0, 2F])
        - 'A': Approximate Ricker wavelet (peak at F, support [0, 2.5F])
        - 'S': Single sine (peak at F, support [0, F])
        - 'H': Hanning weighted four sine (peak at F, support [0, 4F])
        - 'N': N-wave (peak at F, support [0, F])
        - 'M': Miracle wave (peak at 0, support [0, infinity])
        - 'G': Gaussian (peak at 0, support [0, infinity])
        - 'T': Tone burst / gated sinewave (peak at F, support [0, 0.4s])
        - 'C': Sinc function (uniform spectrum [0, F])
        - 'E': One-sided exponential

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
    Original MATLAB code by mbp, based on 1988 Fortran version from SPARC
    """
    t = np.asarray(t)
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
    Peak occurs at ``frequency``, with support approximately [0, 2·frequency].

    Examples
    --------
    >>> time = np.linspace(0, 0.1, 1000)
    >>> s = ricker_wavelet(time, 50.0)

    References
    ----------
    Original MATLAB code: Ricker.m
    """
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

    Examples
    --------
    >>> # Generate 1-second chirp from 100 to 1000 Hz
    >>> s, t = lfm_chirp(100, 1000, 1.0, 10000)

    >>> # Can also use scipy.signal.chirp for similar functionality
    >>> from scipy.signal import chirp
    >>> t = np.linspace(0, 1, 10000)
    >>> s_scipy = chirp(t, 100, 1, 1000)

    References
    ----------
    Original MATLAB code: lfm.m
    """
    T = duration  # local alias for the sweep-duration symbol in the phase law
    N = int(T * sample_rate)
    if N <= 0:
        return np.array([]), np.array([])
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

    Examples
    --------
    >>> # Generate 5-cycle 1000 Hz tone burst
    >>> s, t = tone_burst(1000.0, 5, 48000)

    >>> # Without windowing
    >>> s_rect, t = tone_burst(1000.0, 5, 48000, window=False)
    """
    from scipy.signal.windows import hann

    # ``T`` is the requested burst duration in seconds; the sample count
    # ``N`` is the *nearest* integer that keeps ``n_cycles`` faithful at
    # the given ``sample_rate``. ``time`` is built as
    # ``np.arange(N) / sample_rate`` so ``dt == 1 / sample_rate`` exactly.
    if frequency <= 0:
        raise ConfigurationError(
            f"tone_burst: frequency must be > 0; got {frequency}.")
    T = n_cycles / frequency
    N = int(round(T * sample_rate))
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

    The phase is: φ(t) = (2π/b) * log(1 + b*t/P1)
    where b = (fmin - fmax)/(fmin*fmax*T) and P1 = 1/fmin

    Examples
    --------
    >>> s, t = hfm_chirp(1000, 5000, 0.1, 48000)

    References
    ----------
    Original MATLAB: hfm.m
    """
    T = duration  # local alias for the sweep-duration symbol in the phase law
    N = int(T * sample_rate)
    if N <= 0:
        return np.array([]), np.array([])
    deltat = T / N
    time = np.linspace(0.0, T - deltat, N)

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

    Peak frequency is at F, with support from 0 to 4F
    (or [0, 3F] also acceptable).

    Translated from OALIB Nwave.m

    Examples
    --------
    >>> # Generate 100 Hz N-wave
    >>> t = np.linspace(-0.01, 0.02, 1000)
    >>> s = nwave(t, 100.0)
    >>> print(f"Non-zero samples: {np.sum(s != 0)}")
    """
    omega = 2 * np.pi * frequency
    s = np.sin(omega * time) - 0.5 * np.sin(2 * omega * time)

    # Zero outside [0, 1/frequency]
    s[(time > 1 / frequency) | (time < 0)] = 0

    return s
