"""
Waveform generation functions for underwater acoustics.

This module contains functions for generating various acoustic pulse types
commonly used in underwater acoustic applications including:
- Standard pulse shapes (Ricker, Gaussian, LFM, etc.)
- Chirps and frequency-modulated signals
- Test signals
"""

import numpy as np
from typing import Tuple, Literal
import warnings
import math


def ssrp(Pxx, Fxx, duration=1, scale=1, *,
         n_fft=65536, fs=None, interp='linear'):
    """
    Spectral Synthesis of Random Processes.

    Generate a time-domain noise realisation whose one-sided PSD matches a
    user-supplied target ``Pxx(Fxx)``. The target is resampled onto the
    FFT-native frequency grid ``f_k = k * fs / n_fft`` before synthesis,
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
        IFFT chunk size (must be even, ≥ 4). Defaults to 65536.
    fs : int, optional
        Output sample rate in Hz. Defaults to 2*Fxx[-1]..
    interp : {'linear', 'log', 'pchip', 'nearest'}, optional
        How to resample ``Pxx(Fxx)`` onto the FFT-native grid. ``'log'``
        interpolates ``log10(Pxx)`` vs ``log10(f)`` — recommended for
        broadband PSDs spanning many decades. Frequencies outside
        ``[Fxx[0], Fxx[-1]]`` are set to zero.

    Returns
    -------
    t : ndarray
        Time array in seconds.
    x : ndarray
        Generated signal array.
    fs : int
        Sampling frequency in Hz.

    Examples
    --------
    >>> import numpy as np
    >>> f = np.logspace(0, 4, 64)
    >>> Pxx = 1e-6 / (1 + (f / 100) ** 2)
    >>> t, x, fs = ssrp(Pxx, f, duration=10,
    ...                 n_fft=2**16, fs=40_000, interp='log')
    """
    MAX_NFFT = 262144

    Pxx = np.asarray(Pxx, dtype=float)
    Fxx = np.asarray(Fxx, dtype=float)
    if Pxx.ndim != 1 or Fxx.shape != Pxx.shape:
        raise ValueError(
            f"ssrp: Pxx and Fxx must be 1-D arrays of equal length; "
            f"got Pxx.shape={Pxx.shape} and Fxx.shape={Fxx.shape}"
        )
    if Pxx.size < 2:
        raise ValueError(
            f"ssrp: Pxx must have at least 2 points (got {Pxx.size})"
        )
    if not np.all(np.diff(Fxx) > 0):
        raise ValueError("ssrp: Fxx must be strictly increasing")

    if fs is None:
        fs = 2 * Fxx[-1]

    if n_fft is None:
        n_fft = 65536
    elif n_fft < 16:
        warnings.warn(
            f"ssrp: n_fft={n_fft} below minimum 16; raising to 65536.",
            UserWarning, stacklevel=2,
        )
        n_fft = 65536
    elif n_fft > MAX_NFFT:
        warnings.warn(
            f"ssrp: n_fft={n_fft} above MAX_NFFT={MAX_NFFT}; "
            f"clamping to {MAX_NFFT}.",
            UserWarning, stacklevel=2,
        )
        n_fft = MAX_NFFT
    if not _is_power_of_two(n_fft):
        rounded = _closest_power_of_two(n_fft)
        warnings.warn(
            f"ssrp: n_fft={n_fft} is not a power of two; "
            f"rounding to {rounded}.",
            UserWarning, stacklevel=2,
        )
        n_fft = rounded

    N = n_fft // 2 - 1
    dF = fs / n_fft
    f_grid = np.arange(1, N + 1) * dF
    Pxx_grid = _resample_psd(Pxx, Fxx, f_grid, interp)
    v = Pxx_grid * dF / 4

    chunk_size = n_fft
    overlap_size = chunk_size // 4
    samples_needed = int(duration * fs)
    num_chunks = int(np.ceil(samples_needed / (chunk_size - overlap_size)))

    x_total = np.zeros(samples_needed)
    t_total = np.arange(samples_needed) / fs

    for i in range(num_chunks):
        vi = np.random.randn(N)
        vq = np.random.randn(N)
        w = (vi + 1j * vq) * np.sqrt(v)
        spectrum = np.concatenate(([0.0], w, [0.0]))
        chunk = np.fft.irfft(spectrum, chunk_size) * chunk_size

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

    return t_total, x_total, int(fs)


def _resample_psd(Pxx, Fxx, f_target, method):
    """Resample a one-sided PSD onto ``f_target``; out-of-range bins → 0."""
    if method == 'linear':
        return np.maximum(
            np.interp(f_target, Fxx, Pxx, left=0.0, right=0.0), 0.0)

    in_range = (f_target >= Fxx[0]) & (f_target <= Fxx[-1])
    out = np.zeros_like(f_target)

    if method == 'log':
        if Fxx[0] <= 0 or np.any(Pxx <= 0):
            raise ValueError(
                "ssrp: interp='log' requires strictly positive Pxx and Fxx"
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

    raise ValueError(
        f"ssrp: unknown interp={method!r}; "
        "valid: 'linear', 'log', 'pchip', 'nearest'."
    )


def _is_power_of_two(x):
    return x > 0 and x.is_integer() and ((int(x) & (int(x) - 1)) == 0)


def _closest_power_of_two(x):
    n = round(math.log2(x))
    return 2 ** n


def cans(
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
    >>> s, title = cans(t, omega, 'R')
    >>> print(title)
    Ricker wavelet

    >>> # Generate pseudo-Gaussian pulse
    >>> s_gauss, _ = cans(t, omega, 'P')

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
        raise ValueError(
            f"Unknown pulse type: '{pulse_type}'. "
            "Valid types: P, R, A, S, H, N, M, G, T, C, E"
        )

    return s, pulse_title


def ricker_wavelet(time: np.ndarray, F: float) -> np.ndarray:
    """
    Generate a Ricker wavelet (Mexican hat wavelet).

    The Ricker wavelet is the second derivative of a Gaussian and is
    commonly used in seismic and acoustic applications.

    Parameters
    ----------
    time : ndarray
        Time vector
    F : float
        Nominal source frequency in Hz

    Returns
    -------
    s : ndarray
        Ricker wavelet time series

    Notes
    -----
    Peak occurs at frequency F, with support approximately [0, 2F].

    Examples
    --------
    >>> time = np.linspace(0, 0.1, 1000)
    >>> s = ricker_wavelet(time, 50.0)

    References
    ----------
    Original MATLAB code: Ricker.m
    """
    u = 2 * np.pi * F * time - 8  # Dimensionless time
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
    fmin: float, fmax: float, T: float, sample_rate: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a Linear Frequency Modulated (LFM) pulse (chirp).

    Creates a signal that sweeps linearly from fmin to fmax over duration T.

    Parameters
    ----------
    fmin : float
        Minimum frequency in Hz
    fmax : float
        Maximum frequency in Hz
    T : float
        Duration of time-series in seconds
    sample_rate : float
        Samples per second (Hz)

    Returns
    -------
    s : ndarray
        LFM signal
    time : ndarray
        Time vector

    Notes
    -----
    The instantaneous frequency is:
        f(t) = fmin + (fmax - fmin) * t / (2*T)

    This is a standard chirp used in sonar and radar applications.

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
    N = int(T * sample_rate)
    deltat = T / N
    time = np.linspace(0.0, T - deltat, N)

    # Instantaneous frequency
    f_inst = fmin + (fmax - fmin) * time / (2 * T)
    s = np.sin(2.0 * np.pi * f_inst * time)

    return s, time


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
    s : ndarray
        Tone burst signal
    time : ndarray
        Time vector

    Examples
    --------
    >>> # Generate 5-cycle 1000 Hz tone burst
    >>> s, t = tone_burst(1000.0, 5, 48000)

    >>> # Without windowing
    >>> s_rect, t = tone_burst(1000.0, 5, 48000, window=False)
    """
    from scipy.signal.windows import hann

    T = n_cycles / frequency
    N = int(T * sample_rate)
    time = np.linspace(0, T, N)

    s = np.sin(2 * np.pi * frequency * time)

    if window:
        s = s * hann(N)

    return s, time


def hfm_chirp(
    fmin: float, fmax: float, T: float, sample_rate: float
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
    T : float
        Duration in seconds
    sample_rate : float
        Sample rate in Hz

    Returns
    -------
    s : ndarray
        HFM signal
    time : ndarray
        Time vector

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
    N = int(T * sample_rate)
    deltat = T / N
    time = np.linspace(0.0, T - deltat, N)

    b = (fmin - fmax) / (fmin * fmax * T)
    P1 = 1 / fmin
    s = np.sin((2 * np.pi / b) * np.log(1 + b * time / P1))

    return s, time


def bpsk_modulate(
    s_bipolar: np.ndarray, fc: float, fs: float, chips_per_sec: float
) -> np.ndarray:
    """
    Encode binary sequence as Binary Phase Shift Keying (BPSK) signal.

    Parameters
    ----------
    s_bipolar : ndarray
        Binary source sequence (+1/-1 values)
    fc : float
        Carrier frequency in Hz
    fs : float
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

    Examples
    --------
    >>> # Binary sequence
    >>> bits = np.array([1, -1, 1, 1, -1, 1])
    >>>
    >>> # BPSK modulation
    >>> fc = 12000  # 12 kHz carrier
    >>> fs = 48000  # 48 kHz sample rate
    >>> chips_per_sec = 3000  # 3k chips/sec
    >>> s = bpsk_modulate(bits, fc, fs, chips_per_sec)

    References
    ----------
    Original MATLAB code by Michael B. Porter, April 2000
    """
    samples_per_chip = int(fs / chips_per_sec)

    if fs / chips_per_sec != samples_per_chip:
        raise ValueError("samples_per_chip must be an integer")

    deltat = 1 / fs
    t_chip = np.arange(0, samples_per_chip * deltat, deltat)
    sinwave = np.sin(2 * np.pi * fc * t_chip)

    # Outer product: each column is one chip
    s_matrix = np.outer(sinwave, s_bipolar)

    # Flatten to 1D signal
    s = s_matrix.flatten(order="F")

    return s


def nwave(time: np.ndarray, F: float) -> np.ndarray:
    """
    Generate an N-wave pulse.

    An N-wave is a characteristic waveform shape consisting of a sin wave
    minus half of its second harmonic, creating an N-shaped pulse.

    Parameters
    ----------
    time : ndarray
        Time vector
    F : float
        Nominal source frequency in Hz

    Returns
    -------
    s : ndarray
        N-wave signal, zero outside [0, 1/F]

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
    omega = 2 * np.pi * F
    s = np.sin(omega * time) - 0.5 * np.sin(2 * omega * time)

    # Zero outside [0, 1/F]
    s[(time > 1 / F) | (time < 0)] = 0

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
    - Nearly flat autocorrelation (ideal for matched filtering)
    - Balanced +1/-1 symbols

    Formulas from Proakis, Digital Communications

    Translated from OALIB mseq.m by Michael B. Porter

    Examples
    --------
    >>> # Generate m-sequence of order 5
    >>> s = mseq(5)
    >>> print(f"Length: {len(s)} (should be 2^5-1 = 31)")

    >>> # Check autocorrelation
    >>> shat = np.fft.fft(s)
    >>> scorr = np.real(np.fft.ifft(shat * np.conj(shat)))
    """
    if m < 2 or m > 15 or m != int(m):
        raise ValueError("m must be an integer between 2 and 15")

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

    # Successive shifts with feedback (Proakis p. 433)
    seed = np.zeros(m)
    seed[0] = 1  # All zero except first element
    s = np.zeros(length)

    for ii in range(length):
        out = np.zeros(m)
        out[: m - 1] = seed[1:m]
        out[m - 1] = np.mod(np.dot(c, seed), 2)  # Addition mod 2
        seed = out
        s[ii] = out[0]

    # Convert 0/1 to -1/+1
    s[s == 0] = -1

    return s


def make_mseq_probe(fmin: float, fmax: float, fs: float, T_tot: float) -> np.ndarray:
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
    fs : float
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

    Chip rate is (fmax - fmin) / 2. Output is normalized to 0.95 of
    full scale.

    Translated from OALIB makemseq.m by mbp.

    Examples
    --------
    >>> # Generate 10-second probe, 1-2 kHz
    >>> probe = make_mseq_probe(1000, 2000, 10000, 10.0)
    >>> print(f"Probe length: {len(probe)} samples")
    """
    lead_time = 0.2  # seconds

    # M-sequence parameters
    fc = 0.5 * (fmin + fmax)  # center frequency
    chips_per_sec = 0.5 * (fmax - fmin)

    # Generate base m-sequence (order 10 → length 1023)
    s_m = mseq(10)
    s = bpsk_modulate(s_m, fc, fs, chips_per_sec)

    # Repeat m-sequence to fill time
    Nreps = int(np.floor(T_tot * chips_per_sec / len(s_m)))
    if Nreps < 1:
        Nreps = 1
    probe = np.tile(s, Nreps)
    leader = np.zeros(int(lead_time * fs))
    probe_max = np.max(np.abs(probe))
    if probe_max > 0:
        probe = np.concatenate([leader, 0.95 * probe / probe_max])
    else:
        probe = np.concatenate([leader, probe])

    # Zero-fill to total duration
    n = len(probe)
    if n < T_tot * fs:
        probe = np.concatenate([probe, np.zeros(int(T_tot * fs - n))])

    return probe


def make_noise_waveform(fc: float, BW: float, T: float, fs: float) -> np.ndarray:
    """
    Generate bandpass-filtered Gaussian random noise waveform.

    Creates a noise time series centered at a specified frequency with
    a given bandwidth, useful for noise probes or testing.

    Parameters
    ----------
    fc : float
        Center frequency in Hz
    BW : float
        Bandwidth in Hz
    T : float
        Duration in seconds
    fs : float
        Sample rate in Hz

    Returns
    -------
    nts : ndarray
        Noise time series (column vector)

    Notes
    -----
    The algorithm:
    1. Generate Gaussian white noise at bandwidth rate
    2. Resample to sampling rate fs
    3. Heterodyne with carrier frequency fc

    This creates bandpass noise centered at fc with bandwidth BW.

    Translated from OALIB makenoise.m by mbp (27 Sept 2007)

    Examples
    --------
    >>> # Generate 1 kHz noise, 200 Hz bandwidth, 1 second
    >>> nts = make_noise_waveform(1000, 200, 1.0, 10000)
    >>> print(f"Noise signal: {len(nts)} samples")
    """
    deltat = 1 / fs
    N = int(T / deltat)  # number of samples
    time = np.arange(0, T, deltat).reshape(-1, 1)  # column vector
    deltat2 = 1 / BW
    N2 = int(T / deltat2)

    nts = np.random.randn(N2, 1)  # Gaussian white noise

    # Resample to fs rate
    from scipy.signal import resample

    nts = resample(nts.flatten(), N).reshape(-1, 1)

    # Heterodyne with carrier
    nts = np.sin(2 * np.pi * fc * time) * nts

    return nts


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("Testing waveform generation functions")
    print("=" * 70)

    # Test 1: CANS pulses
    print("\n1. Testing cans() - various pulse types")
    print("-" * 70)

    t = np.linspace(-0.05, 0.15, 2000)
    f0 = 100.0  # Hz
    omega = 2 * np.pi * f0

    pulse_types = ["P", "R", "G", "S", "N"]
    for ptype in pulse_types:
        s, title = cans(t, omega, ptype)
        peak_val = np.max(np.abs(s))
        peak_idx = np.argmax(np.abs(s))
        peak_time = t[peak_idx]
        print(f"  {ptype}: {title:30s} - peak {peak_val:.3f} at t={peak_time:.4f}s")

    # Test 2: Ricker wavelet
    print("\n2. Testing ricker_wavelet()")
    print("-" * 70)

    time = np.linspace(0, 0.1, 1000)
    f_ricker = 50.0
    s_ricker = ricker_wavelet(time, f_ricker)

    print(f"  Frequency: {f_ricker} Hz")
    print(f"  Duration: {time[-1]} s")
    print(f"  Peak amplitude: {np.max(np.abs(s_ricker)):.4f}")
    print(f"  Peak time: {time[np.argmax(np.abs(s_ricker))]:.4f} s")

    # Test 3: Gaussian pulse
    print("\n3. Testing gaussian_pulse()")
    print("-" * 70)

    time_gauss = np.linspace(0, 1, 1000)
    delay = 0.5
    duration = 0.05
    s_gauss = gaussian_pulse(time_gauss, delay, duration)

    print(f"  Delay: {delay} s")
    print(f"  Duration: {duration} s")
    print(f"  Peak amplitude: {np.max(s_gauss):.4f}")
    print(f"  Peak location: {time_gauss[np.argmax(s_gauss)]:.4f} s")

    # Test 4: LFM chirp
    print("\n4. Testing lfm_chirp()")
    print("-" * 70)

    fmin, fmax = 100.0, 1000.0
    T_chirp = 0.5
    fs = 10000
    s_lfm, t_lfm = lfm_chirp(fmin, fmax, T_chirp, fs)

    print(f"  Frequency sweep: {fmin} to {fmax} Hz")
    print(f"  Duration: {T_chirp} s")
    print(f"  Sample rate: {fs} Hz")
    print(f"  Samples generated: {len(s_lfm)}")

    # Test 5: Tone burst
    print("\n5. Testing tone_burst()")
    print("-" * 70)

    freq_tone = 1000.0
    n_cycles = 10
    s_tone, t_tone = tone_burst(freq_tone, n_cycles, 48000, window=True)

    print(f"  Frequency: {freq_tone} Hz")
    print(f"  Cycles: {n_cycles}")
    print(f"  Duration: {t_tone[-1]:.4f} s")
    print(f"  Peak amplitude: {np.max(np.abs(s_tone)):.4f}")
    print("\n" + "=" * 70)
    print("Creating visualization...")

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle("Waveform Generation Examples", fontsize=14, fontweight="bold")

    # Plot 1: Ricker wavelet
    axes[0, 0].plot(time, s_ricker, "b-", linewidth=1.5)
    axes[0, 0].set_title("Ricker Wavelet (50 Hz)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Gaussian pulse
    axes[0, 1].plot(time_gauss, s_gauss, "r-", linewidth=1.5)
    axes[0, 1].set_title("Gaussian Pulse")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Pseudo-Gaussian from cans
    s_pg, _ = cans(t, omega, "P")
    axes[1, 0].plot(t, s_pg, "g-", linewidth=1.5)
    axes[1, 0].set_title("Pseudo-Gaussian (100 Hz)")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: N-wave
    s_nwave, _ = cans(t, omega, "N")
    axes[1, 1].plot(t, s_nwave, "m-", linewidth=1.5)
    axes[1, 1].set_title("N-Wave (100 Hz)")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Amplitude")
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 5: LFM Chirp
    axes[2, 0].plot(t_lfm[:1000], s_lfm[:1000], "c-", linewidth=1)
    axes[2, 0].set_title(f"LFM Chirp ({fmin}-{fmax} Hz)")
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 0].set_ylabel("Amplitude")
    axes[2, 0].grid(True, alpha=0.3)

    # Plot 6: Tone burst
    axes[2, 1].plot(t_tone, s_tone, "orange", linewidth=1.5)
    axes[2, 1].set_title(f"Tone Burst ({freq_tone} Hz, {n_cycles} cycles)")
    axes[2, 1].set_xlabel("Time (s)")
    axes[2, 1].set_ylabel("Amplitude")
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("waveforms_test.png", dpi=100, bbox_inches="tight")
    print("Saved figure: waveforms_test.png")

    print("\n" + "=" * 70)
    print("All waveform tests completed successfully!")
