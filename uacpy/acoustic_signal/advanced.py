"""
Advanced signal processing functions for underwater acoustics.

This module provides advanced signal processing tools including:
- Baseband/passband conversion
- Numerically controlled oscillators (NCO)
- Goertzel algorithm for single-frequency DFT
- Periodic correlation
- Zero-phase filtering
- Time vector generation

-------------------------------------------------------------------------------
Portions of this file are adapted from arlpy (https://github.com/org-arl/arlpy)
Copyright (c) 2016-2020, Acoustic Research Laboratory
All rights reserved.

Redistributed under the terms of the 3-clause BSD license.  The full
license text, including the required disclaimer and no-endorsement clause,
is reproduced in:

    uacpy/third_party/arlpy/LICENSE

See uacpy/third_party/arlpy/NOTICE for the list of arlpy-adapted functions
in this file.
-------------------------------------------------------------------------------
"""

import numpy as np
import scipy.signal as sp
from typing import Union, Optional, Callable, Tuple, Generator


def time(n: Union[int, np.ndarray], fs: float) -> np.ndarray:
    """
    Generate a time vector for time series.

    Parameters
    ----------
    n : int or array_like
        Number of samples, or the time series itself
    fs : float
        Sampling rate in Hz

    Returns
    -------
    ndarray
        Time vector starting at time 0

    Examples
    --------
    >>> import numpy as np
    >>> t = time(100000, fs=250000)
    >>> print(f"Duration: {t[-1]:.4f} seconds")

    >>> # Or pass the signal directly
    >>> x = np.random.randn(1000)
    >>> t = time(x, fs=250000)
    """
    if hasattr(n, "__len__"):
        n = np.asarray(n).shape[0]
    return np.arange(n, dtype=np.float64) / fs


def cw(
    fc: float,
    duration: float,
    fs: float,
    window: Optional[Union[str, tuple]] = None,
    complex_output: bool = False
) -> np.ndarray:
    """
    Generate a continuous wave (sinusoidal) pulse.

    Parameters
    ----------
    fc : float
        Frequency of the pulse in Hz
    duration : float
        Duration of the pulse in seconds
    fs : float
        Sampling rate in Hz
    window : str or tuple, optional
        Window function to apply (None = rectangular)
        For supported windows, see scipy.signal.get_window
    complex_output : bool, optional
        If True, return complex signal (default: False)

    Returns
    -------
    ndarray
        Generated CW pulse (real or complex)

    Examples
    --------
    >>> x1 = cw(fc=27000, duration=0.5, fs=250000)
    >>> x2 = cw(fc=27000, duration=0.5, fs=250000, window='hamming')
    >>> x3 = cw(fc=27000, duration=0.5, fs=250000, window=('kaiser', 4.0))
    >>> x4 = cw(fc=27000, duration=0.5, fs=250000, complex_output=True)
    """
    n = int(round(duration * fs))
    t = time(n, fs)

    if complex_output:
        x = np.exp(2j * np.pi * fc * t)
    else:
        x = np.sin(2 * np.pi * fc * t)

    if window is not None:
        w = sp.get_window(window, n, False)
        x *= w

    return x


def sweep(
    f1: float,
    f2: float,
    duration: float,
    fs: float,
    method: str = 'linear',
    window: Optional[Union[str, tuple]] = None
) -> np.ndarray:
    """
    Generate frequency modulated sweep (chirp).

    Parameters
    ----------
    f1 : float
        Starting frequency in Hz
    f2 : float
        Ending frequency in Hz
    duration : float
        Duration of the pulse in seconds
    fs : float
        Sampling rate in Hz
    method : str, optional
        Type of sweep: 'linear', 'quadratic', 'logarithmic', 'hyperbolic'
        (default: 'linear')
    window : str or tuple, optional
        Window function to apply (None = rectangular)

    Returns
    -------
    ndarray
        Generated sweep signal

    Examples
    --------
    >>> x1 = sweep(20000, 30000, duration=0.5, fs=250000)
    >>> x2 = sweep(20000, 30000, duration=0.5, fs=250000, window='hamming')
    >>> x3 = sweep(20000, 30000, duration=0.5, fs=250000, method='logarithmic')
    """
    n = int(round(duration * fs))
    t = time(n, fs)
    x = sp.chirp(t, f1, duration, f2, method)

    if window is not None:
        w = sp.get_window(window, n, False)
        x *= w

    return x


def envelope(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Generate a Hilbert envelope of the real signal x.

    Parameters
    ----------
    x : ndarray
        Real passband signal
    axis : int, optional
        Axis along which to compute envelope (default: -1)

    Returns
    -------
    ndarray
        Envelope of the signal

    Examples
    --------
    >>> x = cw(1000, 0.1, 10000) * np.hanning(1000)
    >>> env = envelope(x)
    """
    from scipy.signal import hilbert
    return np.abs(hilbert(x, axis=axis))


def bb2pb(
    x: np.ndarray,
    fd: float,
    fc: float,
    fs: Optional[float] = None,
    axis: int = -1
) -> np.ndarray:
    """
    Convert baseband signal to passband.

    Parameters
    ----------
    x : ndarray
        Complex baseband signal
    fd : float
        Sampling rate of baseband signal in Hz
    fc : float
        Carrier frequency in Hz
    fs : float, optional
        Sampling rate of passband signal in Hz (default: None = same as fd)
    axis : int, optional
        Axis along which to process (default: -1)

    Returns
    -------
    ndarray
        Real passband signal

    Examples
    --------
    >>> # Generate baseband signal
    >>> x_bb = np.exp(2j * np.pi * 1000 * np.arange(1000) / 50000)
    >>> # Convert to passband
    >>> x_pb = bb2pb(x_bb, fd=50000, fc=27000, fs=250000)
    """
    if fs is None:
        fs = fd

    # Resample if necessary
    if fs != fd:
        x = sp.resample(x, int(x.shape[axis] * fs / fd), axis=axis)

    # Generate carrier
    n = x.shape[axis]
    t = time(n, fs)
    carrier = np.exp(2j * np.pi * fc * t)

    # Upconvert
    if axis != -1:
        carrier = np.expand_dims(carrier, axis=tuple(i for i in range(x.ndim) if i != axis))

    y = x * carrier
    return np.real(y)


def pb2bb(
    x: np.ndarray,
    fs: float,
    fc: float,
    fd: Optional[float] = None,
    flen: int = 127,
    cutoff: Optional[float] = None,
    axis: int = -1
) -> np.ndarray:
    """
    Convert passband signal to baseband.

    Parameters
    ----------
    x : ndarray
        Real passband signal
    fs : float
        Sampling rate of passband signal in Hz
    fc : float
        Carrier frequency in Hz
    fd : float, optional
        Sampling rate of baseband signal in Hz (default: None = same as fs)
    flen : int, optional
        Length of lowpass filter (default: 127)
    cutoff : float, optional
        Cutoff frequency for lowpass filter in Hz (default: None = fd/4)
    axis : int, optional
        Axis along which to process (default: -1)

    Returns
    -------
    ndarray
        Complex baseband signal

    Examples
    --------
    >>> # Convert passband to baseband
    >>> x_bb = pb2bb(x_pb, fs=250000, fc=27000, fd=50000)
    """
    if fd is None:
        fd = fs

    if cutoff is None:
        cutoff = fd / 4

    # Generate carrier
    n = x.shape[axis]
    t = time(n, fs)
    carrier = np.exp(-2j * np.pi * fc * t)

    # Downconvert
    if axis != -1:
        carrier = np.expand_dims(carrier, axis=tuple(i for i in range(x.ndim) if i != axis))

    y = x * carrier

    # Lowpass filter
    h = sp.firwin(flen, cutoff, fs=fs)
    y = sp.lfilter(h, 1, y, axis=axis)

    # Resample if necessary
    if fs != fd:
        y = sp.resample(y, int(y.shape[axis] * fd / fs), axis=axis)

    return y


def lfilter0(b: np.ndarray, a: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Filter signal with zero-phase filtering.

    Filters signal using forward-backward filtering to achieve zero phase distortion.

    Parameters
    ----------
    b : ndarray
        Numerator coefficients of filter
    a : ndarray
        Denominator coefficients of filter
    x : ndarray
        Input signal
    axis : int, optional
        Axis along which to filter (default: -1)

    Returns
    -------
    ndarray
        Filtered signal with zero phase

    Examples
    --------
    >>> # Design lowpass filter
    >>> b = sp.firwin(51, 0.3)
    >>> # Apply zero-phase filtering
    >>> y = lfilter0(b, 1, x)
    """
    return sp.filtfilt(b, a, x, axis=axis)


def correlate_periodic(a: np.ndarray, v: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute periodic correlation using FFT.

    Fast computation of periodic correlation, useful for correlation with
    periodic sequences like m-sequences.

    Parameters
    ----------
    a : ndarray
        First signal
    v : ndarray, optional
        Second signal (default: None = autocorrelation)

    Returns
    -------
    ndarray
        Periodic correlation

    Examples
    --------
    >>> # Autocorrelation
    >>> x = np.random.randn(1000)
    >>> r = correlate_periodic(x)

    >>> # Cross-correlation
    >>> y = np.random.randn(1000)
    >>> r = correlate_periodic(x, y)
    """
    if v is None:
        v = a

    a = np.asarray(a)
    v = np.asarray(v)

    if len(a) != len(v):
        raise ValueError('Signals must have same length for periodic correlation')

    # Use FFT for fast computation
    n = len(a)
    A = np.fft.fft(a, n)
    V = np.fft.fft(v, n)
    return np.real(np.fft.ifft(A * np.conj(V)))


def goertzel(
    f: Union[float, np.ndarray],
    x: np.ndarray,
    fs: float = 2.0,
    filter: bool = False
) -> Union[complex, np.ndarray]:
    """
    Compute single-frequency DFT using Goertzel algorithm.

    More efficient than FFT when only a few frequency bins are needed.

    Parameters
    ----------
    f : float or array_like
        Frequency or frequencies in Hz
    x : ndarray
        Input signal
    fs : float, optional
        Sampling rate in Hz (default: 2.0)
    filter : bool, optional
        If True, return filtered output for each sample (default: False)

    Returns
    -------
    complex or ndarray
        DFT coefficient(s) at specified frequency/frequencies
        If filter=True, returns time-series output

    Examples
    --------
    >>> # Single frequency
    >>> x = cw(1000, 1.0, 10000)
    >>> X = goertzel(1000, x, fs=10000)
    >>> print(f"Magnitude: {abs(X):.2f}")

    >>> # Multiple frequencies
    >>> freqs = [500, 1000, 1500]
    >>> X = goertzel(freqs, x, fs=10000)

    References
    ----------
    Goertzel, G. (1958). "An Algorithm for the Evaluation of Finite Trigonometric
    Series". The American Mathematical Monthly, 65(1), 34-35.
    """
    f = np.atleast_1d(f)
    k = np.round(f * len(x) / fs).astype(int)
    w = 2 * np.pi * k / len(x)

    if filter:
        # Return filtered output for each sample
        result = np.zeros((len(f), len(x)), dtype=complex)
        for i, wi in enumerate(w):
            coeff = 2 * np.cos(wi)
            s0 = 0
            s1 = 0
            exp_factor = np.exp(-1j * wi)

            for j, xj in enumerate(x):
                s0, s1 = xj + coeff * s0 - s1, s0
                result[i, j] = s0 - exp_factor * s1

        return result.squeeze()
    else:
        # Return single DFT coefficient per frequency
        result = np.zeros(len(f), dtype=complex)
        for i, wi in enumerate(w):
            coeff = 2 * np.cos(wi)
            s0 = 0
            s1 = 0

            for xj in x:
                s0, s1 = xj + coeff * s0 - s1, s0

            result[i] = s0 - np.exp(-1j * wi) * s1

        return result.squeeze()


def nco(
    fc: Union[float, np.ndarray],
    fs: float = 2.0,
    phase0: float = 0,
    wrap: float = 2 * np.pi,
    func: Optional[Callable] = None
) -> np.ndarray:
    """
    Generate numerically controlled oscillator (NCO) output.

    Parameters
    ----------
    fc : float or array_like
        Frequency in Hz (can be time-varying)
    fs : float, optional
        Sampling rate in Hz (default: 2.0)
    phase0 : float, optional
        Initial phase in radians (default: 0)
    wrap : float, optional
        Phase wrap value in radians (default: 2*pi)
    func : callable, optional
        Function to apply to phase (default: lambda x: np.exp(1j*x))

    Returns
    -------
    ndarray
        NCO output (complex by default)

    Examples
    --------
    >>> # Constant frequency NCO
    >>> y = nco(1000, fs=10000)

    >>> # Frequency-modulated NCO
    >>> fc = 1000 + 500 * np.sin(2*np.pi*10*np.arange(10000)/10000)
    >>> y = nco(fc, fs=10000)

    >>> # Real-valued output
    >>> y = nco(1000, fs=10000, func=np.cos)
    """
    if func is None:
        func = lambda x: np.exp(1j * x)

    fc = np.atleast_1d(fc)

    # Compute phase
    if len(fc) == 1:
        # Constant frequency
        n = 1000  # Default length for constant frequency
        phase = phase0 + 2 * np.pi * fc[0] * np.arange(n) / fs
    else:
        # Time-varying frequency
        phase = phase0 + 2 * np.pi * np.cumsum(fc) / fs

    # Wrap phase
    if wrap is not None:
        phase = np.mod(phase, wrap)

    return func(phase)


def nco_gen(
    fc: float,
    fs: float = 2.0,
    phase0: float = 0,
    wrap: float = 2 * np.pi,
    func: Optional[Callable] = None
) -> Generator[Union[complex, float], None, None]:
    """
    Create a numerically controlled oscillator (NCO) generator.

    This generator version is useful for real-time or streaming applications.

    Parameters
    ----------
    fc : float
        Frequency in Hz
    fs : float, optional
        Sampling rate in Hz (default: 2.0)
    phase0 : float, optional
        Initial phase in radians (default: 0)
    wrap : float, optional
        Phase wrap value in radians (default: 2*pi)
    func : callable, optional
        Function to apply to phase (default: lambda x: np.exp(1j*x))

    Yields
    ------
    complex or float
        NCO output sample

    Examples
    --------
    >>> gen = nco_gen(1000, fs=10000)
    >>> samples = [next(gen) for _ in range(100)]
    """
    if func is None:
        func = lambda x: np.exp(1j * x)

    phase = phase0
    dphi = 2 * np.pi * fc / fs

    while True:
        yield func(phase)
        phase += dphi
        if wrap is not None:
            phase = np.mod(phase, wrap)


def lfilter_gen(b: np.ndarray, a: np.ndarray) -> Callable:
    """
    Create a generator-based filter for streaming data.

    Parameters
    ----------
    b : ndarray
        Numerator coefficients
    a : ndarray
        Denominator coefficients

    Returns
    -------
    callable
        Generator function that can be called to create filter instances

    Examples
    --------
    >>> # Create filter generator
    >>> b, a = sp.butter(4, 0.3)
    >>> filt = lfilter_gen(b, a)()

    >>> # Process streaming data
    >>> for sample in data_stream:
    >>>     filtered_sample = filt.send(sample)
    """
    def _lfilter_gen():
        # Initialize filter state
        zi = np.zeros(max(len(b), len(a)) - 1)

        y = yield None  # Prime the generator

        while True:
            # Filter single sample
            y, zi = sp.lfilter(b, a, [y], zi=zi)
            y = yield y[0]

    return _lfilter_gen


def mseq(spec: Union[int, list], n: Optional[int] = None) -> np.ndarray:
    """
    Generate m-sequence (maximum length sequence).

    Parameters
    ----------
    spec : int or list
        If int: degree of polynomial (generates default m-sequence)
        If list: feedback tap specification
    n : int, optional
        Length of sequence (default: None = full period)

    Returns
    -------
    ndarray
        Binary m-sequence (+1, -1)

    Examples
    --------
    >>> # Generate degree-5 m-sequence
    >>> m = mseq(5)
    >>> print(f"Length: {len(m)}")

    >>> # Generate with custom taps
    >>> m = mseq([5, 2], n=100)

    References
    ----------
    Golomb, S. W. (1967). Shift Register Sequences.
    """
    if isinstance(spec, int):
        # Default feedback taps for common degrees
        default_taps = {
            2: [2, 1],
            3: [3, 1],
            4: [4, 1],
            5: [5, 2],
            6: [6, 1],
            7: [7, 1],
            8: [8, 4, 3, 2],
            9: [9, 4],
            10: [10, 3],
        }
        if spec not in default_taps:
            raise ValueError(f'No default taps for degree {spec}')
        spec = default_taps[spec]

    degree = max(spec)
    taps = spec

    # Initialize shift register
    state = np.ones(degree, dtype=int)
    max_length = 2**degree - 1

    if n is None:
        n = max_length

    # Generate sequence
    seq = np.zeros(n, dtype=int)
    for i in range(n):
        seq[i] = state[-1]
        feedback = sum(state[t-1] for t in taps) % 2
        state = np.roll(state, 1)
        state[0] = feedback

    # Convert to +1/-1
    return 2 * seq - 1


def resample(
    data: np.ndarray,
    up_factor: int,
    down_factor: int
) -> np.ndarray:
    """
    Resample signal by rational factor.

    Parameters
    ----------
    data : ndarray
        Input signal
    up_factor : int
        Upsampling factor
    down_factor : int
        Downsampling factor

    Returns
    -------
    ndarray
        Resampled signal

    Examples
    --------
    >>> # Upsample by 3
    >>> y = resample(x, 3, 1)

    >>> # Downsample by 2
    >>> y = resample(x, 1, 2)

    >>> # Resample by 3/2
    >>> y = resample(x, 3, 2)
    """
    return sp.resample_poly(data, up_factor, down_factor)
