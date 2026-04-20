"""
Signal processing functions for underwater acoustics.

This module contains signal processing computations including:
- Beamforming algorithms
- Plane wave replica generation
- Delay-and-sum processing
- Noise incorporation
"""

import numpy as np
from typing import Union, Tuple, Optional
from scipy.signal.windows import hann

from uacpy.core.constants import DEFAULT_SOUND_SPEED


def planewave_rep(
    phone_coords: np.ndarray,
    angles: np.ndarray,
    freq: float,
    c: float = DEFAULT_SOUND_SPEED,
    window: bool = False
) -> np.ndarray:
    """
    Generate matrix of plane wave steering vectors.

    Creates a matrix e(angle, phone) of plane wave steering vectors for
    beamforming applications. Angles are relative to broadside.

    Parameters
    ----------
    phone_coords : ndarray
        Array of phone/hydrophone coordinates (positions along array axis)
        Can be row or column vector
    angles : ndarray
        Array of angles in degrees relative to broadside
        Can be row or column vector
    freq : float
        Frequency in Hz
    c : float, optional
        Reference sound speed in m/s for steering vectors (default: 1480.0)
    window : bool, optional
        If True, apply Hanning window to steering vectors (default: False)

    Returns
    -------
    ndarray
        Matrix of steering vectors with shape (n_angles, n_phones)
        Each row is a normalized steering vector for a specific angle

    Notes
    -----
    The steering vectors are computed as:
        e = exp(i * k0 * sin(theta) * phone_coords)
    where k0 = 2*pi*freq/c is the wavenumber

    Each steering vector is normalized such that norm(e) = 1

    Examples
    --------
    >>> # Linear array with 10 elements spaced 0.5m apart
    >>> phones = np.arange(10) * 0.5
    >>> angles_deg = np.arange(-90, 91, 1)
    >>> freq = 1000.0  # 1 kHz
    >>> e = planewave_rep(phones, angles_deg, freq)
    >>> print(f"Steering matrix shape: {e.shape}")
    Steering matrix shape: (181, 10)

    References
    ----------
    Original MATLAB code by mbp, October 1999
    """
    # Ensure angles is a column vector
    angles = np.asarray(angles)
    if angles.ndim == 1:
        angles = angles.reshape(-1, 1)
    elif angles.shape[1] != 1:
        angles = angles.T

    # Ensure phone_coords is a row vector
    phone_coords = np.asarray(phone_coords)
    if phone_coords.ndim == 1:
        phone_coords = phone_coords.reshape(1, -1)
    elif phone_coords.shape[0] != 1:
        phone_coords = phone_coords.T

    theta_rad = np.deg2rad(angles)  # Convert to radians
    Nelts = phone_coords.shape[1]

    # Generate matrix of steering vectors
    omega = 2 * np.pi * freq
    k0 = omega / c
    e = np.exp(1j * k0 * np.sin(theta_rad) @ phone_coords)

    # Window and normalize
    if window:
        window_vec = hann(Nelts).reshape(1, -1)
        e = e * window_vec

    # Normalize each steering vector
    for itheta in range(e.shape[0]):
        e[itheta, :] = e[itheta, :] / np.linalg.norm(e[itheta, :])

    return e


def beamform(
    pressure: np.ndarray,
    phone_coords: np.ndarray,
    freq: float,
    angles: Optional[np.ndarray] = None,
    SL: float = 150.0,
    NL: float = 0.0,
    c: float = DEFAULT_SOUND_SPEED
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple beamformer for underwater acoustic data.

    Performs conventional plane-wave beamforming on pressure field data
    using steering vectors.

    Parameters
    ----------
    pressure : ndarray
        Pressure field data with shape (n_depths, n_ranges) or (n_phones, n_ranges)
        Can be complex-valued
    phone_coords : ndarray
        Hydrophone/receiver depth coordinates
    freq : float
        Frequency in Hz
    angles : ndarray, optional
        Beam angles in degrees relative to broadside (default: -90 to 90 in 1° steps)
    SL : float, optional
        Source level in dB (default: 150.0)
    NL : float, optional
        Noise level in dB (default: 0.0)
    c : float, optional
        Reference sound speed for steering vectors in m/s (default: 1480.0)

    Returns
    -------
    power : ndarray
        Beamformed power in dB with shape (n_angles, n_ranges)
    angles_out : ndarray
        Angles used for beamforming
    peak : float
        Peak power value in dB

    Notes
    -----
    The beamformer computes:
        power = 20 * log10(|e * pressure|) + SL - NL
    where e is the steering vector matrix

    Examples
    --------
    >>> # pressure field shape: (101 depths, 50 ranges)
    >>> pressure = np.random.randn(101, 50) + 1j*np.random.randn(101, 50)
    >>> depths = np.linspace(0, 100, 101)
    >>> freq = 100.0  # 100 Hz
    >>> power, angles, peak = beamform(pressure, depths, freq)
    >>> print(f"Power shape: {power.shape}, Peak: {peak:.1f} dB")

    References
    ----------
    Original MATLAB code by mbp, 2 March 2001
    """
    if angles is None:
        angles = np.arange(-90, 91, 1)

    # Set up replica vectors
    e = planewave_rep(phone_coords, angles, freq, c=c)

    # Compute beamformed output
    # e has shape (n_angles, n_phones)
    # pressure has shape (n_phones, n_ranges)
    beamformed = e @ pressure

    # Calculate power in dB
    power = 20 * np.log10(np.abs(beamformed)) + SL - NL
    peak = np.max(power)

    return power, angles, peak




def add_noise(
    timeseries: np.ndarray,
    sample_rate: float,
    source_level_db: float,
    noise_level_db: float,
    fc: float,
    bandwidth: float
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
    source_level_db : float
        Source level in dB (total power)
    noise_level_db : float
        Noise amplitude in dB (power spectral density, not total power)
    fc : float
        Center frequency for band-limited noise in Hz
    bandwidth : float
        Bandwidth for band-limited noise in Hz

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
    - Power spectral density specified by noise_level_db

    Total noise power = PSD + 10*log10(BW)

    Examples
    --------
    >>> # Clean signal (0 dB reference)
    >>> clean_signal = np.random.randn(48000)
    >>> clean_signal = clean_signal / np.max(np.abs(clean_signal))
    >>>
    >>> # Add 185 dB source level and 40 dB noise
    >>> noisy = add_noise(clean_signal, 48000, 185.0, 40.0, 10000.0, 10000.0)

    References
    ----------
    Original MATLAB code by mbp, 4/09
    """
    # Convert dB to linear scale
    SL = 10.0 ** (source_level_db / 20.0)

    # Calculate noise amplitude
    # A is the amplitude corresponding to the PSD
    # Factor of sqrt(2) accounts for real signal
    # sqrt(bandwidth) converts PSD to total power
    flow = fc - bandwidth / 2
    fhigh = fc + bandwidth / 2
    A = np.sqrt(2) * np.sqrt(fhigh - flow) * 10.0 ** (noise_level_db / 20.0)

    # Generate band-limited noise
    T = len(timeseries) / sample_rate
    noise_ts = make_bandlimited_noise(fc, bandwidth, T, sample_rate) * A

    # Scale signal by source level and add noise
    if timeseries.ndim == 1:
        rts = timeseries * SL + noise_ts
    else:
        # Multiple receivers
        rts = timeseries * SL + noise_ts.reshape(-1, 1)

    return rts


def make_bandlimited_noise(
    fc: float,
    bandwidth: float,
    duration: float,
    sample_rate: float
) -> np.ndarray:
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

    Returns
    -------
    ndarray
        Band-limited noise time series

    Notes
    -----
    The noise is generated in the frequency domain and transformed to time domain.
    This ensures precise control over the frequency content.

    Examples
    --------
    >>> noise = make_bandlimited_noise(10000.0, 5000.0, 1.0, 48000.0)
    >>> print(f"Generated {len(noise)} samples")
    """
    from scipy.signal import butter, filtfilt

    # Generate white Gaussian noise
    n_samples = int(duration * sample_rate)
    noise = np.random.randn(n_samples)

    # Design bandpass filter
    flow = fc - bandwidth / 2
    fhigh = fc + bandwidth / 2

    # Ensure frequencies are valid
    flow = max(flow, 1.0)  # At least 1 Hz
    fhigh = min(fhigh, sample_rate / 2 - 1)  # Below Nyquist

    # Normalize frequencies to Nyquist
    nyquist = sample_rate / 2
    low = flow / nyquist
    high = fhigh / nyquist

    # Ensure normalized frequencies are in valid range (0, 1)
    low = max(min(low, 0.99), 0.01)
    high = max(min(high, 0.99), 0.02)

    # Create bandpass filter
    b, a = butter(4, [low, high], btype='band')

    # Apply filter
    filtered_noise = filtfilt(b, a, noise)

    return filtered_noise


def fourier_synthesis(
    pressure_freq: np.ndarray,
    freq_vec: np.ndarray,
    source_spectrum: Optional[np.ndarray] = None,
    Tstart: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fourier synthesis to make time series from frequency-domain transfer function.

    Converts frequency-domain pressure field to time domain using inverse FFT,
    optionally weighted by a source spectrum.

    Parameters
    ----------
    pressure_freq : ndarray
        Frequency-domain pressure field
        Shape: (n_freq, n_depths, n_ranges) or (n_freq, n_receivers)
    freq_vec : ndarray
        Frequency vector in Hz
    source_spectrum : ndarray, optional
        Source spectrum (complex) at frequencies in freq_vec
        If None, assumes unit spectrum (impulse response)
    Tstart : float, optional
        Starting time offset in seconds (default: 0.0)

    Returns
    -------
    rmod : ndarray
        Time-domain received signal
        Shape matches input pressure_freq with frequency dim converted to time
    time : ndarray
        Time vector in seconds

    Notes
    -----
    The process:
    1. Apply time-shift via phase rotation: exp(i * 2*pi * Tstart * f)
    2. Weight by source spectrum if provided
    3. Inverse FFT to convert to time domain
    4. Scale by 2 and take real part (conjugate symmetry)

    The time sampling is determined by the frequency spacing:
    - deltaf = freq_vec[1] - freq_vec[0]
    - Tmax = 1 / deltaf
    - deltat = Tmax / Nfreq

    Examples
    --------
    >>> # Generate frequency-domain transfer function
    >>> freqs = np.linspace(10, 1000, 100)
    >>> H_freq = np.random.randn(100, 50, 20) + 1j*np.random.randn(100, 50, 20)
    >>>
    >>> # Convert to time domain (impulse response)
    >>> h_time, t = fourier_synthesis(H_freq, freqs)
    >>>
    >>> # With source spectrum
    >>> s_hat = np.exp(-(freqs - 500)**2 / (2*100**2))  # Gaussian spectrum
    >>> r_time, t = fourier_synthesis(H_freq, freqs, source_spectrum=s_hat)

    References
    ----------
    Original MATLAB code: stack.m by mbp, 9/96
    Updated 2014 for compatibility with current file formats
    """
    Nfreq = len(freq_vec)
    original_shape = pressure_freq.shape

    # Reshape to (Nfreq, -1) for processing
    if pressure_freq.ndim == 1:
        pressure_work = pressure_freq.reshape(-1, 1)
    else:
        n_receivers = np.prod(original_shape[1:])
        pressure_work = pressure_freq.reshape(Nfreq, n_receivers)

    # Apply time-shift via phase rotation
    if Tstart != 0.0:
        for irec in range(pressure_work.shape[1]):
            pressure_work[:, irec] = (pressure_work[:, irec] *
                                     np.exp(1j * 2 * np.pi * Tstart * freq_vec))

    # Weight by source spectrum if provided
    if source_spectrum is not None:
        for irec in range(pressure_work.shape[1]):
            pressure_work[:, irec] = pressure_work[:, irec] * source_spectrum

    # Inverse FFT to get time series
    rmod_work = np.fft.ifft(pressure_work, n=Nfreq, axis=0)

    # Since spectrum is conjugate symmetric, result should be real
    # Factor of 2 accounts for negative frequencies being zeroed
    rmod_work = 2 * np.real(rmod_work)

    # Reshape back to original shape (with freq dim → time dim)
    if pressure_freq.ndim == 1:
        rmod = rmod_work.flatten()
    else:
        new_shape = (Nfreq,) + original_shape[1:]
        rmod = rmod_work.reshape(new_shape)

    # Set up time vector based on FFT sampling rules
    deltaf = freq_vec[1] - freq_vec[0] if len(freq_vec) > 1 else 1.0
    Tmax = 1 / deltaf
    deltat = Tmax / Nfreq
    time = np.linspace(0.0, Tmax - deltat, Nfreq)

    return rmod, time


