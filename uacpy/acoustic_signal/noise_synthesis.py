"""Stochastic signal synthesis: PSD-to-time-series noise realisation, band-limited noise, SNR mixing and Fourier synthesis."""

import math
import warnings
from typing import Tuple, Optional

import numpy as np

from uacpy.core.exceptions import ConfigurationError


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
        IFFT chunk size (must be even, ≥ 4). Defaults to 65536.
    sample_rate : int, optional
        Output sample rate in Hz. Defaults to 2*Fxx[-1]..
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
    sample_rate : int
        Sampling frequency in Hz.

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
        raise ConfigurationError("synthesize_noise_from_psd: Fxx must be strictly increasing")

    if sample_rate is None:
        sample_rate = 2 * Fxx[-1]

    if n_fft is None:
        n_fft = 65536
    elif n_fft < 16:
        warnings.warn(
            f"synthesize_noise_from_psd: n_fft={n_fft} below minimum 16; raising to 65536.",
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

    return t_total, x_total, int(sample_rate)


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
                "synthesize_noise_from_psd: interp='log' requires strictly positive Pxx and Fxx"
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
    n = round(math.log2(x))
    return 2 ** n


def make_noise_waveform(
    fc: float, bandwidth_hz: float, T: float, sample_rate: float, *, rng=None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate bandpass-filtered Gaussian random noise waveform.

    Creates a noise time series centered at a specified frequency with
    a given bandwidth, useful for noise probes or testing.

    Parameters
    ----------
    fc : float
        Center frequency in Hz
    bandwidth_hz : float
        Bandwidth in Hz
    T : float
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
        Noise time series, 1-D of length ``int(T*sample_rate)``.

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
    """
    N = int(T * sample_rate)  # number of samples
    # Build the time axis from the same N used for resample so the carrier
    # and the resampled noise always have matching length (np.arange(0, T,
    # 1/sample_rate) can yield N±1 samples from float accumulation).
    time = np.arange(N) / sample_rate
    N2 = int(T * bandwidth_hz)
    if N < 1 or N2 < 1:
        raise ConfigurationError(
            f"make_noise_waveform: T*sample_rate ({N}) and T*bandwidth_hz "
            f"({N2}) must each resolve to at least one sample; got T={T}, "
            f"sample_rate={sample_rate}, bandwidth_hz={bandwidth_hz}")

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


def _noise_equivalent_bandwidth(sos, sample_rate: float, n_freq: int = 8192):
    """One-sided noise-equivalent bandwidth (Hz) of ``sosfiltfilt(sos, ...)``.

    ``sosfiltfilt`` applies the cascade twice, so the power response is ``|H|**4``
    and the equivalent rectangular bandwidth is ``∫|H|**4 df / max|H|**4``. This
    is what a unit-RMS band-limited realisation actually spreads its power over
    — narrower than the nominal -3 dB ``bandwidth``.
    """
    from scipy.signal import sosfreqz
    f, h = sosfreqz(sos, worN=int(n_freq), fs=float(sample_rate))
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
        _bandpass_design(fc, bandwidth, sample_rate), sample_rate)
    A = np.sqrt(neb * 10.0 ** (noise_level / 10.0))

    # Generate band-limited noise — independent realisation per receiver
    # so cross-channel correlation is zero (required for beamforming and
    # array-gain assertions).
    T = len(timeseries) / sample_rate
    rng = np.random.default_rng() if rng is None else rng

    if timeseries.ndim == 1:
        noise_ts = make_bandlimited_noise(
            fc, bandwidth, T, sample_rate, rng=rng)[1] * A
        rts = timeseries * SL + noise_ts
    else:
        n_rcv = timeseries.shape[1]
        noise_block = np.column_stack([
            make_bandlimited_noise(fc, bandwidth, T, sample_rate, rng=rng)[1]
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
    """
    from scipy.signal import sosfiltfilt
    n_samples = int(duration * sample_rate)
    time = np.arange(n_samples) / sample_rate
    rng = np.random.default_rng() if rng is None else rng
    noise = rng.standard_normal(n_samples)

    sos = _bandpass_design(fc, bandwidth, sample_rate)
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

    Examples
    --------
    >>> # Generate frequency-domain transfer function
    >>> freqs = np.linspace(10, 1000, 100)
    >>> H_freq = np.random.randn(100, 50, 20) + 1j*np.random.randn(100, 50, 20)
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
    if pressure_freq.ndim == 1:
        pressure_work = pressure_freq.reshape(-1, 1).copy()
    else:
        n_receivers = np.prod(original_shape[1:])
        pressure_work = pressure_freq.reshape(Nfreq, n_receivers).copy()
    if Tstart != 0.0:
        for irec in range(pressure_work.shape[1]):
            pressure_work[:, irec] = (pressure_work[:, irec] *
                                      np.exp(1j * 2 * np.pi * Tstart * frequencies))
    elif len(frequencies) > 0 and frequencies[0] > 0:
        warnings.warn(
            f"fourier_synthesis: frequencies[0]={frequencies[0]:.3g} Hz > 0 with "
            "Tstart=0. The IFFT treats input as starting from DC, which "
            "introduces a phase ramp in the synthesised time series. Pass "
            "Tstart matching the physical arrival time (e.g. r/c0) to align "
            "the trace with travel time.",
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
