"""Communication-channel models: AWGN, static multipath, and time-varying fading.

uacpy's propagation models give the *deterministic* channel impulse response;
this module adds the *stochastic / time-varying* channel needed to stress-test a
receiver — Rayleigh/Rician fading tap-delay lines with a Doppler spectrum, the
single dominant impairment of mobile underwater links (Istepanian & Stojanovic).

References
----------
Proakis & Salehi. *Digital Communications* (fading channels, Jakes spectrum).
Stojanovic, in Istepanian & Stojanovic. *Underwater Acoustic DSP & Comms*
    (time-varying multipath, doubly-spread channels).
"""

from __future__ import annotations

import numpy as np

from uacpy.core.exceptions import ConfigurationError


def awgn(signal, snr_db, *, rng=None):
    """Add complex (or real) AWGN for a target in-band SNR (dB)."""
    x = np.asarray(signal)
    rng = np.random.default_rng() if rng is None else rng
    p = np.mean(np.abs(x) ** 2)
    n0 = p / (10.0 ** (float(snr_db) / 10.0))
    if np.iscomplexobj(x):
        noise = np.sqrt(n0 / 2) * (rng.standard_normal(x.shape)
                                   + 1j * rng.standard_normal(x.shape))
    else:
        noise = np.sqrt(n0) * rng.standard_normal(x.shape)
    return x + noise


def multipath_channel(delays_s, gains, sample_rate):
    """Static FIR impulse response from sparse arrivals ``(delay, gain)``.

    Returns the complex tap vector ``h`` (nearest-sample placement). ``gains``
    may be complex (carry per-path phase).
    """
    d = np.asarray(delays_s, dtype=float)
    g = np.asarray(gains)
    if d.shape != g.shape:
        raise ConfigurationError("multipath_channel: delays and gains shape mismatch")
    idx = np.round(d * float(sample_rate)).astype(int)
    if np.any(idx < 0):
        raise ConfigurationError("multipath_channel: delays must be >= 0")
    h = np.zeros(int(idx.max()) + 1, dtype=complex)
    for i, gg in zip(idx, g):
        h[i] += gg
    return h


def apply_channel(signal, h):
    """Convolve a signal with a static channel ``h`` (returns full convolution)."""
    return np.convolve(np.asarray(signal), np.asarray(h))


def fading_taps(n_taps, n_samples, doppler_hz, sample_rate, *, rician_k=0.0,
                rng=None):
    """Time-varying complex tap gains ``H`` of shape ``(n_taps, n_samples)``.

    Each tap is a band-limited complex-Gaussian process (max Doppler
    ``doppler_hz``) — Rayleigh fading; ``rician_k > 0`` adds a line-of-sight
    component with the given Rice K-factor (linear). Per-tap unit average power.
    """
    rng = np.random.default_rng() if rng is None else rng
    fs = float(sample_rate)
    n = int(n_samples)
    # white complex Gaussian per tap, low-pass filtered to the Doppler bandwidth
    H = (rng.standard_normal((n_taps, n)) + 1j * rng.standard_normal((n_taps, n)))
    if doppler_hz > 0:
        f = np.fft.fftfreq(n, d=1.0 / fs)
        mask = (np.abs(f) <= float(doppler_hz)).astype(float)
        H = np.fft.ifft(np.fft.fft(H, axis=1) * mask[None, :], axis=1)
    # normalise each tap to unit average power
    H /= np.sqrt(np.mean(np.abs(H) ** 2, axis=1, keepdims=True))
    if rician_k > 0:
        k = float(rician_k)
        H = np.sqrt(k / (k + 1)) + np.sqrt(1 / (k + 1)) * H
    return H


def apply_fading_channel(signal, taps, delays_samples):
    """Apply a time-varying channel: ``y[n] = sum_i taps[i,n] * x[n - d_i]``.

    ``taps`` is ``(n_taps, >=len(signal))``; ``delays_samples`` the integer tap
    delays. Returns ``y`` of length ``len(signal) + max(delay)``.
    """
    x = np.asarray(signal, dtype=complex)
    n = x.size
    d = np.asarray(delays_samples, dtype=int)
    taps = np.asarray(taps, dtype=complex)
    if taps.shape[0] != d.size:
        raise ConfigurationError("apply_fading_channel: taps rows must match delays")
    if taps.shape[1] < n:
        raise ConfigurationError("apply_fading_channel: taps shorter than signal")
    y = np.zeros(n + int(d.max()), dtype=complex)
    for i, di in enumerate(d):
        y[di:di + n] += taps[i, :n] * x
    return y


def plot_channel(h, sample_rate, title="", axes=None):
    """Two-panel channel view: |h[n]| (delay) and |H(f)| (frequency response).

    Returns ``(fig, axes)``.
    """
    import matplotlib.pyplot as plt
    h = np.asarray(h, dtype=complex)
    fs = float(sample_rate)
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    else:
        fig = axes[0].figure
    t = np.arange(h.size) / fs * 1e3
    axes[0].stem(t, np.abs(h))
    axes[0].set_xlabel("Delay [ms]"); axes[0].set_ylabel("|h|")
    axes[0].set_title(f"[channel] impulse response {title}", loc="left")
    axes[0].grid(alpha=0.3)
    nfft = max(1024, 2 * h.size)
    f = np.fft.rfftfreq(nfft, d=1.0 / fs)
    H = 20 * np.log10(np.abs(np.fft.rfft(h, nfft)) + 1e-12)
    axes[1].plot(f, H)
    axes[1].set_xlabel("Frequency [Hz]"); axes[1].set_ylabel("|H(f)| [dB]")
    axes[1].set_title("frequency response", loc="left"); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    return fig, axes
