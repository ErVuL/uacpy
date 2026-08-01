"""Communication-channel models: AWGN, static multipath, and time-varying fading.

uacpy's propagation models give the *deterministic* channel impulse response;
this module adds the *stochastic / time-varying* channel needed to stress-test a
receiver — band-limited (flat Doppler PSD) Rayleigh/Rician fading tap-delay
lines, the single dominant impairment of mobile underwater links
(Istepanian & Stojanovic).

References
----------
Proakis & Salehi. *Digital Communications* (fading channels).
Stojanovic, in Istepanian & Stojanovic. *Underwater Acoustic DSP & Comms*
    (time-varying multipath, doubly-spread channels).
"""

from __future__ import annotations

import numpy as np

from uacpy.acoustic_signal.channel import impulse_response
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


def multipath_channel(gains, delays_s, sample_rate, *, fractional=False):
    """Static FIR tap vector from sparse arrivals ``(gain, delay)``.

    Adapter over :func:`uacpy.acoustic_signal.impulse_response` — same
    arguments, returning only the complex tap vector that :func:`apply_channel`
    and the equalizers consume. ``gains`` may be complex (carry per-path
    phase). A tap-delay line is an integer-tap FIR, hence ``fractional=False``
    by default.
    """
    _, h = impulse_response(gains, delays_s, sample_rate, fractional=fractional)
    return h.astype(complex)


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
    """Apply a time-varying channel: ``y[n] = sum_i taps[i, n-d_i] * x[n-d_i]``.

    Each tap's gain is sampled at the **input** time of the sample it delays
    (``taps[i, m]`` multiplies ``x[m]``), which is why ``taps`` only needs
    ``len(signal)`` columns. The textbook tap-delay line samples the gain at
    the output time instead (``taps[i, n]·x[n-d_i]``); the two differ only
    by a per-tap gain shift of ``d_i`` samples and are statistically
    indistinguishable when Doppler × delay spread ≪ 1 — always true for a
    physical underwater channel.

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
