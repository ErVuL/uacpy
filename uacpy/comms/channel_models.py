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

import warnings

import numpy as np

from uacpy.acoustic_signal.channel import impulse_response
from uacpy.core.exceptions import ConfigurationError
from uacpy.core._warn_frames import USER_FRAME_SKIP

# Retained DFT bins below which a band-limited tap process has too few degrees
# of freedom for its envelope to be Rayleigh.
_MIN_DOPPLER_BINS = 8


def awgn(signal, snr_db, *, rng=None):
    """Add complex (or real) AWGN for a target in-band SNR (dB).

    A zero-power signal is returned unchanged with a UserWarning: the SNR
    target scales the noise power off the signal power, so zero signal
    power means zero noise power at any ``snr_db``.

    Parameters
    ----------
    signal : array_like
        Clean signal, real or complex. A complex signal is given ``n0/2`` in
        each quadrature, so ``snr_db`` means the same thing for a
        complex-baseband and a real-passband input.
    snr_db : float
        Target in-band signal-to-noise ratio in dB, referred to the mean
        power of ``signal``.
    rng : numpy.random.Generator, optional
        Random generator for the noise realisation. Pass a seeded generator
        for a reproducible result.

    Returns
    -------
    ndarray
        ``signal`` plus the noise realisation, same shape and dtype class.
    """
    x = np.asarray(signal)
    rng = np.random.default_rng() if rng is None else rng
    p = np.mean(np.abs(x) ** 2)
    if p == 0:
        warnings.warn(
            "awgn: the signal has zero power, so the noise power that "
            "realises the requested SNR is zero and the signal is returned "
            "unchanged.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
    n0 = p / (10.0 ** (float(snr_db) / 10.0))
    if np.iscomplexobj(x):
        # n0 is the *total* noise power, split n0/2 into each quadrature so
        # E|noise|^2 = n0 and the requested SNR means the same thing for a
        # complex-baseband and a real-passband signal.
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
    taps = np.asarray(h)
    if taps.size == 0:
        raise ConfigurationError(
            "apply_channel: empty channel h (0 taps); pass at least one tap "
            "— h=[1.0] is the identity channel.")
    return np.convolve(np.asarray(signal), taps)


def fading_taps(n_taps, n_samples, doppler_hz, sample_rate, *, rician_k=0.0,
                rng=None):
    """Time-varying complex tap gains ``H`` of shape ``(n_taps, n_samples)``.

    Each tap is a band-limited complex-Gaussian process (max Doppler
    ``doppler_hz``) — Rayleigh fading; ``rician_k > 0`` adds a line-of-sight
    component with the given Rice K-factor (linear). Unit average power over
    the ensemble, not per realisation. ``doppler_hz = 0`` is the static limit:
    only the DC bin survives, so each tap is a single complex-Gaussian draw
    held constant over the block; a negative ``doppler_hz`` raises
    :class:`~uacpy.core.exceptions.ConfigurationError`.

    The Doppler band is resolved to ``sample_rate/n_samples``, so it spans
    ``2·doppler_hz·n_samples/sample_rate`` DFT bins. That bin count is the
    process's degrees of freedom: below a handful the block is too short to
    realise the requested spread and the envelope is not Rayleigh, which
    warns.

    Parameters
    ----------
    n_taps : int
        Number of channel taps, one row of ``H`` each.
    n_samples : int
        Block length in samples, one column of ``H`` each. It also sets the
        Doppler resolution, ``sample_rate/n_samples``.
    doppler_hz : float
        Maximum Doppler shift in Hz — the half-width of the flat Doppler
        band. ``0`` is the static limit; negative raises
        :class:`~uacpy.core.exceptions.ConfigurationError`.
    sample_rate : float
        Sample rate in Hz.
    rician_k : float, optional
        Rice K-factor, linear (not dB), as the ratio of line-of-sight power
        to diffuse power. Defaults to ``0.0``, which is pure Rayleigh.
    rng : numpy.random.Generator, optional
        Random generator for the tap draw. Pass a seeded generator for a
        reproducible result.

    Returns
    -------
    ndarray
        Complex tap gains of shape ``(n_taps, n_samples)``.
    """
    rng = np.random.default_rng() if rng is None else rng
    fs = float(sample_rate)
    n = int(n_samples)
    # Written as the negation of the admissible condition so NaN is refused
    # too: `doppler_hz < 0` is False for NaN, which then made `|f| <= nan`
    # all-False and every tap NaN. `isfinite` is the other half — `|f| <= inf`
    # is all-True, which disables the Doppler low-pass entirely and returns
    # unfiltered white noise that looks like a plausible fading process.
    if not (np.isfinite(doppler_hz) and doppler_hz >= 0):
        raise ConfigurationError(
            f"fading_taps: doppler_hz must be >= 0 Hz and finite; got "
            f"{doppler_hz!r} (0 is the static limit — a constant complex gain "
            f"per tap).")
    # White complex Gaussian per tap, low-pass filtered to the Doppler
    # bandwidth. doppler_hz = 0 keeps the DC bin alone, so each tap is one
    # complex-Gaussian draw held constant over the block.
    H = (rng.standard_normal((n_taps, n)) + 1j * rng.standard_normal((n_taps, n)))
    f = np.fft.fftfreq(n, d=1.0 / fs)
    mask = (np.abs(f) <= float(doppler_hz)).astype(float)
    n_bins = int(mask.sum())
    H = np.fft.ifft(np.fft.fft(H, axis=1) * mask[None, :], axis=1)
    if doppler_hz > 0 and n_bins < _MIN_DOPPLER_BINS:
        warnings.warn(
            f"fading_taps: doppler_hz={float(doppler_hz):g} spans {n_bins} DFT "
            f"bin(s) at sample_rate/n_samples={fs / n:g} Hz resolution, so the "
            f"tap process is drawn from fewer than {_MIN_DOPPLER_BINS} degrees "
            f"of freedom and its envelope statistics are not Rayleigh. Lengthen "
            f"n_samples to at least {int(np.ceil(_MIN_DOPPLER_BINS * fs / (2.0 * float(doppler_hz))))} "
            f"samples, or raise doppler_hz.",
            UserWarning, stacklevel=2)
    # Normalise by the filter's analytic gain, not per realisation: dividing each
    # tap by its own RMS pins |H| to a constant when few bins survive, which is a
    # unit-modulus phase, not a fading process.
    H /= np.sqrt(2.0 * n_bins / n)
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
        raise ConfigurationError(
            f"apply_fading_channel: taps rows must match delays — taps has "
            f"{taps.shape[0]} rows for {d.size} delays. fading_taps("
            f"n_taps={d.size}, ...) sizes them to match.")
    if taps.shape[1] < n:
        raise ConfigurationError(
            f"apply_fading_channel: taps shorter than signal — taps carries "
            f"{taps.shape[1]} columns for a {n}-sample signal. Call "
            f"fading_taps(..., n_samples={n}) to cover it.")
    if d.size and d.min() < 0:
        # y[di:di+n] indexes from the END for a negative di, so a negative
        # delay either raises a bare broadcast ValueError or — when
        # di < -n <= -len(y) — quietly places the echo near the tail of y
        # (delays [10, -8] on a 6-sample input put it at samples 8-13).
        raise ConfigurationError(
            f"apply_fading_channel: delays_samples must be >= 0; got "
            f"{d.min()}. A tap cannot arrive before the signal.")
    y = np.zeros(n + int(d.max()), dtype=complex)
    for i, di in enumerate(d):
        y[di:di + n] += taps[i, :n] * x
    return y
