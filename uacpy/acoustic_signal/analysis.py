"""Spectral and level estimators (pure functions): PSD, PPSD (Welch
density-scaled) and SEL (band-integrated sound exposure level). dB references
default to 1 uPa. Plotting lives in :mod:`uacpy.visualization.plots.signal`.
"""

import math
from collections import namedtuple

import numpy as np
import scipy.signal as _sig

from uacpy.core.exceptions import ConfigurationError
from uacpy.core.constants import REFERENCE_PRESSURE_WATER
from uacpy.core.acoustics import power_to_db


PPSDResult = namedtuple(
    "PPSDResult",
    "frequencies levels pdf mean_db std_db binwidth_db seg_duration")


def psd(data, fs, *, window="hann", nperseg=8192, noverlap=4096, nfft=None,
        scaling="density", ref=REFERENCE_PRESSURE_WATER):
    """Welch power spectral density. Returns ``(freqs, psd_linear)`` in Pa²/Hz."""
    freqs, Pxx = _sig.welch(data, fs, window=window, nperseg=nperseg,
                            noverlap=noverlap, nfft=nfft, scaling=scaling)
    return freqs, Pxx


def ppsd(data, fs, *, seg_duration=1.0, overlap_pct=50, ddB=1.0, lvlmin=0,
         lvlmax=150, window="hann", nperseg=8192, noverlap=4096,
         scaling="density", ref=REFERENCE_PRESSURE_WATER):
    """Probability density of Welch PSD levels over time segments.

    Segments the signal(s), computes a Welch PSD per segment and histograms the
    dB levels per frequency. Returns a :class:`PPSDResult`. 2-D input uses the
    longer axis as time; pass a list of 1-D arrays to be explicit.
    """
    if isinstance(data, list):
        signals = data
    else:
        data = np.asarray(data)
        if data.ndim == 1:
            signals = [data]
        elif data.ndim == 2:
            if data.shape[0] < data.shape[1]:
                signals = [data[i, :] for i in range(data.shape[0])]
            else:
                signals = [data[:, i] for i in range(data.shape[1])]
        else:
            raise ConfigurationError(
                "ppsd: data must be 1-D, 2-D, or a list of 1-D arrays; "
                f"got ndim={data.ndim}")

    chunk_size = int(seg_duration * fs)
    overlap_samples = int(chunk_size * overlap_pct / 100)
    step = chunk_size - overlap_samples
    if step <= 0:
        raise ConfigurationError(
            f"ppsd: overlap_pct ({overlap_pct}) too high — chunks never "
            "advance; require overlap_pct < 100.")

    levels = np.arange(lvlmin, lvlmax + ddB, ddB)
    psd_list = []
    for sig in signals:
        nps = nperseg
        nov = noverlap
        if chunk_size < nps:
            nps = chunk_size
            nov = int(chunk_size * overlap_pct / 100)
        for i in range(0, len(sig) - chunk_size + 1, step):
            chunk = sig[i: i + chunk_size]
            freqs, p = _sig.welch(chunk, fs, window=window, nperseg=nps,
                                  noverlap=nov, scaling=scaling)
            psd_list.append(p)

    if len(psd_list) == 0:
        raise ConfigurationError(
            "ppsd: no PSD segments computed; seg_duration="
            f"{seg_duration}s vs signal length={len(signals[-1])/fs:.2f}s")

    psd_array = np.array(psd_list)
    psd_segments_dB = power_to_db(psd_array, ref)
    mean_psd = np.mean(psd_segments_dB, axis=0)
    std_psd = np.std(psd_segments_dB, axis=0)

    pdf_matrix = np.zeros((len(levels) - 1, len(freqs)))
    for i in range(len(freqs)):
        hist, _ = np.histogram(psd_segments_dB[:, i], bins=levels, density=True)
        pdf_matrix[:, i] = hist
    pdf_matrix[pdf_matrix == 0] = np.nan

    return PPSDResult(freqs, levels, pdf_matrix, mean_psd, std_psd, ddB,
                      seg_duration)


def _sel_adjust_fmin_fmax(fmin, fmax, band_type, fs):
    """Snap configured band edges to band boundaries for this ``fs``."""
    if band_type == "octave":
        fmin = 2 ** np.floor(math.log2(fmin))
        fmax = 2 ** np.ceil(math.log2(fmax))
        if fmax > fs / 2:
            fmax = 2 ** np.floor(math.log2(fmax))
    elif band_type == "third_octave":
        base = math.pow(2, 1 / 6)
        fmin = base ** np.floor(math.log(fmin, base))
        fmax = base ** np.ceil(math.log(fmax, base))
        if fmax > fs / 2:
            fmax = base ** np.floor(math.log(fmax, base))
    return fmin, fmax


def _sel_bands(fmin, fmax, band_type, num_bands, fs):
    """Generate ``(low, center, high)`` frequency bands."""
    if fmin <= 0 or fmax <= fmin:
        raise ConfigurationError(
            f"sel: require fmin > 0 and fmax > fmin; got fmin={fmin}, fmax={fmax}")
    if band_type in ("octave", "third_octave"):
        fmin, fmax = _sel_adjust_fmin_fmax(fmin, fmax, band_type, fs)
    bands = []
    if band_type == "octave":
        base = math.sqrt(2)
        f_center = fmin
        while f_center < fmax:
            bands.append((f_center / base, f_center, f_center * base))
            f_center *= 2
        if bands and bands[-1][2] > fmax:
            bands[-1] = (bands[-1][0], bands[-1][1], fmax)
    elif band_type == "third_octave":
        base = math.pow(2, 1 / 6)
        f_center = fmin
        while f_center < fmax:
            bands.append((f_center / base, f_center, f_center * base))
            f_center *= math.pow(2, 1 / 3)
        if bands and bands[-1][2] > fmax:
            bands[-1] = (bands[-1][0], bands[-1][1], fmax)
    elif band_type == "linear":
        if num_bands <= 0:
            raise ConfigurationError(
                f"sel: num_bands must be positive for linear bands; got {num_bands}")
        bw = (fmax - fmin) / num_bands
        f_low = fmin
        for _ in range(num_bands):
            f_high = f_low + bw
            bands.append((f_low, (f_low + f_high) / 2, f_high))
            f_low = f_high
        if bands and bands[-1][2] > fmax:
            bands[-1] = (bands[-1][0], bands[-1][1], fmax)
    else:
        raise ConfigurationError(
            f"sel: unknown band_type={band_type!r}; valid: 'octave', "
            "'third_octave', 'linear'")
    return bands


def sel(data, fs, *, fmin=8.9125, fmax=22387, band_type="third_octave",
        num_bands=30, ref=REFERENCE_PRESSURE_WATER, integration_time=None,
        chunk_size=262144, nfft=None):
    """Sound Exposure Level per frequency band. Returns ``(sel_pa2s, bands)``.

    Uses a rectangular (boxcar) window with ``noverlap=0`` and no detrending so
    the summed PSD equals the band exposure exactly (Parseval); do not change
    this — a smoothing window would corrupt the energy identity.
    """
    if integration_time is not None:
        data = data[:min(int(integration_time * fs), len(data))]

    bands = _sel_bands(fmin, fmax, band_type, num_bands, fs)
    if chunk_size > len(data):
        chunk_size = len(data)
    if nfft is None:
        nfft = fs
    nfft = int(nfft)

    window = _sig.windows.boxcar(nfft)
    f = np.fft.rfftfreq(nfft, d=1 / fs)
    edges = np.array([b[0] for b in bands] + [bands[-1][2]])
    bin_band = np.digitize(f, edges) - 1
    band_bins = [np.where(bin_band == k)[0] for k in range(len(bands))]
    out = np.zeros(len(bands))

    for i in range(0, len(data), chunk_size):
        chunk = data[i: min(i + chunk_size, len(data))]
        n_seg = max(1, -(-len(chunk) // nfft))
        pad = n_seg * nfft - len(chunk)
        if pad:
            chunk = np.pad(chunk, (0, pad))
        _f, _t, Sxx = _sig.spectrogram(
            chunk, fs, window=window, noverlap=0, nfft=nfft,
            detrend=False, scaling="density")
        Sxx_sum = np.sum(Sxx, axis=1)
        for k, idx in enumerate(band_bins):
            out[k] += np.sum(Sxx_sum[idx])

    return out, bands
