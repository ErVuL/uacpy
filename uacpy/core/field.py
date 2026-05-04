"""Broadband IFFT helpers used by :class:`uacpy.core.results.TransferFunction`.

Implements frequency-aware bin placement, complex-spectrum interpolation
(prevents ghost echoes when the FFT bin spacing is finer than the data
spacing), and ``psif_envelope`` re-modulation that lands the arrival at
physical time ``r/c``. Lives in its own module to keep
:mod:`uacpy.core.results` compact.

``Field`` is also exposed here as an alias for
:class:`~uacpy.core.results.Result` — handy for ``isinstance`` type
guards in code that wants the union type.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from uacpy.core.constants import DEFAULT_SOUND_SPEED
from uacpy.core.results import (
    Result,
    TimeSeriesField,
    TimeTrace,
    TransferFunction,
)


# ``Field`` is an alias for the Result base class — useful as the union
# type for ``isinstance`` checks across all model outputs.
Field = Result


def _ifft_to_trace(
    tf: TransferFunction,
    *,
    depth: Optional[float],
    range_m: Optional[float],
    source_spectrum: Optional[np.ndarray],
    window: str,
    nfft: Optional[int],
    t_start: Optional[float],
) -> TimeTrace:
    """IFFT one (depth, range) cell of a TransferFunction → TimeTrace.

    Implements frequency-aware bin placement (each ``f_k`` lands at
    ``round(f_k/df)`` rather than at index ``k``), complex spectrum
    interpolation when the FFT bin spacing is finer than the data spacing
    (suppresses periodic ghost echoes), and the ``psif_envelope``
    re-modulation that lands the arrival at physical time ``r/c``.
    """
    data = tf.data                                # (n_d, n_r, n_f)
    freqs = np.asarray(tf.frequencies, dtype=float)
    n_d, n_r, n_freq = data.shape

    if n_freq < 2:
        raise ValueError("Need at least 2 frequencies for IFFT")

    d_idx = (
        int(np.argmin(np.abs(tf.depths - depth))) if depth is not None
        else n_d // 2
    )
    r_idx = (
        int(np.argmin(np.abs(tf.ranges - range_m))) if range_m is not None
        else 0
    )
    actual_depth = float(tf.depths[d_idx])
    actual_range = float(tf.ranges[r_idx])

    # Trailing-axis spectrum extraction — no axis kwarg needed.
    spectrum = data[d_idx, r_idx, :].copy()
    spectrum = np.nan_to_num(spectrum, nan=0.0)

    df_data = float(freqs[1] - freqs[0])
    df = min(df_data, 1.0)               # cap at 1 Hz for ≥ 1-second window

    bin_indices = np.round(freqs / df).astype(int)
    max_bin = int(bin_indices[-1])

    is_envelope = (tf.phase_reference == 'psif_envelope')

    if nfft is None:
        if is_envelope and 'Nsam' in tf.metadata:
            nfft_min = int(tf.metadata.get('Nsam', 4 * n_freq))
        else:
            nfft_min = 4 * n_freq
        nfft = max(nfft_min, max_bin + 1)
        nfft_pow2 = 1
        while nfft_pow2 < nfft:
            nfft_pow2 *= 2
        nfft = nfft_pow2

    if window == 'hann':
        win = np.hanning(n_freq)
    elif window == 'hamming':
        win = np.hamming(n_freq)
    elif window == 'blackman':
        win = np.blackman(n_freq)
    elif window == 'tukey':
        from scipy.signal import windows
        win = windows.tukey(n_freq, alpha=0.5)
    elif window == 'none':
        win = np.ones(n_freq)
    else:
        raise ValueError(f"Unknown window: {window}")

    dt = 1.0 / (nfft * df)

    if t_start is None:
        if is_envelope:
            cmin = tf.metadata.get('cmin', tf.metadata.get('c0', DEFAULT_SOUND_SPEED))
            if 'Nsam' in tf.metadata and 'fs' in tf.metadata:
                T_window = tf.metadata['Nsam'] / tf.metadata['fs']
            else:
                T_window = 1.0 / df_data
            lead = min(0.5 * T_window, 0.25)
            t_start = max(0.0, actual_range / cmin - lead)
        else:
            c0 = tf.metadata.get('c0', DEFAULT_SOUND_SPEED)
            t_start = max(0.0, actual_range / c0 - 2.0 / df)

    spectrum = spectrum * win
    if source_spectrum is not None:
        spectrum = spectrum * np.asarray(source_spectrum)

    padded = np.zeros(nfft, dtype=complex)
    min_bin = int(bin_indices[0])
    max_bin_fill = int(bin_indices[-1])

    if df < df_data * 0.99 and n_freq >= 4:
        # Demodulate, interpolate the slow residual phase, remodulate.
        c0 = tf.metadata.get('c0', DEFAULT_SOUND_SPEED)
        t_demod = actual_range / c0
        demod = np.exp(1j * 2.0 * np.pi * freqs * t_demod)
        spec_demod = spectrum * demod

        from scipy.interpolate import interp1d
        fill_bins = np.arange(min_bin, min(max_bin_fill + 1, nfft))
        fill_freqs = fill_bins * df
        re_interp = interp1d(freqs, spec_demod.real, kind='linear',
                             bounds_error=False, fill_value=0.0)
        im_interp = interp1d(freqs, spec_demod.imag, kind='linear',
                             bounds_error=False, fill_value=0.0)
        spec_interp = re_interp(fill_freqs) + 1j * im_interp(fill_freqs)
        remod = np.exp(1j * 2.0 * np.pi * fill_freqs * (t_start - t_demod))
        padded[fill_bins] = spec_interp * remod
    else:
        spectrum = spectrum * np.exp(1j * 2.0 * np.pi * freqs * t_start)
        valid = (bin_indices >= 0) & (bin_indices < nfft)
        padded[bin_indices[valid]] = spectrum[valid]

    result = 2.0 * np.real(np.fft.ifft(padded))
    time = t_start + np.arange(nfft) * dt

    return TimeTrace(
        data=result,
        time=time,
        depth=actual_depth,
        range_m=actual_range,
        model=tf.model,
        backend=tf.backend,
        source_depths=tf.source_depths,
        frequency=tf.frequency,
        metadata={
            'window': window,
            'source_model': tf.model,
        },
    )


def _synthesize_time_series(
    tf: TransferFunction,
    *,
    source_waveform: np.ndarray,
    sample_rate: float,
    t_start: Optional[float],
    window: str,
    nfft: Optional[int],
) -> TimeSeriesField:
    """Convolve every grid cell of a TransferFunction with a source waveform.

    Output shape: ``(n_d, n_r, n_t)``.
    """
    wf = np.asarray(source_waveform, dtype=float).ravel()
    n_src = len(wf)
    if n_src < 2:
        raise ValueError("source_waveform must have at least 2 samples")

    src_fft = np.fft.rfft(wf)
    src_freqs = np.fft.rfftfreq(n_src, 1.0 / sample_rate)

    from scipy.interpolate import interp1d
    re_interp = interp1d(src_freqs, src_fft.real, bounds_error=False, fill_value=0.0)
    im_interp = interp1d(src_freqs, src_fft.imag, bounds_error=False, fill_value=0.0)
    source_spectrum = re_interp(tf.frequencies) + 1j * im_interp(tf.frequencies)

    n_d, n_r, _ = tf.data.shape
    depths = np.asarray(tf.depths)
    ranges = np.asarray(tf.ranges)

    # Lock t_start using the first cell's policy so all traces share a clock.
    if t_start is None:
        t0_trace = _ifft_to_trace(
            tf, depth=float(depths[0]), range_m=float(ranges[0]),
            source_spectrum=source_spectrum,
            window=window, nfft=nfft, t_start=None,
        )
        t_start = float(t0_trace.metadata['t_start'])

    out = None
    time_vec = None
    for di in range(n_d):
        for ri in range(n_r):
            tr = _ifft_to_trace(
                tf, depth=float(depths[di]), range_m=float(ranges[ri]),
                source_spectrum=source_spectrum,
                window=window, nfft=nfft, t_start=t_start,
            )
            if out is None:
                time_vec = tr.time
                out = np.zeros((n_d, n_r, len(tr.data)), dtype=tr.data.dtype)
            out[di, ri, :] = tr.data

    return TimeSeriesField(
        data=out,
        depths=depths,
        ranges=ranges,
        time=time_vec,
        model=tf.model,
        backend=tf.backend,
        source_depths=tf.source_depths,
        frequencies=tf.frequencies,
        metadata={
            'source_waveform_fs': sample_rate,
            'window': window,
        },
    )
