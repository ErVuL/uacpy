"""The array half of the broadband-to-time-series synthesis: the grid, the
sum of a band at the instants asked for, the per-cell synthesis and its
diagnostics, shared by :func:`uacpy.acoustic_signal.synthesize_time_series`,
:func:`uacpy.acoustic_signal.impulse_response_from_transfer_function`,
:meth:`uacpy.Field.synthesize_time_series` and
:meth:`uacpy.Field.impulse_response` (and so by :meth:`uacpy.Field.to_time_trace`
and :meth:`BeamformedField.to_time_trace`, which call them on one cell).

A package-internal module: its names are public to the package, not to
users.
"""

from __future__ import annotations

import warnings
from typing import Callable, NamedTuple, Optional, Tuple

import numpy as np

from uacpy.acoustic_signal.windows import _taper
from uacpy.core.constants import DEFAULT_SOUND_SPEED
from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, NumericsWarning, ValidityWarning,
)
from uacpy.core._fourier import _BLOCK_ELEMENTS
from uacpy.core._warn_frames import USER_FRAME_SKIP


# A record of n samples per cell can demand a multi-GB buffer from a too-high
# sample_rate or a too-fine frequency grid. The synthesis refuses a record
# longer than 2**26 ≈ 67 M samples.
MAX_SYNTHESIS_NFFT = 1 << 26

#: Peak bytes the synthesis holds per sample of one cell's record: the
#: padded spectrum and its complex transform, then the transform, the real
#: trace, its time axis and the envelope (57 B measured with tracemalloc on
#: one cell; pocketfft's own scratch comes on top). About 3.8 GB per cell at
#: the cap.
SYNTHESIS_BYTES_PER_SAMPLE = 60

#: How close ``sample_rate / df`` must be to a whole number ``n`` for a
#: ``1/df`` record at ``sample_rate`` to be the ``n``-point inverse-FFT
#: lattice: the sample instants ``k/sample_rate`` and ``k/(n·df)`` then part
#: by less than 1e-6 of a sample at the record's end, a phase of at most
#: ``π·1e-6`` on a line at the Nyquist frequency (:func:`band_sum`). It
#: admits the floating-point dust of a grid built by ``linspace`` or
#: ``arange`` on a long record without changing its sample count.
_LATTICE_TOLERANCE = 1e-6


class SynthesisPlan(NamedTuple):
    """The grid a synthesis runs on (:func:`synthesis_grid`).

    ``freqs`` is the uniform ascending band (Hz) and ``df`` its spacing,
    which makes the record ``1/df`` long; the record is sampled at
    ``sample_rate`` (Hz) on ``n_samples`` instants; ``win`` is the band
    window, one weight per frequency."""
    freqs: np.ndarray
    df: float
    sample_rate: float
    n_samples: int
    win: np.ndarray


def record_samples(sample_rate: float, df: float) -> int:
    """Samples a ``1/df`` record holds at ``sample_rate``: ``sample_rate/df``
    when that is a whole number (to :data:`_LATTICE_TOLERANCE`), else the
    next whole number above it, so every sample lies inside the record."""
    ratio = float(sample_rate) / float(df)
    whole = int(round(ratio))
    if whole >= 1 and abs(ratio - whole) <= _LATTICE_TOLERANCE:
        return whole
    return int(np.ceil(ratio))


#: Least margin (samples) an arrival list's impulse response keeps either
#: side of its arrivals, beside the arrivals' own delay span: room for the
#: band-limited impulse of a path between samples, or at a phase other
#: than 0 or pi, to fall away before the record ends.
ARRIVAL_IR_MARGIN_SAMPLES = 256


def arrival_ir_margin(span: float, sample_rate: float) -> float:
    """The margin (s) an arrival list's record keeps either side of its
    arrivals: their delay ``span`` (s), and at least
    :data:`ARRIVAL_IR_MARGIN_SAMPLES` samples at ``sample_rate``."""
    return max(float(span), ARRIVAL_IR_MARGIN_SAMPLES / float(sample_rate))


def _chirp(q: float, k: np.ndarray) -> np.ndarray:
    """``exp(i·pi·q·k²)`` for whole ``k``, its phase computed from ``q`` and
    the exact whole ``k²`` and reduced modulo 2. Powering the complex step
    ``w = exp(i·pi·q)`` instead (``w**(k²)``, as ``scipy.signal.czt`` does)
    carries the step's own angle rounding times ``k²``: measured 1.4e-7 of
    the peak at 350 000 samples, against 2e-10 here."""
    k2 = np.asarray(k, dtype=np.int64) ** 2
    return np.exp(1j * np.pi * np.remainder(q * k2.astype(float), 2.0))


def _chirp_z(G: np.ndarray, q: float, n_out: int) -> np.ndarray:
    """``X[..., n] = sum_k G[..., k] exp(2 pi i q k n)`` for ``n < n_out``,
    by Bluestein's chirp-z algorithm (``kn = (k² + n² − (n−k)²)/2``): one
    FFT convolution of length ``next_fast_len(K + n_out − 1)`` per row."""
    # Imported here: scipy is not needed to hold a Field, only to
    # synthesise from one (test_lazy_imports).
    import scipy.fft as _sp_fft
    n_in = G.shape[-1]
    length = _sp_fft.next_fast_len(n_in + n_out - 1)
    kernel = np.zeros(length, dtype=complex)
    kernel[:n_out] = np.conj(_chirp(q, np.arange(n_out)))
    if n_in > 1:
        kernel[length - n_in + 1:] = np.conj(
            _chirp(q, np.arange(n_in - 1, 0, -1)))
    kernel = _sp_fft.fft(kernel)
    weighted = G * _chirp(q, np.arange(n_in))
    out = _sp_fft.ifft(_sp_fft.fft(weighted, n=length, axis=-1) * kernel,
                       axis=-1)[..., :n_out]
    out *= _chirp(q, np.arange(n_out))
    return out


def band_sum(G, freqs, *, sample_rate: float, t_start, n_samples: int
             ) -> np.ndarray:
    """``sum_k G[m, k] exp(2 pi i f_k (t_start[m] + n/sample_rate))`` for
    ``n < n_samples``: the analytic sum of a band at exactly the instants
    asked for. Returns complex ``(M, n_samples)``.

    ``G`` is ``(M, n_f)`` on the uniform ascending ``freqs`` (Hz),
    ``t_start`` a scalar or one start (s) per row. When ``sample_rate/df``
    is a whole number ``N`` (:func:`record_samples`) and the record fits in
    it, the band is placed on the ``N``-point inverse-FFT lattice: sample
    ``k`` at bin ``round(f_0/(fs/N)) + k``, the common offset of the band
    from the lattice taken back out by ``exp(2 pi i·offset·t)``. That is
    exact when ``sample_rate/df`` is ``N`` itself; one within
    :data:`_LATTICE_TOLERANCE` (``ε``) of it has its spacing taken as
    ``fs/N``, which moves a line's phase at the record's end by up to
    ``π·ε`` (1.1e-6 of the peak for a band to just below the Nyquist
    frequency at ``ε = 9e-7``). Otherwise — or when the record asked for is
    under half the lattice, which the chirp-z then evaluates more cheaply —
    the same sum is evaluated by chirp-z (:func:`_chirp_z`), at exactly
    ``t_start + n/sample_rate``; it agrees with the direct sum to about
    1e-11 of the peak. No sample is interpolated or resampled on either
    path."""
    import scipy.fft as _sp_fft
    G = np.asarray(G, dtype=complex)
    freqs = np.asarray(freqs, dtype=float)
    fs = float(sample_rate)
    n_rows = G.shape[0]
    starts = np.broadcast_to(np.asarray(t_start, dtype=float), (n_rows,))
    elapsed = np.arange(n_samples) / fs
    if freqs.size == 0 or n_samples == 0:
        return np.zeros((n_rows, n_samples), dtype=complex)
    # The record's start, as the phase every line carries there.
    G = G * np.exp(2j * np.pi * np.remainder(
        freqs[None, :] * starts[:, None], 1.0))
    if freqs.size == 1:
        return G[:, :1] * np.exp(2j * np.pi * freqs[0] * elapsed)[None, :]
    df = float(freqs[-1] - freqs[0]) / (freqs.size - 1)
    lattice = record_samples(fs, df)
    whole = abs(fs / df - lattice) <= _LATTICE_TOLERANCE
    ramp_hz = freqs[0]
    if whole and n_samples <= lattice <= 2 * n_samples:
        step = fs / lattice
        first = int(np.floor(freqs[0] / step + 0.5))
        bins = first + np.arange(freqs.size)
        keep = (bins >= 0) & (bins < lattice)
        ramp_hz = freqs[0] - first * step
        chunk = max(1, _BLOCK_ELEMENTS // lattice)
        # One chunk holding the whole record keeps the transform as the
        # result, so a row holds one complex record at a time.
        out = (None if chunk >= n_rows and n_samples == lattice
               else np.empty((n_rows, n_samples), dtype=complex))
        for a in range(0, n_rows, chunk):
            rows = slice(a, min(a + chunk, n_rows))
            padded = np.zeros((rows.stop - rows.start, lattice), dtype=complex)
            padded[:, bins[keep]] = G[rows][:, keep]
            transform = np.fft.ifft(padded, axis=-1)
            del padded
            # ifft carries 1/N; the sum asked for carries none.
            transform *= lattice
            if out is None:
                out = transform
            else:
                out[rows] = transform[:, :n_samples]
    else:
        out = np.empty((n_rows, n_samples), dtype=complex)
        span = _sp_fft.next_fast_len(freqs.size + n_samples - 1)
        chunk = max(1, _BLOCK_ELEMENTS // span)
        for a in range(0, n_rows, chunk):
            rows = slice(a, min(a + chunk, n_rows))
            out[rows] = _chirp_z(G[rows], df / fs, n_samples)
    if ramp_hz != 0.0:
        out *= np.exp(2j * np.pi * np.remainder(ramp_hz * elapsed, 1.0))
    return out


#: The synthesis' documented band window: none, spelled ``'boxcar'``. The
#: received signal is ``S(f)·H(f)`` with no extra filter (Jensen et al.,
#: *Computational Ocean Acoustics*, sect. 8.2.1.1); a window across the band
#: filters the pulse. Every synthesis takes it for ``window=None``.
SYNTHESIS_WINDOW = 'boxcar'


def band_window_name(window) -> str:
    """``window`` as the name of a band window: ``None`` is the synthesis'
    documented default, :data:`SYNTHESIS_WINDOW`."""
    return SYNTHESIS_WINDOW if window is None else window


def recorded_band_window(window) -> Optional[str]:
    """What a synthesised Field records as its ``synthesis_window`` for
    ``window``: the band window's name, or ``None`` when no window was
    applied (``'boxcar'``), so a record says one thing for no taper."""
    name = band_window_name(window)
    return None if name == 'boxcar' else name


def synthesis_grid(
    freqs: np.ndarray,
    *,
    window: str,
    sample_rate: float,
    who: str,
) -> SynthesisPlan:
    """Validate a frequency axis and size the synthesis record on it
    (:class:`SynthesisPlan`): :func:`record_samples` at exactly
    ``sample_rate`` over ``1/df``. Shared by the Field synthesis and the
    array functions.
    """
    freqs = np.asarray(freqs, dtype=float)
    n_freq = freqs.size

    if n_freq < 2:
        raise ConfigurationError(
            f"{who}: need at least 2 frequencies for IFFT; got {n_freq}."
        )
    if float(np.min(freqs)) < 0.0:
        # The synthesis is one-sided: a negative bin has no place on it and
        # would be dropped from the sum without a word.
        raise ConfigurationError(
            f"{who}: the frequency axis reaches {float(np.min(freqs)):g} Hz; "
            f"the synthesis takes the one-sided band, 0 Hz and up.")

    # The transfer function is sampled at df, so the trace it represents
    # without aliasing is exactly 1/df long; a longer record needs a finer
    # frequency grid, which means more model runs. Inventing the samples in
    # between by interpolation would be a convolution with a triangular
    # kernel — a sinc^2(pi df t) taper in time.
    from uacpy.acoustic_signal.channel import _uniform_frequency_step
    df = _uniform_frequency_step(freqs, who=who)

    n_samples = record_samples(sample_rate, df)
    if n_samples > MAX_SYNTHESIS_NFFT:
        raise ConfigurationError(
            f"{who}: a {1.0 / df:.4g} s record (1/df, df = {df:.4g} Hz) "
            f"at sample_rate={float(sample_rate):g} Hz is "
            f"{n_samples:,} samples per cell (~"
            f"{n_samples * SYNTHESIS_BYTES_PER_SAMPLE / 1e9:.1f} GB of "
            f"working memory per cell), above the "
            f"{MAX_SYNTHESIS_NFFT:,}-sample safety cap.",
            remediation="Use a coarser frequency grid, or the "
                        "impulse-response route, which convolves a "
                        "signal of any length in blocks: "
                        "Field.impulse_response(sample_rate) and "
                        "apply(), or acoustic_signal."
                        "apply_impulse_response_blocks.")
    win = _taper(band_window_name(window), n_freq, who=who)
    return SynthesisPlan(freqs, df, float(sample_rate), n_samples, win)


def zero_no_mode_bins(
    spectra: np.ndarray,
    no_mode_bins,
    *,
    who: str,
    frequencies: Optional[np.ndarray] = None,
) -> np.ndarray:
    """``spectra`` ``(M, n_f)`` with the bins where a normal-mode model's
    binary found no mode taken as zero, and one warning naming them.

    ``no_mode_bins`` are the bin indices the model recorded
    (:attr:`~uacpy.core.results.Field.no_mode_bins`, Kraken), wherever they
    sit in the band. There the binary kept no mode in its window — below
    the first cutoff no mode has a real horizontal wavenumber (Jensen et
    al., *Computational Ocean Acoustics*, sect. 2.4.4.4, Eq. 2.154) — so the
    modal sum is empty, and field.exe returns zero pressure for an empty
    mode set (``KrakenField/EvaluateMod.f90:27-31``). A
    ``RunMode.TIME_SERIES`` run takes those bins as zero for the same
    reason, and so does every synthesis from ``H(f)`` (traces, exposure and
    peak maps, impulse responses), so the one field gives one answer by
    every route. What a zero leaves out is the continuous spectrum, the
    steep near-field energy the bottom absorbs within a few ranges (ibid.,
    sect. 1.6), which a wavenumber-integration model (Scooter) holds.

    In a listed bin each cell that holds no data (NaN) is zeroed, and a
    cell a caller filled in keeps its value; a cell NaN at every bin is no
    data, not a no-mode bin, and stays NaN. ``frequencies`` (Hz, the band
    axis) lets the warning name the bins."""
    bins = [int(i) for i in (no_mode_bins or ())]
    spectra = np.asarray(spectra)
    if not bins:
        return spectra
    live = ~np.all(np.isnan(spectra), axis=1)
    if not np.any(live):
        return spectra
    block = spectra[np.ix_(live, bins)]
    empty = np.isnan(block)
    if not empty.any():
        return spectra
    gap = [i for i, hit in zip(bins, empty.any(axis=0)) if hit]
    out = np.array(spectra, copy=True)
    out[np.ix_(live, bins)] = np.where(empty, 0.0, block)
    named = ""
    if frequencies is not None:
        freqs = np.asarray(frequencies, dtype=float)
        more = f" and {len(gap) - 8} more" if len(gap) > 8 else ""
        named = (" (" + ", ".join(f"{freqs[i]:g}" for i in gap[:8]) + more
                 + " Hz)")
    warnings.warn(
        f"{who}: {len(gap)} bin(s){named} hold no mode of the normal-mode "
        f"model (Field.no_mode_bins); they are taken as zero, as field.exe's "
        f"empty modal sum and a RunMode.TIME_SERIES run take them. The "
        f"continuous-spectrum field a wavenumber-integration model "
        f"(Scooter) holds there is not in the result; a band that avoids "
        f"those frequencies leaves them out.",
        ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP)
    return out


def warn_unsolved_bins(
    spectra: np.ndarray,
    *,
    who: str,
    cell_label: Optional[Callable[[int], str]] = None,
    frequencies: Optional[np.ndarray] = None,
) -> None:
    """Warn once about the NaN bins in a batch of cell spectra ``(M, n_f)``.

    A NaN bin is a frequency the model did not solve, not one carrying no
    energy, so it is never zeroed: filling it would put a spectral notch the
    model never produced into a trace that then looks finite and ordinary.
    The NaNs are kept and propagate through the IFFT, which makes the whole
    trace no-data — a trace cannot be synthesised from a spectrum with holes
    in it. (The bins where a normal-mode model found no mode are the
    exception, zeroed beforehand by :func:`zero_no_mode_bins`.)

    One warning covers the whole batch, however many cells it holds: the
    count of cells with unsolved bins and of the bins themselves, the
    frequencies when ``frequencies`` (Hz) is given, and — with
    ``cell_label`` (row index -> the cell's coordinates as text, e.g.
    ``'depth 10 m, range 1500 m'``) — the first such cell. Cells that are
    entirely NaN (masked below the seafloor, say: nothing about them was
    solved) are counted apart. ``who`` is the public entry point's name and
    prefixes the diagnostic.
    """
    nan_bins = np.isnan(spectra)
    all_nan = np.all(nan_bins, axis=1)
    partial = np.any(nan_bins, axis=1) & ~all_nan
    n_cells = spectra.shape[0]
    n_f = spectra.shape[1]
    parts = []
    if np.any(partial):
        bins = np.flatnonzero(np.any(nan_bins[partial], axis=0))
        named = ""
        if frequencies is not None:
            freqs = np.asarray(frequencies, dtype=float)[bins[:8]]
            more = f" and {bins.size - 8} more" if bins.size > 8 else ""
            named = f" ({', '.join(f'{f:g}' for f in freqs)}{more} Hz)"
        first = ("" if cell_label is None else
                 f", first at {cell_label(int(np.flatnonzero(partial)[0]))}")
        parts.append(
            f"{int(partial.sum())} of {n_cells} cell(s) of H(f) hold bins the "
            f"model did not solve — {bins.size} of {n_f} bins{named}"
            f"{first} — so their traces are NaN rather than carrying a notch "
            f"at those frequencies. Re-run them, or narrow the band to the "
            f"bins that solved.")
    if np.any(all_nan):
        first = ("" if cell_label is None else
                 f", first at {cell_label(int(np.flatnonzero(all_nan)[0]))}")
        parts.append(
            f"{int(all_nan.sum())} of {n_cells} cell(s) are entirely NaN (no "
            f"valid model output there{first}); their traces are NaN, not "
            f"silence.")
    if parts:
        warnings.warn(f"{who}: " + " ".join(parts),
                      NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)


# Energy (dB) an exposure may lose outside its band unreported, and the
# change (dB) the band may make to the peak of the source pulse it limits:
# under 0.1 dB a level does not move at the 0.1 dB it is quoted to. uacpy
# guidance.
_EXPOSURE_SHARE_LOSS_DB = 0.1
_PEAK_CHANGE_DB = 0.1

# Level (dB re the spectrum's peak) bounding a waveform's support. uacpy
# guidance.
_SUPPORT_FLOOR_DB = -40.0


def band_limited_pulse(wf, sample_rate, freqs):
    """What the band ``freqs`` (Hz, uniform) does to the waveform ``wf``:
    ``(peak_change_dB, support_hz)`` — the peak of ``wf`` band-limited to
    the band (its spectrum zeroed outside ``freqs[0] - Δf/2`` to
    ``freqs[-1] + Δf/2``) against its own, and the span (Hz) where its
    spectrum stays within ``_SUPPORT_FLOOR_DB`` of its peak. On an FFT four
    times the waveform's length, at least."""
    wf = np.asarray(wf, dtype=float).ravel()
    freqs = np.asarray(freqs, dtype=float)
    nfft = 1 << int(np.ceil(np.log2(max(4 * wf.size, 16))))
    spectrum = np.fft.rfft(wf, nfft)
    grid = np.fft.rfftfreq(nfft, 1.0 / float(sample_rate))
    half = 0.5 * (freqs[1] - freqs[0]) if freqs.size > 1 else 0.0
    inside = (grid >= freqs[0] - half) & (grid <= freqs[-1] + half)
    limited = np.fft.irfft(spectrum * inside, nfft)
    peak = float(np.max(np.abs(wf)))
    change = (20.0 * np.log10(max(float(np.max(np.abs(limited))), 1e-300)
                              / peak) if peak > 0.0 else 0.0)
    level = np.abs(spectrum)
    kept = grid[level >= level.max() * 10.0 ** (_SUPPORT_FLOOR_DB / 20.0)]
    return change, (float(kept[0]), float(kept[-1]))


def _hann_weight_at_peak(mag, freqs, who) -> str:
    """What ``window='hann'`` costs: its weight on the band's bin where the
    source spectrum ``mag`` peaks (the synthesis's own window, ``_taper``)."""
    weight = _taper('hann', mag.size, who=who)[int(np.argmax(mag))]
    level = ("removes it" if weight <= 0.0
             else f"weights it by {20.0 * np.log10(weight):.1f} dB")
    return (f"window='hann' trades the ringing for a level bias: it {level} "
            f"at the spectrum's peak "
            f"({float(np.asarray(freqs)[int(np.argmax(mag))]):g} Hz)")


def _energy_words(weighting) -> str:
    """``'energy'``, or the weighted energy ``weighting`` names."""
    if weighting is None:
        return "energy"
    if isinstance(weighting, str):
        return f"{weighting}-weighted energy"
    return "weighted energy"


def warn_band_edge_cuts_spectrum(source_spectrum, window: str,
                                  who: str, *, freqs, waveform=None,
                                  sample_rate=None,
                                  energy_share: Optional[float] = None,
                                  weighting=None) -> None:
    """Warn when an untapered synthesis's band ``freqs`` (Hz) cuts the
    source spectrum, judged by what the cut changes.

    Untapered (``window`` ``'boxcar'``, or ``None``, its documented
    default, :func:`band_window_name`) the band edge is a rectangular
    filter; it is harmless where ``S(f)`` has already fallen away, and
    rings (Gibbs overshoot, energy outside the pulse) where it has not.

    ``energy_share`` (the band's share of the waveform's energy,
    :func:`band_energy_share`) is given for an exposure, which integrates
    the whole record: there the ringing moves no energy, and what the cut
    costs is the energy outside the band, so the warning is given when that
    loses more than ``_EXPOSURE_SHARE_LOSS_DB`` and states the share
    instead of advising a taper, which would cost the exposure more (a Hann
    window read a burst 17 dB light, :meth:`Field.sound_exposure_level`).
    A weighted exposure's share is of the weighted energy
    (:func:`band_energy_share` with its ``weighting``), and the warning
    says so.

    For a peak or a trace with its ``waveform`` the warning is given when
    the band moves the peak of the pulse it limits by more than
    ``_PEAK_CHANGE_DB`` (:func:`band_limited_pulse`), and names the
    waveform's support as the band to take, with what a Hann window would
    cost at the spectrum's peak (a damped 200 Hz sine in a 25-4000 Hz band:
    -34.5 dB, which took a peak map 26 dB low)."""
    if band_window_name(window) != 'boxcar' or source_spectrum is None:
        return
    mag = np.abs(np.asarray(source_spectrum))
    if mag.size < 2 or not np.any(mag > 0):
        return
    edge = max(mag[0], mag[-1])
    if edge == 0.0:
        # A spectrum already zero at both edges is not cut: the rectangular
        # band edge multiplies zero there, so there is no step to ring.
        return
    edge_dB = 20.0 * np.log10(edge / mag.max())
    freqs = np.asarray(freqs, dtype=float)
    if energy_share is not None:
        kept_dB = 10.0 * np.log10(max(energy_share, 1e-300))
        if -kept_dB > _EXPOSURE_SHARE_LOSS_DB:
            warnings.warn(
                f"{who}: the band's edge cuts the source spectrum at "
                f"{edge_dB:.1f} dB re its peak, and the band holds "
                f"{100 * energy_share:.3g} % of the source waveform's "
                f"{_energy_words(weighting)} ({kept_dB:.2f} dB): the "
                f"exposure counts that share. Widen the frequency band to "
                f"the waveform's support"
                + ("" if weighting is None else " under that weighting")
                + " to count the rest; a window would remove more of it.",
                NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)
    elif waveform is not None and sample_rate is not None:
        change, (low, high) = band_limited_pulse(waveform, sample_rate, freqs)
        if abs(change) > _PEAK_CHANGE_DB:
            warnings.warn(
                f"{who}: the band ({freqs[0]:g}-{freqs[-1]:g} Hz) cuts the "
                f"source waveform: limited to it, the pulse's peak moves "
                f"{change:+.2f} dB, and the untapered synthesis rings at the "
                f"band edges. Widen the frequency band to the waveform's "
                f"support, {low:g}-{high:g} Hz (its spectrum within "
                f"{-_SUPPORT_FLOOR_DB:.0f} dB of its peak); "
                f"{_hann_weight_at_peak(mag, freqs, who)}.",
                NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)


# A Fourier-synthesised record is periodic with period 1/Δf, so an arrival
# train still ringing at the record's end continues onto its start: the
# trace is truncated and the tail wraps. Level (dB re the trace's own
# envelope maximum) of the envelope over the record's first / last
# ``_RECORD_EDGE_FRACTION`` above which that is reported. Measured on a
# 100 m Pekeris at 4 km, 25-75 Hz Gaussian pulse: a 1 s record (Δf = 1 Hz)
# ending before the 2.89 s main arrival reads 0 dB at its head and -1.8 dB
# at its tail (Kraken, Scooter) and -13.6 dB at its tail (Bellhop), while
# the 2 s and 4 s records of the same run read below -54 dB at both edges.
# Example 24 (Bellhop, 5 km Pekeris, 1/Δf = 0.73 s) reads -17.7 dB at its
# tail, where a 2.9 s record shows steep bottom-bounce arrivals continuing
# to 3.76 s, past the 3.70 s record end. uacpy guidance from these uacpy
# measurements; the grounded statement is that the record "must be
# selected large enough that it contains the entire transient response"
# (Jensen et al., Computational Ocean Acoustics, sect. 8.2.1.3).
_RECORD_EDGE_LEVEL_DB = -20.0


_RECORD_EDGE_FRACTION = 0.05

#: The fast/slow path-speed spread the window-wrap warning assumes
#: (see :func:`record_start`): 5 % of the surface speed, uacpy guidance.
_ASSUMED_PATH_SPEED_SPREAD = 0.05

#: The least share of a synthesis record placed before the estimated first
#: arrival (see :func:`record_lead`): clear of the head region the edge
#: check reads (``_RECORD_EDGE_FRACTION``).
RECORD_LEAD_FRACTION = 0.1

#: The band-limited onset a record's lead must hold, in units of
#: ``1/bandwidth``: the sidelobes of an untapered band's impulse,
#: ``~1/(π·B·t)``, fall under the edge check's -20 dB
#: (``_RECORD_EDGE_LEVEL_DB``) about ``3.2/B`` before the peak.
RECORD_ONSET_BANDWIDTHS = 4.0


def record_lead(record_s: float, bandwidth: float) -> float:
    """The lead (s) of a synthesis record ``record_s`` long over a band
    ``bandwidth`` Hz wide: the larger of :data:`RECORD_LEAD_FRACTION` of
    the record and :data:`RECORD_ONSET_BANDWIDTHS` ``/ bandwidth``, never
    more than half the record."""
    onset = (RECORD_ONSET_BANDWIDTHS / bandwidth if bandwidth > 0.0
             else 0.5 * record_s)
    return min(0.5 * record_s, max(RECORD_LEAD_FRACTION * record_s, onset))


def record_edge_levels(analytic: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-row envelope maxima over the record's head and tail, in dB re
    each row's own envelope maximum (``-inf`` for a row with no energy).

    ``analytic`` is the complex synthesis before ``2·Re``: it holds only
    positive frequencies, so its modulus is the trace's envelope."""
    env = np.abs(analytic)
    n = max(1, int(round(_RECORD_EDGE_FRACTION * env.shape[-1])))
    with np.errstate(invalid='ignore', divide='ignore'):
        peak = np.max(env, axis=-1)
        ok = np.isfinite(peak) & (peak > 0.0)
        head = np.full(env.shape[0], -np.inf)
        tail = np.full(env.shape[0], -np.inf)
        head[ok] = 20.0 * np.log10(np.max(env[ok, :n], axis=-1) / peak[ok])
        tail[ok] = 20.0 * np.log10(np.max(env[ok, -n:], axis=-1) / peak[ok])
    return head, tail


def record_edges_are_judgeable(record_s: float, pulse_s: Optional[float],
                                 lead_s: float) -> bool:
    """Whether an auto-placed record's edges can be quiet at all.

    :func:`record_start` puts the estimated first arrival ``lead_s`` in
    (:func:`record_lead`), so a lone arrival there occupies
    ``[lead_s, lead_s + pulse_s]`` and reaches the tail region
    ``[T(1 - f), T]`` (``f`` = ``_RECORD_EDGE_FRACTION``) unless
    ``pulse_s < T(1 - f) - lead_s``. Below that
    the edge energy measures the grid against the pulse — which the
    DFT-period check and the model TIME_SERIES route's derived-grid notice
    name — not an arrival train running past the record. ``pulse_s=None``
    (a raw source spectrum of unknown duration) is not judged."""
    if pulse_s is None:
        return False
    return record_s * (1.0 - _RECORD_EDGE_FRACTION) - lead_s > pulse_s


def record_edge_finding(head_dB: np.ndarray,
                        tail_dB: np.ndarray) -> Optional[Tuple[float, str]]:
    """``(worst_dB, edge)`` when an auto-placed record holds energy above
    :data:`_RECORD_EDGE_LEVEL_DB` at its ``'end'`` or ``'start'`` (the
    louder of the two), else ``None``; ``head_dB`` / ``tail_dB`` from
    :func:`record_edge_levels`."""
    worst_tail = float(np.max(tail_dB)) if tail_dB.size else -np.inf
    worst_head = float(np.max(head_dB)) if head_dB.size else -np.inf
    worst = max(worst_tail, worst_head)
    if not worst > _RECORD_EDGE_LEVEL_DB:
        return None
    return worst, ('end' if worst_tail >= worst_head else 'start')


def warn_record_edge_energy(head_dB: np.ndarray, tail_dB: np.ndarray, *,
                             time: np.ndarray, df: float, who: str) -> None:
    """Warn when an auto-placed synthesised record ends (or starts) inside
    the arrivals.

    Callers invoke this only for a record whose start was auto-estimated
    (:func:`record_start`), which puts :func:`record_lead` before the
    estimated first arrival, clear of the head region read here: both
    edges of such a record should be quiet.
    The tail is the primary signal — energy not yet decayed at the record's
    end continues, by the 1/Δf periodicity, onto its start — and the head
    catches the same train already wrapped. A ``t_start`` the caller pinned
    is a placement, and may open on an arrival (whose band-limited precursor
    then sits at the record end) by choice, so it is not judged."""
    finding = record_edge_finding(head_dB, tail_dB)
    if finding is None:
        return
    worst, edge = finding
    warnings.warn(
        f"{who}: the synthesised record [{time[0]:.4g}, {time[-1]:.4g}] s "
        f"holds energy at its {edge} ({worst:.1f} dB re the trace maximum, "
        f"above {_RECORD_EDGE_LEVEL_DB:.0f} dB), so it cuts through the "
        f"arrival train: the record is periodic with 1/Δf = {1.0 / df:.4g} s "
        f"and the energy past its end wraps onto its start. Refine Δf (more "
        f"frequencies, or a narrower spacing) so 1/Δf spans the arrivals, "
        f"or pass t_start= to place the record.",
        NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)


def synthesize_traces(
    spectra: np.ndarray,
    plan: SynthesisPlan,
    *,
    source_spectrum: Optional[np.ndarray],
    t_start,
    who: str,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Synthesise a batch of cell spectra ``(M, n_f)`` on ``plan``: returns
    ``(traces, (head_dB, tail_dB))``, the traces ``(M, plan.n_samples)``
    with sample ``n`` of row ``m`` at ``t_start[m] + n/plan.sample_rate``
    (``t_start`` a scalar or one start per row), the pair from
    :func:`record_edge_levels`.

    ``p(t) = 2·Re Σ_k w_k·H_k·S_k·e^{2πi f_k t}·Δf``, a Riemann sum of the
    continuous inverse transform (:func:`band_sum` evaluates it), so the
    amplitude does not depend on the sampling. ``w_k`` is the band window,
    halved at 0 Hz and on the Nyquist frequency of the output rate: a
    sampled record holds those lines once, with no separate
    negative-frequency twin for the ``2·Re`` to stand for. A line above the
    Nyquist frequency cannot be sampled at this rate and is left out; a
    waveform's spectrum is zero there (:func:`waveform_spectrum_at`), and a
    bare response (``source_spectrum=None``) warns that part of ``H`` is
    dropped."""
    freqs, rate = plan.freqs, plan.sample_rate
    spectra = spectra * (plan.win * plan.df)
    if source_spectrum is not None:
        spectra = spectra * np.asarray(source_spectrum)
    nyquist = 0.5 * rate
    tol = 1e-9 * plan.df
    kept = freqs <= nyquist + tol
    if not np.all(kept) and source_spectrum is None:
        dropped = spectra[:, ~kept]
        if np.any(np.isfinite(dropped) & (dropped != 0.0)):
            warnings.warn(
                f"{who}: the band reaches {freqs[-1]:g} Hz, above the "
                f"Nyquist frequency {nyquist:g} Hz of the "
                f"{rate:g} Hz output; the part of H above it cannot be "
                f"sampled at that rate and is dropped.",
                NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)
    weights = np.where((np.abs(freqs) <= tol)
                       | (np.abs(freqs - nyquist) <= tol), 0.5, 1.0)
    analytic = band_sum(spectra[:, kept] * weights[kept], freqs[kept],
                        sample_rate=rate, t_start=t_start,
                        n_samples=plan.n_samples)
    return 2.0 * analytic.real, record_edge_levels(analytic)


def refuse_record_shorter_than_waveform(df: float, wf: np.ndarray,
                                        sample_rate: float, who: str) -> None:
    """Refuse a ``1/df`` record that cannot hold the waveform ``wf``.

    A Fourier synthesis on a grid of spacing ``df`` samples the waveform's
    spectrum every ``df``, which is the spectrum of the waveform folded
    with period ``1/df`` (Oppenheim, Willsky & Nawab, *Signals and
    Systems*, Problem 5.53(b): transform samples coarser than the signal's
    duration correspond to more than one sequence), and the received record
    is periodic with that period (JKPS eq. 8.9). A waveform whose support —
    first to last non-zero sample — is longer than the record therefore
    comes back folded onto itself, and nothing in the result shows it. The
    convolution routes hold a signal of any length, so the refusal names
    them."""
    nonzero = np.flatnonzero(wf)
    support = int(nonzero[-1] - nonzero[0] + 1) if nonzero.size else 0
    record = float(sample_rate) / float(df)
    if support > record + _LATTICE_TOLERANCE:
        raise ConfigurationError(
            f"{who}: the {1.0 / df:.4g} s record (1/Δf, Δf = {df:.4g} Hz) is "
            f"shorter than the {support / float(sample_rate):.4g} s source "
            f"waveform ({support} samples against {record:.6g}), so a "
            f"Fourier synthesis on this grid returns the waveform folded "
            f"onto itself, not the received signal.",
            remediation=(f"Convolve instead, which holds a signal of any "
                         f"length: Field.impulse_response(sample_rate) and "
                         f".apply(), or acoustic_signal."
                         f"impulse_response_from_transfer_function and "
                         f"apply_impulse_response on arrays; or refine the "
                         f"grid to Δf <= {float(sample_rate) / support:.4g} "
                         f"Hz."))


def band_energy_share(source_spectrum, freqs, wf, sample_rate,
                      weighting=None) -> float:
    """The share of the waveform ``wf``'s energy inside the band ``freqs``:
    ``2 Σ |S(f_k)|² Δf`` over the band (a 0 Hz bin counted once) against
    ``Σ wf² / sample_rate``, Parseval's two sides for a real signal, at most
    1. ``source_spectrum`` is the continuous spectrum on ``freqs``
    (:func:`~uacpy.acoustic_signal.spectrum_at.waveform_spectrum_at`).

    With ``weighting`` (any form :func:`~uacpy.acoustics.frequency_weighting`
    takes) it is the share of the WEIGHTED energy, the quantity a weighted
    exposure integrates (Southall et al. 2019: the weighted spectrum
    integrated across frequency): both sides weighted by ``10**(W/10)``,
    the whole waveform's on an FFT four times its length, at least."""
    freqs = np.asarray(freqs, dtype=float)
    wf = np.asarray(wf, dtype=float).ravel()
    total = float(np.sum(wf ** 2)) / sample_rate
    if freqs.size < 2 or not total > 0.0:
        return 1.0
    power = np.abs(np.asarray(source_spectrum)) ** 2
    if weighting is not None:
        # Deferred: core.acoustics loads its own dependencies on first use.
        from uacpy.core.acoustics.weighting import frequency_weighting
        power = power * 10.0 ** (frequency_weighting(freqs, weighting) / 10.0)
        nfft = 1 << int(np.ceil(np.log2(max(4 * wf.size, 16))))
        grid = np.fft.rfftfreq(nfft, 1.0 / float(sample_rate))
        # One-sided Parseval: DC and Nyquist once, every other bin twice.
        once = (grid == 0.0) | (grid == grid[-1])
        density = (np.abs(np.fft.rfft(wf, nfft)) ** 2
                   * 10.0 ** (frequency_weighting(grid, weighting) / 10.0))
        total = float(np.sum(np.where(once, 1.0, 2.0) * density)) / (
            nfft * float(sample_rate))
        if not total > 0.0:
            return 1.0
    twice = np.where(freqs > 0.0, 2.0, 1.0)
    band = float(np.sum(twice * power)) * float(freqs[1] - freqs[0])
    return min(band / total, 1.0)


def waveform_synthesis_setup(freqs, wf, sample_rate, *, window, who,
                              exposure=False, weighting=None):
    """``(source_spectrum, plan)`` for synthesising a waveform through
    ``H(f)`` on ``freqs``: the checks, the exact source spectrum on the axis
    and the shared grid at ``sample_rate`` (:func:`synthesis_grid`). The
    one setup behind :meth:`Field.synthesize_time_series` and
    :func:`uacpy.acoustic_signal.synthesize_time_series`.

    A record ``1/Δf`` shorter than the waveform is refused
    (:func:`refuse_record_shorter_than_waveform`). That says nothing about
    the CHANNEL: how long the arrivals ring is not visible in ``H(f)``
    sampled every ``Δf``, and a late arrival folds to a fixed place
    mid-record — so that check lives where the arrivals are
    (``Arrivals.synthesis_band``; Bellhop's BROADBAND run on its default
    grid). ``exposure`` words the band-edge warning for an exposure, which
    integrates every record (:func:`warn_band_edge_cuts_spectrum`), its
    share of the energy judged under the exposure's ``weighting``."""
    n_src = wf.size
    if n_src < 2:
        raise ConfigurationError(
            f"{who}: source_waveform must have at least 2 samples; got "
            f"{n_src}.")
    # NaN-closed (``not (sr > 0)``, not ``sr <= 0``): nan compares False
    # against both bounds, so the plain inequality passes it through to the
    # record length, which raises a raw ValueError instead of this typed one.
    if not np.isfinite(sample_rate) or not (sample_rate > 0):
        raise ConfigurationError(
            f"{who}: sample_rate must be positive and finite; got "
            f"{sample_rate}.")
    freqs = np.asarray(freqs, dtype=float)
    if freqs.size > 1:
        # Refuse a non-uniform axis HERE, not where the grid below reaches
        # it: the source spectrum is evaluated on this axis first, and its
        # chirp-z contour assumes exactly the grid the grid's guard
        # describes. (With fewer than 2 frequencies there is no spacing to
        # check; the grid raises on the count itself.)
        from uacpy.acoustic_signal.channel import _uniform_frequency_step
        df = _uniform_frequency_step(freqs, who=who)
        refuse_record_shorter_than_waveform(df, wf, sample_rate, who)
    from uacpy.acoustic_signal.spectrum_at import waveform_spectrum_at
    source_spectrum = waveform_spectrum_at(wf, sample_rate, freqs)
    plan = synthesis_grid(freqs, window=window, sample_rate=sample_rate,
                          who=who)
    warn_band_edge_cuts_spectrum(
        source_spectrum, window, who, freqs=freqs, waveform=wf,
        sample_rate=sample_rate,
        energy_share=(band_energy_share(source_spectrum, freqs, wf,
                                        sample_rate, weighting)
                      if exposure else None),
        weighting=weighting)
    return source_spectrum, plan


def synthesize_cells(spectra, source_spectrum, plan, t_start, *, who):
    """Synthesise every row of ``spectra`` ``(M, n_f)`` on ``plan``:
    returns ``(traces (M, n_samples), elapsed, head_dB, tail_dB)``, sample
    ``n`` of row ``m`` at ``t_start[m] + elapsed[n]`` (``t_start`` a scalar
    or one start per row; ``elapsed = n / plan.sample_rate``).

    Chunked over the cell axis so the ``(cells × n_samples)`` complex
    scratch stays bounded (~4M elements ≈ 64 MB) however many cells there
    are.
    """
    n_cells = spectra.shape[0]
    n_samples = plan.n_samples
    starts = np.broadcast_to(np.asarray(t_start, dtype=float), (n_cells,))
    out = np.empty((n_cells, n_samples), dtype=np.float64)
    head_dB = np.full(n_cells, -np.inf)
    tail_dB = np.full(n_cells, -np.inf)
    chunk = max(1, _BLOCK_ELEMENTS // max(n_samples, 1))
    for a in range(0, n_cells, chunk):
        idx = np.arange(a, min(a + chunk, n_cells))
        out[idx], (head_dB[idx], tail_dB[idx]) = synthesize_traces(
            spectra[idx], plan, source_spectrum=source_spectrum,
            t_start=starts[idx], who=who)
    return out, np.arange(n_samples) / plan.sample_rate, head_dB, tail_dB


def record_start(range: float, record_s: float, *, bandwidth: float,
                 c_max: float, c0: float, c_slow: float, who: str,
                 fold_report: Optional[dict] = None) -> float:
    """Start of a synthesis record ``record_s`` seconds long over a band
    ``bandwidth`` Hz wide for a cell at ``range``: :func:`record_lead`
    before the estimated first arrival, the rest of it after.

    The estimate anchors on the faster of the stamped speeds ``c_max``
    (the fastest speed of the paths the producer models, or of the
    waveguide its run resolved) and ``c0`` (a surface water speed), 0
    meaning not stamped; it warns when there is neither, or only ``c0``
    while the record is short against the travel time. It warns too when
    the record ends before ``range / c_slow``, the arrival at the slowest
    stated water speed (0: none stated): the later paths then fold onto
    the record's start. With ``fold_report`` that finding is stored there
    (``'late_arrival'``: the record's start and end, the arrival time, the
    speed and the range) for the caller to say with what it measures on
    the record, instead of warned. ``t_start=`` on the caller replaces this
    placement.
    """
    # The earliest arrival travels at the FASTEST speed in the waveguide,
    # so r/c_fast bounds it from below. Candidates are physical speeds:
    # 'c_max', the fastest speed of the paths the producer models, which
    # a producer states as ``Field.speeds.water_max`` when it differs from
    # the waveguide's (Bellhop traces the water alone), else the fastest
    # speed of the waveguide the run resolved
    # (``Field.speeds.waveguide_max``); and 'c0' (Bellhop, the sea-surface
    # water speed ``Field.speeds.surface``).
    # Anchoring on a speed above c_fast opens the window too early: the
    # error adds to the lead, the late multipath
    # tail falls past the end of the record and wraps to the beginning
    # — so no algorithmic speed (e.g. a PE expansion point) may enter
    # this max, and c_min never binds it.
    # The 1500 m/s default is a fallback for when nothing physical was
    # stamped, never a candidate beside a stamped speed: a stamped speed
    # BELOW it (cold or fresh water) must win, or the window opens early by
    # r·(1/c − 1/1500) and the arrival wraps a whole record with the time
    # axis mislabelled and nothing to show for it.
    stamped = [speed for speed in (c_max, c0) if speed > 0.0]
    anchor_speed = max(stamped) if stamped else DEFAULT_SOUND_SPEED
    travel = range / anchor_speed
    # The anchor is a lower bound on the travel time, so the lead only has
    # to hold the band-limited onset of the first path; the rest of the
    # record holds the multipath behind it.
    lead = record_lead(record_s, bandwidth)
    t_start = max(0.0, travel - lead)
    if (c_slow > 0.0 and t_start + record_s < range / c_slow
            and fold_report is not None):
        fold_report['late_arrival'] = (t_start, t_start + record_s,
                                       range / c_slow, c_slow, range)
    elif c_slow > 0.0 and t_start + record_s < range / c_slow:
        warnings.warn(
            f"{who}: the {record_s:.3g}s record [{t_start:.4g}, "
            f"{t_start + record_s:.4g}] s ends before r/c = "
            f"{range / c_slow:.4g} s, the arrival at the slowest water "
            f"speed ({c_slow:g} m/s) at {range:.0f} m, so every later path "
            f"folds onto the record's start. Lengthen the record (a finer "
            f"frequency grid; output_duration= on a TIME_SERIES run) or "
            f"place it with t_start=.",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    if not c_max and not c0 and t_start > 0.0:
        # With no stamped speed at all the anchor is the 1500 m/s
        # default, which bounds nothing: a fast seabed (head waves at
        # 2-6 km/s) puts the earliest arrival well before r/1500, past
        # any lead the window can offer.
        warnings.warn(
            f"{who}: the response stamped no sound speed (a Field's "
            f"speeds state no water_max, waveguide_max or surface; a "
            f"BeamformedField carries none), so the window is anchored on "
            f"the {DEFAULT_SOUND_SPEED:g} m/s default at t_start="
            f"{t_start:.3g}s. Any path faster than that (e.g. a head "
            f"wave in a fast seabed) arrives before the window and "
            f"wraps to the end of the record. Pass t_start= to pin the "
            f"window start.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    # With only a c0 the anchor carries the fast/slow path spread as
    # error, taken as 5 % (_ASSUMED_PATH_SPEED_SPREAD, uacpy guidance);
    # when it exceeds the lead the first arrival can fall before the
    # window and wrap to the end of the record.
    #
    # Where the 5 % lives, exactly: c0 is a SURFACE sample, so it is the
    # fastest water speed only on a downward-refracting profile. The
    # window wraps when r(1/c0 - 1/c_fast) > T/2, and this test fires when
    # 0.05·r/c0 > T/2, so it covers the wrap iff c0 >= 0.95·c_fast — i.e.
    # while the surface sample is within 5 % of the profile maximum. An
    # upward-refracting column with a wider spread (a cold surface over a
    # deep sound channel) leaves a band of window lengths uncovered.
    elif not c_max and t_start > 0.0 and \
            _ASSUMED_PATH_SPEED_SPREAD * travel > lead:
        warnings.warn(
            f"{who}: the {record_s:.3g}s synthesis window is "
            f"short against the {travel:.3g}s travel time at "
            f"{range:.0f} m, and the model reported no maximum "
            f"sound speed, so the window start is an estimate — the "
            f"earliest arrival may fall before it and wrap to the end of "
            f"the record. Pass t_start= to pin it, or refine the "
            f"frequency grid (the window is 1/Δf).",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    return t_start


def reduced_record_starts(ranges, record_s: float, *, bandwidth: float,
                          c_max: float, c0: float, c_slow: float, who: str
                          ) -> Tuple[np.ndarray, float, float]:
    """``(starts, reduction_speed, lead)`` of a running window over the
    range axis ``ranges`` (m).

    Each range's ``record_s``-long record opens :func:`record_lead` before
    ``|r| / reduction_speed``, so on the reduced time
    ``t − |r|/reduction_speed`` every range shares one axis,
    ``[−lead, record_s − lead)``: Jensen et al., *Computational Ocean
    Acoustics*, sect. 8.2.1.3, eq. 8.15, the record start "running" with
    range. The reduction speed is the anchor :func:`record_start` places a
    single record by: the faster of the stamped ``c_max`` and ``c0``, else
    the 1500 m/s default with a FallbackWarning. The record then has to
    hold only the dispersion, not the travel time across the range span.
    Eq. 8.16 sizes it ``T ≥ r_max·(1/u_min − 1/c_max)``, ``u_min`` the
    minimum modal group velocity over the band; that minimum (the Airy
    phase, the tail of a modal arrival, sect. 2.4.5.2) lies below the water
    speed and is not known here. What this checks instead is the arrival
    at the slowest stated water speed ``c_slow`` (0: none stated) at the
    farthest range: it warns when the record ends before
    ``r_max·(1/c_slow − 1/c_red)``. A silent check therefore does not show
    that an Airy-phase tail, or a steep bottom-bounce multiple slower than
    ``c_slow``, fits the record: the eq. 8.16 window "does not ensure
    inclusion of the steep-angle multiples. If these or other late arrivals
    are of interest, the time window must be extended" (sect. 8.2.2)."""
    distance = np.abs(np.atleast_1d(np.asarray(ranges, dtype=float)))
    stamped = [speed for speed in (c_max, c0) if speed > 0.0]
    speed = max(stamped) if stamped else DEFAULT_SOUND_SPEED
    lead = record_lead(record_s, bandwidth)
    starts = distance / speed - lead
    farthest = float(distance.max()) if distance.size else 0.0
    if c_slow > 0.0 and record_s - lead < farthest * (1.0 / c_slow
                                                      - 1.0 / speed):
        warnings.warn(
            f"{who}: the {record_s:.3g}s record ends "
            f"{record_s - lead:.4g} s after r/{speed:g} m/s, before the "
            f"arrival at the slowest water speed ({c_slow:g} m/s) at "
            f"{farthest:.0f} m, {farthest * (1.0 / c_slow - 1.0 / speed):.4g} "
            f"s after it, so the later paths fold onto the record's start. "
            f"Lengthen the record (a finer frequency grid; "
            f"output_duration= on a TIME_SERIES run).",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)
    if not stamped and farthest > 0.0:
        warnings.warn(
            f"{who}: the response stamped no sound speed (a Field's speeds "
            f"state no water_max, waveguide_max or surface), so the reduced "
            f"time is t − r/{DEFAULT_SOUND_SPEED:g} m/s, the default. Any "
            f"path faster than that (e.g. a head wave in a fast seabed) "
            f"arrives before its range's record and wraps to the end of it. "
            f"Pass a numeric t_start= to place one shared record instead.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
    elif not c_max and _ASSUMED_PATH_SPEED_SPREAD * farthest / speed > lead:
        warnings.warn(
            f"{who}: the reduction speed is the surface speed {speed:g} m/s, "
            f"and the model reported no maximum sound speed, so each "
            f"range's record start is an estimate — at {farthest:.0f} m a "
            f"faster path may arrive before it and wrap to the end of the "
            f"record. Refine the frequency grid (the record is 1/Δf).",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
    return starts, float(speed), float(lead)


def source_waveform_problem(source_waveform) -> Optional[str]:
    """Why ``source_waveform`` is not a usable transmitted pulse, or
    ``None`` when it is.

    The one rule every entry point that takes a source waveform applies —
    the Field synthesis methods, :func:`~uacpy.acoustic_signal.synthesize_time_series`
    and a model's TIME_SERIES run: a real, finite, 1-D array of pressure
    samples. The synthesis is ``2·Re`` of a one-sided sum over the waveform's
    spectrum, so a complex waveform has no meaning there, and a
    ``(time, signal)`` pair or a stacked 2-D array would be read as one
    flattened signal."""
    if isinstance(source_waveform, tuple):
        return ("source_waveform must be the 1-D signal, not a (time, "
                "signal) pair — pass lfm_chirp(...)[1].")
    wf = np.asarray(source_waveform)
    if np.iscomplexobj(wf):
        return ("source_waveform must be a real pressure pulse; got a "
                "complex array. Pass np.real(w) if its real part is the "
                "pulse you mean.")
    if wf.ndim != 1:
        return (f"source_waveform must be a 1-D signal; got an array of "
                f"shape {wf.shape}. The generators return a (time, signal) "
                f"pair, so pass lfm_chirp(...)[1].")
    if wf.dtype.kind not in 'biuf':
        return (f"source_waveform must hold numbers; got an array of dtype "
                f"{wf.dtype}.")
    if not np.all(np.isfinite(wf)):
        return "source_waveform contains non-finite values (NaN/inf)."
    return None


def require_source_waveform(source_waveform, who: str) -> np.ndarray:
    """``source_waveform`` as a float64 1-D array, refused with
    :func:`source_waveform_problem`'s reason when it is not a usable
    pulse."""
    problem = source_waveform_problem(source_waveform)
    if problem is not None:
        raise ConfigurationError(f"{who}: {problem}")
    return np.asarray(source_waveform, dtype=float)


def waveform_spectrum_on(frequencies, source_waveform, sample_rate,
                         source_spectrum, who: str):
    """``source_spectrum``, or with ``source_waveform`` given in its place,
    the waveform's exact spectrum on ``frequencies``
    (:func:`~uacpy.acoustic_signal.waveform_spectrum_at`). Refuses both at
    once, a waveform without ``sample_rate``, and a waveform that breaks the
    one rule (:func:`source_waveform_problem`)."""
    if source_waveform is None:
        return source_spectrum
    if source_spectrum is not None:
        raise ConfigurationError(
            f"{who}: pass either source_spectrum= (already on this "
            f"field's frequency axis) or source_waveform= (sampled in "
            f"time), not both.")
    if sample_rate is None:
        raise ConfigurationError(
            f"{who}: source_waveform= needs sample_rate= to have a "
            f"spectrum at all.")
    wf = require_source_waveform(source_waveform, who)
    from uacpy.acoustic_signal.spectrum_at import waveform_spectrum_at
    return waveform_spectrum_at(wf, sample_rate,
                                np.asarray(frequencies, dtype=float))
