"""The constant-Q transform (Brown 1991): geometric frequency bins, each a
fixed number of cycles long.

:func:`constant_q_transform` and :func:`constant_q_spectrogram` give the
coefficients and a time-resolved power panel; the kernels, the per-bin
powers and the histogram twin here are what
:func:`~uacpy.acoustic_signal.constant_q` and
:func:`~uacpy.acoustic_signal.probabilistic_constant_q` estimate with.
"""

from __future__ import annotations

import warnings
from collections import namedtuple
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import get_window
from uacpy.core.exceptions import ConfigurationError, NumericsWarning
from uacpy.core.constants import REFERENCE_PRESSURE_WATER
from uacpy.core.acoustics import power_to_dB
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core._validate import (
    require_at_most_nyquist, require_positive_finite_scalar,
    require_real_signal,
)
from uacpy.acoustic_signal._results import POWER_UNITS, PlottedResult
from uacpy.acoustic_signal.timefreq import SpectrogramResult
from uacpy.acoustic_signal.spectral import (
    ProbabilisticSpectralEstimate, _BIN_SCALINGS, _split_records,
    _warn_levels_outside_window, level_histogram,
)


class CQTResult(PlottedResult,
                namedtuple("CQTResult", "frequencies coefficients")):
    """Constant-Q transform: geometric ``frequencies`` and complex ``coefficients``.

    The tuple is the measurement, so ``frequencies, coefficients = ...``
    keeps working; :meth:`plot` is the one obvious way to draw it.
    """

    __slots__ = ()

    _plotter = "plot_constant_q_transform"
    _plot_fields = ("frequencies", "coefficients")

    def _field_units(self):
        return {"frequencies": "Hz", "coefficients": "Pa"}


class CQSpectrogramResult(PlottedResult,
                          namedtuple("CQSpectrogramResult",
                                     SpectrogramResult._fields)):
    """Constant-Q spectrogram: geometric ``frequencies``, ``times``, ``power``.

    The tuple is the measurement, so ``frequencies, times, power = ...``
    keeps working; :meth:`plot` is the one obvious way to draw it. What the
    panel holds rides on ``scaling`` (``'density'`` Pa²/Hz or
    ``'spectrum'`` Pa²), as on :class:`SpectrogramResult`, so the plot
    labels the unit the power is in.
    """

    _attrs = ("scaling",)
    _plotter = "plot_constant_q_spectrogram"
    _plot_fields = SpectrogramResult._plot_fields
    _plot_defaults = ("scaling",)

    def __new__(cls, frequencies, times, power, scaling="density"):
        self = super().__new__(cls, frequencies, times, power)
        self.scaling = str(scaling)
        return self

    def _field_units(self):
        return {"frequencies": "Hz", "times": "s",
                "power": POWER_UNITS[self.scaling]}


#: The constant-Q helpers speak the same two-valued vocabulary the bin
#: estimators do; an energy is not one of them (see
#: :func:`_require_bin_scaling`).
_SCALINGS = _BIN_SCALINGS


# ── shared kernel construction ──────────────────────────────────────────────
def _cq_frequencies(freq_min, freq_max, bins_per_octave):
    if freq_min <= 0 or freq_max <= freq_min:
        raise ConfigurationError(
            "constant-Q: require 0 < freq_min < freq_max; got "
            f"freq_min={freq_min}, freq_max={freq_max}.")
    n = int(np.floor(bins_per_octave * np.log2(freq_max / freq_min))) + 1
    return freq_min * 2.0 ** (np.arange(n) / float(bins_per_octave))


def _cq_quality(bins_per_octave):
    """Constant quality factor Q = 1 / (2**(1/B) - 1)."""
    return 1.0 / (2.0 ** (1.0 / float(bins_per_octave)) - 1.0)


def _cq_kernels(frequencies, Q, sample_rate, window):
    """List of ``(N_k, kernel, density_factor)`` per bin.

    ``kernel = w·exp(-2j·pi·f_k·n/sample_rate) / Σw`` (so ``|X|**2`` is band power, the
    'spectrum' scaling). ``density_factor = (Σw)**2 / (sample_rate·Σw**2)`` converts that
    band power to a one-sided PSD (per Hz), matching ``scipy.signal.welch``
    density scaling; the one-sided factor 2 is applied in
    :func:`_cq_band_power`.
    """
    kernels = []
    for fk in frequencies:
        Nk = max(1, int(np.ceil(Q * sample_rate / fk)))
        w = get_window(window, Nk, fftbins=True)
        sw = float(np.sum(w))
        sw2 = float(np.sum(w * w))
        n = np.arange(Nk)
        ker = (w * np.exp(-2j * np.pi * fk * n / sample_rate)) / sw
        # |X|**2 captures only the positive-frequency sideband of a real signal
        # (a tone A·cos gives |X|=A/2); _cq_band_power doubles it to one-sided
        # band power A**2/2. density_factor then divides by the noise-equivalent
        # bandwidth sample_rate·Σw²/(Σw)² to give a one-sided PSD (welch-density).
        density_factor = sw * sw / (sample_rate * sw2)
        kernels.append((Nk, ker.astype(np.complex128), density_factor))
    return kernels


# A bin whose one-sided band power is inflated by more than this much warns.
# The bias is 10*log10(1 + ratio**2) and rises smoothly with f_k/fs: measured
# on a Hann window at B=24 it is under 0.005 dB for every f_k/fs <= 0.4865,
# first crosses 0.01 dB at f_k/fs ~ 0.4875, and reaches 3.01 dB at f_k = fs/2.
# 3.01 dB is the phase average, not a bound: at exactly fs/2 the image lands on
# DC and the reading follows the tone's own phase as 10*log10(4 cos**2 phi),
# so a cosine reads +6.02 dB and a sine is identically zero on the grid.
_CQ_IMAGE_BIAS_WARN_DB = 0.01


def _cq_image_ratio(frequencies, sample_rate, kernels):
    """``|W(2 f_k)| / sum(w)`` per bin — the negative-frequency image leak.

    A real tone at ``f_k`` carries a ``+f_k`` and a ``-f_k`` component. The
    kernel demodulates the first to DC and the second to ``-2 f_k``, where the
    window's own transfer function ``W`` attenuates it; this ratio is what
    survives. It inflates the one-sided band power ``2|X|**2`` by
    ``1 + ratio**2`` averaged over frame phase, so the estimators read a
    coherent tone high by that factor. The ratio is negligible while
    ``2 f_k`` sits well outside the window's main lobe and rises to 1 at
    ``f_k = sample_rate/2``, where the image lands on DC. That last point is the one
    place the frame average does not apply: an image on DC keeps the same
    phase in every frame, so the reading there is ``10*log10(4 cos**2 phi)``
    in the tone's own phase (+6.02 dB for a cosine) rather than the 3.01 dB
    the ``1 + ratio**2`` average gives.

    Broadband noise is untouched: its band power is ``sigma**2 * sum(w**2) /
    sum(w)**2`` at every bin, independent of ``f_k``, so the correct remedy is
    to report the affected bins rather than divide the bias out.
    """
    ratios = np.empty(len(kernels))
    for i, (Nk, ker, _) in enumerate(kernels):
        n = np.arange(Nk)
        image = np.sum(ker * np.exp(-2j * np.pi * frequencies[i] * n / sample_rate))
        ratios[i] = abs(image)
    return ratios


def _cq_frame(x, center, kernels):
    """``(coeffs, valid)``: complex constant-Q band amplitudes with each bin's
    window centred on sample ``center`` (zero-padded at the signal edges), and a
    per-bin bool — ``True`` where the window lay fully inside the signal."""
    nx = x.size
    coeffs = np.empty(len(kernels), dtype=np.complex128)
    valid = np.empty(len(kernels), dtype=bool)
    for i, (Nk, ker, _) in enumerate(kernels):
        start = center - Nk // 2
        stop = start + Nk
        inside = start >= 0 and stop <= nx
        valid[i] = inside
        if inside:
            coeffs[i] = np.sum(x[start:stop] * ker)
        else:
            a, b = max(start, 0), min(stop, nx)
            seg = np.zeros(Nk, dtype=float)
            seg[a - start: b - start] = x[a:b]
            coeffs[i] = np.sum(seg * ker)
    return coeffs, valid


#: Frame samples one block of a bin's frame matrix holds (32 MiB of float64),
#: so a long record is correlated in bounded memory: the frames overlap, and
#: gathering them all at once would copy the record four times per bin.
_CQ_BLOCK_SAMPLES = 1 << 22


def _cq_step(Nk):
    """Samples between one bin's frames: a quarter of its kernel.

    The estimators average POWER, so what has to tile the record is the
    squared window. Hann² stepped by a quarter of its length sums to a
    constant, so every sample carries the same weight in the bin's average and
    a transient counts once wherever it falls. A half-length step tiles Hann
    itself but leaves Hann² rippling between 0.5 and 1 (3 dB).
    """
    return max(1, Nk // 4)


def _cq_coefficients(x, starts, Nk, ker):
    """Complex band amplitudes of the ``Nk``-sample frames of ``x`` that start
    at ``starts``; every frame lies inside ``x``."""
    frames = sliding_window_view(x, Nk)
    basis = np.stack([ker.real, ker.imag], axis=1)
    coeffs = np.empty(starts.size, dtype=np.complex128)
    block = max(1, _CQ_BLOCK_SAMPLES // Nk)
    for a in range(0, starts.size, block):
        re_im = frames[starts[a:a + block]] @ basis
        coeffs[a:a + block] = re_im[:, 0] + 1j * re_im[:, 1]
    return coeffs


def _cq_band_power(coeffs, density_factor, scaling):
    """One-sided band power ``2|X|**2`` (a tone reads ``A**2/2``), or that
    power per hertz under ``scaling='density'``.

    There is no exposure here: an energy is :func:`sound_exposure`'s, whose
    bins are orthogonal so that a band sum IS an energy. Constant-Q kernels
    overlap by construction, so no factor turns these into one.
    """
    p = 2.0 * np.abs(coeffs) ** 2
    return p * density_factor if scaling == "density" else p


def _cq_bin_powers(x, kernels, scaling):
    """Per bin, the band power of every frame lying fully inside ``x``.

    Each bin steps by :func:`_cq_step` of its own kernel, so every sample of
    the record reaches every bin. Returns one 1-D array per bin, empty for a
    bin whose kernel is longer than the record. The frames of a bin sit on a
    grid centred in the record, the leftover samples split between its ends.
    """
    powers = []
    for Nk, ker, density_factor in kernels:
        if Nk > x.size:
            powers.append(np.empty(0))
            continue
        step = _cq_step(Nk)
        n_frames = (x.size - Nk) // step + 1
        offset = (x.size - Nk - (n_frames - 1) * step) // 2
        starts = offset + step * np.arange(n_frames)
        powers.append(_cq_band_power(_cq_coefficients(x, starts, Nk, ker),
                                     density_factor, scaling))
    return powers


def _cq_spectrogram_cells(x, sample_rate, kernels, hop, scaling):
    """``(times, power)``: constant-Q power on cells ``hop`` samples wide.

    The cells are centred on ``arange(0, n, hop)``. A bin whose own step
    (:func:`_cq_step`) is shorter than ``hop`` averages ``ceil(hop/step)``
    frames spaced evenly across each cell, so every sample reaches every bin;
    a longer bin reads the one frame centred on the cell, whose kernel already
    spans it. Frames reaching past the record are zero-padded, as on
    :func:`constant_q_transform`.
    """
    centers = np.arange(0, x.size, hop)
    pad = max(Nk for Nk, _, _ in kernels) // 2 + hop + 1
    padded = np.pad(x, pad)
    power = np.empty((len(kernels), centers.size), dtype=float)
    for i, (Nk, ker, density_factor) in enumerate(kernels):
        per_cell = max(1, -(-hop // _cq_step(Nk)))
        offsets = (np.arange(per_cell) + 0.5) * hop / per_cell - hop / 2.0
        frame_centers = np.rint(centers[:, None] + offsets[None, :])
        starts = frame_centers.astype(np.int64).ravel() - Nk // 2 + pad
        p = _cq_band_power(_cq_coefficients(padded, starts, Nk, ker),
                           density_factor, scaling)
        power[i] = p.reshape(centers.size, per_cell).mean(axis=1)
    return centers / sample_rate, power


def _check_scaling(scaling, who):
    if scaling not in _SCALINGS:
        raise ConfigurationError(
            f"{who}: scaling must be one of {_SCALINGS}; got {scaling!r}.")


def _resolve_hop(hop, kernels, n_samples, who):
    """The spectrogram's cell width in samples: ``hop``, or by default an
    eighth of the longest kernel, capped so a short record still has eight
    cells."""
    n_lowest = kernels[0][0]
    if hop is None:
        hop = max(1, min(n_lowest // 8, max(1, n_samples // 8)))
    hop = int(hop)
    if hop < 1:
        raise ConfigurationError(f"{who}: hop must be >= 1; got {hop}.")
    return hop


def _cq_setup(data, sample_rate, freq_min, freq_max, bins_per_octave, window, who,
              drops_short_bins):
    """Validate, build the kernel bank and warn about bins that never fit;
    ``drops_short_bins`` says whether ``who`` drops such a bin (the
    averaging estimators) or evaluates it on a zero-padded window."""
    x = require_real_signal(
        data, who, why="; a complex array would be silently real-cast.")
    fs = require_positive_finite_scalar(sample_rate, who,
                                        "sample_rate", " Hz")
    fmax_given = freq_max is not None
    if freq_max is None:
        freq_max = fs / 2.0
    # The analyser side of the Nyquist split: freq_max == fs/2 is admitted
    # because a caller may ask for exactly that bin.
    require_at_most_nyquist(freq_max, fs, who, "freq_max",
                            "the bin has no signal to analyse")
    freqs = _cq_frequencies(freq_min, freq_max, bins_per_octave)
    Q = _cq_quality(bins_per_octave)
    kernels = _cq_kernels(freqs, Q, fs, window)
    n_lowest = kernels[0][0]
    if n_lowest > x.size:
        fate = ("dropped from the average" if drops_short_bins else
                "evaluated on a zero-padded window and read low there")
        # Every public constant-Q estimator funnels through this setup helper,
        # so the frame the user wrote is two deep here, not one: a
        # hand-counted ``stacklevel=2`` named this module's own call line for
        # all four of them (measured). ``skip_file_prefixes`` walks out to the
        # first frame outside the package instead, which is the caller's own
        # line whichever estimator they chose.
        warnings.warn(
            f"{who}: lowest bin needs {n_lowest} samples (Q·fs/freq_min) but the "
            f"signal has {x.size}; low-frequency bins never fit a full window "
            f"and are {fate}. Raise freq_min or lengthen the signal.",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)
    bias_dB = 10.0 * np.log10(1.0 + _cq_image_ratio(freqs, fs, kernels) ** 2)
    hot = np.flatnonzero(bias_dB > _CQ_IMAGE_BIAS_WARN_DB)
    if not fmax_given and hot.size and hot[0] > 0:
        # Unset, the range ends at the last bin below the first one whose
        # image leaks more than _CQ_IMAGE_BIAS_WARN_DB, so the default call
        # reads tone levels it can stand behind and has nothing to warn
        # about. A caller who asks for more names freq_max, and is warned.
        keep = int(hot[0])
        freqs, kernels = freqs[:keep], kernels[:keep]
        hot = hot[:0]
    if hot.size:
        # Reached only for a caller's explicit ``freq_max`` (an unset one stops
        # below this region), so this is a statement about that ``freq_max``
        # and has to name the caller's line. Same frame walk
        # as the short-signal warning above and for the same reason: measured,
        # ``stacklevel=2`` here names four *different* lines of this module,
        # one per public estimator, and never the caller's — which also
        # collapses every call site onto one dedup key.
        warnings.warn(
            f"{who}: {hot.size} bin(s) above {freqs[hot[0]]:.4g} Hz sit "
            f"close enough to Nyquist that the tone's negative-frequency "
            f"image leaks through the analysis window: a coherent tone in "
            f"those bins reads high by up to {bias_dB[hot].max():.2f} dB "
            f"averaged over frame phase (a bin at exactly fs/2 follows the "
            f"tone's own phase instead: +6.02 dB for a cosine) "
            f"(highest bin {freqs[-1]:.4g} Hz, f/fs = {freqs[-1] / fs:.4f}). "
            f"Broadband noise in the same bins is unaffected. Lower freq_max to "
            f"read tone levels there.",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)
    return x, fs, freqs, kernels


# ── public API ──────────────────────────────────────────────────────────────
def constant_q_transform(data, sample_rate, *, freq_min=20.0, freq_max=None,
                         bins_per_octave=24, window="hann"):
    """Complex constant-Q spectrum of one frame centred on the signal.

    Returns a :class:`CQTResult` ``(frequencies, coefficients)`` — the centre
    frequencies (Hz) and complex band amplitudes (``abs`` is the magnitude
    spectrum). Best for short transients; for longer signals use
    :func:`constant_q_spectrogram`, or :func:`constant_q`. An unset ``freq_max``
    stops below the near-Nyquist bins whose image leaks into a tone's level,
    as on :func:`constant_q`.

    Parameters
    ----------
    data : array_like
        The signal; one frame centred on it is analysed.
    sample_rate : float
        Sample rate (Hz).
    freq_min, freq_max : float, optional
        First and last bin centre (Hz); ``None`` for ``freq_max`` stops below
        the near-Nyquist bins (see above). Default ``freq_min`` 20 Hz.
    bins_per_octave : int, optional
        Bins per octave, which fixes Q. Default 24.
    window : str, optional
        Kernel taper. Default ``'hann'``.
    """
    x, fs, freqs, kernels = _cq_setup(
        data, sample_rate, freq_min, freq_max, bins_per_octave, window,
        "constant_q_transform", drops_short_bins=False)
    coeffs, _ = _cq_frame(x, x.size // 2, kernels)
    return CQTResult(freqs, coeffs)


def constant_q_spectrogram(data, sample_rate, *, freq_min=20.0, freq_max=None,
                           bins_per_octave=24, hop=None, window="hann",
                           scaling="density"):
    """Constant-Q spectrogram: constant-Q power over time.

    Reports power on cells ``hop`` samples wide (default: 1/8 of the longest
    window, i.e. of the lowest-frequency bin), centred on ``times``. Every
    sample reaches every bin: a bin whose kernel is shorter than a cell
    averages frames spaced at most a quarter kernel apart across it, so a
    transient between two cell centres still shows in its cell, and power
    times the cell width summed over the cells is the energy the bin saw.
    ``scaling='density'``
    (default, as on :func:`constant_q` and :func:`spectrogram`) returns a
    one-sided PSD per Hz; ``scaling='spectrum'`` returns one-sided band power
    ``2|X_cq|**2`` (a tone of amplitude ``A`` reads ``A**2/2``). An unset
    ``freq_max`` stops below the near-Nyquist bins whose image leaks into a
    tone's level, as on :func:`constant_q`. Returns a :class:`CQSpectrogramResult`
    ``(frequencies, times, power)`` with ``power`` shaped ``(n_freqs,
    n_frames)`` — every frame is kept, so edge frames carry visible edge
    effects.

    Parameters
    ----------
    data : array_like
        Pressure record (Pa).
    sample_rate : float
        Sample rate (Hz).
    freq_min, freq_max : float, optional
        First and last bin centre (Hz); ``None`` for ``freq_max`` stops below
        the near-Nyquist bins (see above). Default ``freq_min`` 20 Hz.
    bins_per_octave : int, optional
        Bins per octave, which fixes Q. Default 24.
    hop : int, optional
        Cell width in samples; ``None`` is an eighth of the longest kernel.
    window : str, optional
        Kernel taper. Default ``'hann'``.
    scaling : {'density', 'spectrum'}, optional
        PSD per Hz or band power. Default ``'density'``.
    """
    _check_scaling(scaling, "constant_q_spectrogram")
    x, fs, freqs, kernels = _cq_setup(
        data, sample_rate, freq_min, freq_max, bins_per_octave, window,
        "constant_q_spectrogram", drops_short_bins=False)
    hop = _resolve_hop(hop, kernels, x.size, "constant_q_spectrogram")
    times, power = _cq_spectrogram_cells(x, fs, kernels, hop, scaling)
    return CQSpectrogramResult(freqs, times, power, scaling=scaling)


def _constant_q_estimate(data, sample_rate, *, window, scaling, who,
                         freq_min=20.0, freq_max=None, bins_per_octave=24):
    """Time-averaged constant-Q power per geometric bin, as ``(freqs, power)``.

    What :func:`constant_q` returns once it has wrapped this in a
    :class:`SpectralEstimate`. Each bin averages its own frames, stepped by a
    quarter of its kernel (:func:`_cq_step`) so that every sample counts
    once, and only frames whose window lies fully inside the signal; a bin
    with no such frame is ``NaN``.
    """
    _check_scaling(scaling, who)
    x, fs, freqs, kernels = _cq_setup(
        data, sample_rate, freq_min, freq_max, bins_per_octave, window,
        who, drops_short_bins=True)
    avg = np.array([p.mean() if p.size else np.nan
                    for p in _cq_bin_powers(x, kernels, scaling)])
    return freqs, avg


def _probabilistic_constant_q(data, sample_rate, *, window, scaling, who,
                              freq_min=20.0, freq_max=None, bins_per_octave=24,
                              level_step_dB=1.0, level_min_dB=0, level_max_dB=150,
                              ref=REFERENCE_PRESSURE_WATER):
    """Probability density of constant-Q power levels over time.

    Histograms the per-bin dB levels of the constant-Q frames, the constant-Q
    analogue of :func:`probabilistic_welch` — note the two histogram
    different populations: each sample here is a *single unaveraged frame*,
    whereas ``probabilistic_welch`` histograms averages over ``segment_duration``
    chunks,
    so the level spread here is wider for the same signal, and ``mean_dB``
    — the mean of those dB levels — sits ``10*gamma/ln(10)`` = 2.51 dB below
    the power mean :func:`constant_q` returns from the same record
    (measured 2.506 +/- 0.003 dB, the median over bins, for four seeds of
    60 s of white noise at 8 kHz and ``bins_per_octave=24``; single bins
    scatter about it by up to 0.5 dB below 100 Hz, where a bin holds fewer
    independent frames). That figure is the two-degrees-of-freedom case:
    a bin essentially *at* Nyquist loses its quadrature component, so
    ``|X|**2`` tends toward one dof and the offset climbs toward
    ``10*(gamma+ln2)/ln(10)`` = 5.52 dB — measured 5.24 dB at
    ``f_k/fs = 0.4995`` (same four seeds). This
    does not contradict "broadband noise is not affected" above: the band
    *power* there is unchanged, and it is the shape of its distribution, hence
    the mean of the logs, that moves. ``probabilistic_welch`` carries the same bias wherever
    its own ``nperseg`` is clamped to the ``segment_duration`` chunk, which is one
    look too. Compare a target curve against ``constant_q``; ``mean_dB``
    is the centre of the histogram. Only frames whose
    window lay fully inside the signal contribute (per bin). Returns a
    :class:`ProbabilisticSpectralEstimate` — ``(frequencies, level_edges,
    pdf)`` carrying ``.mean_dB``, ``.std_dB``, ``.ref``, ``.scaling`` and
    ``.method='constant_q'``;
    ``pdf`` is shaped ``(n_levels, n_freqs)`` and density-normalised per
    frequency column (an empty bin is 0; a bin with no level inside the
    window is a ``NaN`` column). With ``scaling='density'`` the
    levels are PSD levels (dB re ref²/Hz) rather than band-power levels
    (dB re ref²). The ``ref`` those levels are stated against comes back in the
    result rather than having to be assumed downstream — a consumer that
    hardcodes the package default is 120 dB out for a caller working in µPa.
    """
    _check_scaling(scaling, who)
    # Every record's frames go into the one histogram per bin, as every
    # segment does on the Welch twin; the kernel bank depends only on the
    # rate and the ladder, so each record yields the same bins.
    pooled = None
    for record in _split_records(data, who):
        x, fs, freqs, kernels = _cq_setup(
            record, sample_rate, freq_min, freq_max, bins_per_octave, window,
            who, drops_short_bins=True)
        powers = _cq_bin_powers(x, kernels, scaling)
        pooled = powers if pooled is None else [
            np.concatenate(pair) for pair in zip(pooled, powers)]
    level_edges = np.arange(level_min_dB, level_max_dB + level_step_dB, level_step_dB)
    pdf = np.zeros((level_edges.size - 1, freqs.size))
    mean_dB = np.full(freqs.size, np.nan)
    std_dB = np.full(freqs.size, np.nan)
    for i in range(freqs.size):
        with np.errstate(divide="ignore"):
            vals = power_to_dB(pooled[i], ref)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            pdf[:, i] = np.nan
            continue
        pdf[:, i], mean_dB[i], std_dB[i] = level_histogram(vals, level_edges)
    with np.errstate(divide="ignore"):
        _warn_levels_outside_window(
            who, [power_to_dB(p, ref) for p in pooled], freqs, level_edges)
    # The reference, the scaling and the method are part of what the levels
    # MEAN, so they travel with them rather than being restated by every
    # consumer. ``segment_duration`` is None: each sample is one unaveraged frame,
    # and a frame here is per bin rather than one duration for the estimate.
    return ProbabilisticSpectralEstimate(
        freqs, level_edges, pdf, mean_dB=mean_dB, std_dB=std_dB,
        level_step_dB=level_step_dB, segment_duration=None, ref=float(ref),
        scaling=str(scaling), method="constant_q")
