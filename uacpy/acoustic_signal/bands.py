"""The standard band ladders a level is reported on.

:func:`octave_bands` and :func:`decidecade_bands` (IEC 61260-1 base-10)
give the edges of each ladder;
:func:`standard_bands` selects the bands a range is reported on and
:func:`band_levels` integrates a spectral density over them. The same ladders
are the bands :func:`~uacpy.acoustic_signal.sound_exposure` is written on.
"""

from __future__ import annotations

import math
import warnings
from collections import namedtuple

import numpy as np
from uacpy.core.acoustics import integrate_psd
from uacpy.core.acoustics.levels import BAND_EDGE_RTOL
from uacpy.core.exceptions import ConfigurationError, NumericsWarning
from uacpy.core.constants import REFERENCE_PRESSURE_WATER
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core._validate import (
    require_finite_signal, require_increasing_axis,
    require_positive_finite_scalar,
)
from uacpy.acoustic_signal._results import PlottedResult, level_unit
from uacpy.acoustic_signal.spectral import (
    _BAND_WINDOW, _TARGET_BATCH_SAMPLES, _estimate,
)


#: Reference frequency every ladder is anchored at [Hz]. It is a band centre
#: in the base-2 and the base-10 systems alike, which is what makes a band
#: table comparable with anyone else's: "1, 10, 100, 1000, 10,000 Hz … are
#: also standard 1/3-octave-band f_o's" (Pierce, *Acoustics*, §2.2.1;
#: IEC 61260-1).
_REF_FREQ = 1000.0


#: The ladder each :data:`~uacpy.acoustic_signal.spectral.BAND_TYPES` name
#: stands for, as ``(bands per octave, base)``: both base 10, the one base
#: IEC 61260-1:2014 specifies (octave ratio ``G = 10**(3/10)``). The base-10
#: third-octave is the decidecade, a tenth of a decade — ISO 18405's term,
#: "the nearly equivalent and preferred decidecade" to the third-octave
#: (Abraham, *Underwater Acoustic Signal Processing*, §3.3.1.1) — so it has
#: that one name, and one octave is exactly three decidecades.
_LADDERS = {"octave": (1, 10), "decidecade": (3, 10)}

#: The bases :func:`octave_bands` builds: 10 (IEC 61260-1), or 2 for exact
#: octaves.
_OCTAVE_BASES = (10, 2)


def _ladder(indices, fraction, base):
    """``(lower, centres, upper)`` of the bands ``indices`` of the
    1/``fraction``-octave ladder of ``base`` anchored at 1 kHz.

    Base 2: centres ``1000 * 2**(k/b)`` and edges ``centre * 2**(±1/(2b))``,
    the exact fractional octaves (Pierce, *Acoustics*, §2.2: ``f1 =
    2^(-1/2N) f0``, ``f2 = 2^(1/2N) f0``). Base 10: the octave ratio is
    ``G = 10**(3/10)`` (IEC 61260-1), so centres ``1000 * 10**(3k/(10b))``
    and edges ``centre * 10**(±3/(20b))``. Each base-10 exponent is one
    quotient of integers held exactly in a double, hence the correctly
    rounded value: for ``b = 3`` it is the same double as ``k/10`` and
    ``1/20``, the decidecade exponents.
    """
    k = np.asarray(indices, dtype=float)
    if base == 10:
        centres = _REF_FREQ * 10.0 ** (3.0 * k / (10.0 * fraction))
        half = 3.0 / (20.0 * fraction)
    else:
        centres = _REF_FREQ * 2.0 ** (k / fraction)
        half = 1.0 / (2.0 * fraction)
    radix = float(base)
    return centres * radix ** (-half), centres, centres * radix ** half


def _ladder_position(f, fraction, base):
    """Where ``f`` sits on a ladder, in band indices from the 1 kHz band."""
    if base == 10:
        return 10.0 * fraction / 3.0 * math.log10(f / _REF_FREQ)
    return fraction * math.log2(f / _REF_FREQ)


def _overlapping_bands(freq_min, freq_max, fraction, base, who):
    """Every band of the 1/``fraction``-octave ladder of ``base`` that
    overlaps ``[freq_min, freq_max]``, as ``(lower, centres, upper)``."""
    if freq_min <= 0 or freq_max <= freq_min:
        raise ConfigurationError(
            f"{who}: need 0 < freq_min < freq_max; "
            f"got freq_min={freq_min!r}, freq_max={freq_max!r}.")
    # One band of margin each side of the log position; the overlap test
    # below is what selects.
    first = math.floor(_ladder_position(freq_min, fraction, base)) - 1
    last = math.ceil(_ladder_position(freq_max, fraction, base)) + 1
    lower, centres, upper = _ladder(np.arange(first, last + 1), fraction, base)
    keep = (upper >= freq_min) & (lower <= freq_max)
    return lower[keep], centres[keep], upper[keep]


def octave_bands(freq_min, freq_max, *, base=10):
    """Octave band ``(lower, centres, upper)`` edges overlapping
    ``[freq_min, freq_max]``, anchored at 1 kHz.

    ``base=10`` (the default) is the IEC 61260-1:2014 octave: centres
    ``1000 * 10**(3k/10)``, edges ``centre * 10**(±3/20)`` — every third
    decidecade, and the ladder ``band_type='octave'`` names on
    :func:`sound_exposure`, :func:`standard_bands` and :func:`band_levels`.
    ``base=2`` gives exact octaves: centres ``1000 * 2**k``, edges
    ``centre * 2**(±1/2)`` (Pierce, *Acoustics*, §2.2); the two part by
    0.24 % a step, 15.85 kHz against 16 kHz four octaves up. Returns three
    arrays of equal length covering every band that overlaps the requested
    range.

    Parameters
    ----------
    freq_min, freq_max : float
        The frequency range (Hz).
    base : {10, 2}, optional
        IEC base-10 octaves or exact octaves (see above). Default 10.
    """
    if base not in _OCTAVE_BASES:
        raise ConfigurationError(
            f"octave_bands: base must be 10 (IEC 61260-1, the default) or 2 "
            f"(exact octaves); got base={base!r}.")
    return _overlapping_bands(freq_min, freq_max, 1, base, "octave_bands")


def decidecade_bands(freq_min, freq_max):
    """Decidecade band ``(lower, center, upper)`` edges spanning ``[freq_min, freq_max]``.

    Centre frequencies are ``1000 * 10^(n/10)`` (IEC 61260-1 base-10); band edges
    are ``center * 10^(±1/20)``. Returns three arrays of equal length covering
    every band that overlaps the requested range.

    Parameters
    ----------
    freq_min, freq_max : float
        The frequency range (Hz).
    """
    return _overlapping_bands(freq_min, freq_max, *_LADDERS["decidecade"],
                              "decidecade_bands")


def _standard_bands(freq_min, freq_max, *, band_type, sample_rate, n_bands,
                    who, range_names=("freq_min", "freq_max")):
    """:func:`standard_bands` for ``who``, whose messages name the range
    by ``range_names``."""
    low, high = range_names
    if freq_min <= 0 or freq_max <= freq_min:
        raise ConfigurationError(
            f"{who}: require {low} > 0 and {high} > {low}; got "
            f"{low}={freq_min}, {high}={freq_max}.")
    nyquist = None if sample_rate is None else sample_rate / 2.0
    if band_type == "decidecade":
        # A band is kept when its CENTRE is in the requested range and its
        # whole width is sampled. Selecting on the centre rather than on the
        # edges is what makes the range robust to how it was written: the
        # nominal edges are irrational (10**4.35 = 22387.21...), so the
        # rounded 22387 anyone types drops the 20 kHz band by a hundredth of
        # a hertz, while its centre is 387 Hz clear of the limit. A band cut
        # by Nyquist is excluded outright: it would report the part that was
        # sampled as if it were the band.
        lower, centres, upper = decidecade_bands(freq_min, freq_max)
        keep = (centres >= freq_min) & (centres <= freq_max)
        if nyquist is not None:
            keep &= upper <= nyquist
        return lower[keep], centres[keep], upper[keep]
    if band_type == "octave":
        # From the band holding ``freq_min`` to the band holding ``freq_max``,
        # clamped to the highest band whose whole width is sampled. Rounding
        # the log position onto the centre grid, rather than flooring onto
        # the half-band grid, is what stops the ladder shifting by half a
        # step with the parity of the floor; clamping the upper EDGE rather
        # than the centre keeps a band that is fully below Nyquist.
        fraction, base = _LADDERS[band_type]
        first = round(_ladder_position(freq_min, fraction, base))
        last = math.ceil(_ladder_position(freq_max, fraction, base) - 0.5)
        if nyquist is not None:
            last = min(last, math.floor(
                _ladder_position(nyquist, fraction, base) - 0.5))
        return _ladder(np.arange(first, last + 1), fraction, base)
    if band_type == "linear":
        if n_bands is None or n_bands <= 0:
            raise ConfigurationError(
                f"{who}: n_bands must be positive for linear bands; got "
                f"{n_bands}.")
        edges = freq_min + (freq_max - freq_min) * np.arange(n_bands + 1) / n_bands
        # The last edge is freq_max itself, so a bin sitting exactly on it (the
        # Nyquist bin of a full-span request) is inside the top band.
        edges[-1] = freq_max
        return edges[:-1], (edges[:-1] + edges[1:]) / 2, edges[1:]
    raise ConfigurationError(
        f"{who}: unknown band_type={band_type!r}; valid: "
        "'decidecade' (the default) and 'octave' (IEC 61260-1 base-10), "
        "'linear'")


def standard_bands(freq_min, freq_max, *, band_type="decidecade",
                   sample_rate=None, n_bands=None):
    """The ``(lower, centres, upper)`` bands :func:`sound_exposure` reports a
    range ``[freq_min, freq_max]`` on.

    ``band_type`` is one of
    :data:`~uacpy.acoustic_signal.spectral.BAND_TYPES`:

    - ``'decidecade'`` (IEC 61260-1 / ISO 18405 base-10, the default): the
      bands whose centre lies in the range. Selecting on the centre makes
      the range robust to rounded edges — ``freq_max=22387`` keeps the 20 kHz
      band, whose exact upper edge is 22387.21 Hz.
    - ``'octave'`` (IEC 61260-1 base-10): from the band holding ``freq_min`` to
      the band holding ``freq_max``. An octave is three decidecades wide, so
      selecting on the centre would drop up to half an octave of the range
      at each end (measured: 15 % of a 100-400 Hz request).
    - ``'linear'``: ``n_bands`` equal-width bands spanning the range.

    Given ``sample_rate``, a band of a standard ladder whose upper edge lies
    above Nyquist is dropped: a band cut by Nyquist would report the part
    that was sampled as if it were the band. The arrays can then be empty.
    Every ladder is anchored at 1 kHz, so the bands do not move with the
    range asked for.

    Parameters
    ----------
    freq_min, freq_max : float
        The frequency range (Hz).
    band_type : str, optional
        The ladder (see above). Default ``'decidecade'``.
    sample_rate : float, optional
        Drop standard bands whose upper edge is above its Nyquist.
    n_bands : int, optional
        Band count of ``band_type='linear'``.
    """
    return _standard_bands(freq_min, freq_max, band_type=band_type,
                           sample_rate=sample_rate, n_bands=n_bands,
                           who="standard_bands",
                           range_names=("freq_min", "freq_max"))


def _fractional_bin_weights(nperseg, sample_rate, bands):
    """Per band, the rfft bins it overlaps and the fraction of each it takes.

    Bin ``k > 0`` stands for the interval ``[f_k - df/2, f_k + df/2]``, the
    Nyquist bin for its lower half ``[fs/2 - df/2, fs/2]``, and a band takes
    each bin in proportion to the part of that interval inside
    ``[low, high]``. The DC bin is the record's mean, energy AT 0 Hz rather
    than spread over a half-bin, so it belongs to no band (band edges are
    > 0). The fractions of one bin over adjacent bands sum to 1, so
    energy is still counted once: the total over the bands is the energy
    between the first band's lower edge and the last band's upper edge, the
    edge bins counted by the part inside. Whole-bin sums instead measure
    each band over the width of the bins whose centres fall in it — on a
    flat spectrum at 1 Hz bins the 10 Hz decidecade band (2.31 Hz wide)
    counted 3 bins, +1.14 dB, and the 16 Hz band -0.86 dB.
    """
    df = sample_rate / nperseg
    f = np.fft.rfftfreq(nperseg, d=1 / sample_rate)
    lo = np.maximum(f - df / 2.0, 0.0)
    hi = np.minimum(f + df / 2.0, sample_rate / 2.0)
    width = hi - lo
    out = []
    for low, _, high in bands:
        overlap = np.minimum(hi, high) - np.maximum(lo, low)
        overlap[0] = 0.0                  # DC: a point at 0 Hz, in no band
        idx = np.flatnonzero(overlap > 0.0)
        out.append((idx, overlap[idx] / width[idx]))
    return out


def _band_estimate(data, sample_rate, *, scaling="exposure",
                   freq_min=8.9125, freq_max=22387, band_type="decidecade",
                   n_bands=30, nperseg=None, window=_BAND_WINDOW,
                   batch_size=None, who="sound_exposure", **welch_kwargs):
    """Standard-band estimate: one value per IEC 61260-1 / ISO 18405 band.

    Returns ``(centres, values, bands)``, reference-free and linear.
    ``scaling='exposure'`` gives the sound exposure in each band (Pa²·s),
    ``'spectrum'`` that divided by the record duration (Pa², the band power)
    and ``'density'`` the band power per hertz of the band's own width
    (Pa²/Hz). ``bands`` are the ``(low, centre, high)`` edges those values sit
    on.

    Uses a rectangular (boxcar) window with ``noverlap=0`` and no detrending so
    the summed PSD equals the band exposure exactly (Parseval). A tapering
    window would corrupt that identity, which is why the public door offers
    none: :func:`sound_exposure` has no ``window``, ``noverlap``, ``detrend``
    or ``average`` argument, and the bin estimators refuse
    ``scaling='exposure'`` outright.

    The default ``freq_min``/``freq_max`` are the base-10 band edges of the nominal
    10 Hz — 20 kHz reporting range: ``10^0.95 = 8.9125`` Hz is the lower edge of
    the 10 Hz band and ``10^4.35 = 22387`` Hz the upper edge of the 10^4.3 =
    19953 Hz ("20 kHz") band. For ``band_type='octave'``
    they select from the band holding ``freq_min`` to the band holding ``freq_max``
    under Nyquist (:func:`standard_bands`); ``'linear'`` uses them as given.
    ``nperseg`` is the FFT segment length, the same quantity Welch calls
    ``nperseg``, and defaults to ``sample_rate`` — 1 Hz wide bins.

    **Band selectivity:** the *total* exposure (sum over all bands) is
    Parseval-exact **over the covered band** — it equals ``sum(data**2) /
    sample_rate`` restricted to ``[bands[0][0], bands[-1][2]]``, each FFT bin
    split between the bands its interval overlaps in proportion to the
    overlap (:func:`_fractional_bin_weights`) so that it is counted once.
    Energy outside that span is dropped, which always includes DC (band
    edges must be > 0). So the total
    is *not* the whole signal's exposure whenever the bands do not span the
    full spectrum: for white noise sampled at 2 kHz the default decidecade
    request covers 8.9-891 Hz (the highest whole band under Nyquist) and
    returns about 88 % of it, and even a DC-to-Nyquist ``'linear'`` request
    falls short by the DC bin alone. Compare a total against the band span it
    covers, not against ``sum(data**2) / sample_rate``.

    Each band is a fractionally weighted sum of rectangular FFT bins, not an
    IEC 61260 fractional-octave filter. A tone that does not fall on a bin
    centre leaks into each adjacent band at a floor of roughly -33 dB relative
    to its own band (IEC 61260 class-1 filters provide 60-75 dB of stopband
    rejection), and a tone whose bin straddles a band edge is split between
    the two bands in proportion. Band levels of broadband signals are
    accurate; strong tonals bleed into neighbouring bands at about that
    level.

    **Batch alignment:** ``batch_size`` is how many samples are read at a
    time, so a multi-hour record never materialises as one segment matrix.
    Unset, it is a whole number of segments near
    :data:`_TARGET_BATCH_SAMPLES`, so the default never pays the penalty
    below. It is a memory knob, not an estimator one — but only while each
    batch holds whole segments. Each batch is zero-padded up to a whole number of
    ``nperseg`` segments; the padding adds no energy — the total stays
    exact — but a tone truncated by the padded partial segment leaks part of
    its power into neighbouring bands. With ``batch_size`` not a multiple of
    ``nperseg`` *every* batch ends in such a partial segment and
    neighbouring-band levels wobble by a few tenths of a dB (a warning is
    issued when the signal is actually batched); with a multiple, only the
    signal's final partial batch does.
    """
    data = require_finite_signal(data, who)
    if np.iscomplexobj(data):
        raise ConfigurationError(
            f"{who}: data must be real (got complex input); band "
            "exposure is "
            "defined for a real pressure time series. Demodulate to a real "
            "signal first.")
    sample_rate = require_positive_finite_scalar(
        sample_rate, who, "sample_rate", " Hz")
    bands = np.column_stack(_standard_bands(
        freq_min, freq_max, band_type=band_type, sample_rate=sample_rate,
        n_bands=n_bands, who=who))
    if len(data) <= 0:
        raise ConfigurationError(
            f"{who}: no samples to integrate. Provide a non-empty "
            f"signal; got {len(data)} sample(s) at "
            f"sample_rate={sample_rate:g} Hz.")
    if nperseg is None:
        # One-hertz bins: the resolution a band ladder is read at, which
        # puts two or more lines in even the narrowest low band.
        nperseg = sample_rate
    nperseg = int(nperseg)
    if batch_size is None:
        # Whole segments, about a quarter-million samples of them: a batch
        # that ends mid-segment pays the zero-padding penalty on EVERY batch,
        # and a fixed sample count would never divide the default nperseg
        # (the sample rate). Written in segments, the default never does.
        batch_size = max(1, round(_TARGET_BATCH_SAMPLES / nperseg)) * nperseg
    batch_size = min(int(batch_size), max(1, len(data)))

    if len(bands) == 0:
        raise ConfigurationError(
            f"{who}: no {band_type} band fits below Nyquist "
            f"({sample_rate / 2:g} Hz) with the requested freq_min/freq_max — the "
            f"snapped lower edge sits above the Nyquist-clamped upper edge; "
            f"got freq_min={freq_min!r}, freq_max={freq_max!r}.",
            remediation="Raise sample_rate, or pass a lower freq_min explicitly.")
    if len(data) > batch_size and batch_size % nperseg:
        warnings.warn(
            f"{who}: batch_size ({batch_size}) is not a multiple of "
            f"nperseg ({nperseg}), so every batch ends in a zero-padded "
            "partial segment; tonal energy truncated there leaks into "
            "neighbouring bands (band wobble of a few tenths of a dB). Pass "
            "a batch_size that is a multiple of nperseg.",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)
    band_bins = _fractional_bin_weights(nperseg, sample_rate, bands)
    # A band no bin reaches sums to exactly 0 Pa²·s, which reads as a measured
    # silence rather than as "not measured" — 'linear' bands are used as given,
    # so a freq_max above Nyquist produces whole empty bands (the octave ladders
    # clamp to Nyquist instead).
    empty = [k for k, (idx, _) in enumerate(band_bins) if idx.size == 0]
    if empty:
        named = ', '.join(f"{bands[k][0]:.4g}-{bands[k][2]:.4g}" for k in empty[:4])
        if len(empty) > 4:
            named += f", ... ({len(empty)} in all)"
        warnings.warn(
            f"{who}: {len(empty)} of {len(bands)} bands overlap no FFT "
            f"bin and "
            f"are returned as exactly 0 Pa²·s, which is not a measurement: "
            f"{named} Hz. A band above Nyquist ({sample_rate / 2:g} Hz) has no "
            f"data at all. Lower freq_max.",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)
    out = np.zeros(len(bands))

    for i in range(0, len(data), batch_size):
        chunk = data[i: min(i + batch_size, len(data))]
        # One estimator, not two: the per-bin exposure of this batch is what
        # the Welch route returns under ``scaling='exposure'`` — boxcar, no
        # overlap, no detrending, and the batch padded to whole segments so
        # no sample is dropped — and a band's exposure is the bins it covers.
        # Summing (rather than averaging) over segments is what makes it an
        # energy, and that sum is already inside that route: it multiplies
        # the mean by the padded duration.
        #
        # The batching is memory, not method: a multi-hour record never
        # materialises as one segment matrix. Each batch's own exposure adds,
        # because energy does.
        per_bin = _estimate(
            chunk, sample_rate, method="welch", scaling="exposure",
            window=window, nperseg=nperseg, who=who, **welch_kwargs).power
        for k, (idx, weight) in enumerate(band_bins):
            out[k] += per_bin[idx] @ weight

    centres = np.array([b[1] for b in bands], dtype=float)
    edges = np.asarray(bands, dtype=float)
    if scaling == "exposure":
        values = out
    else:
        # Band power is the exposure spread back over the record it was
        # integrated from; a density then divides by the band's own width,
        # which is what makes decidecade levels comparable across bands that
        # get wider with frequency.
        values = out / (len(data) / sample_rate)
        if scaling == "density":
            values = values / (edges[:, 2] - edges[:, 0])
    return centres, values, edges


# ──────────────────────────────────────────────────────────────────────
# Band levels
#
# A PSD integrated onto a standard ladder. ``sound_exposure`` is written on
# the same edges.
# ──────────────────────────────────────────────────────────────────────

class BandLevels(PlottedResult, namedtuple("BandLevels", "centres levels")):
    """Band levels on a standard ladder: the band ``centres`` (Hz) and one
    ``levels`` value each (dB re ``ref²``, the band power), so
    ``centres, levels = ...`` unpacks it; :meth:`plot` draws it with
    :func:`~uacpy.plot.plot_band_levels` against its own ``ref``.

    What the levels are stated on rides on attributes: ``lower`` and
    ``upper``, the band edges (Hz); ``band_type``, the ladder
    (``'decidecade'`` or ``'octave'``); and ``ref``, the
    reference pressure (Pa) the levels are in dB re the square of.
    """

    _attrs = ("lower", "upper", "band_type", "ref")
    _plotter = "plot_band_levels"
    _plot_fields = ("centres", "levels")
    _plot_defaults = ("ref", "band_type")

    def __new__(cls, centres, levels, *, lower, upper, band_type, ref):
        self = super().__new__(cls, centres, levels)
        self.lower = lower
        self.upper = upper
        self.band_type = str(band_type)
        self.ref = float(ref)
        return self

    def _field_units(self):
        return {"centres": "Hz", "levels": level_unit(self.ref, "spectrum")}


#: The ladder names as prose, for messages.
_LADDER_WORDS = {"decidecade": "decidecade", "octave": "octave"}


def _band_levels(psd, frequencies, band_type, ref, who):
    """:func:`band_levels` for ``who``, which its messages name."""
    if band_type not in _LADDERS:
        raise ConfigurationError(
            f"{who}: band_type={band_type!r} has no standard band edges; "
            f"valid: {', '.join(repr(t) for t in _LADDERS)}. Integrate any "
            f"other band with uacpy.acoustics.integrate_psd.")
    word = _LADDER_WORDS[band_type]
    psd = np.asarray(psd, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    ref = require_positive_finite_scalar(ref, who, "ref", " Pa")
    if np.any(psd < 0):
        raise ConfigurationError(
            f"{who}: psd contains negative values; a power "
            "spectral density is non-negative, so this input is a dB level "
            "or a signed spectrum, whose band integral is not a band level. "
            f"Got {int(np.count_nonzero(psd < 0))} negative value(s), "
            f"minimum {psd.min():g}.")
    if frequencies.shape != psd.shape:
        raise ConfigurationError(
            f"{who}: psd shape {psd.shape} and frequencies "
            f"shape {frequencies.shape} differ.")
    if frequencies.size > 1 and np.any(np.diff(frequencies) <= 0):
        raise ConfigurationError(
            f"{who}: frequencies must be strictly increasing. "
            "A two-sided np.fft.fftfreq grid is not — take the one-sided "
            "np.fft.rfftfreq half (and the matching half of the PSD). Got "
            f"{int(np.count_nonzero(np.diff(frequencies) <= 0))} "
            f"non-increasing step(s), first at index "
            f"{int(np.argmax(np.diff(frequencies) <= 0))}.")
    require_increasing_axis(frequencies, f"{who}: frequencies")
    if frequencies.size < 2:
        # A one-point axis has freq_min == freq_max, which would reach the ladder
        # as "need 0 < freq_min < freq_max" — an error naming two arguments this
        # caller never passed.
        raise ConfigurationError(
            f"{who}: frequencies needs at least 2 samples to "
            f"span a band; got {frequencies.size}. A single point has no "
            f"width, so no {word} band covers it.")
    pos = frequencies > 0
    # Tested after the DC bin is dropped: a two-sample rfftfreq grid passes
    # the size guard above yet leaves a single positive frequency, which has
    # no width for a band and would reach the ladder as freq_min == freq_max
    # — an error naming two arguments this caller never passed.
    if int(np.count_nonzero(pos)) < 2:
        raise ConfigurationError(
            f"{who}: frequencies "
            f"[{frequencies[0]:g}, {frequencies[-1]:g}] Hz holds "
            f"{int(np.count_nonzero(pos))} positive sample(s) once the DC "
            f"bin is dropped, so the grid spans no {word} band. Use a "
            f"longer FFT so the one-sided grid holds at least two positive "
            f"frequencies.")
    freq_min = frequencies[pos].min()
    freq_max = frequencies[pos].max()
    lower, centers, upper = _overlapping_bands(freq_min, freq_max,
                                               *_LADDERS[band_type], who)
    levels = np.full(centers.size, np.nan)
    n_coarse = 0
    for i, (lo, hi) in enumerate(zip(lower, upper)):
        # Integrate over [lo, hi] itself: splice the edges into the in-band
        # grid points and interpolate the PSD onto them.
        #
        # A band the supplied grid does not fully cover is left ``nan``.
        # Clipping the nodes to the support instead returned the integral over
        # the covered part, which is not that band's level — measured 3.8 dB
        # (first band) and 3.2 dB (last) off their own trend on a flat PSD,
        # and 5.5 dB low on the realistic psd -> band_levels path.
        if (lo < freq_min * (1.0 - BAND_EDGE_RTOL)
                or hi > freq_max * (1.0 + BAND_EDGE_RTOL)):
            continue
        interior = frequencies[(frequencies > lo) & (frequencies < hi)]
        if interior.size < 2:
            n_coarse += 1
        power = integrate_psd(psd, frequencies, lo, hi)
        # A covered band with no power is -inf, as every empty band is.
        with np.errstate(divide='ignore'):
            levels[i] = 10.0 * np.log10(power / ref ** 2)
    if n_coarse:
        warnings.warn(
            f"{who}: {n_coarse} band(s) hold fewer than two "
            "interior PSD grid points and rest almost entirely on interpolated "
            "band edges; the PSD grid is too coarse to resolve them. Use a "
            "finer-resolution PSD for a fully integrated level.",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    return BandLevels(centers, levels, lower=lower, upper=upper,
                      band_type=band_type, ref=ref)


def band_levels(psd, frequencies, *, band_type="decidecade",
                ref=REFERENCE_PRESSURE_WATER):
    """Integrate a one-sided PSD into the band levels of a standard ladder.

    Parameters
    ----------
    psd : array_like
        One-sided power spectral density [pressure²/Hz, e.g. Pa²/Hz].
    frequencies : array_like
        Frequencies [Hz] matching ``psd`` (monotonic, > 0).
    band_type : {'decidecade', 'octave'}
        The ladder: :func:`decidecade_bands` (the default) or
        :func:`octave_bands`, both IEC 61260-1 base-10.
    ref : float
        Reference pressure (default ``1e-6`` Pa = 1 µPa, the water standard).

    Returns
    -------
    BandLevels
        ``centres`` [Hz] and ``levels`` [dB re ``ref²``] of every band of the
        ladder overlapping the grid, with the band edges, ``band_type`` and
        ``ref``; bands with no spectral support are ``nan``, and a covered
        band that carries no power is ``-inf``.

    Notes
    -----
    Each band is integrated over its full support ``[lo, hi]`` by
    :func:`uacpy.core.acoustics.integrate_psd`: the band edges are spliced
    into the in-band grid points and the PSD is interpolated onto them, so the
    edge intervals carry their true width. A band reaching past
    the ends of ``frequencies`` is returned as ``nan`` — a partial integral is
    not a band level.

    **The first and last band are normally ``nan``, and that is structural.**
    The band set is every band of the ladder *overlapping*
    ``[min(frequencies), max(frequencies)]``, so the band holding the first
    frequency starts below it and the band holding the last ends above it
    unless both land exactly on band edges — which no ``rfftfreq`` grid
    does. Those two ``nan`` levels are the diagnostic; they are not warned
    about, because a warning that fires on every well-formed call cannot
    distinguish "the grid is too short" from "the function was called". The
    returned arrays stay parallel to the ladder function on the same span,
    so a caller masks with ``np.isfinite(levels)``.

    A band holding fewer than two interior grid points rests almost entirely on
    its interpolated edges; a ``NumericsWarning`` names how many such bands
    the grid produced. That one *is* a warning: it qualifies levels that came
    back finite.

    The integral runs over the exact band width, as :func:`sound_exposure`'s
    overlap-weighted bin sums do, so the two carry the same band widths on
    the same ladder.
    """
    return _band_levels(psd, frequencies, band_type, ref, "band_levels")


def decidecade_band_levels(psd, frequencies, *, ref=REFERENCE_PRESSURE_WATER):
    """Integrate a one-sided PSD into decidecade band levels:
    :func:`band_levels` on the default ``band_type='decidecade'`` ladder.

    Parameters
    ----------
    psd : array_like
        One-sided power spectral density [pressure²/Hz, e.g. Pa²/Hz].
    frequencies : array_like
        Frequencies [Hz] matching ``psd`` (monotonic, > 0).
    ref : float
        Reference pressure (default ``1e-6`` Pa = 1 µPa, the water standard).

    Returns
    -------
    BandLevels
        ``centres`` [Hz] and ``levels`` [dB re ``ref²``], with the band edges;
        bands with no spectral support are ``nan`` (normally the first and
        last, see :func:`band_levels`), and a covered band that carries no
        power is ``-inf``.
    """
    return _band_levels(psd, frequencies, "decidecade", ref,
                        "decidecade_band_levels")
