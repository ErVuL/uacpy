"""The frequency grid a run propagates by the base rules, resolved once with
the notice that states it (:func:`resolve_band`): the band a single
BROADBAND carrier expands to, the grid a TIME_SERIES pulse implies, and the
checks on the pulse the grid is derived from."""

from typing import NamedTuple, Optional

import numpy as np

from uacpy.acoustic_signal._synthesis import source_waveform_problem
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.run_settings import RunMode
from uacpy.models._defaults import (
    DEFAULT_BROADBAND_BANDWIDTH_FACTOR, DEFAULT_BROADBAND_N_FREQS,
)


# Fewest frequencies an auto-derived TIME_SERIES grid may carry. Δf is
# 1/waveform-duration, so a short pulse over a narrow band can derive 2-3 bins
# — below what the band-edge taper in ``_ifft_to_trace`` can act on, and too
# few to represent an arrival. Costs one model run per extra bin.
# Odd on purpose: RAM re-parameterises a uniform grid as an (fc, Q, T) sweep
# that is symmetric about a bin, so an odd count round-trips exactly while an
# even one marches a superset (ram/_band.py resolve_broadband_grid).
_MIN_TIMESERIES_FREQS = 9

#: The frequency arrays a band engine's settings record: the bins the binary
#: marched and the bins the run asked for.
FREQUENCY_ARRAY_FIELDS = ('marched_frequencies', 'requested_frequencies')


#: Spectral support (dB below the peak) of the pulse that sets the band of an
#: auto-derived TIME_SERIES grid.
_TIME_SERIES_THRESHOLD_DB = -40.0

#: The core band (dB below the peak) that bounds how far the -40 dB support
#: may reach: at most one core-band width beyond each core edge. A pulse
#: with a hard edge (a rectangular tone burst, an untapered chirp) has a
#: spectrum that falls 6-12 dB per octave, so its -40 dB support runs tens
#: of main lobes out — a 4.2-cycle rectangular 80 Hz burst at 2 kHz reaches
#: 991 Hz, where SPARC marched 208.6 s against 0.92 s over 40-160 Hz — while
#: the energy past the limit is small: 0.69 % for that burst (the band-limited
#: pulse correlates 0.998 with the pulse, -0.06 dB), 0.11 % for a 4.0-cycle
#: one, 0.007 % for an 8 kHz untapered 1-2 kHz chirp. The -40 dB support of
#: a Gaussian burst, a Ricker or a Hann-tapered tone lies inside the limit,
#: so it never binds on them (``log/fix-r2-models_scratch/b1_band_rules``).
_TIME_SERIES_CORE_DB = -20.0

#: The band is measured on the pulse zero-padded to at least this many
#: times its own length, so the DFT lattice cannot sit on the nulls of the
#: pulse's spectrum: unpadded, an integer-cycle burst's DFT is one line (a
#: 4.0-cycle 80 Hz burst gave the band [80, 100] Hz, SPARC -1.10 dB, corr
#: 0.90) and a 40-sample Hann-tapered tone lost its main lobe's skirts
#: ([90, 110] Hz, 8.8 % of its energy, -0.80 dB).
_BAND_OVERSAMPLING = 16

# Where an auto-derived band edge came from, as the grid records it.
_BAND_EDGE_SUPPORT = "the -40 dB support of the pulse's spectrum"
_BAND_EDGE_LIMIT = ("the -20 dB band widened by its own width (the -40 dB "
                    "support reaches {support:.4g} Hz)")
_BAND_EDGE_DC = "Δf, the lowest bin above DC"


def frequencies_text(frequencies, *, head: int = 3) -> str:
    """``frequencies`` as a message prints them, in Hz: every value to four
    significant digits when there are at most ``2 * head``, else the first
    and last ``head`` around an ellipsis."""
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    shown = [f"{v:.4g}" for v in f]
    if f.size > 2 * head:
        shown = shown[:head] + ['…'] + shown[-head:]
    return f"[{', '.join(shown)}] Hz"


class _TimeSeriesGrid(NamedTuple):
    """A TIME_SERIES grid derived from a pulse, with what the notice
    quotes: the band's own bin count before subdivision, the pulse's
    ``Δf = 1/duration``, the grid's spacing and its edges, where each edge
    came from, the band the -40 dB support alone would have set (on the
    same lattice) and the fraction of the pulse's energy outside the band.
    """
    frequencies: np.ndarray
    n_band: int
    df_waveform: float
    df_grid: float
    freq_min: float
    freq_max: float
    freq_min_origin: str
    freq_max_origin: str
    support_min: float
    support_max: float
    energy_outside: float

    @property
    def limited(self) -> bool:
        """Whether the core-band limit cut the -40 dB support."""
        return (self.freq_min_origin.startswith('the -20 dB')
                or self.freq_max_origin.startswith('the -20 dB'))


def _time_series_grid(source_waveform, sample_rate, threshold_dB):
    """The TIME_SERIES frequency grid a pulse implies, or the reason (a
    string) there is none.

    ``source_waveform`` is the pulse as the record holds it (zero-padded to
    the record), so Δf is ``sample_rate / n_samples`` = 1 / record. The band
    is measured on the pulse zero-padded to :data:`_BAND_OVERSAMPLING`
    times its length: the spectral support above ``threshold_dB`` below the
    peak, limited to one :data:`_TIME_SERIES_CORE_DB` band-width beyond each
    edge of that core band. Its edges are the record's bins inside that
    band (a pulse whose support the record's lattice samples well gets the
    same bins as when measured on the lattice itself). A pulse with DC
    content starts the band at Δf, not 0 Hz. A band of fewer than
    :data:`_MIN_TIMESERIES_FREQS` bins is subdivided (equivalent to
    zero-padding the pulse), because a short pulse over a narrow band can
    derive only 2-3 bins — too few for the band-edge taper to leave an
    interior, and too few to represent an arrival at all.
    """
    wf = np.asarray(source_waveform, dtype=float).ravel()
    n = wf.size
    if n < 2:
        return "source_waveform has fewer than two samples."
    fs = float(sample_rate)
    df = fs / n
    nonzero = np.flatnonzero(wf)
    if nonzero.size == 0:
        return "source_waveform is identically zero."
    # The fine lattice holds the record's: every m-th fine bin is a record
    # bin, so a record bin is fine bin m*i.
    m = max(1, -(-_BAND_OVERSAMPLING * (int(nonzero[-1]) + 1) // n))
    spectrum = np.abs(np.fft.rfft(wf, m * n))
    peak = spectrum.max()
    if peak <= 0:
        return "source_waveform is identically zero."

    def support(level_dB):
        above = np.flatnonzero(spectrum >= peak * 10.0 ** (level_dB / 20.0))
        return (int(above[0]), int(above[-1])) if above.size else None

    wide = support(threshold_dB)
    if wide is None:
        return (f"source_waveform has no spectral content above "
                f"{threshold_dB} dB.")
    core_lo, core_hi = support(_TIME_SERIES_CORE_DB) or wide
    width = core_hi - core_lo
    k_lo, k_hi = max(wide[0], core_lo - width), min(wide[1], core_hi + width)
    fine_df = fs / (m * n)
    record_freqs = np.fft.rfftfreq(n, 1.0 / fs)
    edge_origin = [_BAND_EDGE_SUPPORT, _BAND_EDGE_SUPPORT]
    if k_lo > wide[0]:
        edge_origin[0] = _BAND_EDGE_LIMIT.format(support=wide[0] * fine_df)
    if k_hi < wide[1]:
        edge_origin[1] = _BAND_EDGE_LIMIT.format(support=wide[1] * fine_df)

    def record_band(lo, hi):
        """The record's bins inside fine bins ``[lo, hi]``."""
        i_lo, i_hi = -(-lo // m), hi // m
        if i_hi < i_lo:                  # narrower than one record bin
            i_lo, i_hi = lo // m, -(-hi // m)
        f_lo = max(float(record_freqs[i_lo]), df)
        f_hi = float(record_freqs[i_hi])
        if f_hi <= f_lo:
            f_hi = f_lo + df
        return f_lo, f_hi

    freq_min, freq_max = record_band(k_lo, k_hi)
    # A support edge is the outermost record bin still inside the support,
    # where the pulse stands above the threshold, so a band ending there cuts
    # it above it; the next bin out lies past the support, below it.
    if edge_origin[0] == _BAND_EDGE_SUPPORT and freq_min - df >= df:
        freq_min -= df
    if edge_origin[1] == _BAND_EDGE_SUPPORT and freq_max + df <= 0.5 * fs:
        freq_max += df
    if freq_min == df and k_lo < m:
        edge_origin[0] = _BAND_EDGE_DC
    support_min, support_max = record_band(*wide)
    energy = spectrum ** 2
    fine = np.arange(energy.size) * fine_df
    inside = (fine >= freq_min - 0.5 * fine_df) & (fine <= freq_max + 0.5 * fine_df)
    energy_outside = float(1.0 - energy[inside].sum() / energy.sum())
    n_band = int(round((freq_max - freq_min) / df)) + 1
    refined = max(n_band, _MIN_TIMESERIES_FREQS)
    return _TimeSeriesGrid(
        frequencies=np.linspace(freq_min, freq_max, refined), n_band=n_band,
        df_waveform=df, df_grid=(freq_max - freq_min) / (refined - 1),
        freq_min=freq_min, freq_max=freq_max,
        freq_min_origin=edge_origin[0], freq_max_origin=edge_origin[1],
        support_min=support_min, support_max=support_max,
        energy_outside=max(energy_outside, 0.0))


def _real_pulse(source_waveform):
    """``(pulse, None)`` for a usable TIME_SERIES pulse, as a float64 1-D
    array, else ``(None, reason)``: the one waveform rule every entry point
    that takes a source waveform applies
    (:func:`~uacpy.acoustic_signal._synthesis.source_waveform_problem`)."""
    problem = source_waveform_problem(source_waveform)
    if problem is not None:
        return None, problem
    return np.asarray(source_waveform, dtype=float), None


def _positive_finite(value) -> bool:
    """Whether ``value`` is a positive finite number."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(x) and x > 0.0)


class BandResolution(NamedTuple):
    """The frequency grid (Hz) a call asks for by the base rules, or
    ``None`` where the engine's own settings choose it, with the notice that
    states how it was derived (``None`` when there is nothing to say). The
    notice is text: stage 3 gives it as a warning on the modes an engine
    announces (``spec.traits.announced_band_modes``)."""
    frequencies: Optional[np.ndarray]
    notice: Optional[str] = None


def pad_waveform_to_duration(source_waveform, sample_rate, output_duration):
    """Zero-pad ``source_waveform`` so its duration is at least
    ``output_duration`` seconds. Returns the (possibly padded) array
    unchanged when ``output_duration`` is ``None`` or already met.

    Used by every IFFT-based TIME_SERIES wrapper so the user can
    request a longer output than the source pulse without having to
    pre-pad: ``Field.synthesize_time_series`` sets output duration
    = waveform duration, and the auto-derived broadband grid uses
    ``Δf = 1 / waveform_duration``.
    """
    if (
        output_duration is None
        or source_waveform is None
        or sample_rate is None
        or sample_rate <= 0
    ):
        return source_waveform
    wf = np.asarray(source_waveform, dtype=float).ravel()
    n_needed = int(np.ceil(float(output_duration) * float(sample_rate)))
    if wf.size >= n_needed:
        return source_waveform
    pad = np.zeros(n_needed - wf.size, dtype=wf.dtype)
    return np.concatenate([wf, pad])


def broadband_band(source, n_freqs=None, bandwidth_factor=None, *,
                   model_name: str) -> BandResolution:
    """The BROADBAND grid of ``source``: a multi-element
    ``source.frequencies`` *is* the band; a single centre frequency ``fc``
    expands to ``n_freqs`` bins spanning ``fc·(1 ± bandwidth_factor/2)``
    (``None`` takes the package defaults
    :data:`~uacpy.models._defaults.DEFAULT_BROADBAND_N_FREQS` and
    :data:`~uacpy.models._defaults.DEFAULT_BROADBAND_BANDWIDTH_FACTOR`), its
    lower edge floored at 1 Hz, which the notice then says.

    Raises :class:`ConfigurationError` for a band that cannot be built:
    fewer than two bins, a non-positive ``bandwidth_factor``, or a sub-1 Hz
    centre the floor empties. An axis the package DERIVES must be ascending
    and distinct (a caller's Source may list its own frequencies in any
    order, never twice)."""
    src_f = np.atleast_1d(np.asarray(source.frequencies, dtype=float))
    if src_f.size > 1:
        return BandResolution(src_f)
    fc = float(src_f[0])
    if n_freqs is None:
        n_freqs = DEFAULT_BROADBAND_N_FREQS
    n_freqs = int(n_freqs)
    # ``np.linspace`` degenerates below two points: 1 returns the lower
    # band edge alone — a grid silently mislabelled as the band — and 0
    # an empty grid, so neither can span fc·(1 ± bandwidth_factor/2).
    if n_freqs < 2:
        raise ConfigurationError(
            f"{model_name} broadband: n_freqs = {n_freqs} cannot "
            f"span a frequency band — the expanded grid needs at least "
            f"its two edges.",
            remediation=(
                "Use n_freqs >= 2, or pass frequencies=[fc] to run a "
                "single bin."
            ),
        )
    if bandwidth_factor is None:
        bandwidth_factor = DEFAULT_BROADBAND_BANDWIDTH_FACTOR
    half_bw = 0.5 * float(bandwidth_factor)
    # The lower edge goes to zero or negative once bandwidth_factor >= 2;
    # a 0 Hz bin is not a runnable model frequency, so it is floored at
    # 1 Hz (the notice says so).
    lo_requested = fc * (1.0 - half_bw)
    hi = fc * (1.0 + half_bw)
    lo = max(lo_requested, 1.0)
    notice = None
    if lo_requested < 1.0:
        notice = (
            f"{model_name} broadband: bandwidth_factor="
            f"{bandwidth_factor:g} puts the lower band edge at "
            f"{lo_requested:.4g} Hz; floored at 1 Hz, so the band is no "
            f"longer centred on fc = {fc:g} Hz nor the requested width.")
    if not hi > lo:
        # A non-positive bandwidth_factor inverts or collapses the band at
        # any fc, and a sub-1 Hz fc can put the floored lower edge at or
        # above the upper edge.
        if float(bandwidth_factor) <= 0:
            raise ConfigurationError(
                f"{model_name} broadband: bandwidth_factor = "
                f"{bandwidth_factor:g} gives the band [{lo:g}, {hi:g}] Hz "
                f"around fc = {fc:g} Hz. The band is "
                f"fc*(1 +/- bandwidth_factor/2), so bandwidth_factor "
                f"must be positive."
            )
        raise ConfigurationError(
            f"{model_name} broadband: the band "
            f"[{lo:g}, {hi:g}] Hz is empty after the 1 Hz floor "
            f"(fc = {fc:g} Hz, bandwidth_factor = {bandwidth_factor:g}). "
            f"Sub-1 Hz centre frequencies need an explicit frequencies= "
            f"grid."
        )
    return BandResolution(np.linspace(lo, hi, n_freqs), notice)


def time_series_band(source_waveform, sample_rate, *,
                     model_name: str,
                     record_chosen: bool = False) -> BandResolution:
    """The TIME_SERIES grid a pulse implies (:func:`_time_series_grid`):
    Δf = ``sample_rate / n_samples`` (= 1 / waveform duration), band edges
    from the spectral support above :data:`_TIME_SERIES_THRESHOLD_DB` below
    the peak, limited to one -20 dB band-width beyond the -20 dB band; the
    notice names the band, the spacing and the record it makes (and, when
    the limit cut the support, the energy left out and the solves the full
    support would cost). ``record_chosen`` (the caller passed
    ``output_duration=``) drops the record-length part of the notice, since
    the record is then the caller's choice. ``None`` frequencies, and no
    notice, for a pulse
    of fewer than two samples or a non-positive rate. Raises
    :class:`ConfigurationError` for a pulse with no usable band."""
    if source_waveform is None or sample_rate is None:
        return BandResolution(None)
    wf = np.asarray(source_waveform, dtype=float).ravel()
    if wf.size < 2 or sample_rate <= 0:
        return BandResolution(None)
    threshold_dB = _TIME_SERIES_THRESHOLD_DB
    grid = _time_series_grid(wf, sample_rate, threshold_dB)
    if isinstance(grid, str):
        raise ConfigurationError(
            f"{model_name}.run(run_mode=TIME_SERIES): {grid}.")
    note = ""
    if grid.frequencies.size != grid.n_band:
        note = (f" (waveform Δf = {grid.df_waveform:.4g} Hz subdivided "
                f"so the {grid.n_band}-bin band resolves an arrival)")
    limit = ""
    if grid.limited:
        n_full = max(int(round((grid.support_max - grid.support_min)
                               / grid.df_waveform)) + 1,
                     _MIN_TIMESERIES_FREQS)
        limit = (
            f" The pulse's -40 dB support spans "
            f"{grid.support_min:.4g}-{grid.support_max:.4g} Hz: its "
            f"spectrum falls slowly (a hard edge in the pulse), so the "
            f"band stops one -20 dB band-width beyond the -20 dB band, "
            f"leaving {100.0 * grid.energy_outside:.2g} % of the pulse's "
            f"energy outside it. `frequencies=` over the full support "
            f"includes it, at {n_full} solves instead of "
            f"{grid.frequencies.size}.")
    # Name the record as well as the spacing. They are one number written
    # two ways — a synthesised trace is 1/Δf long — but the spacing alone
    # leaves the consequence to be derived: this Δf comes from the SOURCE
    # pulse, and a channel whose multipath outlasts it folds the tail onto
    # the early trace, where it reads as extra early arrivals rather than as
    # a mistake.
    record = (1.0 / grid.df_grid if grid.df_grid > 0 else float('inf'))
    notice = (
        f"{model_name}.run(run_mode=TIME_SERIES): no "
        f"`frequencies=` passed; auto-derived "
        f"{grid.frequencies.size} freqs from "
        f"the source waveform ({grid.freq_min:.2f}-{grid.freq_max:.2f} Hz, "
        f"Δf={grid.df_grid:.4g} Hz, threshold {threshold_dB:.0f} dB)"
        f"{note}. "
        f"That Δf makes the record {record:.4g} s long, and it is set "
        f"by the pulse, not by the channel: any arrival later than "
        f"{record:.4g} s after the first folds back onto the early "
        f"trace. Pass `output_duration=` to buy a longer record, or "
        f"`frequencies=` to set the grid yourself and silence this."
        f"{limit}")
    if record_chosen:
        notice = (f"{model_name}.run(run_mode=TIME_SERIES):{limit}"
                  if limit else None)
    return BandResolution(grid.frequencies, notice)


def resolve_band(mode, source, frequencies, time, *, model_name: str,
                 n_freqs=None, bandwidth_factor=None) -> BandResolution:
    """The frequency grid (Hz) a call asks for, by the base rules, once,
    with its notice: ``frequencies=`` when given; else, for ``BROADBAND``,
    :func:`broadband_band` of the source (with the model's ``n_freqs`` /
    ``bandwidth_factor``); for ``TIME_SERIES``, a multi-element
    ``source.frequencies`` (it names the band as explicitly as
    ``frequencies=`` does), else :func:`time_series_band` of the pulse of
    the time settings ``time`` (``None`` without a usable pulse); in every
    other mode, ``source.frequencies``."""
    if frequencies is not None:
        return BandResolution(
            np.atleast_1d(np.asarray(frequencies, dtype=float)))
    if mode == RunMode.BROADBAND:
        return broadband_band(source, n_freqs, bandwidth_factor,
                              model_name=model_name)
    if mode == RunMode.TIME_SERIES:
        src_f = np.atleast_1d(np.asarray(source.frequencies, dtype=float))
        if src_f.size > 1:
            return BandResolution(src_f)
        if (time is None or time.source_waveform is None
                or time.sample_rate is None):
            return BandResolution(None)
        return time_series_band(time.source_waveform, time.sample_rate,
                                model_name=model_name,
                                record_chosen=time.output_duration
                                is not None)
    return BandResolution(
        np.atleast_1d(np.asarray(source.frequencies, dtype=float)))
