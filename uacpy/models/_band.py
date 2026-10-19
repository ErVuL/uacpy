"""The frequency grid a run propagates by the base rules, resolved once with
the notice that states it (:func:`resolve_band`): the band a single
BROADBAND carrier expands to, the grid a TIME_SERIES pulse implies, and the
checks on the pulse the grid is derived from."""

from typing import NamedTuple, Optional

import numpy as np

from uacpy.acoustic_signal._synthesis import (
    refuse_record_shorter_than_waveform, source_waveform_problem,
)
from uacpy.core._validate import (
    reject_complex, require_positive, steps_are_uniform,
)
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.run_settings import RunMode, TimeSettings
from uacpy.models._budget import memory_budget
from uacpy.models._defaults import (
    DEFAULT_BROADBAND_BANDWIDTH_FACTOR, DEFAULT_BROADBAND_N_FREQUENCIES,
)


# Fewest frequencies an auto-derived TIME_SERIES grid may carry, uacpy
# guidance (no source gives a minimum). Δf is
# 1/waveform-duration, so a short pulse over a narrow band can derive 2-3 bins
# — below what a band-edge taper (``window='hann'``) can act on, and too
# few to represent an arrival. Costs one model run per extra bin.
# Odd on purpose: RAM re-parameterises a uniform grid as an (fc, Q, T) sweep
# that is symmetric about a bin, so an odd count round-trips exactly while an
# even one marches a superset (ram/_band.py resolve_broadband_grid).
_MIN_TIMESERIES_FREQS = 9

#: The frequency arrays a band engine's settings record: the bins the binary
#: marched and the bins the run asked for.
FREQUENCY_ARRAY_FIELDS = ('marched_frequencies', 'requested_frequencies')


#: Spectral support (dB below the peak) of the pulse that sets the band of an
#: auto-derived TIME_SERIES grid. uacpy guidance: no engine manual or
#: textbook names a spectral threshold for the band a pulse needs; this and
#: the band constants below are uacpy's, measured where stated.
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
#: so it never binds on them.
_TIME_SERIES_CORE_DB = -20.0

#: The band is measured on the pulse zero-padded to at least this many
#: times its own length, so the DFT lattice cannot sit on the nulls of the
#: pulse's spectrum: unpadded, an integer-cycle burst's DFT is one line (a
#: 4.0-cycle 80 Hz burst gave the band [80, 100] Hz, SPARC -1.10 dB, corr
#: 0.90) and a 40-sample Hann-tapered tone lost its main lobe's skirts
#: ([90, 110] Hz, 8.8 % of its energy, -0.80 dB).
_BAND_OVERSAMPLING = 16

#: Peak bytes the band derivation holds per point of its zero-padded FFT
#: (:func:`_fine_fft_length`): measured 16.8-17.0 B on 16 000-64 000-sample
#: noise records and 21.2 B on a short pulse in a long record, with the
#: record's own arrays; rounded up, it is what the derivation is weighed at
#: before it allocates (:func:`time_series_band`).
_BAND_DERIVATION_BYTES_PER_POINT = 24

#: Model frequencies past which the notice of a grid derived from the pulse
#: names its cost, one solve each. The derivation is sized for a pulse:
#: Δf = 1/record, so a recording passed as the pulse derives record ×
#: bandwidth bins (a 21.85 s 6-14 kHz recording: 144 060), where the
#: cheaper route is the channel's impulse response once and a convolution.
_MANY_TIMESERIES_FREQS = 10_000

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


def _fine_fft_length(wf: np.ndarray) -> int:
    """Length of the zero-padded FFT the band of the pulse ``wf`` is
    measured on: :data:`_BAND_OVERSAMPLING` times the pulse up to its last
    non-zero sample, in whole records (0 for an all-zero pulse)."""
    n = wf.size
    nonzero = np.flatnonzero(wf)
    if nonzero.size == 0:
        return 0
    return max(1, -(-_BAND_OVERSAMPLING * (int(nonzero[-1]) + 1) // n)) * n


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
    m = _fine_fft_length(wf) // n
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


def check_run_frequencies(frequencies, *, model_name: str) -> None:
    """Refuse a ``run(frequencies=...)`` grid that a
    :class:`~uacpy.core.source.Source` would refuse as its own frequencies:
    complex, not 1-D, empty, non-finite, not positive, or listing one
    frequency twice. Any order is accepted, as on the Source: the axis
    indexes the result in the order given."""
    label = f"{model_name}.run: frequencies="
    reject_complex(frequencies, label)
    try:
        f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    except (TypeError, ValueError):
        raise ConfigurationError(
            f"{label} must be numbers in Hz; got {frequencies!r}.") from None
    if f.ndim != 1 or f.size == 0:
        raise ConfigurationError(
            f"{label} must be a scalar or a non-empty 1-D array of "
            f"frequencies in Hz; got shape {f.shape}.")
    require_positive(f, label, hint="Hz")
    unique, counts = np.unique(f, return_counts=True)
    if np.any(counts > 1):
        raise ConfigurationError(
            f"{label} lists {unique[counts > 1].tolist()} Hz more than "
            f"once; each frequency is one result bin.",
            remediation="Pass each frequency once "
                        "(np.unique(frequencies) keeps them sorted).")


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


def refuse_a_record_shorter_than_the_pulse(settings, *,
                                           model_name: str) -> None:
    """Stage 3 of a TIME_SERIES run whose trace is synthesised from the
    transfer function on ``settings.frequencies``: the refusal that
    synthesis gives a record ``1/Δf`` shorter than the pulse
    (:func:`~uacpy.acoustic_signal._synthesis.refuse_record_shorter_than_waveform`,
    the same function, so one rule), raised before the engine runs. A grid
    that is not uniform is left to the synthesis, which refuses it."""
    time, freqs = settings.time, settings.frequencies
    if (settings.mode != RunMode.TIME_SERIES or time is None
            or time.source_waveform is None or freqs is None):
        return
    f = np.atleast_1d(np.asarray(freqs, dtype=float))
    if f.ndim != 1 or f.size < 2:
        return
    df = float(f[1] - f[0])
    if df <= 0 or not steps_are_uniform(np.diff(f), df):
        return
    refuse_record_shorter_than_waveform(
        df, np.asarray(time.source_waveform), float(time.sample_rate),
        f"{model_name}.run")


def time_settings(kwargs, *, pads_pulse: bool) -> TimeSettings:
    """The TIME_SERIES request of a checked call's keywords ``kwargs``. The
    pulse is stored as the synthesis uses it (real, float64, zero-padded to
    ``output_duration`` when ``pads_pulse``, the engine's
    ``spec.traits.pads_pulse_to_output_duration``) when it is a usable
    pulse, the float64 1-D array :func:`_real_pulse` admits. A pulse the
    engine will refuse is stored as ``None`` and the refusal is the
    engine's (``PropagationModel._check_time_series_request``)."""
    sample_rate = kwargs['sample_rate']
    output_duration = kwargs['output_duration']
    pulse = None
    if kwargs['source_waveform'] is not None:
        clean, problem = _real_pulse(kwargs['source_waveform'])
        if problem is None:
            pulse = np.asarray(clean, dtype=float).ravel()
            if _positive_finite(sample_rate) and pads_pulse:
                pulse = np.asarray(pad_waveform_to_duration(
                    pulse, float(sample_rate), output_duration),
                    dtype=float)
    return TimeSettings(
        source_waveform=pulse,
        sample_rate=(float(sample_rate) if _positive_finite(sample_rate)
                     else None),
        output_duration=(None if output_duration is None
                         else float(output_duration)),
        t_start=kwargs['t_start'])


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


def broadband_band(source) -> BandResolution:
    """The BROADBAND grid of ``source``: a multi-element
    ``source.frequencies`` *is* the band; a single centre frequency ``fc``
    expands to :data:`~uacpy.models._defaults.DEFAULT_BROADBAND_N_FREQUENCIES`
    bins spanning ``fc·(1 ± b/2)``, ``b`` =
    :data:`~uacpy.models._defaults.DEFAULT_BROADBAND_BANDWIDTH_FACTOR`, built
    as given (a sub-1 Hz centre included). Any other band is the run's
    ``frequencies=``."""
    src_f = np.atleast_1d(np.asarray(source.frequencies, dtype=float))
    if src_f.size > 1:
        return BandResolution(src_f)
    half_bw = 0.5 * DEFAULT_BROADBAND_BANDWIDTH_FACTOR
    return BandResolution(np.linspace(float(src_f[0]) * (1.0 - half_bw),
                                      float(src_f[0]) * (1.0 + half_bw),
                                      DEFAULT_BROADBAND_N_FREQUENCIES))


def time_series_band(source_waveform, sample_rate, *,
                     model_name: str,
                     record_chosen: bool = False,
                     solves_each_bin: bool = True) -> BandResolution:
    """The TIME_SERIES grid a pulse implies (:func:`_time_series_grid`):
    Δf = ``sample_rate / n_samples`` (= 1 / waveform duration), band edges
    from the spectral support above :data:`_TIME_SERIES_THRESHOLD_DB` below
    the peak, limited to one -20 dB band-width beyond the -20 dB band; the
    notice names the band, the spacing and the record it makes (and, when
    the limit cut the support, the energy left out and the solves the full
    support would cost). ``record_chosen`` (the caller passed
    ``output_duration=``) drops the record-length part of the notice, since
    the record is then the caller's choice; whether it holds the channel is
    measured on the synthesised trace, as on every IFFT record
    (:func:`~uacpy.models._extract.warn_if_circular_record_rings`). ``None``
    frequencies, and no
    notice, for a pulse
    of fewer than two samples or a non-positive rate. Raises
    :class:`ConfigurationError` for a pulse with no usable band, and for one
    whose band measurement would not fit in the host's free memory (weighed
    from its length before it allocates, :func:`memory_budget`). A grid of
    more than :data:`_MANY_TIMESERIES_FREQS` frequencies on an engine that
    solves each one (``solves_each_bin``; False for Bellhop, whose
    delay-and-sum traces the rays once and only labels its trace with the
    grid) is returned, and its notice names the solves and the cheaper
    route: a recording passed as the pulse, or an ``output_duration=`` the
    pulse is padded to that is far longer than the channel needs."""
    if source_waveform is None or sample_rate is None:
        return BandResolution(None)
    wf = np.asarray(source_waveform, dtype=float).ravel()
    if wf.size < 2 or sample_rate <= 0:
        return BandResolution(None)
    who = f"{model_name}.run(run_mode=TIME_SERIES)"
    n_fine = _fine_fft_length(wf)
    memory_notice = memory_budget(
        _BAND_DERIVATION_BYTES_PER_POINT * n_fine, model_name=who,
        what='TIME_SERIES band measurement',
        detail=(f"measuring the band of the {wf.size}-sample source "
                f"waveform takes a {n_fine}-point FFT, about "
                f"{_BAND_DERIVATION_BYTES_PER_POINT * n_fine / 1024 ** 3:.2g} "
                f"GiB."),
        remediation=("Compute the channel once and convolve: a TIME_SERIES "
                     "run with a short pulse spanning the band (or a "
                     "BROADBAND H(f) on a frequencies= grid), then convolve "
                     "the recording with its impulse response; or pass "
                     "frequencies= to set the grid."))
    threshold_dB = _TIME_SERIES_THRESHOLD_DB
    grid = _time_series_grid(wf, sample_rate, threshold_dB)
    if isinstance(grid, str):
        raise ConfigurationError(f"{who}: {grid}.")
    cost = ""
    if solves_each_bin and grid.frequencies.size > _MANY_TIMESERIES_FREQS:
        duration = wf.size / float(sample_rate)
        derived = (f"derives {grid.frequencies.size} model frequencies "
                   f"({grid.freq_min:.4g}-{grid.freq_max:.4g} Hz at Δf = "
                   f"{grid.df_grid:.4g} Hz), one solve each, over the "
                   f"{_MANY_TIMESERIES_FREQS} a pulse's grid is sized for")
        # The grid scales with the record's length (Δf = 1/record), so the
        # waveform before its zero-padding would derive this share of it:
        # a shorter output_duration= helps only while that share fits.
        nonzero = np.flatnonzero(wf)
        signal = int(nonzero[-1]) + 1 if nonzero.size else wf.size
        alone = grid.frequencies.size * signal / wf.size
        if record_chosen and alone <= _MANY_TIMESERIES_FREQS:
            cost = (f" The {duration:.4g} s record (the source waveform "
                    f"zero-padded to output_duration=) {derived}. A shorter "
                    f"output_duration= (the record needs to outlast the "
                    f"channel's spread of arrivals, not more), or "
                    f"frequencies=, sets a smaller grid.")
        else:
            padded = (f" ({signal / float(sample_rate):.4g} s of signal, "
                      f"which alone derives about {alone:.0f}, so no "
                      f"output_duration= brings it under)"
                      if record_chosen else "")
            cost = (f" The {duration:.4g} s source waveform{padded} "
                    f"{derived}. Computing the channel once and convolving "
                    f"costs fewer solves: a TIME_SERIES run with a short "
                    f"pulse spanning the band (or a BROADBAND H(f) on a "
                    f"frequencies= grid), then the recording convolved with "
                    f"its impulse response.")
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
        f"That Δf makes the record {record:.4g} s long, set by the "
        f"pulse, not by the channel; the run warns if the response "
        f"outlasts it. Pass `output_duration=` to buy a longer record, "
        f"or `frequencies=` to set the grid yourself and silence this."
        f"{limit}{cost}")
    if record_chosen:
        notice = (f"{model_name}.run(run_mode=TIME_SERIES):{limit}{cost}"
                  if limit or cost else None)
    if memory_notice is not None:
        notice = (memory_notice.message if notice is None
                  else f"{notice} {memory_notice.message}")
    return BandResolution(grid.frequencies, notice)


def resolve_band(mode, source, frequencies, time, *, model_name: str,
                 solves_each_bin: bool = True) -> BandResolution:
    """The frequency grid (Hz) a call asks for, by the base rules, once,
    with its notice: ``frequencies=`` when given; else, for ``BROADBAND``,
    :func:`broadband_band` of the source; for ``TIME_SERIES``, a multi-element
    ``source.frequencies`` (it names the band as explicitly as
    ``frequencies=`` does), else :func:`time_series_band` of the pulse of
    the time settings ``time`` (``None`` without a usable pulse; its notice
    names the solves of a large grid when the engine ``solves_each_bin``);
    in every other mode, ``source.frequencies``."""
    if frequencies is not None:
        return BandResolution(
            np.atleast_1d(np.asarray(frequencies, dtype=float)))
    if mode == RunMode.BROADBAND:
        return broadband_band(source)
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
                                is not None,
                                solves_each_bin=solves_each_bin)
    return BandResolution(
        np.atleast_1d(np.asarray(source.frequencies, dtype=float)))
