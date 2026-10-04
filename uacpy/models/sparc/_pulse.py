"""The pulse one SPARC deck marches: the ``pulse_type`` alphabet and which
pulse a run resolves to, the pulse band ``[freq_min, freq_max]`` and where each edge
came from, the notices about that band and the source frequencies, and the
rows of the ``STSFIL`` series a pulse read from file is written with."""

from typing import NamedTuple, Optional, Tuple

import numpy as np

from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, NumericsWarning,
)
from uacpy.models._band import (
    _time_series_grid, _TIME_SERIES_THRESHOLD_DB, _TimeSeriesGrid,
    pad_waveform_to_duration, time_series_band,
)
from uacpy.core.run_settings import Notice

# The pulse a TIME_SERIES run marches when ``pulse_type`` is unpinned: the
# run's own waveform, staged as STSFIL, or the canned wavelet without one.
_WAVEFORM_PULSE = 'FN+B'
_CANNED_PULSE = 'PN+B'


# SPARC pulse_type alphabets (per Scooter/sparc.f90:126-148 GetPar SELECT CASE).
# Pos 1: pulse shape — the 10 letters accepted by sparc.f90's parser.
# ``T`` and ``C`` exist in tslib/cans.f90 (the pulse evaluator) but
# sparc.f90's GetPar rejects them with "Unknown source type" before
# cans.f90 is reached.
_PULSE_TYPE_POS1 = set('PRASHNGFBM')
# Pos 2: post-processing applied to the pulse samples
# (tslib/sourceMod.f90:69-70).
#   'H' = pre-envelope (|analytic signal|), 'Q' = Hilbert transform.
#   Any other character (including ' ' or 'N') means "no transform".
_PULSE_TYPE_POS2 = {' ', 'N', 'H', 'Q'}
# Pos 3: sign flag. '-' inverts the pulse (tslib/sourceMod.f90:178); any
# other character keeps it.
_PULSE_TYPE_POS3 = {' ', '+', '-'}
# Pos 4: per-wavenumber band-pass of the source pulse, set in sparc.f90's
# MARCH (Scooter/sparc.f90:391-401, mirrored in Matlab/Sparc/march.m:93-102).
#   'L' cuts below k·cLow/2π, 'H' cuts above k·cHigh/2π, 'B' does both.
#   'N' skips the filter entirely (tslib/sourceMod.f90:68); any other
#   character (including ' ') leaves the full band [0, 10·fMax] in place.
_PULSE_TYPE_POS4 = {' ', 'N', 'L', 'H', 'B'}

#: ``tslib/sourceMod.f90:7`` — the most samples STSFIL may hold in
#: total (rows × source depths, ``MaxNST``); ``:116`` stops the read past
#: it. The row count has its own bound, ``MaxNt``, of the same value: the
#: read loop ``DO it = 1, MaxNt`` (:105) ends normally only on end-of-file,
#: and falling out of it is the same fatal (:120), so the rows must stay
#: strictly below ``MaxNt``.
_MAX_STS_POINTS = 10_000_000
_MAX_STS_ROWS = 10_000_000


def _validate_pulse_type(pulse_type: str) -> str:
    """
    Validate a 4-character SPARC pulse_type string.

    Parameters
    ----------
    pulse_type : str
        The raw string the user passed. Short strings are right-padded
        with spaces to length 4 (matching sparcM.m's handling).

    Returns
    -------
    pulse_type : str
        The normalized 4-character string.

    Raises
    ------
    ConfigurationError
        If any character falls outside the alphabets read from
        ``Scooter/sparc.f90:126-148`` (shape) and ``:391-401`` (filter), and
        from ``tslib/sourceMod.f90:68-70,178`` (post-process, sign, filter).
    """
    if not isinstance(pulse_type, str):
        raise ConfigurationError(
            f"pulse_type must be a string, got {type(pulse_type).__name__}."
        )
    if len(pulse_type) > 4:
        raise ConfigurationError(
            f"pulse_type must be at most 4 characters, got {pulse_type!r}."
        )
    pulse_type = pulse_type.ljust(4)

    def _bad(pos, char, allowed):
        return ConfigurationError(
            f"Invalid pulse_type character {char!r} at position {pos} "
            f"(must be one of {sorted(allowed)!r}). "
            f"See Acoustics-Toolbox/Scooter/sparc.f90."
        )

    if pulse_type[0] not in _PULSE_TYPE_POS1:
        raise _bad(1, pulse_type[0], _PULSE_TYPE_POS1)
    if pulse_type[1] not in _PULSE_TYPE_POS2:
        raise _bad(2, pulse_type[1], _PULSE_TYPE_POS2)
    if pulse_type[2] not in _PULSE_TYPE_POS3:
        raise _bad(3, pulse_type[2], _PULSE_TYPE_POS3)
    if pulse_type[3] not in _PULSE_TYPE_POS4:
        raise _bad(4, pulse_type[3], _PULSE_TYPE_POS4)
    return pulse_type


class _PulseBand(NamedTuple):
    """The pulse band ``[freq_min, freq_max]`` (Hz) a run marches, with where
    each edge came from, and the base rule's grid when the band was derived
    from the run's waveform (``None`` otherwise)."""
    freq_min: float
    freq_max: float
    freq_min_origin: str
    freq_max_origin: str
    grid: Optional[_TimeSeriesGrid] = None


# Where a derived band edge came from, as SparcSettings records it. A
# waveform's edge names the base rule's own origin of it
# (:func:`~uacpy.models._band._time_series_grid`, the band every synthesising
# engine derives for the same pulse and record).
_BAND_FROM_WAVEFORM = (
    "{edge}, the run's source_waveform padded to the {time_max:g} s record")
_BAND_FROM_OCTAVE = (
    "one octave around source.frequencies[0], the canned pulse's frequency")
_BAND_FROM_FREQUENCIES = "the span of run(frequencies=...)"


def resolve_pulse_type(source_waveform, *, pulse_type) -> Tuple[str, str]:
    """``(pulse_type, origin)`` of the pulse the deck is written with:
    the pinned ``pulse_type``; else the run's ``source_waveform``
    marched from STSFIL (``'FN+B'``); else the canned ``'PN+B'``
    wavelet."""
    if pulse_type is not None:
        return pulse_type, 'SPARC(pulse_type=…)'
    if source_waveform is not None:
        return (_WAVEFORM_PULSE,
                "the run's source_waveform, marched from STSFIL")
    return (_CANNED_PULSE,
            "the canned wavelet (the run was handed no source_waveform)")


def derived_band(source, time, time_max: float, *,
                 pulse_type) -> Optional[_PulseBand]:
    """The band SPARC derives when neither ``SPARC(freq_min=, freq_max=)`` nor
    ``run(frequencies=)`` sets it: for a pulse read from STSFIL, the
    band the base rule (:func:`_time_series_grid`) derives from the
    waveform zero-padded to SPARC's own ``time_max`` record; for a canned
    pulse, one octave around ``source.frequencies[0]``. ``None`` for a
    waveform the base rule derives no band from."""
    waveform = None if time is None else time.source_waveform
    pulse, _origin = resolve_pulse_type(waveform, pulse_type=pulse_type)
    if pulse[0] in 'FB':
        grid = _time_series_grid(
            pad_waveform_to_duration(
                waveform, time.sample_rate, time_max),
            time.sample_rate, _TIME_SERIES_THRESHOLD_DB)
        if isinstance(grid, str):
            return None
        return _PulseBand(
            float(grid.freq_min), float(grid.freq_max),
            _BAND_FROM_WAVEFORM.format(edge=grid.freq_min_origin,
                                       time_max=time_max),
            _BAND_FROM_WAVEFORM.format(edge=grid.freq_max_origin,
                                       time_max=time_max),
            grid)
    freq = float(np.atleast_1d(
        np.asarray(source.frequencies, dtype=float))[0])
    return _PulseBand(max(freq / 2.0, 0.1), freq * 2.0,
                      _BAND_FROM_OCTAVE, _BAND_FROM_OCTAVE)


def resolve_pulse_band(source, frequencies, time, time_max: float, *,
                       pinned_f_min, pinned_f_max, pulse_type,
                       model_name) -> _PulseBand:
    """The pulse band the deck carries, edge by edge, with each edge's
    origin: the pinned ``freq_min`` / ``freq_max``; else the span of
    ``run(frequencies=)``; else the band :func:`derived_band` derives
    with the ``time_max`` record.

    Refuses a waveform the base rule derives no band from (the
    synthesising engines' refusal) and a resolved band with
    ``freq_min >= freq_max``.
    """
    derived = None
    if pinned_f_min is None or pinned_f_max is None:
        if frequencies is not None:
            grid = np.atleast_1d(np.asarray(frequencies, dtype=float))
            derived = _PulseBand(
                float(grid.min()), float(grid.max()),
                _BAND_FROM_FREQUENCIES, _BAND_FROM_FREQUENCIES)
        else:
            derived = derived_band(source, time, time_max,
                                   pulse_type=pulse_type)
            if derived is None:
                # The base rule's own refusal, with its reason.
                time_series_band(
                    pad_waveform_to_duration(
                        time.source_waveform, time.sample_rate, time_max),
                    time.sample_rate, model_name=model_name)
    if pinned_f_min is not None:
        freq_min, freq_min_origin = float(pinned_f_min), 'SPARC(freq_min=…)'
    else:
        freq_min, freq_min_origin = derived.freq_min, derived.freq_min_origin
    if pinned_f_max is not None:
        freq_max, freq_max_origin = float(pinned_f_max), 'SPARC(freq_max=…)'
    else:
        freq_max, freq_max_origin = derived.freq_max, derived.freq_max_origin
    if not freq_min < freq_max:
        raise ConfigurationError(
            f"SPARC pulse band requires freq_min < freq_max; got "
            f"freq_min={freq_min:g} Hz ({freq_min_origin}), freq_max={freq_max:g} Hz "
            f"({freq_max_origin}).",
            remediation=(
                "Pin both edges with SPARC(freq_min=..., freq_max=...), or pass "
                "a frequencies= grid that spans a band."),
        )
    return _PulseBand(freq_min, freq_max, freq_min_origin, freq_max_origin,
                      None if derived is None else derived.grid)


def multi_frequency_notice(source, octave_band: bool):
    """``(note, warning)`` when ``source`` lists more than one frequency,
    else ``None``.

    ``TIME_SERIES`` is not in ``spec.traits.single_frequency_modes`` (the mode is
    broadband by nature), but SPARC's deck carries one frequency — the
    one the attenuation is converted at and a canned pulse is centred on
    — so a multi-frequency Source silently loses every entry but the
    first; say which one is being read. ``octave_band``: the band is the
    auto octave around that frequency.
    """
    freqs = np.atleast_1d(np.asarray(source.frequencies, dtype=float))
    if freqs.size <= 1:
        return None
    head = (f"SPARC drives one pulse band per run: reading "
            f"source.frequencies[0] = {float(freqs[0]):.6g} Hz and "
            f"ignoring the other {freqs.size - 1} "
            f"({list(freqs[1:])}). ")
    if octave_band:
        tail = ("The auto band is one octave around it; pass "
                "SPARC(freq_min=..., freq_max=...) to cover the whole source "
                "band in the single run.")
    else:
        tail = ("It is the frequency the deck converts the attenuation "
                "at and centres a canned pulse on; the pulse band is "
                "set separately (SPARC(freq_min=..., freq_max=...), "
                "frequencies= or the source waveform).")
    return Notice(f"source.frequencies[1:] ignored "
            f"({freqs.size - 1} entries)", head + tail, FallbackWarning)


def band_limit_notice(band: _PulseBand):
    """``(note, warning)`` when the band derived from the waveform was
    cut short of the pulse's -40 dB support by the base rule's cost limit
    (:data:`~uacpy.models._band._TIME_SERIES_CORE_DB`) on an edge the
    deck uses, else ``None``. States the band, the energy it leaves out
    and what the full support would cost: the march grows about as
    ``freq_max**3`` — Nk with ``freq_max`` (``sparc.f90:112-116``), the
    Courant step count of each wavenumber with ``k`` (``:265``,
    ``:271``) and the mesh with ``freq_max`` (20 points per wavelength) —
    measured 208.6 s at 991 Hz against 0.92 s at 160 Hz on a 4.2-cycle
    rectangular burst, 225x for a (991/160)**3 = 237.6x prediction."""
    grid = band.grid
    limited = [origin for origin in (band.freq_min_origin,
                                     band.freq_max_origin)
               if origin.startswith('the -20 dB')]
    if grid is None or not limited:
        return None
    factor = (grid.support_max / band.freq_max) ** 3
    return Notice(
        f"pulse band limited to {band.freq_min:g}-{band.freq_max:g} Hz (the "
        f"-40 dB support reaches {grid.support_max:g} Hz)",
        f"SPARC: the pulse's -40 dB support spans "
        f"{grid.support_min:.4g}-{grid.support_max:.4g} Hz — its "
        f"spectrum falls slowly (a hard edge in the pulse) — so the "
        f"marched band stops at {band.freq_min:.4g}-{band.freq_max:.4g} Hz, one "
        f"-20 dB band-width beyond the -20 dB band, leaving "
        f"{100.0 * grid.energy_outside:.2g} % of the pulse's energy "
        f"outside it. Marching the full support costs about "
        f"(freq_max ratio)^3 = {factor:.3g} times this run (Nk, the time "
        f"steps of each wavenumber and the mesh all grow with freq_max). "
        f"SPARC(freq_min=..., freq_max=...) or run(frequencies=...) sets the "
        f"band.", NumericsWarning)


def source_series_rows(pulse: str, waveform, sample_rate, n_depths: int,
                       freq_min: float, freq_max: float) -> Tuple[int, int]:
    """``(samples, rows)`` of the STSFIL series ``waveform`` is written
    with when ``pulse`` is read from file, ``(0, 0)`` for a canned
    pulse; refused when the binary cannot read or band-pass it.

    Unless ``pulse_type[3] == 'N'`` the binary band-passes the series
    once per wavenumber (``tslib/sourceMod.f90:68``,
    ``tslib/bandpassc.f90``) between the cuts ``sparc.f90:391-401`` sets
    for that wavenumber — ``k·cLow/2π`` under ``'L'``/``'B'``,
    ``k·cHigh/2π`` under ``'H'``/``'B'``, else 0 and ``10·fMax`` — which
    (a) stops on any length that is not a power of two
    (``bandpassc.f90:24-25``), so the series is zero-padded up to one,
    and (b) truncates each cut to an integer FFT bin of width
    ``1/(Nt·Δt)`` (``bandpassc.f90:14-16``). The wavenumber loop sweeps
    those cuts across the deck's pulse band ``[freq_min, freq_max]``, so a
    padded series shorter than ``1/(freq_max - freq_min)`` gives every
    wavenumber's cut the same one or two bins and the per-wavenumber
    filtering collapses; that length is refused with the sample count
    that fixes it. The rows are bounded by
    :data:`_MAX_STS_POINTS` / :data:`_MAX_STS_ROWS` over the
    ``n_depths`` source depths each row carries.
    """
    if pulse[0] not in 'FB':
        return 0, 0
    n = int(np.asarray(waveform).size)
    sample_rate = float(sample_rate)
    filtered = pulse[3] != 'N'
    n_write = (1 << max(1, int(np.ceil(np.log2(n))))) if filtered else n
    if (n_write * n_depths > _MAX_STS_POINTS
            or n_write >= _MAX_STS_ROWS):
        raise ConfigurationError(
            f"SPARC: STSFIL would hold {n_write} rows x {n_depths} "
            f"source depth(s) = {n_write * n_depths} points, over the "
            f"{_MAX_STS_POINTS} the binary reads "
            f"(tslib/sourceMod.f90:7,105-120).",
            remediation=(
                f"Shorten or decimate source_waveform to at most "
                f"{min(_MAX_STS_POINTS // n_depths, _MAX_STS_ROWS - 1)} samples "
                f"(after zero-padding to a power of two when the pulse "
                f"is band-passed)."
            ),
        )
    if filtered:
        duration = n_write / sample_rate
        needed = 1.0 / (freq_max - freq_min)
        if duration < needed * (1.0 - 1e-9):
            n_needed = int(np.ceil(needed * sample_rate))
            raise ConfigurationError(
                f"SPARC: source_waveform spans {duration:.6g} s "
                f"({n_write} samples at {sample_rate:.6g} Hz, after "
                f"zero-padding to a power of two for the band-pass), "
                f"shorter than 1/(freq_max - freq_min) = {needed:.6g} s. The "
                f"binary band-passes the series once per wavenumber "
                f"between cuts it sweeps across the "
                f"{freq_min:.6g}-{freq_max:.6g} Hz pulse band "
                f"(sparc.f90:391-401) and truncates each cut to an FFT "
                f"bin of width 1/duration (tslib/bandpassc.f90:14-16), "
                f"so this record gives every wavenumber the same bins.",
                remediation=(
                    f"np.pad(source_waveform, (0, {n_needed - n})) "
                    f"(zero-pad to at least {n_needed} samples), or "
                    f"SPARC(pulse_type='{pulse[:3]}N') to "
                    f"skip the band-pass."
                ),
            )
    return n, n_write
