"""OAST: the OASES wavenumber-integration transmission-loss program."""

import warnings
from dataclasses import dataclass
from types import MappingProxyType
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.models._spec import ModelSpec
from uacpy.models._window import (
    check_pinned_window, scholte_wave_notice, oases_integration_edge,
    refuse_inverted_window,
)
from uacpy.core.run_settings import EngineSettings, OutputSpec, RunMode
from uacpy.core.results import Field
from uacpy.core.exceptions import (
    ConfigurationError, NumericsWarning, UnsupportedFeatureError,
)
from uacpy.io.oases_writer import (
    write_oast_input, oases_wavenumber_bounds,
    refuse_non_power_of_two_wavenumbers, oases_frequency_sweep,
    refuse_unusable_integration_window, oases_source_layer_speed,
)
from uacpy.io._parsers import oast_depth_average_field, parse_oast_tl
from uacpy.models.base import DEFAULT_RUN_TIMEOUT, _host_options
from uacpy.models.oases._common import (
    _OASES_TRAITS, _SCTOUT_BARE_FORT46, _geometry_options,
    _oases_binary, _reject_plane_geometry_letter_on_a_point_source,
    _check_n_wavenumbers_knob, _offset_ignored_notice, _pinned_count_note,
    _reject_unreadable_oast_options, resolve_integration_window,
)
from uacpy.models.oases._base import OASES
from uacpy.core.engine_defaults import (
    OASES_PRESSURE_LETTER, OAST_INTEGRATION_OFFSET, OAST_RANGE_MIN, OAST_VREC,
    default_c_low,
)
from uacpy.models._extract import attach_output_paths, result_kwargs
from uacpy.models._launch import require_output


#: File root of every deck and output one OAST run writes. OASS's mean-field
#: producer reads its ``.045`` under this stem.
_OAST_BASE_NAME = 'oast_run'


#: Native range step, in wavelengths of the slowest water speed, above
#: which the dB interpolation onto receiver.ranges is warned about —
#: uacpy's guidance, read off the measurement below (OAST's step is 2*pi
#: over the wavenumber window, so it follows c_low/c_high, not the
#: receivers). The default grid sits at 0.5-0.6 wavelengths. Measured on a 100 m Pekeris
#: guide at 100 and 400 Hz (Scooter on a 0.5 m grid as the truth,
#: sampled at OAST's native positions and interpolated in dB): at the
#: native step the error is median 0.01-0.02 dB, p90 <= 0.09 dB, worst
#: sample 2-14 dB inside a null; at one
#: wavelength p90 0.13-0.35 dB; at two 0.5-1.3 dB; at four 1.8-3.8 dB,
#: with the deepest nulls reading up to 20-34 dB shallow.
_COARSE_RANGE_STEP_WAVELENGTHS = 1.0


@dataclass(frozen=True, eq=False)
class OASTSettings(EngineSettings):
    """The settings one :class:`OAST` run resolved, before launching:
    ``OAST().run_settings(env, source, receiver).engine``, and
    ``result.run_settings.engine`` on the field it produced.

    Attributes
    ----------
    options : str
        The option line the deck carries (Block II), ``'P'`` included for a
        line Source.
    frequency_sweep : tuple of (float, float, int)
        Block III's ``FREQ1 FREQ2 NFREQ``.
    c_low, c_high : float
        Block VII's phase-speed window ``CMIN CMAX`` (m/s).
    c_low_origin, c_high_origin : str
        Where ``c_low`` and ``c_high`` came from: the constructor, or
        :func:`~uacpy.io.oases_writer.oases_wavenumber_bounds` — the
        sentence :func:`~uacpy.core.engine_defaults.default_c_low` states for
        ``c_low``, OASES' "no upper limit" 1e8 for ``c_high``.
    n_wavenumbers : int or None
        Block VII's wavenumber count ``NW``; ``None`` is OASES' automatic
        sampling.
    integration_window : tuple of (int, int) or None
        Block VII's ``IC1 IC2``: the given window, the whole ``(1, NW)``
        under a pinned ``n_wavenumbers``, ``None`` under automatic sampling
        (where OASES sets them).
    integration_offset : float
        The contour offset (dB/wavelength) on the frequency line.
    range_min, range_max : float
        Block VIII's ``XLEFT XRIGHT`` (m), which set OAST's native FFT range
        grid.
    vrec : float
        The frequency line's receiver velocity (m/s), read only under
        ``'d'``.
    dip_angle_deg : float or None
        The dip-slip source's dip (degrees), read only under ``'4'``.
    """

    options: str
    frequency_sweep: Tuple[float, float, int]
    c_low: float
    c_low_origin: str
    c_high: float
    c_high_origin: str
    n_wavenumbers: Optional[int]
    integration_window: Optional[Tuple[int, int]]
    integration_offset: float
    range_min: float
    range_max: float
    vrec: float
    dip_angle_deg: Optional[float]

    def __post_init__(self):
        # Lists (the to_dict form) are stored as the tuples a frozen record
        # holds.
        freq_min, freq_max, n = self.frequency_sweep
        object.__setattr__(self, 'frequency_sweep',
                           (float(freq_min), float(freq_max), int(n)))
        if self.integration_window is not None:
            object.__setattr__(self, 'integration_window',
                               tuple(int(v) for v in self.integration_window))
        super().__post_init__()


def _oast_options(options, complex_contour, compute_depth_average) -> str:
    """The OASES option line for this run.

    A raw ``options`` string is written verbatim; otherwise the letters
    are derived from the typed flags on top of ``'K T'`` (bulk pressure,
    :data:`~uacpy.core.engine_defaults.OASES_PRESSURE_LETTER`, + TL table).
    With the default flags the line is ``K T J``: the manual's worked OAST
    deck, ``N I T J Z`` (oast.tex:476), with the bulk pressure in place of
    the normal stress and without its plot-only letters. The constructor
    rejects the two being combined, and ``compute_contour=True`` (``'C'``,
    see :func:`_check_oast_knobs`), so no typed flag adds ``'C'``.
    """
    if options is not None:
        return options
    opt = [OASES_PRESSURE_LETTER, 'T']
    if complex_contour is None or complex_contour:
        opt.append('J')
    if compute_depth_average:
        opt.append('A')
    return ' '.join(opt)


def _check_oast_knobs(options, complex_contour, compute_contour,
                      compute_depth_average, vrec) -> None:
    """Refuse an OAST knob no run could use: a raw ``options``
    string beside the typed option flags it would discard,
    ``compute_contour=True``, whose contour files uacpy does not read, a
    raw line without ``'T'``, the TL table uacpy reads, a raw line whose
    TL is not a pressure (:func:`~uacpy.models.oases._common.
    _reject_unreadable_oast_options`), and a
    ``vrec`` the option line cannot carry. Run at construction and again
    by every run (:meth:`OAST._validate_engine`), since the attributes
    can be reassigned in between."""
    if compute_contour:
        raise ConfigurationError(
            "OAST(compute_contour=True): option 'C' makes OAST write "
            "range-depth contour files on units 28/29 (unoast31.f:332-334) "
            "holding, decimated in range, the TL the returned Field already "
            "holds (unoast31.f:776-797); uacpy reads no contour file.",
            remediation="Drop compute_contour: the returned Field is the TL "
                        "on the receiver grid, and "
                        "result.components['native'] the TL on OAST's own "
                        "range grid when the two differ.")
    if options is not None:
        pinned = [n for n, value in (
            ('compute_contour', compute_contour),
            ('compute_depth_average', compute_depth_average),
            ('complex_contour', complex_contour))
            if value is not None]
        if pinned:
            raise ConfigurationError(
                f"OAST: options={options!r} replaces the whole "
                f"option line, so {', '.join(pinned)} would be discarded "
                f"silently. Pass either the raw string or the typed "
                f"flags, not both — note the derived string includes "
                f"'J' (complex integration contour) by default, which a "
                f"raw string must repeat to keep."
            )
    # VREC, the fifth frequency-line token OAST reads only under the
    # lowercase 'd' (dynamics) option. The derived option line never
    # carries 'd', so a value given without a raw options string that
    # does would be written into a 4-token line the binary stops
    # reading after COFF (unoast31.f:125-127) — rejected here the way
    # dip_angle_deg is rejected without '4'.
    option_line = _oast_options(options, complex_contour,
                                compute_depth_average)
    # OAST writes its TL curves only under 'T' (GETOPT sets PLTL,
    # unoast31.f:1036-1038; PLTLOS runs under it, :629); without them the
    # run leaves nothing uacpy reads.
    if 'T' not in set(option_line):
        raise ConfigurationError(
            f"OAST: options={option_line!r} carries no 'T': OAST writes the "
            f"TL table (.plp/.plt) uacpy reads only under 'T' (PLTL, "
            f"unoast31.f:1036-1038, :629).",
            remediation=f"Add 'T' to the option line (e.g. "
                        f"options={option_line + ' T'!r}), or pass "
                        f"options=None for the default 'K T J'.")
    _reject_unreadable_oast_options(option_line)
    if vrec and 'd' not in set(option_line) - set(' \t\n'):
        raise ConfigurationError(
            f"OAST(vrec={vrec:g}) cannot reach the binary: INFREQ "
            f"reads the fifth frequency-line token only under the "
            f"lowercase 'd' option (unoast31.f:125-127), and the option "
            f"line for this run ({option_line!r}) does not "
            f"carry it.",
            remediation=("Pass a raw options string with 'd' (e.g. "
                         "options='K J T d') alongside vrec, or drop "
                         "vrec."),
        )


def _oast_multi_frequency_refusal(source) -> Exception:
    """OAST's refusal of a frequency sweep, naming the model that does
    compute one (the base decides when a sweep is refused).

    The deck and the reader both handle ``NFREQ > 1``: OAST writes one TL
    curve per plotted receiver per frequency and ``read_oast_tl`` returns
    one TL Field per frequency. What cannot follow is ONE
    :class:`~uacpy.core.results.Field`. OAST rebuilds its range grid
    *inside* the frequency loop, so ``DX`` scales as ``1/f`` and the
    sweep has one range axis per frequency — each Field the reader returns
    carries its own for exactly that reason — while
    a Field carries a single ``'range'`` coordinate.

    Putting them on a common axis would mean interpolating dB, which is
    all the ``.plt`` carries: no complex pressure exists on disk to
    interpolate instead. That is the smearing this model already warns
    about and tells callers to avoid, and here it would be unavoidable
    rather than a choice, since at most one frequency's native grid can
    be the shared one.

    :class:`OASP` is the multi-frequency wavenumber-integration product:
    it writes complex pressure to a ``.trf`` on one caller-anchored
    range grid and returns a ``(depth, range, frequency)`` Field. So the
    sweep is refused here rather than approximated.

    The base refusal's remedy is ``RunMode.BROADBAND``, which OAST does
    not implement — following it would land on ``UnsupportedFeatureError``
    naming no alternative at all.
    """
    freqs = np.atleast_1d(np.asarray(source.frequencies, dtype=float))
    return UnsupportedFeatureError(
        'OAST',
        f"a {freqs.size}-frequency source "
        f"({freqs.min():g}-{freqs.max():g} Hz) — OAST rebuilds its range "
        f"grid inside the frequency loop (DX proportional to 1/f), so a "
        f"sweep has one range axis per frequency and no single Field can "
        f"carry it; and the .plt holds only real TL, so there is no "
        f"complex pressure to resample onto a common axis without "
        f"smearing interference nulls",
        alternatives=['OASP (complex pressure on one range grid, '
                      'returns a (depth, range, frequency) Field)'],
    )


def _oast_settings(option_line, *, c_low_pinned, c_high_pinned,
                   n_wavenumbers, integration_window, integration_offset,
                   range_min,
                   range_max, vrec, dip_angle_deg, env, source,
                   receiver) -> OASTSettings:
    """Every value the OAST deck is written from, resolved once on the
    projected ``env``: the option line (with ``'P'`` for a line
    Source), Block III's frequency line, Block VII's phase-speed window
    and wavenumber count, and Block VIII's range window. A window whose
    lower edge (:func:`~uacpy.models._window.oases_integration_edge`)
    leaves out a Scholte wave the geometry can see is noticed
    (:func:`~uacpy.models._window.scholte_wave_notice`)."""
    options = _geometry_options(option_line, source)
    c_low, c_high = oases_wavenumber_bounds(env)
    if c_low_pinned is not None:
        c_low = c_low_pinned
    if c_high_pinned is not None:
        c_high = c_high_pinned
    refuse_inverted_window('OAST', c_low, c_high, c_low=c_low_pinned,
                           c_high=c_high_pinned)
    c_low_origin = ('OAST(c_low=…)' if c_low_pinned is not None
                    else default_c_low(env)[1])
    c_high_origin = ('OAST(c_high=…)' if c_high_pinned is not None
                     else "1e8, OASES' no upper limit (oast.tex:487, :568)")
    frequency_sweep = oases_frequency_sweep(source.frequencies, 'OAST')
    window, window_notice = resolve_integration_window(
        'OAST', integration_window, n_wavenumbers, 'oast.tex:548-549')
    range_min = OAST_RANGE_MIN if range_min is None else range_min
    range_max = (float(receiver.ranges.max()) if range_max is None
                 else range_max)
    edge, edge_text = oases_integration_edge(
        float(c_low), float(c_high), automatic=n_wavenumbers is None,
        frequency=frequency_sweep[0], range_min=range_min,
        range_max=range_max,
        source_speed=oases_source_layer_speed(
            env, float(np.atleast_1d(source.depths)[0])),
        full_hankel='f' in options)
    notices = tuple(n for n in [scholte_wave_notice(
        env, source, receiver, edge, frequency_sweep[0], model_name='OAST',
        edge_text=edge_text,
        remediation="Leave c_low unset (its derived value sits under "
                    "every Scholte wave) or pass c_low below the wave "
                    "speed to include it.")] if n is not None)
    return OASTSettings(
        options=options,
        frequency_sweep=frequency_sweep,
        c_low=float(c_low),
        c_low_origin=c_low_origin,
        c_high=float(c_high),
        c_high_origin=c_high_origin,
        n_wavenumbers=n_wavenumbers,
        integration_window=window,
        integration_offset=integration_offset,
        range_min=range_min,
        range_max=range_max,
        vrec=vrec,
        dip_angle_deg=dip_angle_deg,
        # OAST reads the offset token under 'J', 'O' (complex frequency) or
        # 'd' (dynamics) — unoast31.f:126-133 — and applies it only under
        # 'J' (:499).
        notices=notices + tuple(n for n in [_offset_ignored_notice(
            'OAST', integration_offset, n_wavenumbers, options,
            offset_letters='JOd'), window_notice] if n is not None),
    )


def _write_oast_deck(deck: Path, inputs) -> None:
    """The OAST deck at ``deck``, written by
    :func:`~uacpy.io.oases_writer.write_oast_input` from the settings
    ``inputs`` carries."""
    engine = inputs.settings.engine
    writer_kwargs: dict = {
        'integration_offset': engine.integration_offset,
        'n_wavenumbers': engine.n_wavenumbers,
        'integration_window': engine.integration_window,
        'vrec': engine.vrec,
        'c_low': engine.c_low,
        'c_high': engine.c_high,
        'range_min': engine.range_min,
        'range_max': engine.range_max,
    }
    if engine.dip_angle_deg is not None:
        writer_kwargs['dip_angle_deg'] = engine.dip_angle_deg
    write_oast_input(
        filepath=deck,
        env=inputs.env,
        source=inputs.source,
        receiver=inputs.receiver,
        options=engine.options,
        **writer_kwargs,
    )


def _warn_if_native_range_step_is_coarse(native_ranges, source,
                                         env) -> None:
    """Warn when the dB interpolation onto ``receiver.ranges`` reads
    from native samples more than a wavelength apart.

    OAST's native FFT range step is ``2*pi`` over the wavenumber window,
    so it follows the phase-speed window, not the receivers: about half
    a wavelength by default, coarse only when ``c_low``/``c_high`` pin
    a narrow window (see :data:`_COARSE_RANGE_STEP_WAVELENGTHS`).
    """
    steps = np.diff(np.asarray(native_ranges, dtype=float))
    if steps.size == 0:
        return
    step = float(np.max(steps))
    frequency = float(np.atleast_1d(source.frequencies)[0])
    c_water = float(np.min(env.ssp.sound_speed))
    wavelength = c_water / frequency
    if step <= _COARSE_RANGE_STEP_WAVELENGTHS * wavelength:
        return
    warnings.warn(
        f"OAST: receiver.ranges are off OAST's native FFT range grid, "
        f"whose step ({step:.4g} m) is {step / wavelength:.2g} "
        f"wavelengths ({wavelength:.4g} m at {c_water:.6g} m/s and "
        f"{frequency:g} Hz), so TL is linearly interpolated IN dB "
        f"between samples more than a wavelength apart, where interference "
        f"structure between them is not seen (one wavelength is uacpy's "
        f"guidance; docs/models/oases.md shows measured errors). The step "
        f"is 2*pi over the phase-speed window: widen c_low/c_high to "
        f"refine it, read OAST's own samples from "
        f"result.components['native'], or use OASP (.trf complex pressure "
        f"on receiver.ranges).",
        NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP,
    )


def _oast_on_receiver_grid(native: Field, native_ranges, receiver, source,
                           env) -> Field:
    """``native``, OAST's TL on its own FFT range grid, on the receiver
    grid: itself when the ranges match, else resampled by linear
    interpolation in dB, keeping ``native`` — the TL OAST computed — as
    ``result.components['native']``; a coarse native step and ranges
    outside the native span are warned about."""
    receiver_ranges = np.atleast_1d(np.asarray(receiver.ranges, dtype=float))
    receiver_depths = np.atleast_1d(np.asarray(receiver.depths, dtype=float))
    ranges_match = (
        len(native_ranges) == len(receiver_ranges)
        and np.allclose(native_ranges, receiver_ranges)
    )
    if ranges_match:
        result = native
    else:
        _warn_if_native_range_step_is_coarse(native_ranges, source, env)
        result = native.resample_to(
            ranges=receiver_ranges, depths=receiver_depths)
        result._components = {**result.components, 'native': native}
        result.metadata['interpolated'] = True
        # Interpolation cannot extrapolate: any requested range off
        # the native grid's hull comes back NaN, and the hull does
        # not start at r = 0 — the FFT grid's first sample sits one
        # range step out (measured 1.875 m on a 100-m Pekeris case).
        native_min = float(np.min(native_ranges))
        native_max = float(np.max(native_ranges))
        outside = ((receiver_ranges < native_min)
                   | (receiver_ranges > native_max))
        if outside.any():
            offending = np.array2string(
                receiver_ranges[outside], precision=4,
                max_line_width=200)
            warnings.warn(
                f"OAST: {int(outside.sum())} receiver range(s) "
                f"{offending} m lie outside the native FFT range "
                f"grid ({native_min:g}-{native_max:g} m); dB "
                f"interpolation cannot extrapolate, so those columns "
                f"are NaN (no data). Move the receivers inside the "
                f"native span — its first sample is at "
                f"{native_min:g} m, not r = 0 — or use OASP, whose "
                f".trf grid starts at receiver.ranges.min().",
                NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
    return result


class OAST(OASES):
    """
    OAST - OASES Transmission Loss Model

    Computes transmission loss using wavenumber integration. Best for
    range-independent environments with depth-dependent sound speed profiles.

    Parameters
    ----------
    executable : Path, optional
        Path to OAST binary. Auto-detected if ``None``.
    compute_contour : bool, optional
        ``True`` is refused (``ConfigurationError``): option ``'C'`` writes
        range-depth contour files on units 28/29 (``unoast31.f:332-334``)
        holding, decimated in range, the TL the returned Field already holds
        (``unoast31.f:776-797``), and uacpy reads none of them. ``None`` or
        ``False`` adds no ``'C'``.
    compute_depth_average : bool, optional
        Add the ``'A'`` option: OAST also computes the depth-averaged TL,
        ``-10·log10`` of ``|p|²`` averaged with equal weights over every
        receiver depth (``oasfun22.f:209-242``; "the averaging is performed
        over the specified number of receivers", oast.tex:144-146), returned
        as ``result.components['depth_average']``, a real dB Field on
        ``{'range'}`` over OAST's native range grid. Effective default
        ``False``.
    complex_contour : bool, optional
        ``'J'`` option (complex integration contour). Effective default
        ``True``.
    options : str, optional
        Raw OASES options string (e.g. ``'K J T A'``), written verbatim;
        ``None`` derives it from ``compute_contour`` /
        ``compute_depth_average`` / ``complex_contour``. Combining a raw
        string with any of those three flags raises
        ``ConfigurationError`` — the string replaces the whole option
        line, so a flag passed alongside it would be discarded. A raw
        string without ``'T'`` is refused: OAST writes the TL table uacpy
        reads only under ``'T'`` (``unoast31.f:1036-1038``, ``:629``).
    integration_offset : float
        Wavenumber-integration contour offset (dB/wavelength). Default 0, which
        under the 'J' option of the default option line is not "no offset":
        OASES takes any value below 1e-10 as a request for its own default,
        60*c/(f*R_max) dB/wavelength with R_max the FFT range
        (unoast31.f:499-501), the value oast.tex:293-303 recommends ("This
        value is the default which is applied if COFF is specified to
        0.0"). A value above 1e-10 is used as given.
    n_wavenumbers : int, optional
        Number of wavenumber samples; ``None`` lets OASES choose: NW = -1,
        the automatic sampling the manual offers "to obtain correct answers
        in the first attempt without the usual convergence testing"
        (oast.tex:543-550). The manual requires a power of two
        (``NW = 2^M``, oast.tex:73), and another count is refused naming the
        two neighbouring powers: OAST cuts the integral at the deck's count
        before rounding its grid up (unoast31.f:441-458). A pinned count is
        written as given, with an info-level note (``verbose='info'``) that
        the manual leaves it to the convergence test.
    integration_window : (int, int), optional
        ``(IC1, IC2)``, the first and last wavenumber samples integrated:
        the samples outside are zeroed and the edge smoothed
        (oast.tex:528-538). Two whole numbers with ``1 <= IC1 <= IC2 <=
        n_wavenumbers`` (oast.tex:74-75); it applies with a pinned
        ``n_wavenumbers``, and under automatic sampling, where "the
        parameters IC1 and IC2 have no effect" (oast.tex:548-549), the run
        says so. ``None`` integrates the whole window.
    c_low, c_high : float, optional
        ``CMIN``/``CMAX`` (m/s), Block VII's phase-speed window
        (``unoast31.f:213``). ``None`` derives the pair
        (:func:`oases_wavenumber_bounds`): ``c_low`` is 0.90 × the slowest
        wave of the problem — every medium's speed, shear speeds included,
        and every Scholte wave a liquid carries against a solid,
        solved from Brekhovskikh & Lysanov §3.4.2, eq. 3.4.29 — the margin of
        ``scooter.f90:258``
        (:func:`~uacpy.core.engine_defaults.default_c_low`), so an elastic
        seabed's shear and interface branches enter the integral; ``c_high``
        is 1e8. A pinned window whose lower edge — the ``CMIN`` OAST's
        automatic sampling (AUTSAM) reaches on the deck's ``c_low``, range
        axis and frequency
        (:func:`~uacpy.models._window.oases_integration_edge`), or
        ``c_low`` itself with a pinned ``n_wavenumbers`` — lies above a
        Scholte wave is
        noticed when the source or a receiver lies within one of its
        wavelengths of that interface
        (:func:`~uacpy.models._window.scholte_wave_notice`).
    range_min, range_max : float, optional
        First and last range (m) of OAST's native TL grid, ``XLEFT`` /
        ``XRIGHT`` (unoast31.f:482-490); ``None`` → 0, the manual's worked
        deck (oast.tex:489), and ``receiver.range_max``, the range OAST's
        automatic sampling sizes its wavenumber step against
        (unoast31.f:427-430).
    vrec : float
        Receiver velocity (m/s) for OAST's source/receiver **dynamics**
        option, the fifth token of the frequency line. The binary reads it
        only when the option string carries lowercase ``'d'``
        (``unoast31.f:125-127``), which the derived option line never
        emits — so a non-zero ``vrec`` without a raw ``options`` string
        containing ``'d'`` raises ``ConfigurationError`` (the same
        contract as ``dip_angle_deg``, whose value is read only under
        ``'4'``). Default 0.

        Despite the binary's own "Doppler compensation" wording
        (``unoast31.f:1097-1098``), ``oast.tex:215-222`` is explicit that source
        and receiver move *at the same speed and direction*, so **there is
        no Doppler shift** — only a Green's function different from the
        static one. OASP's ``'d'`` is the real radial-Doppler option and
        carries three tokens (``IT VS VR``, ``oasp.tex:227-234``).
    dip_angle_deg : float, optional
        Fault dip angle (degrees) for the dip-slip moment source that
        option ``'4'`` selects (``unoast31.f:1117-1122``). INSRC reads it
        from the source record — as the second token without ``'L'``
        (``oaseun31.f:1114``), as the seventh with it
        (``oaseun31.f:1089``). ``None`` writes 0 when ``'4'`` is present
        and nothing when it is not; a value set without ``'4'`` is
        refused.
    use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
        Standard plumbing (see :class:`PropagationModel`).

    Notes
    -----
    Range-independent wavenumber integration; consumes layered seabed
    columns natively. **Collapse defaults (overrides of
    :data:`DEFAULT_COLLAPSE`).** Per-model: ``'ssp': 'mean'`` (the layer
    stack is kept).

    ``COHERENT_TL`` here returns a **real dB Field** — ``kind='pressure'``,
    ``unit='dB'``, real dtype (transmission loss is the ``unit`` axis of a
    pressure field, not a registered ``kind``) — where :class:`OASP` returns
    complex Pa pressure for the same run mode: OAST's ``.plt`` carries only
    real TL, so there is no complex pressure to hand back and no phase
    reference to tag. Use :class:`OASP` when the consumer needs complex
    pressure — coherent array processing, phase, or off-grid receiver
    ranges through sharp interference nulls.

    The derived option line asks for OASES' output parameter ``'K'``, the
    bulk pressure: the acoustic pressure in a fluid (``oasaun31.f:1448-1449``,
    ``:1471-1472``) and, for a receiver inside an elastic layer, the bulk
    pressure of the compressional potential alone (``oasaun31.f:1509-1513``,
    oasp.tex:178-183). A raw ``options`` string may ask for the normal stress
    ``'N'`` instead: σzz, ``-p`` in a fluid (oasp.tex:185), so its TL in the
    water is the pressure's, but inside an elastic layer the kernel builds
    it from both the compressional and the shear potentials
    (``oasaun31.f:1489-1491``) and it is not a pressure there. A raw line
    asking for a velocity or another stress (``'V'``, ``'H'``, ``'R'``,
    ``'S'``), or for ``'N'`` and ``'K'`` together, is refused: its TL would
    be labelled ``kind='pressure'``.

    ``Source(source_type='line')`` runs OAST's plane geometry (option
    ``'P'``, written for it); a raw ``options`` string carrying ``'P'`` with
    a point Source is refused, since the binary would compute a line source.

    ``OAST().run_settings(env, source, receiver).engine`` is the
    :class:`OASTSettings` a run would write its deck from — the option line,
    the phase-speed window and where it came from, the wavenumber count, the
    range window — without launching anything; every result carries its own
    as ``result.run_settings.engine``.

    Examples
    --------
    >>> from uacpy.models import OAST
    >>> oast = OAST()
    >>> result = oast.run(env, source, receiver)
    """

    # Declarative metadata (see PropagationModel / ModelSpec). OAST:
    # range-independent wavenumber integration; multi-layer fluid + elastic
    # bottom honoured. Single spectral solve → mean SSP / median bottom column.
    # A line Source runs in OAST's plane geometry, option 'P' (ICDR = 1,
    # unoast31.f:1046-1050), which _resolve_engine_settings writes for it.
    spec = ModelSpec(
        modes=(RunMode.COHERENT_TL,),
        # A rough interface is column 7 (RG) of the layer record
        # (oaseun31.f:54).
        supports={'layered_bottom', 'elastic_media',
                  'rough_surface', 'rough_bottom'},
        # A line Source runs in the plane geometry, option 'P'.
        source_types=frozenset({'point', 'line'}),
        collapse={'ssp': 'mean'},
        traits=_OASES_TRAITS,
    )
    provenance_id = 'oases'
    _BINARIES = MappingProxyType({'oast': _oases_binary('oast')})
    # The .plt carries real TL only: a dB pressure field with no phase.
    outputs = MappingProxyType({
        RunMode.COHERENT_TL: OutputSpec(
            'Field', kind='pressure', unit='dB', coherent=True),
    })

    def __init__(
        self,
        *,
        executable: Optional[Path] = None,
        compute_contour: Optional[bool] = None,
        compute_depth_average: Optional[bool] = None,
        complex_contour: Optional[bool] = None,
        options: Optional[str] = None,
        integration_offset: float = OAST_INTEGRATION_OFFSET,
        n_wavenumbers: Optional[int] = None,
        integration_window: Optional[Tuple[int, int]] = None,
        c_low: Optional[float] = None,
        c_high: Optional[float] = None,
        range_min: Optional[float] = None,
        range_max: Optional[float] = None,
        vrec: float = OAST_VREC,
        dip_angle_deg: Optional[float] = None,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        cleanup: Optional[bool] = None,
        timeout: Optional[float] = DEFAULT_RUN_TIMEOUT,
        collapse: Optional[Dict[str, str]] = None,
    ):
        super().__init__(**_host_options(locals()))
        # Kept as passed so ``copy()`` round-trips and so an explicit flag
        # alongside a raw ``options`` string is distinguishable from the
        # default. Resolved by ``_oast_options``.
        self.compute_contour = compute_contour
        self.compute_depth_average = compute_depth_average
        self.complex_contour = complex_contour
        # Raw OASES option string (e.g. ``'K J T A'``). ``None`` lets the
        # wrapper derive it from compute_contour / compute_depth_average /
        # complex_contour; a raw string replaces the deck's option line
        # outright, so the two ways of specifying it are exclusive.
        self.options = options
        self.integration_offset = float(integration_offset)
        # ``None`` lets the OASES kernel pick its own wavenumber sample count.
        # Stored as given, so a count that is not a whole number is refused
        # by _check_knobs rather than truncated.
        self.n_wavenumbers = n_wavenumbers
        # Block VII's IC1/IC2, stored as given (checked by _check_knobs).
        self.integration_window = integration_window
        # Block VII's CMIN/CMAX. ``None`` leaves the derivation
        # (oases_wavenumber_bounds) in place; a value replaces that side of
        # the window.
        self.c_low = float(c_low) if c_low is not None else None
        self.c_high = float(c_high) if c_high is not None else None
        # Plot-axis bounds in metres. ``None`` → 0 / receiver.range_max.
        self.range_min = float(range_min) if range_min is not None else None
        self.range_max = float(range_max) if range_max is not None else None
        # VREC, the fifth frequency-line token, read only under 'd'.
        self.vrec = float(vrec)
        # DA, the dip angle INSRC reads only under the '4' (dip-slip) option.
        self.dip_angle_deg = float(dip_angle_deg) if dip_angle_deg is not None else None
        self._check_knobs()
        if self.n_wavenumbers is not None:
            self._log(_pinned_count_note('OAST', self.n_wavenumbers))

        # Run modes, capability flags and collapse defaults come from the
        # class-level ``spec`` (applied by PropagationModel.__init__).
        self._exe = self._resolve_executable(
            executable, lambda: self._find_binary('oast'),
        )

    def _check_knobs(self) -> None:
        """Refuse a constructor knob no run could use
        (:func:`_check_oast_knobs`, a wavenumber count that is not a power
        of two, :func:`~uacpy.io.oases_writer.
        refuse_non_power_of_two_wavenumbers`, and a pinned phase-speed
        window that is not positive or not ordered,
        :func:`~uacpy.models._window.check_pinned_window`). Run at
        construction and again by every
        run (:meth:`_validate_engine`), since the attributes can be
        reassigned in between."""
        _check_n_wavenumbers_knob(self.n_wavenumbers)
        if self.n_wavenumbers is not None:
            refuse_non_power_of_two_wavenumbers('OAST', self.n_wavenumbers)
        check_pinned_window('OAST', c_low=self.c_low, c_high=self.c_high)
        refuse_unusable_integration_window('OAST', self.integration_window,
                                           self.n_wavenumbers)
        _check_oast_knobs(self.options, self.complex_contour,
                          self.compute_contour, self.compute_depth_average,
                          self.vrec)

    def _multi_frequency_refusal(self, mode, source) -> Exception:
        """The refusal of a frequency sweep, naming the model that does
        compute one (:func:`_oast_multi_frequency_refusal`)."""
        return _oast_multi_frequency_refusal(source)

    def _validate_engine(self, env, source, receiver, run_mode, *,
                         request=None) -> None:
        """Stage 2: the family's boundary refusals
        (:meth:`OASES._validate_engine`), then the refusal of a carrier OAST
        cannot run — a raw ``'P'`` with a point Source
        (:func:`_reject_plane_geometry_letter_on_a_point_source`). A
        frequency sweep is refused before this, in :meth:`_check_carriers`,
        with :meth:`_multi_frequency_refusal`."""
        super()._validate_engine(env, source, receiver, run_mode,
                                 request=request)
        _reject_plane_geometry_letter_on_a_point_source(
            'OAST', _oast_options(self.options, self.complex_contour,
                                  self.compute_depth_average),
            self.options, source,
            'unoast31.f:1046-1050')

    def _resolve_engine_settings(self, env, source, receiver, settings, *,
                                 given_env, request=None) -> 'OASTSettings':
        """Stage 3: :func:`_oast_settings` for this model's knobs."""
        return _oast_settings(
            _oast_options(self.options, self.complex_contour,
                          self.compute_depth_average),
            c_low_pinned=self.c_low, c_high_pinned=self.c_high,
            n_wavenumbers=self.n_wavenumbers,
            integration_window=self.integration_window,
            integration_offset=self.integration_offset,
            range_min=self.range_min, range_max=self.range_max,
            vrec=self.vrec, dip_angle_deg=self.dip_angle_deg,
            env=env, source=source, receiver=receiver)

    def _write_input(self, inputs) -> Path:
        """Stage 4: the OAST deck (:func:`_write_oast_deck`)."""
        deck = inputs.work_dir / f'{_OAST_BASE_NAME}.dat'
        self._log(f"Writing OAST input file: {deck} "
                  f"(options={inputs.settings.engine.options})")
        _write_oast_deck(deck, inputs)
        return deck

    def _launch(self, inputs, deck: Path) -> None:
        """Stage 4: run ``oast`` on the deck, and refuse a run that wrote no
        TL table (FOR020 ``.plt``, per the OASES documentation's FOR019 →
        ``.plp`` / FOR020 → ``.plt``), quoting the binary's streams — OASES
        writes no print file, so they are its only record."""
        proc = self._execute(deck.stem, inputs.work_dir)
        require_output(
            self.model_name, [inputs.work_dir / f'{deck.stem}.plt'],
            what='a TL table (FOR020 .plt)', process=proc,
            hint=('Compare against the reference decks in '
                  'third_party/oases/tloss/.'),
        )

    def _read_output(self, inputs, deck: Path) -> dict:
        """Stage 4: the ``.plt`` TL curves as
        the OAST TL parser returns them."""
        output_file = inputs.work_dir / f'{deck.stem}.plt'
        self._log(f"Reading OAST output: {output_file}")
        return parse_oast_tl(
            filepath=output_file,
            receiver_depths=inputs.receiver.depths,
        )

    def _to_result(self, inputs, deck: Path, raw: dict) -> Field:
        """Stage 5: the real dB TL :class:`Field` on the receiver grid.

        OAST writes only real TL (dB) to its ``.plt`` — there is no complex
        pressure on disk — so receiver ranges off OAST's native FFT range
        grid are obtained by linear interpolation IN dB, which at the
        default grid's half-wavelength step moves the TL by a median
        0.01-0.02 dB (p90 <= 0.09 dB) and only the deepest nulls by more
        (2-14 dB at the worst sample of each measured case); a grid coarser
        than a wavelength is warned about
        (:func:`_warn_if_native_range_step_is_coarse`). The TL on OAST's own
        grid is ``result.components['native']`` whenever the two differ.
        Under option ``'A'`` both carry OAST's depth-averaged TL, on its
        native range grid, as ``components['depth_average']``.
        """
        source = inputs.source
        receiver = inputs.receiver
        tl_data = raw['tl']
        native_depths = raw['depths']
        native_ranges = raw['ranges']
        metadata = raw['metadata']

        frequency = float(np.atleast_1d(source.frequencies)[0])
        components = {}
        if raw['depth_average'] is not None:
            components['depth_average'] = oast_depth_average_field(
                raw['depth_average'], native_ranges,
                **result_kwargs(self.model_name, self.provenance, source,
                                backend='oast', frequencies=frequency))
        kw = result_kwargs(
            self.model_name, self.provenance, source,
            backend='oast', frequencies=frequency, components=components,
        )
        kw['metadata'].update(metadata)
        # The .plt holds real dB with no phase, so the run mode is the one
        # record that this is the coherent TL (Field.coherent reads it).
        native = Field(
            data=tl_data,
            coords={'depth': native_depths, 'range': native_ranges},
            run_mode=inputs.settings.mode,
            **kw,
        )
        result = _oast_on_receiver_grid(native, native_ranges, receiver,
                                        source, inputs.env)

        attach_output_paths(
            result, inputs.work_dir, deck.stem,
            primary_files=(('plt_file', '.plt'),),
            cleanup=self.cleanup,
        )

        self._log("OAST simulation complete")
        return result

    # The wrapper script third_party/oases/bin/oast also names FOR002 (the
    # external source array option 'l' reads) and FOR023 (the tabulated
    # reflection coefficients options 't'/'b' read, oaseun31.f:3727, :3757),
    # but uacpy rejects all three letters (_UNWRITTEN_OPTION_BLOCKS), so
    # those units are never opened and need no entries here.
    _FOR_FILES: dict = {}
    # '.045' is the option-'s' boundary-operator dump (SCTOUT, unoast31.f:1073;
    # FOR045 is set for every run by _oases_subprocess_env). It is listed here
    # because it is an OASS/OASSP *input*: a pinned work_dir that kept a
    # previous run's .rhs would let the mean-field launch's _require_output
    # accept it as this run's, which is the reuse this list exists to close.
    _OUTPUT_SUFFIXES = ('.plt', '.plp', '.045')
    _OUTPUT_FORT_FILES = _SCTOUT_BARE_FORT46
