"""OAST: the OASES wavenumber-integration transmission-loss program."""

import warnings
from dataclasses import dataclass
from types import MappingProxyType
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.models._spec import ModelSpec
from uacpy.core.run_settings import EngineSettings, OutputSpec, RunMode
from uacpy.core.results import Field
from uacpy.core.exceptions import (
    ConfigurationError, NumericsWarning, UnsupportedFeatureError,
)
from uacpy.io.oases_writer import (
    write_oast_input, oases_wavenumber_bounds, _resolve_freq_sweep,
)
from uacpy.io._parsers import parse_oast_tl
from uacpy.models.base import DEFAULT_RUN_TIMEOUT_S
from uacpy.models.oases._common import (
    _OASES_TRAITS, _SCTOUT_BARE_FORT46, _geometry_options,
    _oases_find_executable, _reject_plane_geometry_letter_on_a_point_source,
    _check_n_wavenumbers_knob, _warn_offset_ignored_under_auto_sampling,
)
from uacpy.models.oases._base import OASES
from uacpy.core.engine_defaults import OAST_INTEGRATION_OFFSET, OAST_RANGE_MIN, OAST_VREC


#: File root of every deck and output one OAST run writes. OASS's mean-field
#: producer reads its ``.045`` under this stem.
_OAST_BASE_NAME = 'oast_run'


#: Native range step, in wavelengths of the slowest water speed, above
#: which the dB interpolation onto receiver.ranges is warned about. The
#: default grid sits at 0.5-0.6 wavelengths. Measured on a 100 m Pekeris
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
        :func:`~uacpy.io.oases_writer.oases_wavenumber_bounds` on the water
        column.
    n_wavenumbers : int or None
        Block VII's wavenumber count ``NW``; ``None`` is OASES' automatic
        sampling.
    integration_offset : float
        The contour offset (dB/wavelength) on the frequency line.
    range_min_m, range_max_m : float
        Block VIII's ``XLEFT XRIGHT`` (m), which set OAST's native FFT range
        grid.
    vrec : float
        The frequency line's receiver velocity (m/s), read only under
        ``'d'``.
    dip_angle : float or None
        The dip-slip source's dip (degrees), read only under ``'4'``.
    """

    options: str
    frequency_sweep: Tuple[float, float, int]
    c_low: float
    c_low_origin: str
    c_high: float
    c_high_origin: str
    n_wavenumbers: Optional[int]
    integration_offset: float
    range_min_m: float
    range_max_m: float
    vrec: float
    dip_angle: Optional[float]

    def __post_init__(self):
        # A list (the to_dict form) is stored as the tuple a frozen record
        # holds.
        freq_min, freq_max, n = self.frequency_sweep
        object.__setattr__(self, 'frequency_sweep',
                           (float(freq_min), float(freq_max), int(n)))
        super().__post_init__()


def _oast_options(options, complex_contour, compute_contour,
                  compute_depth_average) -> str:
    """The OASES option line for this run.

    A raw ``options`` string is written verbatim; otherwise the letters
    are derived from the typed flags on top of ``'N T'`` (normal stress
    + TL table). The constructor rejects the two being combined.
    """
    if options is not None:
        return options
    opt = ['N', 'T']
    if complex_contour is None or complex_contour:
        opt.append('J')
    if compute_contour:
        opt.append('C')
    if compute_depth_average:
        opt.append('A')
    return ' '.join(opt)


def _check_oast_knobs(options, complex_contour, compute_contour,
                      compute_depth_average, vrec) -> None:
    """Refuse an OAST knob no run could use: a raw ``options``
    string beside the typed option flags it would discard, and a
    ``vrec`` the option line cannot carry. Run at construction and again
    by every run (:meth:`OAST._validate_engine`), since the attributes
    can be reassigned in between."""
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
    # dip_angle is rejected without '4'.
    option_line = _oast_options(options, complex_contour,
                                compute_contour, compute_depth_average)
    if vrec and 'd' not in set(option_line) - set(' \t\n'):
        raise ConfigurationError(
            f"OAST(vrec={vrec:g}) cannot reach the binary: INFREQ "
            f"reads the fifth frequency-line token only under the "
            f"lowercase 'd' option (unoast31.f:125-127), and the option "
            f"line for this run ({option_line!r}) does not "
            f"carry it.",
            remediation=("Pass a raw options string with 'd' (e.g. "
                         "options='N J T d') alongside vrec, or drop "
                         "vrec."),
        )


def _oast_multi_frequency_refusal(source) -> Exception:
    """OAST's refusal of a frequency sweep, naming the model that does
    compute one (the base decides when a sweep is refused).

    The deck and the reader both handle ``NFREQ > 1``: OAST writes one TL
    curve per plotted receiver per frequency and ``read_oast_tl`` returns
    a ResultStack over frequency. What cannot follow is the
    :class:`~uacpy.core.results.Field`. OAST rebuilds its range grid
    *inside* the frequency loop, so ``DX`` scales as ``1/f`` and the
    sweep has one range axis per frequency — each slab of the reader's
    stack carries its own for exactly that reason — while
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
                   n_wavenumbers, integration_offset, range_min,
                   range_max, vrec, dip_angle, env, source,
                   receiver) -> OASTSettings:
    """Every value the OAST deck is written from, resolved once on the
    projected ``env``: the option line (with ``'P'`` for a line
    Source), Block III's frequency line, Block VII's phase-speed window
    and wavenumber count, and Block VIII's range window."""
    options = _geometry_options(option_line, source)
    ssp_data = env.ssp.extend_to(env.depth).to_pairs()
    c_low, c_high = oases_wavenumber_bounds(ssp_data)
    derived = "oases_wavenumber_bounds(water column)"
    if c_low_pinned is not None:
        c_low = c_low_pinned
    if c_high_pinned is not None:
        c_high = c_high_pinned
    c_low_origin = 'OAST(c_low=…)' if c_low_pinned is not None else derived
    c_high_origin = ('OAST(c_high=…)' if c_high_pinned is not None
                     else derived)
    return OASTSettings(
        options=options,
        frequency_sweep=_resolve_freq_sweep(
            'write_oast_input', source,
            float(source.frequencies[0])),
        c_low=float(c_low),
        c_low_origin=c_low_origin,
        c_high=float(c_high),
        c_high_origin=c_high_origin,
        n_wavenumbers=n_wavenumbers,
        integration_offset=integration_offset,
        range_min_m=(OAST_RANGE_MIN if range_min is None
                     else range_min),
        range_max_m=(float(receiver.ranges.max())
                     if range_max is None else range_max),
        vrec=vrec,
        dip_angle=dip_angle,
    )


def _write_oast_deck(deck: Path, inputs) -> None:
    """The OAST deck at ``deck``, written by
    :func:`~uacpy.io.oases_writer.write_oast_input` from the settings
    ``inputs`` carries."""
    engine = inputs.settings.engine
    writer_kwargs: dict = {
        'integration_offset': engine.integration_offset,
        'n_wavenumbers': engine.n_wavenumbers,
        'vrec': engine.vrec,
        'c_low': engine.c_low,
        'c_high': engine.c_high,
        'range_min': engine.range_min_m,
        'range_max': engine.range_max_m,
    }
    if engine.dip_angle is not None:
        writer_kwargs['dip_angle'] = engine.dip_angle
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
        f"between samples more than a wavelength apart. Measured on a "
        f"Pekeris guide, the p90 error grows from 0.1-0.35 dB at one "
        f"wavelength to 0.5-1.3 dB at two and 1.8-3.8 dB at four, and "
        f"interference nulls read up to 20-34 dB shallow. The step is "
        f"2*pi over the phase-speed window: widen c_low/c_high to refine "
        f"it, read the native grid from metadata['native_ranges'], "
        f"or use OASP (.trf complex pressure on receiver.ranges).",
        NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP,
    )


def _oast_on_receiver_grid(native: Field, native_ranges, receiver, source,
                           env) -> Field:
    """``native``, OAST's TL on its own FFT range grid, on the receiver
    grid: itself when the ranges match, else resampled by linear
    interpolation in dB with the native ranges in
    ``metadata['native_ranges']``; a coarse native step and ranges
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
        result.metadata['native_ranges'] = native_ranges
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
        Add ``'C'`` option (range-depth contour plot). Effective default
        ``False``.
    compute_depth_average : bool, optional
        Add ``'A'`` option (depth-averaged TL). Effective default ``False``.
    complex_contour : bool, optional
        ``'J'`` option (complex integration contour). Effective default
        ``True``.
    options : str, optional
        Raw OASES options string (e.g. ``'N J T C'``), written verbatim;
        ``None`` derives it from ``compute_contour`` /
        ``compute_depth_average`` / ``complex_contour``. Combining a raw
        string with any of those three flags raises
        ``ConfigurationError`` — the string replaces the whole option
        line, so a flag passed alongside it would be discarded.
    integration_offset : float
        Wavenumber-integration contour offset (dB/wavelength). Default 0, which
        under the 'J' option of the default option line is not "no offset":
        OASES takes any value below 1e-10 as a request for its own default,
        60*c/(f*R_max) dB/wavelength with R_max the FFT range
        (unoast31.f:499-501). A value above 1e-10 is used as given.
    n_wavenumbers : int, optional
        Number of wavenumber samples; ``None`` lets OASES choose.
    c_low, c_high : float, optional
        ``CMIN``/``CMAX`` (m/s), Block VII's phase-speed window
        (``unoast31.f:213``). ``None`` derives the pair from the water
        column alone (:func:`oases_wavenumber_bounds`), which is what an
        elastic seabed needs widening: its shear and interface branches have
        phase speeds *below* the slowest water speed, so they fall outside
        the derived ``k_max`` and never enter the integral. ``c_low``
        admits them; ``c_high`` caps the evanescent tail.
    range_min, range_max : float, optional
        TL plot range axis bounds (m); ``None`` → 0 /
        ``receiver.range_max``.
    vrec : float
        Receiver velocity (m/s) for OAST's source/receiver **dynamics**
        option, the fifth token of the frequency line. The binary reads it
        only when the option string carries lowercase ``'d'``
        (``unoast31.f:125-127``), which the derived option line never
        emits — so a non-zero ``vrec`` without a raw ``options`` string
        containing ``'d'`` raises ``ConfigurationError`` (the same
        contract as ``dip_angle``, whose value is read only under
        ``'4'``). Default 0.

        Despite the binary's own "Doppler compensation" wording
        (``unoast31.f:1097-1098``), ``oast.tex:215-222`` is explicit that source
        and receiver move *at the same speed and direction*, so **there is
        no Doppler shift** — only a Green's function different from the
        static one. OASP's ``'d'`` is the real radial-Doppler option and
        carries three tokens (``IT VS VR``, ``oasp.tex:227-234``).
    dip_angle : float, optional
        Fault dip angle (degrees) for the dip-slip moment source that
        option ``'4'`` selects (``unoast31.f:1117-1122``). INSRC reads it
        from the source record — as the second token without ``'L'``
        (``oaseun31.f:1114``), as the seventh with it
        (``oaseun31.f:1089``). ``None`` writes 0 when ``'4'`` is present
        and raises when it is not.
    use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
        Standard plumbing (see :class:`PropagationModel`).

    Notes
    -----
    Range-independent wavenumber integration; consumes layered seabed
    columns natively. **Collapse defaults (overrides of
    :data:`DEFAULT_COLLAPSE`).** Per-model: ``'ssp': 'mean'``,
    ``'bottom_range': 'median'`` (the layer stack is kept).

    ``COHERENT_TL`` here returns a **real dB Field** — ``kind='pressure'``,
    ``unit='dB'``, real dtype (transmission loss is the ``unit`` axis of a
    pressure field, not a registered ``kind``) — where :class:`OASP` returns
    complex Pa pressure for the same run mode: OAST's ``.plt`` carries only
    real TL, so there is no complex pressure to hand back and no phase
    reference to tag. Use :class:`OASP` when the consumer needs complex
    pressure — coherent array processing, phase, or off-grid receiver
    ranges through sharp interference nulls.

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
        supports={'layered_bottom', 'elastic_media',
                  'rough_surface', 'rough_bottom'},
        source_types=frozenset({'point', 'line'}),
        collapse={'ssp': 'mean', 'bottom_range': 'median'},
        traits=_OASES_TRAITS,
    )
    provenance_id = 'oases'
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
        c_low: Optional[float] = None,
        c_high: Optional[float] = None,
        range_min: Optional[float] = None,
        range_max: Optional[float] = None,
        vrec: float = OAST_VREC,
        dip_angle: Optional[float] = None,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        cleanup: Optional[bool] = None,
        timeout: float = DEFAULT_RUN_TIMEOUT_S,
        collapse: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
            cleanup=cleanup, timeout=timeout, collapse=collapse,
        )
        # Kept as passed so ``copy()`` round-trips and so an explicit flag
        # alongside a raw ``options`` string is distinguishable from the
        # default. Resolved by ``_oast_options``.
        self.compute_contour = compute_contour
        self.compute_depth_average = compute_depth_average
        self.complex_contour = complex_contour
        # Raw OASES option string (e.g. ``'N J T C'``). ``None`` lets the
        # wrapper derive it from compute_contour / compute_depth_average /
        # complex_contour; a raw string replaces the deck's option line
        # outright, so the two ways of specifying it are exclusive.
        self.options = options
        self.integration_offset = float(integration_offset)
        # ``None`` lets the OASES kernel pick its own wavenumber sample count.
        # Stored as given, so a count that is not a whole number is refused
        # by _check_knobs rather than truncated.
        self.n_wavenumbers = n_wavenumbers
        # Block VII's CMIN/CMAX. ``None`` leaves the writer's water-column
        # derivation in place; a value replaces that side of the window.
        self.c_low = float(c_low) if c_low is not None else None
        self.c_high = float(c_high) if c_high is not None else None
        # Plot-axis bounds in metres. ``None`` → 0 / receiver.range_max.
        self.range_min = float(range_min) if range_min is not None else None
        self.range_max = float(range_max) if range_max is not None else None
        # VREC, the fifth frequency-line token, read only under 'd'.
        self.vrec = float(vrec)
        # DA, the dip angle INSRC reads only under the '4' (dip-slip) option.
        self.dip_angle = float(dip_angle) if dip_angle is not None else None
        self._check_knobs()
        # OAST reads the offset token under 'J', 'O' (complex frequency) or
        # 'd' (dynamics) — unoast31.f:126-133 — and applies it only under 'J'
        # (:499). The derived line carries 'J' unless complex_contour=False.
        _warn_offset_ignored_under_auto_sampling(
            'OAST', self.integration_offset, self.n_wavenumbers,
            options=_oast_options(self.options, self.complex_contour,
                                  self.compute_contour,
                                  self.compute_depth_average),
            offset_letters='JOd')

        # Run modes, capability flags and collapse defaults come from the
        # class-level ``spec`` (applied by PropagationModel.__init__).
        self._exe = self._resolve_executable(
            executable, lambda: _oases_find_executable(self, 'oast'),
        )

    def _check_knobs(self) -> None:
        """Refuse a constructor knob no run could use
        (:func:`_check_oast_knobs`). Run at construction and again by every
        run (:meth:`_validate_engine`), since the attributes can be
        reassigned in between."""
        _check_n_wavenumbers_knob(self.n_wavenumbers)
        _check_oast_knobs(self.options, self.complex_contour,
                          self.compute_contour, self.compute_depth_average,
                          self.vrec)

    def _multi_frequency_refusal(self, mode, source) -> Exception:
        """The refusal of a frequency sweep, naming the model that does
        compute one (:func:`_oast_multi_frequency_refusal`)."""
        return _oast_multi_frequency_refusal(source)

    def _validate_engine(self, env, source, receiver, run_mode, *,
                         request=None) -> None:
        """Stage 2: the refusal of a carrier OAST cannot run — a raw ``'P'``
        with a point Source
        (:func:`_reject_plane_geometry_letter_on_a_point_source`). A
        frequency sweep is refused before this, in :meth:`_check_carriers`,
        with :meth:`_multi_frequency_refusal`."""
        _reject_plane_geometry_letter_on_a_point_source(
            'OAST', _oast_options(self.options, self.complex_contour,
                                  self.compute_contour,
                                  self.compute_depth_average),
            self.options, source,
            'unoast31.f:1046-1050')

    def _resolve_engine_settings(self, env, source, receiver, settings, *,
                                 given_env, request=None) -> 'OASTSettings':
        """Stage 3: :func:`_oast_settings` for this model's knobs."""
        return _oast_settings(
            _oast_options(self.options, self.complex_contour,
                          self.compute_contour,
                          self.compute_depth_average),
            c_low_pinned=self.c_low, c_high_pinned=self.c_high,
            n_wavenumbers=self.n_wavenumbers,
            integration_offset=self.integration_offset,
            range_min=self.range_min, range_max=self.range_max,
            vrec=self.vrec, dip_angle=self.dip_angle,
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
        self._require_output(
            [inputs.work_dir / f'{deck.stem}.plt'],
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
        (:func:`_warn_if_native_range_step_is_coarse`). The native grid is
        ``metadata['native_ranges']``.
        """
        source = inputs.source
        receiver = inputs.receiver
        tl_data = raw['tl']
        native_depths = raw['depths']
        native_ranges = raw['ranges']
        metadata = raw['metadata']

        kw = self._result_kwargs(
            source,
            backend='oast',
            frequencies=float(np.atleast_1d(source.frequencies)[0]),
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

        self._attach_output_paths(
            result, inputs.work_dir, deck.stem,
            primary_files=(('plt_file', '.plt'),),
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
