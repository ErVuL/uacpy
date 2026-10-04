"""OASR: the OASES plane-wave reflection-coefficient program."""

import warnings
import dataclasses
from dataclasses import dataclass
from types import MappingProxyType
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.source import VALID_SOURCE_TYPES
from uacpy.models._spec import ModelSpec
from uacpy.core.run_settings import EngineSettings, OutputSpec, RunMode
from uacpy.core.results import PhaseReference, ReflectionCoefficient
from uacpy.core.exceptions import (
    ConfigurationError, UnsupportedFeatureError, ValidityWarning,
)
from uacpy.io.oases_writer import write_oasr_input, REFL_TYPE_TO_OPTION
from uacpy.io.oases_reader import read_oasr_reflection_coefficients
from uacpy.models.base import DEFAULT_RUN_TIMEOUT_S
from uacpy.models.oases._common import (
    _OASES_TRAITS, _SCTOUT_BARE_FORT46, _oases_find_executable,
)
from uacpy.models.oases._sampling import _oases_resample_frequencies
from uacpy.models.oases._base import OASES
from uacpy.core.engine_defaults import OASR_ANGLE_TYPE, OASR_REFLECTION_TYPE


#: OASR's default grazing-angle grid as ``(first, last, count)`` in
#: degrees: 0 to 90 in 0.5° steps. A tuple, so no caller can edit
#: the default through an array it was handed.
OASR_DEFAULT_ANGLES_DEG = (0.0, 90.0, 181)


def default_grazing_angles() -> np.ndarray:
    """A fresh array of :data:`OASR_DEFAULT_ANGLES_DEG`."""
    return np.linspace(*OASR_DEFAULT_ANGLES_DEG)


#: File root of every deck and table one OASR run writes.
_OASR_BASE_NAME = 'oasr_run'


@dataclass(frozen=True, eq=False)
class OASRSettings(EngineSettings):
    """The settings one :class:`OASR` run resolved, before launching:
    ``OASR().run_settings(env, source, receiver).engine``, and
    ``result.run_settings.engine`` on the table it produced.

    Attributes
    ----------
    options : str
        The option line the deck carries: the raw ``options`` string, or
        the ``reflection_type`` letter plus ``'T'``.
    reflection_type : str
        The coefficient the option line computes.
    angles_deg : ndarray
        The angle axis as given (degrees, read as ``angle_type``).
    angle_type : str
        ``'grazing'`` or ``'incidence'``.
    frequency_sweep : tuple of (float, float, int)
        Block IV's ``FREQ1 FREQ2 NFREQ``: the frequency bounds (Hz) and
        count.
    log_swept : bool
        Whether option ``'C'`` makes the sweep logarithmic.
    plot_angle_step, plot_frequency_step : int or None
        OASR's plot decimations ``NAOU`` / ``NFOU``; ``None`` leaves the
        writer's default.
    interface_roughness : tuple or None
        Per-interface roughness entries, top to bottom.
    """

    options: str
    reflection_type: str
    angles_deg: np.ndarray
    angle_type: str
    frequency_sweep: Tuple[float, float, int]
    log_swept: bool
    plot_angle_step: Optional[int]
    plot_frequency_step: Optional[int]
    interface_roughness: Optional[tuple]

    _ARRAY_FIELDS = ('angles_deg',)

    def __post_init__(self):
        # A list (the to_dict form) is stored as the tuple a frozen record
        # holds.
        freq_min, freq_max, n = self.frequency_sweep
        object.__setattr__(self, 'frequency_sweep',
                           (float(freq_min), float(freq_max), int(n)))
        if self.interface_roughness is not None:
            object.__setattr__(self, 'interface_roughness',
                               tuple(self.interface_roughness))
        super().__post_init__()


def _oasr_is_log_swept(options) -> bool:
    """True when option ``'C'`` puts OASR on a logarithmic sweep.

    ``oasr.tex:135`` documents ``C`` as a plot option ("Loss contours
    plotted in frequency and grazing angle"), but ``unoasr21.f:123-125``
    computes ``F1LOG``/``DFLOG`` and ``:243`` evaluates the coefficients at
    ``EXP(F1LOG + (JJ-1)*DFLOG)`` — so it changes the frequencies the
    physics is computed at. ``oast.tex:200-203`` confirms it from the other
    side, calling ``C`` the way to obtain "consistent logarithmic sampling".
    """
    return 'C' in (options or '')


def _zero_reflection_reason(reflection_type: str) -> Optional[str]:
    """Why ``reflection_type`` returns a column of zeros from OASR, or
    ``None`` for a type that returns a coefficient."""
    if reflection_type == 'P-SV':
        return ("the incident medium at the seabed is the water column, "
                "a fluid, which carries no SV wave (oasr.tex:143-145)")
    if reflection_type == 'P-Slow':
        return ("it is the Biot slow wave and needs a poro-elastic "
                "medium, which no uacpy carrier expresses")
    return None


def _resolve_reflection_type(options, reflection_type) -> str:
    """The coefficient the deck actually computes.

    Derived from the raw ``options`` letters when one is pinned, so
    the recorded provenance describes the run rather than an unused
    constructor argument. Falls back to ``'P-P'``, OASES' own default
    when no reflection letter appears.

    ``REFL_TYPE_TO_OPTION`` lists the four as peers, but GETOPT does not
    treat them that way (``unoasr21.f:349-378``): ``N``/``S``/``B`` each
    assign ``IPARM`` — the *wave parameter* — while ``'t'`` flips the
    independent ``transmit`` flag, and ``oasjun21.f:80-84`` then picks
    ``trcoef(iparm)`` over ``rfcoef(iparm)``. So ``'N T t'`` is a P-wave
    *transmission* run, not a contradiction: ``'t'`` decides which of the
    two tables is read and wins here, and among the parameter letters the
    last one in the line wins, because each occurrence overwrites
    ``IPARM``.
    """
    if options is None:
        return reflection_type or OASR_REFLECTION_TYPE
    opts = str(options)
    if REFL_TYPE_TO_OPTION['transmission'] in opts:
        return 'transmission'
    by_letter = {letter: name
                 for name, letter in REFL_TYPE_TO_OPTION.items()
                 if name != 'transmission'}
    seen = [by_letter[ch] for ch in opts if ch in by_letter]
    return seen[-1] if seen else OASR_REFLECTION_TYPE


def _reject_shear_reflection_types(options, reflection_type) -> None:
    """Refuse the two reflection types that are identically zero here.

    OASR reflects off the seabed, so the incident medium is the water
    column — a fluid, which carries no shear. `oasr.tex:143-145` defines
    option ``'S'`` as "the P-SV wave reflection coefficient"; with no SV
    wave possible in the incident medium that coefficient is zero for
    **every** environment uacpy can express, and OASES duly writes a
    column of zeros to the ``.rco``. Measured: `max|R| = 0.000000` for
    ``'P-SV'`` against `0.999947` for the default, with an elastic ice
    canopy on the surface making no difference — the canopy is not the
    incident medium at the seabed.

    ``'P-Slow'`` (option ``'B'``) is the Biot slow wave and needs a
    poro-elastic medium; no uacpy carrier expresses one.

    Raised rather than warned: there is no configuration in which either
    returns a usable number, so a warning would leave the caller holding
    an all-zero array that looks like a computed result. A raw
    ``options`` line selecting one is warned about instead
    (:func:`_warn_zero_reflection_options`).
    """
    if reflection_type is None:
        return
    rt = _resolve_reflection_type(options, reflection_type)
    why = _zero_reflection_reason(rt)
    if why is not None:
        raise UnsupportedFeatureError(
            'OASR',
            f"reflection_type={rt!r} — {why}, so OASES returns a column "
            f"of zeros rather than a coefficient. Use "
            f"reflection_type='P-P' (the default) or 'transmission'",
        )


def _warn_zero_reflection_options(options, reflection_type) -> None:
    """Warn once, at construction, when a raw ``options`` line selects a
    reflection type that returns a column of zeros
    (:func:`_reject_shear_reflection_types` says why).

    Reached via the raw ``options=`` escape hatch, which is documented
    as written verbatim. Warn rather than raise so the override stays an
    override — but the zeros are just as useless here."""
    if reflection_type is not None:
        return
    rt = _resolve_reflection_type(options, reflection_type)
    why = _zero_reflection_reason(rt)
    if why is None:
        return
    warnings.warn(
        f"OASR(options={options!r}) selects {rt}, which returns a "
        f"column of zeros: {why}. The deck is written as given.",
        ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP,
    )


def _reject_slowness_sampling(options) -> None:
    """Reject raw ``options`` carrying ``'p'`` (slowness sampling).

    ``write_oasr_input`` always writes Block V as grazing angles in
    degrees; under ``'p'`` the binary reads those same three numbers as
    slownesses in s/km (oasjun21.f:36-37), so the computed grid would
    not be the one the deck asked for.
    """
    if 'p' not in (options or ''):
        return
    raise UnsupportedFeatureError(
        'OASR',
        "option 'p' (slowness sampling): write_oasr_input writes "
        "Block V in grazing degrees, which oasjun21.f:36-37 would "
        "reinterpret as slownesses in s/km — the computed grid would "
        "not be the one requested",
        alternatives=["angle sampling (the default)",
                      "angle_type='incidence'"],
        alternatives_label='samplings',
    )


def _oasr_settings(*, options, reflection_type, angles, angle_type,
                   plot_angle_step, plot_frequency_step,
                   interface_roughness, source) -> OASRSettings:
    """The option line, the angle axis and the frequency sweep the OASR
    deck is written from, for these knobs and ``source``.

    The deck carries the sweep as ``(freq_min, freq_max, N)`` only, so a
    ``source.frequencies`` vector that is not equispaced — or, under
    option ``'C'``, not log-spaced — is evaluated on the grid the
    binary generates, with a warning naming it
    (:func:`_oases_resample_frequencies`).
    """
    reflection_type = _resolve_reflection_type(options, reflection_type)
    log_swept = _oasr_is_log_swept(options)
    source_freqs = np.atleast_1d(np.asarray(source.frequencies,
                                            dtype=float))
    if source_freqs.size > 1:
        freq_min, freq_max, n_freq = _oases_resample_frequencies(
            source_freqs, 'OASR', log_spaced=log_swept,
            argument='source.frequencies')
    else:
        freq_min = freq_max = float(source_freqs[0])
        n_freq = 1
    options = (options if options is not None
               else f"{REFL_TYPE_TO_OPTION[reflection_type]} T")
    return OASRSettings(
        options=options,
        reflection_type=reflection_type,
        angles_deg=(angles if angles is not None
                    else default_grazing_angles()),
        angle_type=angle_type,
        frequency_sweep=(freq_min, freq_max, n_freq),
        log_swept=log_swept,
        plot_angle_step=plot_angle_step,
        plot_frequency_step=plot_frequency_step,
        interface_roughness=(None if interface_roughness is None
                             else tuple(interface_roughness)),
    )


class OASR(OASES):
    """
    OASR - OASES Reflection Coefficients Model

    Computes plane wave reflection coefficients at the bottom interface.

    Notes
    -----
    Range-independent reflection-vs-angle/freq solver. Supports
    elastic / poro-elastic bottom layers. Returns a
    :class:`ReflectionCoefficient` with ``theta`` (1-D grazing-angle grid in
    degrees, whichever ``angle_type`` the input angles were given in; the
    ``angle_type`` used is stamped in the metadata) and ``R``/``phi`` shaped
    ``(n_angles, n_frequencies)`` for a sweep, 1-D for one frequency. A
    multi-frequency Source, equispaced (``np.linspace(f1, f2, n)``), gives
    the broadband sweep; the deck carries only ``(freq_min, freq_max, N)``, so a
    vector that is not equispaced is evaluated on
    ``np.linspace(freq_min, freq_max, N)`` with a warning (on
    ``np.geomspace`` under option ``'C'``, which makes the sweep
    logarithmic).

    ``OASR().run_settings(env, source, receiver).engine`` is the
    :class:`OASRSettings` a run would write its deck from — the option
    line, the angle axis, the frequency sweep — without launching anything;
    every result carries its own as ``result.run_settings.engine``.

    **Collapse defaults (overrides of :data:`DEFAULT_COLLAPSE`).**
    Per-model: ``'bottom_range': 'median'`` (the layer stack is
    kept). SSP collapse left at the global ``'r0'`` because the
    SSP boundary speed is essentially irrelevant to the reflection
    coefficient.

    Examples
    --------
    >>> from uacpy.models import OASR
    >>> oasr = OASR(angles=np.linspace(0, 90, 100))
    >>> refl = oasr.run(env, source, receiver)
    """

    # Declarative metadata (see PropagationModel / ModelSpec). OASR:
    # range-independent reflection vs angle/freq; layered bottom honoured.
    # Only the bottom stack matters — SSP collapse left at the global default
    # ('r0'), as the SSP is essentially irrelevant to the reflection coeff.
    # No 'rough_surface': the OASR deck has no sea surface — its layer 1 is
    # the water half-space the plane wave arrives through, and INENVI
    # discards that record's RG (oaseun31.f:377) — so surface roughness
    # cannot reach the computation and _project_environment discloses the
    # drop instead.
    # The reflection type is read from a raw options= line when one is
    # given (a reflection_type beside it is refused), so a re-run takes the
    # knob as given.
    _UNPINNED_FIELDS = frozenset({'options', 'reflection_type'})

    spec = ModelSpec(
        modes=(RunMode.REFLECTION,),
        supports={'layered_bottom', 'elastic_media', 'rough_bottom'},
        # A plane-wave reflection coefficient is independent of source
        # geometry, and this deck writer proves it: OASR.run reads only
        # ``source.frequencies`` and ``oases_writer.py``'s OASR block never
        # touches ``source.depths`` or ``source.source_type``. Refusing a
        # source type nothing reads would break reusing one Source across
        # models, so every type is accepted — the same reasoning Bounce
        # states for the same run mode.
        source_types=VALID_SOURCE_TYPES,
        collapse={'bottom_range': 'median'},
        # The water is a lossless half-space and a plane-wave interface
        # reflection coefficient has no path for volume absorption to act
        # on.
        traits=dataclasses.replace(_OASES_TRAITS,
                                   consumes_volume_absorption=False),
    )
    provenance_id = 'oases'
    # The .trc table in the travelling-wave phase convention the reader
    # stamps on every ReflectionCoefficient.
    outputs = MappingProxyType({
        RunMode.REFLECTION: OutputSpec(
            'ReflectionCoefficient',
            phase_reference=PhaseReference.TRAVELLING_WAVE.value),
    })

    def __init__(
        self,
        *,
        executable: Optional[Path] = None,
        angles: Optional[np.ndarray] = None,
        angle_type: str = OASR_ANGLE_TYPE,
        reflection_type: Optional[str] = None,
        options: Optional[str] = None,
        plot_angle_step: Optional[int] = None,
        plot_frequency_step: Optional[int] = None,
        interface_roughness: Optional[list] = None,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        cleanup: Optional[bool] = None,
        timeout: float = DEFAULT_RUN_TIMEOUT_S,
        collapse: Optional[Dict[str, str]] = None,
    ):
        """
        Parameters
        ----------
        executable : Path, optional
            Path to OASR binary. Auto-detected if ``None``.
        angles : ndarray, optional
            Angle grid in degrees, uniformly spaced. Default:
            ``np.linspace(0, 90, 181)``. OASR's deck carries only
            ``(ANGLE1, ANGLE2, NANG)`` and generates the grid itself
            (``unoasr21.f:173-177``), so a non-uniform array raises
            ``ConfigurationError`` rather than being silently resampled.
            Default ``linspace(0, 90, 181)``.
        angle_type : str, optional
            'grazing' (OASES native) or 'incidence' (converted via
            ``grazing = 90 - incidence`` before being written to the input
            file). Default: 'grazing'.
        reflection_type : str, optional
            ``'P-P'`` (effective default) | ``'transmission'``, translated to
            OASR option letter ``'N'`` / ``'t'``. ``'P-SV'`` and ``'P-Slow'``
            raise :class:`UnsupportedFeatureError`: the incident medium at the
            seabed is the water column, which carries no SV wave, and no uacpy
            carrier expresses the poro-elastic medium the Biot slow wave needs.
            Mutually exclusive with a raw ``options`` string.
        options : str, optional
            Raw OASES option string, written verbatim. ``None`` lets the
            wrapper derive it from ``reflection_type``. Passing both
            raises ``ConfigurationError`` — the deck would run whichever
            reflection the raw string selects, not the named one.
        plot_angle_step : int, optional
            OASR's ``NAOU``: decimates only OASES' own reflection-vs-angle
            *plot* curves. The ``.rco``/``.trc`` tables uacpy reads back
            carry every angle regardless, so this cannot change any number
            in the returned :class:`ReflectionCoefficient`. ``None`` leaves
            the knob out of the writer call, which applies its own
            ``max(1, n_angles // 10)`` — every 18th sample on the default
            181-angle grid, not every sample.
        plot_angle_step, plot_frequency_step : int, optional
            ``NAOU`` / ``NFOU``, which decimate only OASES' own plot curves; the
            tables uacpy reads carry every sample.
        plot_frequency_step : int, optional
            OASR's ``NFOU`` (``unoasr21.f:116``): decimates only OASES' own
            reflection-vs-frequency *plot* curves; like ``NAOU`` it never
            touches the tables uacpy returns. ``None`` → the writer's
            ``max(1, n_frequencies // 10)``.
        interface_roughness : list, optional
            Per-interface RMS roughness in metres, one entry per layer
            record, ordered top → bottom. Entry 0 is OASR's layer-1
            (water half-space) record, whose RG INENVI discards
            (``oaseun31.f:377``); entry 1 is the reflecting water/seabed
            interface. An entry may be a float, or a ``(RG, CL, M)``
            tuple / ``{'RG', 'CL', 'M'}`` dict to select the Goff-Jordan
            roughness spectrum. Unset entries fall back to the
            environment's own interface roughness.
        use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
            Standard plumbing (see :class:`PropagationModel`).
        """
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
            cleanup=cleanup, timeout=timeout, collapse=collapse,
        )
        self.angles = (
            np.asarray(angles, dtype=float) if angles is not None else None
        )
        self.angle_type = angle_type
        # Kept as passed so ``copy()`` round-trips and so an explicit value
        # alongside a raw ``options`` string is distinguishable from the
        # default. Resolved by ``_resolve_reflection_type``.
        self.reflection_type = reflection_type
        self.options = options
        self._check_knobs()
        _warn_zero_reflection_options(self.options, self.reflection_type)
        self.plot_angle_step = (
            int(plot_angle_step)
            if plot_angle_step is not None else None
        )
        self.plot_frequency_step = (
            int(plot_frequency_step)
            if plot_frequency_step is not None else None
        )
        self.interface_roughness = (
            list(interface_roughness) if interface_roughness else None
        )

        # OASR is strictly a boundary-reflection solver; it does not produce
        # transmission loss. Declare that explicitly.
        # Run modes, capability flags and collapse defaults come from the
        # class-level ``spec`` (applied by PropagationModel.__init__).
        self._exe = self._resolve_executable(
            executable, lambda: _oases_find_executable(self, 'oasr'),
        )

    def _check_knobs(self) -> None:
        """Refuse a constructor knob no run could use: a raw ``options``
        string beside the ``reflection_type`` it would discard, a typed
        reflection type that is identically zero
        (:func:`_reject_shear_reflection_types`) and slowness sampling
        (:func:`_reject_slowness_sampling`). Run at construction and again
        by every run (:meth:`_validate_engine`), since the attributes can be
        reassigned in between."""
        if self.options is not None and self.reflection_type is not None:
            raise ConfigurationError(
                f"OASR: options={self.options!r} replaces the whole option "
                f"line, so reflection_type={self.reflection_type!r} would be "
                f"recorded in the result metadata without ever reaching the "
                f"deck. Pass either the raw string or reflection_type, not "
                f"both."
            )
        _reject_shear_reflection_types(self.options, self.reflection_type)
        _reject_slowness_sampling(self.options)

    def _resolve_engine_settings(self, env, source, receiver, settings, *,
                                 given_env, request=None) -> 'OASRSettings':
        """Stage 3: :func:`_oasr_settings` for this model's knobs."""
        return _oasr_settings(
            options=self.options, reflection_type=self.reflection_type,
            angles=self.angles, angle_type=self.angle_type,
            plot_angle_step=self.plot_angle_step,
            plot_frequency_step=self.plot_frequency_step,
            interface_roughness=self.interface_roughness, source=source)

    def _write_input(self, inputs) -> Path:
        """Stage 4: the OASR deck, written by
        :func:`~uacpy.io.oases_writer.write_oasr_input` from the resolved
        settings."""
        engine = inputs.settings.engine
        deck = inputs.work_dir / f'{_OASR_BASE_NAME}.dat'
        freq_min, freq_max, n_frequencies = engine.frequency_sweep
        writer_kwargs: dict = {
            'freq_min': freq_min,
            'freq_max': freq_max,
            'n_frequencies': n_frequencies,
        }
        if engine.plot_angle_step is not None:
            writer_kwargs['plot_angle_step'] = \
                engine.plot_angle_step
        if engine.plot_frequency_step is not None:
            writer_kwargs['plot_frequency_step'] = \
                engine.plot_frequency_step
        if engine.interface_roughness is not None:
            writer_kwargs['interface_roughness'] = list(
                engine.interface_roughness)
        self._log(f"Writing OASR input file: {deck}")
        write_oasr_input(
            filepath=deck,
            env=inputs.env,
            source=inputs.source,
            receiver=inputs.receiver,
            options=engine.options,
            angles=np.array(engine.angles_deg),
            angle_type=engine.angle_type,
            reflection_type=engine.reflection_type,
            **writer_kwargs,
        )
        return deck

    # Under option 'T' REFLEC writes both tables on every sample — slowness
    # in s/km to unit 22 (.rco) and grazing angle in degrees to unit 23
    # (.trc), oasjun21.f:103-104 — so the pair holds the same coefficients
    # under two abscissae and the choice is uacpy's. Grazing degrees is
    # always the right one: the only option that would make the slowness
    # abscissa the requested grid is 'p' (oasjun21.f:36-37 rereads
    # ANGLE1/ANGLE2/NANG as slowness), and _reject_slowness_sampling refuses
    # that outright, in __init__ and again in _validate_engine.
    _TABLE_SEARCH = ('.trc', '.rco')

    def _launch(self, inputs, deck: Path) -> None:
        """Stage 4: run ``oasr`` on the deck, and refuse a run that wrote
        no reflection-coefficient table, quoting the binary's streams."""
        proc = self._execute(deck.stem, inputs.work_dir)
        self._require_output(
            [inputs.work_dir / f'{deck.stem}{ext}'
             for ext in self._TABLE_SEARCH],
            what='a reflection-coefficient table', process=proc,
        )

    def _read_output(self, inputs, deck: Path):
        """Stage 4: the grazing-angle table as
        :func:`~uacpy.io.oases_reader.read_oasr_reflection_coefficients`
        returns it."""
        output_file = self._require_output(
            [inputs.work_dir / f'{deck.stem}{ext}'
             for ext in self._TABLE_SEARCH],
            what='a reflection-coefficient table',
        )
        self._log(f"Reading OASR output: {output_file}")
        return read_oasr_reflection_coefficients(output_file)

    def _to_result(self, inputs, deck: Path, raw):
        """Stage 5: the :class:`ReflectionCoefficient` as the run's own —
        ``theta`` in grazing degrees, the travelling-wave phase of
        ``inputs.settings.output`` the reader stamps — with the reflection
        type as its :attr:`~ReflectionCoefficient.reflection_type` and the
        sampling and angle types in its ``metadata``."""
        engine = inputs.settings.engine
        table = ReflectionCoefficient(
            angles=raw.angles, magnitude=raw.magnitude, phase=raw.phase,
            reflection_type=engine.reflection_type, **raw.id_kwargs())
        # read_oasr_reflection_coefficients refuses a slowness table, so
        # the table read is angle-sampled.
        field = self._stamp_file_result(
            table, inputs.source, backend='oasr',
            sampling_type='angle',
            angle_type=engine.angle_type,
        )
        self._attach_output_paths(
            field, inputs.work_dir, deck.stem,
            primary_files=(
                ('trc_file', '.trc'),
                ('rco_file', '.rco'),
            ),
        )

        self._log("OASR simulation complete")
        return field

    # Units 22 and 23 are the two reflection-coefficient tables option 'T'
    # opens (unoasr21.f:201-205). They are written together, not either/or:
    # oasjun21.f:103 puts slowness on 22 and :104 grazing angle on 23, and
    # the run reads the grazing one (_TABLE_SEARCH).
    # third_party/oases/bin/oasr also names FOR002 and FOR004, but those
    # units OASR never opens, so they need no entries here.
    _FOR_FILES = {
        'FOR022': 'rco',
        'FOR023': 'trc',
    }
    # Every unit is env-assigned, so the outputs always land on the
    # base_name suffixes; a run additionally leaves .021 behind.
    # '.045' as for OAST: option 's' (SCTOUT, unoasr21.f:400) dumps the
    # boundary operators there, and a stale one would be read as this run's.
    _OUTPUT_SUFFIXES = ('.rco', '.trc', '.plt', '.plp', '.021', '.045')
    _OUTPUT_FORT_FILES = _SCTOUT_BARE_FORT46
