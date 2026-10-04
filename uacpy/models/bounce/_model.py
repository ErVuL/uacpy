"""The :class:`Bounce` engine class: its knobs, its declarations and
the protocol hooks that validate, resolve, write, launch and read one BOUNCE
deck."""

from pathlib import Path
from types import MappingProxyType
from typing import Dict, Optional, Union

import numpy as np

from uacpy.models._band import frequencies_text
from uacpy.models.base import DEFAULT_RUN_TIMEOUT_S, PropagationModel
from uacpy.core.source import VALID_SOURCE_TYPES
from uacpy.models._spec import (
    _SINGLE_FREQUENCY_MODES, EngineTraits, ModelSpec,
)
from uacpy.core.run_settings import OutputSpec, RunMode
from uacpy.core.results import PhaseReference, ReflectionCoefficient
from uacpy.models._defaults import DEFAULT_C_MIN
from uacpy.core.exceptions import (
    ConfigurationError,
    ModelExecutionError,
    UnsupportedFeatureError,
)
from uacpy.io.refl_io import (
    read_reflection_coefficient, dedupe_reflection_file, _scale_irc_impedance,
)
from uacpy.io.oalib_writer import write_bounce_input_file
from uacpy.models.bounce._plan import (
    check_knobs, check_phase_speed_window, refuse_precalc_seabed,
    refuse_too_few_angles, refuse_unsized_table,
    reject_c_low_above_the_water, resolve_c_high, resolve_c_low,
    resolve_n_mesh,
    resolve_rmax_m, tabulated_angle_count,
)
from uacpy.models.bounce._settings import BounceSettings

# BOUNCE always writes both tables, so both are cleared before a launch — but a
# tabulated seabed makes one of them an *input* too: write_bottom_section stages
# the user's table there and misc/RefCoef.f90:39 ('F' -> .brc) / :94 ('P' ->
# .irc) open it with STATUS='OLD' before ComputeReflectionCoefficient rewrites
# it. That one must survive the pre-launch sweep. A 'precalc' seabed (.irc) is
# refused before the launch (``Bounce._validate_engine``), so only the .brc is
# ever staged.
_STAGED_TABLE_SUFFIX = {'file': '.brc'}
_BOUNCE_OUTPUTS = ('.brc', '.irc')

# File root of every deck and table one run writes.
_BASE_NAME = 'bounce_run'


class Bounce(PropagationModel):
    """
    BOUNCE - Reflection Coefficient Model (Acoustics Toolbox)

    Computes plane wave reflection coefficients for a stack of acoustic/elastic
    layers. The reflection coefficient is written to both .BRC (Bottom Reflection
    Coefficient) and .IRC (Internal Reflection Coefficient) files.

    Model Support:
    - .BRC files: BELLHOP, SCOOTER, KRAKENC
    - .IRC files: KRAKENC, SCOOTER (BELLHOP has no 'P' branch)
    - SPARC: does not support reflection files

    Notes
    -----
    Only emits ``RunMode.REFLECTION``. The result always carries the
    in-memory reflection coefficient as typed attributes
    (``.angles``, ``.magnitude``, ``.phase``); the standalone Python user does not
    need the on-disk files.

    To **chain to another model** (Bellhop / Scooter / Kraken reading
    ``acoustic_type='file'`` or ``'precalc'``), pin ``work_dir=`` so the
    ``.brc`` / ``.irc`` files outlive the call. The same uniform
    ``(work_dir, cleanup)`` rule every other model uses applies here:

    - ``Bounce(work_dir='./bounce_out')`` ⇒ files persist there
      (``cleanup=False`` because the user owns the dir);
      ``result.metadata['brc_file']`` is a valid path.
    - ``Bounce()`` (no ``work_dir``) ⇒ uacpy uses a temp dir,
      ``cleanup=True`` ⇒ files are removed when ``run()`` returns;
      ``result.metadata`` does not carry the (now stale) file paths.

    **Collapse defaults (overrides of :data:`DEFAULT_COLLAPSE`).**
    BOUNCE produces ONE BRC consumed across the whole receiver-range
    axis; the median sample is the most representative single profile.
    Per-model: ``'bottom_range': 'median'`` (the layer stack is kept since
    BOUNCE consumes layered seabed columns natively).

    Model characteristics:

    - BOUNCE uses the same environmental file format as KRAKEN
    - The reflection coefficient depends on impedance contrast
    - Supports acoustic, elastic, and poro-elastic layers
    - Tabulated reflection coefficients cover angles from phase velocities [c_low, c_high]
    - Both tables go through the standard AT reflection-coefficient path:
      ``.brc`` via ``acoustic_type='file'``, ``.irc`` via
      ``acoustic_type='precalc'``. Kraken routes either to krakenc.exe.

    Defaults auto-derived at ``run()`` time:

    - ``rmax_m=None`` → ``receiver.range_max`` (or 10 km if 0).
    - ``c_low=None`` → ``min(DEFAULT_C_MIN, min(env.ssp))``, AT's
      ``bounce.htm`` rule "the lowest speed in the problem"; with
      ``c_high=None`` → ``DEFAULT_C_MAX_UNBOUNDED`` this tabulates the full
      0–90° grazing span in cold and brackish water as well as ordinary sea
      water. Lower ``c_high`` to concentrate the samples on a narrower
      angular band. ``c_low`` cannot be raised past the water sound speed —
      that truncates the grazing wedge instead (``run()`` refuses it).
    - TopOpt position 4 reads ``env.absorption``; AT adds a lettered volume
      term (Thorp, Biological) to the sediment media BOUNCE tabulates. A law
      the water SSP rows carry (Francois-Garrison, a constant, a table)
      does not reach it: a BOUNCE deck holds no water rows. ``supports_feature(
      'volume_attenuation')`` still answers ``False`` (see
      ``consumes_volume_absorption`` below).

    With ``verbose='info'`` the resolved ``rmax_m`` is logged.
    ``Bounce().run_settings(env, source, receiver).engine`` is the
    :class:`BounceSettings` a run would use — ``c_low``, ``rmax_m`` and
    where each came from, the ``NkTab`` count, the mesh — without launching
    anything; every result carries its own as
    ``result.run_settings.engine``.

    Examples
    --------
    Compute reflection coefficients for use in other models:

    >>> from uacpy.models import Bounce
    >>> from uacpy.core import Environment, Source, Receiver, BoundaryProperties
    >>> import numpy as np
    >>>
    >>> # Define environment with elastic bottom
    >>> bottom = BoundaryProperties(
    ...     acoustic_type='half-space',
    ...     sound_speed=1600,
    ...     shear_speed=400,
    ...     density=1.8,
    ...     attenuation=0.2,
    ...     shear_attenuation=0.5
    ... )
    >>> env = Environment(name="test", bathymetry=100, bottom=bottom)
    >>> source = Source(depths=50, frequencies=50)
    >>> receiver = Receiver(depths=np.array([50]))
    >>>
    >>> # Pin work_dir so the .brc/.irc files persist for the consumer; a
    >>> # temporary directory keeps them out of the caller's own tree. The
    >>> # files are removed when the ``with`` block exits, so the consumer
    >>> # runs inside it.
    >>> import tempfile
    >>> from uacpy.models import Scooter
    >>> with tempfile.TemporaryDirectory() as d:
    ...     bounce = Bounce(c_low=1400, rmax_m=10000, work_dir=d)
    ...     result = bounce.run(env, source, receiver)
    ...     # Output files can be used by different models:
    ...     # - .brc file → Bellhop, Scooter, Kraken (run as krakenc)
    ...     # - .irc file → Kraken (run as krakenc), Scooter
    ...     bottom_with_rc = BoundaryProperties(
    ...         acoustic_type='file',
    ...         reflection_file=result.metadata['brc_file'],
    ...     )
    ...     env_with_rc = Environment(name="test", bathymetry=100,
    ...                               bottom=bottom_with_rc)
    ...     tl = Scooter().compute_tl(env_with_rc, source, receiver)

    References
    ----------
    - Porter, M.B., "The KRAKEN Normal Mode Program", SACLANT Undersea Research
      Centre Memorandum SM-245, 1991
    - Acoustics Toolbox: http://oalib.hlsresearch.com/
    """

    # ``consumes_volume_absorption`` stays at the default False although
    # TopOpt position 4 does reach the engine (AttenMod.f90 adds a lettered
    # Thorp or biological term to every CRCI call, so to the sediment media
    # BOUNCE tabulates): the flag also gates the lossless-water notice, and
    # BOUNCE tabulates R(theta) at an interface whose `receiver` is read only
    # for `range_max`, which sizes the table's angular resolution rather than
    # describing a path. The notice would quote that knob as a propagation
    # distance.

    # Declarative metadata read and validated by PropagationModel. BOUNCE
    # emits only plane-wave reflection coefficients; it consumes layered and
    # elastic seabed columns natively (so those env shapes are *not*
    # collapsed) but handles no range dependence. It produces ONE BRC used
    # across the whole receiver-range axis, so a range-dependent bottom is
    # reduced to its most-representative single column (median).
    spec = ModelSpec(
        modes=(RunMode.REFLECTION,),
        supports={'layered_bottom', 'elastic_media'},
        # Reflection coefficients are independent of source geometry, so
        # rejecting one would break reusing a Source across models.
        source_types=VALID_SOURCE_TYPES,
        collapse={'bottom_range': 'median'},
        traits=EngineTraits(
            none_receiver_modes=frozenset({RunMode.REFLECTION}),
            # BOUNCE writes one frequency into the ``.env`` and returns one
            # table, so its REFLECTION takes a single frequency; the base
            # rule refuses a multi-frequency Source (OASR's REFLECTION
            # sweeps, so the mode stays out of the package-wide set).
            single_frequency_modes=(_SINGLE_FREQUENCY_MODES
                                    | {RunMode.REFLECTION}),
        ),
    )
    provenance_id = 'acoustics_toolbox'

    #: The settings' ``n_angles`` is the count the deck tabulates, which
    #: follows from the recorded ``rmax_m``; as a knob it would re-derive
    #: ``rmax_m`` instead (:meth:`from_run_settings`).
    _UNPINNED_FIELDS = frozenset({'n_angles'})
    # The .brc table as ``misc/RefCoef.f90`` reads it back: |R| and a phase
    # in the travelling-wave convention every AT consumer applies.
    outputs = MappingProxyType({
        RunMode.REFLECTION: OutputSpec(
            'ReflectionCoefficient',
            phase_reference=PhaseReference.TRAVELLING_WAVE.value),
    })
    def __init__(
        self,
        *,
        executable: Optional[Path] = None,
        c_low: Optional[float] = None,
        c_high: Optional[float] = None,
        rmax_m: Optional[float] = None,
        n_angles: Optional[int] = None,
        interp_ssp: Optional[str] = None,
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
            Path to bounce executable. Auto-detected if None.
        c_low : float, optional
            Minimum phase velocity (m/s) for tabulation. ``None`` (default)
            derives it from the environment at ``run()`` time as
            ``min(DEFAULT_C_MIN, min(env.ssp))`` — AT's ``bounce.htm`` asks
            for "the lowest speed in the problem (say 1400.0)", and 1400 is
            that sentence's example, not its rule, so a column with any water
            slower than 1400 m/s needs the lower value.
            Must be strictly positive (BOUNCE rejects ``c_low <= 0`` — the
            angular grid is derived from ``kx = omega/c``) and must not
            exceed the water sound speed at the seafloor, which BOUNCE takes
            as its reference speed (``bounce.f90:186-195``): above it the
            table starts at ``atan2(sqrt(k0**2 - kMax**2), kMax)`` rather
            than 0 deg and the grazing wedge is unrecoverable. ``run()``
            raises rather than emit such a table.
        c_low, c_high : float, optional
            Phase-velocity bounds for tabulation (m/s). ``c_low`` must be
            strictly positive (BOUNCE rejects ``c_low <= 0``); ``None``
            (default) derives it from the environment at ``run()`` time as
            ``min(DEFAULT_C_MIN, min(env.ssp))``. ``c_high``: ``None``
            (default) resolves at ``run()`` time to ``DEFAULT_C_MAX_UNBOUNDED``.
        c_high : float, optional
            Maximum phase velocity (m/s) for tabulation. Default: 1e9, which
            trips ``bounce.f90:47``'s ``IF ( cHigh > 1.0E6 ) kMin = 0.0`` and
            so tabulates the full 0–90° grazing span. A finite ``c_high``
            stops the table at ``acos(c0 / c_high)`` and every consumer
            silently returns ``R = 0, phi = 0`` above the last tabulated
            angle (``misc/RefCoef.f90:144-149``, both of whose warning WRITEs
            are commented out). Must be strictly greater than c_low.
            ``None`` stands for the default.
        rmax_m : float, optional
            Maximum range (m) for angular sampling. ``None`` (default)
            auto-derives from ``receiver.range_max`` at ``run()`` time,
            falling back to 10000 m when no receiver range is available.
            Ignored when ``n_angles`` is provided. (Internally converted to
            km because BOUNCE's input format is in km.)
        n_angles : int, optional
            Explicit override for the number of angular samples (``NkTab``
            in AT's bounce). If None (default), bounce computes NkTab
            internally from ``rmax_m``. When provided, uacpy sets ``rmax_m``
            such that bounce's internal formula yields approximately
            ``n_angles`` samples. A whole number, at least 2.
        interp_ssp : str, optional
            Sample-connection scheme written into ``TopOpt(1)``. A BOUNCE deck
            carries only the seabed stack — its media are 2-point linear slabs —
            so this rarely changes the answer, and ``'quad'`` is rejected
            outright with :class:`UnsupportedFeatureError` (no water column
            means no ``.ssp`` for it to read).
        use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
            Standard plumbing (see :class:`PropagationModel`).

        Raises
        ------
        ConfigurationError
            For a knob no run could use: a value that is not a number, a
            non-positive speed or range, ``c_high <= c_low``, an
            ``n_angles`` that is not a whole number of at least 2.
        """
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
            cleanup=cleanup, timeout=timeout, collapse=collapse,
        )
        # BOUNCE meshes only the seabed stack — write_bounce_input_file emits
        # no water column and no .ssp — so the range-dependent 'quad' scheme
        # has no file to read and ERROUTs at NMedia.
        # UnsupportedFeatureError, the type every AT wrapper raises for
        # 'quad' (oalib_writer.reject_unsupported_ssp_interp).
        if interp_ssp is not None and str(interp_ssp).lower() in (
                'q', 'quad', 'quadratic'):
            raise UnsupportedFeatureError(
                'Bounce',
                "interp_ssp='quad' — BOUNCE reflection decks carry no water "
                "column, so no .ssp file is written for the quad scheme to "
                "read and the binary stops at NMedia",
                alternatives=["'linear' (default)", "'n2linear'", "'pchip'",
                              "'spline'"],
                alternatives_label='SSP interpolations',
            )
        self.interp_ssp = interp_ssp

        self.c_low = c_low
        self.c_high = c_high
        self.rmax_m = rmax_m
        self.n_angles = n_angles
        self._check_knobs()

        # Run modes, capability flags and collapse defaults come from the
        # class-level ``spec`` (applied by PropagationModel.__init__).
        self._exe = self._resolve_executable(
            executable,
            lambda: self._find_executable_in_paths(
                'bounce', bin_subdirs='oalib',
                dev_subdir='Acoustics-Toolbox/Kraken',
            ),
        )

    def _validate_geometry(self, env, source, receiver, run_mode=None) -> None:
        """No-op: BOUNCE reads no source or receiver geometry.

        Its Fortran driver stops after ``TopOpt``, the SSP, ``BotOpt``,
        ``cLow``/``cHigh`` and ``RMax`` (``bounce.f90``) — it never calls
        ``ReadSzRz``. The plane-wave reflection coefficient is independent of
        source position, so the field models' depth and range-coverage checks
        would reject and warn about geometry that never reaches the binary.
        ``receiver`` is read only to auto-derive ``rmax_m``.
        """

    def _multi_frequency_refusal(self, mode, source) -> Exception:
        """The base rule's refusal of a multi-frequency Source, naming
        Bounce's remedies (it has no broadband mode)."""
        src_freqs = np.atleast_1d(source.frequencies)
        return ConfigurationError(
            f"Bounce tabulates the reflection coefficient at one "
            f"frequency; got {len(src_freqs)}: "
            f"{frequencies_text(src_freqs)}. Loop "
            f"over single-frequency Sources, or use OASR for a "
            f"multi-frequency reflection sweep."
        )

    def _check_knobs(self) -> None:
        """Refuse a knob no run could use
        (:func:`~uacpy.models.bounce._plan.check_knobs`)."""
        check_knobs(c_low=self.c_low, c_high=self.c_high,
                    rmax_m=self.rmax_m, n_angles=self.n_angles)

    def _validate_engine(self, env, source, receiver, run_mode, *,
                         request=None) -> None:
        """Stage 2: the refusals of carriers BOUNCE cannot run — on the
        projected environment, so ``validate_inputs`` refuses what ``run``
        refuses.

        - ``receiver=None`` with nothing else to size the table
          (``rmax_m`` and ``n_angles`` both unset);
        - a ``'precalc'`` (``.irc``) seabed, which the binary cannot re-read
          and rewrite in one launch.
        """
        refuse_unsized_table(receiver, rmax_m=self.rmax_m,
                             n_angles=self.n_angles)
        refuse_precalc_seabed(self.model_name, env)

    def _resolve_engine_settings(self, env, source, receiver, settings, *,
                                 given_env, request=None) -> BounceSettings:
        """Stage 3: every setting of the deck, resolved once, and the
        refusals of a deck BOUNCE cannot run.

        ``c_low`` reads the environment as given (``given_env``; see
        :func:`~uacpy.models.bounce._plan.resolve_c_low`), and everything
        else the projected ``env`` the deck is written from. Refused: a
        window with ``c_low`` at or above ``c_high``, fewer than two
        tabulated angles, a ``c_low`` above the water speed at the
        seafloor, a sediment layer that needs more mesh than one medium
        holds.
        """
        c_low = resolve_c_low(given_env, c_low=self.c_low)
        c_high, c_high_origin = resolve_c_high(self.c_high)
        check_phase_speed_window(c_low, c_high=c_high)
        frequency = float(np.atleast_1d(source.frequencies)[0])
        rmax_m, rmax_origin = resolve_rmax_m(
            receiver, frequency, c_low, n_angles=self.n_angles,
            rmax_m=self.rmax_m, c_high=c_high)
        self._log(f"rmax_m = {rmax_m:.1f} m (from {rmax_origin})")
        n_angles = tabulated_angle_count(rmax_m, frequency, c_low,
                                         c_high=c_high)
        # NkTab is bounded below only. No upper cap: the binary holds four
        # NkTab-length tables (xTab/fTab/gTab/ITab, bounce.f90:52) — tens of
        # bytes per tabulated angle, ~5 MB at a count whose Green's-function
        # cube costs Scooter gigabytes (the sibling Scooter guards with
        # ``scooter._plan.reject_oversized_green_cube``) — and refuses a
        # failed allocation itself through its IAllocStat test.
        refuse_too_few_angles(n_angles, frequency, c_low, rmax_m=rmax_m,
                              rmax_origin=rmax_origin, c_high=c_high)
        self._log(f"NkTab = {n_angles} tabulated angles", level='debug')
        reject_c_low_above_the_water(env, c_low)
        seabed_type = env.bottom.halfspace_at(range=0.0).acoustic_type
        n_mesh = resolve_n_mesh(env, frequency)
        self._log(f"mesh points per medium = {n_mesh}", level='debug')
        return BounceSettings(
            c_low=c_low,
            c_low_origin=('Bounce(c_low=…)' if self.c_low is not None else
                          f"min({DEFAULT_C_MIN:g}, min(env.ssp))"),
            c_high=c_high,
            c_high_origin=c_high_origin,
            rmax_m=rmax_m,
            rmax_origin=rmax_origin,
            n_angles=n_angles,
            n_mesh=tuple(n_mesh),
            staged_table_suffix=_STAGED_TABLE_SUFFIX.get(seabed_type),
        )

    def _write_input(self, inputs) -> Path:
        """Stage 4: the BOUNCE deck, written by
        :func:`~uacpy.io.oalib_writer.write_bounce_input_file` from the
        resolved settings.

        BOUNCE uses the KRAKEN ENV format plus ``cLow``/``cHigh`` and
        ``RMax`` (in km, converted from ``rmax_m``). It does NOT call
        ``ReadSzRz``: its Fortran driver reads only TopOpt, SSP, BotOpt,
        cLow/cHigh, RMax, so the deck carries no source or receiver depth
        block and ``inputs.receiver`` is not read here.
        """
        deck = inputs.work_dir / f'{_BASE_NAME}.env'
        self._log(f"Writing input file: {deck}")
        engine = inputs.settings.engine
        write_bounce_input_file(
            deck, inputs.env, inputs.source,
            interp_ssp=self.interp_ssp,
            n_mesh=list(engine.n_mesh),
            c_low=engine.c_low,
            c_high=engine.c_high,
            rmax_m=engine.rmax_m,
            verbose=self.verbose,
        )
        return deck

    def _launch(self, inputs, deck: Path) -> None:
        """Stage 4: run ``bounce`` on the deck through the shared
        binary-launch helper. The table a ``'file'`` seabed staged next to
        the deck as input (``inputs.settings.engine.staged_table_suffix``, see
        :data:`_STAGED_TABLE_SUFFIX`) is kept out of the stale-output sweep,
        so the binary can still read it."""
        self._log("Running...")
        base_name = deck.stem
        staged = inputs.settings.engine.staged_table_suffix
        self._run_and_attach_prt(
            [str(self._exe), base_name], inputs.work_dir, base_name,
            stale_outputs=tuple(s for s in _BOUNCE_OUTPUTS if s != staged))

    def _read_output(self, inputs, deck: Path) -> ReflectionCoefficient:
        """Stage 4: the ``.brc`` table as
        :func:`~uacpy.io.refl_io.read_reflection_coefficient` returns it,
        after the two corrections the tables need where they are made."""
        base_name = deck.stem
        work_dir = inputs.work_dir
        # A missing or empty .brc means the binary died silently; the
        # raised error carries the .prt tail with the actual cause.
        brc_file = self._require_output(
            [work_dir / f'{base_name}.brc'],
            what='a reflection-coefficient table (.brc)',
            prt_base=base_name, work_dir=work_dir,
        )
        # Normalise the raw BOUNCE table before it is read back into the
        # ReflectionCoefficient result: bellhopcuda's strict monotonicity
        # check rejects the duplicate near-zero angles bounce.f90 emits when
        # many high-c samples round to the same kx, and the phase column
        # needs unwrapping because the incrementing branch of the Fortran's
        # own unwrap (Kraken/bounce.f90:219) cannot fire. Staging repeats
        # this for the consumer's copy; the rewrite is idempotent.
        dedupe_reflection_file(brc_file)
        # The .irc's g column carries the deck's density scale, which is
        # relative to the water's; the consumer applies it on the
        # absolute scale, so it is rescaled once, here, where the file
        # is made (see _scale_irc_impedance).
        irc_file = work_dir / f'{base_name}.irc'
        if irc_file.exists() and irc_file.stat().st_size > 0:
            _scale_irc_impedance(irc_file, float(inputs.env.water_density))

        self._log(f"Reading output: {brc_file}")
        table = read_reflection_coefficient(str(brc_file))
        if table.angles.size == 0:
            # The deck asked for n_angles rows and the binary wrote none:
            # that is an outcome of the run, not a bad configuration, so
            # it is a ModelExecutionError carrying the .prt tail the
            # message points at.
            engine = inputs.settings.engine
            exc = ModelExecutionError(
                self.model_name, return_code=0, stdout=None,
                stderr=(
                    f"Bounce produced an empty reflection-coefficient "
                    f"table — {brc_file.name} has no angle rows, although "
                    f"the deck asked for {engine.n_angles} (RMax = "
                    f"{engine.rmax_m:g} m, from {engine.rmax_origin})."
                ),
            )
            self._attach_prt_tail(exc, work_dir, base_name)
            raise exc
        return table

    def _to_result(self, inputs, deck: Path,
                   raw: ReflectionCoefficient) -> ReflectionCoefficient:
        """Stage 5: the :class:`ReflectionCoefficient` — ``theta`` (grazing
        angles, degrees), ``R`` (magnitude) and ``phi`` (phase, radians) in
        the phase convention of ``inputs.settings.output`` — and, when the
        work directory outlives the run, the ``brc_file`` / ``irc_file`` paths
        a consumer reads through ``acoustic_type='file'`` / ``'precalc'``.
        The deck's window and RMax are the run settings'
        (``run_settings.engine``)."""
        settings = inputs.settings
        engine = settings.engine
        table = ReflectionCoefficient(
            angles=raw.angles,
            magnitude=raw.magnitude,
            phase=raw.phase,
            **self._result_kwargs(
                inputs.source,
                frequencies=float(settings.frequencies[0]),
                phase_reference=settings.output.phase_reference,
                n_points=raw.n_angles,
            ),
        )
        self._attach_output_paths(
            table, inputs.work_dir, deck.stem,
            primary_files=(
                ('brc_file', '.brc'),
                ('irc_file', '.irc'),
            ),
        )
        self._log("Simulation complete")
        return table
