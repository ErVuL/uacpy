"""
BOUNCE - Reflection Coefficient Computation Module

BOUNCE computes reflection coefficients for a stack of acoustic/elastic layers.
Part of the Acoustics Toolbox (OALIB).

Outputs:
- .BRC file: Bottom Reflection Coefficient (BotOpt 'F')
  -> Read by BELLHOP (``Bellhop/bellhop.f90:136`` loads the table, ``:688-693``
     applies it in the ``Reflect2D`` contained in that same file;
     ``Bellhop/ReflectMod.f90`` holds a near-identical ``Reflect2D`` that only
     bellhop3D links, per ``Bellhop/Makefile:4,8``), SCOOTER, KRAKENC
- .IRC file: Internal Reflection Coefficient (BotOpt 'P')
  -> Read by KRAKENC (``Kraken/BCImpedancecMod.f90:105``), SCOOTER
     (``Scooter/scooter.f90:357``) and KRAKEL; BELLHOP has no 'P' branch at
     all. Real KRAKEN parses it (``Kraken/BCImpedanceMod.f90:118``) but its
     mode search runs with ``ComplexFlag = .FALSE.``, which discards the
     table for a rigid boundary (:121-125) — uacpy therefore routes an
     ``.irc``/``.brc`` environment to krakenc.exe.

Note: SPARC does not support reflection coefficient files.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union

from uacpy.models.base import (
    PropagationModel, RunMode, ModelSpec, VALID_SOURCE_TYPES,
)
from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.results import Result
from uacpy.core.constants import (
    DEFAULT_C_MIN, DEFAULT_C_MAX_UNBOUNDED,
    parse_boundary_type,
)
from uacpy.core.exceptions import (
    ConfigurationError, ExecutableNotFoundError, ModelExecutionError,
    UnsupportedFeatureError,
)
from uacpy.io.refl_io import read_reflection_coefficient, dedupe_reflection_file
from uacpy.io.oalib_writer import write_bounce_input_file, writable_layers
from uacpy.io.units import m_to_km

# bounce.f90 zeroes kMin (drops the 1/cHigh term in NkTab) once cHigh > 1e6.
_KMIN_CUTOFF_CHIGH = 1.0e6

# BOUNCE always writes both tables, so both are cleared before a launch — but a
# tabulated seabed makes one of them an *input* too: write_bottom_section stages
# the user's table there and misc/RefCoef.f90:39 ('F' -> .brc) / :92 ('P' ->
# .irc) open it with STATUS='OLD' before ComputeReflectionCoefficient rewrites
# it. That one must survive the pre-launch sweep.
_STAGED_TABLE_SUFFIX = {'file': '.brc', 'precalc': '.irc'}
_BOUNCE_OUTPUTS = ('.brc', '.irc')

# Mesh density of each medium of the sediment stack: 20 points per wavelength,
# the same density misc/ReadEnvironmentMod.f90:103 uses for its own automatic
# mesh, floored so a thin layer still gets a usable mesh. The cap bounds the
# difference-equation grid bounce.f90:78 allocates (B1..B4/rho/cP/cS sized from
# the sum over media); a stack that needs more than that is refused rather than
# clipped, since a clipped count below Nneeded/2 is a deck the binary rejects
# (ReadEnvironmentMod.f90:110-112).
_MESH_POINTS_PER_WAVELENGTH = 20
_MIN_MESH_POINTS = 100
_MAX_MESH_POINTS = 20000


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

    Parameters
    ----------
    executable : Path, optional
        Path to ``bounce``. Auto-detected if ``None``.
    c_low, c_high : float, optional
        Phase-velocity bounds for tabulation (m/s). ``c_low`` must be
        strictly positive (BOUNCE rejects ``c_low <= 0``). Defaults
        ``DEFAULT_C_MIN`` / ``DEFAULT_C_MAX_UNBOUNDED``.
    rmax : float, optional
        Max range (m) for angular sampling. ``None`` (default) auto-
        derives from ``receiver.range_max`` at ``run()`` time, falling
        back to ``10000`` m when no receiver range is available.
        Ignored when ``n_angles`` is provided.
    n_angles : int, optional
        Explicit number of angular samples (``NkTab`` in
        ``bounce.f90``). When provided, uacpy back-derives ``rmax`` to
        hit ``~n_angles``.
    interp_ssp : str, optional
        Sample-connection scheme written into ``TopOpt(1)``. A BOUNCE deck
        carries only the seabed stack — its media are 2-point linear slabs —
        so this rarely changes the answer, and ``'quad'`` is rejected
        outright (no water column means no ``.ssp`` for it to read).
    use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
        Standard plumbing (see :class:`PropagationModel`).

    Notes
    -----
    Only emits ``RunMode.REFLECTION``. The result always carries the
    in-memory reflection coefficient as typed attributes
    (``.theta``, ``.R``, ``.phi``); the standalone Python user does not
    need the on-disk files.

    To **chain to another model** (Bellhop / Scooter / Kraken /
    Kraken reading ``acoustic_type='file'``), pin ``work_dir=`` so the
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
    - **Recommended workflow**: BOUNCE -> .brc -> SCOOTER (most reliable)
    - Both tables go through the standard AT reflection-coefficient path:
      ``.brc`` via ``acoustic_type='file'``, ``.irc`` via
      ``acoustic_type='precalc'``. Kraken routes either to krakenc.exe.

    Defaults auto-derived at ``run()`` time:

    - ``rmax=None`` → ``receiver.range_max`` (or 10 km if 0).
    - ``c_low`` / ``c_high`` constructor defaults
      (``DEFAULT_C_MIN`` / ``DEFAULT_C_MAX_UNBOUNDED``) tabulate the full
      0–90° grazing span; raise ``c_low`` or lower ``c_high`` to
      concentrate the samples on a narrower angular band.
    - TopOpt position 4 reads ``env.absorption``.

    With ``verbose='info'`` the resolved ``rmax`` is logged.

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
    >>> # Pin work_dir so the .brc/.irc files persist for the consumer.
    >>> bounce = Bounce(c_low=1400, c_high=10000, rmax=10000,
    ...                 work_dir='./bounce_out')
    >>> result = bounce.run(env, source, receiver)
    >>>
    >>> # Output files can be used by different models:
    >>> # - .brc file → Use with BELLHOP, SCOOTER, KRAKENC (experimental)
    >>> # - .irc file → Use with KRAKEN
    >>>
    >>> # Example: Use .brc with SCOOTER
    >>> from uacpy.models import Scooter
    >>> bottom_with_rc = BoundaryProperties(
    ...     acoustic_type='file',
    ...     reflection_file=result.metadata['brc_file']
    ... )
    >>> env_with_rc = Environment(name="test", bathymetry=100, bottom=bottom_with_rc)
    >>> scooter = Scooter()
    >>> tl = scooter.compute_tl(env_with_rc, source, receiver)

    References
    ----------
    - Porter, M.B., "The KRAKEN Normal Mode Program", SACLANT Undersea Research
      Centre Memorandum SM-245, 1991
    - Acoustics Toolbox: http://oalib.hlsresearch.com/
    """

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
    )
    source = 'acoustics_toolbox'

    def __init__(
        self,
        executable: Optional[Path] = None,
        c_low: float = DEFAULT_C_MIN,
        c_high: float = DEFAULT_C_MAX_UNBOUNDED,
        rmax: Optional[float] = None,
        n_angles: Optional[int] = None,
        interp_ssp: Optional[str] = None,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        cleanup: Optional[bool] = None,
        timeout: float = 600.0,
        collapse: Optional[Dict[str, str]] = None,
    ):
        """
        Parameters
        ----------
        executable : Path, optional
            Path to bounce executable. Auto-detected if None.
        c_low : float, optional
            Minimum phase velocity (m/s) for tabulation. Default: 1400.
            Must be strictly positive (BOUNCE rejects ``c_low <= 0`` — the
            angular grid is derived from ``kx = omega/c``).
        c_high : float, optional
            Maximum phase velocity (m/s) for tabulation. Default: 1e9, which
            trips ``bounce.f90:47``'s ``IF ( cHigh > 1.0E6 ) kMin = 0.0`` and
            so tabulates the full 0–90° grazing span. A finite ``c_high``
            stops the table at ``acos(c0 / c_high)`` and every consumer
            silently returns ``R = 0, phi = 0`` above the last tabulated
            angle (``misc/RefCoef.f90:144-149``, both of whose warning WRITEs
            are commented out). Must be strictly greater than c_low.
        rmax : float, optional
            Maximum range (m) for angular sampling. ``None`` (default)
            auto-derives from ``receiver.range_max`` at ``run()`` time,
            falling back to 10000 m when no receiver range is available.
            Ignored when ``n_angles`` is provided. (Internally converted to
            km because BOUNCE's input format is in km.)
        n_angles : int, optional
            Explicit override for the number of angular samples (``NkTab``
            in AT's bounce). If None (default), bounce computes NkTab
            internally from ``rmax``. When provided, uacpy sets ``rmax``
            such that bounce's internal formula yields approximately
            ``n_angles`` samples.
        """
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
            cleanup=cleanup, timeout=timeout, collapse=collapse,
        )
        # BOUNCE meshes only the seabed stack — write_bounce_input_file emits
        # no water column and no .ssp — so the range-dependent 'quad' scheme
        # has no file to read and ERROUTs at NMedia.
        if interp_ssp is not None and str(interp_ssp).lower() == 'quad':
            raise ConfigurationError(
                "Bounce(interp_ssp='quad') is not usable: BOUNCE reflection "
                "decks carry no water column, so no .ssp file is written for "
                "the quad scheme to read and the binary stops at NMedia.",
                remediation="Use 'linear' (default), 'n2linear', 'pchip' or "
                            "'cubic'.",
            )
        self.interp_ssp = interp_ssp

        self.c_low = c_low
        self.c_high = c_high
        self.rmax = rmax
        self.n_angles = n_angles

        # Validate phase velocity bounds up front
        self._validate_phase_speed_bounds()

        # Run modes, capability flags and collapse defaults now come from the
        # class-level ``spec`` (applied by PropagationModel.__init__).
        #
        # Keep the user's ``executable`` arg verbatim (``None`` when
        # auto-detected) so ``model.copy()`` re-resolves the binary instead of
        # re-pinning the already-resolved absolute path. The resolved path
        # lives in ``self._exe``.
        self.executable = Path(executable) if executable is not None else None
        if self.executable is None:
            self._exe = self._find_executable_in_paths(
                'bounce',
                bin_subdirs='oalib',
                dev_subdir='Acoustics-Toolbox/Kraken',
            )
        else:
            self._exe = self.executable

        if not self._exe.exists():
            raise ExecutableNotFoundError('Bounce', str(self._exe))

    def _validate_geometry(self, env, source, receiver, run_mode=None) -> None:
        """No-op: BOUNCE reads no source or receiver geometry.

        Its Fortran driver stops after ``TopOpt``, the SSP, ``BotOpt``,
        ``cLow``/``cHigh`` and ``RMax`` (``bounce.f90``) — it never calls
        ``ReadSzRz``. The plane-wave reflection coefficient is independent of
        source position, so the field models' depth and range-coverage checks
        would reject and warn about geometry that never reaches the binary.
        ``receiver`` is read only to auto-derive ``rmax``.
        """

    def _validate_phase_speed_bounds(self) -> None:
        """Phase-velocity bounds invariant, enforced at construction AND at
        ``run()`` (a single source of truth for both call sites)."""
        if self.c_low <= 0:
            raise ConfigurationError(
                f"Bounce requires c_low > 0 strictly (got {self.c_low}). "
                "c_low is the smallest phase velocity on the tabulated grid; "
                "0 would give an infinite wavenumber."
            )
        if self.c_high <= self.c_low:
            raise ConfigurationError(
                f"c_high ({self.c_high}) must be strictly greater than "
                f"c_low ({self.c_low})."
            )
        if self.rmax is not None and float(self.rmax) <= 0:
            raise ConfigurationError(
                f"Bounce requires rmax > 0 (got {self.rmax}). RMax sets the "
                f"angular sampling density — bounce.f90:49 makes the number of "
                f"tabulated angles proportional to it, and "
                f"misc/ReadEnvironmentMod.f90:140 stops outright on a negative "
                f"value."
            )

    def _n_ktab(self, rmax_m: float, frequency: float) -> int:
        """Angles BOUNCE will tabulate for this deck.

        Reproduces ``bounce.f90:45-49`` on the RMax string the writer is about
        to emit: ``kMin = omega / cHigh`` (zeroed at :47 once ``cHigh > 1e6``),
        ``kMax = omega / cLow`` and
        ``NkTab = INT( 1000 * RMax_km * ( kMax - kMin ) / 2 pi )``.

        The ``.6f`` round-trip is the deck's own precision: ``RMax`` is written
        in km at six decimals (``oalib_writer.write_phase_speed_and_rmax``), so
        the count has to come off the rounded value the binary will read back,
        not off the exact metres.
        """
        rmax_km = float(f"{m_to_km(rmax_m):.6f}")
        omega = 2.0 * np.pi * float(frequency)
        k_min = 0.0 if self.c_high > _KMIN_CUTOFF_CHIGH else omega / self.c_high
        k_max = omega / self.c_low
        return int(1000.0 * rmax_km * (k_max - k_min) / (2.0 * np.pi))

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional[RunMode] = None,
        *,
        frequencies: Optional[np.ndarray] = None,
        source_waveform: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
        output_duration: Optional[float] = None,
    ) -> Result:
        """
        Run BOUNCE reflection coefficient computation.

        ``.brc`` / ``.irc`` files are written into the model's
        ``work_dir`` (constructor kwarg). Pin the location with
        ``Bounce(work_dir='./bounce_out')`` — a pinned work dir defaults
        ``cleanup=False`` so the files outlive the call and can be
        consumed by Bellhop / Scooter / Kraken; an unpinned
        temp work dir is wiped after ``run()``.

        Parameters
        ----------
        env : Environment
            Ocean environment (bottom properties define the stack).
        source : Source
            Source definition. Only ``frequencies[0]`` is read; BOUNCE
            consumes no source geometry.
        receiver : Receiver
            Read only for ``range_max``, which auto-derives ``rmax`` when the
            constructor left it ``None``.
        run_mode : RunMode, optional
            Must be ``RunMode.REFLECTION`` (the only mode BOUNCE emits)
            or ``None`` (defaults to REFLECTION). Other values raise
            :class:`UnsupportedFeatureError`.

        Returns
        -------
        ReflectionCoefficient
            Typed result with ``theta`` (grazing angles in degrees),
            ``R`` (reflection coefficient magnitude), and ``phi`` (phase
            in radians). The persisted ``.brc`` / ``.irc`` paths live on
            ``result.metadata['brc_file']`` and
            ``result.metadata['irc_file']``.

        Notes
        -----
        - For full 90-degree coverage, set ``c_low=1400, c_high=1e9``
          (``c_high=1e9`` triggers ``kmin=0`` in the Fortran; see
          ``bounce.f90``).
        - Larger ``rmax`` gives finer angular resolution.
        - ``.brc`` is consumed by Bellhop / Scooter / Kraken via
          ``BoundaryProperties(acoustic_type='file', reflection_file=…)``.
          ``.irc`` is consumed by Kraken (true normal modes).
        """
        self._reject_unsupported_run_kwargs(
            frequencies=frequencies, source_waveform=source_waveform,
            sample_rate=sample_rate, output_duration=output_duration)
        run_mode = self._resolve_run_mode(run_mode)
        self._validate_phase_speed_bounds()

        # BOUNCE writes one frequency into the .env and returns one table, so a
        # multi-frequency Source would be silently truncated. RunMode.REFLECTION
        # itself stays out of ``_SINGLE_FREQUENCY_MODES`` — OASR does sweep.
        src_freqs = np.atleast_1d(source.frequencies)
        if len(src_freqs) > 1:
            raise ConfigurationError(
                f"Bounce tabulates the reflection coefficient at one "
                f"frequency; got {len(src_freqs)}: {list(src_freqs)}. Loop "
                f"over single-frequency Sources, or use OASR for a "
                f"multi-frequency reflection sweep."
            )

        # Per-call rmax. ``n_angles`` (below) overrides via the inverse of
        # bounce.f90:49  NkTab = INT(1000*RMax_km*(kMax-kMin)/(2π)).
        rmax_origin = 'Bounce(rmax=…)'
        if self.rmax is not None:
            rmax = float(self.rmax)
        else:
            if receiver is None:
                raise TypeError(
                    "Bounce.run(receiver=None): a Receiver is required to "
                    "auto-derive rmax (the range the reflection table is "
                    "propagated to). Pass a Receiver, or pin the range with "
                    "Bounce(rmax=...)."
                )
            recv_rmax = float(receiver.range_max)
            rmax = recv_rmax if recv_rmax > 0 else 10000.0
            rmax_origin = ('receiver.range_max' if recv_rmax > 0
                           else 'the 10 km fallback (no receiver range)')
        if self.n_angles is not None:
            if self.n_angles < 2:
                raise ConfigurationError(
                    f"n_angles must be >= 2 (got {self.n_angles}). "
                    f"bounce.f90:172 spaces the wavenumber grid as "
                    f"Deltak = (kMax - kMin) / (NkTab - 1), so a single "
                    f"tabulated angle divides by zero and the binary spins "
                    f"until the model timeout."
                )
            f_hz = float(np.atleast_1d(source.frequencies)[0])
            omega = 2.0 * np.pi * f_hz
            inv_c_diff = 1.0 / float(self.c_low)
            if self.c_high is not None and self.c_high <= _KMIN_CUTOFF_CHIGH:
                inv_c_diff -= 1.0 / float(self.c_high)
            if omega * inv_c_diff <= 0:
                raise ConfigurationError(
                    f"Cannot derive rmax from n_angles={self.n_angles}: "
                    f"omega·(1/cLow - 1/cHigh) is non-positive "
                    f"(omega={omega:.3g}, 1/cLow-1/cHigh={inv_c_diff:.3g})."
                )
            # NkTab = INT(1000 * RMax_km * (kMax - kMin) / 2π) and
            # 1000 * RMax_km IS RMax in metres, so the km conversion cancels
            # out of the inversion and this lands directly in metres.
            rmax = (
                float(self.n_angles) * 2.0 * np.pi / (omega * inv_c_diff)
            )
            rmax_origin = f'n_angles={self.n_angles}'

        # Logged after the n_angles branch, which overwrites rmax.
        self._log(f"rmax = {rmax:.1f} m (from {rmax_origin})")

        frequency = float(src_freqs[0])
        n_ktab = self._n_ktab(rmax, frequency)
        if n_ktab < 2:
            consequence = (
                "BOUNCE would write an empty reflection-coefficient table"
                if n_ktab == 0 else
                "bounce.f90:172 spaces the grid as "
                "Deltak = (kMax - kMin) / (NkTab - 1), so a single angle "
                "divides by zero and the binary spins until the model timeout"
            )
            raise ConfigurationError(
                f"This deck asks for {n_ktab} tabulated angle(s): "
                f"bounce.f90:49 derives NkTab = INT(1000 * RMax_km * "
                f"(kMax - kMin) / 2 pi) from RMax = {rmax:g} m at "
                f"{frequency:.4g} Hz with c_low={self.c_low:g} and "
                f"c_high={self.c_high:g} m/s. {consequence}.",
                remediation=(
                    f"rmax came from {rmax_origin} — raise it (or raise "
                    f"n_angles), or widen the phase-speed window so more "
                    f"wavenumbers fall inside it."
                ),
            )
        self._log(f"NkTab = {n_ktab} tabulated angles", level='debug')

        env = self._project_environment(env)
        self.validate_inputs(env, source, receiver, run_mode=run_mode)

        seabed_type = env.bottom.halfspace_at(range=0.0).acoustic_type
        if seabed_type == 'precalc':
            raise UnsupportedFeatureError(
                self.model_name,
                "a 'precalc' (.irc) seabed — misc/RefCoef.f90:103-104 leaves "
                "xTab/fTab/gTab/iTab allocated for the table it just read, so "
                "bounce.f90:52 cannot allocate them for the table it is about "
                "to write and stops with 'Too many points in reflection "
                "coefficient'",
                alternatives=[
                    "acoustic_type='file' with the equivalent .brc table, "
                    "which BOUNCE reads through its own RBot array",
                    "feed the .irc straight to Kraken or Scooter instead of "
                    "re-running BOUNCE on it",
                ],
            )

        fm = self._setup_file_manager()

        try:
            base_name = 'bounce_run'
            input_file = fm.get_path(f'{base_name}.env')

            self._log(f"Writing input file: {input_file}")
            self._write_bounce_input(
                filepath=input_file,
                env=env,
                source=source,
                c_low=self.c_low,
                c_high=self.c_high,
                rmax=rmax,
            )

            self._log("Running...")
            self._execute(input_file, fm.work_dir,
                          staged_input=_STAGED_TABLE_SUFFIX.get(seabed_type))

            brc_file = fm.get_path(f'{base_name}.brc')

            if not brc_file.exists():
                exc = ModelExecutionError(
                    self.model_name, return_code=0, stdout=None,
                    stderr=(
                        f"BOUNCE did not produce {brc_file}; "
                        f"check {fm.work_dir}/{base_name}.prt for diagnostics."
                    ),
                )
                self._attach_prt_tail(exc, fm.work_dir, base_name)
                raise exc

            # Normalise the raw BOUNCE table before it is read back into the
            # ReflectionCoefficient result: bellhopcuda's strict monotonicity
            # check rejects the duplicate near-zero angles bounce.f90 emits when
            # many high-c samples round to the same kx, and the phase column
            # needs unwrapping because the incrementing branch of the Fortran's
            # own unwrap (Kraken/bounce.f90:219) cannot fire. Staging repeats
            # this for the consumer's copy; the rewrite is idempotent. The .irc
            # is left byte-for-byte as BOUNCE wrote it: a different,
            # fixed-format layout that only the Fortran BotOpt='P' path reads.
            dedupe_reflection_file(brc_file)

            self._log(f"Reading output: {brc_file}")
            result = read_reflection_coefficient(str(brc_file), boundary='bottom')

            theta_out = np.atleast_1d(np.asarray(result.get('theta', []), dtype=float))
            if theta_out.size == 0:
                raise ConfigurationError(
                    f"Bounce produced an empty reflection-coefficient table — "
                    f"{brc_file.name} has no angle rows, although the deck asked "
                    f"for {n_ktab} (RMax = {rmax:g} m, from {rmax_origin}). "
                    f"Inspect {brc_file.with_suffix('.prt').name} for BOUNCE "
                    f"diagnostics."
                )

            from uacpy.core.results import ReflectionCoefficient

            field = ReflectionCoefficient(
                theta=result.get('theta', np.array([])),
                R=result.get('R', np.array([])),
                phi=result.get('phi', np.array([])),
                **self._result_kwargs(
                    source,
                    frequencies=frequency,
                    n_points=result.get('n_pts', 0),
                    c_low=self.c_low,
                    c_high=self.c_high,
                    rmax=rmax,
                    full_result=result,
                ),
            )
            self._attach_output_paths(
                field, fm.work_dir, base_name,
                primary_files=(
                    ('brc_file', '.brc'),
                    ('irc_file', '.irc'),
                ),
            )

            self._log("Simulation complete")
            return field

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    def _resolve_n_mesh(self, env: Environment, frequency: float) -> list:
        """Mesh-point count for each medium of the sediment stack.

        ``write_bounce_input_file`` omits the water column, so the media are
        exactly the writable sediment layers (a bare half-space seabed carries
        none and returns an empty list).

        ``misc/ReadEnvironmentMod.f90:101-112`` sizes every medium
        independently: ``c = alphaR``, then ``IF ( betaR > 0.0 ) c = betaR``,
        ``deltaz = c / freq / 20``, ``Nneeded = INT( thickness / deltaz )``,
        and it aborts with *Mesh is too coarse* when the deck asks for fewer
        than ``Nneeded / 2``. The meshing speed is therefore the medium's
        **shear** speed wherever it has one — an ordinary sand (cs ~ 200 m/s
        against cp ~ 1700 m/s) needs an order of magnitude more points than its
        compressional wavelength suggests. ``doc/bounce.htm`` states the same
        rule.
        """
        counts = []
        for layer in writable_layers(env.bottom.at(range=0.0)):
            thickness = float(layer.thickness)
            shear = float(getattr(layer, 'shear_speed', 0.0) or 0.0)
            speed = shear if shear > 0.0 else float(layer.sound_speed)
            needed = _MESH_POINTS_PER_WAVELENGTH * thickness * frequency / speed
            if needed > _MAX_MESH_POINTS:
                raise ConfigurationError(
                    f"Bounce: the {thickness:g} m sediment layer needs "
                    f"{int(np.ceil(needed))} mesh points at {frequency:.4g} Hz "
                    f"({_MESH_POINTS_PER_WAVELENGTH} per wavelength of its "
                    f"{speed:.1f} m/s "
                    f"{'shear' if shear > 0.0 else 'compressional'} speed), "
                    f"above the {_MAX_MESH_POINTS}-point ceiling on a single "
                    f"BOUNCE medium.",
                    remediation="Lower the frequency, split the layer into "
                                "thinner ones, or drop the shear speed if the "
                                "layer is meant to be fluid.",
                )
            counts.append(max(_MIN_MESH_POINTS, int(np.ceil(needed))))
        self._log(f"mesh points per medium = {counts}", level='debug')
        return counts

    def _write_bounce_input(
        self,
        filepath: Path,
        env: Environment,
        source: Source,
        c_low: float,
        c_high: float,
        rmax: float,
    ):
        """
        Write BOUNCE input file using ATEnvWriter

        BOUNCE uses ENV format similar to KRAKEN with additional sections:
        - c_low, c_high (phase velocity bounds)
        - RMax in km (for angular sampling) — converted from ``rmax``

        BOUNCE does NOT call ``ReadSzRz``; its Fortran driver reads only
        TopOpt, SSP, BotOpt, cLow/cHigh, RMax. We therefore omit the
        source/receiver depth blocks.
        """
        from uacpy.io.oalib_writer import resolve_ssp_topopt
        ssp_topopt = resolve_ssp_topopt(env, self.interp_ssp)
        bottom_type = parse_boundary_type(env.bottom.halfspace_at(range=0.0).acoustic_type)

        frequency = float(source.frequencies[0])
        n_mesh = self._resolve_n_mesh(env, frequency)

        write_bounce_input_file(
            filepath, env, source,
            ssp_topopt=ssp_topopt,
            bottom_type=bottom_type,
            n_mesh=n_mesh,
            c_low=c_low,
            c_high=c_high,
            rmax=rmax,
            verbose=self.verbose,
        )

    def _execute(self, input_file: Path, work_dir: Path,
                 staged_input: Optional[str] = None):
        """Execute BOUNCE binary via the shared binary-launch helper.

        ``staged_input`` is the suffix ``write_bottom_section`` copied next to
        the deck as *input* for this run (see :data:`_STAGED_TABLE_SUFFIX`); it
        is kept out of the stale-output sweep so the binary can still read it.
        """
        base_name = input_file.stem
        self._run_and_attach_prt(
            [str(self._exe), base_name], work_dir, base_name,
            stale_outputs=tuple(s for s in _BOUNCE_OUTPUTS
                                if s != staged_input))
