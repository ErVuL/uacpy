"""
BOUNCE - Reflection Coefficient Computation Module

BOUNCE computes reflection coefficients for a stack of acoustic/elastic layers.
Part of the Acoustics Toolbox (OALIB).

Outputs:
- .BRC file: Bottom Reflection Coefficient
  -> Used by: BELLHOP, SCOOTER, KRAKENC
- .IRC file: Internal Reflection Coefficient
  -> Used by: KRAKEN (NOT KRAKENC - use .BRC with KRAKENC)

Note: SPARC does not support reflection coefficient files.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union

from uacpy.models.base import PropagationModel, RunMode
from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.results import Result
from uacpy.core.constants import (
    DEFAULT_C_MIN, DEFAULT_C_MAX,
    parse_boundary_type,
)
from uacpy.core.exceptions import (
    ConfigurationError, ExecutableNotFoundError, ModelExecutionError,
)
from uacpy.io.refl_io import read_reflection_coefficient
from uacpy.io.oalib_writer import (
    write_absorption_block, write_bottom_section,
    write_header, write_layer_sections, write_ssp_section,
)


class Bounce(PropagationModel):
    """
    BOUNCE - Reflection Coefficient Model (Acoustics Toolbox)

    Computes plane wave reflection coefficients for a stack of acoustic/elastic
    layers. The reflection coefficient is written to both .BRC (Bottom Reflection
    Coefficient) and .IRC (Internal Reflection Coefficient) files.

    Model Support:
    - .BRC files: BELLHOP, SCOOTER, KRAKENC
    - .IRC files: KRAKEN (uses internal reflection coefficient)
    - SPARC: does not support reflection files

    Parameters
    ----------
    executable : Path, optional
        Path to ``bounce``. Auto-detected if ``None``.
    c_low, c_high : float, optional
        Phase-velocity bounds for tabulation (m/s). ``c_low`` must be
        strictly positive (BOUNCE rejects ``c_low <= 0``); ``c_high=1e9``
        is a valid recommendation for ~full 90° coverage. Defaults
        ``DEFAULT_C_MIN`` / ``DEFAULT_C_MAX``.
    rmax : float, optional
        Max range (m) for angular sampling. ``None`` (default) auto-
        derives from ``receiver.range_max`` at ``run()`` time, falling
        back to ``10000`` m when no receiver range is available.
        Ignored when ``n_angles`` is provided.
    n_angles : int, optional
        Explicit number of angular samples (``NkTab`` in
        ``bounce.f90``). When provided, uacpy back-derives ``rmax`` to
        hit ``~n_angles``.
    use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
        Standard plumbing (see :class:`PropagationModel`).

    Notes
    -----
    Only emits ``RunMode.REFLECTION``. The result always carries the
    in-memory reflection coefficient as typed attributes
    (``.theta``, ``.R``, ``.phi``); the standalone Python user does not
    need the on-disk files.

    To **chain to another model** (Bellhop / Scooter / Kraken /
    KrakenC reading ``acoustic_type='file'``), pin ``work_dir=`` so the
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
    Per-model: ``'bottom': 'median'``,
    ``'rd_layered_layers': 'preserve'`` (BOUNCE consumes
    ``LayeredBottom`` natively).

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

    Notes
    -----
    - BOUNCE uses the same environmental file format as KRAKEN
    - The reflection coefficient depends on impedance contrast
    - Supports acoustic, elastic, and poro-elastic layers
    - Tabulated reflection coefficients cover angles from phase velocities [c_low, c_high]
    - **Recommended workflow**: BOUNCE -> .brc -> SCOOTER (most reliable)
    - KRAKENC consumes .brc files directly via the standard AT reflection
      coefficient path
    - For KRAKEN, use .irc files (internal reflection coefficient)

    Defaults auto-derived at ``run()`` time:

    - ``rmax=None`` → ``receiver.range_max`` (or 10 km if 0).
    - ``c_low`` / ``c_high`` constructor defaults
      (``DEFAULT_C_MIN`` / ``DEFAULT_C_MAX``) bracket trapped+leaky modes
      for the typical seawater range; override for narrowband studies.
    - TopOpt position 4 reads ``env.absorption``.

    With ``verbose='info'`` the resolved ``rmax`` is logged.

    References
    ----------
    - Porter, M.B., "The KRAKEN Normal Mode Program", SACLANT Undersea Research
      Centre Memorandum SM-245, 1991
    - Acoustics Toolbox: http://oalib.hlsresearch.com/
    """

    def __init__(
        self,
        executable: Optional[Path] = None,
        c_low: float = DEFAULT_C_MIN,
        c_high: float = DEFAULT_C_MAX,
        rmax: Optional[float] = None,
        n_angles: Optional[int] = None,
        interp_ssp: Optional[str] = None,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        **kwargs,
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
            Maximum phase velocity (m/s) for tabulation. Default: 10000.
            A value of ``1e9`` is a valid recommendation for full 90-deg
            coverage (kraken doc: "c_high large => kmin ~ 0 => grazing
            angles near 0 included"). Must be strictly greater than c_low.
        rmax : float, optional
            Maximum range (m) for angular sampling. Default: 10000. Ignored
            when ``n_angles`` is provided. (Internally converted to km
            because BOUNCE's input format is in km.)
        n_angles : int, optional
            Explicit override for the number of angular samples (``NkTab``
            in AT's bounce). If None (default), bounce computes NkTab
            internally from ``rmax``. When provided, uacpy sets ``rmax``
            such that bounce's internal formula yields approximately
            ``n_angles`` samples.
        """
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir, **kwargs,
        )
        self.interp_ssp = interp_ssp

        # BOUNCE produces ONE BRC consumed across the whole range axis;
        # the median sample is the most representative single profile.
        # ``rd_layered_layers='preserve'`` keeps the layer stack (BOUNCE
        # handles LayeredBottom natively).
        self._set_collapse_defaults({
            'bottom': 'median',
            'rd_layered_layers': 'preserve',
        })

        self.c_low = c_low
        self.c_high = c_high
        self.rmax = rmax
        self.n_angles = n_angles

        # Validate phase velocity bounds up front
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

        # BOUNCE computes plane-wave reflection coefficients, not TL.
        self._supported_modes = [RunMode.REFLECTION]
        self._supports_altimetry = False
        self._supports_range_dependent_bathymetry = False
        self._supports_range_dependent_ssp = False
        self._supports_range_dependent_bottom = False
        self._supports_layered_bottom = True
        self._supports_range_dependent_layered_bottom = False
        self._supports_elastic_media = True
        self._supports_multi_source_depth = False
        if executable is None:
            self.executable = self._find_executable_in_paths(
                'bounce',
                bin_subdirs='oalib',
                dev_subdir='Acoustics-Toolbox/Kraken',
            )
        else:
            self.executable = Path(executable)

        if not self.executable.exists():
            raise ExecutableNotFoundError('Bounce', str(self.executable))

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional[RunMode] = None,
        **kwargs
    ) -> Result:
        """
        Run BOUNCE reflection coefficient computation.

        ``.brc`` / ``.irc`` files are written into the model's
        ``work_dir`` (constructor kwarg). Bounce defaults
        ``cleanup=False`` so they outlive the call and can be consumed
        by Bellhop / Scooter / Kraken / KrakenC. Pass
        ``Bounce(work_dir='./bounce_out')`` to pin the location.

        Parameters
        ----------
        env : Environment
            Ocean environment (bottom properties define the stack).
        source : Source
            Source definition (frequency is used).
        receiver : Receiver
            Receiver definition (not used by BOUNCE, but required for API).
        run_mode : RunMode, optional
            Must be ``RunMode.REFLECTION`` (the only mode BOUNCE emits)
            or ``None`` (defaults to REFLECTION). Other values raise
            :class:`UnsupportedFeatureError`.
        **kwargs
            Additional parameters passed to the ENV file writer.

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
        - ``.brc`` is consumed by Bellhop / Scooter / KrakenC via
          ``BoundaryProperties(acoustic_type='file', reflection_file=…)``.
          ``.irc`` is consumed by Kraken (true normal modes).
        """
        run_mode = self._resolve_run_mode(run_mode)

        if self.c_low <= 0:
            raise ConfigurationError(
                f"Bounce requires c_low > 0 strictly (got {self.c_low})."
            )
        if self.c_high <= self.c_low:
            raise ConfigurationError(
                f"c_high ({self.c_high}) must be strictly greater than "
                f"c_low ({self.c_low})."
            )

        # Per-call rmax. ``n_angles`` (below) overrides via the inverse of
        # bounce.f90:49  NkTab = INT(1000*RMax_km*(kMax-kMin)/(2π)).
        if self.rmax is not None:
            rmax = float(self.rmax)
        else:
            recv_rmax = float(receiver.range_max) if receiver is not None else 0.0
            if recv_rmax > 0:
                rmax = recv_rmax
                self._log(
                    f"rmax auto-derived from receiver.range_max = "
                    f"{rmax:.1f} m"
                )
            else:
                rmax = 10000.0
                self._log(
                    "rmax auto-derived = 10000.0 m (no receiver range available)"
                )
        if self.n_angles is not None:
            if self.n_angles <= 0:
                raise ConfigurationError(
                    f"n_angles must be > 0 (got {self.n_angles})."
                )
            f_hz = float(np.atleast_1d(source.frequencies)[0])
            omega = 2.0 * np.pi * f_hz
            inv_c_diff = 1.0 / float(self.c_low)
            if self.c_high is not None and self.c_high < 1e8:
                inv_c_diff -= 1.0 / float(self.c_high)
            if omega * inv_c_diff <= 0:
                raise ConfigurationError(
                    f"Cannot derive rmax from n_angles={self.n_angles}: "
                    f"omega·(1/cLow - 1/cHigh) is non-positive "
                    f"(omega={omega:.3g}, 1/cLow-1/cHigh={inv_c_diff:.3g})."
                )
            rmax = (
                float(self.n_angles) * 2.0 * np.pi / (omega * inv_c_diff)
            )

        env = self._project_environment(env)
        self.validate_inputs(env, source, receiver, run_mode=run_mode)

        fm = self._setup_file_manager()

        try:
            base_name = 'bounce_run'
            input_file = fm.get_path(f'{base_name}.env')

            self._log(f"Writing input file: {input_file}")
            self._write_bounce_input(
                filepath=input_file,
                env=env,
                source=source,
                receiver=receiver,
                c_low=self.c_low,
                c_high=self.c_high,
                rmax=rmax,
                **kwargs
            )

            self._log("Running...")
            self._execute(input_file, fm.work_dir)

            brc_file = fm.get_path(f'{base_name}.brc')
            irc_file = fm.get_path(f'{base_name}.irc')

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

            # bellhopcuda's strict monotonicity check on .brc/.irc rejects
            # the duplicate near-zero angles bounce.f90 emits when many
            # high-c samples round to the same kx — rewrite both files
            # with a strictly-increasing angle axis.
            self._dedupe_reflection_file(brc_file)
            if irc_file.exists():
                self._dedupe_reflection_file(irc_file)

            self._log(f"Reading output: {brc_file}")
            result = read_reflection_coefficient(str(brc_file), boundary='bottom')

            from uacpy.core.results import ReflectionCoefficient
            frequency = source.frequencies[0] if hasattr(source.frequencies, '__len__') else source.frequencies

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

    def _write_bounce_input(
        self,
        filepath: Path,
        env: Environment,
        source: Source,
        receiver: Receiver,
        c_low: float,
        c_high: float,
        rmax: float,
        **kwargs
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
        surface_type = parse_boundary_type(env.surface.acoustic_type)
        bottom_type = parse_boundary_type(env.halfspace_at_range(0.0).acoustic_type)

        frequency = source.frequencies[0] if hasattr(source.frequencies, '__len__') else source.frequencies
        c_water = float(env.ssp.to_pairs()[0, 1])
        wavelength = c_water / frequency
        n_mesh = max(100, int(20 * env.depth / wavelength))

        with open(filepath, 'w') as f:
            write_header(
                f, env, source,
                ssp_topopt=ssp_topopt,
                surface_type=surface_type,
            )
            write_absorption_block(f, env)

            write_ssp_section(
                f, env, env.depth,
                n_mesh=n_mesh,
                roughness=0.0
            )

            # Layered sediments (no-op when env.bottom is a plain halfspace).
            write_layer_sections(f, env, env.depth)

            write_bottom_section(
                f, env,
                bottom_type=bottom_type,
                filepath=filepath,
                verbose=self.verbose
            )

            # BOUNCE-SPECIFIC SECTIONS

            # Phase velocity bounds (define angular coverage)
            f.write(f"{c_low:.2f} {c_high:.2f}\n")

            # Maximum range in km (for angular sampling resolution)
            f.write(f"{rmax / 1000.0:.2f}\n")

            # BOUNCE does NOT read source/receiver depths — do NOT emit
            # them. AT's bounce.f90 stops after RMax.

    def _execute(self, input_file: Path, work_dir: Path):
        """Execute BOUNCE binary via base-class subprocess helper."""
        base_name = input_file.stem
        try:
            result = self._run_subprocess(
                [str(self.executable), base_name],
                cwd=work_dir,
            )
        except ModelExecutionError as exc:
            self._attach_prt_tail(exc, work_dir, base_name)
            raise
        if self.verbose and result.stdout:
            self._log(f"Bounce output:\n{result.stdout}", level='debug')

    @staticmethod
    def _dedupe_reflection_file(filepath: Path) -> None:
        """Rewrite a .brc/.irc file with a strictly-increasing angle axis.

        BOUNCE's Fortran driver tabulates reflection coefficients by
        sweeping phase velocity (kx = omega/c), which — for the c_low/c_high
        defaults — produces many samples that round to the same grazing
        angle (hundreds of duplicate 0-degree rows are typical). Bellhop
        tolerates non-decreasing angles but bellhopcuda enforces strict
        monotonicity in ``bhc::setup()`` and aborts with "Bottom
        reflection coefficients must be monotonically increasing".

        This helper loads the file, keeps only rows whose angle strictly
        exceeds the previous kept row, and rewrites it in the original
        3-column BOUNCE format (angle_deg, |R|, phase_deg). The IRC file
        has the same layout so the same routine works for both.
        """
        filepath = Path(filepath)
        with open(filepath, 'r') as fh:
            lines = fh.readlines()

        if not lines:
            return

        kept_rows = []  # list of (angle, mag, phase_deg) as strings
        last_angle = -np.inf
        for line in lines[1:]:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 3:
                continue
            try:
                angle = float(parts[0])
            except ValueError:
                continue
            if angle > last_angle:
                kept_rows.append((parts[0], parts[1], parts[2]))
                last_angle = angle

        # If nothing survived dedup (degenerate case), leave the file
        # alone — downstream reader will surface the real error.
        if not kept_rows:
            return

        # Preserve the original numeric-format header style by simply
        # rewriting the count with the same trailing newline.
        with open(filepath, 'w') as fh:
            # BOUNCE pads the count with leading whitespace; match that
            # so any downstream tool expecting free-format reads happily.
            fh.write(f"{len(kept_rows):12d}\n")
            for a, r, p in kept_rows:
                fh.write(f"   {a}        {r}        {p}\n")
