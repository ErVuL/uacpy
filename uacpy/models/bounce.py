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

import shutil
import tempfile
import warnings
import weakref
import numpy as np
from pathlib import Path
from typing import Optional

from uacpy.models.base import PropagationModel, RunMode, _UNSET, _resolve_overrides
from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.results import Result, ReflectionCoefficient
from uacpy.core.constants import (
    DEFAULT_SOUND_SPEED, DEFAULT_C_MIN, DEFAULT_C_MAX,
    AttenuationUnits, VolumeAttenuation,
    parse_ssp_type, parse_boundary_type,
)
from uacpy.core.exceptions import UnsupportedFeatureError, ConfigurationError, ExecutableNotFoundError, ModelExecutionError
from uacpy.io.refl_io import read_reflection_coefficient
from uacpy.io.oalib_writer import write_bio_layers, write_bottom_section, write_fg_params, write_header, write_ssp_section


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
        Path to bounce executable. If None, searches standard locations.
    use_tmpfs : bool, optional
        Use tmpfs for I/O operations. Default False.
    verbose : bool, optional
        Print detailed execution information. Default False.
    work_dir : Path, optional
        Working directory for I/O files. Default uses temp directory.

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
    >>> # Compute reflection coefficients
    >>> bounce = Bounce()
    >>> result = bounce.run(env, source, receiver,
    ...                     c_low=1400, c_high=10000, rmax_m=10000)
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
        rmax_m: float = 10000.0,
        volume_attenuation: Optional[str] = None,
        francois_garrison_params: Optional[tuple] = None,
        bio_layers: Optional[list] = None,
        n_angles: Optional[int] = None,
        use_tmpfs: bool = False,
        verbose: bool = False,
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
        rmax_m : float, optional
            Maximum range (m) for angular sampling. Default: 10000. Ignored
            when ``n_angles`` is provided. (Internally converted to km
            because BOUNCE's input format is in km.)
        volume_attenuation : str, optional
            'T' (Thorp), 'F' (Francois-Garrison), 'B' (Biological). Default: None.
        n_angles : int, optional
            Explicit override for the number of angular samples (``NkTab``
            in AT's bounce). If None (default), bounce computes NkTab
            internally from ``rmax_m``. When provided, uacpy sets
            ``rmax_m`` such that bounce's internal formula yields
            approximately ``n_angles`` samples.
        """
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir, **kwargs,
        )

        self.c_low = c_low
        self.c_high = c_high
        self.rmax_m = rmax_m
        self.volume_attenuation = volume_attenuation
        self.francois_garrison_params = francois_garrison_params
        self.bio_layers = bio_layers
        self.n_angles = n_angles
        self._validate_volume_attenuation_params()

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
        # BOUNCE writes a layered AT env; layered bottoms honored.
        self._supports_layered_bottom = True
        self._supports_elastic_media = True
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
        *,
        output_brc: bool = True,
        output_irc: bool = True,
        output_dir: Optional[Path] = None,
        c_low=_UNSET,
        c_high=_UNSET,
        rmax_m=_UNSET,
        volume_attenuation=_UNSET,
        n_angles=_UNSET,
        **kwargs
    ) -> Result:
        """
        Run BOUNCE reflection coefficient computation

        Parameters
        ----------
        env : Environment
            Ocean environment (bottom properties define the stack)
        source : Source
            Source definition (frequency is used)
        receiver : Receiver
            Receiver definition (not used by BOUNCE, but required for API)
        output_brc : bool, optional
            Output .BRC file (for BELLHOP/SCOOTER). Default True.
        output_irc : bool, optional
            Output .IRC file (for KRAKEN). Default True.
        **kwargs
            Additional parameters passed to ENV file writer

        Returns
        -------
        result : Result
            Field containing reflection coefficient data with:
            - angles: grazing angles (degrees)
            - R_magnitude: reflection coefficient magnitudes
            - R_phase: reflection coefficient phases (degrees)
            - brc_file: path to .BRC file (if output_brc=True)
            - irc_file: path to .IRC file (if output_irc=True)

        Notes
        -----
        - For full 90-degree coverage, set c_low=1400, c_high=1e9
          (c_high=1e9 triggers kmin=0 in the Fortran: see bounce.f90)
        - Larger rmax_m gives finer angular resolution
        - The output .BRC / .IRC files can be used directly by other
          models. Pass ``output_dir=<path>`` to persist them across cleanup.
        """
        # Bounce only emits reflection coefficients — guard against the
        # caller asking for something else explicitly.
        if run_mode is not None and run_mode != RunMode.REFLECTION:
            raise UnsupportedFeatureError(
                self.model_name, str(run_mode),
                alternatives=["RunMode.REFLECTION"],
            )

        self._warn_unknown_kwargs(
            kwargs, allowed=('francois_garrison_params', 'bio_layers'),
        )

        with _resolve_overrides(
            self,
            c_low=c_low,
            c_high=c_high,
            rmax_m=rmax_m,
            volume_attenuation=volume_attenuation,
            n_angles=n_angles,
        ):
            if self.c_low <= 0:
                raise ConfigurationError(
                    f"Bounce requires c_low > 0 strictly (got {self.c_low})."
                )
            if self.c_high <= self.c_low:
                raise ConfigurationError(
                    f"c_high ({self.c_high}) must be strictly greater than "
                    f"c_low ({self.c_low})."
                )

            # n_angles → rmax_m: bounce.f90:49 sets
            #     NkTab = INT(1000 * RMax_km * (kMax - kMin) / (2*pi))
            # with kMax = omega/cLow, kMin = omega/cHigh (or 0 if cHigh
            # is "infinity"). Inverting:
            #     RMax_m = NkTab * 2*pi / (omega * (1/cLow - 1/cHigh))
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
                        f"Cannot derive rmax_m from n_angles={self.n_angles}: "
                        f"omega·(1/cLow - 1/cHigh) is non-positive "
                        f"(omega={omega:.3g}, 1/cLow-1/cHigh={inv_c_diff:.3g})."
                    )
                rmax_from_angles_m = (
                    float(self.n_angles) * 2.0 * np.pi / (omega * inv_c_diff)
                )
                self.rmax_m = rmax_from_angles_m

            env = self._project_environment(env)
            self.validate_inputs(env, source, receiver, run_mode=run_mode)

            fm = self._setup_file_manager()

            try:
                base_name = 'bounce_run'
                input_file = fm.get_path(f'{base_name}.env')

                self._log(f"Writing input file: {input_file}", level='info')
                # Caller-supplied francois_garrison_params / bio_layers in
                # run() override the constructor defaults.
                fg = kwargs.pop('francois_garrison_params',
                                self.francois_garrison_params)
                bl = kwargs.pop('bio_layers', self.bio_layers)
                self._write_bounce_input(
                    filepath=input_file,
                    env=env,
                    source=source,
                    receiver=receiver,
                    c_low=self.c_low,
                    c_high=self.c_high,
                    rmax_m=self.rmax_m,
                    volume_attenuation=self.volume_attenuation,
                    francois_garrison_params=fg,
                    bio_layers=bl,
                    **kwargs
                )

                self._log("Running...", level='info')
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

                self._log(f"Reading output: {brc_file}", level='info')
                result = read_reflection_coefficient(str(brc_file), boundary='bottom')

                # Reflection-coefficient files must outlive the work dir
                # because Bellhop/Scooter/Kraken consume them downstream. If
                # ``output_dir`` was passed, persist there (user-owned lifetime);
                # otherwise copy to a private tempdir tied to the Field via
                # weakref.finalize.
                if output_dir is not None:
                    persist_dir = Path(output_dir)
                    persist_dir.mkdir(parents=True, exist_ok=True)
                    field_finalizer = None
                else:
                    # Honour use_tmpfs (RAM-backed /dev/shm) so the
                    # persisted .brc/.irc share the same medium as the
                    # FileManager work_dir.
                    tmp_root = '/dev/shm' if self.use_tmpfs else None
                    persist_dir = Path(tempfile.mkdtemp(
                        prefix='bounce_persist_', dir=tmp_root,
                    ))
                    field_finalizer = persist_dir

                persisted_brc = persist_dir / brc_file.name
                shutil.copy(brc_file, persisted_brc)
                persisted_irc = None
                if irc_file.exists():
                    persisted_irc = persist_dir / irc_file.name
                    shutil.copy(irc_file, persisted_irc)

                result['brc_file'] = str(persisted_brc)
                if persisted_irc is not None:
                    result['irc_file'] = str(persisted_irc)

                from uacpy.core.results import ReflectionCoefficient
                frequency = source.frequencies[0] if hasattr(source.frequencies, '__len__') else source.frequencies

                field = ReflectionCoefficient(
                    theta=result.get('theta', np.array([])),
                    R=result.get('R', np.array([])),
                    phi=result.get('phi', np.array([])),
                    **self._result_kwargs(
                        source,
                        frequencies=frequency,
                        brc_file=result.get('brc_file'),
                        irc_file=result.get('irc_file'),
                        n_pts=result.get('n_pts', 0),
                        c_low=self.c_low,
                        c_high=self.c_high,
                        rmax_m=self.rmax_m,
                        volume_attenuation=self.volume_attenuation,
                        full_result=result,
                    ),
                )

                if field_finalizer is not None:
                    weakref.finalize(
                        field, shutil.rmtree, str(field_finalizer),
                        ignore_errors=True,
                    )

                self._log("Simulation complete", level='info')
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
        rmax_m: float,
        volume_attenuation: Optional[str] = None,
        francois_garrison_params: Optional[tuple] = None,
        bio_layers: Optional[list] = None,
        **kwargs
    ):
        """
        Write BOUNCE input file using ATEnvWriter

        BOUNCE uses ENV format similar to KRAKEN with additional sections:
        - c_low, c_high (phase velocity bounds)
        - RMax in km (for angular sampling) — converted from ``rmax_m``

        BOUNCE does NOT call ``ReadSzRz``; its Fortran driver reads only
        TopOpt, SSP, BotOpt, cLow/cHigh, RMax. We therefore omit the
        source/receiver depth blocks.
        """
        # Parse types
        ssp_type = parse_ssp_type(env.ssp.interp)
        surface_type = parse_boundary_type(env.surface.acoustic_type)
        bottom_type = parse_boundary_type(env.bottom.acoustic_type)

        # Resolve volume attenuation
        vol_atten = None
        if volume_attenuation:
            vol_atten = VolumeAttenuation.from_string(volume_attenuation)

        # Calculate dense mesh for BOUNCE (needs ~20 points per wavelength)
        frequency = source.frequencies[0] if hasattr(source.frequencies, '__len__') else source.frequencies
        c_water = float(env.ssp.to_pairs()[0, 1])
        wavelength = c_water / frequency
        n_mesh = max(100, int(20 * env.depth / wavelength))

        with open(filepath, 'w') as f:
            # Write standard ENV sections using ATEnvWriter
            write_header(
                f, env, source,
                ssp_type=ssp_type,
                surface_type=surface_type,
                attenuation_unit=AttenuationUnits.DB_PER_WAVELENGTH,
                volume_attenuation=vol_atten,
            )

            # F/B volume-attenuation parameter blocks (after TopOpt)
            if vol_atten == VolumeAttenuation.FRANCOIS_GARRISON:
                if francois_garrison_params is None:
                    raise ConfigurationError(
                        "volume_attenuation='F' requires "
                        "francois_garrison_params=(T, S, pH, z_bar)",
                    )
                write_fg_params(f, francois_garrison_params)
            elif vol_atten == VolumeAttenuation.BIOLOGICAL:
                if not bio_layers:
                    raise ConfigurationError(
                        "volume_attenuation='B' requires bio_layers",
                    )
                write_bio_layers(f, bio_layers)

            write_ssp_section(
                f, env, env.depth,
                n_mesh=n_mesh,
                roughness=0.0
            )

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
            f.write(f"{rmax_m / 1000.0:.2f}\n")

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
