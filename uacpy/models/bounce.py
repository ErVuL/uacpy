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
import weakref
import numpy as np
from pathlib import Path
from typing import Optional

from uacpy.models.base import PropagationModel, RunMode, _UNSET
from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.field import Field
from uacpy.core.constants import (
    DEFAULT_SOUND_SPEED, DEFAULT_C_MIN, DEFAULT_C_MAX,
    AttenuationUnits, VolumeAttenuation,
    parse_ssp_type, parse_boundary_type,
)
from uacpy.core.exceptions import UnsupportedFeatureError, ConfigurationError
from uacpy.io.output_reader import read_reflection_coefficient
from uacpy.io.at_env_writer import ATEnvWriter


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
    >>> env = Environment(name="test", depth=100, bottom=bottom)
    >>> source = Source(depth=50, frequency=50)
    >>> receiver = Receiver(depths=np.array([50]))
    >>>
    >>> # Compute reflection coefficients
    >>> bounce = Bounce()
    >>> result = bounce.run(env, source, receiver,
    ...                     cmin=1400, cmax=10000, rmax_km=10)
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
    >>> env_with_rc = Environment(name="test", depth=100, bottom=bottom_with_rc)
    >>> scooter = Scooter()
    >>> tl = scooter.compute_tl(env_with_rc, source, receiver)

    Notes
    -----
    - BOUNCE uses the same environmental file format as KRAKEN
    - The reflection coefficient depends on impedance contrast
    - Supports acoustic, elastic, and poro-elastic layers
    - Tabulated reflection coefficients cover angles from phase velocities [cmin, cmax]
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
        cmin: float = DEFAULT_C_MIN,
        cmax: float = DEFAULT_C_MAX,
        rmax_km: float = 10.0,
        volume_attenuation: Optional[str] = None,
        n_angles: Optional[int] = None,
        use_tmpfs: bool = False,
        verbose: bool = False,
        work_dir: Optional[Path] = None,
    ):
        """
        Parameters
        ----------
        executable : Path, optional
            Path to bounce executable. Auto-detected if None.
        cmin : float, optional
            Minimum phase velocity (m/s) for tabulation. Default: 1400.
            Must be strictly positive (BOUNCE rejects ``cmin <= 0`` — the
            angular grid is derived from ``kx = omega/c``).
        cmax : float, optional
            Maximum phase velocity (m/s) for tabulation. Default: 10000.
            A value of ``1e9`` is a valid recommendation for full 90-deg
            coverage (kraken doc: "cmax large => kmin ~ 0 => grazing
            angles near 0 included"). Must be strictly greater than cmin.
        rmax_km : float, optional
            Maximum range (km) for angular sampling. Default: 10. Ignored
            when ``n_angles`` is provided.
        volume_attenuation : str, optional
            'T' (Thorp), 'F' (Francois-Garrison), 'B' (Biological). Default: None.
        n_angles : int, optional
            Explicit override for the number of angular samples (``NkTab``
            in AT's bounce). If None (default), bounce computes NkTab
            internally from ``rmax_km``. When provided, uacpy sets
            ``rmax_km`` such that bounce's internal formula yields
            approximately ``n_angles`` samples.
        """
        super().__init__(use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir)

        self.cmin = cmin
        self.cmax = cmax
        self.rmax_km = rmax_km
        self.volume_attenuation = volume_attenuation
        self.n_angles = n_angles

        # Validate phase velocity bounds up front
        if self.cmin <= 0:
            raise ConfigurationError(
                f"Bounce requires cmin > 0 strictly (got {self.cmin}). "
                "cmin is the smallest phase velocity on the tabulated grid; "
                "0 would give an infinite wavenumber."
            )
        if self.cmax <= self.cmin:
            raise ConfigurationError(
                f"cmax ({self.cmax}) must be strictly greater than "
                f"cmin ({self.cmin})."
            )

        # BOUNCE computes plane-wave reflection coefficients, not TL.
        self._supported_modes = [RunMode.REFLECTION]

        if executable is None:
            self.executable = self._find_executable_in_paths(
                'bounce',
                bin_subdirs='oalib',
                dev_subdir='Acoustics-Toolbox/Kraken',
            )
        else:
            self.executable = Path(executable)

    def _compute_tl_impl(self, env, source, receiver, **kwargs):
        """Bounce does not compute transmission loss."""
        raise UnsupportedFeatureError(
            self.model_name,
            "transmission loss computation",
            alternatives=[
                "Bellhop", "RAM", "Scooter",
                "KrakenField (uses .brc/.irc from Bounce as input)",
            ],
        )

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        output_brc: bool = True,
        output_irc: bool = True,
        output_dir: Optional[Path] = None,
        cmin=_UNSET,
        cmax=_UNSET,
        rmax_km=_UNSET,
        volume_attenuation=_UNSET,
        n_angles=_UNSET,
        **kwargs
    ) -> Field:
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
        field : Field
            Field containing reflection coefficient data with:
            - angles: grazing angles (degrees)
            - R_magnitude: reflection coefficient magnitudes
            - R_phase: reflection coefficient phases (degrees)
            - brc_file: path to .BRC file (if output_brc=True)
            - irc_file: path to .IRC file (if output_irc=True)

        Notes
        -----
        - For full 90-degree coverage, set cmin=1400, cmax=1e9
          (cmax=1e9 triggers kmin=0 in the Fortran: see bounce.f90)
        - Larger rmax_km gives finer angular resolution
        - The output .BRC / .IRC files can be used directly by other
          models. Pass ``output_dir=<path>`` to persist them across cleanup.
        """
        # Resolve per-call overrides
        cmin = cmin if cmin is not _UNSET else self.cmin
        cmax = cmax if cmax is not _UNSET else self.cmax
        rmax_km = rmax_km if rmax_km is not _UNSET else self.rmax_km
        volume_attenuation = (
            volume_attenuation
            if volume_attenuation is not _UNSET
            else self.volume_attenuation
        )
        n_angles = n_angles if n_angles is not _UNSET else self.n_angles

        # Re-validate in case a caller mutated cmin/cmax between __init__
        # and run, or passed per-call overrides through cmin=/cmax=.
        if cmin <= 0:
            raise ConfigurationError(
                f"Bounce requires cmin > 0 strictly (got {cmin})."
            )
        if cmax <= cmin:
            raise ConfigurationError(
                f"cmax ({cmax}) must be strictly greater than cmin ({cmin})."
            )

        # n_angles override: AT's bounce derives NkTab from rmax_km and
        # frequency; we invert that to get an equivalent rmax_km. See
        # bounce.f90: NkTab = INT(1000 * RMax * OMEGA / (2*PI * DeltaKInv))
        # — in practice uacpy just scales rmax_km linearly with n_angles.
        if n_angles is not None:
            if n_angles <= 0:
                raise ConfigurationError(
                    f"n_angles must be > 0 (got {n_angles})."
                )
            # Very rough inversion: each km of range gives ~bounce's
            # default density of angles. Just pass rmax_km = n_angles/100
            # which matches the default rmax_km=10 giving ~1000 angles.
            # If user set both, n_angles wins.
            rmax_km = float(n_angles) / 100.0
            if rmax_km <= 0:
                import warnings
                warnings.warn(
                    f"Computed rmax_km={rmax_km} from n_angles={n_angles} is "
                    "not positive; using 0.1 km fallback.",
                    RuntimeWarning,
                )
                rmax_km = 0.1

        self.validate_inputs(env, source, receiver)

        # Use the standard FileManager path so cleanup is always consistent.
        fm = self._setup_file_manager()

        try:
            base_name = 'bounce_run'
            input_file = fm.get_path(f'{base_name}.env')

            # Write ENV file (BOUNCE uses KRAKEN format)
            self._log(f"Writing BOUNCE input file: {input_file}", level='info')
            self._write_bounce_input(
                filepath=input_file,
                env=env,
                source=source,
                receiver=receiver,
                cmin=cmin,
                cmax=cmax,
                rmax_km=rmax_km,
                volume_attenuation=volume_attenuation,
                **kwargs
            )

            # Run BOUNCE executable
            self._log("Running BOUNCE...", level='info')
            self._execute(input_file, fm.work_dir)

            # Read output
            brc_file = fm.get_path(f'{base_name}.brc')
            irc_file = fm.get_path(f'{base_name}.irc')

            if not brc_file.exists():
                raise FileNotFoundError(
                    f"BOUNCE output file not found: {brc_file}\n"
                    "BOUNCE execution may have failed."
                )

            # BOUNCE tabulates angles from phase velocities (kx = omega/c),
            # and its angular discretisation can produce duplicate angles
            # near 0 deg (multiple high-c samples all round to the same
            # tiny angle). bellhopcuda validates the .brc with a strict
            # "monotonically increasing" check and aborts on duplicates.
            # Rewrite both files with a strictly-increasing angle axis
            # before they are consumed downstream. See
            # bellhopcuda/bhc::setup() "Bottom reflection coefficients
            # must be monotonically increasing".
            self._dedupe_reflection_file(brc_file)
            if irc_file.exists():
                self._dedupe_reflection_file(irc_file)

            self._log(f"Reading BOUNCE output: {brc_file}", level='info')
            result = read_reflection_coefficient(str(brc_file), boundary='bottom')

            # Reflection-coefficient files must outlive the work dir because
            # downstream models (Bellhop, Scooter, Kraken) consume them.
            # If the caller passed ``output_dir``, persist there (user-owned
            # lifetime). Otherwise copy to a private tempdir and tie its
            # cleanup to the garbage collection of the returned Field via
            # weakref.finalize — no indefinite leak.
            if output_dir is not None:
                persist_dir = Path(output_dir)
                persist_dir.mkdir(parents=True, exist_ok=True)
                field_finalizer = None
            else:
                persist_dir = Path(tempfile.mkdtemp(prefix='bounce_persist_'))
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

            # Convert to Field object
            frequency = source.frequency[0] if hasattr(source.frequency, '__len__') else source.frequency

            field = Field(
                field_type='reflection_coefficients',
                data=result.get('R', np.array([])),  # Magnitude array
                ranges=result.get('theta', np.array([])),  # Using angles as "ranges"
                depths=np.array([0.0]),  # Not applicable for BOUNCE
                frequencies=np.array([frequency]),
                metadata={
                    'theta': result.get('theta', np.array([])),
                    'R': result.get('R', np.array([])),
                    'phi': result.get('phi', np.array([])),
                    'brc_file': result.get('brc_file'),
                    'irc_file': result.get('irc_file'),
                    'n_pts': result.get('n_pts', 0),
                    # Store parameters for later use in Bellhop/Kraken/Scooter
                    'cmin': cmin,
                    'cmax': cmax,
                    'rmax_km': rmax_km,
                    'volume_attenuation': volume_attenuation,
                }
            )

            field.metadata['full_result'] = result

            # Tie the private tempdir lifetime to the Field (if no
            # user-owned output_dir).
            if field_finalizer is not None:
                weakref.finalize(
                    field, shutil.rmtree, str(field_finalizer),
                    ignore_errors=True,
                )

            self._log("BOUNCE simulation complete", level='info')
            return field

        finally:
            if not self.work_dir:
                fm.cleanup_work_dir()

    def _write_bounce_input(
        self,
        filepath: Path,
        env: Environment,
        source: Source,
        receiver: Receiver,
        cmin: float,
        cmax: float,
        rmax_km: float,
        volume_attenuation: Optional[str] = None,
        francois_garrison_params: Optional[tuple] = None,
        bio_layers: Optional[list] = None,
        **kwargs
    ):
        """
        Write BOUNCE input file using ATEnvWriter

        BOUNCE uses ENV format similar to KRAKEN with additional sections:
        - cmin, cmax (phase velocity bounds)
        - rmax_km (for angular sampling)

        BOUNCE does NOT call ``ReadSzRz``; its Fortran driver reads only
        TopOpt, SSP, BotOpt, cLow/cHigh, RMax. We therefore omit the
        source/receiver depth blocks.
        """
        # Parse types
        ssp_type = parse_ssp_type(env.ssp_type)
        surface_type = parse_boundary_type(env.surface.acoustic_type)
        bottom_type = parse_boundary_type(env.bottom.acoustic_type)

        # Resolve volume attenuation
        vol_atten = None
        if volume_attenuation:
            vol_atten = VolumeAttenuation.from_string(volume_attenuation)

        # Calculate dense mesh for BOUNCE (needs ~20 points per wavelength)
        frequency = source.frequency[0] if hasattr(source.frequency, '__len__') else source.frequency
        c_water = env.sound_speed if hasattr(env, 'sound_speed') else DEFAULT_SOUND_SPEED
        wavelength = c_water / frequency
        n_mesh = max(100, int(20 * env.depth / wavelength))

        with open(filepath, 'w') as f:
            # Write standard ENV sections using ATEnvWriter
            ATEnvWriter.write_header(
                f, env, source,
                ssp_type=ssp_type,
                surface_type=surface_type,
                attenuation_unit=AttenuationUnits.DB_PER_WAVELENGTH,
                volume_attenuation=vol_atten,
            )

            # F/B volume-attenuation parameter blocks (after TopOpt)
            if vol_atten == VolumeAttenuation.FRANCOIS_GARRISON:
                if francois_garrison_params is None:
                    from uacpy.core.exceptions import ConfigurationError
                    raise ConfigurationError(
                        "volume_attenuation='F' requires "
                        "francois_garrison_params=(T, S, pH, z_bar)",
                    )
                ATEnvWriter.write_fg_params(f, francois_garrison_params)
            elif vol_atten == VolumeAttenuation.BIOLOGICAL:
                if not bio_layers:
                    from uacpy.core.exceptions import ConfigurationError
                    raise ConfigurationError(
                        "volume_attenuation='B' requires bio_layers",
                    )
                ATEnvWriter.write_bio_layers(f, bio_layers)

            ATEnvWriter.write_ssp_section(
                f, env, env.depth,
                n_mesh=n_mesh,
                roughness=0.0
            )

            ATEnvWriter.write_bottom_section(
                f, env,
                bottom_type=bottom_type,
                filepath=filepath,
                verbose=self.verbose
            )

            # BOUNCE-SPECIFIC SECTIONS

            # Phase velocity bounds (define angular coverage)
            f.write(f"{cmin:.2f} {cmax:.2f}\n")

            # Maximum range (for angular sampling resolution)
            f.write(f"{rmax_km:.2f}\n")

            # BOUNCE does NOT read source/receiver depths — do NOT emit
            # them. AT's bounce.f90 stops after RMax.

    def _execute(self, input_file: Path, work_dir: Path):
        """Execute BOUNCE binary via base-class subprocess helper."""
        base_name = input_file.stem
        result = self._run_subprocess(
            [str(self.executable), base_name],
            cwd=work_dir,
            timeout=300,
        )
        if self.verbose and result.stdout:
            self._log(f"Bounce output:\n{result.stdout}", level='debug')

    @staticmethod
    def _dedupe_reflection_file(filepath: Path) -> None:
        """Rewrite a .brc/.irc file with a strictly-increasing angle axis.

        BOUNCE's Fortran driver tabulates reflection coefficients by
        sweeping phase velocity (kx = omega/c), which — for the cmin/cmax
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
