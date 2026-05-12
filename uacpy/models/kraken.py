"""
Kraken Normal Mode Suite - Unified Architecture

The Kraken suite provides normal mode modeling for underwater acoustics:
- **Kraken**: Normal mode computation (real arithmetic)
- **KrakenC**: Complex normal modes (elastic media, attenuation)
- **KrakenField**: Field computation from modes (TL, pressure)
- **KrakenField** with ``mode_coupling='coupled'``: Range-dependent coupled modes

This module follows the OASES architecture pattern where each executable
is wrapped by a separate class inheriting from a common base.

Note
----
The Acoustics Toolbox also ships ``krakel.exe`` (true elastic normal
modes with shear support using an FEM discretisation). It is bundled in
``uacpy/uacpy/bin/oalib/`` but NOT wrapped by uacpy at this time. Users
who need elastic modes can either:

* drive KrakenC (which handles elastic half-spaces via complex
  wavenumbers), or
* invoke ``krakel.exe`` manually with a Kraken-format .env file.

Usage
-----
```python
from uacpy.models import Kraken, KrakenC, KrakenField

# Compute modes
kraken = Kraken()
modes = kraken.run(env, source, receiver)  # Returns Modes typed result

# Compute TL field from modes
field_model = KrakenField()
result = field_model.run(env, source, receiver)  # Returns PressureField (TL)

# Complex modes for elastic bottom
krakenc = KrakenC()
modes = krakenc.run(env, source, receiver)

# Range-dependent via coupled modes
field_model = KrakenField(mode_coupling='coupled', n_segments=10)
result = field_model.run(env, source, receiver)
```
"""

import os
import re
import warnings
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Union

from uacpy.models.base import PropagationModel, RunMode
from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.results import Result, Modes, PressureField, TransferFunction
from uacpy.core.constants import (
    parse_boundary_type,
    DEFAULT_SOUND_SPEED,
)
import inspect
import shutil
from uacpy.core.exceptions import (
    ConfigurationError, ExecutableNotFoundError, ModelExecutionError,
)
from uacpy.io.oalib_writer import (
    write_absorption_block, write_bottom_section, write_broadband_freqs,
    write_header, write_layer_sections,
    write_multi_profile_env, write_phase_speed_and_rmax,
    write_receiver_depths, write_source_depths, write_ssp_section,
    write_fieldflp,
)
from uacpy.io.oalib_reader import read_shd_file, read_shd_bin
from uacpy.models.coupled_modes import segment_environment_by_range


class _KrakenBase(PropagationModel):
    """Base class for all Kraken models with shared functionality

    Parameters
    ----------
    c_low : float, optional
        Lower phase speed limit (m/s). None = auto (0.95 * min SSP speed).
        Must be non-negative and strictly less than ``c_high``.
    c_high : float, optional
        Upper phase speed limit (m/s). None = auto (1.05 * max of SSP and bottom speed).
        Must be strictly greater than ``c_low``.
    n_mesh : int, optional
        Total number of mesh points PER MEDIUM used by the finite-difference
        mode solver (AT's ``NMESH`` column on the SSP mesh line). 0 = let
        Kraken pick automatically from frequency / wavelength. Default: 0.
        Note: this is NOT a "points per wavelength" density — it is a total
        point count per medium.
    roughness : float, optional
        Bottom roughness (m). Default: 0.0.
    leaky_modes : bool, optional
        If True, override ``c_high`` to 1e9 so Kraken/KrakenC attempt to
        compute leaky modes (trapped modes with phase speeds above the
        halfspace P-wave speed). KrakenC is strongly recommended in this
        mode because it handles complex wavenumbers. See the Kraken doc:
        "CHIGH will attempt to compute leaky modes...". Default: False.

    Notes
    -----
    Defaults auto-derived at ``run()`` time (override only when tuning):

    - ``c_low=None`` → ``min(env.ssp) × 0.95``
    - ``c_high=None`` → ``max(max(env.ssp), env.bottom.sound_speed) × 1.05``
    - ``n_mesh=0`` → Kraken picks mesh from frequency / wavelength.
    - TopOpt position 4 reads ``env.absorption`` (``Thorp`` / ``FrancoisGarrison``
      / ``Biological`` / ``ConstantAbsorption`` / ``None``).

    With ``verbose='info'`` the resolved ``c_low`` / ``c_high`` are logged.
    """

    def __init__(
        self,
        c_low: Optional[float] = None,
        c_high: Optional[float] = None,
        n_mesh: int = 0,
        roughness: float = 0.0,
        interp_ssp: Optional[str] = None,
        leaky_modes: bool = False,
        top_reflection_file: Optional[Path] = None,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        **kwargs
    ):
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir, **kwargs,
        )
        self.interp_ssp = interp_ssp
        self.c_low = c_low
        self.c_high = c_high
        self.n_mesh = n_mesh
        self.roughness = roughness
        self.leaky_modes = leaky_modes
        self.top_reflection_file = (
            Path(top_reflection_file) if top_reflection_file is not None else None
        )

        if leaky_modes:
            # Override c_high so kraken/krakenc attempts leaky modes.
            # See Kraken doc: "CHIGH will attempt to compute leaky modes..."
            self.c_high = 1e9

        # CLOW/CHIGH validation (Kraken doc: 0 <= cLow < cHigh)
        self._validate_phase_speed_limits()

    def _validate_phase_speed_limits(self):
        """Check 0 <= c_low < c_high when either is explicitly set."""
        cl = self.c_low
        ch = self.c_high
        if cl is not None and cl < 0:
            raise ConfigurationError(
                f"c_low must be >= 0, got {cl}"
            )
        if cl is not None and ch is not None and ch <= cl:
            raise ConfigurationError(
                f"c_high ({ch}) must be strictly greater than c_low ({cl})"
            )

    def _check_kraken_ssp_type(self):
        """Reject SSP interpolation choices kraken does not implement.

        Per AT ``sspMod.f90:61-89`` kraken accepts codes A (analytic),
        N (N^2-linear), C (C-linear), P (PCHIP), S (spline). The 'Q'
        quadrilateral code is Bellhop-only (see RangeDepSSPFile.htm).

        Default ``self.interp_ssp=None`` resolves to 'linear' for
        Kraken's env (auto-quad only applies to Bellhop), so the
        rejection only fires on explicit 'quad'.
        """
        if self.interp_ssp is None:
            return
        if str(self.interp_ssp).lower() in ('q', 'quad', 'quadratic'):
            raise ConfigurationError(
                "Kraken/KrakenC/KrakenField do not support the 'quad' "
                "SSP interpolation — it is Bellhop-only. Pick one of "
                "'linear' (C-linear), 'n2linear', 'pchip', 'cubic' / "
                "'spline'."
            )

    def _build_modes_field(self, modes, n_modes, source):
        """Wrap a modes-reader payload as a :class:`Modes` Result.

        Returns the full mode set the reader produced; callers cap the
        count via :meth:`Modes.first_n` if they passed an ``n_modes``
        request. Shared by Kraken.run and KrakenC.run.
        """
        k_arr = modes.get('k', np.array([]))
        phi_arr = modes.get('phi', np.array([]))
        z_arr = modes.get('z', np.array([]))

        result = Modes(
            k=k_arr,
            phi=phi_arr,
            depths=z_arr,
            n_modes=len(k_arr),
            **self._result_kwargs(
                source,
                backend=Path(self.executable).name if self.executable else self.model_name.lower(),
                frequencies=float(source.frequencies[0]),
                n_modes_requested=n_modes,
                leaky_modes=self.leaky_modes,
            ),
        )
        if n_modes is not None:
            result = result.first_n(int(n_modes))
        return result

    @staticmethod
    def _compute_rmax_m(receiver, fallback_m: float = 100_000.0) -> float:
        """Derive field-computation RMax (m) from receiver ranges.

        Adds a 5 % buffer so field.exe doesn't clip the outermost ranges.
        Falls back to ``fallback_m`` if the receiver has no explicit range
        vector (e.g. mode-only Kraken runs).
        """
        if receiver is None:
            return float(fallback_m)
        ranges = getattr(receiver, 'ranges', None)
        if ranges is None or len(np.atleast_1d(ranges)) == 0:
            return float(fallback_m)
        rmax_m = float(np.max(np.asarray(ranges, dtype=float)))
        if rmax_m <= 0:
            return float(fallback_m)
        return rmax_m * 1.05

    def _write_kraken_env(self, filepath, env, source, **kwargs):
        """
        Write Kraken environment file using shared ATEnvWriter

        Kraken has additional sections beyond the standard ENV format:
        - Phase speed limits (cLow, cHigh)
        - Maximum range (RMax)
        - Optional broadband frequency vector (TopOpt(6)='B')
        """
        # Reject 'quad' SSP interp (Bellhop-only)
        self._check_kraken_ssp_type()
        # Re-validate in case caller mutated attributes after __init__
        self._validate_phase_speed_limits()

        from uacpy.io.oalib_writer import resolve_ssp_topopt
        ssp_topopt = resolve_ssp_topopt(env, self.interp_ssp)
        surface_type = parse_boundary_type(env.surface.acoustic_type)
        bottom_acoustic_type = env.halfspace_at_range(0.0).acoustic_type
        bottom_type = parse_boundary_type(bottom_acoustic_type)

        # Top reflection coefficient file (.trc): override surface BC to 'F'
        # and copy the user-supplied file next to the .env file. AT's
        # ReadEnvironmentBell.f90 reads <base>.trc when TopOpt(2:2)='F'.
        top_reflection_file = getattr(self, 'top_reflection_file', None)
        if top_reflection_file is not None:
            from uacpy.core.constants import BoundaryType
            src = Path(top_reflection_file)
            if not src.exists():
                raise ConfigurationError(
                    f"top_reflection_file not found: {src}"
                )
            surface_type = BoundaryType.FILE
            trc_dest = Path(filepath).with_suffix('.trc')
            shutil.copy(src, trc_dest)

        receiver_obj = kwargs.get('receiver_obj', None)
        receiver_depths = kwargs.get('receiver_depths', [100.0])
        rmax_m = kwargs.get('rmax_m', None)
        if rmax_m is None:
            rmax_m = self._compute_rmax_m(receiver_obj, fallback_m=100_000.0)
        frequencies = kwargs.get('frequencies', None)

        with open(filepath, 'w') as f:
            write_header(
                f, env, source,
                ssp_topopt=ssp_topopt,
                surface_type=surface_type,
                frequencies=frequencies,
            )
            write_absorption_block(f, env)

            write_ssp_section(
                f, env, env.depth,
                n_mesh=self.n_mesh,
                roughness=self.roughness
            )

            # Write sediment layers if layered bottom
            write_layer_sections(
                f, env, env.depth
            )

            write_bottom_section(
                f, env,
                bottom_type=bottom_type,
                emit_reflection_table_block=False,
            )

            # KRAKEN-SPECIFIC SECTIONS

            from uacpy.io.oalib_writer import resolve_phase_speed_bounds
            cl, ch = resolve_phase_speed_bounds(env, self.c_low, self.c_high)
            if self.c_low is None or self.c_high is None:
                self._log(
                    f"c_low / c_high auto-derived from env.ssp + bottom = "
                    f"{cl:.1f} / {ch:.1f} m/s"
                )
            write_phase_speed_and_rmax(
                f, env,
                rmax_m=rmax_m,
                c_low=cl, c_high=ch,
            )

            # Source depths (use ATEnvWriter helper for full non-uniform support)
            write_source_depths(f, source)

            # Receiver depths: receiver_obj (if present) preserves a non-
            # uniform array verbatim; otherwise fall back to the depths array.
            write_receiver_depths(
                f, receiver_obj if receiver_obj is not None else receiver_depths,
            )

            # Broadband frequency vector (read by ReadfreqVec AFTER SD/RD)
            if frequencies is not None and len(np.atleast_1d(frequencies)) > 1:
                write_broadband_freqs(f, np.asarray(frequencies))

    def _run_kraken_executable(self, base_name: str, work_dir: Path):
        """Execute Kraken/KrakenC via base-class subprocess runner."""
        try:
            self._run_subprocess(
                [str(self.executable), base_name],
                cwd=work_dir,
            )
        except ModelExecutionError as exc:
            self._attach_prt_tail(exc, work_dir, base_name)
            raise

    def _compute_modes_impl(self, env, source, n_modes, **kwargs):
        """Override base class: use a dense depth grid for mode sampling.

        ``PropagationModel._compute_modes_impl`` uses a dummy receiver
        with a single depth of 0, which clips the .mod eigenfunctions to
        one sample per mode — useless for plotting or downstream field
        reconstruction. Here we build a dense grid spanning the full
        water column (and any configured sediment depth) so the returned
        Field contains meaningful mode shapes.

        Parameters
        ----------
        env, source, n_modes : see PropagationModel.compute_modes
        mode_depth_grid : array-like, optional (kwarg)
            User-supplied mode sampling depths. If omitted the grid is
            ``max(100, total_depth * mode_points_per_meter)`` points
            linearly spaced from 0 to the total media depth.
        """
        from uacpy.core.receiver import Receiver as _Receiver

        override = kwargs.pop('mode_depth_grid', None)
        if override is not None:
            mode_depths = np.asarray(override, dtype=float)
        else:
            total_depth = env.depth
            if env.has_layered_bottom():
                for layer in env.bottom.layers:
                    total_depth += layer.thickness
            ppm = float(getattr(self, 'mode_points_per_meter', 0.0) or 0.0)
            n_pts = max(100, int(round(float(total_depth) * ppm)))
            mode_depths = np.linspace(0.0, float(total_depth), n_pts)

        dense_receiver = _Receiver(depths=mode_depths, ranges=[0.0])
        # Kraken/KrakenC.run ignore run_mode (they only compute modes); only
        # pass it if the concrete class advertises the kwarg (KrakenField).
        sig = inspect.signature(self.run)
        if 'run_mode' in sig.parameters:
            return self.run(
                env, source, dense_receiver,
                run_mode=RunMode.MODES, n_modes=n_modes, **kwargs,
            )
        return self.run(
            env, source, dense_receiver,
            n_modes=n_modes, **kwargs,
        )

    def _read_modes_file(self, filepath: Path) -> Dict:
        """Read a Kraken ``.mod`` file using the binary reader."""
        from uacpy.io.modes_reader import read_modes_bin

        # read_modes_bin expects the filename without extension and appends
        # its own ('.moA'); strip '.mod' before handing it over.
        filepath_str = str(filepath)
        if filepath_str.endswith('.mod'):
            basename = filepath_str[:-4]
        else:
            basename = filepath_str

        # read_modes_bin will add .moA, but Kraken produces .mod
        # So we need to rename the file first
        mod_file = basename + '.mod'
        moa_file = basename + '.moA'

        if os.path.exists(mod_file) and not os.path.exists(moa_file):
            shutil.copy(mod_file, moa_file)

        # Check if modes file exists and is not empty
        # Kraken produces empty files when it encounters unsupported conditions
        # Check both .mod and .moA files (read_modes_bin expects .moA)
        files_to_check = [mod_file, moa_file, basename + '.modfil']
        is_empty = False
        for check_file in files_to_check:
            if os.path.exists(check_file) and os.path.getsize(check_file) == 0:
                is_empty = True
                break

        if is_empty:
            raise ModelExecutionError(
                self.model_name, return_code=0, stdout=None,
                stderr=self._modes_error_message(basename),
            )

        # Try to read modes, catch IndexError from reading empty binary files
        try:
            modes_data = read_modes_bin(basename, freq=0.0)
        except IndexError as e:
            raise ModelExecutionError(
                self.model_name, return_code=0, stdout=None,
                stderr=self._modes_error_message(basename, original_error=e),
            ) from e

        return modes_data

    @staticmethod
    def _modes_error_message(basename, original_error=None):
        """Build error message for invalid mode files, checking .prt for clues.

        Only suggest KrakenC when the PRT evidences an actual elastic
        configuration (acousto-elastic boundary or a non-zero shear speed
        in the halfspace summary) or when kraken reports it couldn't find
        modes at cLow — not on any stray 'elastic' token in the PRT.
        """
        prt_file = basename + '.prt'
        error_msg = "Kraken did not produce valid modes. "
        if os.path.exists(prt_file):
            with open(prt_file, 'r') as f:
                prt_content = f.read()

            # 1. True "acousto-elastic" mention (used in AT PRT for elastic HS)
            has_acousto_elastic = bool(
                re.search(r'acousto[-\s]*elastic', prt_content, re.IGNORECASE)
            )

            # 2. Non-zero shear speed anywhere in the halfspace summary.
            #    AT prints lines like "Shear speed = <value>" and
            #    "Bot. Shear speed  = <value>". A non-zero value means
            #    the elastic code path is engaged.
            has_nonzero_shear = False
            for m in re.finditer(
                r'[Ss]hear\s*speed\s*=?\s*([0-9.+\-eE]+)',
                prt_content,
            ):
                try:
                    if abs(float(m.group(1))) > 0.0:
                        has_nonzero_shear = True
                        break
                except ValueError:
                    pass

            # 3. Kraken-specific failure string
            modes_not_found = 'modes not found at cLow' in prt_content

            if has_acousto_elastic or has_nonzero_shear or modes_not_found:
                error_msg += (
                    "Kraken (real arithmetic) failed. This is typical when "
                    "the environment has an acousto-elastic bottom "
                    "(non-zero shear speed) or when modes cannot be found "
                    "at cLow. Try KrakenC (complex mode version) which "
                    "handles shear and leaky modes. Alternatives: Bellhop, "
                    "RAM, Scooter, OAST."
                )
            else:
                error_msg += f"Check the .prt file for details: {prt_file}"
                if original_error:
                    error_msg += f". Original error: {original_error}"
        elif original_error:
            error_msg += f"Original error: {original_error}"
        return error_msg


class Kraken(_KrakenBase):
    """
    Kraken - Normal Mode Computation (Real Arithmetic)

    Computes normal modes for range-independent environments using real
    arithmetic. Best for fluid media (water and sediment) without significant
    attenuation or elastic effects.

    For elastic bottoms or complex attenuation, use KrakenC instead.
    For field computation from modes, use KrakenField.
    For range-dependent scenarios, use KrakenField(mode_coupling='coupled').

    Parameters
    ----------
    executable : Path, optional
        Path to ``kraken.exe``. Auto-detected if ``None``.
    mode_points_per_meter : float, optional
        Mode-grid sampling density. Default ``1.5``.
    c_low, c_high : float, optional
        Phase-speed bounds (m/s). ``None`` ⇒ ``0.95 × min SSP`` /
        ``1.05 × max SSP+bottom``.
    n_mesh : int, optional
        Mesh points per medium. ``0`` ⇒ Kraken auto-picks. Default ``0``.
    roughness : float, optional
        Bottom RMS roughness (m). Default ``0``.
    leaky_modes : bool, optional
        Override ``c_high`` to ``1e9`` so kraken attempts leaky modes.
        KrakenC is recommended in this mode (complex arithmetic).
    top_reflection_file : Path, optional
        Path to ``.trc`` — sets surface BC ``TopOpt(2)='F'``.
    use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
        Standard plumbing (see :class:`PropagationModel`).

    Returns
    -------
    Modes
        Typed modal result with ``k`` (complex wavenumbers, shape ``(M,)``),
        ``phi`` (mode shapes ``(nz, M)``), ``depths``, ``frequencies``
        (length-1), ``n_modes``.

    Notes
    -----
    **No auto-route to KrakenC.** If the env carries elastic media
    (``shear_speed > 0``) or ``leaky_modes=True`` is set, ``kraken.exe``
    will likely fail to find modes; the wrapper inspects the ``.prt``
    log and raises ``ModelExecutionError`` with a message suggesting
    KrakenC. Instantiate ``KrakenC`` directly for elastic / leaky-mode
    environments. (``KrakenField`` *does* auto-route internally because
    it owns the modes-solve step; ``Kraken`` is the stand-alone modes
    solver and leaves the choice to the user.)

    **Collapse defaults (overrides of :data:`DEFAULT_COLLAPSE`).**
    Modes solve an eigenproblem of the whole water column — no
    source-side. Per-model defaults: ``'ssp': 'mean'``,
    ``'bottom': 'median'``. Other keys keep the global defaults.

    Examples
    --------
    >>> kraken = Kraken()
    >>> modes = kraken.run(env, source, receiver)
    >>> print(f"Computed {len(modes.k)} modes")
    """

    def __init__(
        self,
        executable: Optional[Path] = None,
        mode_points_per_meter: float = 1.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mode_points_per_meter = float(mode_points_per_meter)
        self._supported_modes = [RunMode.MODES]
        # Elastic media are flagged supported because KrakenC handles
        # them; the stand-alone ``Kraken`` (real-arithmetic) class will
        # fail at runtime on elastic envs and the wrapper's PRT-aware
        # error message tells the user to switch to KrakenC. (Auto-route
        # lives inside KrakenField only — see KrakenField._select_kraken_exe.)
        self._supports_altimetry = False
        self._supports_range_dependent_bathymetry = False
        self._supports_range_dependent_ssp = False
        self._supports_range_dependent_bottom = False
        self._supports_layered_bottom = True
        self._supports_range_dependent_layered_bottom = False
        self._supports_elastic_media = True
        self._supports_multi_source_depth = False
        # Modes solve an eigenproblem over the whole water column —
        # there is no "source-side" profile. Median/mean samples are
        # representative of the path the modes describe.
        self._set_collapse_defaults({
            'ssp': 'mean',
            'bottom': 'median',
        })
        if executable is None:
            self.executable = self._find_executable_in_paths(
                'kraken.exe',
                bin_subdirs='oalib',
                dev_subdir='Acoustics-Toolbox/Kraken',
            )
        else:
            self.executable = Path(executable)

        if not self.executable.exists():
            raise ExecutableNotFoundError(self.model_name, str(self.executable))

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional[RunMode] = None,
        *,
        n_modes: Optional[int] = None,
        **kwargs
    ) -> Result:
        """
        Compute normal modes

        Parameters
        ----------
        env : Environment
            Must be range-independent
        source : Source
            Used for frequency
        receiver : Receiver
            Used for depth grid
        n_modes : int, optional
            Maximum number of modes to return. Kraken itself has no mode-
            count cap (mode count is bounded by ``cHigh`` vs the halfspace
            P-wave speed), but when ``n_modes`` is provided the returned
            ``Field`` is clipped to at most that many modes. To reduce
            Kraken's own work, lower ``c_high`` or set it just below the
            halfspace P-wave speed. Use ``KrakenField.run(n_modes=N)`` to
            apply ``MLimit`` in the FLP file during field reconstruction.

        Returns
        -------
        Modes
        """
        run_mode = self._resolve_run_mode(run_mode)

        env = self._project_environment(env)
        self.validate_inputs(env, source, receiver, run_mode=run_mode)

        fm = self._setup_file_manager()
        base_name = 'modes'

        try:
            env_file = fm.get_path(f'{base_name}.env')
            self._log(f"Writing environment file: {env_file}")
            self._write_kraken_env(
                env_file, env, source,
                receiver_obj=receiver,
                receiver_depths=receiver.depths,
                **kwargs,
            )

            self._log("Running Kraken...")
            self._run_kraken_executable(base_name, fm.work_dir)

            modes_file = fm.get_path(base_name)
            self._log(f"Reading mode file: {modes_file}.mod")
            modes = self._read_modes_file(modes_file)

            self._log("Simulation complete")

            field = self._build_modes_field(modes, n_modes, source)
            self._attach_output_paths(
                field, fm.work_dir, base_name,
                primary_files=(('mod_file', '.mod'),),
            )
            return field

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()


class KrakenC(_KrakenBase):
    """
    KrakenC - Complex Normal Modes

    Computes normal modes using complex arithmetic for environments with:
    - Elastic (solid) bottoms with shear waves
    - Significant attenuation
    - Range-dependent attenuation

    Uses krakenc.exe which handles complex wavenumbers and mode shapes.

    Parameters
    ----------
    Same as :class:`Kraken` (``executable``, ``mode_points_per_meter``,
    ``c_low``, ``c_high``, ``n_mesh``, ``roughness``,
    ``leaky_modes``, ``top_reflection_file`` plus standard plumbing).
    The default ``executable`` resolves to ``krakenc.exe``.

    Notes
    -----
    **Collapse defaults (overrides of :data:`DEFAULT_COLLAPSE`).**
    Same rationale as :class:`Kraken`: ``'ssp': 'mean'``,
    ``'bottom': 'median'``.

    Examples
    --------
    >>> krakenc = KrakenC()
    >>> modes = krakenc.run(env_with_elastic_bottom, source, receiver)
    """

    def __init__(
        self,
        executable: Optional[Path] = None,
        mode_points_per_meter: float = 1.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mode_points_per_meter = float(mode_points_per_meter)
        self._supported_modes = [RunMode.MODES]
        self._supports_altimetry = False
        self._supports_range_dependent_bathymetry = False
        self._supports_range_dependent_ssp = False
        self._supports_range_dependent_bottom = False
        self._supports_layered_bottom = True
        self._supports_range_dependent_layered_bottom = False
        self._supports_elastic_media = True
        self._supports_multi_source_depth = False
        # Modes solve an eigenproblem over the whole water column —
        # there is no "source-side" profile. Median/mean samples are
        # representative of the path the modes describe.
        self._set_collapse_defaults({
            'ssp': 'mean',
            'bottom': 'median',
        })
        if executable is None:
            self.executable = self._find_executable_in_paths(
                'krakenc.exe',
                bin_subdirs='oalib',
                dev_subdir='Acoustics-Toolbox/Kraken',
            )
        else:
            self.executable = Path(executable)

        if not self.executable.exists():
            raise ExecutableNotFoundError(self.model_name, str(self.executable))

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional[RunMode] = None,
        *,
        n_modes: Optional[int] = None,
        **kwargs
    ) -> Result:
        """
        Compute complex normal modes.

        Uses complex arithmetic, required for environments with elastic
        (solid) boundaries that support shear waves, or with significant
        volume attenuation.

        Returns
        -------
        Modes
            Typed modal result with ``k``, ``phi``, ``z``, ``n_modes``.
        """
        run_mode = self._resolve_run_mode(run_mode)

        env = self._project_environment(env)
        self.validate_inputs(env, source, receiver, run_mode=run_mode)

        fm = self._setup_file_manager()
        base_name = 'modes'

        try:
            env_file = fm.get_path(f'{base_name}.env')
            self._log(f"Writing environment file: {env_file}")
            self._write_kraken_env(
                env_file, env, source,
                receiver_obj=receiver,
                receiver_depths=receiver.depths,
                **kwargs,
            )

            self._log("Running KrakenC...")
            self._run_kraken_executable(base_name, fm.work_dir)

            modes_file = fm.get_path(base_name)
            self._log(f"Reading mode file: {modes_file}.mod")
            modes = self._read_modes_file(modes_file)

            self._log("Simulation complete")

            field = self._build_modes_field(modes, n_modes, source)
            self._attach_output_paths(
                field, fm.work_dir, base_name,
                primary_files=(('mod_file', '.mod'),),
            )
            return field

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()


class KrakenField(_KrakenBase):
    """
    KrakenField - Field Computation from Normal Modes

    Computes acoustic field (TL, pressure) using the AT pipeline:
    kraken.exe (modes) → field.exe (field reconstruction) → .shd output.

    Supports range-independent and range-dependent environments via
    adiabatic or coupled mode theory (delegated to AT's field.exe).

    Note
    ----
    Range-dependent bathymetry is supported via environment segmentation
    (multi-profile .env + field.exe coupled/adiabatic modes). Sea-surface
    altimetry (non-flat sea surface) is NOT supported; Bellhop is the
    only uacpy model that supports altimetry.

    Parameters
    ----------
    mode_coupling : str, optional
        ``'adiabatic'`` (default) or ``'coupled'``. Controls how
        ``field.exe`` handles range-dependent mode transitions.
    coherent : bool, optional
        Coherent mode addition (default ``True``). ``coupled`` +
        ``coherent=False`` is rejected — ``field.exe`` has no
        coupled-incoherent path.
    n_segments : int, optional
        Range segments for RD scenarios. Default ``10``.
    mode_points_per_meter : float, optional
        Mode-depth grid density. Default ``1.5`` pts/m.
    source_beam_pattern_file : Path, optional
        Bellhop-style ``.sbp`` file; sets ``field.exe`` ``Opt(3)='*'``.
    source_type : str, optional
        ``field.exe`` Opt(1): ``'R'`` cylindrical (default) | ``'X'``
        Cartesian line | ``'S'`` scaled-cylindrical.
    executable, field_executable : Path, optional
        ``kraken.exe`` and ``field.exe`` paths. Auto-detected if ``None``.
    c_low, c_high, n_mesh, roughness, leaky_modes, top_reflection_file : optional
        Inherited from :class:`_KrakenBase` — same semantics as :class:`Kraken`.
    use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
        Standard plumbing (see :class:`PropagationModel`).

    Notes
    -----
    ``field.exe`` ``Opt(3)`` only accepts ``'*'``, ``'O'``, or ``' '``
    (``field.f90:83-90``); anything else raises FATAL ERROR. Purely
    elastic component selection (H/V/T/N) is not reachable through
    ``field.exe`` — an upstream Fortran limitation.

    **Auto-route to KrakenC** when ``env`` carries shear (delegates the
    modes step to ``krakenc.exe``).

    **Collapse defaults (overrides of :data:`DEFAULT_COLLAPSE`).**
    RD bathymetry and RD-SSP are honoured natively (segments). Per-model
    defaults: ``'bottom': 'median'`` (median over RD halfspace samples),
    ``'rd_layered_layers': 'preserve'`` (KrakenField consumes
    ``LayeredBottom`` natively, so RDLB collapses to the median range
    with the layer stack kept).

    Examples
    --------
    >>> field_model = KrakenField()
    >>> result = field_model.run(env, source, receiver)

    >>> # Range-dependent with coupled modes
    >>> field_model = KrakenField(mode_coupling='coupled', n_segments=20)
    >>> result = field_model.run(env_rd, source, receiver)

    >>> # Incoherent mode addition
    >>> field_model = KrakenField(coherent=False)
    >>> result = field_model.run(env, source, receiver)
    """

    # Valid source types for field.exe option col 1 (AT field.f90:71-79).
    _ALLOWED_SOURCE_TYPE = ('R', 'X', 'S')

    def __init__(
        self,
        mode_points_per_meter: float = 1.5,
        mode_coupling: str = 'adiabatic',
        coherent: bool = True,
        n_segments: int = 10,
        executable: Optional[Path] = None,
        field_executable: Optional[Path] = None,
        source_beam_pattern_file: Optional[Path] = None,
        source_type: str = 'R',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._supported_modes = [
            RunMode.COHERENT_TL,
            RunMode.BROADBAND,
            RunMode.TIME_SERIES,
        ]
        self._supports_altimetry = False
        self._supports_range_dependent_bathymetry = True
        self._supports_range_dependent_ssp = True
        self._supports_range_dependent_bottom = False
        self._supports_range_dependent_layered_bottom = False
        self._supports_layered_bottom = True
        self._supports_elastic_media = True
        self._supports_multi_source_depth = False
        # KrakenField segments RD-bathy / RD-SSP natively; only the
        # bottom and RDLB axes still collapse. Median across range is
        # the representative single profile per segment.
        self._set_collapse_defaults({
            'bottom': 'median',
            'rd_layered_layers': 'preserve',
        })
        if mode_coupling not in ('adiabatic', 'coupled'):
            raise ConfigurationError(
                f"mode_coupling must be 'adiabatic' or 'coupled', "
                f"got {mode_coupling!r}"
            )

        if executable is None:
            self.executable = self._find_executable_in_paths(
                'kraken.exe',
                bin_subdirs='oalib',
                dev_subdir='Acoustics-Toolbox/Kraken',
            )
        else:
            self.executable = Path(executable)

        if not self.executable.exists():
            raise ExecutableNotFoundError(self.model_name, str(self.executable))

        if field_executable is None:
            self._field_exe = self._find_executable_in_paths(
                'field.exe',
                bin_subdirs='oalib',
                dev_subdir='Acoustics-Toolbox/Kraken',
            )
        else:
            self._field_exe = Path(field_executable)

        if not self._field_exe.exists():
            raise ExecutableNotFoundError(
                f"{self.model_name} (field.exe)", str(self._field_exe),
            )

        self.mode_points_per_meter = mode_points_per_meter
        self.mode_coupling = mode_coupling
        self.coherent = coherent
        self.n_segments = n_segments
        self.source_beam_pattern_file = (
            Path(source_beam_pattern_file)
            if source_beam_pattern_file is not None else None
        )

        # Validate source_type
        source_type = str(source_type).upper()
        if source_type not in self._ALLOWED_SOURCE_TYPE:
            raise ConfigurationError(
                f"source_type must be one of {self._ALLOWED_SOURCE_TYPE}, "
                f"got {source_type!r}"
            )
        self.source_type = source_type

    def _build_field_option(self, is_range_dependent: bool) -> str:
        """Build the 4-character option string for field.exe.

        Columns follow AT ``field.f90`` / ``ReadModes.f90``:

        * pos 1: source geometry — 'R' point source (cylindrical), 'X'
          line source (Cartesian), 'S' scaled point source.
        * pos 2: coupling — 'C' coupled modes, 'A' adiabatic.
          For NProf > 1 we honour ``mode_coupling``; for range-independent
          runs we default to 'C' (coupled) so the option string is fully
          populated rather than containing a padding blank. AT's
          field.f90 treats NProf == 1 identically for 'A' and 'C'.
        * pos 3: source beam pattern — '*' to apply a .sbp file, else
          ' ' (omnidirectional). Field.exe rejects any other character
          (``field.f90:83-90``), so the elastic Comp selector (H/V/T/N)
          is not exposed here; it is only reachable if a user invokes
          ReadModes directly.
        * pos 4: 'C' coherent TL, 'I' incoherent.
        """
        pos1 = self.source_type
        if is_range_dependent:
            pos2 = 'C' if self.mode_coupling.lower() == 'coupled' else 'A'
        else:
            # Range-independent: AT doesn't require 'A'/'C', but setting
            # 'C' keeps the option string fully specified and matches what
            # AT's own field.f90 does internally when NProf == 1.
            pos2 = 'C'
        # pos3: '*' => field.exe reads <base>.sbp, else omnidirectional.
        pos3 = '*' if self.source_beam_pattern_file is not None else ' '
        pos4 = 'C' if self.coherent else 'I'
        return f"{pos1}{pos2}{pos3}{pos4}"

    @staticmethod
    def _total_media_depth(env):
        """Return total depth through ocean + sediment layers."""
        depth = env.depth
        if env.has_layered_bottom():
            for layer in env.bottom.layers:
                depth += layer.thickness
        return depth

    def _select_kraken_exe(self, env):
        """Return 'kraken.exe' or 'krakenc.exe' based on environment."""
        needs_krakenc = (
            env.has_elastic_bottom()
            or env.has_elastic_surface()
            # leaky_modes requires complex arithmetic for convergence.
            or getattr(self, 'leaky_modes', False)
        )
        if needs_krakenc:
            if self.model_name == 'Kraken':
                warnings.warn(
                    f"{self.model_name}: env contains elastic media or "
                    f"leaky_modes=True; auto-routing to krakenc.exe "
                    f"(complex-arithmetic Kraken) for the modes solve.",
                    UserWarning, stacklevel=3,
                )
            return self._find_executable_in_paths(
                'krakenc.exe',
                bin_subdirs='oalib',
                dev_subdir='Acoustics-Toolbox/Kraken',
            )
        return self.executable  # kraken.exe

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode=None,
        frequencies: Optional[np.ndarray] = None,
        n_modes: Optional[int] = None,
        source_waveform=None,
        sample_rate=None,
        **kwargs
    ) -> Result:
        """
        Compute TL field using normal modes.

        Uses AT pipeline: kraken.exe → field.exe → .shd.

        Parameters
        ----------
        env : Environment
            Can be range-dependent or range-independent.
        source : Source
            Source configuration.
        receiver : Receiver
            Receiver grid.
        run_mode : RunMode, optional
            ``COHERENT_TL`` (default), ``BROADBAND``, or ``TIME_SERIES``.
            ``TIME_SERIES`` requires ``source_waveform`` + ``sample_rate``.
        frequencies : ndarray, optional
            Frequency vector (Hz) for native broadband computation. Uses
            TopOpt(6)='B' so kraken writes one multi-frequency .mod file
            and field.exe handles all frequencies in a single pass.
        n_modes : int, optional
            Max number of modes used by field.exe (FLP ``MLimit``).
        source_waveform : ndarray, optional
            Source pulse for ``TIME_SERIES`` mode.
        sample_rate : float, optional
            Sampling rate of ``source_waveform`` in Hz.
        """
        # Early gate: coupled-mode field calculations cannot be
        # combined with incoherent mode addition. AT's field.f90
        # (KrakenField/field.f90 around line 123-127) calls ERROUT
        # on Opt(2:2)='C' + Opt(4:4)='I', which surfaces in Python
        # as an opaque "no .shd file" error. Fail loudly up front.
        if (
            env.is_range_dependent
            and self.mode_coupling == 'coupled'
            and not self.coherent
        ):
            raise ConfigurationError(
                "KrakenField: coupled mode calculations do not support "
                "incoherent addition of modes. Use mode_coupling="
                "'adiabatic' with coherent=False, or keep "
                "mode_coupling='coupled' with coherent=True."
            )

        # field.f90's Comp selector at line 169 calls Extract(...) per
        # ReadModes.f90:315-324, which has no default branch for
        # ELASTIC media when Comp ∈ {'*', 'O', ' '} (the wrapper's
        # only options). Receivers in elastic layers therefore see
        # uninitialised Phi(j). Refuse loudly.
        if (
            env.has_layered_bottom()
            and any(
                (getattr(layer, 'shear_speed', 0) or 0) > 0
                for layer in env.bottom.layers
            )
        ):
            rcv_depths = np.atleast_1d(np.asarray(receiver.depths, dtype=float))
            if rcv_depths.max() > env.depth:
                raise ConfigurationError(
                    "KrakenField: receiver depth exceeds water depth and "
                    "an elastic layer is present in the bottom. AT's "
                    "field.f90 elastic-component selector (Comp) only "
                    "supports {'*','O',' '} which fall through "
                    "uninitialised in elastic media (ReadModes.f90:315-324). "
                    "Constrain receiver to the water column or use a "
                    "fluid bottom."
                )

        # Default run mode: BROADBAND if a freq vector is provided,
        # else single-frequency coherent TL.
        smart_default = (
            RunMode.BROADBAND
            if frequencies is not None and len(np.atleast_1d(frequencies)) > 1
            else RunMode.COHERENT_TL
        )
        run_mode = self._resolve_run_mode(run_mode, default=smart_default)

        if run_mode in (RunMode.BROADBAND, RunMode.TIME_SERIES):
            if run_mode == RunMode.TIME_SERIES and (
                source_waveform is None or sample_rate is None
            ):
                raise ConfigurationError(
                    "KrakenField.run(run_mode=TIME_SERIES) requires "
                    "source_waveform and sample_rate. For the broadband "
                    "transfer function H(f), use run_mode=RunMode.BROADBAND."
                )
            tf = self._compute_broadband_field(
                env, source, receiver,
                frequencies=frequencies, n_modes=n_modes,
                **kwargs
            )
            if run_mode == RunMode.TIME_SERIES:
                return tf.synthesize_time_series(
                    source_waveform=source_waveform,
                    sample_rate=sample_rate,
                )
            return tf

        env = self._project_environment(env)
        self.validate_inputs(env, source, receiver, run_mode=run_mode)
        return self._compute_field_via_exe(
            env, source, receiver, n_modes=n_modes, **kwargs
        )

# ── field.exe pipeline ──────────────────────────────────────────────

    def _compute_field_via_exe(
        self, env, source, receiver,
        return_pressure=False, n_modes=None, frequencies=None,
        **kwargs,
    ):
        """Compute field using kraken.exe → field.exe AT pipeline.

        Parameters
        ----------
        return_pressure : bool
            If True, return complex pressure Field instead of TL.
        n_modes : int, optional
            Maximum number of modes to use during field reconstruction
            (mapped to ``MLimit`` in the FLP file).
        frequencies : ndarray, optional
            Broadband frequency vector. If given and multi-entry, kraken
            runs once with TopOpt(6)='B' producing a multi-freq .mod file
            that field.exe handles natively.
        """
        fm = self._setup_file_manager()
        self.file_manager = fm
        base_name = 'kfield'

        broadband = (
            frequencies is not None
            and len(np.atleast_1d(frequencies)) > 1
        )
        freq_vec = np.asarray(frequencies, dtype=float) if broadband else None

        try:
            is_rd = env.is_range_dependent
            segments = None
            profile_ranges_m = None
            n_profiles = 1

            if is_rd:
                # Segment environment for multi-profile kraken run
                segments = segment_environment_by_range(env, n_segments=self.n_segments)
                n_profiles = len(segments)

                # Max total depth must account for NMedia padding
                # (write_multi_profile_env pads profiles with fewer
                # media using 0.1 m layers, which can push total past
                # the deepest real profile).
                def _n_media_seg(seg_env):
                    n = 1
                    if seg_env.has_layered_bottom():
                        n += len(seg_env.bottom.layers)
                    return n

                max_n_media = max(_n_media_seg(seg) for _, seg in segments)
                if max_n_media < 2:
                    max_n_media = 2  # AT requires NMedia>=2 for RD
                max_total_depth = max(
                    self._total_media_depth(seg_env) + 0.1 * (max_n_media - _n_media_seg(seg_env))
                    for _, seg_env in segments
                )

                profile_ranges_m = np.array([s[0] for s in segments])
                self._log(f"Range-dependent: {n_profiles} profiles, "
                          f"mode_coupling={self.mode_coupling}")
            else:
                max_total_depth = self._total_media_depth(env)

            # 1. Write .env file
            env_file = fm.get_path(f'{base_name}.env')

            # Mode depths must cover the full ocean + sediment for all
            # profiles. Use max total media depth across all segments.
            n_mode_depths = max(100, int(max_total_depth * self.mode_points_per_meter))
            mode_depths = np.linspace(0, max_total_depth, n_mode_depths)
            receiver_for_modes = Receiver(depths=mode_depths, ranges=receiver.ranges)

            if is_rd and segments is not None:
                rmax_m = float(np.max(receiver.ranges))

                # Compute fixed mesh N for all profiles to ensure
                # consistent .mod record length (LRecordLength must not
                # increase between profiles — krakenc.f90 line 629).
                # Use 20 pts/wavelength based on max depth, like AT convention.
                freq = float(source.frequencies[0])
                all_c = []
                for _, seg in segments:
                    all_c.extend([float(c) for c in seg.ssp.data[:, 0]])
                    if seg.bottom is not None:
                        # Walk LayeredBottom via env helper — handles both
                        # halfspace and stratified columns.
                        all_c.append(seg.halfspace_at_range(0.0).sound_speed)
                min_c = min(all_c) if all_c else DEFAULT_SOUND_SPEED
                n_mesh_fixed = max(500, int(max_total_depth * freq / min_c * 20))

                if broadband:
                    # Multi-profile broadband not supported by
                    # write_multi_profile_env. Previous versions silently
                    # dropped the freq vector — that lost user output
                    # without signalling. Fail loudly instead.
                    raise ConfigurationError(
                        "KrakenField does not support range-dependent "
                        "broadband runs. Pass a single frequency or make "
                        "the environment range-independent."
                    )

                write_multi_profile_env(
                    filepath=env_file,
                    segments=segments,
                    source=source,
                    receiver=receiver_for_modes,
                    interp_ssp=self.interp_ssp,
                    n_mesh=n_mesh_fixed,
                    roughness=self.roughness,
                    c_low=self.c_low,
                    c_high=self.c_high,
                    rmax_m=rmax_m,
                )
            else:
                self._write_kraken_env(
                    env_file, env, source,
                    receiver_obj=receiver_for_modes,
                    receiver_depths=mode_depths,
                    rmax_m=float(np.max(receiver.ranges)),
                    frequencies=freq_vec if broadband else None,
                    **kwargs
                )

            # 2. Run kraken.exe → .mod (using base-class subprocess helper)
            kraken_exe = self._select_kraken_exe(env)
            self._log(f"Running {kraken_exe.name}...")
            try:
                self._run_subprocess(
                    [str(kraken_exe), base_name],
                    cwd=fm.work_dir,
                )
            except ModelExecutionError as exc:
                self._attach_prt_tail(exc, fm.work_dir, base_name)
                raise

            # 3. Write .flp file
            flp_file = fm.get_path(f'{base_name}.flp')
            option = self._build_field_option(is_rd)
            pos = {
                's': {'z': source.depths},
                'r': {'z': receiver.depths, 'r': receiver.ranges},
            }
            flp_kwargs = dict(
                title=getattr(env, 'name', ''),
                n_profiles=n_profiles,
                profile_ranges_km=(
                    profile_ranges_m / 1000.0 if profile_ranges_m is not None else None
                ),
            )
            if n_modes is not None:
                flp_kwargs['M_limit'] = int(n_modes)
            write_fieldflp(flp_file, option, pos, **flp_kwargs)

            # Copy source beam pattern file when requested. field.exe reads
            # <base>.sbp when Opt(3:3)='*' — we set that in
            # _build_field_option above.
            sbp_path = getattr(self, 'source_beam_pattern_file', None)
            if sbp_path is not None:
                src = Path(sbp_path)
                if not src.exists():
                    raise ConfigurationError(
                        f"source_beam_pattern_file not found: {src}"
                    )
                shutil.copy(src, fm.get_path(f'{base_name}.sbp'))

            # 4. Run field.exe → .shd
            # field.exe may exit non-zero on a successful run because of a
            # known Fortran teardown bug; the .shd file is still valid.
            # Catch only ``ModelExecutionError`` and warn so any wider
            # failure (e.g. ``FileNotFoundError`` for the binary itself)
            # still propagates.
            self._log(f"Running field.exe (option='{option}')...")
            try:
                self._run_subprocess(
                    [str(self._field_exe), base_name],
                    cwd=fm.work_dir,
                )
            except ModelExecutionError as exc:
                warnings.warn(
                    f"{self.model_name}: field.exe exited with non-zero "
                    f"status ({exc}); attempting to read the .shd output "
                    "anyway (known Fortran cleanup issue).",
                    UserWarning,
                    stacklevel=2,
                )

            # 5. Read .shd output
            # field.exe may exit with non-zero (memory cleanup issues in
            # Fortran) even when computation succeeds, so check for .shd first.
            shd_file = fm.get_path(f'{base_name}.shd')
            if not shd_file.exists():
                exc = ModelExecutionError(
                    self.model_name, return_code=0, stdout=None,
                    stderr=(
                        "field.exe did not produce a .shd file; check the "
                        f".prt log at {fm.get_path(base_name + '.prt')}"
                    ),
                )
                self._attach_prt_tail(exc, fm.work_dir, base_name)
                raise exc

            if broadband:
                shd0 = read_shd_bin(str(shd_file))
                freqs_read = np.asarray(shd0['freqVec'], dtype=float)
                # New layout: (n_d, n_r, n_f).
                p_stack = np.zeros(
                    (len(receiver.depths), len(receiver.ranges), len(freqs_read)),
                    dtype=np.complex128,
                )
                for i_freq, fr in enumerate(freqs_read):
                    shd_i = read_shd_bin(str(shd_file), freq=float(fr))
                    # field.exe (EvaluateMod.f90:34,42) emits the modal sum
                    # under the engineering carrier exp(-ikr) but with a
                    # leading factor i·√(2π)·exp(iπ/4); Scooter's Hankel
                    # path produces -exp(iπ/4)/√(2πr). The two prefactors
                    # differ by an overall -1 (NOT a conjugation), so a
                    # plain negation aligns the two travelling-wave fields.
                    p_stack[:, :, i_freq] = -shd_i['pressure'][0, 0, :, :]
                field = TransferFunction(
                    data=p_stack,
                    depths=receiver.depths,
                    ranges=receiver.ranges,
                    phase_reference='travelling_wave',
                    **self._result_kwargs(
                        source,
                        backend='field.exe',
                        frequencies=freqs_read,
                        mode_coupling=self.mode_coupling if is_rd else 'none',
                        n_profiles=n_profiles,
                        native_broadband=True,
                    ),
                )
            elif return_pressure:
                # Negate to match Scooter / Bellhop / RAM (see broadband
                # branch above for the prefactor algebra).
                shd_data = read_shd_bin(str(shd_file))
                p = -shd_data['pressure'][0, 0, :, :]  # (nrz, nrr)
                field = PressureField(
                    data=p,
                    depths=receiver.depths,
                    ranges=receiver.ranges,
                    **self._result_kwargs(
                        source,
                        backend='field.exe',
                        frequencies=float(np.atleast_1d(source.frequencies)[0]),
                        mode_coupling=self.mode_coupling if is_rd else 'none',
                        n_profiles=n_profiles,
                    ),
                )
            else:
                field = read_shd_file(shd_file)
                kw = self._result_kwargs(
                    source,
                    backend='field.exe',
                    frequencies=float(np.atleast_1d(source.frequencies)[0]),
                    mode_coupling=self.mode_coupling if is_rd else 'none',
                    n_profiles=n_profiles,
                )
                extras = kw.pop('metadata', {})
                field.tag(**kw, **extras)

            self._attach_output_paths(
                field, fm.work_dir, base_name,
                primary_files=(
                    ('shd_file', '.shd'),
                    ('mod_file', '.mod'),
                ),
            )

            self._log("KrakenField simulation complete")
            return field

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    # ── Broadband ───────────────────────────────────────────────────────

    def _compute_broadband_field(
        self, env, source, receiver,
        frequencies=None, n_modes=None, **kwargs,
    ):
        """
        Compute broadband transfer function.

        Issues ONE kraken.exe run with ``TopOpt(6)='B'`` and the broadband
        frequency vector. kraken writes a single multi-frequency .mod
        file, and field.exe handles every frequency natively (O(N) work
        instead of O(N) subprocess startups).

        Returns
        -------
        Field
            Transfer function field with shape ``(n_depths, n_freqs, n_ranges)``
            containing complex pressure. Axis order matches Bellhop/RAM/Scooter.
        """
        fc = float(source.frequencies[0])
        if frequencies is None:
            frequencies = np.linspace(fc * 0.5, fc * 2.0, 64)
        frequencies = np.asarray(frequencies, dtype=float)

        self._log(f"Broadband: {len(frequencies)} frequencies, "
                  f"{frequencies[0]:.1f}-{frequencies[-1]:.1f} Hz")

        env = self._project_environment(env)
        self.validate_inputs(env, source, receiver, run_mode=RunMode.BROADBAND)
        return self._compute_field_via_exe(
            env, source, receiver,
            frequencies=frequencies,
            n_modes=n_modes,
            **kwargs,
        )
