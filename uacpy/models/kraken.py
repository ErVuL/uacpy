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
modes = kraken.run(env, source, receiver)  # Returns Field with mode data

# Compute TL field from modes
field_model = KrakenField()
result = field_model.run(env, source, receiver)  # Returns Field with TL

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
import numpy as np
from pathlib import Path
from typing import Optional, Dict

from uacpy.models.base import PropagationModel, RunMode, _UNSET, _resolve_overrides
from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.field import Field
from uacpy.core.constants import (
    AttenuationUnits, VolumeAttenuation,
    parse_ssp_type, parse_boundary_type,
    C_LOW_FACTOR, C_HIGH_FACTOR,
)
from uacpy.core.exceptions import ConfigurationError
from uacpy.io.at_env_writer import ATEnvWriter
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
    volume_attenuation : str, optional
        Volume attenuation formula: 'T' (Thorp), 'F' (Francois-Garrison),
        'B' (Biological), or None (no volume attenuation). Default: None.
    francois_garrison_params : tuple, optional
        (T, S, pH, z_bar) required when ``volume_attenuation='F'``:
        temperature (degC), salinity (psu), pH, mean depth (m).
    bio_layers : list of tuples, optional
        Biological attenuation layers when ``volume_attenuation='B'``.
        Each entry is (Z1, Z2, f0, Q, a0): top depth (m), bottom depth (m),
        resonance frequency (Hz), quality factor, absorption coefficient.
    leaky_modes : bool, optional
        If True, override ``c_high`` to 1e9 so Kraken/KrakenC attempt to
        compute leaky modes (trapped modes with phase speeds above the
        halfspace P-wave speed). KrakenC is strongly recommended in this
        mode because it handles complex wavenumbers. See the Kraken doc:
        "CHIGH will attempt to compute leaky modes...". Default: False.
    """

    def __init__(
        self,
        c_low: Optional[float] = None,
        c_high: Optional[float] = None,
        n_mesh: int = 0,
        roughness: float = 0.0,
        volume_attenuation: Optional[str] = None,
        francois_garrison_params: Optional[tuple] = None,
        bio_layers: Optional[list] = None,
        leaky_modes: bool = False,
        top_reflection_file: Optional[Path] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.c_low = c_low
        self.c_high = c_high
        self.n_mesh = n_mesh
        self.roughness = roughness
        self.volume_attenuation = volume_attenuation
        self.francois_garrison_params = francois_garrison_params
        self.bio_layers = bio_layers
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

        # Inherits base validation (PropagationModel._validate_volume_attenuation_params)
        self._validate_volume_attenuation_params()

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

    def _check_kraken_ssp_type(self, env):
        """Reject SSP interpolation types kraken does not implement.

        Per AT ``sspMod.f90:61-89`` kraken accepts codes A (analytic),
        N (N^2-linear), C (C-linear), P (PCHIP), S (spline). The 'Q'
        quadrilateral code is Bellhop-only (see RangeDepSSPFile.htm).
        """
        ssp_type = getattr(env, 'ssp_type', None)
        if ssp_type is None:
            return
        val = str(ssp_type).lower()
        if val in ('q', 'quad', 'quadratic', 'ssptype.quadratic'):
            raise ConfigurationError(
                "Kraken/KrakenC/KrakenField do not support the 'Q' "
                "(quadrilateral/quadratic) SSP interpolation type — it is "
                "Bellhop-only. Use one of: 'c-linear', 'n2linear', 'pchip', "
                "'spline', or 'analytic'."
            )

    def _build_modes_field(self, modes, n_modes, source):
        """Clip modes to ``n_modes`` and wrap them in a Field.

        Shared by Kraken.run and KrakenC.run.
        """
        k_arr = modes.get('k', np.array([]))
        phi_arr = modes.get('phi', np.array([]))
        z_arr = modes.get('z', np.array([]))
        if n_modes is not None and len(k_arr) > n_modes:
            k_arr = k_arr[:n_modes]
            # read_modes_bin always returns phi with shape (NMat, M) so
            # we only need the 2-D clip; a 1-D branch would be dead.
            if phi_arr.ndim == 2 and phi_arr.shape[1] > n_modes:
                phi_arr = phi_arr[:, :n_modes]

        return Field(
            field_type='modes',
            data=phi_arr,
            depths=z_arr,
            metadata={
                'k': k_arr,
                'phi': phi_arr,
                'z': z_arr,
                'frequency': float(source.frequency[0]),
                'n_modes': len(k_arr),
                'n_modes_requested': n_modes,
                'leaky_modes': self.leaky_modes,
            },
        )

    @staticmethod
    def _compute_rmax_km(receiver, fallback_km: float = 100.0) -> float:
        """Derive field-computation RMax (km) from receiver ranges.

        Adds a 5 % buffer so field.exe doesn't clip the outermost ranges.
        Falls back to ``fallback_km`` if the receiver has no explicit range
        vector (e.g. mode-only Kraken runs).
        """
        if receiver is None:
            return float(fallback_km)
        ranges = getattr(receiver, 'ranges', None)
        if ranges is None or len(np.atleast_1d(ranges)) == 0:
            return float(fallback_km)
        rmax_m = float(np.max(np.asarray(ranges, dtype=float)))
        if rmax_m <= 0:
            return float(fallback_km)
        return rmax_m / 1000.0 * 1.05

    def _write_kraken_env(self, filepath, env, source, **kwargs):
        """
        Write Kraken environment file using shared ATEnvWriter

        Kraken has additional sections beyond the standard ENV format:
        - Phase speed limits (cLow, cHigh)
        - Maximum range (RMax)
        - Optional broadband frequency vector (TopOpt(6)='B')
        """
        # Reject Q/quad SSP type (Bellhop-only)
        self._check_kraken_ssp_type(env)
        # Re-validate in case caller mutated attributes after __init__
        self._validate_phase_speed_limits()
        self._validate_volume_attenuation_params()

        # Parse types (parse_* normalises string aliases like 'halfspace' vs 'half-space')
        ssp_type = parse_ssp_type(env.ssp_type)
        surface_type = parse_boundary_type(env.surface.acoustic_type)
        bottom_type = parse_boundary_type(env.bottom.acoustic_type)

        # Top reflection coefficient file (.trc): override surface BC to 'F'
        # and copy the user-supplied file next to the .env file. AT's
        # ReadEnvironmentBell.f90 reads <base>.trc when TopOpt(2:2)='F'.
        top_reflection_file = getattr(self, 'top_reflection_file', None)
        if top_reflection_file is not None:
            from uacpy.core.constants import BoundaryType
            import shutil
            src = Path(top_reflection_file)
            if not src.exists():
                raise ConfigurationError(
                    f"top_reflection_file not found: {src}"
                )
            surface_type = BoundaryType.FILE
            trc_dest = Path(filepath).with_suffix('.trc')
            shutil.copy(src, trc_dest)

        # Parse volume attenuation from instance attribute
        vol_atten = None
        if self.volume_attenuation:
            vol_atten = VolumeAttenuation.from_string(self.volume_attenuation)

        # Use instance attributes for model tuning parameters
        receiver_obj = kwargs.get('receiver_obj', None)
        receiver_depths = kwargs.get('receiver_depths', [100.0])
        rmax_km = kwargs.get('rmax_km', None)
        if rmax_km is None:
            rmax_km = self._compute_rmax_km(receiver_obj, fallback_km=100.0)
        frequencies = kwargs.get('frequencies', None)

        with open(filepath, 'w') as f:
            # Write standard ENV sections using ATEnvWriter
            ATEnvWriter.write_header(
                f, env, source,
                ssp_type=ssp_type,
                surface_type=surface_type,
                attenuation_unit=AttenuationUnits.DB_PER_WAVELENGTH,
                volume_attenuation=vol_atten,
                frequencies=frequencies,
            )

            # Francois-Garrison / Biological follow-up lines (after TopOpt)
            if vol_atten == VolumeAttenuation.FRANCOIS_GARRISON:
                ATEnvWriter.write_fg_params(f, self.francois_garrison_params)
            elif vol_atten == VolumeAttenuation.BIOLOGICAL:
                ATEnvWriter.write_bio_layers(f, self.bio_layers)

            ATEnvWriter.write_ssp_section(
                f, env, env.depth,
                n_mesh=self.n_mesh,
                roughness=self.roughness
            )

            # Write sediment layers if layered bottom
            ATEnvWriter.write_layer_sections(
                f, env, env.depth
            )

            ATEnvWriter.write_bottom_section(
                f, env,
                bottom_type=bottom_type
            )

            # KRAKEN-SPECIFIC SECTIONS

            # Phase speed limits (cLow, cHigh)
            c_min = min([c for _, c in env.ssp_data])
            c_max = max([c for _, c in env.ssp_data] + [env.bottom.sound_speed])
            c_low = self.c_low if self.c_low is not None else c_min * C_LOW_FACTOR
            c_high = self.c_high if self.c_high is not None else c_max * C_HIGH_FACTOR
            f.write(f"{c_low:.1f} {c_high:.1f}\n")

            # Maximum range (km) - used for field computation
            f.write(f"{rmax_km:.1f}\n")

            # Source depths (use ATEnvWriter helper for full non-uniform support)
            ATEnvWriter.write_source_depths(f, source)

            # Receiver depths (use full list via ATEnvWriter; receiver_obj has
            # priority so non-uniform arrays survive verbatim).
            if receiver_obj is not None:
                ATEnvWriter.write_receiver_depths(f, receiver_obj)
            else:
                rd = np.asarray(receiver_depths, dtype=float)
                f.write(f"{len(rd)}\n")
                depths_str = " ".join([f"{d:.6f}" for d in rd])
                f.write(f"{depths_str} /\n")

            # Broadband frequency vector (read by ReadfreqVec AFTER SD/RD)
            if frequencies is not None and len(np.atleast_1d(frequencies)) > 1:
                ATEnvWriter.write_broadband_freqs(f, np.asarray(frequencies))

    def _run_kraken_executable(self, base_name: str, work_dir: Path):
        """Execute Kraken/KrakenC via base-class subprocess runner."""
        self._run_subprocess(
            [str(self.executable), base_name],
            cwd=work_dir,
        )

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
            User-supplied mode sampling depths. If omitted, a 100-point
            linspace from 0 to total media depth is used.
        """
        from uacpy.core.receiver import Receiver as _Receiver

        override = kwargs.pop('mode_depth_grid', None)
        if override is not None:
            mode_depths = np.asarray(override, dtype=float)
        else:
            total_depth = env.depth
            if (
                hasattr(env, 'bottom_layered')
                and env.bottom_layered is not None
            ):
                for layer in env.bottom_layered.layers:
                    total_depth += layer.thickness
            mode_depths = np.linspace(0.0, float(total_depth), 100)

        dense_receiver = _Receiver(depths=mode_depths, ranges=[0.0])
        # Kraken/KrakenC.run ignore run_mode (they only compute modes); only
        # pass it if the concrete class advertises the kwarg (KrakenField).
        import inspect
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
        from uacpy.io.output_reader import read_modes_bin

        # read_modes_bin expects the filename without extension and appends
        # its own ('.moA'); strip '.mod' before handing it over.
        filepath_str = str(filepath)
        if filepath_str.endswith('.mod'):
            basename = filepath_str[:-4]
        else:
            basename = filepath_str

        # read_modes_bin will add .moA, but Kraken produces .mod
        # So we need to rename the file first
        import shutil
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
            raise RuntimeError(self._modes_error_message(basename))

        # Try to read modes, catch IndexError from reading empty binary files
        try:
            modes_data = read_modes_bin(basename, freq=0.0)
        except IndexError as e:
            raise RuntimeError(self._modes_error_message(basename, original_error=e))

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
        Path to kraken.exe. Auto-detected if None.
    use_tmpfs : bool
        Use tmpfs for I/O performance (Linux only)
    verbose : bool
        Enable verbose output
    work_dir : Path, optional
        Working directory for temporary files

    Returns
    -------
    Field
        Field object with mode data in metadata.

        Single-frequency runs (``Kraken.run``) expose:

        - 'k': modal wavenumbers (complex array, shape ``(M,)``)
        - 'phi': mode shapes, shape ``(nz, M)``
        - 'z': depth grid for modes
        - 'frequency': single frequency (Hz)
        - 'n_modes': number of returned modes

        These keys are NOT the same as the broadband keys produced by
        ``KrakenField`` with a frequency vector (which returns a
        ``field_type='transfer_function'`` field with ``frequencies``,
        ``depths``, ``ranges`` axes on ``.data`` and no 'k'/'phi' entries).

    Examples
    --------
    >>> kraken = Kraken()
    >>> modes = kraken.run(env, source, receiver)
    >>> print(f"Computed {len(modes.metadata['k'])} modes")
    """

    def __init__(self, executable: Optional[Path] = None, **kwargs):
        super().__init__(**kwargs)
        self._supported_modes = [RunMode.MODES]

        if executable is None:
            self.executable = self._find_executable_in_paths(
                'kraken.exe',
                bin_subdirs='oalib',
                dev_subdir='Acoustics-Toolbox/Kraken',
            )
        else:
            self.executable = Path(executable)

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        n_modes: Optional[int] = None,
        c_low=_UNSET,
        c_high=_UNSET,
        n_mesh=_UNSET,
        roughness=_UNSET,
        volume_attenuation=_UNSET,
        **kwargs
    ) -> Field:
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
        c_low, c_high, n_mesh, roughness, volume_attenuation : optional
            Per-call overrides for constructor defaults.

        Returns
        -------
        Field
            Mode data in metadata dict
        """
        self.validate_inputs(env, source, receiver)

        if env.is_range_dependent:
            raise ValueError(
                "Kraken does not support range-dependent environments for mode computation.\n"
                "Use KrakenField for range-dependent scenarios."
            )

        with _resolve_overrides(self, c_low=c_low, c_high=c_high, n_mesh=n_mesh,
                                roughness=roughness, volume_attenuation=volume_attenuation):
            fm = self._setup_file_manager()
            base_name = 'modes'

            try:
                # Write environment file
                env_file = fm.get_path(f'{base_name}.env')
                self._log(f"Writing environment file: {env_file}", level='info')
                self._write_kraken_env(
                    env_file, env, source,
                    receiver_obj=receiver,
                    receiver_depths=receiver.depths,
                    **kwargs,
                )

                # Run Kraken
                self._log("Running Kraken...", level='info')
                self._run_kraken_executable(base_name, fm.work_dir)

                # Read modes (read_modes expects basename without extension)
                modes_file = fm.get_path(base_name)  # Don't add .mod - read_modes handles it
                self._log(f"Reading mode file: {modes_file}.mod", level='info')
                modes = self._read_modes_file(modes_file)

                self._log("Simulation complete", level='info')

                return self._build_modes_field(modes, n_modes, source)

            finally:
                if not self.work_dir:
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
    Same as Kraken

    Examples
    --------
    >>> krakenc = KrakenC()
    >>> modes = krakenc.run(env_with_elastic_bottom, source, receiver)
    """

    def __init__(self, executable: Optional[Path] = None, **kwargs):
        super().__init__(**kwargs)
        self._supported_modes = [RunMode.MODES]

        if executable is None:
            self.executable = self._find_executable_in_paths(
                'krakenc.exe',
                bin_subdirs='oalib',
                dev_subdir='Acoustics-Toolbox/Kraken',
            )
        else:
            self.executable = Path(executable)

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        n_modes: Optional[int] = None,
        c_low=_UNSET,
        c_high=_UNSET,
        n_mesh=_UNSET,
        roughness=_UNSET,
        volume_attenuation=_UNSET,
        **kwargs
    ) -> Field:
        """
        Compute complex normal modes.

        Uses complex arithmetic, which is required for environments with
        elastic (solid) boundaries that support shear waves, or with
        significant volume attenuation.

        Parameters
        ----------
        env : Environment
            Must be range-independent.
        source : Source
            Used for frequency.
        receiver : Receiver
            Used for depth grid.
        n_modes : int, optional
            Maximum number of modes to return. KrakenC itself has no mode-
            count cap; when ``n_modes`` is provided the returned ``Field``
            is clipped to at most that many modes. See ``Kraken.run``.
        c_low, c_high, n_mesh, roughness, volume_attenuation : optional
            Per-call overrides for constructor defaults.

        Returns
        -------
        Field
            Field with field_type='modes'. Mode data (wavenumbers, mode shapes)
            available in metadata dict with keys 'k', 'phi', 'z', 'frequency',
            'n_modes'.
        """
        self.validate_inputs(env, source, receiver)

        if env.is_range_dependent:
            raise ValueError("KrakenC requires range-independent environment")

        with _resolve_overrides(self, c_low=c_low, c_high=c_high, n_mesh=n_mesh,
                                roughness=roughness, volume_attenuation=volume_attenuation):
            fm = self._setup_file_manager()
            base_name = 'modes'

            try:
                # Write environment file (same format as Kraken)
                env_file = fm.get_path(f'{base_name}.env')
                self._log(f"Writing environment file: {env_file}", level='info')
                self._write_kraken_env(
                    env_file, env, source,
                    receiver_obj=receiver,
                    receiver_depths=receiver.depths,
                    **kwargs,
                )

                # Run KrakenC (uses krakenc.exe instead of kraken.exe)
                self._log("Running KrakenC...", level='info')
                self._run_kraken_executable(base_name, fm.work_dir)

                # Read modes (read_modes expects basename without extension)
                modes_file = fm.get_path(base_name)  # Don't add .mod - read_modes handles it
                self._log(f"Reading mode file: {modes_file}.mod", level='info')
                modes = self._read_modes_file(modes_file)

                self._log("Simulation complete", level='info')

                return self._build_modes_field(modes, n_modes, source)

            finally:
                if not self.work_dir:
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
    mode_coupling : str
        'adiabatic' or 'coupled' (default: 'adiabatic').
        Controls how field.exe handles range-dependent mode transitions.
    coherent : bool
        If True (default), coherent mode addition. If False, incoherent.
        Note: ``coupled`` + ``coherent=False`` is rejected up front — AT's
        field.exe does not support coupled-incoherent calculations.
    n_segments : int
        Number of range segments for range-dependent scenarios (default: 10).
    mode_points_per_meter : float, optional
        Depth grid density for mode computation (default: 1.5 pts/m). Used
        in both range-dependent and range-independent paths to size the
        internal mode-depth grid.
    source_beam_pattern_file : Path, optional
        Path to a Bellhop-style .sbp source beam pattern file. When set,
        the file is copied to the work dir and field.exe is invoked with
        Opt(3:3)='*' so it applies the beam pattern.
    source_type : str, optional
        field.exe option column 1 (AT ``field.f90:71-79``):

        * 'R' (default) — cylindrical point source (pressure, 2-D),
        * 'X' — Cartesian line source (2-D),
        * 'S' — scaled-cylindrical point source.

    Note
    ----
    Field.exe Opt(3:3) only accepts ``'*'``, ``'O'``, or ``' '`` (see
    ``field.f90:83-90`` — anything else raises FATAL ERROR). For
    acoustic-only environments the default ``' '`` is fine — ``Comp`` is
    only consulted inside the ELASTIC branch of ``EXTRACT`` in
    ``ReadModes.f90:315-324``. Purely elastic component selection
    (H/V/T/N) is not reachable through field.exe — a limitation of the
    upstream Fortran, not uacpy.

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
            RunMode.TIME_SERIES,
            RunMode.TRANSFER_FUNCTION,
        ]

        if mode_coupling not in ('adiabatic', 'coupled'):
            raise ValueError(
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

        if field_executable is None:
            self._field_exe = self._find_executable_in_paths(
                'field.exe',
                bin_subdirs='oalib',
                dev_subdir='Acoustics-Toolbox/Kraken',
            )
        else:
            self._field_exe = Path(field_executable)

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
        if hasattr(env, 'bottom_layered') and env.bottom_layered is not None:
            for layer in env.bottom_layered.layers:
                depth += layer.thickness
        return depth

    def _select_kraken_exe(self, env):
        """Return 'kraken.exe' or 'krakenc.exe' based on environment."""
        needs_krakenc = False
        if hasattr(env, 'bottom') and env.bottom is not None:
            if (hasattr(env.bottom, 'shear_speed')
                    and env.bottom.shear_speed is not None
                    and env.bottom.shear_speed > 0):
                needs_krakenc = True
        # leaky_modes requires complex arithmetic for convergence.
        if getattr(self, 'leaky_modes', False):
            needs_krakenc = True
        if needs_krakenc:
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
        c_low=_UNSET,
        c_high=_UNSET,
        n_mesh=_UNSET,
        roughness=_UNSET,
        volume_attenuation=_UNSET,
        mode_coupling=_UNSET,
        n_segments=_UNSET,
        mode_points_per_meter=_UNSET,
        **kwargs
    ) -> Field:
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
            COHERENT_TL (default), TIME_SERIES, or TRANSFER_FUNCTION.
        frequencies : ndarray, optional
            Frequency vector (Hz) for native broadband computation. Uses
            TopOpt(6)='B' so kraken writes one multi-frequency .mod file
            and field.exe handles all frequencies in a single pass.
        n_modes : int, optional
            Max number of modes used by field.exe (FLP ``MLimit``).
        mode_coupling : str, optional
            'adiabatic' or 'coupled' (per-call override).
        n_segments : int, optional
            Number of range segments (per-call override).
        """
        if mode_coupling is not _UNSET and mode_coupling not in (
            'adiabatic', 'coupled'
        ):
            raise ValueError(
                f"mode_coupling must be 'adiabatic' or 'coupled', "
                f"got {mode_coupling!r}"
            )

        with _resolve_overrides(
            self, c_low=c_low, c_high=c_high, n_mesh=n_mesh,
            roughness=roughness, volume_attenuation=volume_attenuation,
            mode_coupling=mode_coupling, n_segments=n_segments,
            mode_points_per_meter=mode_points_per_meter,
        ):
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

            if run_mode is None:
                # Default run mode: TRANSFER_FUNCTION if a freq vector is
                # provided, else single-frequency coherent TL.
                if frequencies is not None and len(np.atleast_1d(frequencies)) > 1:
                    run_mode = RunMode.TRANSFER_FUNCTION
                else:
                    run_mode = RunMode.COHERENT_TL

            if run_mode in (RunMode.TIME_SERIES, RunMode.TRANSFER_FUNCTION):
                return self._compute_broadband_field(
                    env, source, receiver,
                    frequencies=frequencies, n_modes=n_modes,
                    **kwargs
                )

            self.validate_inputs(env, source, receiver)
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
        from uacpy.io.at_env_writer import ATEnvWriter
        from uacpy.io.flp_writer import write_fieldflp
        from uacpy.io.output_reader import read_shd_file, read_shd_bin

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
            profile_ranges_km = None
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
                    if hasattr(seg_env, 'bottom_layered') and seg_env.bottom_layered is not None:
                        n += len(seg_env.bottom_layered.layers)
                    return n

                max_n_media = max(_n_media_seg(seg) for _, seg in segments)
                if max_n_media < 2:
                    max_n_media = 2  # AT requires NMedia>=2 for RD
                max_total_depth = max(
                    self._total_media_depth(seg_env) + 0.1 * (max_n_media - _n_media_seg(seg_env))
                    for _, seg_env in segments
                )

                profile_ranges_km = np.array([s[0] for s in segments])
                self._log(f"Range-dependent: {n_profiles} profiles, "
                          f"mode_coupling={self.mode_coupling}", level='info')
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
                rmax_km = float(np.max(receiver.ranges)) / 1000.0

                # Compute fixed mesh N for all profiles to ensure
                # consistent .mod record length (LRecordLength must not
                # increase between profiles — krakenc.f90 line 629).
                # Use 20 pts/wavelength based on max depth, like AT convention.
                freq = float(source.frequency[0])
                all_c = []
                for _, seg in segments:
                    all_c.extend([c for _, c in seg.ssp_data])
                    if hasattr(seg, 'bottom') and seg.bottom is not None:
                        all_c.append(seg.bottom.sound_speed)
                min_c = min(all_c) if all_c else 1500.0
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

                ATEnvWriter.write_multi_profile_env(
                    filepath=env_file,
                    segments=segments,
                    source=source,
                    receiver=receiver_for_modes,
                    volume_attenuation=None,
                    n_mesh=n_mesh_fixed,
                    roughness=self.roughness,
                    c_low=self.c_low,
                    c_high=self.c_high,
                    rmax_km=rmax_km,
                )
            else:
                self._write_kraken_env(
                    env_file, env, source,
                    receiver_obj=receiver_for_modes,
                    receiver_depths=mode_depths,
                    rmax_km=float(np.max(receiver.ranges)) / 1000.0,
                    frequencies=freq_vec if broadband else None,
                    **kwargs
                )

            # 2. Run kraken.exe → .mod (using base-class subprocess helper)
            kraken_exe = self._select_kraken_exe(env)
            self._log(f"Running {kraken_exe.name}...", level='info')
            self._run_subprocess(
                [str(kraken_exe), base_name],
                cwd=fm.work_dir,
            )

            # 3. Write .flp file
            flp_file = fm.get_path(f'{base_name}.flp')
            option = self._build_field_option(is_rd)
            pos = {
                's': {'z': source.depth},
                'r': {'z': receiver.depths, 'r': receiver.ranges},
            }
            flp_kwargs = dict(
                title=getattr(env, 'name', ''),
                n_profiles=n_profiles,
                profile_ranges_km=profile_ranges_km,
            )
            if n_modes is not None:
                flp_kwargs['M_limit'] = int(n_modes)
            write_fieldflp(flp_file, option, pos, **flp_kwargs)

            # Copy source beam pattern file when requested. field.exe reads
            # <base>.sbp when Opt(3:3)='*' — we set that in
            # _build_field_option above.
            sbp_path = getattr(self, 'source_beam_pattern_file', None)
            if sbp_path is not None:
                import shutil
                src = Path(sbp_path)
                if not src.exists():
                    raise ConfigurationError(
                        f"source_beam_pattern_file not found: {src}"
                    )
                shutil.copy(src, fm.get_path(f'{base_name}.sbp'))

            # 4. Run field.exe → .shd
            # field.exe may exit with non-zero (Fortran cleanup issues) even
            # when computation succeeds, so we inspect the .shd file below
            # rather than trusting the return code.
            self._log(f"Running field.exe (option='{option}')...", level='info')
            try:
                self._run_subprocess(
                    [str(self._field_exe), base_name],
                    cwd=fm.work_dir,
                )
            except Exception:
                # Swallow — checked via .shd existence below.
                pass

            # 5. Read .shd output
            # field.exe may exit with non-zero (memory cleanup issues in
            # Fortran) even when computation succeeds, so check for .shd first.
            shd_file = fm.get_path(f'{base_name}.shd')
            if not shd_file.exists():
                raise RuntimeError(
                    "field.exe did not produce a .shd file; check the "
                    f".prt log at {fm.get_path(base_name + '.prt')}"
                )

            if broadband:
                # Broadband runs currently only support ONE source depth.
                # read_shd_bin returns pressure shape (Ntheta, Nsz, Nrz, Nrr);
                # collapsing on Nsz via ``[0, 0]`` would silently drop the
                # other source depths. Fail loudly instead of producing
                # a misleading single-source-depth transfer function.
                sd = np.atleast_1d(source.depth)
                if len(sd) > 1:
                    raise ConfigurationError(
                        "KrakenField broadband/transfer-function mode "
                        "currently supports only a single source depth "
                        f"(got {len(sd)}). Loop over source depths "
                        "externally, or use a single depth."
                    )

                shd0 = read_shd_bin(str(shd_file))
                freqs_read = np.asarray(shd0['freqVec'], dtype=float)
                p_stack = np.zeros(
                    (len(receiver.depths), len(freqs_read), len(receiver.ranges)),
                    dtype=np.complex128,
                )
                for i_freq, fr in enumerate(freqs_read):
                    shd_i = read_shd_bin(str(shd_file), freq=float(fr))
                    p_stack[:, i_freq, :] = -shd_i['pressure'][0, 0, :, :]
                field = Field(
                    field_type='transfer_function',
                    data=p_stack,
                    depths=receiver.depths,
                    ranges=receiver.ranges,
                    frequencies=freqs_read,
                    metadata={
                        'model': 'KrakenField',
                        'mode_coupling': self.mode_coupling if is_rd else 'none',
                        'n_profiles': n_profiles,
                        'backend': 'field.exe',
                        'native_broadband': True,
                    },
                )
            elif return_pressure:
                # Return complex pressure for broadband/transfer function use.
                # field.exe uses exp(-ikr) with a factor i*sqrt(2pi)*exp(i*pi/4)
                # which produces pressure with opposite sign vs. Scooter's
                # Hankel transform convention. Negate to match the sign used
                # by Scooter, Bellhop, and RAM for consistent time-series.
                shd_data = read_shd_bin(str(shd_file))
                p = -shd_data['pressure'][0, 0, :, :]  # (nrz, nrr)
                field = Field(
                    field_type='pressure',
                    data=p,
                    depths=receiver.depths,
                    ranges=receiver.ranges,
                    frequencies=source.frequency,
                    metadata={
                        'model': 'KrakenField',
                        'mode_coupling': self.mode_coupling if is_rd else 'none',
                        'n_profiles': n_profiles,
                        'backend': 'field.exe',
                    },
                )
            else:
                field = read_shd_file(shd_file)
                field.metadata['model'] = 'KrakenField'
                field.metadata['mode_coupling'] = self.mode_coupling if is_rd else 'none'
                field.metadata['n_profiles'] = n_profiles
                field.metadata['backend'] = 'field.exe'

            self._log("KrakenField simulation complete", level='info')
            return field

        finally:
            if not self.work_dir:
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
        fc = float(source.frequency[0])
        if frequencies is None:
            frequencies = np.linspace(fc * 0.5, fc * 2.0, 64)
        frequencies = np.asarray(frequencies, dtype=float)

        self._log(f"Broadband mode: {len(frequencies)} frequencies, "
                  f"{frequencies[0]:.1f}-{frequencies[-1]:.1f} Hz")

        self.validate_inputs(env, source, receiver)
        return self._compute_field_via_exe(
            env, source, receiver,
            frequencies=frequencies,
            n_modes=n_modes,
            **kwargs,
        )


