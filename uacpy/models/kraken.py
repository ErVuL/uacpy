"""
Kraken Normal Mode Suite - Unified Architecture

The Kraken suite provides normal mode modeling for underwater acoustics:
- **Kraken**: Normal mode computation (real arithmetic)
- **KrakenC**: Complex normal modes (elastic media, attenuation)
- **KrakenField**: Field computation from modes (TL, pressure)
- **KrakenField** with ``mode_coupling='coupled'``: Range-dependent coupled modes

This module follows the OASES architecture pattern where each executable
is wrapped by a separate class inheriting from a common base.

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
import subprocess
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
    TL_FLOOR_PRESSURE, TL_MAX_DB,
)
from uacpy.io.at_env_writer import ATEnvWriter
from uacpy.models.coupled_modes import segment_environment_by_range


class _KrakenBase(PropagationModel):
    """Base class for all Kraken models with shared functionality

    Parameters
    ----------
    c_low : float, optional
        Lower phase speed limit (m/s). None = auto (0.95 * min SSP speed).
    c_high : float, optional
        Upper phase speed limit (m/s). None = auto (1.05 * max of SSP and bottom speed).
    n_mesh : int, optional
        Number of mesh points per wavelength. 0 = auto. Default: 0.
    roughness : float, optional
        Bottom roughness (m). Default: 0.0.
    volume_attenuation : str, optional
        Volume attenuation formula: 'T' (Thorp), 'F' (Francois-Garrison),
        'B' (Biological), or None (no volume attenuation). Default: None.
    """

    def __init__(
        self,
        c_low: Optional[float] = None,
        c_high: Optional[float] = None,
        n_mesh: int = 0,
        roughness: float = 0.0,
        volume_attenuation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.c_low = c_low
        self.c_high = c_high
        self.n_mesh = n_mesh
        self.roughness = roughness
        self.volume_attenuation = volume_attenuation

    def _find_executable(self, name: str) -> Path:
        """Find Kraken executable in standard locations"""
        # Check in uacpy/bin/oalib
        bin_path = Path(__file__).parent.parent / 'bin' / 'oalib' / name
        if bin_path.exists():
            return bin_path

        # Check in third_party (development location)
        dev_path = Path(__file__).parent.parent / 'third_party' / 'Acoustics-Toolbox' / 'Kraken' / name
        if dev_path.exists():
            return dev_path

        # Check in PATH
        import shutil
        result = shutil.which(name)
        if result:
            return Path(result)

        raise FileNotFoundError(
            f"Could not find {name} executable.\n"
            "Please compile Acoustics Toolbox: cd uacpy && ./install.sh"
        )

    def _write_kraken_env(self, filepath, env, source, **kwargs):
        """
        Write Kraken environment file using shared ATEnvWriter

        Kraken has additional sections beyond the standard ENV format:
        - Phase speed limits (cLow, cHigh)
        - Maximum range (RMax)
        """
        # Parse types with backward compatibility
        ssp_type = parse_ssp_type(env.ssp_type)
        surface_type = parse_boundary_type(env.surface.acoustic_type)
        bottom_type = parse_boundary_type(env.bottom.acoustic_type)

        # Parse volume attenuation from instance attribute
        vol_atten = None
        if self.volume_attenuation:
            vol_atten = VolumeAttenuation.from_string(self.volume_attenuation)

        # Use instance attributes for model tuning parameters
        receiver_depths = kwargs.get('receiver_depths', [100.0])
        rmax_km = kwargs.get('rmax_km', 100.0)

        with open(filepath, 'w') as f:
            # Write standard ENV sections using ATEnvWriter
            ATEnvWriter.write_header(
                f, env, source,
                ssp_type=ssp_type,
                surface_type=surface_type,
                attenuation_unit=AttenuationUnits.DB_PER_WAVELENGTH,
                volume_attenuation=vol_atten
            )

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

            # Source depths (Kraken format: count, then one value per line)
            f.write(f"{len(source.depth)}\n")
            for sd in source.depth:
                f.write(f"{sd:.6f} /\n")

            # Receiver depths (Kraken format: count, then min/max or single value)
            f.write(f"{len(receiver_depths)}\n")
            if len(receiver_depths) == 1:
                f.write(f"{receiver_depths[0]:.6f} /\n")
            else:
                # Multiple depths: write as min max range
                f.write(f"{min(receiver_depths):.6f} {max(receiver_depths):.6f} /\n")

    def _run_kraken_executable(self, base_name: str, work_dir: Path):
        """Execute Kraken/KrakenC"""
        cmd = [str(self.executable), base_name]
        self._log(f"Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"Kraken failed: {result.stderr}")

    def _read_modes_file(self, filepath: Path) -> Dict:
        """Read Kraken .mod file using proper binary reader"""
        from uacpy.io.output_reader import read_modes_bin

        # Kraken outputs .mod files (binary format)
        # read_modes_bin expects filename WITHOUT extension (adds .moA internally)
        # But Kraken uses .mod, so we strip the extension
        filepath_str = str(filepath)
        if filepath_str.endswith('.mod'):
            basename = filepath_str[:-4]  # Remove .mod extension
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
        """Build error message for invalid mode files, checking .prt for clues."""
        prt_file = basename + '.prt'
        error_msg = "Kraken did not produce valid modes. "
        if os.path.exists(prt_file):
            with open(prt_file, 'r') as f:
                prt_content = f.read()
            if any(word in prt_content for word in ['ELASTIC', 'elastic', 'Elastic', 'shear', 'Shear']):
                error_msg += (
                    "Kraken (real arithmetic) cannot compute modes with elastic (shear-supporting) boundaries. "
                    "This requires KrakenC (complex mode version). "
                    "Solutions: (1) Use a fluid half-space bottom instead, or (2) use a different model like Bellhop, RAM, Scooter, or OAST."
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
        Field object with mode data in metadata:
        - 'k': modal wavenumbers
        - 'phi': mode shapes
        - 'z': depth grid for modes

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
            self.executable = self._find_executable('kraken.exe')
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
            Max number of modes (None = all)
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
                    receiver_depths=receiver.depths, **kwargs
                )

                # Run Kraken
                self._log("Running Kraken...", level='info')
                self._run_kraken_executable(base_name, fm.work_dir)

                # Read modes (read_modes expects basename without extension)
                modes_file = fm.get_path(base_name)  # Don't add .mod - read_modes handles it
                self._log(f"Reading mode file: {modes_file}.mod", level='info')
                modes = self._read_modes_file(modes_file)

                self._log("Simulation complete", level='info')

                # Return as Field
                return Field(
                    field_type='modes',
                    data=modes.get('phi', np.array([])),
                    depths=modes.get('z', np.array([])),
                    metadata={
                        'k': modes.get('k', np.array([])),
                        'phi': modes.get('phi', np.array([])),
                        'z': modes.get('z', np.array([])),
                        'frequency': float(source.frequency[0]),
                        'n_modes': len(modes.get('k', [])),
                    }
                )

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
            self.executable = self._find_executable('krakenc.exe')
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
            Max number of modes (None = all).
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
                    receiver_depths=receiver.depths, **kwargs
                )

                # Run KrakenC (uses krakenc.exe instead of kraken.exe)
                self._log("Running KrakenC...", level='info')
                self._run_kraken_executable(base_name, fm.work_dir)

                # Read modes (read_modes expects basename without extension)
                modes_file = fm.get_path(base_name)  # Don't add .mod - read_modes handles it
                self._log(f"Reading mode file: {modes_file}.mod", level='info')
                modes = self._read_modes_file(modes_file)

                self._log("Simulation complete", level='info')

                # Return as Field
                return Field(
                    field_type='modes',
                    data=modes.get('phi', np.array([])),
                    depths=modes.get('z', np.array([])),
                    metadata={
                        'k': modes.get('k', np.array([])),
                        'phi': modes.get('phi', np.array([])),
                        'z': modes.get('z', np.array([])),
                        'frequency': float(source.frequency[0]),
                        'n_modes': len(modes.get('k', [])),
                    }
                )

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

    Parameters
    ----------
    mode_coupling : str
        'adiabatic' or 'coupled' (default: 'adiabatic').
        Controls how field.exe handles range-dependent mode transitions.
    coherent : bool
        If True (default), coherent mode addition. If False, incoherent.
    n_segments : int
        Number of range segments for range-dependent scenarios (default: 10).
    mode_points_per_meter : float, optional
        Depth grid density for mode computation (default: 1.5 pts/m).

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

    def __init__(self, mode_points_per_meter: float = 1.5, mode_coupling: str = 'adiabatic',
                 coherent: bool = True, n_segments: int = 10, **kwargs):
        super().__init__(**kwargs)
        self._supported_modes = [RunMode.COHERENT_TL, RunMode.TIME_SERIES]
        self.executable = self._find_executable('kraken.exe')
        self._field_exe = self._find_executable('field.exe')

        self.mode_points_per_meter = mode_points_per_meter
        self.mode_coupling = mode_coupling
        self.coherent = coherent
        self.n_segments = n_segments

    def _build_field_option(self, is_range_dependent: bool) -> str:
        """Build the 4-character option string for field.exe."""
        pos1 = 'R'  # Point source (cylindrical spreading)
        if is_range_dependent:
            pos2 = 'C' if self.mode_coupling.lower() == 'coupled' else 'A'
        else:
            pos2 = ' '
        pos3 = ' '  # Omnidirectional (no beam pattern)
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
        if hasattr(env, 'bottom') and env.bottom is not None:
            if (hasattr(env.bottom, 'shear_speed')
                    and env.bottom.shear_speed is not None
                    and env.bottom.shear_speed > 0):
                return self._find_executable('krakenc.exe')
        return self.executable  # kraken.exe

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode=None,
        frequencies: Optional[np.ndarray] = None,
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
            COHERENT_TL (default) or TIME_SERIES.
        mode_coupling : str, optional
            'adiabatic' or 'coupled' (per-call override).
        n_segments : int, optional
            Number of range segments (per-call override).
        """
        with _resolve_overrides(self, c_low=c_low, c_high=c_high, n_mesh=n_mesh,
                                roughness=roughness, volume_attenuation=volume_attenuation,
                                mode_coupling=mode_coupling, n_segments=n_segments,
                                mode_points_per_meter=mode_points_per_meter):
            if run_mode is None:
                run_mode = RunMode.COHERENT_TL

            if run_mode == RunMode.TIME_SERIES:
                return self._compute_broadband_field(
                    env, source, receiver, frequencies=frequencies, **kwargs
                )

            self.validate_inputs(env, source, receiver)
            return self._compute_field_via_exe(env, source, receiver, **kwargs)

    # ── field.exe pipeline ──────────────────────────────────────────────

    def _compute_field_via_exe(self, env, source, receiver, return_pressure=False, **kwargs):
        """Compute field using kraken.exe → field.exe AT pipeline.

        Parameters
        ----------
        return_pressure : bool
            If True, return complex pressure Field instead of TL.
        """
        from uacpy.io.at_env_writer import ATEnvWriter
        from uacpy.io.flp_writer import write_fieldflp
        from uacpy.io.output_reader import read_shd_file, read_shd_bin

        fm = self._setup_file_manager()
        self.file_manager = fm
        base_name = 'kfield'

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
                    receiver_depths=mode_depths,
                    rmax_km=float(np.max(receiver.ranges)) / 1000.0,
                    **kwargs
                )

            # 2. Run kraken.exe → .mod
            kraken_exe = self._select_kraken_exe(env)
            self._log(f"Running {kraken_exe.name}...", level='info')
            result = subprocess.run(
                [str(kraken_exe), base_name],
                cwd=fm.work_dir, capture_output=True, text=True
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"{kraken_exe.name} failed (rc={result.returncode}): {result.stderr}"
                )

            # 3. Write .flp file
            flp_file = fm.get_path(f'{base_name}.flp')
            option = self._build_field_option(is_rd)
            pos = {
                's': {'z': source.depth},
                'r': {'z': receiver.depths, 'r': receiver.ranges},
            }
            write_fieldflp(
                flp_file, option, pos,
                title=getattr(env, 'name', ''),
                n_profiles=n_profiles,
                profile_ranges_km=profile_ranges_km,
            )

            # 4. Run field.exe → .shd
            self._log(f"Running field.exe (option='{option}')...", level='info')
            result = subprocess.run(
                [str(self._field_exe), base_name],
                cwd=fm.work_dir, capture_output=True, text=True
            )

            # 5. Read .shd output
            # field.exe may exit with non-zero (memory cleanup issues in
            # Fortran) even when computation succeeds, so check for .shd first.
            shd_file = fm.get_path(f'{base_name}.shd')
            if not shd_file.exists():
                raise RuntimeError(
                    f"field.exe failed (rc={result.returncode}): {result.stderr}"
                )

            if return_pressure:
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

    def _compute_broadband_field(self, env, source, receiver, frequencies=None, **kwargs):
        """
        Compute broadband transfer function by running field.exe at each frequency.

        Returns
        -------
        Field
            Transfer function field with data shape (n_depths, n_freqs, n_ranges)
            containing complex pressure.
        """
        fc = float(source.frequency[0])
        if frequencies is None:
            frequencies = np.linspace(fc * 0.5, fc * 2.0, 64)
        frequencies = np.asarray(frequencies, dtype=float)

        self._log(f"Broadband mode: {len(frequencies)} frequencies, "
                  f"{frequencies[0]:.1f}-{frequencies[-1]:.1f} Hz")

        n_depths = len(receiver.depths)
        n_ranges = len(receiver.ranges)
        n_freqs = len(frequencies)

        pressure = np.zeros((n_depths, n_freqs, n_ranges), dtype=np.complex128)

        for i_freq, freq in enumerate(frequencies):
            freq_source = Source(depth=source.depth[0], frequency=freq)
            try:
                field_result = self._compute_field_via_exe(
                    env, freq_source, receiver, return_pressure=True, **kwargs
                )
                pressure[:, i_freq, :] = field_result.data
            except Exception as e:
                self._log(f"  Freq {freq:.1f} Hz: failed ({e})", level='warn')
                continue

            if (i_freq + 1) % 10 == 0 or i_freq == n_freqs - 1:
                self._log(f"  Completed {i_freq+1}/{n_freqs} frequencies")

        return Field(
            field_type='transfer_function',
            data=pressure,
            ranges=receiver.ranges,
            depths=receiver.depths,
            frequencies=frequencies,
            metadata={
                'model': 'KrakenField',
                'nfreq': n_freqs,
                'center_frequency': fc,
            },
        )


