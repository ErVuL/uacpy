"""
Scooter finite-element FFP (Fast Field Program) model.

Computes the acoustic field in the frequency-wavenumber domain using a
finite-element discretization, then transforms ``.grn`` to a range-domain
TL field via the in-tree Python Hankel transform in
:mod:`uacpy.io.grn_reader`. Supports coherent TL, broadband ``H(f)``,
and broadband time-series output.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np

from uacpy.core.exceptions import (
    ConfigurationError, ExecutableNotFoundError, ModelExecutionError,
)
from uacpy.models.base import PropagationModel, RunMode
from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.results import Result
from uacpy.core.constants import parse_boundary_type
from uacpy.io.grn_reader import read_grn_file, grn_to_field, grn_to_transfer_function
from uacpy.io.oalib_writer import (
    write_absorption_block, write_bottom_section, write_broadband_freqs,
    write_header, write_layer_sections,
    write_phase_speed_and_rmax, write_receiver_depths, write_source_depths,
    write_ssp_section,
)


class Scooter(PropagationModel):
    """
    Scooter finite element FFP (Fast Field Program) model

    Frequency-domain solver for underwater acoustics.
    Developed by Michael B. Porter.

    Parameters
    ----------
    executable : Path, optional
        Path to ``scooter.exe``. Auto-detected if ``None``.
    c_low, c_high : float, optional
        Phase-speed bounds (m/s). ``None`` ⇒ ``0.95 × min SSP`` /
        ``1.05 × max SSP+bottom``.
    n_mesh : int, optional
        Mesh points per medium. ``0`` ⇒ auto. Default ``0``.
    roughness : float, optional
        Bottom RMS roughness (m). Default ``0``.
    rmax_multiplier : float, optional
        Padding for k-resolution; Scooter's spectral RMax becomes
        ``receiver.range.max() * rmax_multiplier``. Default ``2.0``.
    source_type : str, optional
        FLP Opt(1): ``'R'`` cylindrical (default) | ``'X'`` Cartesian.
    spectrum : str, optional
        FLP Opt(2): ``'positive'`` (fast, default) | ``'negative'`` | ``'both'``.
    stabilizing_attenuation_off : bool, optional
        Disable Scooter's stabilising attenuation. Default ``False``;
        leave it unless you know what you're doing (the stabiliser
        prevents pole-on-contour blow-ups).
    field_interp : str, optional
        FLP Opt(3): ``'O'`` polynomial (default) | ``'P'`` Padé.
    use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
        Standard plumbing (see :class:`PropagationModel`).

    Notes
    -----
    Range-independent FFP — single spectral solve over the full
    wavenumber axis, Hankel-transformed to range. Supports
    ``LayeredBottom`` and elastic bottoms natively. The Green's-function
    ``.grn`` is converted to range-domain TL via the in-tree Python
    Hankel transform (``uacpy.io.grn_reader``).

    **Collapse defaults (overrides of :data:`DEFAULT_COLLAPSE`).**
    Per-model: ``'ssp': 'mean'``, ``'bottom': 'median'``,
    ``'rd_layered_layers': 'preserve'`` (Scooter consumes
    ``LayeredBottom`` natively).

    Defaults auto-derived at ``run()`` time:

    - ``c_low=None`` → ``min(env.ssp) × 0.95``
    - ``c_high=None`` → ``max(max(env.ssp), env.bottom.sound_speed) × 1.05``
    - Spectral ``RMax = receiver.range_max × rmax_multiplier``
    - ``n_mesh=0`` → Scooter picks from frequency / wavelength.
    - TopOpt position 4 reads ``env.absorption``.

    With ``verbose='info'`` the resolved ``c_low`` / ``c_high`` are logged.

    Examples
    --------
    >>> scooter = Scooter()
    >>> result = scooter.run(env, source, receiver)
    """

    def __init__(
        self,
        executable: Optional[Path] = None,
        c_low: Optional[float] = None,
        c_high: Optional[float] = None,
        n_mesh: int = 0,
        roughness: float = 0.0,
        rmax_multiplier: float = 2.0,
        interp_ssp: Optional[str] = None,
        source_type: str = 'R',
        spectrum: str = 'positive',
        stabilizing_attenuation_off: bool = False,
        field_interp: str = 'O',
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        executable : Path, optional
            Path to scooter executable. Auto-detected if None.
        c_low : float, optional
            Lower phase speed limit (m/s). None = auto (0.95 * min SSP speed).
        c_high : float, optional
            Upper phase speed limit (m/s). None = auto (1.05 * max of SSP and bottom speed).
        n_mesh : int, optional
            Mesh points per wavelength. 0 = auto. Default: 0.
        roughness : float, optional
            Bottom roughness (m). Default: 0.0.
        rmax_multiplier : float, optional
            Multiply max receiver range for wavenumber resolution. Default: 2.0.
        source_type : {'R', 'X'}, optional
            FLP Option(1:1). 'R' = cylindrical (point source, default),
            'X' = Cartesian (line source). The in-tree Hankel transform
            scales by ``√k`` and ``1/√(2πr)`` for 'R' and by
            ``1/√(2π)`` for 'X'.
        spectrum : {'positive', 'negative', 'both'}, optional
            FLP Option(2:2). 'positive' (default) uses only the positive
            wavenumber spectrum (fast, recommended). 'negative' uses only
            the negative branch; 'both' integrates along the full k-axis.
        stabilizing_attenuation_off : bool, optional
            If True, writes ``'0'`` at TopOpt position 7. Scooter then
            sets its stabilising attenuation to zero (see
            ``scooter.f90:81,129``). Leave False (default) unless you
            know what you're doing — the stabiliser is there to prevent
            pole-on-contour blow-ups.
        field_interp : {'O', 'P'}, optional
            FLP Option(3:3). 'O' = polynomial interpolation (default,
            what Scooter's own sample FLP uses), 'P' = Pade.
        """
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
            **kwargs,
        )

        # Range-independent FFP — single spectral solve over the full
        # wavenumber axis, Hankel-transformed to range. Median/mean
        # samples are the representative single profile.
        self._set_collapse_defaults({
            'ssp': 'mean',
            'bottom': 'median',
            'rd_layered_layers': 'preserve',
        })

        self.c_low = c_low
        self.c_high = c_high
        self.interp_ssp = interp_ssp
        if c_low is not None and c_high is not None and c_low >= c_high:
            raise ConfigurationError(
                f"Scooter spectral phase-velocity band requires "
                f"c_low < c_high; got c_low={c_low} m/s, c_high={c_high} m/s."
            )
        self.n_mesh = n_mesh
        self.roughness = roughness
        self.rmax_multiplier = rmax_multiplier

        if source_type not in ('R', 'X'):
            raise ConfigurationError(
                f"Invalid source_type '{source_type}'. Use 'R' (cylindrical) or 'X' (Cartesian)."
            )
        self.source_type = source_type

        spectrum_map = {'positive': 'P', 'negative': 'N', 'both': 'B'}
        if spectrum not in spectrum_map:
            raise ConfigurationError(
                f"Invalid spectrum '{spectrum}'. Use 'positive', 'negative', or 'both'."
            )
        self.spectrum = spectrum
        self._spectrum_code = spectrum_map[spectrum]

        self.stabilizing_attenuation_off = bool(stabilizing_attenuation_off)

        if field_interp not in ('O', 'P'):
            raise ConfigurationError(
                f"Invalid field_interp '{field_interp}'. Use 'O' (polynomial) "
                f"or 'P' (Pade) — see fields.f90:82-90."
            )
        self.field_interp = field_interp

        # Declare supported modes for Scooter.
        # INCOHERENT_TL is NOT implemented — Scooter computes the full
        # coherent field; incoherent TL would require a modal decomposition
        # we don't have here. See run() for the supported branches.
        self._supported_modes = [
            RunMode.COHERENT_TL,
            RunMode.BROADBAND,
            RunMode.TIME_SERIES,
        ]
        # Scooter: range-independent wavenumber integration. Honors
        # multi-layer fluid/elastic bottom natively. Range dependence in
        # any form is collapsed to range-0 with a warning.
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
                'scooter.exe', bin_subdirs='oalib',
                dev_subdir='Acoustics-Toolbox/Scooter',
            )
        else:
            self.executable = Path(executable)

        if not self.executable.exists():
            raise ExecutableNotFoundError('Scooter', str(self.executable))

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode=None,
        frequencies: Optional['np.ndarray'] = None,
        source_waveform=None,
        sample_rate=None,
        **kwargs
    ) -> Result:
        """
        Run Scooter simulation

        Parameters
        ----------
        env : Environment
            Ocean environment (range-dependent environments will be approximated)
        source : Source
            Acoustic source
        receiver : Receiver
            Receiver array
        run_mode : RunMode, optional
            ``COHERENT_TL`` (default) — single-frequency TL.
            ``BROADBAND`` — broadband H(f).
            ``TIME_SERIES`` — real pressure p(t); requires
            ``source_waveform`` + ``sample_rate``.
        frequencies : ndarray, optional
            Frequency vector for BROADBAND/TIME_SERIES. If not provided,
            a default vector spanning fc/2 to 2*fc is generated.
        source_waveform : ndarray, optional
            Source pulse for ``TIME_SERIES`` mode.
        sample_rate : float, optional
            Sampling rate of ``source_waveform`` in Hz.
        **kwargs
            Additional Scooter parameters

        Returns
        -------
        result : Result
            :class:`Field` — narrowband complex pressure for COHERENT_TL,
            broadband complex ``H(f)`` for BROADBAND, real ``p(d, r, t)``
            for TIME_SERIES.
        """
        run_mode = self._resolve_run_mode(run_mode)

        if run_mode == RunMode.TIME_SERIES and (
            source_waveform is None or sample_rate is None
        ):
            raise ConfigurationError(
                "Scooter.run(run_mode=TIME_SERIES) requires source_waveform "
                "and sample_rate. For the broadband transfer function "
                "H(f), use run_mode=RunMode.BROADBAND."
            )

        env = self._project_environment(env)

        # Clip receiver depths to environment depth (with safety margin)
        receiver = self._clip_receiver_depths(receiver, env.depth)

        self.validate_inputs(env, source, receiver, run_mode=run_mode)

        # Broadband mode (BROADBAND or TIME_SERIES) requires a
        # frequency vector. The in-tree Python Hankel transform handles
        # the multi-frequency Green's-function output; fields.exe is
        # not used.
        broadband_freqs = None
        broadband_mode = run_mode in (RunMode.BROADBAND, RunMode.TIME_SERIES)
        if broadband_mode:
            if frequencies is not None:
                broadband_freqs = np.asarray(frequencies, dtype=float)
            else:
                fc = float(source.frequencies[0])
                broadband_freqs = np.linspace(fc * 0.5, fc * 2.0, 64)
            self._log(f"Broadband: {len(broadband_freqs)} frequencies, "
                      f"{broadband_freqs[0]:.1f}-{broadband_freqs[-1]:.1f} Hz")

        fm = self._setup_file_manager()
        self.file_manager = fm

        try:
            base_name = 'model'

            env_file = fm.get_path(f'{base_name}.env')
            self._log(f"Writing environment file: {env_file}")

            self._write_scooter_env(
                env_file, env, source, receiver,
                frequencies=broadband_freqs,
                **kwargs
            )

            self._log("Running...")
            self._run_scooter(base_name, fm.work_dir)

            grn_file = fm.get_path(f'{base_name}.grn')
            if not grn_file.exists():
                exc = ModelExecutionError(
                    self.model_name, return_code=0, stdout=None,
                    stderr=(
                        f"Scooter did not produce {grn_file}; "
                        f"check {fm.work_dir}/{base_name}.prt for diagnostics."
                    ),
                )
                self._attach_prt_tail(exc, fm.work_dir, base_name)
                raise exc

            self._log("Reading Green's function...")
            grn_data = read_grn_file(grn_file)

            if grn_data['nk'] == 0:
                exc = ModelExecutionError(
                    self.model_name, return_code=0,
                    stdout=None,
                    stderr="Scooter produced empty Green's function (nk=0)",
                )
                self._attach_prt_tail(exc, fm.work_dir, base_name)
                raise exc

            transform_kwargs = dict(
                source_type=self.source_type,
                spectrum=self._spectrum_code,
            )
            if broadband_mode:
                self._log(f"Transforming {grn_data['nfreq']} frequencies to range domain...")
                result = grn_to_transfer_function(
                    grn_data, receiver.ranges, **transform_kwargs,
                )
            else:
                self._log("Transforming to range domain (FFT-based Hankel transform)...")
                result = grn_to_field(
                    grn_data, receiver.ranges, method='fft_hankel',
                    **transform_kwargs,
                )
            result.model = self.model_name
            result.backend = 'scooter.exe'
            result.source_depths = np.atleast_1d(np.asarray(source.depths, dtype=float))
            freqs = broadband_freqs if broadband_mode else float(source.frequencies[0])
            result.frequencies = np.atleast_1d(np.asarray(freqs, dtype=float))
            result.phase_reference = 'travelling_wave'

            self._attach_output_paths(
                result, fm.work_dir, base_name,
                primary_files=(('grn_file', '.grn'),),
            )

            self._log("Simulation complete")
            if run_mode == RunMode.TIME_SERIES:
                result = result.synthesize_time_series(
                    source_waveform=source_waveform,
                    sample_rate=sample_rate,
                )
            return result

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    def _write_scooter_env(self, filepath, env, source, receiver, **kwargs):
        """
        Write Scooter environment file using shared ATEnvWriter

        Scooter uses ReadEnvironmentMod format (same as Kraken) with additional sections:
        - Phase speed limits (cLow, cHigh)
        - Maximum range with multiplier (RMax)
        - Receiver ranges (not in standard Kraken format)
        - Supports shear wave parameters in bottom halfspace
        """
        from uacpy.io.oalib_writer import resolve_ssp_topopt
        ssp_topopt = resolve_ssp_topopt(env, self.interp_ssp)
        surface_type = parse_boundary_type(env.surface.acoustic_type)
        bottom_type = parse_boundary_type(env.halfspace_at_range(0.0).acoustic_type)

        frequencies = kwargs.get('frequencies', None)

        # TopOpt position 7: '0' zeroes out Scooter's stabilising attenuation
        # (see scooter.f90:81,129). Leave as ' ' otherwise — the Fortran
        # reader keeps Atten=Deltak (the default stabiliser).
        topopt_extra = '0' if self.stabilizing_attenuation_off else ''

        with open(filepath, 'w') as f:
            write_header(
                f, env, source,
                ssp_topopt=ssp_topopt,
                surface_type=surface_type,
                frequencies=frequencies,
                topopt_extra=topopt_extra,
            )
            write_absorption_block(f, env)

            write_ssp_section(
                f, env, env.depth,
                n_mesh=self.n_mesh,
                roughness=self.roughness
            )

            # Write sediment layers if layered bottom
            write_layer_sections(f, env, env.depth)

            # Scooter honours real shear attenuation on the 'A' halfspace
            # line and writes cLow/cHigh/RMax via write_phase_speed_and_rmax,
            # so the F-type reflection-table bounds line is suppressed here.
            write_bottom_section(
                f, env,
                bottom_type=bottom_type,
                filepath=Path(filepath),
                halfspace_alpha_s_source='env',
                emit_reflection_table_block=False,
            )

            rmax_m = float(receiver.ranges.max()) * self.rmax_multiplier
            from uacpy.io.oalib_writer import resolve_phase_speed_bounds
            cl, ch = resolve_phase_speed_bounds(env, self.c_low, self.c_high)
            if self.c_low is None or self.c_high is None:
                self._log(
                    f"c_low / c_high auto-derived = "
                    f"{cl:.1f} / {ch:.1f} m/s"
                )
            write_phase_speed_and_rmax(
                f, env,
                rmax_m=rmax_m,
                c_low=cl, c_high=ch,
                rmax_format="{:.6f}",
            )

            # Source and receiver depths. Use the shared ATEnvWriter so
            # arbitrary-length depth arrays are written verbatim rather than
            # collapsed to "min max /" (which the Fortran reader expands to a
            # uniformly-spaced vector — losing user-specified samples).
            write_source_depths(f, source)
            write_receiver_depths(f, receiver)

            # Broadband frequency vector (ReadfreqVec reads after source/receiver depths)
            if frequencies is not None and len(frequencies) > 1:
                write_broadband_freqs(f, frequencies)

    def _run_scooter(self, base_name: str, work_dir: Path):
        """Execute Scooter via the shared ``_run_subprocess`` helper."""
        try:
            result = self._run_subprocess(
                [self.executable, base_name],
                cwd=work_dir,
            )
        except ModelExecutionError as exc:
            self._attach_prt_tail(exc, work_dir, base_name)
            raise
        if self.verbose and result.stdout:
            self._log(f"Scooter output:\n{result.stdout}", level='debug')
