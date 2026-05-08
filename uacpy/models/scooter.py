"""
Scooter finite-element FFP (Fast Field Program) model.

Computes the acoustic field in the frequency-wavenumber domain using a
finite-element discretization, then transforms ``.grn`` to a range-domain
TL field via the in-tree Python Hankel transform in
:mod:`uacpy.io.grn_reader`. Supports coherent TL, broadband ``H(f)``,
and broadband time-series output.

The upstream Acoustic Toolbox discontinued the Fortran ``fields.exe``
post-processor in 2020; ``install.sh`` no longer builds it. The
``use_fields_exe`` constructor flag is retained for legacy installs that
have it on disk but defaults to ``False``.
"""

import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from uacpy.core.exceptions import (
    ConfigurationError, ExecutableNotFoundError, ModelExecutionError,
)
from uacpy.models.base import PropagationModel, RunMode, _UNSET, _resolve_overrides
from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.results import Result
from uacpy.core.constants import (
    AttenuationUnits, VolumeAttenuation,
    parse_ssp_type, parse_boundary_type,
    C_LOW_FACTOR, C_HIGH_FACTOR, DEFAULT_C_MIN, DEFAULT_C_MAX,
)
from uacpy.io.grn_reader import read_grn_file, grn_to_field, grn_to_transfer_function
from uacpy.io.oalib_reader import read_shd_file
from uacpy.io.oalib_writer import write_bio_layers, write_broadband_freqs, write_fg_params, write_header, write_layer_sections, write_receiver_depths, write_source_depths, write_ssp_section


class Scooter(PropagationModel):
    """
    Scooter finite element FFP (Fast Field Program) model

    Frequency-domain solver for underwater acoustics.
    Developed by Michael B. Porter.

    Parameters
    ----------
    executable : str or Path, optional
        Path to scooter executable. If None, searches automatically.
    use_tmpfs : bool, optional
        Use RAM filesystem. Default is False.
    verbose : bool, optional
        Verbose output. Default is False.

    Examples
    --------
    >>> scooter = Scooter()
    >>> result = scooter.run(env, source, receiver)
    """

    def __init__(
        self,
        executable: Optional[Path] = None,
        fields_executable: Optional[Path] = None,
        c_low: Optional[float] = None,
        c_high: Optional[float] = None,
        n_mesh: int = 0,
        roughness: float = 0.0,
        rmax_multiplier: float = 2.0,
        volume_attenuation: Optional[str] = None,
        attenuation_unit=AttenuationUnits.DB_PER_WAVELENGTH,
        francois_garrison_params: Optional[tuple] = None,
        bio_layers: Optional[list] = None,
        source_type: str = 'R',
        spectrum: str = 'positive',
        stabilizing_attenuation_off: bool = False,
        field_interp: str = 'O',
        use_fields_exe: bool = False,
        use_tmpfs: bool = False,
        verbose: bool = False,
        work_dir: Optional[Path] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        executable : Path, optional
            Path to scooter executable. Auto-detected if None.
        fields_executable : Path, optional
            Path to fields.exe post-processor. Auto-detected if None.
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
        volume_attenuation : str, optional
            'T' (Thorp), 'F' (Francois-Garrison), 'B' (Biological). Default: None.
        francois_garrison_params : tuple, optional
            Required when ``volume_attenuation='F'``. Tuple
            ``(T, S, pH, z_bar)``: temperature (degC), salinity (psu), pH,
            mean depth (m).
        bio_layers : list of tuples, optional
            Required when ``volume_attenuation='B'``. List of per-layer
            5-tuples ``(Z1, Z2, f0, Q, a0)``: depth range (m),
            resonance frequency (Hz), quality factor, absorption coefficient.
        source_type : {'R', 'X'}, optional
            FLP Option(1:1). 'R' = cylindrical (point source, default),
            'X' = Cartesian (line source). Honoured by both ``fields.exe``
            and the in-tree Hankel transform — the latter scales by
            ``√k`` and ``1/√(2πr)`` for 'R' and by ``1/√(2π)`` for 'X'.
        spectrum : {'positive', 'negative', 'both'}, optional
            FLP Option(2:2). 'positive' (default) uses only the positive
            wavenumber spectrum (fast, recommended). 'negative' uses only
            the negative branch; 'both' integrates along the full k-axis.
            Honoured by both ``fields.exe`` and the in-tree transform.
        stabilizing_attenuation_off : bool, optional
            If True, writes ``'0'`` at TopOpt position 7. Scooter then
            sets its stabilising attenuation to zero (see
            ``scooter.f90:81,129``). Leave False (default) unless you
            know what you're doing — the stabiliser is there to prevent
            pole-on-contour blow-ups.
        field_interp : {'O', 'P'}, optional
            FLP Option(3:3). 'O' = polynomial interpolation (default,
            what Scooter's own sample FLP uses), 'P' = Pade. See
            ``fields.f90:82-90``.
        use_fields_exe : bool, optional
            Default ``False`` — uses the in-tree Python Hankel transform
            in :mod:`uacpy.io.grn_reader`. The upstream Acoustic Toolbox
            discontinued ``fields.exe`` in 2020 and ``install.sh`` no
            longer builds it; setting ``True`` is supported for legacy
            installs that have it on disk and silently falls back to
            the Python transform otherwise. Broadband runs (BROADBAND /
            TIME_SERIES) always use the Python transform regardless
            (``fields.exe`` is single-frequency only).
        """
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
            **kwargs,
        )

        self.c_low = c_low
        self.c_high = c_high
        self.n_mesh = n_mesh
        self.roughness = roughness
        self.rmax_multiplier = rmax_multiplier
        self.volume_attenuation = volume_attenuation
        self.attenuation_unit = AttenuationUnits.from_string(attenuation_unit)
        self.francois_garrison_params = francois_garrison_params
        self.bio_layers = bio_layers

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

        self.use_fields_exe = use_fields_exe

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
        self._supports_layered_bottom = True
        self._supports_elastic_media = True
        if executable is None:
            self.executable = self._find_executable_in_paths(
                'scooter.exe', bin_subdirs='oalib',
                dev_subdir='Acoustics-Toolbox/Scooter',
            )
        else:
            self.executable = Path(executable)

        if not self.executable.exists():
            raise ExecutableNotFoundError('Scooter', str(self.executable))

        if fields_executable is None:
            # Auto-detect lazily; allow Scooter to be instantiated even when
            # fields.exe is missing (user may have use_fields_exe=False).
            self._fields_executable = None
        else:
            self._fields_executable = Path(fields_executable)

        # Inherits base validation (PropagationModel._validate_volume_attenuation_params)
        self._validate_volume_attenuation_params()

    def _get_fields_executable(self) -> Optional[Path]:
        """
        Locate ``fields.exe`` (Scooter post-processor); cache the result.

        Returns ``None`` if the binary cannot be found. Unlike ``scooter.exe``
        the ``fields.exe`` post-processor is optional — the in-tree Python
        Hankel transform in :mod:`uacpy.io.grn_reader` is used as a fallback.
        Upstream Acoustics-Toolbox has flagged Fortran ``fields.exe`` as
        deprecated (see ``Scooter/Makefile`` header), so many installations
        simply don't ship it.
        """
        if self._fields_executable is None:
            try:
                self._fields_executable = self._find_executable_in_paths(
                    'fields.exe', bin_subdirs='oalib',
                    dev_subdir='Acoustics-Toolbox/Scooter',
                )
            except ExecutableNotFoundError:
                self._fields_executable = False  # sentinel: searched, missing
        if self._fields_executable is False:
            return None
        return self._fields_executable

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode=None,
        frequencies: Optional['np.ndarray'] = None,
        c_low=_UNSET,
        c_high=_UNSET,
        n_mesh=_UNSET,
        roughness=_UNSET,
        rmax_multiplier=_UNSET,
        volume_attenuation=_UNSET,
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
        c_low, c_high, n_mesh, roughness, rmax_multiplier, volume_attenuation : optional
            Per-call overrides for constructor defaults.
        **kwargs
            Additional Scooter parameters

        Returns
        -------
        result : Result
            :class:`TLField` for COHERENT_TL, :class:`TransferFunction`
            for BROADBAND, :class:`TimeSeriesField` for TIME_SERIES.
        """
        import numpy as np

        if run_mode is None:
            run_mode = RunMode.COHERENT_TL

        self._warn_unknown_kwargs(kwargs)

        if run_mode == RunMode.TIME_SERIES and (
            source_waveform is None or sample_rate is None
        ):
            raise ValueError(
                "Scooter.run(run_mode=TIME_SERIES) requires source_waveform "
                "and sample_rate. For the broadband transfer function "
                "H(f), use run_mode=RunMode.BROADBAND."
            )

        with _resolve_overrides(self, c_low=c_low, c_high=c_high, n_mesh=n_mesh,
                                roughness=roughness, rmax_multiplier=rmax_multiplier,
                                volume_attenuation=volume_attenuation):
            # Handle range-dependent environments
            env = self._project_environment(env)

            # Clip receiver depths to environment depth (with safety margin)
            receiver = self._clip_receiver_depths(receiver, env.depth)

            self.validate_inputs(env, source, receiver, run_mode=run_mode)

            # Broadband mode (BROADBAND or TIME_SERIES) requires a
            # frequency vector; fields.exe only supports single-frequency GRNs,
            # so we force the in-tree Hankel path in that case.
            broadband_freqs = None
            broadband_mode = run_mode in (RunMode.BROADBAND, RunMode.TIME_SERIES)
            if broadband_mode:
                if frequencies is not None:
                    broadband_freqs = np.asarray(frequencies, dtype=float)
                else:
                    fc = float(source.frequencies[0])
                    broadband_freqs = np.linspace(fc * 0.5, fc * 2.0, 64)
                self._log(f"Broadband mode: {len(broadband_freqs)} frequencies, "
                          f"{broadband_freqs[0]:.1f}-{broadband_freqs[-1]:.1f} Hz")

            # Decide whether to use fields.exe. Broadband runs must use the
            # Python Hankel transform; fields.exe bails on Nfreq > 1. If the
            # user requested fields.exe but the binary is not installed,
            # transparently fall back to the Python transform.
            use_fields = self.use_fields_exe and not broadband_mode
            if self.use_fields_exe and broadband_mode:
                self._log(
                    "fields.exe cannot handle Nfreq > 1; broadband run "
                    "falls back to the in-tree Python Hankel transform "
                    "(use_fields_exe=True is ignored in broadband mode).",
                    level='warn',
                )
            if use_fields and self._get_fields_executable() is None:
                self._log(
                    "fields.exe not found; falling back to in-tree Python "
                    "Hankel transform. Set use_fields_exe=False to silence.",
                    level='warn',
                )
                use_fields = False

            fm = self._setup_file_manager()
            self.file_manager = fm

            try:
                base_name = 'model'

                # Write environment file
                env_file = fm.get_path(f'{base_name}.env')
                self._log(f"Writing environment file: {env_file}")

                self._write_scooter_env(
                    env_file, env, source, receiver,
                    frequencies=broadband_freqs,
                    **kwargs
                )

                # Run Scooter
                self._log("Running Scooter...")
                self._run_scooter(base_name, fm.work_dir)

                # Sanity-check the Green's function output
                grn_file = fm.get_path(f'{base_name}.grn')
                if not grn_file.exists():
                    self._log(f"Green's function file not found: {grn_file}", level='error')
                    raise FileNotFoundError(f"Green's function file not found: {grn_file}")

                if use_fields:
                    # Write FLP and invoke fields.exe -> .shd
                    flp_file = fm.get_path(f'{base_name}.flp')
                    self._log(f"Writing fields FLP file: {flp_file}")
                    self._write_fields_flp(flp_file, env, source, receiver)

                    self._log("Running fields.exe...")
                    self._run_fields(base_name, fm.work_dir)

                    shd_file = fm.get_path(f'{base_name}.shd')
                    if not shd_file.exists():
                        raise FileNotFoundError(
                            f"fields.exe did not produce {shd_file}; "
                            f"check {fm.work_dir}/fields.prt for diagnostics."
                        )
                    self._log("Reading SHD file...")
                    result = read_shd_file(shd_file)
                    result.tag(
                        model=self.model_name,
                        backend='scooter.exe+fields.exe',
                        source_depths=source.depths,
                        frequencies=float(source.frequencies[0]),
                        phase_reference='travelling_wave',
                        source_type=self.source_type,
                        spectrum=self.spectrum,
                        post_processor='fields.exe',
                    )
                else:
                    self._log("Reading Green's function...", level='info')
                    grn_data = read_grn_file(grn_file)

                    if grn_data['nk'] == 0:
                        self._log("Scooter produced empty Green's function (nk=0)", level='error')
                        raise ModelExecutionError(
                            self.model_name, return_code=0,
                            stdout=None,
                            stderr="Scooter produced empty Green's function (nk=0)",
                        )

                    if grn_data['nsd'] > 1:
                        self._log(
                            f"Multi-source-depth GRN ({grn_data['nsd']} sources); "
                            "in-tree Hankel transform returns the field for the "
                            "first source depth only.",
                            level='warn',
                        )

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
                    result.tag(
                        model=self.model_name,
                        backend='scooter.exe',
                        source_depths=source.depths,
                        frequencies=(broadband_freqs if broadband_mode
                                     else float(source.frequencies[0])),
                        phase_reference='travelling_wave',
                        post_processor='in_tree_hankel',
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
        # Parse types (parse_* normalises string aliases like 'halfspace' vs 'half-space')
        ssp_type = parse_ssp_type(env.ssp.interp)
        surface_type = parse_boundary_type(env.surface.acoustic_type)
        bottom_type = parse_boundary_type(env.bottom.acoustic_type)

        # Parse volume attenuation from instance attribute
        vol_atten = None
        if self.volume_attenuation:
            vol_atten = VolumeAttenuation.from_string(self.volume_attenuation)

        # Get kwargs
        frequencies = kwargs.get('frequencies', None)

        # TopOpt position 7: '0' zeroes out Scooter's stabilising attenuation
        # (see scooter.f90:81,129). Leave as ' ' otherwise — the Fortran
        # reader then keeps Atten=Deltak (the default stabiliser).
        topopt_extra = '0' if self.stabilizing_attenuation_off else ''

        with open(filepath, 'w') as f:
            # Write standard ENV sections using ATEnvWriter
            write_header(
                f, env, source,
                ssp_type=ssp_type,
                surface_type=surface_type,
                attenuation_unit=self.attenuation_unit,
                volume_attenuation=vol_atten,
                frequencies=frequencies,
                topopt_extra=topopt_extra,
            )

            # Francois-Garrison / Biological follow-up lines (after TopOpt,
            # before SSP). ReadTopOpt in AT reads these immediately when
            # TopOpt(4)='F'/'B'.
            if vol_atten == VolumeAttenuation.FRANCOIS_GARRISON:
                write_fg_params(f, self.francois_garrison_params)
            elif vol_atten == VolumeAttenuation.BIOLOGICAL:
                write_bio_layers(f, self.bio_layers)

            write_ssp_section(
                f, env, env.depth,
                n_mesh=self.n_mesh,
                roughness=self.roughness
            )

            # Write sediment layers if layered bottom
            write_layer_sections(f, env, env.depth)

            # Bottom section with shear wave support for Scooter
            bottom_code = bottom_type.to_acoustics_toolbox_code()
            sigma = getattr(env.bottom, 'roughness', 0.0)

            # Check for range-dependent bathymetry
            if len(env.bathymetry) > 1:
                f.write(f"'{bottom_code}~' {sigma:.1f}\n")
            else:
                f.write(f"'{bottom_code}' {sigma:.1f}\n")

            # Handle reflection coefficient file (type 'F')
            if bottom_code == 'F':
                # Copy reflection file to working directory
                if env.bottom.reflection_file:
                    brc_source = Path(env.bottom.reflection_file)
                    if brc_source.exists():
                        # Copy to same directory as .env file with matching base name
                        brc_dest = Path(filepath).with_suffix('.brc')
                        shutil.copy(brc_source, brc_dest)
                        self._log(f"Copied reflection file: {brc_source} -> {brc_dest}", level='info')
                    else:
                        self._log(f"Reflection coefficient file not found: {env.bottom.reflection_file}", level='error')
                        raise FileNotFoundError(
                            f"Reflection coefficient file not found: {env.bottom.reflection_file}\n"
                            f"Generate this file using BOUNCE or OASR models."
                        )
                else:
                    self._log("acoustic_type='file' requires reflection_file parameter", level='error')
                    raise ValueError(
                        "acoustic_type='file' requires reflection_file parameter.\n"
                        "Example: BoundaryProperties(acoustic_type='file', reflection_file='path/to/file.brc')"
                    )

                # 'F': no bottom-property line; cLow/cHigh/RMax follow
                # below (ReadEnvironmentMod.f90:133-140).
                pass

            elif bottom_code == 'A':  # Half-space (with optional shear)
                z_bottom = env.bottom.depth if hasattr(env.bottom, 'depth') else env.depth
                cp = env.bottom.sound_speed
                cs = getattr(env.bottom, 'shear_speed', 0.0)
                rho = env.bottom.density
                alpha_p = env.bottom.attenuation
                alpha_s = getattr(env.bottom, 'shear_attenuation', 0.0)
                f.write(f"  {z_bottom:.2f}  {cp:.2f}  {cs:.1f}  "
                       f"{rho:.2f}  {alpha_p:.2f}  {alpha_s:.2f} /\n")

            # cLow/cHigh: BRC-table bounds for 'F'; SSP-derived otherwise.
            if bottom_code == 'F':
                c_low = getattr(env.bottom, 'reflection_cmin', DEFAULT_C_MIN)
                c_high = getattr(env.bottom, 'reflection_cmax', DEFAULT_C_MAX)
            else:
                _ssp_pairs = env.ssp.to_pairs()
                c_min = float(_ssp_pairs[:, 1].min())
                c_max = max(float(_ssp_pairs[:, 1].max()), env.bottom.sound_speed)
                c_low = self.c_low if self.c_low is not None else c_min * C_LOW_FACTOR
                c_high = self.c_high if self.c_high is not None else c_max * C_HIGH_FACTOR
            f.write(f"{c_low:.1f} {c_high:.1f}\n")

            # RMax (km) sets Scooter's k-grid via deltak = pi/RMax. Derived
            # from receiver range, not reflection_rmax_m (different concept).
            rmax = float(receiver.ranges.max() / 1000.0) * self.rmax_multiplier
            f.write(f"{rmax:.6f}\n")

            # Source and receiver depths. Use the shared ATEnvWriter so
            # arbitrary-length depth arrays are written verbatim rather than
            # collapsed to "min max /" (which the Fortran reader expands to a
            # uniformly-spaced vector — losing user-specified samples).
            write_source_depths(f, source)
            write_receiver_depths(f, receiver)

            # Broadband frequency vector (ReadfreqVec reads after source/receiver depths)
            if frequencies is not None and len(frequencies) > 1:
                write_broadband_freqs(f, frequencies)

    def _write_fields_flp(
        self,
        filepath: Path,
        env: Environment,
        source: Source,
        receiver: Receiver,
        options: Optional[str] = None,
    ) -> None:
        """
        Write the FLP parameter file consumed by ``fields.exe``.

        Per ``third_party/Acoustics-Toolbox/doc/fields.htm`` the Scooter
        post-processor's FLP has three records:

        1. 4-character option string (Coords / Spectrum / Interp / SBP)
        2. NRr — number of receiver ranges
        3. Rr(1:NRr) receiver ranges in **km** (terminated with ``/``)

        Unlike Kraken's ``field.exe``, fields.exe takes source and receiver
        *depths* from the ``.grn`` header, not the FLP file.

        Parameters
        ----------
        filepath : Path
            Destination ``.flp`` path.
        env, source, receiver : core objects
            Provided for API symmetry with Scooter's env writer; only
            ``receiver.ranges`` is used here.
        options : str, optional
            Explicit 4-character option override. If None, the string is
            built from ``self.source_type``, ``self._spectrum_code``,
            polynomial interpolation (``'O'``), and no beam pattern.
        """
        if options is None:
            # Pos 1: source_type ('R' cylindrical / 'X' Cartesian)
            # Pos 2: spectrum code ('P' / 'N' / 'B')
            # Pos 3: field_interp ('O' polynomial / 'P' Pade, see
            #        fields.f90:82-90)
            # Pos 4: ' ' (no source beam pattern file)
            options = (
                f"{self.source_type}{self._spectrum_code}"
                f"{self.field_interp} "
            )
        if len(options) != 4:
            raise ValueError(
                f"FLP option string must be 4 characters, got {len(options)!r}"
            )

        ranges_km = receiver.ranges / 1000.0
        with open(filepath, 'w') as f:
            f.write(f"'{options}'\n")
            f.write(f"{len(ranges_km)}\n")
            # Explicit list; fields.exe expands the list if a trailing '/' is
            # given after exactly two values, but listing all ranges avoids
            # the uniform-spacing assumption.
            ranges_str = " ".join(f"{r:.6f}" for r in ranges_km)
            f.write(f"{ranges_str} /\n")

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

    def _run_fields(self, base_name: str, work_dir: Path):
        """Execute ``fields.exe`` to transform ``<base>.grn`` -> ``<base>.shd``."""
        fields_exe = self._get_fields_executable()
        if fields_exe is None:
            raise ExecutableNotFoundError(self.model_name, 'fields.exe')
        try:
            result = self._run_subprocess(
                [fields_exe, base_name],
                cwd=work_dir,
            )
        except ModelExecutionError as exc:
            self._attach_prt_tail(exc, work_dir, 'fields')
            raise
        if self.verbose and result.stdout:
            self._log(f"fields.exe output:\n{result.stdout}", level='debug')
