"""
Scooter finite-element FFP (Fast Field Program) model.

Computes the acoustic field in the frequency-wavenumber domain using a
finite-element discretization, then transforms ``.grn`` to a range-domain
TL field via the in-tree Python Hankel transform in
:mod:`uacpy.io.grn_reader`. Supports coherent TL, broadband ``H(f)``,
and broadband time-series output.
"""

import warnings
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from uacpy.core.exceptions import (
    ConfigurationError, ModelExecutionError,
)
from uacpy.models.base import (
    PropagationModel, RunMode, ModelSpec, USER_FRAME_SKIP,
    _max_roughness, _smooth_surface,
)
from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.results import Result
from uacpy.core.constants import parse_boundary_type
from uacpy.io.grn_reader import read_grn_file, grn_to_field, grn_to_transfer_function
from uacpy.io.oalib_writer import (
    write_scooter_env_file, reject_coarse_at_mesh,
    reject_unsupported_ssp_interp,
    SOURCE_TYPE_CODE as _SOURCE_TYPE_CODE,
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
        ``1.05 × max SSP+bottom``; a vacuum or rigid bottom has no
        half-space speed to cap on, so ``c_high`` resolves to the AT
        "unbounded" sentinel instead (see
        :func:`~uacpy.io.oalib_writer.resolve_phase_speed_bounds`).
    n_mesh : int, optional
        Mesh points per medium. ``0`` ⇒ auto. Default ``0``.
    rmax_multiplier : float, optional
        Multiplier on ``receiver.ranges.max()`` to set Scooter's spectral
        ``RMax``, which fixes the wavenumber sampling:
        ``scooter.f90:69`` takes ``Nk = INT(2000·RMax_km·(kMax−kMin)/π)``,
        i.e. ``Δk ≈ π/(2·RMax_m)``. Default ``None`` → 2.0 for
        ``COHERENT_TL``, 3.0 for ``BROADBAND`` / ``TIME_SERIES``.
    spectrum : str, optional
        FLP Opt(2): ``'positive'`` (fast, default) | ``'negative'`` | ``'both'``.
    stabilizing_attenuation_off : bool, optional
        Disable Scooter's stabilising attenuation. Default ``False``;
        leave it unless you know what you're doing (the stabiliser
        prevents pole-on-contour blow-ups).
    use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
        Standard plumbing (see :class:`PropagationModel`).

    Notes
    -----
    Range-independent FFP — single spectral solve over the full
    wavenumber axis, Hankel-transformed to range. Supports layered
    and elastic bottoms natively. The Green's-function
    ``.grn`` is converted to range-domain TL via the in-tree Python
    Hankel transform (``uacpy.io.grn_reader``).

    **Collapse defaults (overrides of :data:`DEFAULT_COLLAPSE`).**
    Per-model: ``'ssp': 'mean'``, ``'bottom_range': 'median'`` (the layer
    stack is kept since Scooter consumes layered seabed columns natively).

    Defaults auto-derived at ``run()`` time:

    - ``c_low=None`` → ``min(env.ssp) × 0.95``
    - ``c_high=None`` → ``max(max(env.ssp), env.bottom.sound_speed) × 1.05``,
      or the unbounded sentinel for a vacuum / rigid bottom
    - Spectral ``RMax = receiver.range_max × rmax_multiplier``
    - ``n_mesh=0`` → Scooter picks from frequency / wavelength.
    - TopOpt position 4 reads ``env.absorption``.

    With ``verbose='info'`` the resolved ``c_low`` / ``c_high`` are logged.

    Examples
    --------
    >>> scooter = Scooter()
    >>> result = scooter.run(env, source, receiver)
    """

    # Declarative metadata (see PropagationModel / ModelSpec). Scooter:
    # range-independent wavenumber integration. Honours multi-layer
    # fluid/elastic bottom natively; range dependence in any form is
    # collapsed to range-0. Single spectral solve → mean SSP / median
    # bottom column are the representative single profile.
    # INCOHERENT_TL is intentionally absent (no modal decomposition here).
    #
    # No ``rough_bottom``: ``SSP%sigma`` is read in exactly three places in
    # ``Scooter/`` — ``scooter.f90:63`` rejects ``sigma(2:NMedia)``,
    # ``scooter.f90:309`` reads ``sigma(1)``, and ``sparc.f90:177`` belongs to
    # the other binary. The seabed's own slot, ``sigma(NMedia+1)``, is read
    # nowhere, and a sediment-layer roughness lands in the rejected range.
    spec = ModelSpec(
        modes=(RunMode.COHERENT_TL, RunMode.BROADBAND, RunMode.TIME_SERIES),
        supports={'layered_bottom', 'elastic_media', 'rough_surface'},
        source_types=frozenset({'point', 'line', 'scaled'}),
        collapse={'ssp': 'mean', 'bottom_range': 'median'},
    )
    source = 'acoustics_toolbox'

    def __init__(
        self,
        executable: Optional[Path] = None,
        c_low: Optional[float] = None,
        c_high: Optional[float] = None,
        n_mesh: int = 0,
        rmax_multiplier: Optional[float] = None,
        interp_ssp: Optional[str] = None,
        spectrum: str = 'positive',
        stabilizing_attenuation_off: bool = False,
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
            Path to scooter executable. Auto-detected if None.
        c_low : float, optional
            Lower phase speed limit (m/s). None = auto (0.95 * min SSP speed).
        c_high : float, optional
            Upper phase speed limit (m/s). None = auto (1.05 * max of SSP and
            bottom speed; unbounded for a vacuum / rigid bottom).
        n_mesh : int, optional
            Total number of mesh points PER MEDIUM used by the finite-element
            spectral solver (AT's ``NG`` column on the SSP mesh line). 0 = let
            Scooter pick automatically from frequency / wavelength. Default: 0.
            Note: this is NOT a "points per wavelength" density — it is a total
            point count per medium.
        rmax_multiplier : float, optional
            Multiply max receiver range to set the spectral ``RMax``, which
            fixes the wavenumber sampling (see
            :meth:`_resolve_rmax_multiplier`). Default ``None`` → 2.0 for
            ``COHERENT_TL``, 3.0 for ``BROADBAND`` / ``TIME_SERIES``.
        interp_ssp : str, optional
            SSP connection scheme written into ``TopOpt(1)``. ``None``
            (default) resolves to ``'linear'`` (C-linear): Scooter declares
            no range-dependent-SSP capability, so ``run()`` collapses
            ``env.ssp`` to 1-D before the deck is written and the
            range-dependent auto-pick never applies. Explicit values:
            ``'linear'``, ``'n2linear'``, ``'pchip'``, ``'cubic'`` /
            ``'spline'``. ``env.ssp.shape='isovelocity'`` always forces
            ``'C'`` regardless.
        spectrum : {'positive', 'negative', 'both'}, optional
            FLP Option(2:2) in AT nomenclature. uacpy writes no ``.flp`` —
            the letter is applied by the in-tree Hankel transform, which
            follows ``Matlab/Scooter/fieldsco.m:170-181``. 'positive'
            (default) uses only the positive wavenumber spectrum (fast,
            recommended). 'negative' uses only the negative branch; 'both'
            integrates along the full k-axis.
        stabilizing_attenuation_off : bool, optional
            If True, writes ``'0'`` at TopOpt position 7. Scooter then
            replaces its default ``Atten = Deltak`` with zero
            (``scooter.f90:81,129-130``). Leave False (default) unless you
            know what you're doing — the stabiliser is there to prevent
            pole-on-contour blow-ups.
        """
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
            cleanup=cleanup, timeout=timeout, collapse=collapse,
        )

        self.c_low = c_low
        self.c_high = c_high
        self.interp_ssp = interp_ssp
        if c_low is not None and c_high is not None and c_low >= c_high:
            raise ConfigurationError(
                f"Scooter spectral phase-velocity band requires "
                f"c_low < c_high; got c_low={c_low} m/s, c_high={c_high} m/s."
            )
        self.n_mesh = n_mesh
        self.rmax_multiplier = rmax_multiplier

        spectrum_map = {'positive': 'P', 'negative': 'N', 'both': 'B'}
        if spectrum not in spectrum_map:
            raise ConfigurationError(
                f"Invalid spectrum '{spectrum}'. Use 'positive', 'negative', or 'both'."
            )
        self.spectrum = spectrum
        self._spectrum_code = spectrum_map[spectrum]

        self.stabilizing_attenuation_off = bool(stabilizing_attenuation_off)
        if self.stabilizing_attenuation_off:
            # ``scooter.f90:581`` evaluates the FE solve on the contour
            # ``k + i*Atten``, which is what holds the modal poles off the
            # integration path. Zeroing it puts them back on a fixed-Δk grid that
            # cannot resolve them: measured against Kraken on a 100 m Pekeris
            # guide at 100 Hz, TL is 6.7 dB out at 5 km even with the inverse
            # transform using the correct Atten = 0, versus 0.075 dB with the
            # stabiliser left on.
            warnings.warn(
                "Scooter(stabilizing_attenuation_off=True) removes the contour "
                "offset that keeps the modal poles off the integration path "
                "(scooter.f90:581), so the wavenumber integral is evaluated "
                "through them. Measured 6.7 dB TL error at 5 km against Kraken "
                "on a 100 m Pekeris guide, against 0.075 dB with the stabiliser "
                "on. Use it only to inspect the un-damped kernel.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )

        # Run modes, capability flags and collapse defaults come from the
        # class-level ``spec`` (applied by PropagationModel.__init__).
        self._exe = self._resolve_executable(
            executable,
            lambda: self._find_executable_in_paths(
                'scooter.exe', bin_subdirs='oalib',
                dev_subdir='Acoustics-Toolbox/Scooter',
            ),
        )

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
            a default 128-bin vector spanning fc*(1 +/- 0.25) is
            generated; a multi-element ``source.frequencies`` is used
            as the band directly.
        source_waveform : ndarray, optional
            Source pulse for ``TIME_SERIES`` mode.
        sample_rate : float, optional
            Sampling rate of ``source_waveform`` in Hz.
        output_duration : float, optional
            Desired output duration (seconds) for ``TIME_SERIES``. When
            given, the source waveform is zero-padded internally so the
            broadband frequency grid is tight enough (``Δf =
            1/output_duration``). Defaults to
            ``len(source_waveform)/sample_rate``.

        Returns
        -------
        result : Result
            :class:`Field` — narrowband complex pressure for COHERENT_TL,
            broadband complex ``H(f)`` for BROADBAND, real ``p(d, r, t)``
            for TIME_SERIES.
        """
        run_mode = self._resolve_run_mode(run_mode)
        if run_mode not in (RunMode.BROADBAND, RunMode.TIME_SERIES):
            self._warn_ignored_run_kwargs(
                run_mode,
                frequencies=frequencies,
                source_waveform=source_waveform,
                sample_rate=sample_rate,
                output_duration=output_duration,
            )
        source_waveform, frequencies = self._prepare_timeseries(
            run_mode, source, frequencies, source_waveform, sample_rate,
            output_duration,
        )

        env = self._project_environment(env)
        media_depth = self._total_media_depth(env)

        self.validate_inputs(env, source, receiver, run_mode=run_mode)
        reject_unsupported_ssp_interp('Scooter', self.interp_ssp)
        # A 'precalc' bottom is staged verbatim as <base>.irc; a table in the
        # wrong layout (typically a theta/|R|/phase angle table) aborts the
        # binary with a bare Fortran backtrace, so the header is checked here,
        # ahead of the launch.
        self._reject_malformed_irc_bottom(env)

        # Broadband mode (BROADBAND or TIME_SERIES) requires a
        # frequency vector. The in-tree Python Hankel transform handles
        # the multi-frequency Green's-function output; fields.exe is
        # not used.
        broadband_freqs = None
        broadband_mode = run_mode in (RunMode.BROADBAND, RunMode.TIME_SERIES)
        if broadband_mode:
            broadband_freqs = self._resolve_broadband_frequencies(
                source, frequencies,
            )
            self._log(f"Broadband: {len(broadband_freqs)} frequencies, "
                      f"{broadband_freqs[0]:.1f}-{broadband_freqs[-1]:.1f} Hz")

        # A pinned n_mesh is checked at the highest frequency the run will
        # march: the AT reader's "Mesh is too coarse" floor scales with
        # frequency, so a mesh that clears fc can still under-resolve the
        # top of a broadband sweep.
        marched = (broadband_freqs if broadband_freqs is not None
                   else source.frequencies)
        reject_coarse_at_mesh(
            'Scooter', self.n_mesh, env,
            float(np.max(np.atleast_1d(np.asarray(marched, dtype=float)))))

        fm = self._setup_file_manager()

        try:
            base_name = 'model'

            env_file = fm.get_path(f'{base_name}.env')
            self._log(f"Writing environment file: {env_file}")

            self._write_scooter_env(
                env_file, env, source, receiver,
                frequencies=broadband_freqs,
                run_mode=run_mode,
            )

            grn_data = self._run_and_read_grn(fm, base_name)
            result = self._assemble_field_from_grn(
                grn_data, source, receiver, broadband_mode)

            freqs = broadband_freqs if broadband_mode else float(source.frequencies[0])
            self._stamp_result(result, source, backend='scooter',
                               frequencies=freqs, phase_reference='travelling_wave')

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
            return self._mask_unresolvable_depths(
                result, receiver, media_depth)

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    def _project_environment(self, env):
        """Collapse unsupported features, then drop a surface roughness the
        top boundary condition cannot carry.

        ``SSP%sigma(1)`` enters the solve only through the vacuum branch of
        ``Scooter/scooter.f90:309`` (``g = -i·sqrt(omega2/cInside² − x)·
        sigma(1)²``, reached from ``:635`` via ``BCImpedance(x, 'TOP', …)``).
        Every other top boundary — rigid, acousto-elastic, tabulated — takes
        an impedance that never reads the slot, so the deck would carry a
        roughness the run silently ignores. Measured on a 100 m Pekeris guide
        at 100 Hz, ``roughness=2.0`` against ``0.0`` moves TL by 20.3 dB under
        a vacuum surface and by exactly 0.0 dB under a rigid or
        acousto-elastic one.
        """
        env = super()._project_environment(env)
        sigma = _max_roughness(env.surface.properties)
        if not sigma:
            return env
        acoustic_type = env.surface.acoustic_type
        if parse_boundary_type(acoustic_type).to_acoustics_toolbox_code() == 'V':
            return env
        env.surface = _smooth_surface(env.surface)
        warnings.warn(
            f"{self.model_name} reads the sea-surface roughness only for a "
            f"pressure-release (vacuum) surface; the {acoustic_type!r} surface "
            f"takes an impedance that never touches SSP%sigma(1), so "
            f"env.surface.roughness={sigma:g} m was dropped.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
        return env

    def _max_receiver_depth(self, env) -> float:
        return self._total_media_depth(env)

    def _run_and_read_grn(self, fm, base_name):
        """Run the binary and read back its Green's function.

        A missing ``.grn`` and an ``nk=0`` one are both run failures the binary
        does not report through its exit status, so each is turned into a typed
        error carrying the ``.prt`` diagnostics.
        """
        self._log("Running...")
        self._run_scooter(base_name, fm.work_dir)

        # A missing or empty .grn means the binary died silently; the raised
        # error carries the .prt tail with the actual cause.
        grn_file = self._require_output(
            [fm.get_path(f'{base_name}.grn')],
            what="a Green's function (.grn)",
            prt_base=base_name, work_dir=fm.work_dir,
        )

        self._log("Reading Green's function...")
        grn_data = read_grn_file(grn_file)
        if grn_data['nk'] == 0:
            exc = ModelExecutionError(
                self.model_name, return_code=0, stdout=None,
                stderr="Scooter produced empty Green's function (nk=0)",
            )
            self._attach_prt_tail(exc, fm.work_dir, base_name)
            raise exc
        return grn_data

    def _assemble_field_from_grn(self, grn_data, source, receiver,
                                 broadband_mode):
        """Hankel-transform the Green's function onto the receiver ranges.

        Broadband transforms every frequency in the ``.grn`` at once; the
        narrowband path transforms the single frequency slice.
        """
        transform_kwargs = dict(
            source_type=_SOURCE_TYPE_CODE[source.source_type],
            spectrum=self._spectrum_code,
        )
        if broadband_mode:
            self._log(f"Transforming {grn_data['nfreq']} frequencies to "
                      f"range domain...")
            return grn_to_transfer_function(
                grn_data, receiver.ranges, **transform_kwargs)
        self._log("Transforming to range domain (direct-DFT Hankel transform)...")
        return grn_to_field(
            grn_data, receiver.ranges, method='direct_dft', **transform_kwargs)

    def _resolve_rmax_multiplier(self, run_mode: RunMode) -> float:
        """Pick the effective ``rmax_multiplier`` for this run.

        ``scooter.exe`` writes only the wavenumber-domain ``.grn``; the k→r
        step is uacpy's :func:`~uacpy.io.grn_reader._hankel_transform`, a
        direct trapezoidal-rule DFT (``fieldsco.m:5``), not an FFT. What
        ``RMax`` controls is the wavenumber grid the solver samples:
        ``scooter.f90:69`` sets ``Nk = INT(2000·RMax_km·(kMax−kMin)/π)``, so
        ``Δk ≈ π/(2·RMax_m)`` and both cost (``Nk`` samples of the
        finite-element solve) and resolution scale linearly with the
        multiplier. A uniform-``Δk`` DFT is periodic in range with period
        ``2π/Δk ≈ 4·RMax_m`` at the top frequency, so the wrap-around
        replica also moves out proportionally. ``BROADBAND`` /
        ``TIME_SERIES`` use the finer grid: their syntheses sum many
        frequencies, and an under-resolved ``G(k)`` shows up as trapezoidal
        error in every one of them. User-pinned values win.
        """
        if self.rmax_multiplier is not None:
            return float(self.rmax_multiplier)
        return 3.0 if run_mode in (RunMode.TIME_SERIES, RunMode.BROADBAND) else 2.0

    def _write_scooter_env(
        self,
        filepath,
        env,
        source,
        receiver,
        *,
        frequencies=None,
        run_mode=RunMode.COHERENT_TL,
    ):
        """
        Write Scooter environment file using shared ATEnvWriter

        Scooter uses ReadEnvironmentMod format (same as Kraken) with additional sections:
        - Phase speed limits (cLow, cHigh)
        - Maximum range with multiplier (RMax)
        - Supports shear wave parameters in bottom halfspace

        No receiver ranges are written: ``scooter.f90``'s ``GetPar`` never
        calls ``ReadRcvrRanges`` — the ``.env`` ends at ``RMax``, and the
        range axis is applied in-tree when the ``.grn`` is transformed.
        """
        from uacpy.io.oalib_writer import resolve_ssp_topopt
        ssp_topopt = resolve_ssp_topopt(env, self.interp_ssp)
        surface_type = parse_boundary_type(env.surface.acoustic_type)
        bottom_type = parse_boundary_type(env.bottom.halfspace_at(range=0.0).acoustic_type)

        # TopOpt position 7: '0' zeroes out Scooter's stabilising attenuation
        # (scooter.f90:81,129-130). Leave as ' ' otherwise — Scooter keeps
        # Atten=Deltak, the default stabiliser.
        topopt_extra = '0' if self.stabilizing_attenuation_off else ''

        rmax_m = float(receiver.ranges.max()) * self._resolve_rmax_multiplier(run_mode)
        from uacpy.io.oalib_writer import resolve_phase_speed_bounds
        cl, ch = resolve_phase_speed_bounds(env, self.c_low, self.c_high)
        if self.c_low is None or self.c_high is None:
            self._log(
                f"c_low / c_high auto-derived = "
                f"{cl:.1f} / {ch:.1f} m/s"
            )

        write_scooter_env_file(
            filepath, env, source, receiver,
            ssp_topopt=ssp_topopt,
            surface_type=surface_type,
            bottom_type=bottom_type,
            frequencies=frequencies,
            topopt_extra=topopt_extra,
            n_mesh=self.n_mesh,
            rmax_m=rmax_m,
            c_low=cl, c_high=ch,
        )

    def _run_scooter(self, base_name: str, work_dir: Path):
        """Execute Scooter via the shared binary-launch helper."""
        self._run_and_attach_prt([str(self._exe), base_name], work_dir, base_name,
                                 stale_outputs=('.grn',))
