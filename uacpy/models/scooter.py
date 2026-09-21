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
    _slabs_of,
    _line_source_unit_at_1m, _source_sound_speed,
    PropagationModel, RunMode, ModelSpec, USER_FRAME_SKIP,
    _max_roughness, _smooth_surface,
)
from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.results import Result, ResultStack
from uacpy.core.constants import parse_boundary_type
from uacpy.io.grn_reader import (read_grn_file, grn_to_field,
                                 grn_to_transfer_function,
                                 available_memory_bytes)
from uacpy.io.oalib_writer import (
    write_scooter_env_file, reject_coarse_at_mesh,
    reject_unsupported_ssp_interp, resolve_ssp_topopt,
    resolve_phase_speed_bounds,
    SOURCE_TYPE_CODE as _SOURCE_TYPE_CODE,
)

def _nk_consequence(nk: int) -> str:
    """What a wavenumber count below 2 costs the run.

    ``Nk <= 0`` allocates ``k( Nk )`` empty (``scooter.f90:73``) and the
    Green's function comes back with no wavenumber samples at all. ``Nk = 1``
    allocates one sample and then divides by ``Nk - 1 = 0`` at
    ``scooter.f90:77`` and again per frequency at ``:125``, so ``Deltak`` and
    the stabilising ``Atten`` (``:129``) are both infinite and every Green's
    function value is NaN. Neither case changes the exit status, so both reach
    the caller as a full-shape result unless something refuses them.
    """
    if nk == 1:
        return (
            "scooter.f90:77 spaces the grid as "
            "Deltak = (kMax - kMin) / (Nk - 1), so a single sample divides by "
            "zero: the binary writes an all-NaN Green's function at exit 0 "
            "and the transformed field is all-NaN."
        )
    return (
        "The wavenumber vector is empty and the Green's function comes back "
        "with no samples at exit 0."
    )


def _deck_nk(rmax_m: float, f_max: float, c_low: float, c_high: float) -> int:
    """The ``Nk`` ``scooter.f90:69`` will derive from this deck.

    ``Nk = INT( 2000.0 * RMax * ( kMax - kMin ) / pi )`` with ``RMax`` in km
    and ``kMax - kMin = 2*pi*freqVec(Nfreq)*(1/cLow - 1/cHigh)`` from
    ``scooter.f90:67-68``, which reduces to
    ``4000 * RMax_km * f_max * (1/cLow - 1/cHigh)``.

    The deck's own rounding is applied first — ``write_phase_speed_and_rmax``
    writes RMax as ``%.6f`` km and the phase-speed pair as ``%.1f`` — so this
    reproduces the binary's count rather than an unrounded neighbour of it.
    """
    rmax_km = round(float(rmax_m) / 1000.0, 6)
    cl = round(float(c_low), 1)
    ch = round(float(c_high), 1)
    return int(4000.0 * rmax_km * float(f_max) * (1.0 / cl - 1.0 / ch))


#: Ceiling on the complex64 Green's-function cube a Scooter run commits
#: ``read_grn_file`` to allocating whole (``grn_reader.py``:
#: ``np.zeros((nfreq, nsd, nrd, nk), dtype=np.complex64)``): the same 2 GiB
#: budget SPARC's snapshot cap puts on its own Green cube
#: (``sparc._MAX_SNAPSHOT_GREEN_BYTES``).
_MAX_GREEN_CUBE_BYTES = 2 * 1024 ** 3

# Bytes per (nk x nr) element the k->r transform holds at peak: the complex128
# phase array it exponentiates in place (16) plus the complex64 kernel that
# array is cast to (8). Measured against peak RSS at nk x nr = 120000 x 499,
# an estimate built on this bounds the real peak by 1.17x.
_TRANSFORM_BYTES_PER_ELEMENT = 24


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
    taper : float, optional
        Hanning roll-off applied to the wavenumber kernel over this fraction
        of the spectral span at EACH edge, before the Hankel transform
        (``fieldsco.m:taper``). Default ``0`` — OFF, matching the reference
        implementation, which sets ``cmin=1e-10, cmax=1e30`` and calls the
        feature "user play (at your own risk)"
        (``Matlab/Scooter/fieldsco.m:23-32``). COA Sect. 4.5 gives the reason
        to taper: the wavenumber beyond which the integrand is negligible
        "will depend on range, and for multiple ranges it is not desirable to
        truncate at different wavenumbers", so the kernel is instead "forced
        to gradually vanish" at one fixed edge. The sidelobe rates are
        standard windowing rather than COA: 6 dB/octave for a rectangular
        edge against 18 for a Hann one (Abraham, *Underwater Acoustic
        Signal Processing*, Sect. 4.10), i.e. ``1/x`` against ``1/x^3``.

        The fraction is taken in ``k`` and the bounds are applied as phase
        speeds, so they are identical at every frequency of a broadband
        sweep. Tapering smooths an edge; it cannot recover
        spectrum that was never computed, so widen ``c_high`` as well when the
        field is built within a few wavelengths of a boundary, where steep and
        evanescent components still carry energy at the cut.

        COA's criterion is that the taper span "several periods" of
        ``exp(i k r)``, i.e. ``taper * (kMax-kMin) * r_max >> 2*pi``. One
        period is ``2*pi / ((kMax-kMin) * r_max)`` of the band: with the
        default window that is a few per cent at a few kilometres and shrinks
        as ``1/r_max`` — parts in a thousand at tens of kilometres.
        Values far above that attenuate real spectrum, which is occasionally
        what you want (the far evanescent tail is the least well conditioned
        part of the solve) and is never free. Measure it against something
        independent before adopting one.
    spectrum : str, optional
        FLP Opt(2): ``'positive'`` (fast, default) | ``'negative'`` | ``'both'``.
        This is the WAVENUMBER spectrum half — which side of the real k-axis
        the spectral integral covers. Not to be confused with
        :class:`~uacpy.models.OASS` / :class:`~uacpy.models.OASSP`'s
        ``spectrum=``, which selects a ROUGHNESS power-spectrum model
        (``'gaussian'`` / ``'goff-jordan'``). The two share a name and nothing
        else.
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

    # TopOpt position 4 carries env.absorption to the engine.
    _consumes_volume_absorption = True

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
        supports={'layered_bottom', 'elastic_media', 'rough_surface',
                  'multi_source_depth'},
        source_types=frozenset({'point', 'line', 'scaled'}),
        collapse={'ssp': 'mean', 'bottom_range': 'median'},
    )
    source = 'acoustics_toolbox'

    # scooter.exe solves the depth-separated equation for every source depth
    # of the deck in one wavenumber sweep (the .grn carries an NSz axis) and
    # grn_to_field transforms one depth at a time — so COHERENT_TL takes a
    # multi-depth Source in one launch. BROADBAND / TIME_SERIES go through
    # the base's per-depth loop.
    _NATIVE_MULTI_DEPTH_MODES = frozenset({RunMode.COHERENT_TL})

    def __init__(
        self,
        executable: Optional[Path] = None,
        c_low: Optional[float] = None,
        c_high: Optional[float] = None,
        n_mesh: int = 0,
        rmax_multiplier: Optional[float] = None,
        interp_ssp: Optional[str] = None,
        spectrum: str = 'positive',
        taper: float = 0.0,
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
        taper = 0.0 if taper is None else taper
        if not (0.0 <= float(taper) < 0.5):
            raise ConfigurationError(
                f"Scooter: taper is the fraction of the wavenumber span "
                f"rolled off at EACH edge, so it must satisfy "
                f"0 <= taper < 0.5; got {taper!r}."
            )
        self.taper = float(taper)
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

    def _run_single(
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
    ) -> Union[Result, ResultStack]:
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
        self._require_run_triple(env, source, receiver)
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

        # A pinned n_mesh is checked at the deck's freq0 — the frequency AT
        # applies its own floor at. misc/ReadEnvironmentMod.f90:103-112 sizes
        # Nneeded from freq0 during the environment read, and scooter.f90:106
        # then scales N with freq/freq0 for every swept frequency, so a mesh
        # that clears the floor at freq0 stays proportionally as fine across
        # the whole sweep. Testing max(freq) instead rejected meshes the
        # binary would have run.
        reject_coarse_at_mesh(
            'Scooter', self.n_mesh, env,
            float(np.atleast_1d(
                np.asarray(source.frequencies, dtype=float))[0]))

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
            slabs = _slabs_of(result)
            sources = ([source.at_depth(i) for i in range(len(slabs))]
                       if len(slabs) > 1 else [source])
            for slab, slab_source in zip(slabs, sources):
                if source.source_type == 'line':
                    # The 'X' Hankel path returns 1/√(k0·R) in free space
                    # (grn_reader: no √k weighting, 1/√(2π)); ×√k0 is unit
                    # amplitude at 1 m, the package's line-source level.
                    freqs_out = (np.asarray(slab.coords['frequency'],
                                            dtype=float)
                                 if 'frequency' in slab.coords
                                 else np.atleast_1d(source.frequencies)[0])
                    # ``slab_source``, not ``source``: the level is
                    # sqrt(k0) at the SOURCE depth, and on a multi-depth
                    # stack ``source`` still carries every depth, whose
                    # first one is not this slab's.
                    level = _line_source_unit_at_1m(
                        _source_sound_speed(env, slab_source), freqs_out)
                    slab.data = slab.data * (level if level.size == 1
                                             else level[None, None, :])

                freqs = (broadband_freqs if broadband_mode
                         else float(source.frequencies[0]))
                self._stamp_result(slab, slab_source, backend='scooter',
                                   frequencies=freqs,
                                   phase_reference='travelling_wave')
                # Physical fastest compressional speed in the waveguide
                # (water column + sediment + half-space): the time-series
                # synthesis helpers anchor their output window at r / c_max,
                # ahead of the earliest bottom-refracted arrival.
                c_max = self._resolve_c_max(env)
                if c_max is not None:
                    slab.metadata['c_max'] = c_max
                # The taper changes the field by a decibel or two and leaves
                # no other trace, so two otherwise identical results are only
                # distinguishable by it.
                slab.metadata['taper'] = self.taper

                self._attach_output_paths(
                    slab, fm.work_dir, base_name,
                    primary_files=(('grn_file', '.grn'),),
                )

            self._log("Simulation complete")
            if isinstance(result, ResultStack):
                # Narrowband only (the broadband modes loop per depth), so
                # there is no synthesis to finish.
                result.slabs = [
                    self._mask_unresolvable_depths(slab, receiver, media_depth)
                    for slab in result.slabs]
                return result
            result = self._finish_broadband(
                result, run_mode, source_waveform, sample_rate)
            return self._mask_unresolvable_depths(
                result, receiver, media_depth)

        finally:
            fm.finish()

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

    _receivers_reach_sediment = True    # meshes through the sediment stack

    def _run_and_read_grn(self, fm, base_name):
        """Run the binary and read back its Green's function.

        A missing ``.grn`` and one carrying fewer than two wavenumber samples
        are both run failures the binary does not report through its exit
        status, so each is turned into a typed error carrying the ``.prt``
        diagnostics. ``_write_scooter_env`` refuses ``Nk < 2`` before the
        launch; this is the backstop for a count the deck-side arithmetic did
        not predict.
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
        nk = int(grn_data['nk'])
        if nk < 2:
            exc = ModelExecutionError(
                self.model_name, return_code=0, stdout=None,
                stderr=(
                    f"Scooter produced a Green's function with nk={nk} "
                    f"wavenumber sample(s); {_nk_consequence(nk)}"
                ),
            )
            self._attach_prt_tail(exc, fm.work_dir, base_name)
            raise exc
        return grn_data

    def _taper_bounds(self, grn_data):
        """``(cmin, cmax)`` phase-speed bounds for the kernel taper.

        COA Sect. 4.5 prescribes the cure: since the wavenumber beyond which
        the integrand is negligible "will depend on range, and for multiple
        ranges it is not desirable to truncate at different wavenumbers", the
        kernel is "forced to gradually vanish" at one fixed edge. The roll-off
        is Hanning (:func:`~uacpy.io.grn_reader._hanning_taper`, mirroring
        ``fieldsco.m:taper``); a rectangular edge's sidelobes fall at
        6 dB/octave against a Hann edge's 18 (Abraham, Sect. 4.10), i.e.
        ``1/x`` against ``1/x^3``.

        ``taper`` is a fraction of the wavenumber span applied at EACH edge.
        The fraction is taken in ``k`` while the bounds are returned as phase
        speeds, so ``omega`` cancels: they depend only on the deck's own
        ``c_low`` / ``c_high`` and are identical at every frequency of a
        broadband sweep. ``taper=0`` disables it and restores the rectangular
        cut.

        COA also warns that the choice of the truncation point itself "is not
        easily automated" -- tapering smooths an edge, it does not recover
        spectrum that was never computed. Widen ``c_high`` when the field is
        built close to a boundary, where steep and evanescent components still
        carry energy at the edge.
        """
        taper = self.taper
        if taper <= 0.0:
            return None, None
        c = np.asarray(grn_data['cVec'], dtype=float)
        if c.size and not np.all(np.isfinite(c) & (c > 0.0)):
            # ``_hanning_taper`` indexes the RAW grid, so filtering here and
            # letting it index the unfiltered one turns a bad .grn into a
            # numpy ValueError or ZeroDivisionError from inside the transform.
            raise ConfigurationError(
                f"Scooter(taper={taper:.4g}): the Green's function's "
                f"phase-speed grid holds non-finite or non-positive values, "
                f"so the taper's edges cannot be located. Re-run the solver, "
                f"or pass taper=0 to transform it untapered.",
                remediation="A .grn with a corrupt cVec usually means the "
                            "run was interrupted; delete it and re-run.",
            )
        if c.size < 4:
            warnings.warn(
                f"Scooter(taper={taper:.4g}) was requested but the "
                f"Green's function's phase-speed grid holds only {c.size} "
                f"value(s), too few to place a roll-off. The transform runs "
                f"untapered, i.e. as taper=0.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
            return None, None
        inv_lo, inv_hi = 1.0 / c.max(), 1.0 / c.min()   # k/omega at each edge
        span = inv_hi - inv_lo
        if span <= 0.0:
            warnings.warn(
                f"Scooter(taper={taper:.4g}) was requested but the "
                f"Green's function's phase-speed grid spans a single speed, "
                f"so there is no edge to roll off. The transform runs "
                f"untapered, i.e. as taper=0.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
            return None, None
        return (1.0 / (inv_hi - taper * span),
                1.0 / (inv_lo + taper * span))

    def _assemble_field_from_grn(self, grn_data, source, receiver,
                                 broadband_mode):
        """Hankel-transform the Green's function onto the receiver ranges.

        Broadband transforms every frequency in the ``.grn`` at once; the
        narrowband path transforms the single frequency slice.
        """
        cmin, cmax = self._taper_bounds(grn_data)
        if cmin is not None:
            self._log(f"Kernel taper: Hanning roll-off over "
                      f"{self.taper:.4g} of the wavenumber span at each edge "
                      f"(pass band {cmin:.1f}-{cmax:.1f} m/s)")
        transform_kwargs = dict(
            source_type=_SOURCE_TYPE_CODE[source.source_type],
            spectrum=self._spectrum_code,
            cmin=cmin, cmax=cmax,
        )
        if broadband_mode:
            self._log(f"Transforming {grn_data['nfreq']} frequencies to "
                      f"range domain...")
            return grn_to_transfer_function(
                grn_data, receiver.ranges, **transform_kwargs)
        self._log("Transforming to range domain (direct-DFT Hankel transform)...")
        nsd = int(grn_data['nsd'])
        if nsd == 1:
            return grn_to_field(
                grn_data, receiver.ranges, method='direct_dft',
                **transform_kwargs)
        # One slab per source depth of the deck, on the Source's own depth
        # axis (the .grn stores the depths in float32).
        depths = np.atleast_1d(np.asarray(source.depths, dtype=float))
        if depths.size != nsd:
            raise ModelExecutionError(
                self.model_name, return_code=0, stdout=None,
                stderr=(f"the Green's function holds {nsd} source depths for "
                        f"a Source of {depths.size}"))
        return ResultStack(
            [grn_to_field(grn_data, receiver.ranges, method='direct_dft',
                          source_depth_idx=i, **transform_kwargs)
             for i in range(nsd)],
            coordinate=depths, coordinate_name='source_depth')

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

        No receiver *ranges* are written: ``scooter.f90``'s ``GetPar``
        (``:154-178``) reads the environment, then ``ReadSzRz`` and
        ``ReadfreqVec``, and never calls ``ReadRcvrRanges`` — so the deck
        continues past ``RMax`` with the source depths, the receiver depths
        and (broadband only) the frequency vector, and simply stops there.
        The range axis is applied in-tree when the ``.grn`` is transformed.
        """
        ssp_topopt = resolve_ssp_topopt(env, self.interp_ssp)
        surface_type = parse_boundary_type(env.surface.acoustic_type)
        bottom_type = parse_boundary_type(env.bottom.halfspace_at(range=0.0).acoustic_type)

        # TopOpt position 7: '0' zeroes out Scooter's stabilising attenuation
        # (scooter.f90:81,129-130). Leave as ' ' otherwise — Scooter keeps
        # Atten=Deltak, the default stabiliser.
        topopt_extra = '0' if self.stabilizing_attenuation_off else ''

        # Scooter's spectral RMax derives from the maximum receiver range
        # (RMax = range_max × multiplier), and scooter.f90:69 sizes the
        # wavenumber grid from RMax — a non-positive value stops the binary
        # with a bare STOP that names no cause, so it is refused here.
        if receiver.range_max <= 0.0:
            raise ConfigurationError(
                f"Scooter requires a positive receiver range: the spectral "
                f"RMax is receiver.range_max × rmax_multiplier, and "
                f"receiver.range_max = {receiver.range_max:.6g} m would "
                f"write RMax = 0, which scooter.exe rejects with an "
                f"unexplained STOP.",
                remediation="Pass a Receiver with at least one range > 0 m.",
            )
        rmax_m = float(receiver.ranges.max()) * self._resolve_rmax_multiplier(run_mode)
        cl, ch = resolve_phase_speed_bounds(env, self.c_low, self.c_high)
        # The constructor (:195-199) can only compare two pinned bounds. A
        # single pinned bound is only comparable once the other has been
        # derived from this env, and an inverted pair reaches
        # ReadEnvironmentMod.f90:135, which stops the binary after the deck
        # has been written and the process spawned. Catch it here instead and
        # name the bound the user pinned, since that is the one to move.
        if cl >= ch:
            if self.c_low is not None and self.c_high is None:
                pinned = (f"the pinned c_low={self.c_low} m/s is at or above "
                          f"the env-derived c_high = {ch:.1f} m/s")
            elif self.c_high is not None and self.c_low is None:
                pinned = (f"the pinned c_high={self.c_high} m/s is at or "
                          f"below the env-derived c_low = {cl:.1f} m/s")
            else:
                pinned = (f"the env-derived band collapsed to c_low = "
                          f"{cl:.1f} m/s, c_high = {ch:.1f} m/s")
            raise ConfigurationError(
                f"Scooter spectral phase-velocity band requires "
                f"c_low < c_high: {pinned}.",
                remediation="Widen the pinned bound, or leave both unset to "
                            "derive the band from the SSP and bottom.",
            )
        if self.c_low is None or self.c_high is None:
            self._log(
                f"c_low / c_high auto-derived = "
                f"{cl:.1f} / {ch:.1f} m/s"
            )

        # Refuse a wavenumber count the binary cannot space, before the deck
        # is written: scooter.f90 has no Nk test of its own (the only IF
        # naming it is the allocation status at :74), so an Nk below 2 runs
        # to completion at exit 0 and reaches the caller as a full-shape
        # result. The two siblings that derive the same quantity refuse it
        # pre-launch for the same reason — see ``Bounce._n_ktab`` and the Nk
        # guard inside ``SPARC._write_sparc_env``.
        f_deck = self._deck_max_frequency(source, frequencies)
        nk = _deck_nk(rmax_m, f_deck, cl, ch)
        if nk < 2:
            raise ConfigurationError(
                f"This deck asks Scooter for Nk = {nk} wavenumber sample(s): "
                f"scooter.f90:69 derives Nk = INT(2000 * RMax_km * "
                f"(kMax - kMin) / pi) from RMax = {rmax_m:g} m at "
                f"{f_deck:.6g} Hz with "
                f"c_low = {cl:.1f} and c_high = {ch:.1f} m/s. "
                f"{_nk_consequence(nk)}",
                remediation=(
                    "Nk grows with RMax, frequency and the width of the "
                    "phase-speed window: raise rmax_multiplier (RMax = "
                    "receiver.ranges.max() x rmax_multiplier), raise the "
                    "source frequency, or widen c_low/c_high. Lengthening "
                    "the receiver ranges also raises RMax; shortening them "
                    "lowers it."
                ),
            )

        # The deck writes the frequency vector only when it holds more than
        # one entry (``write_broadband_freqs``); otherwise the ``.grn``
        # carries the single header frequency — the same branch
        # ``_deck_max_frequency`` mirrors.
        freqs_deck = (np.atleast_1d(frequencies)
                      if frequencies is not None else None)
        n_freqs = (len(freqs_deck)
                   if freqs_deck is not None and len(freqs_deck) > 1 else 1)
        self._reject_oversized_green_cube(
            nk, n_freqs,
            int(np.atleast_1d(np.asarray(source.depths)).size),
            int(np.atleast_1d(np.asarray(receiver.depths)).size),
            int(np.atleast_1d(np.asarray(receiver.ranges)).size),
            rmax_m=rmax_m, f_deck=f_deck, c_low=cl, c_high=ch,
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

    @staticmethod
    def _deck_max_frequency(source, frequencies):
        """``freqVec( Nfreq )`` for the deck ``_write_scooter_env`` writes.

        ``scooter.f90:67-68`` sizes the wavenumber grid from the *last* entry
        of the frequency vector, so this mirrors the writer's own branch:
        ``write_broadband_freqs`` emits the vector only when it holds more
        than one entry, and otherwise the deck carries the single header
        frequency ``write_header`` takes from ``source.frequencies[0]``.
        """
        if frequencies is not None:
            freqs = np.atleast_1d(frequencies)
            if len(freqs) > 1:
                return float(freqs[-1])
        return float(source.frequencies[0])

    def _reject_oversized_green_cube(
        self, nk: int, n_freqs: int, n_source_depths: int,
        n_receiver_depths: int, n_ranges: int, *, rmax_m: float,
        f_deck: float, c_low: float, c_high: float,
    ) -> None:
        """Warn — or refuse — on the memory this deck commits the run to.

        ``scooter.exe`` holds one frequency's ``Green(NSz, NRz, Nk)`` at a
        time, but the ``.grn`` accumulates every frequency and
        ``read_grn_file`` allocates the whole ``(nfreq, nsd, nrd, nk)``
        complex64 cube in one ``np.zeros`` — the Python process, not the
        binary, takes the hit. ``Nk`` grows linearly with RMax
        (``receiver.ranges.max() × rmax_multiplier``), the top deck frequency
        and the phase-speed span, so a plausible broadband deck reaches tens
        of GB with no single knob looking unreasonable.

        The cube is not the whole bill, so counting it alone under-reads the
        peak: :func:`~uacpy.io.grn_reader._hankel_transform` also builds
        ``outer(k, r)`` in double and exponentiates it in place, then casts the
        result down, so the kernel costs another ``nk × nr × 24`` bytes (16 for
        the complex128 phase, 8 for the complex64 copy it becomes) on top of a
        second copy of the cube. All of that is estimated here. Measured
        against peak RSS on an ``nk x nr`` of 120000 x 499, the estimate is
        1.17x the real peak — tight, and on the safe side.

        It is measured against what the host actually has free rather than a
        fixed constant: a 3 GiB cube is nothing on a 64 GiB workstation and
        fatal on a 4 GiB laptop. Over half of ``MemAvailable`` warns; over all
        of it raises, because that one cannot be made to work by waiting. With
        no reading of the host's memory available, a fixed cap applies.
        """
        cube = (8 * int(n_freqs) * int(n_source_depths)
                * int(n_receiver_depths) * int(nk))
        kernel = (_TRANSFORM_BYTES_PER_ELEMENT * int(nk)
                  * max(int(n_ranges), 1))
        peak = 2 * cube + kernel
        detail = (
            f"Nk = {int(nk)} wavenumber samples: a "
            f"{cube / 1024 ** 3:.1f} GiB complex64 Green's-function cube "
            f"({int(n_freqs)} frequencies x {int(n_source_depths)} source "
            f"depth(s) x {int(n_receiver_depths)} receiver depth(s) x Nk x "
            f"8 B) and a {kernel / 1024 ** 3:.1f} GiB transform kernel "
            f"({int(nk)} x {int(n_ranges)} ranges), about "
            f"{peak / 1024 ** 3:.1f} GiB at peak. scooter.f90:69 derives "
            f"Nk = INT(2000 * RMax_km * (kMax - kMin) / pi) from "
            f"RMax = {rmax_m:g} m at {f_deck:.6g} Hz with "
            f"c_low = {c_low:.1f} and c_high = {c_high:.1f} m/s."
        )
        advice = (
            "Peak memory scales with Nk, which grows with RMax = "
            "receiver.ranges.max() x rmax_multiplier, with the top deck "
            "frequency and with the width of the c_low/c_high phase-speed "
            "window. Fewer receiver depths or ranges shrink it too."
        )
        avail = available_memory_bytes()
        # Both ways past the budget end at one raise, so ``advice`` is written
        # once and the warning below shares it. ``over`` is assigned in each
        # branch rather than built as a conditional inside a concatenation,
        # which is the shape that silently drops a description.
        if avail is None:
            if cube <= _MAX_GREEN_CUBE_BYTES:
                return
            over = (
                f"The host's free memory could not be read, so uacpy falls "
                f"back to a fixed "
                f"{_MAX_GREEN_CUBE_BYTES / 1024 ** 3:.1f} GiB cube cap."
            )
        elif peak > avail:
            over = (f"That is more than the {avail / 1024 ** 3:.1f} GiB this "
                    f"host reports free.")
        else:
            if peak > 0.5 * avail:
                warnings.warn(
                    f"Scooter: {detail} That is over half the "
                    f"{avail / 1024 ** 3:.1f} GiB this host reports free; the "
                    f"run should complete but leaves little headroom. "
                    f"{advice}",
                    UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
                )
            return
        raise ConfigurationError(
            f"This deck asks Scooter for {detail} {over}",
            remediation=advice,
        )

    def _run_scooter(self, base_name: str, work_dir: Path):
        """Execute Scooter via the shared binary-launch helper."""
        self._run_and_attach_prt([str(self._exe), base_name], work_dir, base_name,
                                 stale_outputs=('.grn',))
