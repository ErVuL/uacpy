"""The :class:`Scooter` engine class: its knobs, its declarations and the
protocol hooks that project, validate, resolve, write, launch, read and
transform one Scooter deck."""

import warnings
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Optional, Union

import numpy as np

from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, ModelExecutionError, NumericsWarning,
)
from uacpy.models.base import DEFAULT_RUN_TIMEOUT_S, PropagationModel
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.models._conventions import (
    _line_source_unit_at_1m, _source_sound_speed,
)
from uacpy.models._checks import reject_precalc_surface
from uacpy.models._projection import _max_roughness, _smooth_surface
from uacpy.models._spec import EngineTraits, ModelSpec
from uacpy.models._stacking import _slabs_of
from uacpy.core.run_settings import OutputSpec, RunMode
from uacpy.core.results import GreensFunction, PhaseReference, ResultStack
from uacpy.io.at_codes import boundary_code
from uacpy.io.grn_reader import read_grn_file
from uacpy.io.oalib_writer import (
    write_scooter_env_file, reject_coarse_at_mesh,
    reject_unsupported_ssp_interp,
    reject_biological_edges_under_neighbour_interp,
)
from uacpy.models.scooter._extract import assemble_field_from_grn
from uacpy.models._window import resolve_window
from uacpy.models.scooter._plan import (
    _BROADBAND_MODES, _deck_nk,
    _nk_consequence, check_knobs, deck_max_frequency,
    green_cube_axes, mesh_floor_notice, refuse_nonpositive_range,
    refuse_too_few_wavenumbers, reject_oversized_green_cube,
    resolve_rmax_factor, steep_path_notice,
)
from uacpy.models.scooter._settings import ScooterSettings
from uacpy.core.engine_defaults import SCOOTER_N_MESH

# File root of the deck and of every file the binary writes next to it.
_BASE_NAME = 'model'


class Scooter(PropagationModel):
    """
    Scooter finite element FFP (Fast Field Program) model

    Frequency-domain solver for underwater acoustics.
    Developed by Michael B. Porter.

    Notes
    -----
    Range-independent FFP — single spectral solve over the full
    wavenumber axis, Hankel-transformed to range. Supports layered
    and elastic bottoms natively. The Green's-function
    ``.grn`` is converted to range-domain TL via the in-tree Python
    Hankel transform (:meth:`~uacpy.core.results.GreensFunction.to_field`).

    **Collapse defaults (overrides of :data:`DEFAULT_COLLAPSE`).**
    Per-model: ``'ssp': 'mean'``, ``'bottom_range': 'median'`` (the layer
    stack is kept since Scooter consumes layered seabed columns natively).

    Defaults auto-derived at ``run()`` time:

    - ``c_low=None`` → ``min(env.ssp) × 0.95``
    - ``c_high=None`` → ``max(max(env.ssp), env.bottom.sound_speed) × 1.05``,
      or the unbounded sentinel for a vacuum, rigid, ``'file'`` or
      ``'precalc'`` bottom
    - Spectral ``RMax = receiver.range_max × rmax_factor``
    - ``n_mesh=0`` → Scooter picks from frequency / wavelength.
    - TopOpt position 4 reads ``env.absorption``.

    With ``verbose='info'`` the resolved ``c_low`` / ``c_high`` are logged.
    ``Scooter().run_settings(env, source, receiver).engine`` is the
    :class:`ScooterSettings` a run would use — the phase-speed window and
    where each bound came from, ``RMax`` and its multiplier, the mesh, the
    ``Nk`` count — without launching anything; every result carries its own
    as ``result.run_settings.engine``.

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
        traits=EngineTraits(
            consumes_run_t_start=True,
            # TopOpt position 4 carries env.absorption to the engine.
            consumes_volume_absorption=True,
            # Meshes through the sediment stack.
            receivers_reach_sediment=True,
            # scooter.exe solves the depth-separated equation for every
            # source depth of the deck in one wavenumber sweep (the .grn
            # carries an NSz axis) and GreensFunction.to_field transforms
            # one depth at a time — so COHERENT_TL takes a multi-depth
            # Source in one launch. BROADBAND / TIME_SERIES go through the
            # base's per-depth loop.
            native_multi_depth_modes=frozenset({RunMode.COHERENT_TL}),
        ),
    )
    provenance_id = 'acoustics_toolbox'
    # Complex pressure in the travelling-wave convention of the .grn
    # transform for the TL and H(f) modes; the synthesised p(t) is real.
    outputs = MappingProxyType({
        RunMode.COHERENT_TL: OutputSpec(
            'Field', kind='pressure', unit='Pa',
            phase_reference=PhaseReference.TRAVELLING_WAVE.value,
            coherent=True),
        RunMode.BROADBAND: OutputSpec(
            'Field', kind='pressure', unit='Pa',
            phase_reference=PhaseReference.TRAVELLING_WAVE.value,
            coherent=True),
        RunMode.TIME_SERIES: OutputSpec(
            'Field', kind='pressure', unit='Pa',
            phase_reference=PhaseReference.TIME_DOMAIN_NATIVE.value),
    })

    def __init__(
        self,
        *,
        executable: Optional[Path] = None,
        c_low: Optional[float] = None,
        c_high: Optional[float] = None,
        n_mesh: int = SCOOTER_N_MESH,
        rmax_factor: Optional[float] = None,
        interp_ssp: Optional[str] = None,
        wavenumber_spectrum: str = 'positive',
        taper: float = 0.0,
        stabilizing_attenuation_off: bool = False,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        cleanup: Optional[bool] = None,
        timeout: float = DEFAULT_RUN_TIMEOUT_S,
        collapse: Optional[Dict[str, str]] = None,
    ):
        """
        Parameters
        ----------
        executable : Path, optional
            Path to ``scooter.exe``. Auto-detected if ``None``.
        c_low : float, optional
            Lower phase speed limit (m/s). None = auto (0.95 * min SSP
            speed). The auto value reads the water only, so seabed
            interface (Scholte) waves slower than it are outside the
            integration window: for a source or receiver near an elastic
            seabed, set ``c_low`` below the slowest shear speed
            (``docs/models/scooter.md``, "Layered and elastic seabeds").
        c_low, c_high : float, optional
            Phase-speed bounds (m/s). ``None`` ⇒ ``0.95 × min SSP`` /
            ``1.05 × max(SSP, bottom)``; a vacuum, rigid, ``'file'`` or
            ``'precalc'`` bottom has no half-space speed to cap on, so ``c_high``
            resolves to the AT "unbounded" sentinel (1e9 m/s) instead (see
            :func:`~uacpy.models._window.resolve_phase_speed_bounds`). The
            default ``c_low`` reads the water only, so the interface (Scholte)
            waves an elastic seabed carries below it are left out: for a source
            or receiver near the seabed, set ``c_low`` below the slowest shear
            speed (``docs/models/scooter.md``, "Layered and elastic seabeds").
        c_high : float, optional
            Upper phase speed limit (m/s). None = auto (1.05 * max of SSP and
            bottom speed; unbounded, 1e9 m/s, for a vacuum, rigid, 'file' or
            'precalc' bottom, which has no half-space speed).
        n_mesh : int, optional
            Total number of mesh points PER MEDIUM used by the finite-element
            spectral solver (AT's ``NG`` column on the SSP mesh line). 0 = let
            Scooter pick automatically from frequency / wavelength. Default: 0.
            Note: this is NOT a "points per wavelength" density — it is a total
            point count per medium.
            Default ``0``.
        rmax_factor : float, optional
            Multiplier on ``receiver.ranges.max()`` to set Scooter's spectral
            ``RMax``, which fixes the wavenumber sampling:
            ``scooter.f90:69`` takes ``Nk = INT(2000·RMax_km·(kMax−kMin)/π)``,
            i.e. ``Δk ≈ π/(2·RMax_m)``. Default ``None`` → 2.0 for
            ``COHERENT_TL``, 3.0 for ``BROADBAND`` / ``TIME_SERIES``.
        interp_ssp : str, optional
            SSP connection scheme written into ``TopOpt(1)``. ``None``
            (default) resolves to ``'linear'`` (C-linear): Scooter declares
            no range-dependent-SSP capability, so ``run()`` collapses
            ``env.ssp`` to 1-D before the deck is written and the
            range-dependent auto-pick never applies. Explicit values:
            ``'linear'``, ``'n2linear'``, ``'pchip'``,
            ``'spline'``. ``env.ssp.kind='isovelocity'`` always forces
            ``'C'`` regardless.
        wavenumber_spectrum : {'positive', 'negative', 'both'}, optional
            FLP Option(2:2) in AT nomenclature. uacpy writes no ``.flp`` —
            the letter is applied by the in-tree Hankel transform, which
            follows ``Matlab/Scooter/fieldsco.m:170-181``. 'positive'
            (default) uses only the positive wavenumber spectrum (fast,
            recommended). 'negative' uses only the negative branch; 'both'
            integrates along the full k-axis.
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
        stabilizing_attenuation_off : bool, optional
            If True, writes ``'0'`` at TopOpt position 7. Scooter then
            replaces its default ``Atten = Deltak`` with zero
            (``scooter.f90:81,129-130``). Leave False (default) unless you
            know what you're doing — the stabiliser is there to prevent
            pole-on-contour blow-ups.
            Default ``False``; leave it unless you know what you're doing (the
            stabiliser prevents pole-on-contour blow-ups).
        use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
            Standard plumbing (see :class:`PropagationModel`).
        """
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
            cleanup=cleanup, timeout=timeout, collapse=collapse,
        )

        self.c_low = c_low
        self.c_high = c_high
        self.taper = 0.0 if taper is None else taper
        self.interp_ssp = interp_ssp
        self.n_mesh = n_mesh
        self.rmax_factor = rmax_factor
        self.wavenumber_spectrum = wavenumber_spectrum
        self._check_knobs()
        self.taper = float(self.taper)

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
                NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP,
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

    def _project_environment(self, env, *, request=None):
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
        env = super()._project_environment(env, request=request)
        sigma = _max_roughness(env.surface.nodes)
        if not sigma:
            return env
        acoustic_type = env.surface.acoustic_type
        if boundary_code(acoustic_type) == 'V':
            return env
        env.surface = _smooth_surface(env.surface)
        warnings.warn(
            f"{self.model_name} reads the sea-surface roughness only for a "
            f"pressure-release (vacuum) surface; the {acoustic_type!r} surface "
            f"takes an impedance that never touches SSP%sigma(1), so "
            f"env.surface.roughness={sigma:g} m was dropped.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
        return env

    def _check_knobs(self) -> None:
        """Refuse a knob no run could use
        (:func:`~uacpy.models.scooter._plan.check_knobs`)."""
        check_knobs(taper=self.taper, c_low=self.c_low,
                    c_high=self.c_high,
                    wavenumber_spectrum=self.wavenumber_spectrum)

    def _validate_engine(self, env, source, receiver, run_mode, *,
                         request=None) -> None:
        """Stage 2: the refusals of carriers Scooter cannot run — on the
        projected environment, so ``validate_inputs`` refuses what ``run``
        refuses.

        - an ``interp_ssp`` the Scooter deck cannot carry;
        - a ``'precalc'`` sea surface: scooter.exe interpolates a top
          ``.irc`` table it never reads and dies with SIGSEGV;
        - a ``'precalc'`` seabed whose ``.irc`` table is not in the layout
          the binary reads: staged verbatim as ``<base>.irc``, a table in the
          wrong layout (typically a theta/|R|/phase angle table) aborts the
          binary with a bare Fortran backtrace;
        - a receiver with no positive range: the spectral ``RMax`` is
          ``receiver.range_max × rmax_factor``, and ``scooter.f90:69``
          sizes the wavenumber grid from it — a non-positive value stops the
          binary with a bare STOP that names no cause.
        """
        reject_precalc_surface(env, model_name=self.model_name)
        reject_unsupported_ssp_interp('Scooter', self.interp_ssp)
        reject_biological_edges_under_neighbour_interp(
            'Scooter', env, self.interp_ssp)
        self._reject_malformed_irc_bottom(env)
        refuse_nonpositive_range(receiver)

    def _resolve_engine_settings(self, env, source, receiver, settings, *,
                                 given_env, request=None) -> ScooterSettings:
        """Stage 3: every setting of the deck, resolved once from the
        projected ``env``, and the refusals of a deck Scooter cannot run.

        Refused: a pinned ``n_mesh`` under AT's floor at the deck's
        ``freq0``, a phase-speed window with ``c_low >= c_high`` once the
        unpinned bound is derived, fewer than two wavenumber samples, a
        Green's-function cube over the memory this host has free.
        """
        broadband = settings.mode in _BROADBAND_MODES
        deck_frequencies = settings.frequencies if broadband else None
        if broadband:
            if deck_frequencies is None:
                raise ConfigurationError(
                    f"Scooter.run(run_mode={settings.mode.name}): the source "
                    f"pulse implies no frequency grid (it needs at least two "
                    f"samples).",
                    remediation="Pass frequencies= to set the grid, or a "
                                "source_waveform of two samples or more.")
            self._log(f"Broadband: {len(deck_frequencies)} frequencies, "
                      f"{deck_frequencies[0]:.1f}-"
                      f"{deck_frequencies[-1]:.1f} Hz")

        # A pinned n_mesh is checked at the deck's freq0 — the frequency AT
        # applies its own floor at. misc/ReadEnvironmentMod.f90:103-112 sizes
        # Nneeded from freq0 during the environment read, and scooter.f90:106
        # then scales N with freq/freq0 for every swept frequency, so a mesh
        # that clears the floor at freq0 stays proportionally as fine across
        # the whole sweep. Testing max(freq) instead rejected meshes the
        # binary would have run.
        freq0 = float(np.atleast_1d(
            np.asarray(source.frequencies, dtype=float))[0])
        reject_coarse_at_mesh('Scooter', self.n_mesh, env, freq0)

        multiplier, multiplier_origin = resolve_rmax_factor(
            settings.mode, rmax_factor=self.rmax_factor)
        rmax_m = float(receiver.ranges.max()) * multiplier
        window = resolve_window(env, c_low=self.c_low, c_high=self.c_high,
                                model_name='Scooter')
        cl, ch = window.c_low, window.c_high
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
        # pre-launch for the same reason — see
        # ``bounce._plan.tabulated_angle_count`` and the Nk guard inside
        # ``SPARC._resolve_engine_settings``.
        f_deck = deck_max_frequency(source, deck_frequencies)
        nk = _deck_nk(rmax_m, f_deck, cl, ch)
        refuse_too_few_wavenumbers(nk, rmax_m, f_deck, cl, ch)

        peak, headroom = reject_oversized_green_cube(
            nk, *green_cube_axes(source, receiver, settings),
            rmax_m=rmax_m, f_deck=f_deck, c_low=cl, c_high=ch,
        )

        # The conditions the run warns about, decided here once; the
        # announcement (``_announce_engine_settings``) only presents them.
        notices = [
            mesh_floor_notice(
                freq0, deck_frequencies if broadband else [freq0],
                n_mesh=self.n_mesh),
            (None if self.c_high is not None else
             steep_path_notice(env, source, receiver, ch)),
            headroom,
        ]
        notices = [n for n in notices if n is not None]
        return ScooterSettings(
            c_low=cl,
            c_low_origin=window.c_low_origin,
            c_high=ch,
            c_high_origin=window.c_high_origin,
            rmax_m=rmax_m,
            rmax_factor=multiplier,
            rmax_factor_origin=multiplier_origin,
            n_mesh=int(self.n_mesh),
            mesh_reference_frequency=freq0,
            deck_max_frequency=f_deck,
            n_wavenumbers=nk,
            taper=self.taper,
            peak_memory_bytes=peak,
            notices=tuple(n for n in notices if n is not None),
        )

    def _write_input(self, inputs) -> Path:
        """Stage 4: the Scooter deck, written by
        :func:`~uacpy.io.oalib_writer.write_scooter_env_file` from the
        resolved settings.

        Scooter uses the ReadEnvironmentMod format (same as Kraken) plus the
        phase-speed limits (``cLow``, ``cHigh``) and the spectral ``RMax``;
        shear parameters in the bottom half-space are carried natively.

        No receiver *ranges* are written: ``scooter.f90``'s ``GetPar``
        (``:154-178``) reads the environment, then ``ReadSzRz`` and
        ``ReadfreqVec``, and never calls ``ReadRcvrRanges`` — so the deck
        continues past ``RMax`` with the source depths, the receiver depths
        and (broadband only) the frequency vector, and simply stops there.
        The range axis is applied in-tree when the ``.grn`` is transformed.
        """
        deck = inputs.work_dir / f'{_BASE_NAME}.env'
        self._log(f"Writing environment file: {deck}")
        settings = inputs.settings
        engine = settings.engine
        write_scooter_env_file(
            deck, inputs.env, inputs.source, inputs.receiver,
            interp_ssp=self.interp_ssp,
            frequencies=(np.array(settings.frequencies)
                         if settings.mode in _BROADBAND_MODES else None),
            # TopOpt position 7: '0' zeroes out Scooter's stabilising
            # attenuation (scooter.f90:81,129-130). Left blank otherwise —
            # Scooter keeps Atten=Deltak, the default stabiliser.
            topopt_extra='0' if self.stabilizing_attenuation_off else '',
            n_mesh=engine.n_mesh,
            rmax_m=engine.rmax_m,
            c_low=engine.c_low, c_high=engine.c_high,
        )
        return deck

    def _launch(self, inputs, deck: Path) -> None:
        """Stage 4: run ``scooter.exe`` on the deck through the shared
        binary-launch helper, clearing a stale ``.grn`` first."""
        self._log("Running...")
        self._run_and_attach_prt([str(self._exe), deck.stem],
                                 inputs.work_dir, deck.stem,
                                 stale_outputs=('.grn',))

    def _read_output(self, inputs, deck: Path) -> GreensFunction:
        """Stage 4: the Green's function as
        :func:`~uacpy.io.grn_reader.read_grn_file` returns it.

        A missing ``.grn`` and one carrying fewer than two wavenumber samples
        are both run failures the binary does not report through its exit
        status, so each is turned into a typed error carrying the ``.prt``
        diagnostics. :meth:`_resolve_engine_settings` refuses ``Nk < 2``
        before the launch; this is the backstop for a count the deck-side
        arithmetic did not predict.
        """
        base_name = deck.stem
        work_dir = inputs.work_dir
        # A missing or empty .grn means the binary died silently; the raised
        # error carries the .prt tail with the actual cause.
        grn_file = self._require_output(
            [work_dir / f'{base_name}.grn'],
            what="a Green's function (.grn)",
            prt_base=base_name, work_dir=work_dir,
        )

        self._log("Reading Green's function...")
        greens_function = read_grn_file(grn_file)
        nk = len(greens_function.phase_speeds)
        if nk < 2:
            exc = ModelExecutionError(
                self.model_name, return_code=0, stdout=None,
                stderr=(
                    f"Scooter produced a Green's function with nk={nk} "
                    f"wavenumber sample(s); {_nk_consequence(nk)}"
                ),
            )
            self._attach_prt_tail(exc, work_dir, base_name)
            raise exc
        return greens_function

    def _to_result(self, inputs, deck: Path, raw: GreensFunction):
        """Stage 5: the Hankel transform of the Green's function onto the
        receiver ranges — complex pressure (``COHERENT_TL``, one slab per
        source depth of the deck), ``H(f)`` (``BROADBAND``) or its synthesis
        with the source pulse (``TIME_SERIES``) — in the package's
        line-source level and the phase convention of
        ``inputs.settings.output``, with depths below the modelled media
        masked."""
        settings = inputs.settings
        source = inputs.source
        env = inputs.env
        broadband = settings.mode in _BROADBAND_MODES
        result = assemble_field_from_grn(
            raw, source, inputs.receiver, broadband, taper=self.taper,
            spectrum=self.wavenumber_spectrum, model_name=self.model_name,
            log=self._log)
        slabs = _slabs_of(result)
        sources = ([source.at_depth(i) for i in range(len(slabs))]
                   if len(slabs) > 1 else [source])
        for slab, slab_source in zip(slabs, sources):
            if source.source_type == 'line':
                # The 'X' Hankel path returns 1/√(k0·R) in free space
                # (hankel_transform: no √k weighting, 1/√(2π)); ×√k0 is unit
                # amplitude at 1 m, the package's line-source level.
                freqs_out = (np.asarray(slab.coords['frequency'],
                                        dtype=float)
                             if 'frequency' in slab.coords
                             else np.atleast_1d(source.frequencies)[0])
                # ``slab_source``, not ``source``: the level is
                # sqrt(k0) at the SOURCE depth, and on a multi-depth
                # stack ``source`` carries every depth, whose first one
                # is not this slab's.
                level = _line_source_unit_at_1m(
                    _source_sound_speed(env, slab_source), freqs_out)
                slab.data = slab.data * (level if level.size == 1
                                         else level[None, None, :])

            freqs = (np.array(settings.frequencies, dtype=float) if broadband
                     else float(source.frequencies[0]))
            self._stamp_result(
                slab, slab_source, backend='scooter', frequencies=freqs,
                phase_reference=PhaseReference.TRAVELLING_WAVE.value)

            self._attach_output_paths(
                slab, inputs.work_dir, deck.stem,
                primary_files=(('grn_file', '.grn'),),
            )

        self._log("Simulation complete")
        media_depth = self._total_media_depth(env)
        if isinstance(result, ResultStack):
            # Narrowband only (the broadband modes loop per depth), so
            # there is no synthesis to finish.
            result.slabs = [
                self._mask_unresolvable_depths(slab, inputs.receiver,
                                               media_depth)
                for slab in result.slabs]
            return result
        if settings.time is not None:
            result = self._finish_broadband(result, settings)
        return self._mask_unresolvable_depths(
            result, inputs.receiver, media_depth)
