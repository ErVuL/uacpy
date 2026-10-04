"""The :class:`SPARC` engine class: its knobs, its declarations and the
protocol hooks that refuse, validate, resolve, write, launch, read and
assemble one SPARC run."""

from pathlib import Path
from types import MappingProxyType
from typing import Dict, Optional, Union

import numpy as np

from uacpy.core.exceptions import (
    ConfigurationError, UnsupportedFeatureError,
)
from uacpy.core.results import PhaseReference
from uacpy.core.run_settings import OutputSpec, RunMode
from uacpy.io.grn_reader import read_grn_file
from uacpy.io.oalib_reader import read_rts_file
from uacpy.io.oalib_writer import (
    write_sparc_env_file, write_sparc_source_time_series,
    reject_unsupported_ssp_interp,
    reject_biological_edges_under_neighbour_interp,
    SOURCE_TYPE_CODE as _SOURCE_TYPE_CODE,
)
from uacpy.models._band import BandResolution
from uacpy.models._spec import EngineTraits, ModelSpec
from uacpy.models._window import resolve_window
from uacpy.models.base import DEFAULT_RUN_TIMEOUT_S, PropagationModel
from uacpy.models.sparc._extract import (
    _output_times, launch_receiver, require_deck_time_grid,
    scale_to_unit_source_level, stack_traces, warn_on_truncated_window,
)
from uacpy.models.sparc._plan import (
    SPARC_WINDOW_TRAVEL_TIMES, _LOOPED_TIME_SERIES_MODES, check_knobs,
    checked_n_mesh,
    lossless_march_notice, profile_speed_bounds, range_alias_notice,
    refuse_too_few_wavenumbers, reject_halfspace_bottom,
    reject_oversized_snapshot, reject_reflection_table_bottom,
    resolve_n_time_samples, resolve_rmax_factor, resolve_run_bases,
)
from uacpy.models.sparc._pulse import (
    _BAND_FROM_OCTAVE, _validate_pulse_type, band_limit_notice,
    multi_frequency_notice, resolve_pulse_band, resolve_pulse_type,
    source_series_rows,
)
from uacpy.models.sparc._settings import SparcSettings
from uacpy.core.engine_defaults import SPARC_N_MESH, SPARC_OUTPUT_MODE


class SPARC(PropagationModel):
    """
    SPARC - Seismo-Acoustic Propagation in Realistic oCeans

    Time-domain FFP model: it marches a source pulse and its product is the
    transient pressure ``p(z, r, t)``. ``RunMode.TIME_SERIES`` is the only
    supported run mode; CW transmission loss is refused (see
    :meth:`run`) — use :class:`~uacpy.Scooter` or :class:`~uacpy.Kraken`
    for that.

    Limitations:
    - Only supports Vacuum or Rigid boundary conditions (no halfspace)
    - ``output_mode='R'`` runs one SPARC simulation per receiver depth and
      ``'D'`` one per receiver range (both looped in the wrapper) — see
      ``max_launches`` for the safety cap on the looped axis. ``'S'`` runs the
      binary once for the whole grid.
    - Longer computation time due to time-domain integration

    Notes
    -----
    Range-independent time-marched FFP. Only ``Vacuum`` / ``Rigid``
    bottom interfaces are supported; a half-space bottom raises
    :class:`ConfigurationError` (set ``acoustic_type='rigid'`` explicitly,
    or use Scooter for a half-space). ``RunMode.TIME_SERIES`` returns a
    :class:`Field` directly. SPARC drives its source pulse via the
    constructor ``pulse_type``: with a canned wavelet (``'P'``, ``'R'``,
    ...) passing ``source_waveform`` / ``sample_rate`` to ``run()`` emits a
    ``FallbackWarning`` (they have no effect on the simulation); with a
    ``pulse_type`` opening ``'F'`` (or ``'B'``, played backwards) the pair
    is required and is staged as the ``STSFIL`` series the binary marches
    (see :func:`~uacpy.io.oalib_writer.write_sparc_source_time_series`).

    The level is the package's: ``p(t)`` of a point source of unit
    amplitude at 1 m, the same level Scooter, Kraken, Bellhop and RAM
    return for the same waveform. The binary's own field is
    ``rho(z_s) s(t - R/c) / (2R)`` (see :data:`_UNIT_SOURCE_GAIN`), so the
    result is that times ``2 / rho(z_s)``.

    All three ``output_mode``s share one output grid: ``n_time_samples``
    samples over ``[0, time_max]`` (``march_start`` only sets where the time
    integration begins, not the output window), so the output rate is
    ``(n_time_samples − 1) / time_max`` whatever ``sample_rate`` the waveform was
    handed with. A grid whose Nyquist sits below the pulse band ``freq_max``
    aliases and emits a ``NumericsWarning`` naming the ``n_time_samples`` that would
    resolve it. ``result.frequencies`` is ``[deck_frequency]`` (SPARC
    marches in time and has no frequency bins); the pulse band
    ``[freq_min, freq_max]`` the run marched is ``result.band_hz``.

    **Collapse defaults (overrides of :data:`DEFAULT_COLLAPSE`).**
    Per-model: ``'ssp': 'mean'``, ``'bottom_range': 'median'`` (the layer
    stack is kept since SPARC consumes layered seabed columns natively).

    Defaults auto-derived at ``run()`` time (``run_settings`` shows them):

    - ``n_mesh=0`` → 20 points per wavelength at ``freq_max``, per medium.
    - ``c_low`` / ``c_high`` → from env SSP and bottom speed.
    - ``rmax`` written as ``receiver.range_max × rmax_factor``.
    - ``dt`` / ``dr`` derived from CFL stability and the source pulse.

    **The march is lossless.** ``sparc.f90:221`` converts each node's
    complex sound speed to single precision with ``REAL(cp(ii), 4)``, which
    keeps its real part only, so ``c2I`` (``:223``), the one path by which
    attenuation enters the march (``CONTRIB``, ``:437-451``), is zero: water
    absorption and every sediment attenuation are ignored. Measured: 0.000 dB
    of water absorption where Scooter applies 0.9-3.4 dB, and a sediment at
    0 or 5 dB/wavelength giving the same trace energy to 1e-5. The
    truncation keeps the explicit scheme uacpy writes (ALPHA = BETA = 0)
    consistent: that step divides by the diagonal of ``A2`` alone, and the
    damping matrix ``c2I`` would build is not lumped — restoring ``Im c``
    turned a 1-3 dB loss into a 4-13 dB gain and made a lossy sediment
    diverge. A run with any attenuation warns; Scooter is the attenuating
    spectral alternative for the same waveguide.

    Examples
    --------
    >>> sparc = SPARC(verbose=False)
    >>> result = sparc.run(env, source, receiver)
    """

    # Declarative metadata (see PropagationModel / ModelSpec). SPARC:
    # range-independent time-marched FFP. Honours a multi-layer fluid
    # bottom; elastic_media is False because the march reads cp alone, so
    # shear is collapsed to fluid up front with the uniform warning. Single
    # solve over the spectrum →
    # mean SSP / median bottom column represent the path.
    spec = ModelSpec(
        modes=(RunMode.TIME_SERIES,),
        supports={'layered_bottom'},
        collapse={'ssp': 'mean', 'bottom_range': 'median'},
        traits=EngineTraits(
            # TopOpt position 4 writes env.absorption into the deck, but the
            # march drops every imaginary sound speed (sparc.f90:221, see the
            # class docstring), so the engine does not honour it.
            consumes_volume_absorption=False,
            # Meshes through the sediment stack.
            receivers_reach_sediment=True,
            # None: the base notices state how a grid derived from the pulse
            # folds a late arrival onto a record ``1/Δf`` long, and SPARC
            # marches in time over its own ``[0, time_max]`` record, folding
            # nothing. SPARC states its own band
            # (:meth:`_resolve_engine_settings`).
            announced_band_modes=frozenset(),
            # ``output_duration`` is the end of SPARC's own record
            # (``time_max``), not zero padding of the pulse, so the time
            # settings keep the pulse unpadded — the series STSFIL carries.
            pads_pulse_to_output_duration=False,
        ),
    )
    provenance_id = 'acoustics_toolbox'

    outputs = MappingProxyType({
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
        n_mesh: int = SPARC_N_MESH,
        interp_ssp: Optional[str] = None,
        output_mode: str = SPARC_OUTPUT_MODE,
        pulse_type: Optional[str] = None,
        # Power of two: p(t) comes back on this many samples and any
        # downstream FFT over the time axis then stays on the radix-2 path.
        n_time_samples: int = 512,
        time_max: Optional[float] = None,
        march_start: float = -0.1,
        courant_factor: float = 0.999,
        max_launches: int = 20,
        rmax_factor: Optional[float] = None,
        freq_min: Optional[float] = None,
        freq_max: Optional[float] = None,
        window_sound_speed: Optional[float] = None,
        timeout: float = DEFAULT_RUN_TIMEOUT_S,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        cleanup: Optional[bool] = None,
        collapse: Optional[Dict[str, str]] = None,
    ):
        """
        Parameters
        ----------
        executable : Path, optional
            Path to ``sparc.exe``. Auto-detected if ``None``.
        c_low : float, optional
            Lower phase speed limit (m/s). None = auto. Default: None.
        c_low, c_high : float, optional
            Phase-speed bounds (m/s). ``None`` ⇒ auto. Default ``None``.
        c_high : float, optional
            Upper phase speed limit (m/s). None = auto. Default: None.
        n_mesh : int, optional
            Total mesh points per medium (not per wavelength). 0 = auto:
            uacpy sizes each medium at 20 points per wavelength at the band
            top ``freq_max`` (``resolve_n_mesh``). Default: 0.
            Default ``0``.
        interp_ssp : str, optional
            SSP connection scheme written into ``TopOpt(1)``. ``None``
            (default) resolves to ``'linear'`` (C-linear): SPARC declares
            no range-dependent-SSP capability, so ``run()`` collapses
            ``env.ssp`` to 1-D before the deck is written and the
            range-dependent auto-pick never applies. Explicit values:
            ``'linear'``, ``'n2linear'``, ``'pchip'``,
            ``'spline'``. ``env.ssp.kind='isovelocity'`` always forces
            ``'C'`` regardless.
        output_mode : str, optional
            'R' (horizontal array), 'D' (vertical array), 'S' (snapshot). Default: 'R'.

            All three return the received time series ``p(z, r, t)`` on the
            same convention. ``'R'`` loops the binary per receiver depth,
            ``'D'`` per receiver range; ``'S'`` runs once and is converted
            in-tree by one inverse Hankel transform per output time
            (:meth:`~uacpy.core.results.GreensFunction.snapshot_to_time_field`).
            CW transmission loss is not offered — see the
            ``RunMode.COHERENT_TL`` refusal in :meth:`run` for why.
        pulse_type : str, optional
            4-character source-pulse code per ``sourceMod.f90``. ``None``
            (default) marches the run's ``source_waveform`` (``'FN+B'``), or the
            canned ``'PN+B'`` without one.
            Position 1 = pulse shape (``PRASHNGFBM`` — ``T``/``C`` are listed
            in ``cans.f90`` but rejected by ``sparc.f90``'s ``GetPar``),
            2 = post-process, 3 = sign, 4 = filter.

            ``cans.f90:31-35``'s ``'R'`` (Ricker) is defined on ``U = ω·T - 5``
            and therefore peaks at ``T = 5/(2π·F) ≈ 0.796/F``, not at ``T = 0``.
            (``cans.f90``'s own per-case "peak at …, support […]" notes describe
            the *spectrum*, not the waveform — its ``'C'`` sinc entry reads
            "uniform spectrum from [0, F]" and its Gaussian says "peak at 0"
            while peaking in time at ``T0 = 0.5/F``.) When aligning expected vs
            measured arrivals downstream, treat the Ricker peak as offset by
            ``+5/(2π·F)`` from the source pulse origin.
        n_time_samples : int, optional
            Number of output time samples. Default ``512``.

            The deck writes the output times as the pair ``0.0 time_max /``, which
            ``ReadVector`` leaves for ``SubTab`` to expand — and ``SubTab`` acts
            only for ``Nx >= 3`` (``misc/subtabulate.f90:24,40``). ``n_time_samples >= 3``
            therefore gets ``n_time_samples`` uniform samples on ``[0, time_max]``;
            ``n_time_samples == 2`` happens to land on the two endpoints unexpanded; and
            ``n_time_samples == 1`` would read only the ``0.0`` — the trailing ``time_max``
            never consumed, a single ``t = 0`` sample returned — so
            ``n_time_samples < 2`` raises :class:`ConfigurationError`.
        time_max : float, optional
            Maximum simulated time (s). ``None`` ⇒ ``run(output_duration=)``
            when given, else ``2.5 ×`` the direct travel time at the slowest
            speed in the profile. That is a heuristic, not a bound on the last
            arrival — a waveguide's tail is set by the slowest modal *group*
            velocity, which falls to zero at cutoff — so a run whose trace is
            still ringing at the end of the window emits a ``NumericsWarning``
            saying the trace is truncated.
        march_start : float, optional
            Time the march begins (s), ``time = march_start + (i-1)·Δt``
            (``sparc.f90:409``). ``cans.f90`` returns zero pulse amplitude for
            ``T <= 0``, so a negative value starts the field from rest before the
            source turns on. Default ``-0.1``.
        courant_factor : float, optional
            Courant safety factor on the marching time step: ``sparc.f90:265``
            sets ``Δt = courant_factor / sqrt(1/crossT² + (0.5·cMax·k)²)``, so ``1.0`` is
            exactly the stability limit and the default sits just inside it.
            Default ``0.999``.
        max_launches : int, optional
            Cap on the axis the wrapper loops the binary over — receiver
            depths for ``output_mode='R'``, receiver ranges for ``'D'``.
            ``'S'`` runs once and is uncapped. Default ``20``.
        rmax_factor : float, optional
            Multiplier applied to ``receiver.ranges.max()`` to set SPARC's
            ``RMax``. ``None`` (default) ⇒ ``4.0``. ``RMax`` fixes the
            wavenumber sampling — ``sparc.f90:116`` takes
            ``Nk = INT(1000·RMax_km·(kMax−kMin)/2π)``, i.e.
            ``Δk ≈ 2π/RMax_m`` — and the range synthesis is a direct ``Δk``
            sum over that grid, so the output is periodic in range with
            period ``2π/Δk = RMax``. The replica that period puts at
            ``r = RMax`` is not the whole story: a receiver at range ``r``
            sees the fold from ``r − RMax``, i.e. the arrival that belongs at
            ``|RMax − r|``, so the alias reaches it at ``(RMax − r)/c`` and
            pushing ``RMax`` just past the receivers does *not* clear it. At
            margin ``m`` the alias lands at ``(m−1)·r/c`` while the auto
            window runs to ``2.5·r/c``, so the margin has to exceed ``3.5``;
            ``4.0`` is the first round value that does. Increasing this knob
            raises ``Nk`` (and SPARC runtime) in proportion. A pinned margin
            (or a pinned ``time_max``) that leaves the alias inside the output
            window emits a ``NumericsWarning`` naming the margin that clears it.
            ``sparc.f90:153-155`` rejects the run outright when the largest
            receiver range exceeds ``RMax``.
        freq_min, freq_max : float, optional
            Pulse frequency band (Hz). ``None`` (default) resolves at
            ``run()`` time, edge by edge: to the span of
            ``run(frequencies=)`` when given; else, when SPARC marches the
            run's ``source_waveform``, to the band every synthesising
            engine resolves from that waveform padded to SPARC's own
            ``time_max`` record (its spectral support above -40 dB, limited
            for a hard-edged pulse to one -20 dB band-width beyond its
            -20 dB band, with a notice; the ``run_settings().frequencies``
            span of Scooter or Kraken for the same pulse and record);
            else, for a canned pulse, to one
            octave around the source frequency (``max(f/2, 0.1)`` ..
            ``2f``). SPARC's work scales with the band's wavenumber span,
            so a much wider band slows it sharply. ``freq_min=0`` is accepted
            (``sparc.f90:114`` clamps the resulting ``kMin`` to 1e-20).
        window_sound_speed : float, optional
            Water sound speed (m/s) used for the travel-time window
            when ``time_max`` is auto. ``None`` (default) → the slowest
            speed in ``env.ssp``, which puts the *direct* arrival well
            inside the window; :data:`DEFAULT_SOUND_SPEED` only when the
            profile carries none.
        timeout : float, optional
            Subprocess timeout per run (s). Default ``600.0`` — sized so the
            suite's longest SPARC runs survive a fully loaded CPU (an
            oversubscribed 8-worker pytest session slows an ~85 s march past
            3x), while a hung binary still dies.
        use_tmpfs, verbose, work_dir, cleanup, collapse : optional
            Standard plumbing (see :class:`PropagationModel`).
        """
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
            timeout=timeout, cleanup=cleanup, collapse=collapse,
        )

        self.c_low = c_low
        self.c_high = c_high
        self.n_mesh = n_mesh
        self.interp_ssp = interp_ssp
        self.output_mode = output_mode
        # Only the snapshot runs a Hankel transform, so it is the only mode
        # that can honour a source geometry; 'R'/'D' are range-/depth-native.
        if output_mode == 'S':
            self._supported_source_types = frozenset(_SOURCE_TYPE_CODE)
        self.pulse_type = (None if pulse_type is None
                           else _validate_pulse_type(pulse_type))
        self.n_time_samples = n_time_samples
        self.time_max = time_max
        self.march_start = march_start
        self.courant_factor = courant_factor
        self.max_launches = max_launches
        self.rmax_factor = rmax_factor
        # Pulse band (``freq_min``/``freq_max``) and ``window_sound_speed`` (used for
        # the travel-time window when ``time_max`` is auto) default to
        # ``None`` and are resolved at ``run()`` time (see
        # :func:`~uacpy.models.sparc._pulse.resolve_pulse_band` and
        # :func:`~uacpy.models.sparc._plan.profile_speed_bounds`).
        self.freq_min = float(freq_min) if freq_min is not None else None
        self.freq_max = float(freq_max) if freq_max is not None else None
        self.window_sound_speed = (
            float(window_sound_speed) if window_sound_speed is not None else None
        )
        self._check_knobs()

        # Run modes, capability flags and collapse defaults come from the
        # class-level ``spec`` (applied by PropagationModel.__init__).
        self._exe = self._resolve_executable(
            executable,
            lambda: self._find_executable_in_paths(
                'sparc.exe', bin_subdirs='oalib',
                dev_subdir='Acoustics-Toolbox/Scooter',
            ),
        )

    def _refuse_run_mode(self, run_mode) -> None:
        """Refuse ``COHERENT_TL`` by name, so the refusal points at a model
        that does compute CW TL; the generic run-mode error can only list
        SPARC's own modes."""
        if run_mode == RunMode.COHERENT_TL:
            raise UnsupportedFeatureError(
                model_name='SPARC',
                feature=(
                    'RunMode.COHERENT_TL — SPARC marches a pulse, and the '
                    'CW field extracted from it is not quantitative '
                    '(no contour offset on the wavenumber sum, a grid sized '
                    'for the whole pulse band, and a per-wavenumber band-pass '
                    'that a scalar source-spectrum deconvolution cannot undo)'
                ),
                alternatives=[
                    'Scooter for wavenumber-integration CW transmission loss',
                    'Kraken for normal-mode CW transmission loss',
                    'SPARC with run_mode=TIME_SERIES for its native p(t)',
                ],
            )

    # ── stage 2: the carriers SPARC cannot run ─────────────────────────

    def _check_time_series_request(self, call) -> None:
        """Stage 2: the waveform pair of the pulse SPARC marches, and the
        run keywords a pinned knob overrides.

        A pulse read from STSFIL (``pulse_type`` opening with ``'F'`` /
        ``'B'``, or unpinned with a ``source_waveform``) needs a usable
        waveform pair (:meth:`_require_source_time_series`); a canned pulse
        ignores the pair, with a warning. ``frequencies=`` sets the pulse
        band and ``output_duration=`` the record unless
        ``SPARC(freq_min=, freq_max=)`` / ``SPARC(time_max=)`` pin them, in which
        case each is ignored with a warning. A missing pair is legal for a
        canned pulse, so the synthesising engines' rule (a TIME_SERIES run
        needs a pulse) does not apply.
        """
        kw = call.kwargs
        pulse, _origin = resolve_pulse_type(kw['source_waveform'],
                                            pulse_type=self.pulse_type)
        if pulse[0] in 'FB':
            self._require_source_time_series(kw['source_waveform'],
                                             kw['sample_rate'])
        else:
            self._warn_ignored_run_kwargs(
                call.mode,
                reason=(
                    "SPARC builds p(t) from its native pulse_type over its "
                    "own time grid at source.frequencies; pass "
                    "SPARC(pulse_type=...) to shape the pulse, or a "
                    "pulse_type opening with 'F' to march the given waveform"
                ),
                source_waveform=kw['source_waveform'],
                sample_rate=kw['sample_rate'],
            )
        band_pinned = self.freq_min is not None and self.freq_max is not None
        self._warn_ignored_run_kwargs(
            call.mode,
            reason=(
                "SPARC(freq_min=, freq_max=) pin the pulse band SPARC marches and "
                "SPARC(time_max=) the end of its record"
            ),
            frequencies=kw['frequencies'] if band_pinned else None,
            output_duration=(kw['output_duration'] if self.time_max is not None
                             else None),
        )

    def _check_knobs(self) -> None:
        """Refuse a knob no run could use
        (:func:`~uacpy.models.sparc._plan.check_knobs`)."""
        check_knobs(c_low=self.c_low, c_high=self.c_high,
                    output_mode=self.output_mode,
                    pulse_type=self.pulse_type,
                    march_start=self.march_start, freq_min=self.freq_min,
                    freq_max=self.freq_max)

    def _validate_engine(self, env, source, receiver, run_mode, *,
                         request=None) -> None:
        """Stage 2: the refusals of carriers SPARC cannot run — on the
        projected environment, so ``validate_inputs`` refuses what ``run``
        refuses.

        - a half-space seabed
          (:func:`~uacpy.models.sparc._plan.reject_halfspace_bottom`);
        - a reflection-table seabed (``'file'`` / ``'precalc'``);
        - an ``interp_ssp`` the SPARC deck cannot carry.
        """
        reject_halfspace_bottom(env)
        reject_reflection_table_bottom(env)
        reject_unsupported_ssp_interp('SPARC', self.interp_ssp)
        reject_biological_edges_under_neighbour_interp(
            'SPARC', env, self.interp_ssp)

    def _require_source_time_series(self, source_waveform,
                                    sample_rate) -> None:
        """Refuse the waveform pair of a pulse read from STSFIL; returns
        nothing. The series STSFIL carries is the run's
        ``settings.time.source_waveform`` (:meth:`_time_settings`).

        Raises :class:`ConfigurationError` when a pulse read from STSFIL
        (a ``pulse_type`` opening with ``'F'`` / ``'B'``) is run without
        both ``source_waveform`` and ``sample_rate``, when the waveform is
        not one real finite series, or when it has fewer than two samples
        (``tslib/sourceMod.f90:47`` takes the step as ``TF(2) - TF(1)``).
        The value checks are the shared :meth:`_require_timeseries_signal`
        ones.
        """
        pulse, _origin = resolve_pulse_type(source_waveform,
                                            pulse_type=self.pulse_type)
        letter = pulse[0]
        if source_waveform is None or sample_rate is None:
            raise ConfigurationError(
                f"SPARC(pulse_type={pulse!r}): the leading "
                f"{letter!r} makes the binary read its source time series "
                f"from a file (tslib/sourceMod.f90:44-46), so run() needs "
                f"both source_waveform (a 1-D pressure series) and "
                f"sample_rate (Hz); got source_waveform="
                f"{'None' if source_waveform is None else 'given'}, "
                f"sample_rate={sample_rate!r}.",
                remediation=(
                    f"SPARC(pulse_type={pulse!r}).run(env, source, "
                    f"receiver, source_waveform=pulse, sample_rate=fs), or a "
                    f"canned pulse_type such as 'PN+B'."
                ),
            )
        self._require_timeseries_signal(
            RunMode.TIME_SERIES, source_waveform, sample_rate)
        waveform = np.atleast_1d(np.asarray(source_waveform))
        if waveform.ndim != 1:
            raise ConfigurationError(
                f"SPARC: source_waveform must be a 1-D series (it is written "
                f"once per source depth into STSFIL); got shape "
                f"{waveform.shape}.",
                remediation="source_waveform=np.ravel(pulse)",
            )
        if waveform.size < 2:
            raise ConfigurationError(
                f"SPARC: source_waveform has {waveform.size} sample(s); the "
                f"binary takes its time step from the first two "
                f"(tslib/sourceMod.f90:47, TF(2) - TF(1)).",
                remediation="Pass a series of at least 2 samples.",
            )

    # ── stage 3: the settings of the deck ──────────────────────────────

    def _requested_frequencies(self, mode, source, frequencies, time):
        """The call's ``frequencies=`` (their span sets the pulse band), or
        ``None``: the band is resolved with the record it pads the pulse to,
        in :meth:`_resolve_engine_settings`, and :meth:`_marched_frequencies`
        then records the result's ``frequencies``. SPARC states its band
        itself, so no notice."""
        if frequencies is None:
            return BandResolution(None)
        return BandResolution(
            np.atleast_1d(np.asarray(frequencies, dtype=float)))

    def _marched_frequencies(self, settings):
        """What the result carries: ``[deck_frequency]``, the centre of the
        band — SPARC marches in time, not over frequency bins; the band
        edges are ``settings.engine.freq_min`` / ``freq_max`` and the result's
        :attr:`~uacpy.core.results.Field.band_hz`."""
        return np.array([settings.engine.deck_frequency])

    def _resolve_engine_settings(self, env, source, receiver, settings, *,
                                 given_env, request=None) -> SparcSettings:
        """Stage 3: every setting of the deck(s), resolved once from the
        projected ``env``, and the refusals of a deck SPARC cannot run.

        Refused: an STSFIL series over the binary's point cap or too short
        for the band-pass, a looped axis over ``max_launches``,
        ``n_time_samples < 2``, fewer than two wavenumbers, a snapshot table over
        the memory this host has free, a mesh over ``sparc.f90``'s static
        storage.
        """
        time = settings.time
        pulse, pulse_type_origin = resolve_pulse_type(time.source_waveform,
                                                 pulse_type=self.pulse_type)
        run_bases = resolve_run_bases(receiver, output_mode=self.output_mode,
                                      max_launches=self.max_launches)

        # RMax is the range period of SPARC's direct Δk sum (Δk ≈ 2π/RMax) —
        # too tight leaks the source's r=RMax image into the receiver area.
        # Margin policy in ``resolve_rmax_factor``. Every launch of
        # the 'R' / 'D' loops shares it, so all of them share one output time
        # grid.
        r_ref = float(receiver.ranges.max())
        margin = resolve_rmax_factor(
            rmax_factor=self.rmax_factor)
        rmax_m = r_ref * margin

        # Time output window (s), anchored on the travel time to the farthest
        # receiver (r_ref) — not on rmax_m, whose rmax_factor factor is
        # a wavenumber-sampling knob (Δk ≈ 2π/RMax): folding the margin into
        # the window stretches [0, time_max] by that factor while n_time_samples stays
        # fixed, aliasing the default output grid.
        #
        # The auto window is 2.5 direct travel times at the slowest speed the
        # profile carries. That is a heuristic, not a bound on the last
        # arrival: in a guide the
        # tail is set by the slowest modal GROUP velocity, which goes to zero
        # at cutoff, and SPARC's vacuum/rigid boundaries (sparc.f90:101-104)
        # leave that tail undamped. An arrival past time_max is simply absent
        # from p(t) with nothing in the output to say so, hence the post-run
        # ``warn_on_truncated_window`` check on the trace that comes back.
        # ``sound_speed`` pins the speed; DEFAULT_SOUND_SPEED is only the
        # fallback for an environment that declares no usable speed at all.
        c_slow, c_fast = profile_speed_bounds(env,
                                              window_sound_speed=self.window_sound_speed)
        if self.time_max is not None:
            time_max, time_max_origin = float(self.time_max), 'SPARC(time_max=…)'
        elif time.output_duration is not None:
            time_max, time_max_origin = (time.output_duration,
                                   'run(output_duration=…)')
        else:
            travel_time = r_ref / c_slow
            time_max = travel_time * SPARC_WINDOW_TRAVEL_TIMES
            time_max_origin = (f"{SPARC_WINDOW_TRAVEL_TIMES:g} × the direct "
                               f"travel time to the farthest receiver at "
                               f"the slowest SSP speed")

        # The band the deck marches, measured on the pulse padded to that
        # record (one decider, which also says where each edge came from).
        band = resolve_pulse_band(
            source, settings.frequencies, time, time_max,
            pinned_f_min=self.freq_min, pinned_f_max=self.freq_max,
            pulse_type=self.pulse_type, model_name=self.model_name)
        freq_min, freq_max = band.freq_min, band.freq_max
        notices = [multi_frequency_notice(
            source, band.freq_max_origin == _BAND_FROM_OCTAVE)]
        n_source_depths = (1 if settings.depth_loop == 'per_depth' else
                           int(np.atleast_1d(np.asarray(source.depths)).size))
        sts_samples, sts_rows = source_series_rows(
            pulse, time.source_waveform, time.sample_rate, n_source_depths,
            freq_min, freq_max)
        notices.append(range_alias_notice(rmax_m, r_ref, c_fast, time_max))

        n_time_samples, grid_notice = resolve_n_time_samples(freq_max, time_max,
                                               n_time_samples=self.n_time_samples)
        notices.append(grid_notice)

        # Refuse a wavenumber count the binary cannot march. sparc.f90:116
        # computes Nk = INT( 1000.0*RMax_km*(kMax-kMin)/(2π) ) with
        # kMin = 2π·fMin/cHigh and kMax = 2π·fMax/cLow, which reduces to
        # rmax_m·(fMax/cLow − fMin/cHigh). Nk <= 0 runs the march loop
        # (sparc.f90:269) zero times, so every output comes back all-zero at
        # exit 0, and :265 reads k(Nk) out of bounds first. Nk = 1 is a
        # single-sample spectrum; require >= 2, the floor bounce.f90's own
        # tabulation needs (Deltak divides by NkTab - 1 at :172).
        # An inverted window (one pinned bound past the derived other) is
        # refused here, before ReadEnvironmentMod.f90:135 stops the binary.
        window = resolve_window(env, c_low=self.c_low, c_high=self.c_high,
                                model_name='SPARC')
        c_low_res, c_high_res = window.c_low, window.c_high
        nk = int(rmax_m * (freq_max / c_low_res - freq_min / c_high_res))
        refuse_too_few_wavenumbers(nk, rmax_m, freq_min, freq_max, c_low_res,
                                   c_high_res)
        notices.append(reject_oversized_snapshot(
            receiver, nk, n_time_samples, output_mode=self.output_mode))
        notices.append(band_limit_notice(band))
        notices.append(lossless_march_notice(env))
        n_mesh = checked_n_mesh(env, freq_max, pinned_n_mesh=self.n_mesh)

        notices = [n for n in notices if n is not None]
        return SparcSettings(
            pulse_type=pulse,
            pulse_type_origin=pulse_type_origin,
            deck_frequency=float(np.atleast_1d(
                np.asarray(source.frequencies, dtype=float))[0]),
            freq_min=freq_min, freq_max=freq_max,
            freq_min_origin=band.freq_min_origin, freq_max_origin=band.freq_max_origin,
            time_max=time_max, time_max_origin=time_max_origin,
            n_time_samples=int(n_time_samples),
            rmax_m=rmax_m,
            rmax_factor=margin,
            rmax_factor_origin=(
                'SPARC(rmax_factor=…)'
                if self.rmax_factor is not None else
                'the default, which clears the range alias from the auto '
                'window'),
            c_low=c_low_res, c_high=c_high_res,
            n_wavenumbers=nk,
            n_mesh=n_mesh,
            output_mode=self.output_mode,
            run_bases=run_bases,
            march_start=float(self.march_start),
            courant_factor=float(self.courant_factor),
            sts_samples=sts_samples,
            sts_rows=sts_rows,
            notices=tuple(n for n in notices if n is not None),
        )

    # ── stage 4: decks, launches, outputs ──────────────────────────────

    def _n_launches(self, settings) -> int:
        """One launch per receiver depth (``'R'``), per receiver range
        (``'D'``), or one (``'S'``): one per ``settings.engine.run_bases``
        entry."""
        return len(settings.engine.run_bases)

    def _write_input(self, inputs) -> Path:
        """Stage 4, per launch: the SPARC deck of launch ``inputs.launch``,
        written by :func:`~uacpy.io.oalib_writer.write_sparc_env_file` from
        the resolved settings, preceded on the first launch by ``STSFIL``
        when the pulse is read from file.

        One STSFIL per work directory: ``sparc.f90:525`` calls SOURCE in
        every output mode's march, and ``sourceMod.f90:97`` opens the fixed
        name in the cwd — the work directory every launch of the R/D loops
        shares.

        SPARC extends ReadEnvironmentMod format with:
        - Output mode in TopOpt (5th character: R=horizontal, D=vertical, S=snapshot)
        - Limited bottom types (only vacuum and rigid, no halfspace)
        - Time-domain pulse parameters
        - Time output parameters
        - Integration parameters
        """
        settings = inputs.settings
        engine = settings.engine
        work_dir = inputs.work_dir
        if engine.sts_rows and inputs.launch == 0:
            write_sparc_source_time_series(
                work_dir / 'STSFIL', inputs.source,
                settings.time.source_waveform, settings.time.sample_rate,
                engine.sts_rows)
        deck = work_dir / f'{engine.run_bases[inputs.launch]}.env'
        receiver, _value = launch_receiver(inputs)
        write_sparc_env_file(
            deck, inputs.env, inputs.source, receiver,
            interp_ssp=self.interp_ssp,
            output_mode=engine.output_mode,
            n_mesh=engine.n_mesh,
            rmax_m=engine.rmax_m,
            c_low=engine.c_low, c_high=engine.c_high,
            pulse_type=engine.pulse_type,
            freq_min=engine.freq_min, freq_max=engine.freq_max,
            n_time_samples=engine.n_time_samples,
            time_max=engine.time_max,
            march_start=engine.march_start, courant_factor=engine.courant_factor,
        )
        return deck

    def _launch(self, inputs, deck: Path) -> None:
        """Stage 4, per launch: run ``sparc.exe`` on ``deck`` — one receiver
        depth (``'R'``), one receiver range (``'D'``), or the whole grid
        (``'S'``)."""
        engine = inputs.settings.engine
        n = len(engine.run_bases)
        if engine.output_mode == 'S':
            self._log("Computing snapshot (whole field per output time)...")
        else:
            spec = _LOOPED_TIME_SERIES_MODES[engine.output_mode]
            if inputs.launch == 0:
                self._log(spec['opening'].format(n=n))
            _receiver, value = launch_receiver(inputs)
            self._log(spec['step'].format(i=inputs.launch + 1, n=n,
                                          value=value))
        self._run_sparc(deck.stem, inputs.work_dir)

    def _run_sparc(self, base_name: str, work_dir: Path):
        """
        Run SPARC as a subprocess (600 s timeout by default).

        Delegates to ``PropagationModel._run_subprocess`` (which raises the
        child stack limit — required because ``MARCH`` declares twelve
        automatic ``COMPLEX(NTot1)`` arrays, ``sparc.f90:354-355``, which
        gfortran puts on the stack). On failure, appends the ``.prt`` tail to
        the raised ``ModelExecutionError`` for easier diagnosis. Override via
        the ``timeout`` constructor kwarg.
        """
        self._run_and_attach_prt(
            [str(self._exe), base_name], work_dir, base_name,
            timeout=self.timeout, stale_outputs=('.grn', '.rts'))

    def _read_output(self, inputs, deck: Path):
        """Stage 4, per launch: the snapshot ``.grn`` as
        :func:`~uacpy.io.grn_reader.read_grn_file` returns it
        (``output_mode='S'``), or the launch's :func:`read_rts_file` record,
        checked to carry the deck's output time grid.

        A missing or empty file means the binary died silently; the raised
        error carries the ``.prt`` tail with the actual cause.
        """
        engine = inputs.settings.engine
        work_dir = inputs.work_dir
        base_name = deck.stem
        if engine.output_mode == 'S':
            grn_file = self._require_output(
                [work_dir / f'{base_name}.grn'],
                what='a snapshot file (.grn)',
                prt_base=base_name, work_dir=work_dir,
            )
            return read_grn_file(grn_file)
        rts_file = self._require_output(
            [work_dir / f'{base_name}.rts'],
            what='a time-series file (.rts)',
            prt_base=base_name, work_dir=work_dir,
        )
        rts_data = read_rts_file(rts_file)
        require_deck_time_grid(
            rts_data, _output_times(engine.time_max, engine.n_time_samples), work_dir,
            base_name, model_name=self.model_name)
        return rts_data

    # ── stage 5: the result ────────────────────────────────────────────

    def _to_result(self, inputs, deck: Path, raw):
        """Stage 5: ``p(z, r, t)`` on the deck's output time grid — the
        per-launch traces stacked (``'R'`` / ``'D'``) or the snapshot
        transformed onto the receiver ranges (``'S'``) — at the package's
        unit-source level
        (:func:`~uacpy.models.sparc._extract.scale_to_unit_source_level`),
        stamped
        with ``frequencies = [deck_frequency]`` (the band's centre; SPARC
        has no frequency bins), the marched band as
        ``band_hz = (freq_min, freq_max)`` and the metadata every
        TIME_SERIES engine carries, with depths below the modelled media
        masked."""
        settings = inputs.settings
        engine = settings.engine
        source = inputs.source
        env = inputs.env
        band_hz = (engine.freq_min, engine.freq_max)
        metadata = {}
        if engine.sts_rows:
            metadata['source_waveform_sample_rate'] = settings.time.sample_rate
        if engine.output_mode == 'S':
            result = raw.snapshot_to_time_field(
                np.atleast_1d(inputs.receiver.ranges),
                source_type=source.source_type,
            )
            self._stamp_result(
                result, source, backend='sparc',
                frequencies=settings.frequencies,
                phase_reference=settings.output.phase_reference)
            result.metadata.update(metadata)
            result = result.replace(band_hz=band_hz)
        else:
            # One launch hands over its record, several their list.
            runs = raw if isinstance(raw, list) else [raw]
            result = stack_traces(inputs, runs, metadata, band_hz=band_hz,
                                  model_name=self.model_name,
                                  provenance=self.provenance)
        self._attach_output_paths(
            result, inputs.work_dir, engine.run_bases[0],
            primary_files=(
                ('grn_file', '.grn'),
                ('rts_file', '.rts'),
            ),
        )
        pinned = (self.time_max is not None
                  or settings.time.output_duration is not None)
        warn_on_truncated_window(
            result, pinned_by=engine.time_max_origin if pinned else None)
        self._log("Simulation complete")
        scale_to_unit_source_level(result, env, source)
        return self._mask_unresolvable_depths(
            result, inputs.receiver, self._total_media_depth(env))
