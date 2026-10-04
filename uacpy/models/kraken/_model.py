"""The :class:`Kraken` model: its knobs, traits and run hooks."""

import warnings
import dataclasses
import time
from types import MappingProxyType
import numpy as np
from pathlib import Path
from typing import Callable, Dict, NamedTuple, Optional, Union
from uacpy.models.base import DEFAULT_RUN_TIMEOUT_S, PropagationModel
from uacpy.models._launch import Launch
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.models._conventions import _source_density, _source_sound_speed
from uacpy.models._spec import EngineTraits, ModelSpec
from uacpy.models._stacking import _slabs_of
from uacpy.core.run_settings import OutputSpec, RunMode
from uacpy._log import log_message
from uacpy.core.boundary import BoundaryProperties
from uacpy.core.source import Source
from uacpy.core.surface import Surface
from uacpy.core.results import Field, PhaseReference, ResultStack
from uacpy.core.constants import NO_ENERGY_DB
from uacpy.core.exceptions import (
    ConfigurationError, ExecutableNotFoundError, FileFormatError,
    ModelExecutionError, NumericsWarning, ValidityWarning,
)
from uacpy.io.oalib_writer import (
    at_env_media, reject_coarse_at_mesh, reject_unsupported_ssp_interp,
    reject_biological_edges_under_neighbour_interp,
)
from uacpy.io.oalib_reader import read_shd_bin
from uacpy.io.modes_reader import read_modes
from uacpy.models.kraken._settings import KrakenLaunch, KrakenSettings
from uacpy.models.kraken._checks import (
    backend_origin,
    deck_bottom_columns,
    drop_roughness_on_tabulated_top,
    elastic_depth_intervals,
    krakenc_incoherent_sum_notice,
    mask_elastic_mode_depths,
    modes_single_profile,
    partition_elastic_subbottom,
    reflection_table_boundaries,
    reinsert_nan_depths,
    reject_acoustic_below_elastic,
    reject_krakenc_over_a_hard_or_free_floor,
    reject_rough_elastic_layer,
)
from uacpy.models._checks import reject_precalc_surface
from uacpy.models.kraken._window import (
    _LEAKY_C_HIGH_FACTOR,
    phase_speed_window,
    steep_path_notice,
    validate_phase_speed_limits,
)
from uacpy.core._validate import (
    require_non_negative, require_strictly_increasing,
)
from uacpy.core.deck_limits import DECK_DEPTH_RESOLUTION_M
from uacpy.models._budget import work_dir_is_memory_backed
from uacpy.models.kraken._grid import (
    build_field_option,
    check_field_tabulation_size,
    merge_depths,
    mode_count_check_mesh,
    mode_file_size_notice,
    dense_mode_depths,
    mode_points_per_meter,
    multi_profile_media,
    multi_profile_n_mesh,
    reject_band_over_field_limit,
    resolve_modes_launch,
    resolve_rmax_m,
    segment_env_for_field,
)
from uacpy.models.kraken._modes import (
    _NO_MODES_ERROUT,
    build_modes_field,
    check_field_modes_trapped,
    check_non_trapped_modes,
    read_group_speeds_from_prt,
    read_modes_file,
    refuse_a_profile_without_modes,
)
from uacpy.models.kraken._extract import (
    assemble_field_from_shd,
    deck_source,
    evaluated_receiver,
    launch_of,
    read_shd,
)
from uacpy.models.kraken._launch import (
    _ADIABATIC_BASE,
    _CHECK_BASE,
    _COUPLED_GAP_WARN_DB,
    _FIELD_BASE,
    _FIELD_PRT_ROOT,
    _KRAKEN_FIELD_OUTPUTS,
    _KRAKEN_MODES_OUTPUTS,
    _PROBE_BASE,
    attach_field_prt_path,
    far_field_gap_dB,
    raise_on_field_fatal,
    require_band_axis,
    require_field_shd,
    tolerate_field_teardown,
    write_decks,
    write_modes_deck,
)
from uacpy.core.engine_defaults import KRAKEN_N_MESH


# The modes whose result is a transfer function H(f) (TIME_SERIES then
# synthesises it with the source pulse).
_BAND_MODES = frozenset({RunMode.BROADBAND, RunMode.TIME_SERIES})


class Kraken(PropagationModel):
    """
    Kraken - Normal modes + field computation (multi-backend).

    One model over the AT Kraken pipeline, mirroring ``RAM``'s
    ``backend=`` dispatcher. The modes binary is selected by ``backend=``
    (``'kraken'`` real / ``'krakenc'`` complex, or ``None`` to auto-pick
    krakenc for elastic / leaky media). ``field.exe`` runs only when the
    requested run mode produces a field:

    * ``RunMode.MODES`` (``compute_modes``) → runs the modes binary only,
      returns a :class:`Modes` result (``k``, ``phi``).
    * ``COHERENT_TL`` / ``BROADBAND`` / ``TIME_SERIES``
      (``compute_tl`` / ``compute_transfer_function`` / ``compute_time_series``)
      → modes binary → ``field.exe`` → ``.shd``, returns a :class:`Field`.
    * ``INCOHERENT_TL`` → same pipeline with ``field.exe`` ``Opt(4:4)='I'``,
      which drops the range phase (``ik = REAL(ik)``) and returns
      ``SQRT(SUM(z**2))`` over the per-mode contributions ``z``
      (``EvaluateMod.f90:43,66``). That is the energy sum
      ``SQRT(SUM(|z|**2))`` only while the mode functions are real, i.e.
      on the ``kraken.exe`` path; with ``backend='krakenc'`` the complex
      ``phi``/``k`` leave cross-mode phase in the square and the run
      warns. The result is a real :class:`Field` in dB (``kind='pressure'``,
      ``unit='dB'`` — transmission loss is the dB unit of the pressure
      kind, not a kind of its own; see :attr:`Field.kind`). It
      carries no phase, so it cannot feed a time-series synthesis.
      ``mode_coupling='coupled'`` has no incoherent path in ``field.exe``
      and is rejected.

    Note that ``run()`` defaults to a field mode (``COHERENT_TL``, or
    ``BROADBAND`` when a multi-frequency ``frequencies=`` is given) — use
    ``compute_modes(...)`` (or ``run(run_mode=RunMode.MODES)``) for modes.

    Supports range-independent and range-dependent environments via
    adiabatic or coupled mode theory (delegated to AT's field.exe); the
    MODES path is range-independent and samples the r=0 profile of any RD
    environment (with a warning).

    Note
    ----
    Range-dependent bathymetry is supported via environment segmentation
    (multi-profile .env + field.exe coupled/adiabatic modes). Sea-surface
    altimetry (non-flat sea surface) is NOT supported; Bellhop and RAM
    (``ramsurf`` backend) support altimetry.

    Note
    ----
    ``Source(beam_pattern=...)`` means something DIFFERENT here than it does
    on Bellhop, and the ``.sbp`` file is not portable between the two.
    ``field.exe`` shades the mode amplitudes rather than launch angles
    (``KrakenField/field.f90:189-211``), which brings the restrictions the
    file format itself does not show:

    1. **Every source depth gets the same shading.** The factor is computed
       once per frequency under ``IF ( iS == 1 )`` inside
       ``SourceDepths: DO iS = 1, Pos%Nsz`` (``:184``) and applied to every
       ``iS`` (``:210``) — the vendored ``field.f90`` is patched for this
       (``third_party/MODIFICATIONS.md``), so a multi-depth ``Source`` with a
       beam pattern shades every depth.
    2. **The reference speed is hard-coded at 1500 m/s**, not the speed at
       the source: ``c0 = 1500`` at ``:203``, carrying Porter's own
       ``!!! ... should be speed at the source depth``. Every angle below is
       computed against it, so in water that is not near 1500 m/s the
       pattern is applied at the wrong angles.
    3. **Sub-1500 m/s modes are all shaded at 0°, and the angle is
       one-sided.** ``kz2 = omega**2/c0**2 - k**2`` (``:205``) is clamped by
       ``WHERE ( kz2 < 0 ) kz2 = 0`` (``:206``), so every mode whose phase
       speed is below ``c0`` gets ``kz2 = 0`` and lands in the same
       ``thetaT = 0`` bin — including the near-horizontal, longest-ranging
       modes. ``thetaT = ATAN( SQRT( kz2 ) / k )`` (``:208``) then takes the
       non-negative root, so ``thetaT`` spans ``[0°, 90°)`` and the negative
       half of the pattern table is never read.

    Bellhop instead shades by the SIGNED take-off angle of each ray, in the
    real ``c(z_s)`` (``Bellhop/bellhop.f90:263,267-274``), so it reads the
    whole table and honours up/down asymmetry. A pattern authored for one
    engine will not reproduce on the other.

    Parameters
    ----------
    executable, field_executable : Path, optional
        ``kraken.exe`` and ``field.exe`` paths. Auto-detected if ``None``;
        a relative path is made absolute against the constructor's cwd.
    mode_coupling : str, optional
        ``'adiabatic'`` (default) or ``'coupled'``. Controls how
        ``field.exe`` handles range-dependent mode transitions.
        ``'coupled'`` marches forward only and projects only the pressure
        onto each new segment's modes (``EvaluateCMMod.f90``), so it does
        not conserve energy and its answer moves with ``n_segments``
        rather than converging. On the shelf of docs/models/kraken.md §6.6
        it sits 7.5 dB above a refined RAM run in the upper 100 m past
        15 km, where the adiabatic run sits 0.13 dB below it; check either
        option against :class:`~uacpy.models.RAM` on a slope.
    n_segments : int, optional
        Range segments for RD scenarios. Default ``None`` lets
        :func:`_segments.segment_environment_by_range` pick segment
        edges from the union of bathymetry / RD-SSP / RD-bottom
        change-points, inserting intermediates wherever the gap
        exceeds :data:`_segments._MAX_SEGMENT_LENGTH_M` (2 km). Pass
        an explicit int to override with a uniform linspace decomposition.
    mode_points_per_meter : float, optional
        Mode-depth grid density in pts/m. ``None`` (default) derives it from
        the run as ``10 * freq_max / c_min`` — the ~10 points/wavelength the
        KRAKEN and FIELD manuals require of the mode tabulation grid — with a
        1.5 pts/m floor. An explicit value is used verbatim and warns if it
        falls under that.
    backend : str, optional
        Force the modes binary: ``'kraken'`` or ``'krakenc'``. ``None``
        (default) auto-selects — ``krakenc.exe`` for elastic media / leaky
        modes (complex eigenvalues) and for a tabulated reflection
        coefficient (``acoustic_type='file'`` / ``'precalc'``, which
        ``kraken.exe`` either refuses or silently replaces with a rigid
        boundary), ``kraken.exe`` otherwise. Forcing ``'kraken'`` on either
        raises ``ConfigurationError``, and so does forcing ``'krakenc'`` onto
        a rigid or vacuum seabed under a fluid column, which krakenc solves
        wrongly at every window (kraken.exe solves it).
    c_low : float, optional
        Lower phase speed limit (m/s). None ⇒ 0.0 when no medium carries
        shear, which makes KRAKEN compute cLow automatically — the
        modal-solver default; when any medium is elastic, None ⇒ the
        minimum compressional speed in the problem (SSP and bottom), which
        keeps the search off the interfacial branch (``_window.c_low_for``).
        A positive c_low skips slower modes and excludes interfacial
        (Scholte / Stoneley) modes; set it to the minimum p-wave speed if KRAKEN
        fails to converge on those. (The 0.95·min-SSP rule is the Scooter/SPARC
        wavenumber-integration default, not Kraken's.) Must be non-negative and
        strictly less than ``c_high``.
    c_high : float, optional
        Upper phase speed limit (m/s). None = auto, per profile of the deck:
        1.05 x the fastest speed among the SSP and the seabed half-space;
        10 x the fastest water speed over a 'file' / 'precalc'
        reflection-table seabed (``_TABLE_BOTTOM_C_HIGH_FACTOR``), and over a
        rigid or vacuum floor on krakenc; unbounded (1e9) over a rigid or
        vacuum floor on kraken.exe, and with ``leaky_modes=True``. Must be
        strictly greater than ``c_low``. ``run_settings(...).engine`` shows
        the window and the rule that set it.
    n_mesh : int, optional
        Total number of mesh points PER MEDIUM used by the finite-difference
        mode solver (AT's ``NMESH`` column on the SSP mesh line). 0 = let
        Kraken pick automatically from frequency / wavelength. Default: 0.
        Note: this is NOT a "points per wavelength" density — it is a total
        point count per medium. On a hard elastic seabed the automatic mesh
        is the coarse end of a converging ladder with holes in it — 100 m of
        water over cp 3000 / cs 1400 at 100 Hz, against Scooter: 0.51 dB
        median at 0 (auto), 0.24 at 250, 0.10 at 500, 0.055 at 1500, 0.050
        at 3000, but 2.9 dB at 200, 400, 1000, 2000 and 4000, where KRAKENC
        lost modes refining its mesh (fluid seabeds sit at 0.01-0.07 dB with
        the default). A krakenc solve of an elastic problem is also solved
        on AT's coarsest accepted mesh, and a run that kept fewer modes than
        that solve warns (see ``_grid.mode_count_check_mesh``); pick another
        ``n_mesh`` then.
    interp_ssp : str, optional
        SSP connection scheme written into ``TopOpt(1)``. ``None``
        (default) resolves to ``'linear'`` (C-linear). Explicit values:
        ``'linear'``, ``'n2linear'``, ``'pchip'``,
        ``'spline'``. ``'quad'`` is Bellhop-only and is
        rejected with an :class:`UnsupportedFeatureError`.
    n_modes : int, optional
        Cap on the number of modes field.exe propagates (FLP ``MLimit``).
        ``None`` (default) uses every mode the solver found. Also truncates a
        :class:`Modes` result to the first ``n_modes``, matching what the
        field evaluation used; :meth:`Modes.first_n` slices an existing result
        after the fact.
    leaky_modes : bool, optional
        If True, raise ``c_high`` to 10 x the fastest speed in each profile
        (``_window._LEAKY_C_HIGH_FACTOR``; KRAKENC's root finder fails or
        returns a wrong field at 1e9) so the modes binary attempts leaky
        modes — modes whose phase speed exceeds the half-space S- or P-wave
        speed, so they radiate into the half-space instead of being trapped.
        This forces ``backend='krakenc'``, because only complex arithmetic can
        represent a leaky eigenvalue: ``doc/kraken.htm:654-657`` has "KRAKENC
        will attempt to compute leaky modes if CHIGH exceeds the phase velocity
        of either the S-wave or P-wave speed in the half-space".

        ``False`` (the default) does **not** mean "no leaky modes". The same
        page claims KRAKEN "will (if necessary) reduce CHIGH so that only
        trapped (non-leaky) modes are computed" (``doc/kraken.htm:648-650``),
        but the vendored source only does that for an **elastic** half-space:
        ``Kraken/kraken.f90:209`` clamps cHigh to the shear speed, while the
        acoustic clamp one branch below is commented out —
        ``Kraken/kraken.f90:212``, ``! cHigh = MIN( cHigh, DBLE( HSBot%cP ) )``.
        Over a fluid seabed cHigh therefore stands as written, and the auto
        value sits 5 % past the bottom speed
        (:data:`~uacpy.models._window.C_HIGH_FACTOR`), so an ordinary run keeps
        a few modes with ``c_p`` above it. :meth:`compute_modes` logs how many
        at ``verbose='info'``, and refuses outright when *every* returned mode
        is one of them. Default: False.

        With the automatic window a field run warns when the receivers need
        paths steeper than it keeps (the near field of a trapped-mode sum,
        measured +11.9 dB at 200 m on a 100 m Pekeris guide); ``True`` is the
        remedy there (within 0.04 dB of Scooter). Refused over a rigid or
        vacuum seabed under a fluid column: there is no half-space to leak
        into, and the krakenc solve it forces fails there.
    top_reflection_file : Path, optional
        A ``.trc`` top-reflection-coefficient table. Overrides the surface
        boundary condition to ``'F'`` and is staged next to the ``.env``.
    rmax_m : float, optional
        ``RMax`` written into the ``.env`` (m here; the writer converts to
        the km the deck expects). It is the mode solver's
        mesh-convergence tolerance, scaled to range, and nothing else:
        ``kraken.f90:80`` / ``krakenc.f90:82`` refine the mesh until
        ``Error·1000·RMax < 1``, where ``Error`` is the change in the
        Richardson-extrapolated eigenvalue between two successive meshes
        and ``1000·RMax`` is RMax in metres. A **larger** RMax is a
        **tighter** tolerance (finer mesh, slower, more accurate).
        ``None`` (default) derives it from the outermost receiver range:
        ×1.05 for narrowband, ×3 for a broadband sweep. A receiver with no
        positive range gives it nothing to scale, and falls back to
        100 km — the tightest of these tolerances. ``compute_modes`` is
        always that case: it takes no receiver at all, because the
        eigenfunctions it returns are not evaluated at one, so its modes are
        converged to the 100 km target regardless of where the caller later
        propagates them.
    mode_depths : ndarray, optional
        Explicit depth grid (m) the ``.mod`` eigenfunctions are sampled
        on by ``compute_modes``. ``None`` (default) builds
        ``max(100, total_media_depth × mode_points_per_meter)`` points
        spanning water + sediment.
    use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
        Standard plumbing (see :class:`PropagationModel`).

    Notes
    -----
    ``field.exe`` ``Opt(3)`` only accepts ``'*'``, ``'O'``, or ``' '``
    (``field.f90:83-90``); anything else raises FATAL ERROR. Purely
    elastic component selection (H/V/T/N) is not reachable through
    ``field.exe`` — an upstream Fortran limitation.

    **Auto-route to krakenc.exe** when ``env`` carries shear (delegates the
    modes step to the complex-arithmetic binary).

    Defaults auto-derived at ``run()`` time (override only when tuning):

    - ``c_low=None`` → ``0.0`` (KRAKEN computes cLow automatically); with
      shear anywhere in the problem → the minimum compressional speed
    - ``c_high=None`` → ``max(max(env.ssp), env.bottom.sound_speed) × 1.05``;
      ``10 × max(env.ssp)`` over a 'file' / 'precalc' reflection-table seabed
      (``_TABLE_BOTTOM_C_HIGH_FACTOR``) and over a rigid or vacuum floor on
      krakenc; ``1e9`` over a rigid or vacuum floor on kraken.exe; ``10 x``
      the fastest speed in the profile with ``leaky_modes=True``
    - ``n_mesh=0`` → Kraken picks mesh from frequency / wavelength.
    - TopOpt position 4 reads ``env.absorption`` (``Thorp`` / ``FrancoisGarrison``
      / ``Biological`` / ``ConstantAbsorption`` / ``None``).

    With ``verbose='info'`` the resolved ``c_low`` / ``c_high`` are logged.

    **Range dependence.** RD bathymetry, RD SSP and an RD bottom are
    honoured natively on the field path (segments): each profile carries the
    seabed column at its own range, layers included. ``compute_modes``
    samples the r = 0 profile of every quantity. A range-dependent surface
    collapses by :data:`DEFAULT_COLLAPSE`.

    Examples
    --------
    >>> kraken = Kraken()
    >>> tl = kraken.compute_tl(env, source, receiver)        # field via field.exe

    >>> modes = kraken.compute_modes(env, source)            # modes only (no receiver)

    >>> # Elastic bottom → complex modes (auto, or force backend='krakenc')
    >>> modes = Kraken(backend='krakenc').compute_modes(env_elastic, source)

    >>> # Range-dependent with coupled modes
    >>> kraken = Kraken(mode_coupling='coupled', n_segments=20)
    >>> tl = kraken.compute_tl(env_rd, source, receiver)
    """

    def _run_kraken_executable(self, base_name: str, work_dir: Path, exe=None,
                               launch: Optional[KrakenLaunch] = None, *,
                               report_prt_warnings: bool = True):
        """Execute the modes binary (``exe`` selects kraken.exe vs krakenc.exe;
        defaults to the resolved kraken.exe) via the shared binary-launch
        helper. ``launch`` is the deck's resolved settings, which the
        no-mode diagnosis quotes; ``report_prt_warnings=False`` for a solve
        whose ``.prt`` does not speak for the run (the mode-count check)."""
        try:
            self._run_and_attach_prt(
                [str(exe or self._exe), base_name], work_dir, base_name,
                stale_outputs=_KRAKEN_MODES_OUTPUTS,
                report_prt_warnings=report_prt_warnings)
        except ModelExecutionError as exc:
            # KRAKENC's no-mode branch re-OPENs a mode file it already holds
            # open (Kraken/krakenc.f90:431-443, after 'No modes OPEN
            # statement' on stdout), so gfortran stops with an OPEN error
            # before the 'No modes for given phase speed interval' ERROUT
            # can say what happened. Measured: a BOUNCE table of an elastic
            # half-space (cp 1600, cs 400) read at 100 Hz with c_high=1e9 —
            # Bounce's full-span table bound — searched 3369 candidate modes,
            # the secant root finder failed on them, and no mode was kept;
            # c_high=1e4 gives 11-13 modes (by table sampling) and
            # c_high=1600 the 5 trapped ones. The same 100 Hz elastic
            # half-space with leaky_modes=True at c_high=1e9 (default c_low)
            # ends here after 26.8 s; with c_low=1400 it ran past a 300 s
            # timeout. The documented fluid case (100 m over a 1650 m/s
            # half-space, 200 Hz) returns 27 leaky modes in under a second.
            text = f"{exc.stdout or ''}\n{exc.stderr or ''}"
            if ('No modes OPEN statement' in text
                    or 'Cannot change STATUS parameter' in text):
                raise ModelExecutionError(
                    self.model_name, return_code=exc.return_code,
                    stdout=exc.stdout,
                    stderr=(
                        f"KRAKENC kept no modes in c_low..c_high = "
                        f"{'auto' if launch is None else f'{launch.c_low:g}'}.."
                        f"{'auto' if launch is None else f'{max(launch.c_high):g}'}"
                        f" m/s (its root "
                        f"finder "
                        f"failed on every candidate; see the RootFinderSecant "
                        f"warnings in the .prt). A c_high far above the "
                        f"seabed's speeds asks for every leaky mode; pass "
                        f"c_high at the seabed compressional speed for the "
                        f"trapped modes, or a few times it for the "
                        f"strongest leaky ones.\n\n{exc.stderr or ''}"),
                ) from exc
            raise

    @classmethod
    def _per_launch_knob_values(cls, engine):
        """``c_low``, ``c_high``, ``n_mesh`` and ``rmax_m`` over the
        launches (``c_high`` one per profile of each; none with
        ``leaky_modes``, which sets it)."""
        launches = engine.launches
        values = {
            'c_low': [launch.c_low for launch in launches],
            'c_high': [value for launch in launches
                       for value in launch.c_high],
            'n_mesh': [launch.n_mesh for launch in launches],
            'rmax_m': [launch.rmax_m for launch in launches],
        }
        if engine.knobs.get('leaky_modes'):
            # leaky_modes sets c_high; the two are refused together, so
            # c_high follows leaky_modes.
            del values['c_high']
        return values

    # Declarative metadata (see PropagationModel / ModelSpec). Kraken: normal
    # modes. The field path segments RD bathymetry, RD SSP and an RD bottom
    # natively: every profile block of the multi-profile .env is a full
    # environment read by its own ReadEnvironment call (kraken.f90:42-46), and
    # the .mod stores each profile's own half-space (ReadModes.f90:69), so
    # each segment carries ``env.bottom.at(range=r)`` with its layers. A
    # range-dependent *surface* still collapses. Honours layered + elastic
    # bottom. The MODES path samples r = 0 (``_checks.modes_single_profile``).
    #
    # No ``'ssp'`` or ``'bottom_range'`` entry: the only readers of those keys
    # are ``_project_environment``'s range-dependent branches, guarded by the
    # ``range_dependent_ssp`` / ``range_dependent_bottom`` capabilities this
    # spec declares, so a Kraken default there would advertise a collapse that
    # never runs.
    spec = ModelSpec(
        modes=(
            RunMode.MODES, RunMode.COHERENT_TL, RunMode.INCOHERENT_TL,
            RunMode.BROADBAND, RunMode.TIME_SERIES,
        ),
        supports={
            'range_dependent_bathymetry',
            'range_dependent_ssp',
            'range_dependent_bottom',
            'layered_bottom',
            'elastic_media',
            'source_beam_pattern',
            'rough_surface',
            'rough_bottom',
        },
        source_types=frozenset({'point', 'line', 'scaled'}),
        traits=EngineTraits(
            consumes_run_t_start=True,
            # MODES tabulates on a depth grid the engine builds
            # (_grid.dense_mode_depths) when no receiver is given.
            none_receiver_modes=frozenset({RunMode.MODES}),
            # TopOpt position 4 carries env.absorption to the engine.
            consumes_volume_absorption=True,
            # Meshes through fluid sediment layers: receivers resolve down to
            # the deepest interface, not just the seafloor.
            receivers_reach_sediment=True,
            # field.exe evaluates every source depth of the .flp from the one
            # .mod kraken.exe wrote — the mode solve reads no source depth at
            # all — and the .shd reader splits its NSz axis into a ResultStack,
            # so the two TL modes take a multi-depth Source in one launch
            # instead of one mode solve per depth (measured 4.3x at eight
            # depths). Two upstream obstacles had to go first: field.f90 shaded
            # only the FIRST source depth of a .sbp run (patched —
            # third_party/MODIFICATIONS.md), and kraken.f90:573 merges Pos%Sz
            # into the mode-tabulation grid that ReadModes.f90:54 interpolates
            # the receiver mode shapes from, so an extra source depth moved a
            # receiver's answer — until the receiver depths joined that grid
            # (``_resolve_field_launch``) and stopped being interpolated at
            # all. MODES takes one too: the modes do not depend on the source
            # depth, and the deck tabulates them at every source depth
            # (kraken.f90:573), where ``Modes.excitation`` reads them.
            # BROADBAND / TIME_SERIES keep the per-depth loop: their .mod
            # carries one block per frequency and the synthesis is per-depth
            # anyway.
            native_multi_depth_modes=frozenset({
                RunMode.MODES, RunMode.COHERENT_TL, RunMode.INCOHERENT_TL,
            }),
            # Below the waveguide's modal cutoff KRAKEN funnels through
            # ERROUT, but "no trapped modes in this phase-speed interval" is
            # a physical answer this model surfaces itself — as an all-NaN
            # field plus a warning, NaN bins from the broadband floor
            # search, or a typed error from ``compute_modes`` — not a run
            # failure.
            benign_fortran_fatals=(_NO_MODES_ERROUT,),
        ),
    )

    provenance_id = 'acoustics_toolbox'

    # A mode set for MODES; complex pressure in the travelling-wave convention
    # (field.exe's sum times -1, see _extract.assemble_field_from_shd) for
    # the coherent field and H(f); real dB with no phase for the magnitude
    # sum of INCOHERENT_TL; the synthesised p(t) for TIME_SERIES.
    outputs = MappingProxyType({
        RunMode.MODES: OutputSpec('Modes'),
        RunMode.COHERENT_TL: OutputSpec(
            'Field', kind='pressure', unit='Pa',
            phase_reference=PhaseReference.TRAVELLING_WAVE.value,
            coherent=True),
        RunMode.INCOHERENT_TL: OutputSpec(
            'Field', kind='pressure', unit='dB', coherent=False),
        RunMode.BROADBAND: OutputSpec(
            'Field', kind='pressure', unit='Pa',
            phase_reference=PhaseReference.TRAVELLING_WAVE.value,
            coherent=True),
        RunMode.TIME_SERIES: OutputSpec(
            'Field', kind='pressure', unit='Pa',
            phase_reference=PhaseReference.TIME_DOMAIN_NATIVE.value),
    })

    # Below the waveguide's modal cutoff KRAKEN funnels through ERROUT, but
    # "no trapped modes in this phase-speed interval" is a physical answer
    # this model surfaces itself — as an all-NaN field plus a warning from
    # ``_compute_field_via_exe``, NaN bins from the broadband floor search,
    # or a typed error from ``compute_modes`` — not a run failure.

    def __init__(
        self,
        *,
        executable: Optional[Path] = None,
        field_executable: Optional[Path] = None,
        backend: Optional[str] = None,
        c_low: Optional[float] = None,
        c_high: Optional[float] = None,
        n_mesh: int = KRAKEN_N_MESH,
        n_modes: Optional[int] = None,
        interp_ssp: Optional[str] = None,
        leaky_modes: bool = False,
        top_reflection_file: Optional[Path] = None,
        rmax_m: Optional[float] = None,
        mode_depths: Optional[np.ndarray] = None,
        mode_points_per_meter: Optional[float] = None,
        mode_coupling: str = 'adiabatic',
        n_segments: Optional[int] = None,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        cleanup: Optional[bool] = None,
        timeout: float = DEFAULT_RUN_TIMEOUT_S,
        collapse: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
            cleanup=cleanup, timeout=timeout, collapse=collapse,
        )
        # --- modal-solver knobs ---
        self.interp_ssp = interp_ssp
        # c_low default 0.0 → KRAKEN computes cLow
        # automatically; a positive c_low skips slower/interfacial (Scholte /
        # Stoneley) modes. Stored raw (None preserved) so repr/copy stay clean;
        # resolved via ``_window.c_low_for`` at write time.
        self.c_low = None if c_low is None else float(c_low)
        self.c_high = c_high
        self.n_mesh = n_mesh
        self.n_modes = n_modes
        self.leaky_modes = leaky_modes
        self.top_reflection_file = (
            Path(top_reflection_file) if top_reflection_file is not None else None
        )
        # rmax_m scales the mode solver's mesh-convergence tolerance;
        # None → derive at run() from receiver.range_max.
        self.rmax_m = rmax_m
        # mode_depths overrides compute_modes's dense grid; None → density.
        self.mode_depths = (
            np.asarray(mode_depths, dtype=float)
            if mode_depths is not None else None
        )
        self.backend = backend
        # Run modes, capability flags and collapse defaults come from the
        # class-level ``spec`` (applied by PropagationModel.__init__).
        self.mode_coupling = mode_coupling
        self._check_knobs()
        self.n_modes = None if n_modes is None else int(n_modes)
        self.rmax_m = float(rmax_m) if rmax_m is not None else None

        # The resolved kraken.exe path lives in ``self._exe``;
        # a run on krakenc looks krakenc.exe up (``_modes_exe``).
        self._exe = self._resolve_executable(
            executable,
            lambda: self._find_executable_in_paths(
                'kraken.exe', bin_subdirs='oalib',
                dev_subdir='Acoustics-Toolbox/Kraken',
            ),
        )

        # field.exe is only needed for field-producing run modes (TL /
        # broadband / time-series), not for MODES. Store the user arg for
        # copy() round-tripping and resolve lazily on first field run so a
        # MODES-only Kraken doesn't require field.exe to be installed.
        self.field_executable = (
            Path(field_executable) if field_executable is not None else None
        )
        # A pinned path is made absolute against the constructor's cwd: every
        # launch runs with ``cwd=`` the scratch dir, where a relative path
        # names nothing (the same rule _resolve_executable applies to
        # ``executable``). Existence is checked lazily, at launch.
        self._field_exe_pinned: Optional[Path] = (
            self.field_executable.expanduser().resolve()
            if self.field_executable is not None else None
        )
        self._field_exe: Optional[Path] = None

        self.mode_points_per_meter = mode_points_per_meter
        self.n_segments = n_segments

    def _resolve_field_executable(self) -> Path:
        """Resolve ``field.exe`` lazily (only field-producing run modes need
        it). Cached in ``self._field_exe`` after the first call."""
        if self._field_exe is not None:
            return self._field_exe
        if self._field_exe_pinned is not None:
            fx = self._field_exe_pinned
        else:
            fx = self._find_executable_in_paths(
                'field.exe',
                bin_subdirs='oalib',
                dev_subdir='Acoustics-Toolbox/Kraken',
            )
        if not fx.exists():
            raise ExecutableNotFoundError(
                f"{self.model_name} (field.exe)", str(fx),
            )
        self._field_exe = fx
        return fx

    def select_backend(self, env, run_mode=None) -> str:
        """Logical name of the modes backend that would run for ``env`` —
        ``'krakenc'`` for elastic media, leaky modes (complex eigenvalues)
        or a tabulated reflection coefficient, ``'kraken'`` otherwise.
        Round-trips with ``backend=`` and mirrors ``RAM.select_backend``
        (``run_mode`` accepted for signature parity; it does not affect the
        modes-binary choice). Pure introspection of the dispatch — no disk
        access; :meth:`_modes_exe` resolves the named binary and
        :meth:`_reject_malformed_irc_bottom` checks a ``'precalc'`` bottom's
        ``.irc`` header on the run path.

        Parameters
        ----------
        env : Environment
            The environment the dispatch is for.
        run_mode : RunMode, optional
            Accepted for signature parity; it does not change the choice.

        Raises
        ------
        UnsupportedFeatureError
            For a ``'precalc'`` sea surface — no Kraken-family binary reads
            a top ``.irc`` table.
        ConfigurationError
            When ``backend='kraken'`` is forced on an environment
            kraken.exe cannot answer (elastic media / leaky modes, or a
            tabulated reflection coefficient).
        """
        reject_precalc_surface(env, model_name=self.model_name)
        complex_modes = (
            env.bottom.is_elastic
            or env.surface.is_elastic
            or getattr(self, 'leaky_modes', False)
        )
        reflection_tables = reflection_table_boundaries(
            env, top_reflection_file=self.top_reflection_file)
        forced = getattr(self, 'backend', None)
        if forced == 'kraken' and complex_modes:
            raise ConfigurationError(
                "Kraken(backend='kraken') answers incorrectly on elastic media "
                "/ leaky modes: kraken.exe clamps c_high to the half-space "
                "shear speed (dropping every faster mode) and its absorption "
                "perturbation skips elastic media (returning Im(k)=0). Use "
                "backend='krakenc', or backend=None for automatic dispatch."
            )
        if forced == 'kraken' and reflection_tables:
            raise ConfigurationError(
                f"Kraken(backend='kraken') cannot honour the tabulated "
                f"reflection coefficient on the "
                f"{', '.join(reflection_tables)}: kraken.exe either stops "
                f"('The option to read a file for the reflection loss is not "
                f"implemented in KRAKEN') or silently substitutes a rigid "
                f"boundary. Use backend='krakenc', or backend=None for "
                f"automatic dispatch."
            )
        if forced == 'krakenc' or (
                forced is None and (complex_modes or reflection_tables)):
            return 'krakenc'
        return 'kraken'

    def _project_environment(self, env, *, request=None):
        """Collapse unsupported features, then express ``top_reflection_file``
        as the surface carrier it is shorthand for.

        Staging goes through the one path that owns it — ``write_header``
        copies ``env.surface.reflection_file`` to ``<root>.trc``, which
        ``misc/RefCoef.f90:64-76`` opens for ``TopOpt(2:2)='F'``.

        This rewrite belongs here, not in a deck writer: ``_write_field_env``
        branches to ``write_multi_profile_env`` for a range-dependent run and
        never calls ``_write_kraken_env``, so a rewrite living there left the
        multi-profile deck with a vacuum ``TopOpt`` and no staged ``.trc`` on
        exactly the runs ``_checks.reflection_table_boundaries`` had already
        routed to ``krakenc``. Every entry path projects first, so doing it
        here reaches both writers.

        The roughness drop runs last, on the resolved surface, so it covers
        both ways a tabulated top reaches the deck — this rewrite and a
        user-built ``Surface(acoustic_type='file')``.
        """
        env = super()._project_environment(env, request=request)
        if self.top_reflection_file is not None:
            if not self.top_reflection_file.exists():
                raise ConfigurationError(
                    f"top_reflection_file not found: {self.top_reflection_file}."
                )
            # Replace only the boundary condition; the roughness carries over
            # so the drop below sees the value the user set, and reports it.
            env.surface = Surface(nodes=[BoundaryProperties(
                acoustic_type='file',
                reflection_file=str(self.top_reflection_file),
                roughness=env.surface.roughness)])
        return drop_roughness_on_tabulated_top(env, model_name=self.model_name)

    def _default_run_mode(self) -> RunMode:
        """``run(run_mode=None)`` solves a TL field (``BROADBAND`` when a
        ``frequencies=`` vector is passed, else ``COHERENT_TL``), not the
        ``MODES`` entry that opens ``spec.modes``. Either choice is a field
        mode, which is all the base's per-depth loop asks."""
        return RunMode.COHERENT_TL

    def _default_run_mode_for(self, frequencies) -> RunMode:
        """``BROADBAND`` for a ``frequencies=`` vector of more than one
        element, else :meth:`_default_run_mode`."""
        if frequencies is not None and len(np.atleast_1d(frequencies)) > 1:
            return RunMode.BROADBAND
        return self._default_run_mode()

    # ── stages 2-3: refusals and settings ───────────────────────────────

    def _check_knobs(self) -> None:
        """Refuse a constructor knob no run could use: a mode cap below 1, a
        non-positive ``rmax_m``, ``leaky_modes=True`` beside a pinned
        ``c_high``, a pinned phase-speed pair out of order, an unknown
        ``backend`` or ``mode_coupling``.

        Run at construction and again by every run (the attributes can be
        reassigned in between). A single pinned bound is held to the other
        once a run derives it (:func:`_window.phase_speed_window`).
        """
        n_modes = self.n_modes
        if n_modes is not None and int(n_modes) < 1:
            raise ConfigurationError(
                f"Kraken(n_modes={n_modes}): the mode cap must be >= 1. "
                "field.exe treats MLimit <= 0 as zero propagating modes "
                "(field.f90:68,185), which returns an empty field.")
        # rmax_m scales the mode solver's mesh-convergence tolerance;
        # None → derive at run() from receiver.range_max. Zero or negative
        # short-circuits that test: ``kraken.f90:80`` leaves the refinement
        # loop as soon as ``Error*1000*RMax < 1`` and ``Error`` starts at
        # 1e10, so RMax <= 0 makes it true on the first, coarsest mesh — no
        # Richardson extrapolation, measured 2.37 dB max |ΔTL| against the
        # default, at exit 0. Bounce rejects the same value.
        # The pinned mode-depth grid is the deck's receiver-depth line, held
        # to what a Receiver's depths are held to.
        if self.mode_depths is not None:
            grid = np.atleast_1d(self.mode_depths)
            if grid.size < 1:
                raise ConfigurationError(
                    "Kraken(mode_depths=...) must hold at least one "
                    "depth.")
            require_non_negative(grid, "Kraken.mode_depths",
                                 hint="metres, positive down from surface")
            require_strictly_increasing(grid, "Kraken.mode_depths",
                                        min_step=DECK_DEPTH_RESOLUTION_M)
        rmax_m = self.rmax_m
        if rmax_m is not None and float(rmax_m) <= 0.0:
            raise ConfigurationError(
                f"Kraken(rmax_m={rmax_m}) must be > 0: RMax scales the mode "
                f"solver's mesh-convergence test (kraken.f90:80, "
                f"Error*1000*RMax < 1), and a non-positive value satisfies it "
                f"on the coarsest mesh, so the modes come back unrefined and "
                f"unextrapolated with nothing in the .prt to say so. Pass the "
                f"longest range the modes will be propagated to, or None to "
                f"derive it from receiver.ranges."
            )
        # leaky_modes sets CHIGH per profile at deck time
        # (_window.phase_speed_window); pinning c_high alongside it is a
        # contradiction refused here rather than resolved by discarding the
        # pinned value silently — and c_high stays stored verbatim so
        # copy()/repr() round-trip the constructor args.
        if self.leaky_modes and self.c_high is not None:
            raise ConfigurationError(
                f"Kraken(leaky_modes=True) sets CHIGH itself, to "
                f"{_LEAKY_C_HIGH_FACTOR:g} x the fastest speed in each "
                f"profile, for krakenc to search leaky modes; an explicit "
                f"c_high={self.c_high} contradicts it. Pass one or the other."
            )
        validate_phase_speed_limits(leaky_modes=self.leaky_modes,
                                    pinned_c_high=self.c_high,
                                    pinned_c_low=self.c_low)
        if self.backend is not None and self.backend not in ('kraken',
                                                             'krakenc'):
            raise ConfigurationError(
                f"Kraken(backend={self.backend!r}) is not a known backend. "
                f"Choose 'kraken', 'krakenc', or None for automatic dispatch."
            )
        if self.mode_coupling not in ('adiabatic', 'coupled'):
            raise ConfigurationError(
                f"mode_coupling must be 'adiabatic' or 'coupled', "
                f"got {self.mode_coupling!r}."
            )

    def _normalise_env(self, env, mode):
        """Stage 1 of a MODES call reduces a range-dependent environment to
        its r = 0 profile (:func:`_checks.modes_single_profile`) before the
        shared projection: normal modes are range-independent, while every
        field mode segments the environment itself."""
        if mode == RunMode.MODES:
            return modes_single_profile(env, collapse=self._collapse,
                                        model_name=self.model_name)
        return env

    def _validate_engine(self, env, source, receiver, run_mode, *,
                         request=None) -> None:
        """Stage 2: the refusals of carriers Kraken cannot run — on the
        projected environment, so ``validate_inputs`` refuses what ``run``
        refuses.

        - an ``interp_ssp`` the Kraken deck cannot carry ('quad' is
          Bellhop-only: ``misc/sspMod.f90:61-89`` has no 'Q' case);
        - a fluid medium below an elastic one, and roughness on an elastic
          layer (:func:`_checks.reject_acoustic_below_elastic`,
          :func:`_checks.reject_rough_elastic_layer`), on the columns the deck
          carries;
        - coupled modes with incoherent addition on a multi-profile deck;
        - a backend the environment rules out (:meth:`select_backend`), a
          'precalc' seabed ``.irc`` in the wrong layout, and krakenc forced
          onto a rigid or vacuum floor
          (:func:`_checks.reject_krakenc_over_a_hard_or_free_floor`).
        """
        reject_unsupported_ssp_interp('Kraken', self.interp_ssp)
        reject_biological_edges_under_neighbour_interp(
            'Kraken', env, self.interp_ssp)
        reject_acoustic_below_elastic(env, run_mode,
                                      model_name=self.model_name)
        reject_rough_elastic_layer(env, run_mode, model_name=self.model_name)
        # ``KrakenField/field.f90:125-129`` calls ERROUT on Opt(2:2)='C' +
        # Opt(4:4)='I', which surfaces in Python as an opaque "no .shd file"
        # error. The deck is multi-profile exactly when the projected
        # environment is range dependent — the predicate the deck resolution
        # reads (:meth:`_resolve_field_launch`).
        if (run_mode == RunMode.INCOHERENT_TL
                and env.is_range_dependent
                and self.mode_coupling == 'coupled'):
            raise ConfigurationError(
                "Kraken: coupled mode calculations do not support "
                "incoherent addition of modes. Use mode_coupling="
                "'adiabatic' with run_mode=RunMode.INCOHERENT_TL, or keep "
                "mode_coupling='coupled' with run_mode=RunMode.COHERENT_TL."
            )
        self.select_backend(env)
        # A 'precalc' bottom is staged verbatim as <root>.irc; a table in the
        # wrong layout (typically a theta/|R|/phase angle table) aborts the
        # binary with a bare Fortran backtrace, so the header is checked
        # ahead of any launch.
        self._reject_malformed_irc_bottom(env)
        reject_krakenc_over_a_hard_or_free_floor(
            env, run_mode, forced_backend=self.backend,
            leaky_modes=self.leaky_modes,
            top_reflection_file=self.top_reflection_file)

    def _resolve_engine_settings(self, env, source, receiver, settings, *,
                                 given_env, request=None) -> KrakenSettings:
        """Stage 3: every setting of every deck of the run, resolved once
        from the projected ``env``, and the refusals of a deck Kraken cannot
        run.

        Refused: a mode tabulation longer than field.exe's ``MaxN``, a
        pinned ``n_mesh`` under AT's floor, a band longer than field.exe's
        ``MaxNfreq``, a phase-speed window with ``c_low >= c_high``, a
        TIME_SERIES pulse that implies no frequency grid.
        """
        mode = settings.mode
        backend = self.select_backend(env)
        notices = []
        if mode == RunMode.MODES:
            # compute_modes passes its mode cap as the call's request.
            asked = (None if request is None
                     else getattr(request.engine_request, 'n_modes', None))
            n_modes = self.n_modes if asked is None else asked
            if receiver is None:
                # The grid the modes are tabulated on, from the caller's
                # environment (the modes are solved on its r = 0 profile).
                tabulation, ppm_notice = dense_mode_depths(
                    given_env, source,
                    pinned_mode_depths=self.mode_depths,
                    pinned_mode_points_per_meter=self.mode_points_per_meter)
                notices.append(ppm_notice)
            else:
                tabulation = receiver.depths
            launch, c_low_origin, c_high_origin, rmax_origin = \
                resolve_modes_launch(env, source, receiver,
                                     tabulation_depths=tabulation,
                                     backend=backend,
                                     collapse=self._collapse,
                                     leaky_modes=self.leaky_modes,
                                     log=self._log,
                                     pinned_c_high=self.c_high,
                                     pinned_c_low=self.c_low,
                                     pinned_n_mesh=self.n_mesh,
                                     pinned_rmax_m=self.rmax_m)
            return KrakenSettings(
                backend=backend,
                backend_origin=backend_origin(
                    env, forced_backend=self.backend,
                    leaky_modes=self.leaky_modes,
                    top_reflection_file=self.top_reflection_file),
                route='modes',
                c_low_origin=c_low_origin,
                c_high_origin=c_high_origin,
                rmax_origin=rmax_origin,
                mode_points_per_meter=None,
                mode_coupling='none',
                n_modes=n_modes,
                evaluated_depths=None,
                receiver_keep=None,
                launches=(launch,),
                notices=tuple(n for n in notices if n is not None),
            )

        # field.exe cannot evaluate the field inside an elastic medium: the
        # binary tabulates the eigenvector over the acoustic media only, so
        # depths below the last acoustic node come back as a straight-line
        # extrapolation of the column above
        # (:func:`_checks.elastic_depth_intervals`). A *fluid* sediment layer
        # above the elastic one evaluates perfectly well, so the exclusion is
        # per medium rather than "everything under the water column"; a fluid
        # layer below one was already refused in stage 2
        # (:func:`_checks.reject_acoustic_below_elastic`), along with the
        # elastic-over-fluid-halfspace stack that hangs krakenc.exe. The
        # columns measured are the ones the deck will carry: every column of a
        # segmented field run — whose union of elastic depth spans is excluded,
        # since a receiver depth elastic at any profile has no field.exe value
        # there.
        elastic_spans = (
            [span for col in deck_bottom_columns(env, mode)
             for span in elastic_depth_intervals(env, col)]
            if env.bottom.is_layered else []
        )
        rcv, keep, partition_notice = partition_elastic_subbottom(
            env, receiver, elastic_spans, model_name=self.model_name)
        notices.append(partition_notice)

        if mode in _BAND_MODES:
            frequencies = settings.frequencies
            if frequencies is None:
                raise ConfigurationError(
                    f"Kraken.run(run_mode={mode.name}): the source pulse "
                    f"implies no frequency grid (it needs at least two "
                    f"samples).",
                    remediation="Pass frequencies= to set the grid, or a "
                                "source_waveform of two samples or more.")
            frequencies = np.asarray(frequencies, dtype=float)
            self._log(f"Broadband: {len(frequencies)} frequencies, "
                      f"{frequencies[0]:.1f}-{frequencies[-1]:.1f} Hz")
            if frequencies.size > 1 and env.is_range_dependent:
                # The multi-profile deck carries one frequency, but KRAKEN
                # solves modes at one frequency whatever the environment,
                # so the band is a launch per bin rather than a refusal.
                route = 'band_by_frequency'
                resolved = [
                    self._resolve_field_launch(
                        env, dataclasses.replace(
                            source, frequencies=np.array([f_i])),
                        rcv, marched_frequencies=None, run_mode=RunMode.COHERENT_TL,
                        backend=backend, notices=notices)
                    for f_i in frequencies]
            elif frequencies.size > 1:
                route = 'band'
                reject_band_over_field_limit(frequencies,
                                             model_name=self.model_name)
                resolved = [self._resolve_field_launch(
                    env, source, rcv, marched_frequencies=frequencies,
                    run_mode=RunMode.COHERENT_TL, backend=backend,
                    notices=notices)]
            else:
                # A single bin cannot use the native deck: both
                # ``write_header`` and ``write_kraken_env_file`` only emit
                # ``'B'`` for a vector, and AT's ``ReadfreqVec`` is not
                # exercised at ``NFreq=1``. It is solved narrowband at the
                # requested frequency and lifted onto a length-1 frequency
                # axis, so the BROADBAND contract holds for every grid size.
                route = 'band_bin'
                resolved = [self._resolve_field_launch(
                    env, dataclasses.replace(
                        source, frequencies=np.array([float(frequencies[0])])),
                    rcv, marched_frequencies=None, run_mode=RunMode.COHERENT_TL,
                    backend=backend, notices=notices)]
        else:
            route = 'field'
            resolved = [self._resolve_field_launch(
                env, source, rcv, marched_frequencies=None, run_mode=mode, backend=backend,
                notices=notices)]
            notices.append(krakenc_incoherent_sum_notice(
                mode, backend, resolved[0][0].n_profiles,
                model_name=self.model_name))

        launches = tuple(launch for launch, *_origins in resolved)
        _launch, c_low_origin, c_high_origin, rmax_origin, ppm = resolved[0]
        if self.c_high is None and not self.leaky_modes:
            notices.append(steep_path_notice(
                env, source, rcv, launches[0].c_high[0]))
        notices = [n for n in notices if n is not None]
        return KrakenSettings(
            backend=backend,
            backend_origin=backend_origin(
                env, forced_backend=self.backend, leaky_modes=self.leaky_modes,
                top_reflection_file=self.top_reflection_file),
            route=route,
            c_low_origin=c_low_origin,
            c_high_origin=c_high_origin,
            rmax_origin=rmax_origin,
            mode_points_per_meter=ppm,
            mode_coupling=(self.mode_coupling if env.is_range_dependent
                           else 'none'),
            n_modes=self.n_modes,
            evaluated_depths=(None if keep is None
                              else np.asarray(rcv.depths, dtype=float)),
            receiver_keep=None if keep is None else tuple(keep),
            launches=launches,
            notices=tuple(n for n in notices if n is not None),
        )

    def _resolve_field_launch(self, env, source, receiver, *, marched_frequencies, run_mode,
                              backend, notices):
        """One launch of a field run: the modes deck (single- or
        multi-profile) and the field.exe option that sums it, for
        ``source`` (its frequency, or the ``marched_frequencies`` of a ``TopOpt(6)='B'``
        deck) on the evaluable ``receiver``. Appends the notices it finds to
        ``notices``. Returns ``(launch, c_low_origin, c_high_origin,
        rmax_origin, mode_points_per_meter)``.

        Mode depths span the full ocean + sediment of every profile, and the
        caller's receiver depths join them. kraken.exe tabulates the mode
        shapes at these depths (merged with the source depths,
        ``kraken.f90:573``) and field.exe interpolates the receiver values off
        that table (``ReadModes.f90:54``), so a receiver that is not on the
        grid is interpolated twice: FE mesh -> grid -> receiver. Putting it on
        the grid leaves one interpolation. It also makes the receiver values
        independent of the source depths, which is what lets a multi-depth
        deck return slabs equal to their stand-alone runs.
        """
        freqs = marched_frequencies if marched_frequencies is not None else source.frequencies
        if env.is_range_dependent:
            segments, n_profiles, profile_ranges_m, max_total_depth = \
                segment_env_for_field(env, freq=freqs, log=self._log,
                                      mode_coupling=self.mode_coupling,
                                      n_segments=self.n_segments)
        else:
            segments, profile_ranges_m = None, None
            max_total_depth = self._total_media_depth(env)

        ppm, ppm_notice = mode_points_per_meter(
            env, freqs,
            pinned_mode_points_per_meter=self.mode_points_per_meter)
        notices.append(ppm_notice)
        n_mode_depths = max(100, int(max_total_depth * ppm))
        mode_depths = np.linspace(0, max_total_depth, n_mode_depths)
        mode_depths = merge_depths(mode_depths, receiver.depths,
                                   max_total_depth)
        check_field_tabulation_size(mode_depths, source)
        notices.append(mode_file_size_notice(
            env, mode_depths, freqs,
            memory_backed=work_dir_is_memory_backed(self.scratch_policy())))

        if segments is not None:
            rmax_m, rmax_origin = resolve_rmax_m(receiver, band=False,
                                                 pinned_rmax_m=self.rmax_m)
            n_mesh = multi_profile_n_mesh(
                segments, float(source.frequencies[0]),
                pinned_n_mesh=self.n_mesh)
            profiles = [seg_env for _range_m, seg_env in segments]
            media = multi_profile_media(segments)
        else:
            # A pinned n_mesh is checked at the deck's freq0 (see
            # :func:`_grid.resolve_modes_launch`).
            reject_coarse_at_mesh('Kraken', self.n_mesh, env, float(
                np.atleast_1d(np.asarray(source.frequencies, dtype=float))[0]))
            rmax_m, rmax_origin = resolve_rmax_m(
                receiver, band=marched_frequencies is not None and np.size(marched_frequencies) > 1,
                pinned_rmax_m=self.rmax_m)
            n_mesh = int(self.n_mesh)
            profiles = [env]
            media = at_env_media(env)
        c_low, c_highs, c_low_origin, c_high_origin = \
            phase_speed_window(
                env, profiles, backend=backend,
                coupled=(segments is not None
                         and self.mode_coupling == 'coupled'),
                         collapse=self._collapse, leaky_modes=self.leaky_modes,
                         log=self._log, pinned_c_high=self.c_high,
                         pinned_c_low=self.c_low)
        freq0 = float(np.atleast_1d(
            np.asarray(source.frequencies, dtype=float))[0])
        launch = KrakenLaunch(
            deck_frequency=freq0,
            marched_frequencies=None if marched_frequencies is None else np.asarray(marched_frequencies, dtype=float),
            tabulation_depths=mode_depths,
            profile_ranges_m=profile_ranges_m,
            c_low=c_low,
            c_high=c_highs,
            rmax_m=rmax_m,
            n_mesh=n_mesh,
            field_option=build_field_option(
                env.is_range_dependent, source, run_mode,
                mode_coupling=self.mode_coupling),
            check_n_mesh=mode_count_check_mesh(
                env, media, n_mesh, freq0,
                backend=backend, c_high_origin=c_high_origin),
        )
        return launch, c_low_origin, c_high_origin, rmax_origin, ppm

    # ── stage 4: decks, launches, outputs ─────────────────────────────────

    def _n_launches(self, settings) -> int:
        """One launch per modes solve the settings hold: one for MODES, a
        narrowband field and a band on one deck (plus the cutoff search when
        a bin is below the modal cutoff, :meth:`_launch`); one per bin of a
        range-dependent band, each reusing the fixed file names the last one
        wrote after it was read."""
        return len(settings.engine.launches)

    def _announce_band_cost(self, per_bin: float, n_bins: int) -> None:
        """Log the cost of a range-dependent band, projected from its first
        bin.

        Projected from the FIRST bin actually solved, rather than from a
        constant. Per-bin cost runs from ~0.03 s on a 3-profile puddle to
        ~47 s on a 21-profile 100 km section — a factor of 1500 — so any
        hard-coded rate is wrong by orders of magnitude on one of them, and
        wrong in the direction that strands the caller. A LOWER bound by
        construction: the band is ascending and per-bin cost climbs with
        frequency, so the cheapest bin sets the estimate. Better to say "at
        least" than to time the whole band before saying anything.
        """
        total = per_bin * n_bins
        log_message(
            "Kraken",
            f"range-dependent broadband: the multi-profile deck "
            f"carries one frequency, so this runs {n_bins} "
            f"separate mode solves and stacks them. The first took "
            f"{per_bin:.2f} s, so expect AT LEAST about {total:.0f} "
            f"s — the first bin is the cheapest, since mode count "
            f"and mesh density both climb with frequency (roughly "
            f"f^2, so a 2:1 band runs about 1.8x this). Cost "
            f"is linear in the band and grows with the number of "
            f"profiles; reduce the frequency count if that is too "
            f"slow, or use a range-independent environment for the "
            f"native broadband deck.",
            verbose=self.verbose,
            # Loud only when the projection is worth interrupting for.
            # Measured from the run, so the threshold means the same thing
            # on every section.
            level="warning" if total > 10.0 else "info",
        )

    def _recover_sub_cutoff_band(self, inputs, launch, exc) -> None:
        """Launch the band of ``launch`` without its sub-cutoff bins.

        A band on one deck can fail when a bin is below the waveguide's
        modal cutoff: kraken writes an empty mode record there, which
        corrupts the multi-frequency ``.mod`` (field.exe then crashes in
        ``ReadModes.f90`` or returns a frequency axis that does not match the
        request). A normal-mode model has no answer below cutoff (the real
        field there is continuous-spectrum energy, Computational Ocean
        Acoustics §2, which needs a wavenumber-integration model like
        Scooter), so the propagating sub-band is solved instead, into the
        same files; :meth:`_read_output` finds it from the ``.shd``'s
        frequency axis and the dropped bins come back NaN — the narrowband
        path's no-data value — with their count in
        :attr:`~uacpy.core.results.Field.sub_cutoff_bins`. The cutoff is
        probed (cheap
        single-frequency mode counts) only on this failure path, and the
        sub-band's deck is resolved by the same resolver as every other
        (:meth:`_sub_band_launch`). Re-raises ``exc`` when every bin
        propagates, since the failure was then something else."""
        frequencies = np.asarray(launch.marched_frequencies, dtype=float)
        floor = self._propagating_frequency_floor(inputs, frequencies)
        if floor == 0:
            raise exc
        if floor >= len(frequencies):
            raise ConfigurationError(
                f"{self.model_name}: no propagating modes at any requested "
                f"frequency ({frequencies[0]:.1f}-{frequencies[-1]:.1f} Hz) — "
                f"every frequency is below the waveguide's modal cutoff.",
                remediation="Raise the frequency band above the modal "
                "cutoff, or use a wavenumber-integration model (Scooter) "
                "for below-cutoff fields.",
            ) from exc
        warnings.warn(
            f"{self.model_name}: {floor} broadband frequency(ies) <= "
            f"{frequencies[floor - 1]:.2f} Hz are below the modal cutoff (no "
            f"propagating modes), so a normal-mode model has no field "
            f"there (use Scooter for the continuous spectrum). Computing "
            f"the {len(frequencies) - floor} propagating frequencies; the "
            f"rest are NaN in H(f) and zero in a TIME_SERIES synthesis.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
        sub_launch = self._sub_band_launch(inputs, frequencies, floor)
        deck = write_decks(inputs, sub_launch, interp_ssp=self.interp_ssp,
                           log=self._log)
        self._run_launch(inputs, sub_launch, deck)

    def _sub_band_launch(self, inputs, band, floor: int) -> KrakenLaunch:
        """The launch of ``band`` without its first ``floor`` bins: a band
        deck, or — for one remaining bin — a narrowband deck at it."""
        propagating = np.asarray(band, dtype=float)[floor:]
        source = (inputs.source if propagating.size > 1 else
                  dataclasses.replace(inputs.source, frequencies=np.array(
                      [float(propagating[0])])))
        sub_launch, *_origins = self._resolve_field_launch(
            inputs.env, source, evaluated_receiver(inputs),
            marched_frequencies=propagating if propagating.size > 1 else None,
            run_mode=RunMode.COHERENT_TL,
            backend=inputs.settings.engine.backend, notices=[])
        return sub_launch

    def _count_modes_at_freq(self, inputs, freq) -> int:
        """Number of trapped modes Kraken finds at a single ``freq`` (Hz),
        solved in the run's work directory under its own root.

        ``0`` means the frequency is below the waveguide's modal cutoff — no
        propagating modes (Computational Ocean Acoustics; Brekhovskikh &
        Lysanov). Uses a single-frequency modes run: a multi-frequency ``.mod``
        that contains a zero-mode record is itself unreadable (the record stride
        breaks at ``M=0``), so the cutoff must be probed one frequency at a
        time. The deck is resolved as a MODES deck on the evaluated receiver
        depths (:func:`_grid.resolve_modes_launch`).

        Only the *below-cutoff* signatures are read as zero: an unreadable or
        absent ``.mod`` after a clean binary exit, and the binary's own empty-
        spectrum ERROUT. Anything else — a missing binary, a crash, a full
        disk — propagates, because the caller turns a zero here into "every
        frequency is below the waveguide's modal cutoff" with the remediation
        "raise the frequency band", which would bury the real cause.
        """
        work_dir = inputs.work_dir
        receiver = evaluated_receiver(inputs)
        source = Source(depths=inputs.source.depths, frequencies=float(freq))
        backend = inputs.settings.engine.backend
        try:
            launch, *_origins = resolve_modes_launch(
                inputs.env, source, receiver,
                tabulation_depths=receiver.depths, backend=backend,
                collapse=self._collapse, leaky_modes=self.leaky_modes,
                log=self._log, pinned_c_high=self.c_high,
                pinned_c_low=self.c_low, pinned_n_mesh=self.n_mesh,
                pinned_rmax_m=self.rmax_m)
            write_modes_deck(work_dir / f'{_PROBE_BASE}.env',
                             inputs.env, source, receiver, launch,
                             interp_ssp=self.interp_ssp)
            self._run_kraken_executable(_PROBE_BASE, work_dir,
                                        exe=self._modes_exe(backend),
                                        launch=launch)
            return read_modes(str(work_dir / _PROBE_BASE),
                              frequency=float(freq)).n_modes
        except (FileFormatError, IndexError, FileNotFoundError) as e:
            # The binary ran; the .mod it left is a zero-mode record the
            # reader cannot stride over (or was never written) — the physical
            # answer is "no modes here".
            self._log(f"_count_modes_at_freq({float(freq):g} Hz): unreadable "
                      f"mode file ({type(e).__name__}: {e}); reading it as "
                      f"below cutoff.", level="debug")
            return 0
        except ModelExecutionError as e:
            # ``_raise_on_fortran_fatal`` already lets the empty-spectrum
            # ERROUT through as a physical outcome; this covers the paths that
            # surface the same banner as an error instead.
            if not any(m in str(e) for m in self._traits.benign_fortran_fatals):
                raise
            self._log(f"_count_modes_at_freq({float(freq):g} Hz): empty "
                      f"spectrum reported; reading it as below cutoff.",
                      level="debug")
            return 0

    def _propagating_frequency_floor(self, inputs, freqs):
        """Index of the first frequency in (sorted, ascending) ``freqs`` that
        has propagating modes — i.e. the end of the contiguous sub-cutoff band.

        Mode count rises monotonically with frequency, so the zero-mode
        frequencies are a prefix ``[0, floor)`` and ``floor`` can be found by
        binary search in O(log N) single-frequency probes (bounded — not one
        probe per frequency). Returns ``len(freqs)`` if none propagate."""
        freqs = np.atleast_1d(freqs)
        lo, hi = 0, len(freqs)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._count_modes_at_freq(inputs, float(freqs[mid])) > 0:
                hi = mid          # modes here → cutoff is at or below mid
            else:
                lo = mid + 1      # no modes here → cutoff is above mid
        return lo

    def _modes_exe(self, backend: str) -> Path:
        """The modes binary of ``backend``: the resolved ``kraken.exe``, or
        ``krakenc.exe`` looked up on first use."""
        if backend == 'krakenc':
            return self._find_executable_in_paths(
                'krakenc.exe',
                bin_subdirs='oalib',
                dev_subdir='Acoustics-Toolbox/Kraken',
            )
        return self._exe

    def _write_input(self, inputs) -> Path:
        """Stage 4: the decks of launch ``inputs.launch``, written by the io
        writers from the resolved settings (:func:`_launch.write_decks`)."""
        return write_decks(inputs, launch_of(inputs),
                           interp_ssp=self.interp_ssp, log=self._log)

    def _launch(self, inputs, deck: Path) -> None:
        """Stage 4: launch ``inputs.launch`` as its route does
        (:data:`_ROUTE_STEPS`)."""
        _ROUTE_STEPS[inputs.settings.engine.route].launch(self, inputs, deck)

    def _launch_as_resolved(self, inputs, deck: Path) -> None:
        """Route step (modes, field, band_bin): the launch as stage 3
        resolved it (:meth:`_run_launch`)."""
        self._run_launch(inputs, launch_of(inputs), deck)

    def _launch_band_deck(self, inputs, deck: Path) -> None:
        """Route step (band): the one deck carrying the whole band, held to
        its frequency axis, and recovered without its sub-cutoff bins when a
        bin fails (:meth:`_recover_sub_cutoff_band`)."""
        launch = launch_of(inputs)
        try:
            self._run_launch(inputs, launch, deck)
            require_band_axis(inputs.work_dir, deck.stem, launch.marched_frequencies,
                              model_name=self.model_name)
        except ModelExecutionError as exc:
            self._recover_sub_cutoff_band(inputs, launch, exc)

    def _launch_band_bin(self, inputs, deck: Path) -> None:
        """Route step (band_by_frequency): one bin of a range-dependent
        band; the first bin announces the band's cost."""
        started = time.perf_counter()
        self._run_launch(inputs, launch_of(inputs), deck)
        if inputs.launch == 0:
            self._announce_band_cost(time.perf_counter() - started,
                                     len(inputs.settings.engine.launches))

    def _run_launch(self, inputs, launch: KrakenLaunch, deck: Path) -> None:
        """The modes binary on the deck of ``launch``, then — on a field
        run — field.exe on the ``.mod`` it wrote.

        Between the two, a narrowband single-profile run reads the
        eigenvalues back: a run below the waveguide's modal cutoff produces
        a complete ``.shd`` that looks like any other
        (:func:`_modes.check_field_modes_trapped`). A band ``.mod`` holds
        one block per frequency (only some of which may be sub-cutoff) and a
        segmented one holds one profile per range, and neither is a single
        statement about the run."""
        exe = self._modes_exe(inputs.settings.engine.backend)
        base = deck.stem
        self._log(f"Running {exe.name}"
                  + ("..." if launch.field_option is not None
                     else " (modes)..."))
        self._run_kraken_executable(base, inputs.work_dir, exe=exe,
                                    launch=launch)
        if launch.check_n_mesh is not None:
            self._check_mode_count(inputs, launch, base, exe)
        if launch.field_option is None:
            return
        if launch.profile_ranges_m is not None:
            refuse_a_profile_without_modes(
                inputs.work_dir / base, inputs.env, launch.profile_ranges_m,
                exe, model_name=self.model_name)
        if launch.marched_frequencies is None and launch.profile_ranges_m is None:
            check_field_modes_trapped(
                inputs.work_dir / base, inputs.env,
                deck_source(inputs, launch), exe, leaky_modes=self.leaky_modes,
                log=self._log, model_name=self.model_name)
        coupled = (launch.profile_ranges_m is not None
                   and launch.field_option[1] == 'C')
        if coupled:
            adiabatic_shd = self._run_adiabatic_pass(inputs.work_dir, base,
                                                     launch.field_option)
        self._run_field_exe(inputs.work_dir, base, launch.field_option)
        if coupled:
            self._warn_on_coupled_gap(inputs.work_dir / f'{base}.shd',
                                      adiabatic_shd)

    def _run_adiabatic_pass(self, work_dir: Path, base: str, option: str
                            ) -> Path:
        """field.exe's adiabatic sum of the coupled launch's own ``.mod``
        (option position 2 'A', ``field.f90:125-136``), under
        :data:`_ADIABATIC_BASE`. Run before the coupled pass so the
        ``field.prt`` left in the work directory is the coupled run's."""
        adiabatic_option = option[0] + 'A' + option[2:]
        flp = (work_dir / f'{base}.flp').read_text()
        (work_dir / f'{_ADIABATIC_BASE}.flp').write_text(
            flp.replace(f"'{option}'", f"'{adiabatic_option}'", 1))
        for suffix in ('.mod', '.sbp'):
            source = work_dir / f'{base}{suffix}'
            link = work_dir / f'{_ADIABATIC_BASE}{suffix}'
            if source.exists() and not link.exists():
                link.symlink_to(source.name)
        return self._run_field_exe(work_dir, _ADIABATIC_BASE, adiabatic_option)

    def _warn_on_coupled_gap(self, coupled_shd: Path, adiabatic_shd: Path
                             ) -> None:
        """Warn when the coupled field departs from the adiabatic sum of the
        same modes by more than :data:`_COUPLED_GAP_WARN_DB` (see there for
        the measurements)."""
        gap = far_field_gap_dB(coupled_shd, adiabatic_shd)
        self._log(f"coupled - adiabatic far-field gap: {gap:+.1f} dB",
                  level="debug")
        if abs(gap) > _COUPLED_GAP_WARN_DB:
            warnings.warn(
                f"{self.model_name}: the coupled-mode field is {abs(gap):.1f} dB "
                f"{'louder' if gap > 0 else 'quieter'} than the adiabatic sum "
                f"of the same modes beyond half the longest range. field.exe's "
                f"coupled projection is not energy-conserving on modes above "
                f"or just below the half-space speed and, over many short "
                f"segments, amplifies or annihilates them (PE Workshop 4c at "
                f"the automatic decomposition: +27.5 dB, the shelf 22 dB above "
                f"the published reference). Use mode_coupling='adiabatic', or "
                f"compare against RAM.",
                NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)

    def _check_mode_count(self, inputs, launch: KrakenLaunch, base: str,
                          exe: Path) -> None:
        """Solve the launch's modes again at the coarser
        ``launch.check_n_mesh`` and warn when a profile or frequency of the
        launch kept fewer modes than that solve did — a mode count that dropped
        as the mesh was refined (see :func:`_grid.mode_count_check_mesh`).

        Measured over five elastic problems (hard, medium and soft
        half-spaces, an elastic layer, an ice canopy over a fluid seabed),
        50-400 Hz and n_mesh 0-4000, against Scooter: it warned on all 12
        runs more than 0.5 dB worse than the problem's best mesh (0.7-5.1 dB
        median) and on 3 whose field had not moved (a lost mode that carries
        nothing there), and never on the automatic mesh. A warning, not a
        refusal, for those 3. A count the coarser solve lost is logged."""
        work_dir = inputs.work_dir
        check = dataclasses.replace(launch, n_mesh=launch.check_n_mesh,
                                    check_n_mesh=None)
        write_modes_deck(work_dir / f'{_CHECK_BASE}.env', inputs.env,
                         deck_source(inputs, launch),
                         inputs.receiver, check,
                         interp_ssp=self.interp_ssp)
        # The check solve's own .prt warnings describe the check, not the
        # run, whose .prt the launch already reported; only the mode counts
        # below speak for the run.
        self._run_kraken_executable(_CHECK_BASE, work_dir, exe=exe,
                                    launch=check, report_prt_warnings=False)
        frequencies = (np.atleast_1d(launch.marched_frequencies) if launch.marched_frequencies is not None
                       else [launch.deck_frequency])
        short = []
        for profile in range(1, launch.n_profiles + 1):
            for frequency in frequencies:
                try:
                    kept, checked = [
                        read_modes(str(work_dir / root),
                                   frequency=float(frequency),
                                   profile=profile).n_modes
                        for root in (base, _CHECK_BASE)]
                except (FileFormatError, IndexError):
                    # An empty mode record breaks the reader's stride; the
                    # run itself reports that case.
                    continue
                where = (f"{float(frequency):g} Hz" if launch.n_profiles == 1
                         else f"profile {profile}, {float(frequency):g} Hz")
                if kept > checked:
                    self._log(f"mode-count check ({where}): the coarser solve "
                              f"at n_mesh = {check.n_mesh} kept {checked} of "
                              f"the run's {kept} modes; the run's set is "
                              f"kept.")
                elif kept < checked:
                    short.append(f"{where}: {kept} vs {checked}")
        if not short:
            return
        warnings.warn(
            f"{self.model_name}: KRAKENC's mode count dropped as its mesh was "
            f"refined — the run's solve (n_mesh = {launch.n_mesh or 'auto'}) "
            f"kept fewer modes than a coarser solve at n_mesh = "
            f"{launch.check_n_mesh} ({'; '.join(short[:4])}"
            f"{'; ...' if len(short) > 4 else ''}). A modal sum missing a "
            f"mode can be several dB off (2.9 dB median, 7.4 dB "
            f"range-averaged near the bed, against Scooter on a hard elastic "
            f"seabed). Compare with another n_mesh, or with Scooter or OAST, "
            f"which integrate over wavenumber with no mode search.",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )

    def _read_output(self, inputs, deck: Path):
        """Stage 4: what the launch wrote, as the io readers return it, read
        as its route does (:data:`_ROUTE_STEPS`)."""
        return _ROUTE_STEPS[inputs.settings.engine.route].read(
            self, inputs, deck)

    def _read_mode_set(self, inputs, deck: Path):
        """Route step (modes): the ``.mod`` as
        :func:`_modes.read_modes_file` reads it (a run with no mode, or with
        only non-trapped ones, raises), with the group speeds krakenc prints
        in its ``.prt``."""
        work_dir = inputs.work_dir
        base = deck.stem
        exe = self._modes_exe(inputs.settings.engine.backend)
        modes_file = work_dir / base
        self._log(f"Reading mode file: {modes_file}.mod")
        modes = read_modes_file(modes_file, model_name=self.model_name)
        # Before any n_modes cap: the reader returns the modes in order of
        # decreasing wavenumber, i.e. increasing phase speed, so if the
        # full set is entirely non-trapped every prefix of it is too.
        check_non_trapped_modes(
            modes.k, inputs.env, float(inputs.source.frequencies[0]),
            exe=exe, leaky_modes=self.leaky_modes, log=self._log,
            model_name=self.model_name)
        # Read before the work directory is let go. Only krakenc fills
        # this column; kraken prints it unassigned.
        vg = read_group_speeds_from_prt(
            work_dir, base, modes.n_modes,
            fills_vg=(exe.stem.lower() == 'krakenc'))
        return {'modes': modes, 'group_velocity': vg}

    def _read_field(self, inputs, deck: Path):
        """Route step (field, band_by_frequency): the narrowband ``.shd`` as
        :func:`~uacpy.io.oalib_reader.read_shd_file` reads it."""
        return read_shd(inputs.work_dir / f'{deck.stem}.shd',
                        launch_of(inputs), one_bin=False)

    def _read_one_bin(self, inputs, deck: Path):
        """Route step (band_bin): a band of one bin, the ``.shd`` as
        :func:`~uacpy.io.oalib_reader.read_shd_bin` reads it."""
        return read_shd(inputs.work_dir / f'{deck.stem}.shd',
                        launch_of(inputs), one_bin=True)

    def _read_band_deck(self, inputs, deck: Path):
        """Route step (band): the ``.shd`` of the one-deck band as
        :func:`~uacpy.io.oalib_reader.read_shd_bin` reads it. A ``.shd``
        carrying fewer frequencies than asked is the propagating sub-band
        :meth:`_launch_band_deck` recovered: the raw output then says how
        many bins were dropped and holds the sub-band's launch
        (:meth:`_sub_band_launch`, the same resolution)."""
        launch = launch_of(inputs)
        shd_file = inputs.work_dir / f'{deck.stem}.shd'
        freqs_read = np.asarray(read_shd_bin(str(shd_file)).frequencies,
                                dtype=float)
        floor = int(np.size(launch.marched_frequencies)) - int(freqs_read.size)
        written = (launch if floor == 0 else
                   self._sub_band_launch(inputs, launch.marched_frequencies, floor))
        raw = read_shd(shd_file, written, one_bin=True)
        if floor:
            raw['sub_cutoff_bins'] = floor
            raw['sub_band'] = written
        return raw

    # ── stage 5: the result ───────────────────────────────────────────────

    def _to_result(self, inputs, deck: Path, raw):
        """Stage 5: the result of the run from the list of launch outputs
        ``raw`` — the :class:`Modes` of a MODES run, or the field in the
        package's conventions (``phase_corr``, ``ρ(z_s)`` and the
        line-source level of :func:`_extract.assemble_field_from_shd`), NaN at
        receiver depths in an elastic sub-bottom, and for ``TIME_SERIES`` its
        synthesis with the request's pulse, made as the run's route makes
        it (:data:`_ROUTE_STEPS`). ``deck`` and ``raw`` are the lists of
        every launch's when the run had several."""
        if not isinstance(deck, list):
            deck, raw = [deck], [raw]
        return _ROUTE_STEPS[inputs.settings.engine.route].result(
            self, inputs, deck, raw)

    def _mode_set_result(self, inputs, deck, raw):
        """Route step (modes): the :class:`Modes` of the run."""
        engine = inputs.settings.engine
        exe = self._modes_exe(engine.backend)
        modes = raw[0]['modes']
        mask_elastic_mode_depths(modes, inputs.env,
                                 model_name=self.model_name)
        result = build_modes_field(
            modes, engine.n_modes, inputs.source, backend_exe=exe,
            group_velocity=raw[0]['group_velocity'],
            water_density=inputs.env.water_density,
            leaky_modes=self.leaky_modes, model_exe=self._exe,
            model_name=self.model_name, result_kwargs=self._result_kwargs
        )
        self._attach_output_paths(
            result, inputs.work_dir, deck[0].stem,
            primary_files=(('mod_file', '.mod'),),
        )
        return result

    @staticmethod
    def _receiver_keep(engine):
        """The receiver depths the deck computed (``None``: all of them)."""
        return (None if engine.receiver_keep is None
                else np.array(engine.receiver_keep, dtype=bool))

    def _field_result(self, inputs, deck, raw):
        """Route step (field): one narrowband field (a stack over a
        multi-depth source), NaN at receiver depths in an elastic
        sub-bottom."""
        settings = inputs.settings
        engine = settings.engine
        exe = self._modes_exe(engine.backend)
        field = self._field_of_launch(
            inputs, engine.launches[0], raw[0], exe,
            run_mode=settings.mode)
        return reinsert_nan_depths(field, inputs.receiver,
                                   self._receiver_keep(engine))

    def _band_deck_result(self, inputs, deck, raw):
        """Route step (band): the band on one deck, its sub-cutoff bins
        restored when the launch recovered the propagating sub-band."""
        engine = inputs.settings.engine
        exe = self._modes_exe(engine.backend)
        frequencies = np.asarray(inputs.settings.frequencies, dtype=float)
        if 'sub_cutoff_bins' in raw[0]:
            tf = self._band_with_sub_cutoff_bins(inputs, raw[0], exe,
                                                 frequencies)
        else:
            tf = self._band_field(inputs, engine.launches[0], raw[0], exe,
                                  frequencies)
        return self._finish_band(inputs, tf)

    def _band_bin_result(self, inputs, deck, raw):
        """Route step (band_bin): a band of one bin."""
        engine = inputs.settings.engine
        tf = self._band_field(
            inputs, engine.launches[0], raw[0],
            self._modes_exe(engine.backend),
            np.asarray(inputs.settings.frequencies, dtype=float))
        return self._finish_band(inputs, tf)

    def _band_by_frequency_result(self, inputs, deck, raw):
        """Route step (band_by_frequency): the band stacked from one
        narrowband launch per bin."""
        tf = self._stack_band_by_frequency(
            inputs, raw, self._modes_exe(inputs.settings.engine.backend),
            np.asarray(inputs.settings.frequencies, dtype=float))
        return self._finish_band(inputs, tf)

    def _finish_band(self, inputs, tf):
        """The transfer function of a band route as the run returns it: NaN
        at receiver depths in an elastic sub-bottom, sub-cutoff bins zeroed
        for a TIME_SERIES synthesis, then the base's broadband finish."""
        settings = inputs.settings
        tf = reinsert_nan_depths(tf, inputs.receiver,
                                 self._receiver_keep(settings.engine))
        n_cut = int(tf.sub_cutoff_bins or 0)
        if settings.mode == RunMode.TIME_SERIES and n_cut:
            # Synthesis needs finite bins: a sub-cutoff bin contributes
            # nothing to the modal pulse, which is field.exe's own zero-mode
            # answer (EvaluateMod.f90, M <= 0 → P = 0).
            data = np.array(tf.data, copy=True)
            data[:, :, :n_cut] = 0.0
            tf.data = data
        return self._finish_broadband(tf, settings)

    def _band_field(self, inputs, launch, raw, exe, frequencies):
        """``H(f)`` of one band launch: the native ``(n_d, n_r, n_f)``
        field, or — for a band of one bin — the narrowband field lifted onto
        a length-1 frequency axis."""
        field = self._field_of_launch(inputs, launch, raw, exe,
                                      run_mode=RunMode.COHERENT_TL)
        if 'frequencies' in raw:
            return field
        d = field.to_dict()
        d['data'] = np.asarray(d['data'])[:, :, None]
        d['coords'] = {**d['coords'], 'frequency': frequencies}
        d['frequencies'] = frequencies
        d['metadata'] = {**d['metadata'], 'native_broadband': False}
        return Field.from_dict(d)

    def _band_with_sub_cutoff_bins(self, inputs, raw, exe, frequencies):
        """``H(f)`` of a band whose lowest bins are below the modal cutoff:
        the propagating sub-band :meth:`_run_launch` solved, on the full
        grid, with the dropped bins NaN and their count in
        :attr:`~uacpy.core.results.Field.sub_cutoff_bins`."""
        floor = raw['sub_cutoff_bins']
        tf = self._band_field(inputs, raw['sub_band'], raw, exe,
                              frequencies[floor:])
        receiver = evaluated_receiver(inputs)
        data_full = np.full(
            tf.data.shape[:2] + (len(frequencies),), np.nan,
            dtype=tf.data.dtype)
        data_full[:, :, floor:] = tf.data        # sub-cutoff bins stay NaN
        return tf.replace(
            data=data_full,
            coords={'depth': receiver.depths, 'range': receiver.ranges,
                    'frequency': frequencies},
            pinned=None, aux_coords=None, frequencies=frequencies,
            sub_cutoff_bins=int(floor),
        )

    def _stack_band_by_frequency(self, inputs, raw, exe, frequencies):
        """``H(f)`` of a range-dependent band: the narrowband field of every
        bin's launch, stacked on a trailing frequency axis.

        The stacked result carries what a NATIVE broadband field carries,
        not what one narrowband slab happens to have. Two things travel
        only on the broadband path and are stamped here: the frequency
        vector (without it the field denies being broadband) and the phase
        reference (without it a complex-weighted sum is refused as "no phase
        reference"). Its identity is the native band's: the modes binary
        as ``backend``, the model source, the source depths and level
        (:meth:`_result_kwargs`). The speed a time-series window anchors on
        is the run's ``run_settings.waveguide``."""
        slabs, first = [], None
        for i, raw_i in enumerate(raw):
            field = self._field_of_launch(
                inputs, inputs.settings.engine.launches[i], raw_i, exe,
                run_mode=RunMode.COHERENT_TL)
            first = first if first is not None else field
            slabs.append(np.asarray(field.data))
        metadata = dict(getattr(first, 'metadata', {}) or {})
        metadata['native_broadband'] = False
        metadata['broadband_assembly'] = ('per-frequency loop '
                                          '(range-dependent deck)')
        return Field(
            data=np.stack(slabs, axis=-1),
            coords={'depth': np.asarray(first.coords['depth'], dtype=float),
                    'range': np.asarray(first.coords['range'], dtype=float),
                    'frequency': frequencies},
            **self._result_kwargs(
                inputs.source, backend=first.backend, frequencies=frequencies,
                phase_reference=getattr(first, 'phase_reference', None),
                **metadata),
        )

    def _field_of_launch(self, inputs, launch, raw, exe, *, run_mode):
        """The field of one field launch in the package's conventions
        (:func:`_extract.assemble_field_from_shd`), stamped with the modes
        binary that ran, the r = 0 mask and the output paths; an empty modal
        sum warns."""
        source = deck_source(inputs, launch)
        receiver = evaluated_receiver(inputs)
        env = inputs.env
        is_rd = launch.profile_ranges_m is not None
        field = assemble_field_from_shd(
            raw, source, receiver, is_rd, launch.n_profiles, run_mode,
            line_c_source=([_source_sound_speed(env, source.at_depth(i))
                            for i in range(np.atleast_1d(
                                source.depths).size)]
                           if source.source_type == 'line' else None),
            source_rho=[_source_density(env, z) for z in
                        np.atleast_1d(np.asarray(source.depths, float))],
                        backend=Path(exe).stem,
                        mode_coupling=self.mode_coupling,
                        result_kwargs=self._result_kwargs,
                        stamp_result=self._stamp_result)

        # No-propagation guard: an empty modal sum — Kraken found 0
        # trapped modes (frequency below the waveguide's modal cutoff, or
        # c_high too low) — leaves field.exe's grid untouched, which the
        # SHD reader surfaces as all-NaN (no-data). Flag it rather than
        # return a silent empty field (compute_modes raises on the same
        # case). Tested on TL so it covers both the complex branches and the
        # real dB INCOHERENT_TL one: an empty sum saturates at the
        # PRESSURE_FLOOR clamp.
        tl = np.asarray(field.dB, dtype=float)
        finite = np.isfinite(tl)
        if not np.any(tl[finite] < NO_ENERGY_DB):
            warnings.warn(
                f"{self.model_name}: no propagating field — the modal sum "
                f"is empty (0 trapped modes: the frequency is below the "
                f"waveguide's modal cutoff, or c_high is too low). The "
                f"returned field is all-NaN (no data), not a physical "
                f"result; raise the frequency or c_high.",
                ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )

        # After the guard above: the guard reads the whole TL grid, and a
        # single-range r=0 request would otherwise look like an empty
        # modal sum.
        if isinstance(field, ResultStack):
            # One launch, one r = 0 notice — not one per slab.
            field.slabs = [self._mask_source_axis(slab, source,
                                                  warn=(i == 0))
                           for i, slab in enumerate(field.slabs)]
        else:
            field = self._mask_source_axis(field, source)

        for slab in _slabs_of(field):
            self._attach_output_paths(
                slab, inputs.work_dir, _FIELD_BASE,
                primary_files=(
                    ('shd_file', '.shd'),
                    ('mod_file', '.mod'),
                ),
            )
            attach_field_prt_path(slab, inputs.work_dir, cleanup=self.cleanup)

        self._log("Kraken simulation complete")
        return field

    # ── field.exe pipeline ──────────────────────────────────────────────

    #: Estimated ``.mod`` size above which a run is warned about — the same
    #: 2 GiB Scooter and SPARC cap their Green's-function files at.

    def _run_field_exe(self, work_dir, base_name, option):
        """Run field.exe → ``.shd`` and return the output path. field.exe may
        exit non-zero on a successful run (known Fortran teardown bug) — warn
        and read the ``.shd`` anyway (:func:`_launch.tolerate_field_teardown`);
        a
        *missing* ``.shd`` is a real failure and raises
        ``ModelExecutionError`` (:func:`_launch.require_field_shd`).

        Tolerating a non-zero exit is exactly why the missing-``.shd`` test
        has to be trustworthy, so any earlier ``.shd`` in a pinned
        ``work_dir`` is cleared first (the launch's ``stale_outputs``).
        ``field.prt`` goes with it — it is read back as a failure signal, so
        an earlier run's copy must not be mistaken for this one's."""
        self._log(f"Running field.exe (option='{option}')...")
        self._launch_binary(Launch(
            argv=(str(self._resolve_field_executable()), base_name),
            cwd=work_dir,
            stale_outputs=tuple(f'{base_name}{suffix}'
                                for suffix in _KRAKEN_FIELD_OUTPUTS),
            # field.exe logs to its own field.prt (field.f90:44), not to
            # <base_name>.prt, which is the modes run's successful log.
            prt_root=_FIELD_PRT_ROOT,
            tolerate_exit=lambda exc: tolerate_field_teardown(
                exc, work_dir, model_name=self.model_name),
            checks=(
                lambda result: raise_on_field_fatal(
                    work_dir, model_name=self.model_name),
                lambda result: self._warn_on_prt_warnings(
                    work_dir, _FIELD_PRT_ROOT),
                lambda result: require_field_shd(work_dir, base_name,
                                                 model_name=self.model_name),
            )))
        return work_dir / f'{base_name}.shd'


class _RouteSteps(NamedTuple):
    """How a Kraken route (:attr:`KrakenSettings.route`) runs stages 4-5:
    the method launching one of its launches, the one reading a launch's
    output, and the one making the result from every launch's output."""
    launch: Callable
    read: Callable
    result: Callable


#: Every route's steps, the one place a route's launch, read and result sit
#: together; its keys are :data:`_settings._ROUTES`.
_ROUTE_STEPS = MappingProxyType({
    'modes': _RouteSteps(Kraken._launch_as_resolved, Kraken._read_mode_set,
                         Kraken._mode_set_result),
    'field': _RouteSteps(Kraken._launch_as_resolved, Kraken._read_field,
                         Kraken._field_result),
    'band': _RouteSteps(Kraken._launch_band_deck, Kraken._read_band_deck,
                        Kraken._band_deck_result),
    'band_bin': _RouteSteps(Kraken._launch_as_resolved, Kraken._read_one_bin,
                            Kraken._band_bin_result),
    'band_by_frequency': _RouteSteps(Kraken._launch_band_bin,
                                     Kraken._read_field,
                                     Kraken._band_by_frequency_result),
})
