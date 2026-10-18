"""The :class:`Kraken` model: its knobs, traits and run hooks."""

import warnings
import dataclasses
import time
from types import MappingProxyType
import numpy as np
from pathlib import Path
from typing import Callable, Dict, NamedTuple, Optional, Sequence, Union
from uacpy.models.base import (
    DEFAULT_RUN_TIMEOUT_S, PropagationModel, _host_options,
)
from uacpy.models._launch import (
    BinarySearch, Launch, _read_prt_if_present, run_launch, run_prt_binary,
    warn_on_prt_warnings,
)
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.models._conventions import _source_density, _source_sound_speed
from uacpy.models._knobs import is_whole_number, positive_finite, whole_count
from uacpy.models._spec import EngineTraits, ModelSpec
from uacpy.models._stacking import _slabs_of
from uacpy.core.run_settings import Notice, OutputSpec, RunMode
from uacpy._log import log_message
from uacpy.core.boundary import BoundaryProperties
from uacpy.core.surface import Surface
from uacpy.core.results import Field, PhaseReference
from uacpy.core.results._superposition import (restack_source_depths,
                                               split_source_depths)
from uacpy.core.constants import NO_ENERGY_DB
from uacpy.core.exceptions import (
    ConfigurationError, ExecutableNotFoundError, FallbackWarning,
    ModelExecutionError, NumericsWarning, ValidityWarning,
)
from uacpy.io.oalib_writer import (
    reject_unsupported_ssp_interp,
    reject_biological_edges_under_neighbour_interp,
)
from uacpy.models.kraken._settings import KrakenLaunch, KrakenSettings
from uacpy.models.kraken._checks import (
    backend_origin,
    deck_bottom_columns,
    drop_roughness_on_tabulated_top,
    elastic_depth_intervals,
    kraken_exe_above_seabed_notice,
    kraken_exe_elastic_notice,
    krakenc_incoherent_sum_notice,
    mask_elastic_mode_depths,
    modes_single_profile,
    partition_elastic_subbottom,
    reflection_table_boundaries,
    reinsert_nan_depths,
    reject_acoustic_below_elastic,
    reject_rough_elastic_layer,
)
from uacpy.models._checks import (
    reject_malformed_irc_bottom, reject_precalc_surface, total_media_depth,
)
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
    band_mesh_floor_notice,
    band_n_mesh,
    build_field_option,
    check_field_tabulation_size,
    merge_depths,
    mode_file_size_notice,
    dense_mode_depths,
    mode_points_per_meter,
    multi_profile_n_mesh,
    reject_band_over_field_limit,
    resolve_modes_launch,
    resolve_rmax,
    rmax_zero_notice,
    segment_env_for_field,
    single_profile_n_mesh,
)
from uacpy.models.kraken._modes import (
    _NO_MODES_ERROUT,
    _ZERO_MODE_REOPEN,
    below_cutoff_remedy,
    build_modes_field,
    end_track_or_ram,
    log_non_trapped_modes,
    printed_mode_tables,
    profile_location,
    profile_without_modes_refusal,
    read_group_speeds_from_prt,
    read_mode_record,
    record_seabed_speed,
    trapped_among,
    trapped_phase_speeds,
    zero_mode_stop,
)
from uacpy.models.kraken._extract import (
    assemble_field_from_shd,
    deck_source,
    evaluated_receiver,
    launch_of,
    read_shd,
)
from uacpy.models.kraken._launch import (
    _FIELD_BASE,
    _FIELD_PRT_ROOT,
    _KRAKEN_FIELD_OUTPUTS,
    _KRAKEN_MODES_OUTPUTS,
    attach_field_prt_path,
    raise_on_field_fatal,
    require_band_axis,
    require_field_shd,
    tolerate_field_teardown,
    write_decks,
)
from uacpy.models._extract import (
    add_run_note, attach_output_paths, finish_broadband, mask_source_axis,
    result_kwargs,
)


# The modes whose result is a transfer function H(f) (TIME_SERIES then
# synthesises it with the source pulse).
_BAND_MODES = frozenset({RunMode.BROADBAND, RunMode.TIME_SERIES})


#: The binary's own words for a solve that ran through yet kept no mode at
#: a frequency: an empty mode table in its ``.prt``
#: (:func:`_modes.printed_tables`), which krakenc prints when it keeps no
#: root at its last mesh, the ``.mod`` record then holding none.
_EMPTY_TABLE = "an empty mode table in its .prt"


def _no_level(data) -> bool:
    """Whether field.exe's output ``data`` holds no level at any receiver:
    every value zero (its empty modal sum) or NaN."""
    data = np.asarray(data)
    return not np.any(np.abs(data[np.isfinite(data)]) > 0.0)


class _Unsummed(NamedTuple):
    """A launch whose modes solve kept no mode, so field.exe had nothing to
    sum: at ``frequency`` (Hz), in ``profile`` (0-based) — the source's
    (0), or a later one the binary stopped on — in the binary's own words
    (``reason``: the message it stopped on, or :data:`_EMPTY_TABLE`)."""
    frequency: float
    profile: int
    reason: str


class _BandDeck(NamedTuple):
    """What a band's launch ran through (:meth:`Kraken._launch_band_deck`):
    the launch of its last deck, the band's bins it carries (``kept``,
    indices into the band) and the binary's own words for each bin it kept
    no mode at (``dropped``, ``{bin index: reason}``)."""
    launch: KrakenLaunch
    kept: np.ndarray
    dropped: dict


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
      (``compute_field`` / ``compute_transfer_function`` / ``compute_time_series``)
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
    MODES path is range-independent and reduces an RD environment to one
    profile by the collapse methods (the package defaults, else the
    caller's ``collapse=``), with a warning.

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
        ``'adiabatic'`` or ``'coupled'``: how ``field.exe`` handles
        range-dependent mode transitions. ``None`` (default) is
        ``'adiabatic'``, field.exe's own default (``doc/field.htm``: "'A'
        Adiabatic mode theory (default)."). A range-independent run has one
        profile, which field.exe evaluates whatever the option
        (``field.f90:214-215``), and a ``MODES`` run sums no field: a value
        set there is not used, and the run says so.
        ``'coupled'`` marches forward only and projects only the pressure
        onto each new segment's modes (``EvaluateCMMod.f90``), so it does
        not conserve energy and its answer moves with ``n_segments``
        rather than converging. On the shelf of docs/models/kraken.md §6.6
        it sits 7.4 dB above a refined RAM run in the upper 100 m past
        15 km, where the adiabatic run sits 0.15 dB below it; check either
        option against :class:`~uacpy.models.RAM` on a slope.
    n_segments : int, optional
        Range segments for RD scenarios. Default ``None`` lets
        :func:`_segments.segment_environment_by_range` pick segment
        edges from the union of bathymetry / RD-SSP / RD-bottom
        change-points, inserting intermediates wherever the gap
        exceeds :data:`_segments._MAX_SEGMENT_LENGTH_M` (2 km), spans a
        seafloor depth change of more than a quarter of the shortest water
        wavelength, or a sound-speed change of more than ``0.5·c/f`` (at the
        highest frequency); the count is capped at
        :data:`_segments._MAX_AUTO_SEGMENTS` (200), thinned evenly with a
        ``NumericsWarning`` when the criteria ask for more. Pass
        an explicit int (at least 2) to override with a uniform linspace
        decomposition.
    mode_points_per_meter : float, optional
        Mode-depth grid density in pts/m. ``None`` (default) derives it from
        the run as ``10 * freq_max / c_min`` — the ~10 points/wavelength the
        KRAKEN and FIELD manuals require of the mode tabulation grid — with a
        1.5 pts/m floor. An explicit value (positive) is used verbatim and
        warns if it falls under that.
    backend : str, optional
        Force the modes binary: ``'kraken'`` or ``'krakenc'``. ``None``
        (default) auto-selects — ``krakenc.exe`` for elastic media / leaky
        modes (complex eigenvalues) and for a tabulated reflection
        coefficient (``acoustic_type='file'`` / ``'precalc'``, which
        ``kraken.exe`` either refuses or silently replaces with a rigid
        boundary), ``kraken.exe`` otherwise. Forcing ``'kraken'`` on a
        reflection table, or beside ``leaky_modes=True``, raises
        ``ConfigurationError``; on elastic media it runs, with one
        ``ValidityWarning`` stating what kraken.exe computes there: the
        trapped modes only, ``c_high`` lowered to an elastic half-space's
        shear speed (``Kraken/kraken.f90:209``, ``:221``), without the
        elastic media's material loss in ``Im(k)`` (``:728``; ``doc/kraken.htm``:
        "for attenuating elastic media, KRAKENC should be used"). Forcing
        ``'krakenc'`` runs over a
        rigid or vacuum floor too: KRAKENC's boundary condition sets the
        vector it shoots for both floors
        (``Kraken/BCImpedancecMod.f90:58-66``).

        Over a fluid seabed ``None`` runs ``kraken.exe``, whose real
        arithmetic solves the seabed above its compressional speed as rigid
        (``Kraken/BCImpedanceMod.f90:83-88``), where KRAKENC computes leaky
        modes (``doc/kraken.htm`` (7) Phase Speed Limits); the derived window
        reaches above that speed. ``run_settings(...).engine.backend``
        records which program ran. A default run says nothing of it;
        ``'kraken'``, or a ``c_high`` pinned above the seabed speed, gives
        one ``ValidityWarning`` saying so (``docs/models/kraken.md`` §3
        compares the two programs on measured cases).

        KRAKENC's root-finder restart option (``TopOpt(5:5) = '.'``,
        ``krakenc.f90:323-327``: up to ``max(10, N/50)`` restarts) is not
        offered and the deck is written with it off: each restart starts
        from ``RANDOM_NUMBER`` draws (``krakenc.f90:393-395``) and the
        toolbox sets no seed.
    c_low : float, optional
        Lower phase speed limit (m/s). None ⇒ 0.0 when no medium carries
        shear, which makes KRAKEN compute cLow automatically — the
        modal-solver default; when any medium is elastic, None ⇒ the
        minimum compressional speed in the problem (SSP and bottom), which
        keeps the search off the interfacial branch (``_window.c_low_for``).
        A positive c_low skips slower modes and excludes interfacial
        (Scholte / Stoneley) modes; set it to the minimum p-wave speed if KRAKEN
        fails to converge on those. (``default_c_low`` in
        ``core/engine_defaults.py`` is the Scooter/SPARC wavenumber-integration
        default, not Kraken's.) Must be non-negative and
        strictly less than ``c_high``.
    c_high : float, optional
        Upper phase speed limit (m/s). None = auto, per profile of the deck:
        1.05 x the fastest speed among the SSP and the seabed half-space;
        10 x the fastest water speed over a 'file' / 'precalc'
        reflection-table seabed (``_TABLE_BOTTOM_C_HIGH_FACTOR``), and over a
        rigid or vacuum floor on krakenc; unbounded (1e9) over a rigid or
        vacuum floor on kraken.exe; 10 x the fastest speed in the profile
        with ``leaky_modes=True`` (``_LEAKY_C_HIGH_FACTOR``). Must be
        strictly greater than ``c_low``. ``run_settings(...).engine`` shows
        the window and the rule that set it.
    n_mesh : int or sequence of int, optional
        Total number of mesh points PER MEDIUM used by the finite-difference
        mode solver (AT's ``NMESH`` column on each medium line): one count
        for every medium, or one per medium, water first. A count is the
        mesh at the deck's frequency line ``source.frequencies[0]``; a band
        deck solves each frequency ``f`` on that count times ``f / freq0``
        (``Kraken/kraken.f90:75``, ``Kraken/krakenc.f90:77``). 0 lets AT
        size the medium at 20 points per wavelength of the deck's frequency
        line, 10 at least (``misc/ReadEnvironmentMod.f90:101-108``;
        ``doc/EnvironmentalFile.htm``: "If you type 0 for the number of mesh
        points, the code will calculate NMESH automatically"). ``None``
        (default) is 0, except that on a band deck a medium whose own count
        would leave fewer than 10 points at the band's lowest frequency is
        written with the least count that leaves 10 there
        (``run_settings(...).engine.launches``). A set value is written as
        given (a range-dependent run takes one count for every medium); 0 on
        a band deck that leaves a medium under 10 points at its lowest
        frequency gives a ``NumericsWarning``.
        Note: this is NOT a "points per wavelength" density — it is a total
        point count per medium. On a hard elastic seabed the automatic mesh
        is the coarse end of a converging ladder with holes in it — 100 m of
        water over cp 3000 / cs 1400 at 100 Hz, against Scooter: 0.51 dB
        median at 0 (auto), 0.24 at 250, 0.10 at 500, 0.055 at 1500, 0.050
        at 3000, but 2.9 dB at 200, 400, 1000, 2000 and 4000, where KRAKENC
        lost modes refining its mesh (fluid seabeds sit at 0.01-0.07 dB with
        the default). Comparing a result across ``n_mesh`` values is how to
        judge it (``docs/models/kraken.md`` §7).
    interp_ssp : str, optional
        SSP connection scheme written into ``TopOpt(1)``. ``None``
        (default) resolves to ``'linear'`` (C-linear). Explicit values:
        ``'linear'``, ``'n2linear'``, ``'pchip'``,
        ``'spline'``. ``'quad'`` is Bellhop-only and is
        rejected with an :class:`UnsupportedFeatureError`.
    n_modes : int, optional
        Cap on the number of modes field.exe propagates (FLP ``MLimit``).
        ``None`` (default) uses every mode the solver found, as
        ``doc/field.htm`` does with a large count: "If the number of modes
        specified exceeds the number computed then the program uses all the
        computed modes." Also truncates a
        :class:`Modes` result to the first ``n_modes``, matching what the
        field evaluation used; :meth:`Modes.first_n` slices an existing result
        after the fact.
    range_offsets : sequence of float, optional
        Receiver range offsets (m), one per receiver depth, shallowest
        first: field.exe's ``Rro`` (``field.f90:147-152``). ``doc/field.htm``:
        "The field is computed by stepping through the ranges, Rr(1:NR), and
        adding in the range displacements, Rro() before computing the field
        on the array. Nonzero values are used to tilt or distort the
        receiving array, thereby simulating the distortion which occurs on
        an array deployed in the ocean." The receiver at depth ``z_i`` and
        range ``r`` reads the field at ``r + ro_i`` (``EvaluateMod.f90:47``,
        ``:72-73``). ``None`` (default) writes zeros, a vertical array.
        field.exe sorts the offsets ascending, as it sorts the receiver
        depths (``misc/SourceReceiverPositions.f90:224``, ``:267-268``), so
        it pairs the smallest offset with the shallowest depth: offsets that
        decrease with depth are refused. Only the range-independent sum
        reads them (``Evaluate``, ``field.f90:214-215``); a range-dependent
        run (``EvaluateAD`` / ``EvaluateCM``) and :meth:`compute_modes` do
        not, with a notice.
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
        a few modes with ``c_p`` above it. :meth:`compute_modes` returns
        them with the rest and logs how many at ``verbose='info'``; a field
        run whose every summed mode is one of them warns (Computational Ocean
        Acoustics §2.4.5.1). Default: False.

        With the automatic window a field run warns when the receivers need
        paths steeper than it keeps (the near field of a trapped-mode sum,
        measured +11.9 dB at 200 m on a 100 m Pekeris guide); ``True`` is the
        remedy there (within 0.04 dB of Scooter). A rigid or vacuum seabed
        has no half-space for a mode to leak into (``docs/models/kraken.md``
        §7).
    top_reflection_file : Path, optional
        A ``.trc`` top-reflection-coefficient table. Overrides the surface
        boundary condition to ``'F'`` and is staged next to the ``.env``.
    rmax : float, optional
        ``RMax`` written into the ``.env`` (m here; the writer converts to
        the km the deck expects): set it to the farthest range you will
        compute a field at — the Acoustics Toolbox manual's RMax, "the
        largest range for which a field calculation will be desired"
        (``doc/kraken.htm:674-685``). It sets how far the mode solver
        converges the mesh, and moves no output range:
        ``kraken.f90:80`` / ``krakenc.f90:82`` refine the mesh until
        ``Error·1000·RMax < 1``, where ``Error`` is the change in the
        Richardson-extrapolated eigenvalue between two successive meshes
        and ``1000·RMax`` is RMax in metres. A **larger** RMax is a
        **tighter** tolerance (finer mesh, slower, more accurate).
        ``None`` (default) is the farthest receiver range, the manual's
        value; with no positive receiver range it is 100 km
        (``_grid.KRAKEN_RMAX_NO_RANGE_M``, uacpy guidance, measured: the
        manual's 0 left a 1.09 rad mode phase error at 10 km on a Pekeris
        guide). ``compute_modes`` takes no receiver, so it is that case
        unless ``rmax`` is set. A set value is written as given, 0
        included — ``doc/kraken.htm`` (8): "If you set it to zero, then no
        mesh doublings will be performed" — with a ``NumericsWarning`` that
        no mesh doubling follows; a negative one is refused.
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
    - ``n_mesh=None`` → 0, AT sizing each medium at the deck's frequency
      line; a band deck keeps 10 points per medium at its lowest frequency.
    - ``mode_coupling=None`` → ``'adiabatic'``, field.exe's default.
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
    >>> field = kraken.compute_field(env, source, receiver)     # field via field.exe

    >>> modes = kraken.compute_modes(env, source)            # modes only (no receiver)

    >>> # Elastic bottom → complex modes (auto, or force backend='krakenc')
    >>> modes = Kraken(backend='krakenc').compute_modes(env_elastic, source)

    >>> # Range-dependent with coupled modes
    >>> kraken = Kraken(mode_coupling='coupled', n_segments=20)
    >>> field = kraken.compute_field(env_rd, source, receiver)
    """

    def _run_kraken_executable(self, base_name: str, work_dir: Path,
                               exe=None):
        """Execute the modes binary (``exe`` selects kraken.exe vs krakenc.exe;
        defaults to the resolved kraken.exe) via the shared binary-launch
        helper. Returns where the binary stopped on a window
        holding no mode and its own words for it, ``(profile, frequency,
        reason)`` — 0-based indices into the deck and the message it
        stopped on — or ``None`` when it solved every one.

        Both binaries stop at the first such solve: the empty-spectrum
        ERROUT (``kraken.f90:962``, ``krakenc.f90:445``), whose solve the
        ``.prt`` names (:func:`_modes.zero_mode_stop`), or — on the deck's
        first solve only — a gfortran error re-OPENing the mode file they
        hold open (:data:`_modes._ZERO_MODE_REOPEN`) before the ERROUT is
        written. Measured on both binaries next to the cut-off: kraken.exe
        at 12.4875 Hz in 30 m over a rigid seabed (f01 = 12.5 Hz), krakenc
        at 30.035 Hz in 30 m over a lossless 1650 m/s half-space. Any other
        failure raises."""
        try:
            run_prt_binary(
                self._launch_context(), [str(exe or self._exe), base_name],
                work_dir, base_name,
                stale_outputs=_KRAKEN_MODES_OUTPUTS)
        except ModelExecutionError as exc:
            text = f"{exc.stdout or ''}\n{exc.stderr or ''}"
            reopen = [marker for marker in _ZERO_MODE_REOPEN if marker in text]
            if reopen:
                return (0, 0, f"'{reopen[0]}'")
            stop = zero_mode_stop(_read_prt_if_present(work_dir / f'{base_name}.prt'))
            if stop is None:
                raise
            return stop + (f"'{_NO_MODES_ERROUT}'",)
        stop = zero_mode_stop(_read_prt_if_present(work_dir / f'{base_name}.prt'))
        return None if stop is None else stop + (f"'{_NO_MODES_ERROUT}'",)

    #: The record's ``mode_coupling`` is ``'none'`` on a range-independent
    #: run, which is no knob value; a re-run derives it again.
    _UNPINNED_FIELDS = frozenset({'mode_coupling'})

    @property
    def _coupling(self) -> str:
        """``mode_coupling``, ``'adiabatic'`` when unset: field.exe's
        default (``doc/field.htm``: "'A' Adiabatic mode theory
        (default).")."""
        return 'adiabatic' if self.mode_coupling is None else self.mode_coupling

    def _unused_knob_notices(self, env, mode, receiver) -> list:
        """One notice per knob the caller set that this run does not use:
        ``mode_coupling`` when the run sums no range-dependent field (a
        ``MODES`` run, or one profile, which field.exe evaluates with
        ``Evaluate`` whatever the option, ``field.f90:214-215``);
        ``n_segments``, read only to segment a range-dependent field run
        (:func:`_segments.segment_env_for_field`); ``mode_depths``, read
        only by a ``MODES`` run with no receiver
        (:func:`_grid.dense_mode_depths`); ``mode_points_per_meter``, not
        read by a ``MODES`` run whose modes are tabulated on the receiver
        depths or on a pinned ``mode_depths``; ``range_offsets``, read only
        by the one-profile ``Evaluate`` of a field run (``field.f90:214-221``)."""
        modes = mode == RunMode.MODES
        unused = []
        if self._caller_set('mode_coupling') and (
                modes or not env.is_range_dependent):
            unused.append((
                'mode_coupling',
                "a MODES run sums no field" if modes else
                "a range-independent run has one profile, which field.exe "
                "evaluates with Evaluate whatever the option "
                "(field.f90:214-215)"))
        if self._caller_set('n_segments') and (
                modes or not env.is_range_dependent):
            unused.append((
                'n_segments',
                "a MODES run solves one profile" if modes else
                "a range-independent run has one profile to segment"))
        if self._caller_set('mode_depths') and not (
                modes and receiver is None):
            unused.append((
                'mode_depths',
                "a MODES run with a receiver tabulates the modes on the "
                "receiver depths" if modes else
                "a field run tabulates the modes on its own grid, at "
                "mode_points_per_meter, with the receiver depths"))
        if self._caller_set('mode_points_per_meter') and modes and (
                receiver is not None or self.mode_depths is not None):
            unused.append((
                'mode_points_per_meter',
                "a MODES run with a receiver tabulates the modes on the "
                "receiver depths" if receiver is not None else
                "the pinned mode_depths are the tabulation grid"))
        if self._caller_set('range_offsets') and (
                modes or env.is_range_dependent):
            unused.append((
                'range_offsets',
                "a MODES run sums no field" if modes else
                "a range-dependent run is summed by EvaluateAD / "
                "EvaluateCM, which take no offsets; only the one-profile "
                "Evaluate reads Rro (field.f90:214-221)"))
        return [Notice(
            f"{name} not used",
            f"{self.model_name}: {name}={self._knob_text(name)} is not "
            f"used: {why}.", FallbackWarning) for name, why in unused]

    def _knob_text(self, name) -> str:
        """``repr`` of knob ``name``, an array summarised."""
        value = getattr(self, name)
        if isinstance(value, np.ndarray):
            what = 'offsets' if name == 'range_offsets' else 'depths'
            return f"<{value.size} {what}>"
        return repr(value)

    def _caller_set(self, name) -> bool:
        """Whether the caller set knob ``name``: given, and not a value a
        model rebuilt from a run's record took from what that run derived
        (``models/_replay.py``, ``_pinned_from_record``)."""
        return (getattr(self, name) is not None
                and name not in getattr(self, '_pinned_from_record', ()))

    def _chose_backend_or_window(self) -> bool:
        """Whether the caller set ``backend`` or ``c_high``: a run on
        kraken.exe above a fluid seabed says so only then
        (:func:`_checks.kraken_exe_above_seabed_notice`)."""
        return self._caller_set('backend') or self._caller_set('c_high')

    @classmethod
    def _per_launch_knob_values(cls, engine):
        """``c_low``, ``c_high``, ``n_mesh`` and ``rmax`` over the
        launches (``c_high`` one per profile of each; none with
        ``leaky_modes``, which sets it)."""
        launches = engine.launches
        values = {
            'c_low': [launch.c_low for launch in launches],
            'c_high': [value for launch in launches
                       for value in launch.c_high],
            'n_mesh': [launch.n_mesh for launch in launches],
            'rmax': [launch.rmax for launch in launches],
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
    # bottom. The MODES path reduces each range-dependent quantity by the
    # caller's collapse method, else the package default
    # (``_checks.modes_single_profile``).
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
            # all — and the .shd reader puts its NSz axis on the Field's
            # source_depth axis,
            # so the two TL modes take a multi-depth Source in one launch
            # instead of one mode solve per depth (measured 4.3x at eight
            # depths). Two conditions make that one launch exact: field.f90
            # shades every source depth of a .sbp run (a uacpy patch — it
            # shaded only the first; third_party/MODIFICATIONS.md), and the
            # receiver depths join the mode-tabulation grid kraken.f90:573
            # merges Pos%Sz into (``_resolve_field_launch``), so
            # ReadModes.f90:54 interpolates no receiver mode shape and an
            # extra source depth cannot move a receiver's answer. MODES takes
            # one too: the modes do not depend on the source
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

    #: The modes binary of each backend, and ``field.exe``, which every
    #: backend's field run launches to sum the modes (key ``'field'``).
    _BINARIES = MappingProxyType({
        key: BinarySearch((name,), ('oalib',), 'Acoustics-Toolbox/Kraken')
        for key, name in (('kraken', 'kraken.exe'), ('krakenc', 'krakenc.exe'),
                          ('field', 'field.exe'))
    })

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

    def __init__(
        self,
        *,
        executable: Optional[Path] = None,
        field_executable: Optional[Path] = None,
        backend: Optional[str] = None,
        c_low: Optional[float] = None,
        c_high: Optional[float] = None,
        n_mesh: Optional[Union[int, Sequence[int]]] = None,
        n_modes: Optional[int] = None,
        range_offsets: Optional[Sequence[float]] = None,
        interp_ssp: Optional[str] = None,
        leaky_modes: bool = False,
        top_reflection_file: Optional[Path] = None,
        rmax: Optional[float] = None,
        mode_depths: Optional[np.ndarray] = None,
        mode_points_per_meter: Optional[float] = None,
        mode_coupling: Optional[str] = None,
        n_segments: Optional[int] = None,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        cleanup: Optional[bool] = None,
        timeout: float = DEFAULT_RUN_TIMEOUT_S,
        collapse: Optional[Dict[str, str]] = None,
    ):
        super().__init__(**_host_options(locals()))
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
        self.range_offsets = (
            np.atleast_1d(np.asarray(range_offsets, dtype=float))
            if range_offsets is not None else None
        )
        self.leaky_modes = leaky_modes
        self.top_reflection_file = (
            Path(top_reflection_file) if top_reflection_file is not None else None
        )
        # rmax scales the mode solver's mesh-convergence tolerance;
        # None → derive at run() from receiver.range_max.
        self.rmax = rmax
        # mode_depths overrides compute_modes's dense grid; None → density.
        self.mode_depths = (
            np.asarray(mode_depths, dtype=float)
            if mode_depths is not None else None
        )
        self.backend = backend
        # Run modes, capability flags and collapse defaults come from the
        # class-level ``spec`` (applied by PropagationModel.__init__).
        self.mode_coupling = mode_coupling
        self.mode_points_per_meter = mode_points_per_meter
        self.n_segments = n_segments
        self._check_knobs()
        self.n_modes = None if n_modes is None else int(n_modes)
        self.rmax = float(rmax) if rmax is not None else None

        # The resolved kraken.exe path lives in ``self._exe``;
        # a run on krakenc looks krakenc.exe up (``_modes_exe``).
        self._exe = self._resolve_executable(
            executable, lambda: self._find_binary('kraken'),
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

    def _resolve_field_executable(self) -> Path:
        """Resolve ``field.exe`` lazily (only field-producing run modes need
        it). Cached in ``self._field_exe`` after the first call."""
        if self._field_exe is not None:
            return self._field_exe
        if self._field_exe_pinned is not None:
            fx = self._field_exe_pinned
        else:
            fx = self._find_binary('field')
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
        :func:`~uacpy.models._checks.reject_malformed_irc_bottom` checks a
        ``'precalc'`` bottom's ``.irc`` header on the run path.

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
            When ``backend='kraken'`` is forced beside ``leaky_modes=True``,
            or on a tabulated reflection coefficient kraken.exe cannot
            read.
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
        if forced == 'kraken' and getattr(self, 'leaky_modes', False):
            raise ConfigurationError(
                "Kraken(backend='kraken', leaky_modes=True): the two settings "
                "contradict. Leaky modes need complex eigenvalues, which only "
                "krakenc.exe computes (kraken.htm: 'KRAKENC will attempt to "
                "compute leaky modes if CHIGH exceeds the phase velocity of "
                "either the S-wave or P-wave speed in the half-space'); "
                "kraken.exe computes trapped modes only ('KRAKEN will (if "
                "necessary) reduce CHIGH so that only trapped (non-leaky) "
                "modes are computed').",
                remediation="Drop backend='kraken' (leaky_modes=True runs "
                            "krakenc.exe), or drop leaky_modes=True."
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
        """Refuse a constructor knob no run could use: a mode cap that is not
        an integer of at least 1, an ``n_mesh`` that is not an integer of at
        least 0 (0 is KRAKEN's automatic mesh), a non-positive or non-finite
        ``mode_points_per_meter``, an ``n_segments`` that is not an integer
        of at least 2, a negative ``rmax``, ``range_offsets`` that are
        not finite or decrease with depth, ``leaky_modes=True`` beside
        a pinned ``c_high``, a pinned phase-speed pair out of order, an
        unknown ``backend`` or ``mode_coupling``.

        Run at construction and again by every run (the attributes can be
        reassigned in between). A single pinned bound is held to the other
        once a run derives it (:func:`_window.phase_speed_window`).
        """
        n_modes = self.n_modes
        if n_modes is not None and not is_whole_number(n_modes):
            raise ConfigurationError(
                f"Kraken(n_modes={n_modes!r}): the mode cap must be an "
                f"integer.")
        if n_modes is not None and n_modes < 1:
            raise ConfigurationError(
                f"Kraken(n_modes={n_modes}): the mode cap must be >= 1. "
                "field.exe treats MLimit <= 0 as zero propagating modes "
                "(field.f90:68,185), which returns an empty field.")
        # The deck's NMESH column is a point count per medium, 0 handing the
        # mesh to KRAKEN; a fraction was truncated and a negative count
        # reached the binary's "Mesh is too coarse" stop.
        if isinstance(self.n_mesh, (list, tuple, np.ndarray)):
            for count in self.n_mesh:
                whole_count('Kraken: each n_mesh count', count, 0)
        elif self.n_mesh is not None:
            whole_count('Kraken: n_mesh', self.n_mesh, 0)
        positive_finite('Kraken: mode_points_per_meter',
                        self.mode_points_per_meter, optional=True)
        # A range-dependent run needs two profiles
        # (_segments.segment_environment_by_range refuses fewer).
        if self.n_segments is not None:
            whole_count('Kraken: n_segments', self.n_segments, 2)
        # rmax scales the mode solver's mesh-convergence tolerance;
        # None → derive at run() from receiver.range_max. ``kraken.f90:80``
        # leaves the refinement loop as soon as ``Error*1000*RMax < 1`` and
        # ``Error`` starts at 1e10, so RMax <= 0 makes it true on the first,
        # coarsest mesh — no Richardson extrapolation, measured 2.37 dB max
        # |ΔTL| against the default, at exit 0. A set 0 is the manual's
        # documented "no mesh doublings" (``doc/kraken.htm`` (8)) and is
        # written as given with a notice (``rmax_zero_notice``); a negative
        # one is refused below, since no range is negative.
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
        # field.exe's Rro: finite metres, in the order its ReadVector leaves
        # them — sorted ascending, as the receiver depths are
        # (misc/SourceReceiverPositions.f90:224, :267-268), so a vector that
        # decreases with depth would reach the wrong receivers.
        if self.range_offsets is not None:
            offsets = np.asarray(self.range_offsets)
            if (offsets.ndim != 1 or offsets.size < 1
                    or not np.all(np.isfinite(offsets))):
                raise ConfigurationError(
                    f"Kraken(range_offsets={self._knob_text('range_offsets')})"
                    f" must be a 1-D sequence of finite offsets (m), one per "
                    f"receiver depth.")
            if np.any(np.diff(offsets) < 0.0):
                k = int(np.argmax(np.diff(offsets) < 0.0))
                raise ConfigurationError(
                    f"Kraken(range_offsets=...) decreases with depth: offset "
                    f"{k + 2} ({offsets[k + 1]:g} m) is below offset {k + 1} "
                    f"({offsets[k]:g} m). field.exe reads the offsets with "
                    f"ReadVector, which sorts them ascending "
                    f"(misc/SourceReceiverPositions.f90:267-268) as it sorts "
                    f"the receiver depths (:224), so it pairs the smallest "
                    f"offset with the shallowest depth and these offsets "
                    f"would reach other receivers than the ones given.",
                    remediation="Give offsets that do not decrease with "
                                "depth (shallowest receiver first); "
                                "field.exe cannot place a deeper receiver "
                                "at a shorter range than a shallower one.")
        rmax = self.rmax
        if rmax is not None and float(rmax) < 0.0:
            raise ConfigurationError(
                f"Kraken(rmax={rmax}) is negative: RMax is the largest "
                f"range for which a field calculation will be desired "
                f"(doc/kraken.htm (8)), and no range is negative.",
                remediation="Pass the farthest range of the field (m), 0 "
                            "for no mesh doubling, or None for the farthest "
                            "receiver range.")
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
        validate_phase_speed_limits(pinned_c_high=self.c_high,
                                    pinned_c_low=self.c_low)
        if self.backend is not None and self.backend not in ('kraken',
                                                             'krakenc'):
            raise ConfigurationError(
                f"Kraken(backend={self.backend!r}) is not a known backend. "
                f"Choose 'kraken', 'krakenc', or None for automatic dispatch."
            )
        if (self.mode_coupling is not None
                and self.mode_coupling not in ('adiabatic', 'coupled')):
            raise ConfigurationError(
                f"mode_coupling must be 'adiabatic' or 'coupled', "
                f"got {self.mode_coupling!r}."
            )

    def _normalise_env(self, env, mode):
        """Stage 1 of a MODES call reduces a range-dependent environment to
        one profile (:func:`_checks.modes_single_profile`: the caller's
        ``collapse`` methods, else the package defaults) before the shared
        projection:
        normal modes are range-independent, while every field mode segments
        the environment itself."""
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
        - a backend the environment rules out (:meth:`select_backend`) and a
          'precalc' seabed ``.irc`` in the wrong layout.
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
                and self._coupling == 'coupled'):
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
        reject_malformed_irc_bottom(self.model_name, env)

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
        notices = [rmax_zero_notice(
            self.rmax if self._caller_set('rmax') else None,
            model_name=self.model_name),
            kraken_exe_elastic_notice(env, backend=backend,
                                      model_name=self.model_name),
            *self._unused_knob_notices(env, mode, receiver)]
        if mode == RunMode.MODES:
            n_modes = self.n_modes
            if receiver is None:
                # The grid the modes are tabulated on, from the one profile
                # the modes are solved on (_normalise_env).
                tabulation, ppm_notice = dense_mode_depths(
                    env, source,
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
                                     pinned_rmax=self.rmax)
            notices.append(kraken_exe_above_seabed_notice(
                env, mode, backend=backend, c_highs=launch.c_high,
                chosen=self._chose_backend_or_window(),
                model_name=self.model_name))
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
        if self.range_offsets is not None and \
                self.range_offsets.size != receiver.depths.size:
            raise ConfigurationError(
                f"Kraken(range_offsets=...) holds {self.range_offsets.size} "
                f"offset(s) for {receiver.depths.size} receiver depth(s); "
                f"field.exe stops unless there is one offset per depth "
                f"('Number of range offsets, NRro, not equal to number of "
                f"depths, NRz', field.f90:149-152).",
                remediation="Give one offset (m) per receiver depth, "
                            "shallowest first.")
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
        notices.append(kraken_exe_above_seabed_notice(
            env, mode, backend=backend,
            c_highs=[c for launch in launches for c in launch.c_high],
            chosen=self._chose_backend_or_window(),
            model_name=self.model_name))
        _launch, c_low_origin, c_high_origin, rmax_origin, ppm = resolved[0]
        if not self._caller_set('c_high') and not self.leaky_modes:
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
            mode_coupling=(self._coupling if env.is_range_dependent
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
            segments, n_profiles, profile_ranges, max_total_depth = \
                segment_env_for_field(env, freq=freqs, log=self._log,
                                      mode_coupling=self._coupling,
                                      n_segments=self.n_segments)
        else:
            segments, profile_ranges = None, None
            max_total_depth = total_media_depth(env)

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

        freq0 = float(np.atleast_1d(
            np.asarray(source.frequencies, dtype=float))[0])
        if segments is not None:
            rmax, rmax_origin = resolve_rmax(receiver,
                                             pinned_rmax=self.rmax)
            n_mesh = multi_profile_n_mesh(
                segments, float(source.frequencies[0]),
                pinned_n_mesh=self.n_mesh)
            profiles = [seg_env for _range_m, seg_env in segments]
        else:
            # A pinned n_mesh is checked at the deck's freq0 (see
            # :func:`_grid.resolve_modes_launch`).
            n_mesh = single_profile_n_mesh(env, self.n_mesh, freq0)
            rmax, rmax_origin = resolve_rmax(
                receiver, pinned_rmax=self.rmax)
            if marched_frequencies is not None:
                lowest = float(np.min(marched_frequencies))
                band_counts = band_n_mesh(env, deck_frequency=freq0,
                                          lowest_frequency=lowest)
                if self.n_mesh is None:
                    n_mesh = band_counts
                elif (self._caller_set('n_mesh') and n_mesh == 0
                      and band_counts != 0):
                    notices.append(band_mesh_floor_notice(
                        band_counts, deck_frequency=freq0,
                        lowest_frequency=lowest, model_name=self.model_name))
            profiles = [env]
        c_low, c_highs, c_low_origin, c_high_origin = \
            phase_speed_window(
                env, profiles, backend=backend,
                coupled=(segments is not None
                         and self._coupling == 'coupled'),
                         collapse=self._collapse,
                         leaky_modes=self.leaky_modes,
                         log=self._log, pinned_c_high=self.c_high,
                         pinned_c_low=self.c_low)
        launch = KrakenLaunch(
            deck_frequency=freq0,
            marched_frequencies=None if marched_frequencies is None else np.asarray(marched_frequencies, dtype=float),
            tabulation_depths=mode_depths,
            profile_ranges=profile_ranges,
            c_low=c_low,
            c_high=c_highs,
            rmax=rmax,
            n_mesh=n_mesh,
            field_option=build_field_option(
                env.is_range_dependent, source, run_mode,
                mode_coupling=self._coupling),
        )
        return launch, c_low_origin, c_high_origin, rmax_origin, ppm

    # ── stage 4: decks, launches, outputs ─────────────────────────────────

    def _n_launches(self, settings) -> int:
        """One launch per deck the settings hold: one for MODES, a
        narrowband field and a band (whose launch re-runs its deck without
        each bin the solver stops on, :meth:`_launch_band_deck`); one per
        bin of a range-dependent band, each reusing the fixed file names the
        last one wrote after it was read."""
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

    def _sub_band_launch(self, inputs, band) -> KrakenLaunch:
        """The launch of the bins ``band`` (Hz) of the run's band, resolved
        as the run's own: a band deck, or for one bin a narrowband deck at
        it."""
        band = np.asarray(band, dtype=float)
        source = (inputs.source if band.size > 1 else
                  dataclasses.replace(inputs.source, frequencies=np.array(
                      [float(band[0])])))
        sub_launch, *_origins = self._resolve_field_launch(
            inputs.env, source, evaluated_receiver(inputs),
            marched_frequencies=band if band.size > 1 else None,
            run_mode=RunMode.COHERENT_TL,
            backend=inputs.settings.engine.backend, notices=[])
        return sub_launch

    def _unsummed_refusal(self, inputs, outcome, launch=None, *, band=None
                          ) -> ConfigurationError:
        """The typed refusal of a launch (``launch``, by default the
        current one) whose modes solve kept no mode (:class:`_Unsummed`) —
        of a whole band (``band``, its frequencies) when it kept none at any
        bin, ``outcome`` being its top bin's — in the binary's own words: a
        later profile it stopped on
        (:func:`_modes.profile_without_modes_refusal`), else the source's
        profile, where below the waveguide's first modal cut-off a mode
        model has no mode to carry the field on (Computational Ocean
        Acoustics §2.4.5)."""
        launch = launch_of(inputs) if launch is None else launch
        exe = self._modes_exe(inputs.settings.engine.backend)
        if outcome.profile > 0:
            return profile_without_modes_refusal(
                outcome.profile, inputs.env, launch.profile_ranges, exe,
                outcome.frequency, outcome.reason, model_name=self.model_name)
        at = (f"at {outcome.frequency:g} Hz" if band is None else
              f"at any requested frequency ({band[0]:g}-{band[-1]:g} Hz)")
        remedy = below_cutoff_remedy(inputs.env.is_range_dependent)
        if self.c_low is not None or self.c_high is not None:
            remedy = ("Leave c_low and c_high unset, for the window derived "
                      "from the environment, or r" + remedy[1:])
        return ConfigurationError(
            f"{self.model_name}: {exe.name} found no mode {at} in its "
            f"phase-speed window c_low..c_high = {float(launch.c_low):g}.."
            f"{float(max(launch.c_high)):g} m/s ({outcome.reason}). Below a "
            f"waveguide's first modal cut-off a mode model has no mode to "
            f"carry the field on (Computational Ocean Acoustics §2.4.5).",
            remediation=remedy)

    def _warn_unsummed_bins(self, inputs, frequencies, unsummed, no_mode
                            ) -> None:
        """The warnings of a band some of whose bins were not summed
        (``unsummed``, ``{bin index: _Unsummed}``): a ValidityWarning for the
        bins ``no_mode``, where the modes binary found no mode in the
        source's profile — zero in a synthesis (the empty modal sum) — and a
        NumericsWarning for the rest, stopped on a later profile, which make
        a synthesis over the band NaN."""
        def listed(bins):
            more = f" and {len(bins) - 8} more" if len(bins) > 8 else ""
            return (", ".join(f"{frequencies[i]:g}" for i in bins[:8])
                    + more)

        name = self._modes_exe(inputs.settings.engine.backend).name
        if no_mode:
            use = 'RAM' if inputs.env.is_range_dependent else 'Scooter'
            reasons = sorted({unsummed[i].reason for i in no_mode})
            warnings.warn(
                f"{self.model_name}: {name} found no mode at "
                f"{len(no_mode)} of {len(frequencies)} band frequencies "
                f"({listed(no_mode)} Hz; {', '.join(reasons)}). Below a "
                f"waveguide's first modal cut-off the field is carried by "
                f"the continuous spectrum, which a mode sum does not hold "
                f"(Computational Ocean Acoustics §2.4.5; {use} holds it). "
                f"They are NaN in H(f) (Field.no_mode_bins) and zero in "
                f"every synthesis from it (a TIME_SERIES run, "
                f"sound_exposure_level, to_time_trace): the empty modal sum.",
                ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP)
        stops = sorted(i for i in unsummed if unsummed[i].profile > 0)
        if stops:
            first = unsummed[stops[0]]
            ranges = inputs.settings.engine.launches[stops[0]].profile_ranges
            r, depth = profile_location(inputs.env, ranges, first.profile)
            remedy = end_track_or_ram(inputs.env, ranges, first.profile)
            warnings.warn(
                f"{self.model_name}: {len(stops)} of {len(frequencies)} band "
                f"frequencies were not summed: at {listed(stops)} Hz {name} "
                f"found no mode in a later "
                f"profile and stopped ({first.reason}; at "
                f"{first.frequency:g} Hz, profile {first.profile + 1}, "
                f"r = {r:g} m, seafloor {depth:g} m), so no field reaches "
                f"past it; at {first.frequency:g} Hz, "
                f"{remedy[0].lower()}{remedy[1:-1]}. They are NaN in H(f), "
                f"so a synthesis over the band is NaN; a band whose grid "
                f"avoids them synthesises.",
                NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)

    def _note_leaky(self, inputs, frequency, record, *, profile=0,
                    range_m=0.0, cp=None) -> None:
        """Record, for :meth:`_warn_leaky`, a profile (0-based, at
        ``range_m`` m) at ``frequency`` Hz whose modes are all faster than
        the compressional speed of its bottom half-space
        (:func:`_modes.record_seabed_speed` of its mode ``record``): the
        phase speeds ``cp`` (m/s) of the binary's printed table, else the
        record's own."""
        c_bottom = record_seabed_speed(record, inputs.env, range_m)
        if c_bottom is None:
            return
        speeds = (trapped_phase_speeds(record.k, frequency, c_bottom)
                  if cp is None else trapped_among(cp, c_bottom))
        if not speeds.size:
            shear = inputs.env.bottom.halfspace_at(
                range=float(range_m)).shear_speed
            inputs.prepared['leaky'].append(
                (float(frequency), profile, float(range_m),
                 bool(float(shear) > 0.0)))

    def _warn_leaky(self, inputs) -> None:
        """One warning for the frequencies and profiles at which every mode
        the binary kept is faster than the half-space's compressional speed
        (:meth:`_note_leaky`): no mode there is trapped. A mode faster than
        the bottom's compressional speed radiates into the bottom, a leaky
        mode (Computational Ocean Acoustics §2.4.5.1; on kraken.exe it is a
        rigid-bottom duct's root, ``Kraken/BCImpedanceMod.f90:83-89``); a
        rigid or vacuum floor has no half-space to radiate into, so every
        root there is a mode and none is noted
        (:func:`_modes.record_seabed_speed`); over an elastic
        half-space the continuous spectrum lies above it (§4.10.1); along a
        track a mode cut off at a profile radiates into the bottom from
        there (§6.9.2). The field returned is Kraken's sum as computed."""
        # A band's bin solved again on its own deck is noted twice.
        leaky = sorted(set(inputs.prepared['leaky']))
        if not leaky:
            return
        named = [f"{f:g} Hz" + ("" if p == 0 else
                                f" in profile {p + 1} (r = {r:g} m)")
                 for f, p, r, _shear in leaky[:8]]
        more = f" and {len(leaky) - 8} more" if len(leaky) > 8 else ""
        elastic = any(shear for *_where, shear in leaky)
        track = any(p > 0 for _f, p, _r, _shear in leaky)
        warnings.warn(
            f"{self.model_name}: every mode "
            f"{self._modes_exe(inputs.settings.engine.backend).name} kept is "
            f"faster than the bottom half-space's compressional speed, at "
            f"{', '.join(named)}{more}: none is trapped."
            + (" kraken.exe's roots there are a rigid-bottom duct's "
               "eigenvalues, not leaky modes: its real arithmetic keeps the "
               "real part, 0, of the half-space's imaginary vertical "
               "wavenumber, the rigid condition "
               "(Kraken/BCImpedanceMod.f90:83-89)."
               if inputs.settings.engine.backend == 'kraken' else "")
            + " A mode faster than the bottom's compressional speed radiates "
            "into the bottom, a leaky mode (Computational Ocean Acoustics "
            "§2.4.5.1"
            + ("; over an elastic half-space the continuous spectrum lies "
               "there, §4.10.1" if elastic else "")
            + ("; from the profile where it is cut off on a track, §6.9.2"
               if track else "")
            + "), and a mode sum does not represent the field there. The "
            "field returned is Kraken's sum as computed; a "
            "wavenumber-integration model (Scooter"
            + (", or OAST over shear" if elastic else "")
            + ") or, on a track, RAM holds the continuous spectrum.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP)

    def _prepare_launches(self, env, settings):
        """What the launches of a run leave for its read and result steps:
        each launch's outcome, by launch index (:meth:`_run_launch`,
        :meth:`_launch_band_deck`), and the frequencies and profiles at
        which every mode the binary kept is faster than the half-space
        (:meth:`_note_leaky`)."""
        return {'outcomes': {}, 'leaky': []}

    def _modes_exe(self, backend: str) -> Path:
        """The modes binary of ``backend``: the resolved ``kraken.exe``, or
        ``krakenc.exe`` looked up on first use."""
        if backend == 'krakenc':
            return self._find_binary('krakenc')
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

    def _launch_or_refuse(self, inputs, deck: Path) -> None:
        """Route step (modes, field, band_bin): the launch as stage 3
        resolved it, refused when its modes solve kept no mode
        (:meth:`_run_launch`, :meth:`_unsummed_refusal`)."""
        outcome = self._run_launch(inputs, launch_of(inputs), deck)
        if outcome is not None:
            raise self._unsummed_refusal(inputs, outcome)

    def _launch_band_deck(self, inputs, deck: Path) -> None:
        """Route step (band): the band on one deck — without the bins the
        modes binary kept no mode at (:meth:`_band_deck_bins`) — summed by
        field.exe. A band with no bin left is refused."""
        launch = launch_of(inputs)
        frequencies = np.asarray(launch.marched_frequencies, dtype=float)
        launch, deck, kept, dropped = self._band_deck_bins(
            inputs, frequencies, launch, deck)
        if not kept.size:
            top = frequencies.size - 1
            raise self._unsummed_refusal(
                inputs, _Unsummed(float(frequencies[top]), 0, dropped[top]),
                launch, band=frequencies)
        self._run_field_exe(inputs.work_dir, deck.stem, launch.field_option)
        if launch.marched_frequencies is not None:
            require_band_axis(inputs.work_dir, deck.stem,
                              launch.marched_frequencies,
                              model_name=self.model_name)
        inputs.prepared['outcomes'][inputs.launch] = _BandDeck(
            launch, kept, dropped)

    def _band_deck_bins(self, inputs, frequencies, launch, deck: Path):
        """The band ``frequencies`` (Hz) on one deck, solved until the modes
        binary runs through it: ``(launch, deck, kept, dropped)`` — the
        launch and deck it ran through last, the bins they carry (indices
        into ``frequencies``) and the binary's own words for each bin it
        kept no mode at (``{bin index: reason}``).

        The binary solves the deck's frequencies in turn and stops at the
        first whose window holds no mode (``Kraken/kraken.f90:962``,
        ``Kraken/krakenc.f90:445``); a zero-mode record would leave the
        multi-frequency ``.mod`` unreadable to field.exe, so that bin leaves
        the deck, which runs again, until the binary solves every bin left.
        A stop on the deck's lowest bin costs one solve, so a band that
        starts below the cut-off re-runs once per such bin. The bins a deck
        that ran through kept no mode at leave it together
        (:meth:`_band_deck_empty_bins`)."""
        kept, dropped = list(range(frequencies.size)), {}
        while True:
            stop = self._solve_modes(inputs, launch, deck)
            if stop is None:
                empty, leaky = self._band_deck_empty_bins(
                    inputs.work_dir / deck.stem, frequencies[kept])
                if not empty:
                    for j, record, cp in leaky:
                        self._note_leaky(inputs, frequencies[kept[j]], record,
                                         cp=cp)
                    return launch, deck, np.array(kept, dtype=int), dropped
                reason = _EMPTY_TABLE
            else:
                empty, reason = [min(stop[1], len(kept) - 1)], stop[2]
            dropped.update({kept[j]: reason for j in empty})
            kept = [b for j, b in enumerate(kept) if j not in set(empty)]
            if not kept:
                return launch, deck, np.zeros(0, dtype=int), dropped
            launch = self._sub_band_launch(inputs, frequencies[kept])
            deck = write_decks(inputs, launch, interp_ssp=self.interp_ssp,
                               log=self._log)

    def _band_deck_empty_bins(self, base: Path, frequencies):
        """``(empty, modes)`` of a band deck the modes binary ran through
        (bins ``frequencies``, Hz; ``base``, its path without a suffix): the
        indices of the bins it kept no mode at — every bin whose printed
        table has no row, else the first whose record holds no mode (a
        zero-mode record can break the reading of the records after it) —
        and, when there are none, per bin ``(index, record, cp)`` for
        :meth:`_note_leaky`: the phase speeds of its printed table
        (:func:`_modes.printed_mode_tables`, whose first row is its slowest
        mode, ``kraken.f90:95-103``) with the first record, which holds the
        deck's half-space, else its own record and ``None``.

        krakenc keeps no root at the last mesh of some bins without
        stopping (30 m over sand, 1, 1.5, 3 and 22.5 Hz on a 0.5-60 Hz
        deck). Reading a record walks the file from its first frequency,
        which on a 340-bin band took twice the band's solve; a printed
        table saves it."""
        tables = printed_mode_tables(
            _read_prt_if_present(base.with_suffix('.prt')), len(frequencies))
        empty = [i for i, table in enumerate(tables)
                 if table is not None and not table[1].size]
        if empty:
            return empty, []
        first = read_mode_record(base, frequencies[0],
                                 model_name=self.model_name)
        if first is None:
            return [0], []
        modes = []
        for i, frequency in enumerate(frequencies):
            if tables[i] is not None:
                modes.append((i, first, tables[i][1]))
                continue
            record = read_mode_record(base, frequency,
                                      model_name=self.model_name)
            if record is None:
                return [i], []
            modes.append((i, record, None))
        return [], modes

    def _launch_band_bin(self, inputs, deck: Path) -> None:
        """Route step (band_by_frequency): one bin of a range-dependent
        band (:meth:`_run_launch`) — a bin not summed is NaN in H(f) — and
        the first bin announces the band's cost."""
        started = time.perf_counter()
        inputs.prepared['outcomes'][inputs.launch] = self._run_launch(
            inputs, launch_of(inputs), deck)
        if inputs.launch == 0:
            self._announce_band_cost(time.perf_counter() - started,
                                     len(inputs.settings.engine.launches))

    def _solve_modes(self, inputs, launch: KrakenLaunch, deck: Path):
        """The modes binary on the deck of ``launch``: where it stopped on a
        window holding no mode, as :meth:`_run_kraken_executable` returns
        it."""
        exe = self._modes_exe(inputs.settings.engine.backend)
        self._log(f"Running {exe.name}"
                  + ("..." if launch.field_option is not None
                     else " (modes)..."))
        return self._run_kraken_executable(deck.stem, inputs.work_dir,
                                           exe=exe)

    def _run_launch(self, inputs, launch: KrakenLaunch, deck: Path):
        """One launch: the modes binary on its deck (:meth:`_solve_modes`)
        and, on a field launch, field.exe on the ``.mod`` it wrote — one
        solve and one sum.

        ``None`` when the solve kept modes; else :class:`_Unsummed` — the
        binary stopped, in the source's profile or a later one, or ran
        through and kept no mode — and nothing is summed. Each profile of a
        field launch whose modes are all faster than its half-space is noted
        for the run's warning (:meth:`_note_leaky`)."""
        frequency = float(deck_source(inputs, launch).frequencies[0])
        stop = self._solve_modes(inputs, launch, deck)
        if stop is not None:
            return _Unsummed(frequency, stop[0], stop[2])
        base = inputs.work_dir / deck.stem
        modes = read_mode_record(base, frequency, model_name=self.model_name)
        if modes is None:
            return _Unsummed(frequency, 0, _EMPTY_TABLE)
        if launch.field_option is None:
            return None
        for profile in range(launch.n_profiles):
            record = (modes if profile == 0 else read_mode_record(
                base, frequency, profile + 1, model_name=self.model_name))
            if record is not None:
                self._note_leaky(
                    inputs, frequency, record, profile=profile,
                    range_m=(0.0 if launch.profile_ranges is None
                             else launch.profile_ranges[profile]))
        self._run_field_exe(inputs.work_dir, deck.stem, launch.field_option)
        return None

    def _read_output(self, inputs, deck: Path):
        """Stage 4: what the launch wrote, as the io readers return it, read
        as its route does (:data:`_ROUTE_STEPS`)."""
        return _ROUTE_STEPS[inputs.settings.engine.route].read(
            self, inputs, deck)

    def _read_mode_set(self, inputs, deck: Path):
        """Route step (modes): the ``.mod`` the launch's modes solve wrote
        (:meth:`_run_launch`), with the group speeds krakenc prints in its
        ``.prt``; a set some of whose modes are non-trapped logs it
        (:func:`_modes.log_non_trapped_modes`)."""
        work_dir = inputs.work_dir
        base = deck.stem
        exe = self._modes_exe(inputs.settings.engine.backend)
        frequency = float(inputs.source.frequencies[0])
        self._log(f"Reading mode file: {work_dir / base}.mod")
        modes = read_mode_record(work_dir / base, frequency,
                                 model_name=self.model_name)
        log_non_trapped_modes(modes.k, inputs.env, frequency,
                              leaky_modes=self.leaky_modes, log=self._log)
        # Read before the work directory is let go. Only krakenc fills
        # this column; kraken prints it unassigned.
        vg = read_group_speeds_from_prt(
            work_dir, base, modes.n_modes,
            fills_vg=(exe.stem.lower() == 'krakenc'))
        return {'modes': modes, 'group_speeds': vg}

    def _read_field(self, inputs, deck: Path):
        """Route step (field): the narrowband ``.shd`` as
        :func:`~uacpy.io.oalib_reader.read_shd_file` reads it."""
        return read_shd(inputs.work_dir / f'{deck.stem}.shd',
                        launch_of(inputs), one_bin=False)

    def _read_band_bin(self, inputs, deck: Path):
        """Route step (band_by_frequency): one bin's narrowband ``.shd``, or
        ``{'unsummed': why}`` for a bin its own solve left unsummed
        (:meth:`_launch_band_bin`)."""
        outcome = inputs.prepared['outcomes'][inputs.launch]
        if outcome is not None:
            return {'unsummed': outcome}
        return self._read_field(inputs, deck)

    def _read_one_bin(self, inputs, deck: Path):
        """Route step (band_bin): a band of one bin, the ``.shd`` as
        :func:`~uacpy.io.oalib_reader.read_shd_bin` reads it."""
        return read_shd(inputs.work_dir / f'{deck.stem}.shd',
                        launch_of(inputs), one_bin=True)

    def _read_band_deck(self, inputs, deck: Path):
        """Route step (band): the ``.shd`` of the deck the band's launch ran
        through (:meth:`_launch_band_deck`, which holds its launch) as
        :func:`~uacpy.io.oalib_reader.read_shd_bin` reads it."""
        band = inputs.prepared['outcomes'][inputs.launch]
        return read_shd(inputs.work_dir / f'{deck.stem}.shd', band.launch,
                        one_bin=True)

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
            group_speeds=raw[0]['group_speeds'],
            water_density=inputs.env.water_density, model_exe=self._exe,
            model_name=self.model_name, provenance=self.provenance
        )
        attach_output_paths(
            result, inputs.work_dir, deck[0].stem,
            primary_files=(('mod_file', '.mod'),),
            cleanup=self.cleanup,
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
        sub-bottom; a profile whose modes are all faster than its half-space
        warns (:meth:`_warn_leaky`)."""
        settings = inputs.settings
        engine = settings.engine
        exe = self._modes_exe(engine.backend)
        field = self._field_of_launch(
            inputs, engine.launches[0], raw[0], exe,
            run_mode=settings.mode)
        self._warn_leaky(inputs)
        return reinsert_nan_depths(field, inputs.receiver,
                                   self._receiver_keep(engine))

    def _band_deck_result(self, inputs, deck, raw):
        """Route step (band): the band on one deck, on the full band grid —
        a bin above the first summed one that the deck left without a field
        solved again on a deck of its own (:meth:`_resolve_failed_bins`),
        and a bin no solve summed NaN (:meth:`_mark_unsummed_bins`)."""
        engine = inputs.settings.engine
        exe = self._modes_exe(engine.backend)
        frequencies = np.asarray(inputs.settings.frequencies, dtype=float)
        band = inputs.prepared['outcomes'][0]
        tf = self._band_field(inputs, band.launch, raw[0], exe,
                              frequencies[band.kept])
        tf = self._on_band_grid(inputs, tf, frequencies, band.kept)
        solved = np.zeros(frequencies.size, dtype=bool)
        solved[band.kept] = True
        tf, solved, outcomes, note = self._resolve_failed_bins(
            inputs, tf, exe, frequencies, solved)
        tf = self._mark_unsummed_bins(
            inputs, tf, frequencies,
            {int(i): outcomes.get(int(i), _Unsummed(
                float(frequencies[i]), 0,
                band.dropped.get(int(i), _EMPTY_TABLE)))
             for i in np.flatnonzero(~solved)})
        result = self._finish_band(inputs, tf)
        if note is not None:
            add_run_note(result, note)
        return result

    def _on_band_grid(self, inputs, tf, frequencies, kept):
        """``tf``, the field of the deck's bins ``kept`` (indices into
        ``frequencies``, Hz), on the band's full grid, NaN at every other
        bin."""
        if len(kept) == frequencies.size:
            return tf
        receiver = evaluated_receiver(inputs)
        data = np.full(tf.data.shape[:2] + (frequencies.size,), np.nan,
                       dtype=tf.data.dtype)
        data[:, :, kept] = tf.data
        return tf.replace(
            data=data,
            coords={'depth': receiver.depths, 'range': receiver.ranges,
                    'frequency': frequencies},
            pinned=None, aux_coords=None, frequencies=frequencies)

    def _mark_unsummed_bins(self, inputs, tf, frequencies, unsummed):
        """``tf`` with its bins where the modes binary found no mode in the
        source's profile recorded in
        :attr:`~uacpy.core.results.Field.no_mode_bins`, after the warnings
        of every bin not summed (``unsummed``, ``{bin index: _Unsummed}``,
        :meth:`_warn_unsummed_bins`)."""
        no_mode = sorted(i for i in unsummed if unsummed[i].profile == 0)
        if unsummed:
            self._warn_unsummed_bins(inputs, frequencies, unsummed, no_mode)
        return tf.replace(no_mode_bins=tuple(no_mode)) if no_mode else tf

    def _resolve_failed_bins(self, inputs, tf, exe, frequencies, solved):
        """``(tf, solved, outcomes, note)`` with every bin above the first
        ``solved`` one that the band deck left without a field — no level at
        any receiver (field.exe's empty sum, or NaN), or not summed — solved
        again on a deck of its own,
        in a subdirectory of the run's. ``solved`` marks the bins a solve
        summed, ``outcomes`` why each bin solved again was not
        (:class:`_Unsummed`, by bin index), ``note`` the line of
        ``run_settings.notes`` naming the bins solved again (``None`` when
        none was).

        The band deck solves all its frequencies in one modes run, and a bin
        can fail there that solves alone. Measured: 30 m of water over sand,
        ``leaky_modes=True``, a 31-1999 Hz krakenc deck kept one 579 m/s
        root at 35 Hz whose eigenvector failed inverse iteration and was
        zeroed (``krakenc.f90:708-711``), so the deck's sum held no field
        there, where 35 Hz on its own deck solved (2 modes, a finite field);
        the bins the deck did solve agree with their own-deck solves to 0.06
        dB there. The bins below the first summed one, which the binary
        stopped on, are left alone. A bin whose own solve keeps no
        mode is NaN, unsummed; one whose own solve sums no field either, or
        fails, keeps what the band deck left, and one warning names it."""
        data = np.asarray(tf.data)
        first = int(np.argmax(solved)) if solved.any() else data.shape[-1]
        failed = [i for i in range(first, data.shape[-1])
                  if not solved[i] or _no_level(data[..., i])]
        if not failed:
            return tf, solved, {}, None
        data = np.array(data, copy=True)
        solved = np.array(solved, copy=True)
        raised, outcomes = {}, {}
        for i in failed:
            launch = self._sub_band_launch(inputs, frequencies[i:i + 1])
            work_dir = inputs.work_dir / f'bin_{i}'
            work_dir.mkdir(exist_ok=True)
            bin_inputs = dataclasses.replace(inputs, work_dir=work_dir)
            # The band run already said what this geometry and window
            # deserve; the bin's own solve repeats it.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                deck = write_decks(bin_inputs, launch,
                                   interp_ssp=self.interp_ssp, log=self._log)
                try:
                    outcome = self._run_launch(bin_inputs, launch, deck)
                    solved[i] = outcome is None
                    if outcome is not None:
                        outcomes[i] = outcome
                        data[..., i] = np.nan
                except ModelExecutionError as exc:
                    raised[i] = str(exc).strip().splitlines()[0]
                    continue
                if not solved[i]:
                    continue
                field = self._band_field(
                    bin_inputs, launch, read_shd(
                        work_dir / f'{deck.stem}.shd', launch, one_bin=True),
                    exe, frequencies[i:i + 1])
            data[..., i] = np.asarray(field.data)[..., 0]

        def listed(bins):
            more = f" and {len(bins) - 8} more" if len(bins) > 8 else ""
            return ", ".join(f"{frequencies[i]:g}" for i in bins[:8]) + more

        still = [i for i in failed if solved[i] and _no_level(data[..., i])]
        parts = []
        if still:
            parts.append(
                f"returned no field (no level at any receiver) at "
                f"{len(still)} of {data.shape[-1]} band frequencies "
                f"({listed(still)} Hz), on the band deck and again on a deck "
                f"of its own")
        if raised:
            parts.append(
                f"failed on a deck of its own at {listed(sorted(raised))} Hz "
                f"({next(iter(raised.values()))}), which the band deck left "
                f"without a field")
        if parts:
            warnings.warn(
                f"{self.model_name}: {Path(exe).name} {'; and '.join(parts)}"
                f": those bins hold what the band deck left there (NaN where "
                f"it summed nothing, which makes a synthesis over the band "
                f"NaN). Its .prt names the failure (keep it with work_dir= "
                f"and cleanup=False); a band whose grid avoids those "
                f"frequencies synthesises.",
                NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)
        note = (f"frequencies {listed(failed)} Hz: the band deck left them "
                f"without a field; each was solved again on a deck of its own "
                f"(bin_<index>/ in the work directory)")
        self._log(note, level="info")
        return tf.replace(data=data), solved, outcomes, note

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
        at receiver depths in an elastic sub-bottom, the no-mode bins of a
        TIME_SERIES run zeroed before its synthesis, then the base's
        broadband finish; a bin whose modes are all faster than the
        half-space warns (:meth:`_warn_leaky`)."""
        self._warn_leaky(inputs)
        settings = inputs.settings
        tf = reinsert_nan_depths(tf, inputs.receiver,
                                 self._receiver_keep(settings.engine))
        no_mode = list(tf.no_mode_bins or ())
        if settings.mode == RunMode.TIME_SERIES and no_mode:
            # The synthesis takes these bins as zero anyway
            # (zero_no_mode_bins) and would say so a second time: the run
            # has named them (_warn_unsummed_bins). Zeroed here, the trace is
            # the same bit for bit and the run warns once.
            data = np.array(tf.data, copy=True)
            data[:, :, no_mode] = 0.0
            tf.data = data
        return finish_broadband(self.model_name, tf, settings)

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
        return Field.from_dict(d)

    def _stack_band_by_frequency(self, inputs, raw, exe, frequencies):
        """``H(f)`` of a range-dependent band: the narrowband field of every
        bin's launch, stacked on a trailing frequency axis, NaN at a bin its
        own solve left unsummed (:meth:`_mark_unsummed_bins`); a band none
        of whose bins was summed is refused for its top bin
        (:meth:`_unsummed_refusal`).

        The stacked result carries what a NATIVE broadband field carries,
        not what one narrowband slab happens to have. Two things travel
        only on the broadband path and are stamped here: the frequency
        vector (without it the field denies being broadband) and the phase
        reference (without it a complex-weighted sum is refused as "no phase
        reference"). Its identity is the native band's: the modes binary
        as ``backend``, the model source, the source depths and level
        (:func:`~uacpy.models._extract.result_kwargs`). The speed a
        time-series window anchors on is the run's
        ``run_settings.waveguide``."""
        launches = inputs.settings.engine.launches
        unsummed = {i: r['unsummed'] for i, r in enumerate(raw)
                    if 'unsummed' in r}
        if len(unsummed) == len(raw):
            top = len(raw) - 1
            raise self._unsummed_refusal(inputs, unsummed[top], launches[top],
                                         band=frequencies)
        slabs, first = [None] * len(raw), None
        for i, raw_i in enumerate(raw):
            if i in unsummed:
                continue
            field = self._field_of_launch(inputs, launches[i], raw_i, exe,
                                          run_mode=RunMode.COHERENT_TL)
            first = first if first is not None else field
            slabs[i] = np.asarray(field.data)
        blank = np.full_like(next(s for s in slabs if s is not None), np.nan)
        slabs = [blank if slab is None else slab for slab in slabs]
        metadata = dict(getattr(first, 'metadata', {}) or {})
        tf = Field(
            data=np.stack(slabs, axis=-1),
            coords={'depth': np.asarray(first.coords['depth'], dtype=float),
                    'range': np.asarray(first.coords['range'], dtype=float),
                    'frequency': frequencies},
            **result_kwargs(
                self.model_name, self.provenance, inputs.source,
                backend=first.backend, frequencies=frequencies,
                phase_reference=getattr(first, 'phase_reference', None),
                **metadata),
        )
        return self._mark_unsummed_bins(inputs, tf, frequencies, unsummed)

    def _field_of_launch(self, inputs, launch, raw, exe, *, run_mode):
        """The field of one field launch in the package's conventions
        (:func:`_extract.assemble_field_from_shd`), stamped with the modes
        binary that ran, the r = 0 mask and the output paths; an empty modal
        sum warns."""
        source = deck_source(inputs, launch)
        receiver = evaluated_receiver(inputs)
        env = inputs.env
        field = assemble_field_from_shd(
            raw, source, receiver, launch.n_profiles, run_mode,
            line_c_source=([_source_sound_speed(env, source.at_depth(i))
                            for i in range(np.atleast_1d(
                                source.depths).size)]
                           if source.source_type == 'line' else None),
            source_rho=[_source_density(env, z) for z in
                        np.atleast_1d(np.asarray(source.depths, float))],
                        backend=Path(exe).stem,
                        mode_coupling=self._coupling,
                        model_name=self.model_name,
                        provenance=self.provenance)

        # An empty modal sum: field.exe's grid holds no level above the
        # no-energy floor at any receiver. Tested on TL so it covers both the
        # complex branches and the real dB INCOHERENT_TL one: an empty sum
        # saturates at the PRESSURE_FLOOR clamp.
        tl = np.asarray(field.dB, dtype=float)
        finite = np.isfinite(tl)
        if not np.any(tl[finite] < NO_ENERGY_DB):
            warnings.warn(
                f"{self.model_name}: field.exe's sum holds no level above the "
                f"no-energy floor at any receiver: the modes it summed carry "
                f"nothing there (krakenc zeroes an eigenvector its inverse "
                f"iteration did not converge on, krakenc.f90:708-711). The "
                f"modes run's .prt names its warnings (keep it with "
                f"work_dir= and cleanup=False).",
                ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )

        # After the guard above: the guard reads the whole TL grid, and a
        # single-range r=0 request would otherwise look like an empty
        # modal sum.
        # Each source depth on its own, and one launch, one r = 0 notice —
        # not one per source depth.
        field = restack_source_depths(
            [mask_source_axis(self.model_name, slab, source, warn=(i == 0))
             for i, slab in enumerate(split_source_depths(field))],
            field, who=f"{self.model_name}.run")

        for slab in _slabs_of(field):
            attach_output_paths(
                slab, inputs.work_dir, _FIELD_BASE,
                primary_files=(
                    ('shd_file', '.shd'),
                    ('mod_file', '.mod'),
                ),
                cleanup=self.cleanup,
            )
            attach_field_prt_path(slab, inputs.work_dir, cleanup=self.cleanup)

        self._log("Kraken simulation complete")
        return field

    # ── field.exe pipeline ──────────────────────────────────────────────

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
        run_launch(self._launch_context(), Launch(
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
                lambda result: warn_on_prt_warnings(
                    self.model_name, work_dir, _FIELD_PRT_ROOT),
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
    'modes': _RouteSteps(Kraken._launch_or_refuse, Kraken._read_mode_set,
                         Kraken._mode_set_result),
    'field': _RouteSteps(Kraken._launch_or_refuse, Kraken._read_field,
                         Kraken._field_result),
    'band': _RouteSteps(Kraken._launch_band_deck, Kraken._read_band_deck,
                        Kraken._band_deck_result),
    'band_bin': _RouteSteps(Kraken._launch_or_refuse, Kraken._read_one_bin,
                            Kraken._band_bin_result),
    'band_by_frequency': _RouteSteps(Kraken._launch_band_bin,
                                     Kraken._read_band_bin,
                                     Kraken._band_by_frequency_result),
})
