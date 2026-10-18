"""The :class:`RAM` model: its knobs, traits, run hooks and the launches of
its binaries."""

import dataclasses
import numpy as np
import warnings
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Dict, Optional, Union
from uacpy.models.base import (
    DEFAULT_RUN_TIMEOUT_S, PropagationModel, StageInputs, _host_options,
)
from uacpy.models._launch import (
    BinarySearch, Launch, openmp_thread_env, require_output, run_launch,
)
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.models._band import BandResolution, broadband_band
from uacpy.models._notices import give_notice
from uacpy.models._knobs import (
    is_real_number, is_whole_number, positive_finite, whole_count,
)
from uacpy.models._spec import EngineTraits, ModelSpec
from uacpy.core.run_settings import OutputSpec, RunMode
from uacpy.core.environment import (
    Environment,
)
from uacpy.core.boundary import BoundaryType
from uacpy.core.results import PhaseReference
from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, NumericsWarning,
    UnsupportedFeatureError,
)
from uacpy.io.mpirams_reader import read_psif
from uacpy.models.ram._settings import RamKnobs, RamSettings
from uacpy.models.ram._band import (
    sweep_on_the_pins, warn_if_band_pins_unused_on_tl,
)
from uacpy.models.ram._stability import DEFAULT_RAMS_ROTATION
from uacpy.models.pe_grid import DEFAULT_N_PADE
from uacpy.models.ram._domain import (
    collapse_surface_to_pressure_release,
    launch_grid,
    resolve_c0,
    warn_if_pinned_absorber_is_below_the_seabed,
    water_alpha_band,
)
from uacpy.models.ram.grid import (
    warn_if_pinned_accuracy_is_unused,
    warn_if_pinned_angle_max_is_below_the_slope,
    warn_if_pinned_rams_dr_factor_is_unused,
    warn_if_pinned_dz_misplaces_the_seafloor,
    warn_if_ramsurf_surface_unresolved,
    warn_if_receivers_see_paths_steeper_than_the_band,
)
from uacpy.models.ram._dispatch import (
    backend_origin,
    check_rams_top_layer_carries_shear,
    prefer_ramgeo,
    validate_forced_backend,
)
from uacpy.models.ram.mpirams import (
    assemble_broadband_field,
    assemble_tl_field,
    check_mpirams_output_range_spacing,
    resolve_mpirams_band,
    resolve_mpirams_tl,
    write_mpirams_deck,
)
from uacpy.models.ram.collins import (
    assemble_collins_band_field,
    assemble_collins_tl_field,
    collins_deck_base,
    raise_on_ram_stop,
    read_collins_output,
    resolve_collins_band,
    resolve_collins_tl,
    write_collins_deck,
)
from uacpy.models._extract import finish_broadband


# Lytaev grid-accuracy target used when the caller does not pin one: the
# target of every example in Lytaev (2023) §5.1, "In all the examples,
# target accuracy ε is set to 10⁻³".
DEFAULT_RAM_ACCURACY = 1e-3


# Output files each family writes into the work dir. Cleared before launch so
# a pinned ``work_dir`` cannot hand an earlier run's output back as this run's
# answer — the binaries write fixed names, with no run-specific stem.
_COLLINS_OUTPUTS = ('tl.grid', 'pcomplex.bin')


_MPIRAMS_OUTPUTS = ('psif.dat',)


#: The run modes that march a frequency band (one mpiramS sweep, or one
#: Collins launch per bin) and return H(f) or its synthesis.
_BAND_MODES = frozenset({RunMode.BROADBAND, RunMode.TIME_SERIES})


class RAM(PropagationModel):
    """
    RAM - Range-dependent Acoustic Model (Parabolic Equation), multi-backend.

    A unified façade that picks one of four vendored Collins-family PE
    binaries at run-time based on the environment:

    ================================  =================================================
    Environment                       Backend selected
    ================================  =================================================
    fluid + flat, layered, narrowband ``ramgeo`` — Collins' RD layered fluid PE
    fluid + flat (simple / broadband) ``mpiramS`` — Dushaw's broadband PE (Q/T loop)
    elastic bottom (any shear>0)      ``rams0.5`` — Collins' elastic PE
    fluid bottom + altimetry          ``ramsurf1.5`` — Collins' rough-surface PE
    elastic + altimetry               ``UnsupportedFeatureError`` (no published PE)
    ================================  =================================================

    ``ramgeo`` tracks sediment layers *parallel to the bathymetry* — the most
    faithful Collins treatment of a sloping layered fluid seabed — and is
    auto-selected for narrowband (COHERENT_TL) layered cases. For a simple
    half-space mpiramS is preferred (there is no layer geometry for ramgeo
    to track), but ramgeo *accepts* a simple bottom when forced. Like the
    other Collins backends it supports every run mode via uacpy's complex-
    envelope patch; auto-dispatch hands broadband / time-series to mpiramS's
    faster native sweep. Pass ``RAM(backend=...)`` to force a backend
    (``'mpirams'`` / ``'ramgeo'`` / ``'rams'`` / ``'ramsurf'``).

    Use ``RAM(...).select_backend(env)`` to inspect the choice without
    actually running, and ``RAM(...).run_settings(env, source,
    receiver).engine`` (a :class:`RamSettings`) for everything a run will
    march — the backend, the PE reference speed, the band and each launch's
    grid (``dr``, ``dz``, ``zmax``, depth points) — before anything is
    written. Range-dependent SSP, bathymetry and (layered) bottom
    are supported by every backend: mpiramS threads them through its native
    range-dependent setup, and the Collins backends emit one ``ram.in``
    profile section per range break (each carrying its range-local SSP and
    Collins-style depth/value bottom profile). Water-column volume
    attenuation (``env.absorption``) is too: every backend receives
    ``alpha(z)`` in dB per local wavelength — per frequency bin on a
    broadband sweep — and applies it to the water wavenumber the way the
    seabed's attenuation is applied (uacpy's patched binaries; see
    ``third_party/MODIFICATIONS.md``).

    Limitations
    -----------
    - The lower boundary at ``zmax`` is an absorbing layer, not a rigid
      Neumann floor — true rigid bottoms are not supported.
    - Collins backends (rams0.5, ramsurf1.5, ramgeo1.5) are single-frequency at the
      Fortran level. uacpy's local patch dumps the complex envelope (see
      ``third_party/MODIFICATIONS.md``); the wrapper drives the binary
      in a Python-side frequency loop to produce ``BROADBAND`` /
      ``TIME_SERIES`` outputs. mpiramS is still faster for fluid+flat
      broadband (in-process Fortran loop with shared setup).

    Run modes
    ---------
    COHERENT_TL:
        Narrowband TL over a range-depth grid. Available on every backend.
        Returns ``Field``.

    BROADBAND:
        Broadband complex pressure field. Returns ``Field`` with ψ(depth,
        range, frequency) for downstream IFFT to time domain. Available on
        every backend: mpiramS sweeps the (fc, Q, T) band inside the Fortran
        loop, the Collins backends run one subprocess per band frequency and
        read the patched ``pcomplex.bin``.

    TIME_SERIES:
        Real pressure p(t) at each receiver. Internally runs BROADBAND
        and convolves with ``source_waveform`` (sampled at ``sample_rate``).
        Returns ``Field`` with shape (n_d, n_r, n_t). Available on every
        backend, same split as BROADBAND.

    Some constructor kwargs are backend-specific. The list below tags each
    one with the backends that consume it; the Collins backends (ramgeo,
    rams0.5, ramsurf1.5) have no deck field for a setting tagged
    ``[mpiramS]``, so uacpy emits a ``FallbackWarning`` when any such setting
    is overridden from its default and the dispatcher then picks a Collins
    backend.

    Notes
    -----
    Defaults auto-derived at ``run()`` time:

    - ``dr=None`` / ``dz=None`` → Lytaev (2023) Padé-error optimizer
      picks the coarsest grid that meets ``accuracy``.
    - ``zmax=None`` → ``_domain.compute_zmax`` (water, sediment stack, a
      pad of real seabed and the absorbing layer).
    - ``c0=None`` → Lytaev Eq. (15) from speed spectrum.
    - ``q_factor`` / ``record_duration`` → narrowband ``(1e6, 1.0)`` for
      ``COHERENT_TL``, which marches the source frequency alone (a pinned
      value is not used there, and the run says so); on ``BROADBAND`` /
      ``TIME_SERIES`` resolved from the source (their entries in
      ``__init__``).
    - Backend (mpiramS / ramgeo / rams0.5 / ramsurf1.5) picked by
      :meth:`select_backend` from ``env`` shape.

    With ``verbose='info'`` the resolved Padé grid is logged per frequency.
    """

    # Declarative metadata (see PropagationModel / ModelSpec). RAM is the
    # range-dependent PE engine: every range-dependence axis is honoured by
    # some backend (mpiramS / ramgeo / rams0.5 / ramsurf1.5), so all flags
    # except multi_source_depth are True. _validate_forced_backend rejects real
    # per-backend mismatches at run() time. No collapse override — RAM uses
    # the base DEFAULT_COLLAPSE.
    spec = ModelSpec(
        modes=(RunMode.COHERENT_TL, RunMode.BROADBAND, RunMode.TIME_SERIES),
        supports={
            'altimetry',
            'range_dependent_bathymetry',
            'range_dependent_ssp',
            'range_dependent_bottom',
            'layered_bottom',
            'elastic_media',
        },
        traits=EngineTraits(
            consumes_run_t_start=True,
            # Every backend applies ``env.absorption`` as a dB/wavelength
            # profile on the water wavenumber
            # (:func:`_domain.water_attenuation_block`).
            consumes_volume_absorption=True,
            # The PE marches through every sediment layer and returns its
            # field down to the top of the absorbing layer, which the
            # automatic domain places below the deepest modelled stack
            # (:func:`_domain.adequate_zmax`); only samples inside that
            # layer come back NaN (:func:`_domain.mask_absorbing_layer`).
            receivers_reach_sediment=True,
        ),
    )

    provenance_id = 'collins_ram'

    #: The binary of each backend. ram1.5 (Collins fluid PE) is not built:
    #: mpiramS runs fluid + flat (broadband and range-dependent bottom).
    #: ramgeo lives in its own vendor dir; rams0.5 and ramsurf1.5 share the
    #: ramsurf/ tree.
    _BINARIES = MappingProxyType({
        'mpirams': BinarySearch(('s_mpiram',), ('mpirams',), 'mpiramS'),
        'ramgeo': BinarySearch(('ramgeo',), ('ramgeo',), 'ramgeo'),
        'rams': BinarySearch(('rams0.5',), ('ramsurf',), 'ramsurf'),
        'ramsurf': BinarySearch(('ramsurf1.5',), ('ramsurf',), 'ramsurf'),
    })

    # Complex pressure in the travelling-wave convention for the TL and H(f)
    # modes (``models/ram/_pe_phase.py`` converts every backend's envelope); the
    # synthesised p(t) is real.
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

    # The stability range defaults to the farthest receiver: the call's. The
    # (Q, T) sweep is derived from the call's frequencies, and a pinned pair
    # replaces the call's band with the full sweep.
    _UNPINNED_FIELDS = frozenset({'stability_range', 'q_factor',
                                  'record_duration'})

    @classmethod
    def _per_launch_knob_values(cls, engine):
        """``dr``, ``dz`` and ``zmax`` over the frequencies marched, and on
        rams0.5 ``rams_rotation_angle_deg`` (the Padé angle each grid marched;
        the other backends' decks do not read it)."""
        grids = engine.grids
        values = {'dr': [grid.dr for grid in grids],
                  'dz': [grid.dz for grid in grids],
                  'zmax': [grid.zmax for grid in grids]}
        if engine.backend == 'rams':
            values['rams_rotation_angle_deg'] = [grid.theta_deg for grid in grids]
        return values

    def __init__(
        self,
        *,
        executable: Optional[Path] = None,
        dr: Optional[float] = None,
        dz: Optional[float] = None,
        zmax: Optional[float] = None,
        n_pade: int = DEFAULT_N_PADE,
        n_stability: Optional[int] = None,
        stability_range: Optional[float] = None,
        q_factor: Optional[float] = None,
        record_duration: Optional[float] = None,
        depth_decimation: int = 1,
        earth_curvature: bool = False,
        absorber_width_wavelengths: float = 20.0,
        absorber_attenuation: Optional[float] = None,
        n_sediment_points: Optional[int] = None,
        c0: Optional[float] = None,
        timeout: float = DEFAULT_RUN_TIMEOUT_S,
        # ``dr`` / ``dz`` are picked by the Lytaev (2023) Padé-error
        # optimizer when not set explicitly. ``accuracy`` is the per-run
        # error budget; ``angle_max_deg`` (degrees) bounds the PE spectrum.
        accuracy: Optional[float] = None,
        angle_max_deg: Optional[float] = None,
        # rams0.5 only — ignored on the fluid backends.
        # ``rams_rotation_angle_deg`` is the Padé rotation angle (degrees, 0–90)
        # rams0.5 reads as ``theta`` for elastic stability; defaults are
        # tuned against Kraken on the Pekeris-elastic problem.
        # ``rams_rotation`` is its rotation flag ``irot`` (1 = on).
        rams_rotation_angle_deg: Optional[Union[float, Callable[[float], float]]] = None,
        rams_rotation: Optional[bool] = None,
        # Divisor of the Lytaev-optimised ``dr`` for the ``rams`` backend,
        # beside the ``c_min/(5·f)`` λ cap ``grid.tighten_rams_dr`` applies;
        # the tighter of the two wins.
        rams_dr_factor: Optional[float] = None,
        backend: Optional[str] = None,
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
            Path to the binary of the backend named by ``backend=`` — the
            s_mpiram binary when ``backend=None``. A relative path is resolved
            against the working directory at construction. With
            ``backend=None`` a run that dispatches to a Collins backend
            raises ``ConfigurationError`` (each binary reads its own deck), so
            pin ``backend=`` together with a Collins build. Auto-detected if
            None. **[all backends]**
        dr : float, optional
            Range step in meters. Default: None (selected by the Lytaev
            (2023) Padé-error optimizer; see ``accuracy`` / ``angle_max_deg``).
            On mpiramS this is the *longest* step the march takes: mpiramS
            lands exactly on each requested output range, so a leg shorter
            than ``dr`` is marched in one step of that leg's length. The
            Collins codes take at most one profile section, bathymetry node
            and altimetry node per step (``ramgeo1.5.f:348, :359``) and
            the manual bounds ``dr`` by the smallest region (``ram.pdf``
            p.8), so a pinned ``dr`` wider than a region of the environment
            is refused (:func:`collins.regions_the_march_cannot_take`), and
            uacpy's own intermediate SSP sections are thinned to fit it.
            **[all backends]**
        dz : float, optional
            Depth step in meters. Default: None (selected by the Lytaev
            optimizer, then snapped so the shallowest bathymetry point sits
            ``SEAFLOOR_CELL_OFFSET`` (a quarter) of a cell below a depth grid
            node on the fluid backends and on a node on rams, capped at
            ``MAX_DEPTH_POINTS`` grid points, and capped at λ_s/14 on rams;
            the search starts at c_min/(15·freq), ``LAMBDA_PER_DZ_FLOOR``,
            where the optimiser asks for finer, and refines below it until the
            seabed's trapped modes score under ``TRAPPED_MODE_SCORE_LIMIT``). A pinned ``dz`` is used as given; on the
            fluid backends (mpiramS, ramgeo, ramsurf) one that puts the
            shallowest seafloor more than ``SEAFLOOR_QUARTER_TOLERANCE`` of a
            cell from the quarter-cell placement — on a node, as round values
            such as 1, 0.5 or 0.25 m do on a 100 m channel, or anywhere from
            mid-cell to the next node — warns and names the aligned value
            ``h/(n + SEAFLOOR_CELL_OFFSET)``, which the automatic grid would
            use (the placement and its tolerance are uacpy's guidance,
            measured in docs/models/ram.md). On ramsurf the automatic ``dz`` also puts the
            depressed surface at ``r = 0`` on a depth node: ramsurf1.5 zeroes
            every row down to ``int(1 + zsrf/dz)``, so its pressure-release
            surface sits at ``floor(zsrf/dz)·dz``, up to one cell above the one
            asked for. **[all backends]**
        zmax : float, optional
            PE domain depth (m). None = auto (seafloor + absorbing layer).
            Default: None.
        n_pade : int, optional
            Number of Padé terms (2-10). Default: 8
            (:data:`~uacpy.models.pe_grid.DEFAULT_N_PADE`, the order the
            grid functions score by default) — Collins' sample deck
            (``np = 8``, ram.pdf p.12 Fig. 2;
            ``third_party/mpiramS/doc/RAM.ORIG/ram.in``), the ``[7/8]``
            order of every table in Lytaev (2023), and the Acoustics
            Toolbox's RAM writer (``Matlab/Slicer/write_ram_in.m:82``,
            ``np = 8``). Every binary finds the Padé roots with a Laguerre
            iteration that stops when it does not converge
            (``ramgeo1.5.f:767-771``, ``mpiramS/src/epade.f90:296-301``);
            at order 8 that happened on a few small ``k0·dr`` steps a pinned
            ``dr`` marched (36.5 Hz at 0.5 m and 75 Hz at 1 m on ramgeo, a
            36.5 Hz bin at 2 m on mpiramS, where order 6 ran), and the run
            raises naming the order and the step; two fewer terms ran every
            such stop measured. **[all backends]**
        n_stability : int, optional
            Number of stability terms, written as given (``0`` turns the
            constraints off). Default ``None``: 1 — Collins' guide, "We
            have found that ns = 1 or 2 is effective for most problems"
            (ram.pdf p.3), and his sample deck's ``ns = 1``.
            **[mpiramS, ramgeo1.5, ramsurf1.5]** — rams0.5's row 5 carries
            (rams_rotation, rams_rotation_angle_deg) in these two slots instead
            and its ``epade`` call takes ``ns = 1`` itself
            (``rams0.5.f:176``), so a pinned value there gives a notice.
        stability_range : float, optional
            Stability range in meters — the range past which the ``n_stability``
            terms are switched off. Default: max output range on mpiramS;
            on the Collins fluid codes ``None`` writes 0, which the binary
            then expands to ``2 × rmax`` (``if(rs.lt.dr) rs = 2.0*rmax``).
            Either way the constraints act at every range, the RAM guide's
            default: "When rs is set to 0, the stability constraints are used
            for all ranges" (``ram.pdf`` p.7); on mpiramS ``rs = rmax`` makes
            ``rsc = rmax − rs = 0`` (``mpiramS/src/ram.f90:68-69``).
            **[mpiramS, ramgeo1.5, ramsurf1.5]**

            The Collins codes test it against the absolute range
            (``ramgeo1.5.f:368``). mpiramS tests the distance left to the
            *current output range* (``ram.f90:251``), so it acts at the range it
            names only when the receiver carries a single range; on a multi-range
            grid the switch-off point is set by the range spacing and the value
            is largely inert, which :meth:`RAM.run` warns about.
        q_factor : float, optional
            Q value for broadband mode (half-bandwidth = fc/q_factor, so the band
            spans 2·fc/q_factor). ``1/q_factor`` is the HALF fractional width, where
            Bellhop's ``bandwidth_factor`` is the full one: the same band is
            ``q_factor = 2/bandwidth_factor``. Default: ``None``, resolved by
            :func:`_band.resolve_broadband_grid` from what the source carries:

            * a multi-frequency source or ``run(frequencies=…)`` — the
              array's own half-width, ``fc / ((n//2 + ½)·Δf)``; a pinned
              ``q_factor`` there must give the same sweep around the
              array's centre bin (the run refuses one that does not, the
              two settings contradicting each other);
            * a TIME_SERIES grid derived from the pulse — the grid's own
              half-width, which a pinned ``q_factor`` replaces;
            * a single frequency with ``record_duration`` pinned —
              ``2/DEFAULT_BROADBAND_BANDWIDTH_FACTOR`` (4), the half-width of the
              default band every engine shares, ``fc·(1 ± 1/4)``;
            * a single-frequency Source on BROADBAND / TIME_SERIES with neither
              pinned — ``run()`` first expands it to the default band shared by
              every engine (``DEFAULT_BROADBAND_N_FREQUENCIES`` bins over
              ``fc·(1 ± DEFAULT_BROADBAND_BANDWIDTH_FACTOR/2)``), then derives
              ``q_factor`` from that array as above;
            * an explicit ``run(frequencies=[fc])``, and every COHERENT_TL
              run — ``1e6``, which collapses the band to one bin so mpiramS
              doesn't sweep ~500 frequencies per call: a band narrower than one
              bin (``fc/q_factor < 1/record_duration``) marches ``fc`` alone (``peramx.f90:362-370``,
              a UACPY patch; :func:`_band.broadband_frequencies` applies the same
              rule to the Collins loop). A COHERENT_TL run marches the source
              frequency alone, so a pinned ``q_factor`` / ``record_duration``
              is not used there and the run gives a notice naming it.

            Used by every backend's broadband mode to derive the frequency
            vector — mpiramS internally, Collins backends as the Python-side
            frequency-loop grid.
        record_duration : float, optional
            Time window width in seconds (broadband resolution df = 1/record_duration).
            Default: ``None``, resolved alongside ``q_factor`` above: ``1/Δf`` for a
            frequency array (a pinned ``record_duration`` there must equal
            it — the run refuses one that does not) and for a pulse-derived
            TIME_SERIES grid (which a pinned value replaces), for a single
            frequency with ``q_factor`` pinned
            the default band's own bin spacing,
            ``(DEFAULT_BROADBAND_N_FREQUENCIES - 1) / (fc·DEFAULT_BROADBAND_BANDWIDTH_FACTOR)``,
            and ``1.0`` for the collapsed single-bin band and every
            COHERENT_TL run.
        depth_decimation : int, optional
            Output depth decimation factor. Default: 1 (no decimation), as
            Collins' sample deck (``ram.pdf`` p.12-13: "ndz depth decimation
            factor for tl.grid (1=no decimation)").
            **[all backends]**
        earth_curvature : bool, optional
            Apply the flat-earth transformation (Earth-curvature correction of
            the water column's depths and speeds, the bathymetry, altimetry and
            source depth before the PE marches in range). Default: False —
            off by default so RAM models the flat Earth every other uacpy engine
            models, and cross-model comparisons compare one ocean (uacpy
            guidance; the vendored mpiramS sample deck also runs flat,
            ``mpiramS/in.pe``: "0 ! flat earth transform"). Earth
            curvature is real physics: it matters beyond about 56 km and moves the
            first convergence zone about 1 % closer, in better agreement with
            measured data (Etter, *Underwater Acoustic Modeling and Simulation*,
            sect. 5.5.2) — turn it on for long-range work.
            **[all backends]** — mpiramS applies it inside the binary
            (``peramx.f90:268-281``); the Collins binaries have no flag, so uacpy
            applies the same formulas to their decks
            (:func:`_domain.deck_water_column`) and un-maps their output depth
            axis.
        absorber_width_wavelengths : float, optional
            Width of the absorbing layer at the base of the PE domain (below
            the modelled sediment stack and a pad of real seabed), in
            wavelengths ``c0/f`` of the PE reference speed — at ``freq_min`` on
            the broadband paths, the longest wavelength in the band.
            Default: 20.0, uacpy guidance, measured on lossless seabeds
            (:func:`_domain.adequate_zmax`); far thicker than the RAM guide's
            "the attenuation is increased over the lower few wavelengths of
            the grid" (``ram.pdf`` p.7; about 3 wavelengths in Collins' sample
            deck) or Jensen et al.'s H/3 (*Computational Ocean Acoustics*,
            Sect. 6.5.3). **[all backends]**
        absorber_attenuation : float, optional
            Attenuation at the floor of the absorbing layer
            (dB/wavelength). The seabed keeps its own attenuation down to
            the top of the absorbing layer — below the sediment stack and a
            pad of real seabed — and only the deepest
            ``absorber_width_wavelengths`` wavelengths are ramped
            linearly from it to this value at the domain floor. Default
            ``None``: 10 dB/wavelength, the floor of Collins' sample ramp
            ``900.0 0.5`` / ``1000.0 10.0`` (``ram.pdf`` p.12, Fig. 2;
            upstream mpiramS used 5, ``third_party/MODIFICATIONS.md:2431-2432``),
            raised to the seabed's own attenuation at the top of the layer
            where that is higher, so the ramp never decreases ("the
            attenuation is increased over the lower few wavelengths",
            ``ram.pdf`` p.7). A set value is written as given; one below
            the seabed half-space's attenuation, which makes the ramp
            decrease, gives one notice.
        n_sediment_points : int, optional
            Number of sediment-profile control points (at least 4), written
            to the deck as given. Default ``None``: enough points to hold
            the interval below to one depth cell, ``ceil(sedlayer/dz) + 3``,
            and at least :data:`mpirams.DEFAULT_N_SEDIMENT_POINTS` (1000,
            uacpy guidance). **[mpiramS]** — ``profl`` lays them out as [sea surface, seafloor,
            ``n_sediment_points-3`` interior points spanning the seabed down to the top
            of the absorbing layer, domain floor] and interpolates linearly between
            them (``mpiramS/src/ram.f90:334-350``), so a material step is resolved
            to ``sedlayer/(n_sediment_points-3)`` where ``sedlayer`` is that span
            (:func:`_domain.absorber_span`). The default keeps that interval far
            below an acoustic wavelength for ordinary geometries: at 50 points a
            200 m span resolves its steps only to 4.3 m (measured: on a 100 m
            guide over a 60 m layer at 800 Hz, 1000 points moved the field
            1.04 dB rms between zmax=700 m and 400 m, one point per depth
            cell 0.16 dB). Collins backends ignore
            this — they consume the layered bottom as Collins-style ``(depth,
            value)`` breakpoints (see ``_seabed.piecewise_breakpoints``).
        c0 : float, optional
            PE reference sound speed (m/s). ``c0`` is the *algorithmic*
            expansion point of the parabolic equation (the speed factored
            out as ``exp(ik₀x)``), **not** a physical input. ``None``
            (default) → uacpy resolves it via Lytaev Eq. (15), the c₀
            that centres the spectrum ``[ξ_min, ξ_max]`` around 0 to
            minimise the Padé approximation error (Collins' own sample uses
            1600 m/s "to obtain an accurate rational approximation for phase
            speeds between 1500 and 1700 m/s", ``ram.pdf`` p.8). All four
            backends (mpiramS, ramgeo, rams, ramsurf) honour the resolved value.
            Pass an explicit float to override.
        timeout : float, optional
            Subprocess timeout (s) for each binary launch, on every
            backend. Default: 600.0.
        use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
            Standard plumbing (see :class:`PropagationModel`).
        accuracy : float, optional
            Lytaev optimiser's per-run accuracy budget (max
            ``|τ · n_steps|``, the accumulated error bound of the steepest
            scored component). Default ``None``: the default chooser
            targets 1e-3 (Lytaev 2023 §5.1: "In all the examples, target
            accuracy ε is set to 10⁻³") and treats it as a target, not a
            bound — it relaxes ``ε`` when no grid meets it, starts the depth
            search at a cost floor (``λ_p/15``, :data:`LAMBDA_PER_DZ_FLOOR`)
            that sits above the optimiser's own ``dz`` for most ordinary
            frequencies, refines from there only as far as the seabed's
            trapped modes require (:data:`TRAPPED_MODE_SCORE_LIMIT`), and
            stops at :data:`MAX_DEPTH_POINTS` — each reported as a log line.
            A pinned ``accuracy`` is met as given
            (:func:`grid.grid_for_pinned_accuracy`): the coarsest grid on
            the Lytaev ladders whose score stays under it, a pinned ``dr``
            or ``dz`` held while the other is searched, with no relaxation,
            floor or cap (a grid past :data:`MAX_DEPTH_POINTS` is marched
            with a cost notice; a Collins depth grid past the binary's
            ``mz`` is refused). The run refuses an ``accuracy`` no grid on
            the ladders meets, and says so when ``dr`` and ``dz`` are both
            pinned, which leaves ``accuracy`` nothing to choose.
        angle_max_deg : float, optional
            Maximum propagation angle (degrees, within (0, 90)) bounding the
            PE spectrum for the Lytaev optimiser and for Eq. (15)'s ``c₀``,
            used as given. Default ``None``: Lytaev (2023) §5.5's
            ``θ_max = max(θ_max^src, θ_max^bottom)`` with the source term
            30° (:data:`_domain.DEFAULT_ANGLE_MAX_DEG`, the aperture of
            Lytaev's §5.1 example)
            and the seabed term the steepest slope of ``env.bathymetry``
            (:func:`_domain.resolve_angle_max`); when no grid meets the
            default accuracy target the default chooser may narrow it to
            20° and then 15°, with a warning. A pinned value narrower than
            the steepest slope gives a notice citing §5.5, and is never
            narrowed.
        rams_rotation_angle_deg : float or callable, optional
            Padé rotation angle in degrees for elastic stability (0 <= theta
            <= 90). ``None`` (default) takes :data:`RAMS_DEFAULT_THETA_DEG`
            (45°, tuned at order 6 against Kraken on the Pekeris-elastic
            scenario in tests/test_cross_model_agreement.py — at the default
            order 8 that scenario's default grid reads 0.528 dB rms from
            Kraken over 1-8 km, against 0.526 at 6; Milinazzo, Zala & Brooke
            (1997) Fig. 11 find 45° in line with their reference from four
            Padé terms up) unless that angle's own growth
            on the steepest propagating components outruns what the seabed leaks
            (``pe_grid.rams_growth_margin``) — then the widest angle on
            :data:`~uacpy.models.pe_grid.RAMS_THETA_LADDER_DEG` that is
            stable is taken, per frequency, with a ``FallbackWarning`` naming it;
            when no angle on the ladder is stable the run is refused. A
            float pins the angle for every frequency and is marched as
            given — an angle the rule finds unstable gives a
            ``NumericsWarning`` naming the stable one; a callable
            ``theta_fn(frequency) -> float`` varies it across a broadband
            run. The run record holds a callable as its repr, so
            :meth:`from_run_settings` refuses that record: rebuild the model
            with the callable. **[rams0.5]**
        rams_rotation : bool, optional
            Rotate the Padé branch cut (rams0.5's ``irot``). Default
            ``None``: on (``DEFAULT_RAMS_ROTATION``), uacpy guidance: no
            source prefers the rotated coefficients (Milinazzo et al. 1997
            take Collins' complex coefficients as their reference); the 45°
            tuning and the rams stability rule were measured with rotation
            on. A set value, ``True`` included, on a backend other than
            rams0.5 gives the not-used notice.
            **[rams0.5]**
        rams_dr_factor : float, optional
            Divisor (≥ 1) of the Lytaev-optimised ``dr`` for the rams
            backend (rotated Padé, Milinazzo-Zala-Brooke 1997), applied
            with an independent ``dr ≤ c_min/(5·f)`` λ cap and the
            stability rule; the tightest wins (:func:`grid.tighten_rams_dr`).
            Default ``None``: 5.0 (:data:`grid.DEFAULT_RAMS_DR_FACTOR`, uacpy
            guidance). A set value is used as given; ``1.0`` leaves the cap
            and the rule alone. It divides the automatic ``dr`` only, so
            beside a pinned ``dr`` — or on another backend — it gives a
            notice that it is not used. **[rams0.5]**
        backend : str, optional
            Force a specific RAM-family backend instead of automatic
            dispatch: ``'mpirams'``, ``'ramgeo'``, ``'rams'`` or
            ``'ramsurf'``. ``None`` (default) auto-selects from the
            environment (see :meth:`select_backend`; uacpy guidance, routing
            by what each code can represent — ``ramgeo1.5.f:3-4``,
            ``ramsurf1.5.f:3-11``). A forced backend
            that cannot represent the environment (e.g. a fluid backend
            for an elastic bottom) raises ``ConfigurationError`` at run
            time.
        """
        super().__init__(**_host_options(locals()))

        # Run modes, capability flags and collapse defaults come from the
        # class-level ``spec`` (applied by PropagationModel.__init__).
        self.backend = backend
        self.dr = dr
        self.dz = dz
        self.zmax = zmax
        self.n_pade = n_pade
        self.n_stability = n_stability
        self.stability_range = stability_range
        self.q_factor = q_factor
        self.record_duration = record_duration
        self.depth_decimation = depth_decimation
        self.earth_curvature = earth_curvature
        self.absorber_width_wavelengths = absorber_width_wavelengths
        self.absorber_attenuation = absorber_attenuation
        self.n_sediment_points = n_sediment_points
        self.c0 = c0
        # ``rams_rotation_angle_deg`` is either a float (used for every frequency) or
        # a callable ``theta_fn(frequency) -> float`` resolved per
        # frequency by ``_stability.theta_for_freq``.
        self.rams_rotation_angle_deg = rams_rotation_angle_deg
        self.rams_rotation = rams_rotation
        self.rams_dr_factor = rams_dr_factor
        # Kept unset when not given so ``copy()`` round-trips: materialising
        # the default here would make every copy look caller-pinned.
        # Resolved on use via ``_accuracy`` / ``_accuracy_explicit``.
        self.accuracy = accuracy
        self.angle_max_deg = angle_max_deg
        self._check_knobs()

        if accuracy is not None:
            self.accuracy = float(accuracy)
        if angle_max_deg is not None:
            self.angle_max_deg = float(angle_max_deg)
        if rams_rotation_angle_deg is not None and not callable(rams_rotation_angle_deg):
            self.rams_rotation_angle_deg = float(rams_rotation_angle_deg)
        if rams_dr_factor is not None:
            self.rams_dr_factor = float(rams_dr_factor)
        # A NumPy integer passes the checks; the decks and the run record
        # carry the Python int.
        for name in ('n_pade', 'n_stability', 'depth_decimation',
                     'n_sediment_points'):
            if getattr(self, name) is not None:
                setattr(self, name, int(getattr(self, name)))

        # The backend name is checked (:meth:`_check_knobs`) before any
        # binary is looked up, and the lookup follows it: a forced Collins
        # backend never executes s_mpiram, so demanding it on disk would
        # refuse a configuration that runs. Auto dispatch (``backend=None``)
        # can reach mpiramS, so that case resolves it here. The remaining
        # Collins binaries resolve per-run via ``_collins_binary``.
        self._exe = self._resolve_executable(
            executable,
            lambda: (
                self._collins_binary(backend)
                if backend is not None and backend != 'mpirams'
                else self._find_binary('mpirams')
            ),
            label=f"RAM:{backend or 'mpirams'}",
        )

        # Warn on low absorbing-layer attenuation: values < 1 dB/wavelength
        # let bottom reflections leak back into the PE domain and contaminate
        # the field. The 1 dB/wavelength threshold is uacpy guidance; the
        # "5-10" the message names are upstream mpiramS's 5
        # (third_party/MODIFICATIONS.md:2431-2432) and the floor of Collins'
        # sample ramp, 10 (ram.pdf p.12, Fig. 2).
        if (self.absorber_attenuation is not None
                and self.absorber_attenuation < 1.0):
            warnings.warn(
                f"RAM absorber_attenuation={self.absorber_attenuation} "
                "dB/wavelength is low; spurious reflections from the PE "
                "domain bottom may contaminate the field. Typical values "
                "are 5-10 dB/wavelength.",
                NumericsWarning,
                skip_file_prefixes=USER_FRAME_SKIP
            )

    def _check_knobs(self) -> None:
        """Refuse a constructor knob no run could use: an unknown backend,
        a Padé order outside the binaries' bound, a non-positive or
        non-finite grid step, window, timeout, absorbing layer or
        ``accuracy``, an ``angle_max_deg`` outside (0, 90) degrees, a sediment
        grid too small to lay out, a ``rams_rotation_angle_deg`` outside [0, 90] degrees,
        a ``rams_rotation`` that is not a bool and a ``rams_dr_factor``
        below 1. A count may be a Python or a NumPy integer.

        Run at construction and again by every run (stage 2,
        :meth:`_check_carriers`), since the attributes can be reassigned in
        between. Garbage scalars
        fail here with a clear Python error instead of a Fortran array-bound
        or divide-by-zero crash 30 seconds into a binary call.
        """
        if self.backend is not None and self.backend not in self._BACKENDS:
            raise ConfigurationError(
                f"RAM(backend={self.backend!r}) is not a known backend. "
                f"Choose one of {sorted(self._BACKENDS)}, or None for "
                f"automatic dispatch."
            )
        # mpiramS allocates its Padé arrays dynamically (epade.f90:30); the
        # Collins binaries carry ``parameter (mp=10)`` and stop above it
        # (``ramgeo1.5.f:142-145``). The bound below is the binaries' own, so
        # n_pade=9 and 10 — legal, and the widest-angle operators available —
        # are accepted rather than refused by a margin inside it.
        n_pade = self.n_pade
        if not is_whole_number(n_pade) or not (2 <= n_pade <= 10):
            raise ConfigurationError(
                f"n_pade must be an integer in [2, 10] (Collins mp=10, "
                f"ramgeo1.5.f:142-145); got {n_pade!r}."
            )
        for name in ('dr', 'dz', 'zmax', 'stability_range', 'c0', 'q_factor',
                     'record_duration', 'accuracy'):
            positive_finite(name, getattr(self, name), optional=True)
        # The source aperture θ bounds the Padé spectrum at ξ = -sin²θ: the
        # bracket is undefined at 90° (``_domain.resolve_angle_max`` caps the
        # seabed term at 89°) and empty at 0°.
        angle_max_deg = self.angle_max_deg
        positive_finite('angle_max_deg', angle_max_deg, optional=True)
        if angle_max_deg is not None and not angle_max_deg < 90.0:
            raise ConfigurationError(
                f"angle_max_deg is the source's maximum propagation angle in "
                f"degrees from the horizontal, within (0, 90); got "
                f"{angle_max_deg!r}.")
        for name in ('timeout', 'absorber_width_wavelengths'):
            positive_finite(name, getattr(self, name))
        positive_finite('absorber_attenuation', self.absorber_attenuation,
                        optional=True)
        # ``profl`` lays out the sub-bottom as [surface, seafloor, nzs-3
        # interior points, domain floor] and only builds interior points when
        # nzs > 3 (mpiramS/src/ram.f90:334-342).
        if self.n_sediment_points is not None:
            whole_count('n_sediment_points', self.n_sediment_points, 4)
        whole_count('depth_decimation', self.depth_decimation, 1)
        if self.n_stability is not None:
            whole_count('n_stability', self.n_stability, 0)
        rams_rotation_angle_deg = self.rams_rotation_angle_deg
        if rams_rotation_angle_deg is not None and not callable(rams_rotation_angle_deg):
            if not is_real_number(rams_rotation_angle_deg):
                raise ConfigurationError(
                    f"rams_rotation_angle_deg must be an angle in degrees or a "
                    f"callable theta_fn(frequency) -> degrees; got "
                    f"{rams_rotation_angle_deg!r}.",
                    remediation=(
                        "A run record holds a callable rams_rotation_angle_deg "
                        "as its repr, which no run can call, so "
                        "RAM.from_run_settings cannot rebuild it: build "
                        "the model with the callable itself, "
                        "RAM(..., rams_rotation_angle_deg=theta_fn)."))
            theta_val = float(rams_rotation_angle_deg)
            if not (0.0 <= theta_val <= 90.0):
                raise ConfigurationError(f"rams_rotation_angle_deg must be in [0, 90] degrees; "
                                         f"got {theta_val!r}.")
        if self.rams_rotation is not None and not isinstance(
                self.rams_rotation, (bool, np.bool_)):
            raise ConfigurationError(
                f"rams_rotation must be True, False or None; got "
                f"{self.rams_rotation!r}.")
        factor = self.rams_dr_factor
        if factor is not None and (not is_real_number(factor)
                                   or not np.isfinite(factor)
                                   or factor < 1.0):
            raise ConfigurationError(
                f"rams_dr_factor must be ≥ 1.0; got "
                f"{factor!r}. It divides the Lytaev dr on rams; use 1.0 "
                f"to march that dr (still under the λ/5 cap)."
            )

    @property
    def _accuracy(self) -> float:
        """The Lytaev accuracy target actually used."""
        return (DEFAULT_RAM_ACCURACY if self.accuracy is None
                else float(self.accuracy))

    @property
    def _accuracy_explicit(self) -> bool:
        """True when the caller pinned ``accuracy``.

        A pinned accuracy is met as given
        (:func:`grid.grid_for_pinned_accuracy`); left unset, the default
        chooser may relax uacpy's own target and reports it.
        """
        return self.accuracy is not None

    def _knob_record(self) -> RamKnobs:
        """This model's knobs as :class:`RamKnobs`."""
        return RamKnobs(
            model_name=self.model_name,
            backend=self.backend,
            q_factor=self.q_factor,
            record_duration=self.record_duration,
            accuracy=self._accuracy,
            accuracy_pinned=self._accuracy_explicit,
            absorber_attenuation=self.absorber_attenuation,
            absorber_width_wavelengths=(
                self.absorber_width_wavelengths),
            depth_decimation=self.depth_decimation,
            earth_curvature=self.earth_curvature,
            n_pade=self.n_pade,
            n_sediment_points=self.n_sediment_points,
            n_stability=self.n_stability,
            rams_dr_factor=self.rams_dr_factor,
            rams_rotation=(DEFAULT_RAMS_ROTATION
                           if self.rams_rotation is None
                           else bool(self.rams_rotation)),
            rams_rotation_angle_deg=self.rams_rotation_angle_deg,
            stability_range=self.stability_range,
            angle_max_deg=self.angle_max_deg,
            dz=self.dz,
            dr=self.dr,
            zmax=self.zmax,
            c0=self.c0)

    # ── stages 1-2: the environment ─────────────────────────────────────

    def _requested_frequencies(self, mode, source, frequencies, time):
        """The frequencies a RAM band run is asked for (Hz): the shared rule
        (``frequencies=``, the grid a TIME_SERIES pulse implies, a lone
        ``fc`` expanded to the default band every engine shares), except
        that a lone ``fc`` with ``q_factor`` or ``record_duration`` pinned
        stays ``[fc]`` — the centre of the ``(fc, q_factor,
        record_duration)`` sweep those knobs describe
        (:func:`_band.resolve_broadband_grid`), whose bins
        ``settings.engine.marched_frequencies`` lists and
        :meth:`_marched_frequencies` records as the run's grid, and that the
        grid a
        TIME_SERIES pulse implies — a default — is rebuilt on those pins
        (:func:`_band.sweep_on_the_pins`: a pinned knob replaces what it
        names, the pulse's grid supplies the other). The shared rule's
        notice is kept
        whichever grid is returned, except the pulse grid's, which
        describes a grid the pins replaced."""
        band = super()._requested_frequencies(mode, source, frequencies,
                                              time)
        if mode not in _BAND_MODES or frequencies is not None:
            return band
        src = np.atleast_1d(np.asarray(source.frequencies, dtype=float))
        pinned = (self.q_factor is not None
                  or self.record_duration is not None)
        if mode == RunMode.TIME_SERIES and band.frequencies is not None:
            if src.size == 1 and pinned:
                # The grid the pulse implies is a default: the pins
                # replace what they name, the pulse's grid the rest.
                return BandResolution(sweep_on_the_pins(
                    band.frequencies, knobs=self._knob_record()))
            # The synthesis reads H(f) on an ascending grid and returns
            # p(t), so the source's bins are marched in increasing order.
            return BandResolution(np.sort(band.frequencies), band.notice)
        if src.size == 1 and pinned:
            return BandResolution(src, band.notice)
        if mode == RunMode.TIME_SERIES:
            # A pulse too short to imply a grid runs on the source's own
            # frequencies, a lone one expanded like BROADBAND's.
            if src.size > 1:
                return BandResolution(src, band.notice)
            return BandResolution(
                broadband_band(source, *self._broadband_band_knobs(),
                               model_name=self.model_name).frequencies,
                band.notice)
        return band

    def _marched_frequencies(self, settings):
        """What a band run's result carries: the requested bins
        (``settings.engine.requested_frequencies``, mpiramS's trim of its
        sweep) or, when none are requested, every bin the engine marches
        (``settings.engine.marched_frequencies``) — the whole ``(fc,
        q_factor, record_duration)`` sweep of a lone ``fc`` with a pinned
        ``q_factor`` / ``record_duration``. A single-frequency run carries
        the requested grid."""
        engine = settings.engine
        if engine is None or settings.mode not in _BAND_MODES:
            return settings.frequencies
        if engine.requested_frequencies is not None:
            return np.asarray(engine.requested_frequencies, dtype=float)
        return np.asarray(engine.marched_frequencies, dtype=float)

    def _project_environment(self, env: Environment, *,
                             request=None) -> Environment:
        """The shared projection, then the sea surface made pressure-release
        (:func:`_domain.collapse_surface_to_pressure_release`): no RAM deck
        carries a surface record, so the environment every backend runs on
        has a vacuum surface."""
        return collapse_surface_to_pressure_release(
            super()._project_environment(env, request=request))

    def _validate_engine(self, env, source, receiver, run_mode, *,
                         request=None) -> None:
        """The RAM-specific bottom-type guard and the mpiramS output-range
        spacing check.

        Every RAM-family deck expresses the seabed as fluid geoacoustic
        values — a half-space or sediment layers (plus shear on rams0.5) —
        and none has a spelling for a vacuum, rigid or tabulated-reflection
        boundary. Writing one of those would silently model whatever
        placeholder geoacoustics the ``BoundaryProperties`` constructor left
        in place, so they are refused here instead.
        """
        for col in env.bottom.columns:
            kind = col.halfspace.acoustic_type
            if not BoundaryType.from_string(kind).is_geoacoustic:
                raise UnsupportedFeatureError(
                    'RAM',
                    f"a bottom with acoustic_type={kind!r} — the RAM decks "
                    f"express the seabed as fluid geoacoustic layers over a "
                    f"half-space, and this type has no layer spelling",
                    alternatives=[
                        'a geoacoustic half-space / layered bottom',
                        "Scooter / Kraken / Bellhop, which take "
                        "'rigid'/'file' natively",
                    ],
                )

        if self.select_backend(env, run_mode) == 'mpirams':
            check_mpirams_output_range_spacing(receiver)

    # ── stage 3: the settings of the run ─────────────────────────────────

    def _resolve_engine_settings(self, env, source, receiver, settings, *,
                                 given_env, request=None) -> RamSettings:
        """Stage 3: the backend and every PE grid the run marches, one per
        binary launch, resolved once from the projected ``env`` — with the
        notices about the grid and the refusals of a grid a binary cannot
        run (a source row no solver writes, a Collins array overrun, a
        sediment block ``zread`` cannot represent, a rotation no step keeps
        stable, ...).

        A band run is resolved for the frequencies ``settings`` holds
        (:meth:`_requested_frequencies`). Every source depth shares the
        grid: ``dz`` is capped by the shallowest, and each depth is checked
        against it.
        """
        collected = []
        engine = self._resolve_ram_settings(env, source, receiver, settings,
                                            notices=collected)
        return dataclasses.replace(engine, notices=tuple(collected))

    def _resolve_ram_settings(self, env, source, receiver,
                              settings, *, notices=None) -> RamSettings:
        """The body of :meth:`_resolve_engine_settings`, its notices going
        to ``notices`` (:func:`~uacpy.models._notices.give_notice`)."""
        mode = settings.mode
        knobs = self._knob_record()
        backend = self.select_backend(env, mode)
        self._log(
            f"Dispatching to {backend} backend "
            f"(elastic_bottom={env.bottom.is_elastic}, "
            f"altimetry={env.altimetry is not None})"
        )
        seafloor = float(env.bathymetry.eval(range=0.0))
        buried = np.atleast_1d(np.asarray(source.depths, dtype=float))
        buried = buried[buried > seafloor]
        if buried.size:
            # A fact of the inputs, said at info: every RAM self-starter
            # plants the source on any depth row and solves the sediment
            # rows too (_check_source_depths), so the field is computed.
            self._log(
                f"source depth(s) {', '.join(f'{z:g}' for z in buried)} m "
                f"lie below the {seafloor:g} m seafloor at r = 0: RAM's "
                f"self-starter plants the source in the sediment and the "
                f"march computes its field there.")
        self._warn_on_mpirams_only_overrides(backend, notices=notices)
        if backend != 'mpirams':
            # A pinned executable of another backend is refused before any
            # grid is resolved.
            self._collins_binary(backend)
        warn_if_pinned_dz_misplaces_the_seafloor(env, backend,
                                                 notices=notices,
                                                 knobs=knobs)
        warn_if_pinned_accuracy_is_unused(notices=notices,
                                          knobs=knobs)
        warn_if_pinned_rams_dr_factor_is_unused(backend, notices=notices,
                                                knobs=knobs)
        warn_if_pinned_angle_max_is_below_the_slope(
            env, notices=notices, knobs=knobs)
        warn_if_pinned_absorber_is_below_the_seabed(
            env, notices=notices, knobs=knobs)
        band = mode in _BAND_MODES
        if band:
            source = dataclasses.replace(
                source, frequencies=np.array(settings.frequencies, dtype=float))
        else:
            warn_if_band_pins_unused_on_tl(notices=notices,
                                           knobs=knobs)
        resolve = (self._resolve_mpirams_grids if backend == 'mpirams'
                   else self._resolve_collins_grids)
        parts = resolve(env, source, receiver, mode, backend, notices=notices)
        warn_if_receivers_see_paths_steeper_than_the_band(
            env, source, receiver, notices=notices,
            knobs=knobs)
        if backend == 'ramsurf':
            warn_if_ramsurf_surface_unresolved(env, parts['grids'][0].dz,
                                               notices=notices,
                                               knobs=knobs)
        return RamSettings(
            backend=backend,
            backend_origin=backend_origin(env, backend,
                                          knobs=knobs),
            c0=resolve_c0(env, knobs=knobs),
            c0_origin=('RAM(c0=…)' if self.c0 is not None
                       else 'Lytaev (2023) Eq. (15)'),
            **parts,
        )

    def _check_source_depths(self, env, source) -> float:
        """The deepest modelled interface (:meth:`_max_receiver_depth`),
        below which receivers warn; ``Source`` itself refuses a negative
        depth.

        A source below the seafloor is not refused here: every RAM
        self-starter plants the delta at ``si = 1 + zs/dz`` on any row of
        the depth grid and solves the full depth operator, sediment rows
        included (``ramgeo1.5.f:389-393``, ``ramsurf1.5.f:396-400``,
        ``rams0.5.f:357-361``, ``mpiramS/src/ram.f90:110-114``), so the PE
        computes a buried source's field. Stage 3 sizes the automatic
        domain to hold it (:func:`_domain.adequate_zmax`) and refuses only
        a source past the grid the binary marches
        (:func:`_domain.check_source_inside_the_domain`).
        """
        return self._max_receiver_depth(env)

    _BACKENDS = tuple(_BINARIES)

    def select_backend(self, env: Environment, run_mode=None) -> str:
        """Inspect which RAM-family binary will run for a given environment.

        Useful for diagnostics and tests — call this before ``run()`` to
        confirm dispatch without executing the binary. When the model was
        constructed with an explicit ``backend=``, that choice is returned
        (after a compatibility check) regardless of the environment.

        Parameters
        ----------
        env : Environment
            The environment the dispatch is for.
        run_mode : RunMode, optional
            The run mode the dispatch is for. Only ``COHERENT_TL`` (the
            default) routes a fluid+flat *layered* bottom to ``ramgeo``;
            broadband / time-series stay on ``mpiramS`` (native
            multi-frequency sweep). ``None`` assumes ``COHERENT_TL``.

        Returns
        -------
        str
            ``'ramgeo'`` (fluid + flat, narrowband layered; sediment layers
            parallel to bathymetry — a simple bottom is accepted when forced),
            ``'mpirams'`` (fluid + flat: simple bottom, or broadband),
            ``'rams'`` (elastic bottom, flat surface),
            ``'ramsurf'`` (fluid bottom, variable surface).

        Raises
        ------
        UnsupportedFeatureError
            For elastic + variable-surface environments — no published
            Collins PE handles that combination. Use OASES for
            range-independent elastic propagation, or approximate by
            either flattening the surface (``rams``) or fluidising the
            bottom (``ramsurf``).
        ConfigurationError
            When a forced ``backend=`` cannot represent the environment.
        """
        if self.backend is not None:
            validate_forced_backend(self.backend, env)
            return self.backend
        elastic = env.bottom.is_elastic
        rough = env.altimetry is not None
        if elastic and rough:
            raise UnsupportedFeatureError(
                'RAM',
                'elastic bottom + sea-surface altimetry',
                alternatives=[
                    "OASES (range-independent elastic + rough)",
                    "drop env.altimetry to use rams0.5 (Collins elastic PE)",
                    "fluidise the bottom (set shear_speed=0) to use ramsurf1.5",
                ]
            )
        if elastic:
            check_rams_top_layer_carries_shear(env)
            return 'rams'
        if rough:
            return 'ramsurf'
        # Fluid + flat: RAMGEO for narrowband TL through a *layered* bottom,
        # because its deck carries the layer geometry itself. mpiramS keeps the
        # broadband path and the simple half-space cases, where there is no
        # layer geometry to track. ramgeo still *accepts* a simple bottom when
        # forced via backend='ramgeo'. See :func:`_dispatch.prefer_ramgeo` for
        # what actually separates the two on a layered stack — following the
        # bathymetry is not it, since mpiramS anchors its own profile at the
        # local seafloor.
        if prefer_ramgeo(env, run_mode):
            return 'ramgeo'
        return 'mpirams'

    def _collins_binary(self, kind: str) -> Path:
        """Resolve the path to a Collins-family binary on disk.

        A pinned ``executable`` is the binary of the backend named by
        ``backend=`` (mpiramS when ``backend=None``); a dispatch that lands
        on any other backend refuses, because each binary reads its own
        deck format.
        """
        if getattr(self, 'executable', None) is not None:
            if kind != self.backend:
                raise ConfigurationError(
                    f"RAM(executable={str(self.executable)!r}) pins the "
                    f"{self.backend or 'mpirams'} binary, but this run "
                    f"dispatches to the {kind!r} backend, which reads a "
                    f"different deck. Pass backend={kind!r} together with "
                    f"the {kind} build, or drop executable= to auto-detect."
                )
            return self._exe
        if kind == 'mpirams' or kind not in self._BINARIES:
            raise ConfigurationError(f"Unknown Collins kind {kind!r}.")
        return self._find_binary(kind)

    # ── stages 4-5: write, launch, read, build the result ─────────────────

    def _n_launches(self, settings) -> int:
        """One binary launch per grid of ``settings.engine``: one for
        mpiramS and a Collins TL run, one per bin for a Collins band — each
        clearing the fixed output names the last one wrote, and read before
        the next writes its deck, so the band reuses one work directory and
        the paths attached to the result describe it."""
        return len(settings.engine.grids)

    def _prepare_launches(self, env, settings) -> Optional[dict]:
        """The frequency-invariant Collins deck of a band run
        (:func:`collins.collins_deck_base`), cut once for every bin, with
        the water attenuation of the whole band
        (:func:`_domain.water_alpha_band`, one ``absorption.alpha`` call and
        so one out-of-range notice for the band) — each launch's
        ``inputs.prepared``; ``None`` for mpiramS and a Collins TL run. A
        local of the run, never an attribute: it describes this ``env``
        only."""
        if settings.engine.backend == 'mpirams':
            return None
        return self._prepare_collins_band(env, settings)

    def _write_input(self, inputs: StageInputs) -> Path:
        """Stage 4: the deck of launch ``inputs.launch``, written by the io
        writers from ``inputs.settings.engine`` — mpiramS's ``in.pe`` and
        the files it names (:func:`mpirams.write_mpirams_deck`), or a
        Collins ``ram.in`` (:func:`collins.write_collins_deck`) — returning
        the path of the deck the binary reads."""
        if inputs.settings.engine.backend == 'mpirams':
            return write_mpirams_deck(inputs, knobs=self._knob_record(),
                                      log=self._log)
        return write_collins_deck(inputs, knobs=self._knob_record())

    def _launch(self, inputs: StageInputs, deck: Path) -> None:
        """Stage 4: run the backend's binary on ``deck`` and require the
        output it writes (quoting the subprocess streams when it wrote
        none: no RAM binary writes a print file): s_mpiram
        (:meth:`_run_binary`) or the launch's Collins binary
        (:meth:`_run_collins_binary`)."""
        if inputs.settings.engine.backend == 'mpirams':
            self._run_binary(inputs.work_dir, dr=launch_grid(inputs).dr)
        else:
            self._run_collins_binary(inputs)

    def _read_output(self, inputs: StageInputs, deck: Path):
        """Stage 4: what the launch wrote — mpiramS's ``psif.dat`` as
        :func:`~uacpy.io.mpirams_reader.read_psif` returns it, or one Collins
        launch's envelope on the receiver grid
        (:func:`collins.read_collins_output`)."""
        if inputs.settings.engine.backend == 'mpirams':
            return read_psif(inputs.work_dir)
        return read_collins_output(inputs, knobs=self._knob_record())

    def _to_result(self, inputs: StageInputs, deck: Path, raw):
        """Stage 5: the :class:`Field` of the run from the launch output
        ``raw`` (the list of them for a band of launches), assembled as the
        backend's family does (:meth:`_assemble_mpirams`,
        :meth:`_assemble_collins`) — complex pressure in
        the travelling-wave convention (``models/ram/_pe_phase.py``), NaN
        inside the absorbing layer, with the grid in its metadata and the output
        paths attached — and, for ``TIME_SERIES``, its synthesis with the
        request's pulse."""
        settings = inputs.settings
        raw = raw if self._n_launches(settings) > 1 else [raw]
        assemble = (self._assemble_mpirams
                    if settings.engine.backend == 'mpirams'
                    else self._assemble_collins)
        field = assemble(inputs, raw)
        if settings.mode == RunMode.TIME_SERIES:
            field = finish_broadband(self.model_name, field, settings)
        return field

    # ── the backend families' steps ─────────────────────────────────────

    def _resolve_mpirams_grids(self, env, source, receiver, mode, backend, *,
                               notices):
        """mpiramS step: its one grid, marching the whole band in its own
        frequency loop (:func:`mpirams.resolve_mpirams_band`) or one
        frequency (:func:`mpirams.resolve_mpirams_tl`)."""
        if mode in _BAND_MODES:
            return resolve_mpirams_band(
                env, source, receiver, notices=notices,
                knobs=self._knob_record(), log=self._log)
        return resolve_mpirams_tl(env, source, receiver, notices=notices,
                                  knobs=self._knob_record(), log=self._log)

    def _resolve_collins_grids(self, env, source, receiver, mode, backend, *,
                               notices):
        """Collins step: one grid per bin of a band, each launched on its
        own (:func:`collins.resolve_collins_band`; a ``TIME_SERIES`` band
        must be uniform), or the one grid of a TL run
        (:func:`collins.resolve_collins_tl`)."""
        if mode in _BAND_MODES:
            return resolve_collins_band(
                env, source, receiver, backend,
                require_uniform=mode == RunMode.TIME_SERIES, notices=notices,
                knobs=self._knob_record(), log=self._log)
        return resolve_collins_tl(env, source, receiver, backend,
                                  notices=notices, knobs=self._knob_record(),
                                  log=self._log)

    def _prepare_collins_band(self, env, settings) -> Optional[dict]:
        """Collins step: the deck every bin of a band shares, with the
        band's water attenuation; ``None`` for a TL run."""
        engine = settings.engine
        if settings.mode not in _BAND_MODES:
            return None
        zmax = engine.grids[0].zmax
        base = collins_deck_base(env, engine.backend, zmax,
                                 knobs=self._knob_record())
        base['water_alpha'] = water_alpha_band(
            env, engine.marched_frequencies, zmax)
        return base

    def _assemble_mpirams(self, inputs: StageInputs, raw):
        """mpiramS step: the band's Field from its one sweep
        (:func:`mpirams.assemble_broadband_field`) or the TL Field
        (:func:`mpirams.assemble_tl_field`)."""
        settings = inputs.settings
        engine = settings.engine
        grid = engine.grids[0]
        if settings.mode in _BAND_MODES:
            return assemble_broadband_field(
                raw[0], inputs.env, inputs.source, inputs.receiver,
                inputs.work_dir, engine, grid, log=self._log,
                model_name=self.model_name, provenance=self.provenance,
                cleanup=self.cleanup, knobs=self._knob_record())
        return assemble_tl_field(
            raw[0], inputs.env, inputs.source, inputs.receiver,
            inputs.work_dir, engine, grid,
            knobs=self._knob_record(), model_name=self.model_name,
            provenance=self.provenance, cleanup=self.cleanup)

    def _assemble_collins(self, inputs: StageInputs, raw):
        """Collins step: the band's Field stacked from its launches
        (:func:`collins.assemble_collins_band_field`) or the TL Field
        (:func:`collins.assemble_collins_tl_field`)."""
        if inputs.settings.mode in _BAND_MODES:
            return assemble_collins_band_field(
                inputs, raw, model_name=self.model_name,
                provenance=self.provenance, cleanup=self.cleanup,
                knobs=self._knob_record())
        return assemble_collins_tl_field(
            inputs, raw[0], model_name=self.model_name,
            provenance=self.provenance, cleanup=self.cleanup,
            knobs=self._knob_record())

    def _run_collins_binary(self, inputs: StageInputs) -> None:
        """Stage 4 of one Collins launch: run the binary in the work
        directory, raise on the stop conditions it reports on stdout, and
        require both files its patched ``outpt`` writes."""
        engine = inputs.settings.engine
        kind = engine.backend
        grid = launch_grid(inputs)
        n = len(engine.grids)
        if n > 1:
            # Progress every ~10% of frequencies; on a 500-freq elastic run
            # each launch takes ~1 s of subprocess overhead, so without this
            # the verbose log goes silent for many minutes.
            k = inputs.launch
            log_every = max(1, n // 10)
            if k % log_every == 0 or k == n - 1:
                self._log(f"{kind} broadband: freq {k + 1}/{n} "
                          f"({grid.frequency:.2f} Hz)")
        binary = self._collins_binary(kind)
        work_dir = inputs.work_dir
        self._log(f"Executing: {binary} (cwd={work_dir})")
        run_launch(self._launch_context(), Launch(
            argv=(str(binary),), cwd=work_dir, timeout=self.timeout,
            stale_outputs=_COLLINS_OUTPUTS,
            # rams0.5 echoes every march step to stdout (rams0.5.f:254), so
            # the text is formatted only when it will be printed (DEBUG).
            stdout_label=kind,
            checks=(
                lambda result: raise_on_ram_stop(
                    kind, binary, result, dr=grid.dr,
                    knobs=self._knob_record()),
                # A missing or empty tl.grid means the binary died
                # silently; the raised error quotes the subprocess streams
                # (the Collins codes write no print file).
                lambda result: require_output(
                    self.model_name, [work_dir / 'tl.grid'],
                    what=f'a TL grid ({kind} tl.grid)', process=result),
                # The patched outpt (third_party/ramsurf/{rams0.5,
                # ramsurf1.5}.f + MODIFICATIONS.md) writes pcomplex.bin
                # alongside tl.grid.
                lambda result: require_output(
                    self.model_name, [work_dir / 'pcomplex.bin'],
                    what=f'a complex-envelope grid ({kind} pcomplex.bin)',
                    process=result,
                    hint=('Rebuild the binaries via install.sh so the '
                          'patched outpt routine emits the complex '
                          'envelope.')),
            )))

    # Settings that only the mpiramS backend consumes. When the dispatcher
    # picks a Collins backend (ramgeo1.5 / ramsurf1.5 / rams0.5) and one of
    # these has been overridden from its default,
    # ``_warn_on_mpirams_only_overrides`` warns rather than silently dropping
    # the override. Each entry is (attribute, default). Q and T are honoured
    # by every backend's broadband mode: the Collins path uses them as the
    # Python-side frequency-loop grid, mpiramS uses them inside the Fortran
    # loop. absorber_width_wavelengths / absorber_attenuation are honoured
    # too — the first sizes zmax in ``_domain.compute_zmax``, both drive the
    # attenuation ramp in ``collins.ramp_absorbing_attenuation``.
    _MPIRAMS_ONLY_SETTINGS = (
        ('n_sediment_points', None),
    )

    # rams0.5's row 5 is ``c0 np irot theta`` (rams0.5.f:109) where the fluid
    # codes read ``c0 np ns rs`` (ramgeo1.5.f:108, ramsurf1.5.f:80), so the
    # two stability pairs are mutually exclusive at the Fortran level and
    # ``write_ram_in`` switches the row per kind. Overriding the pair the
    # selected backend does not read discards the value silently.
    _RAMS_ONLY_SETTINGS = (
        ('rams_rotation_angle_deg', None),
        ('rams_rotation', None),
        ('rams_dr_factor', None),
    )

    _NOT_RAMS_SETTINGS = (
        ('n_stability', None),
        ('stability_range', None),
    )

    def _warn_on_mpirams_only_overrides(self, backend: str, *,
                                        notices=None) -> None:
        """Warn about knobs the selected backend cannot read.

        Three disjoint groups: mpiramS-only options the Collins writers
        have no field for, and the two mutually exclusive row-5 stability
        pairs (``ns/rs`` on the fluid codes, ``irot/theta`` on rams0.5).
        """
        groups = []
        if backend != 'mpirams':
            groups.append(('mpiramS-only', self._MPIRAMS_ONLY_SETTINGS))
        if backend == 'rams':
            groups.append(("mpiramS/ramgeo/ramsurf-only", self._NOT_RAMS_SETTINGS))
        else:
            groups.append(('rams0.5-only', self._RAMS_ONLY_SETTINGS))

        for label, settings in groups:
            nondefault = [
                name for name, default in settings
                if getattr(self, name) != default
            ]
            if nondefault:
                give_notice(notices,
                    f"RAM:{backend} ignores these {label} settings "
                    f"(left at their effective default in the binary): "
                    f"{', '.join(nondefault)}. See the RAM constructor "
                    f"docstring for the per-backend applicability.",
                    FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP
                )

    def _run_binary(self, work_dir: Path, *, dr: float):
        """Execute s_mpiram in ``work_dir``, requiring the ``psif.dat`` it
        writes.

        Its thread count is the one rule of :func:`~uacpy.models._launch.
        openmp_thread_env`: an exported ``OMP_NUM_THREADS`` as given, 1 in a
        ``run_parallel`` worker (a pool of N workers would otherwise start N
        x N threads), otherwise libgomp's one thread per CPU. The count
        cannot change the answer: mpiramS parallelises only its frequency
        loop (``third_party/mpiramS/src/peramx.f90:414-439``, ``!$OMP DO
        SCHEDULE(STATIC,1)`` over the frequency index), each iteration
        writing only its own slice ``psif(:,iff,ir)``, with the per-thread
        state THREADPRIVATE (``param.f90:23``, ``mattri.f90:12``,
        ``fld.f90:10``, ``profiles.f90:8``) and no reduction; the one shared
        bracket start in ``splnlib.f90``'s ``interv`` is a per-call local
        (third_party/MODIFICATIONS.md). Measured bit-identical at 1, 2 and 4
        threads on a flat and a range-dependent BROADBAND run and a
        TIME_SERIES run.

        A stop the binary diagnoses itself (its Padé root finder, at the
        ``dr`` it marched with) is raised naming it (:func:`collins.
        raise_on_ram_stop`) before the missing-output check.
        """
        env, threads = openmp_thread_env()
        self._log(f"Executing mpiramS: {self._exe} (cwd={work_dir}, "
                  f"{threads})")
        run_launch(self._launch_context(), Launch(
            argv=(str(self._exe),), cwd=work_dir, env=env,
            timeout=self.timeout, stale_outputs=_MPIRAMS_OUTPUTS,
            stdout_label='mpirams',
            checks=(
                lambda result: raise_on_ram_stop(
                    'mpirams', self._exe, result, dr=dr,
                    knobs=self._knob_record()),
                # A missing or empty psif.dat means the binary died
                # silently; the raised error quotes the subprocess streams
                # (mpiramS writes no print file).
                lambda result: require_output(
                    self.model_name, [work_dir / 'psif.dat'],
                    what='an output field (psif.dat)', process=result,
                    hint='Check input parameters.'),
            )))
