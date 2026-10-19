"""``fetch_environment`` — GPS (+ date) → a ready-to-run ``Environment``.

The capstone of the on-demand data layer: assemble bathymetry, sound speed
and (optionally) a bottom into a single :class:`~uacpy.core.environment.\
Environment` that drops straight into any propagation model.

    env = uacpy.data.fetch_environment((43.2, 7.5), date='2026-06-14',
                                       bottom='sand')

Each axis is handled the same way: supply it as a **literal** (``ssp=`` /
``bathymetry=`` / ``bottom=`` / ``surface=`` / ``altimetry=``, as
:class:`Environment` takes them, except that a ``bottom=`` number is a mean
grain size ϕ where ``Environment`` reads a half-space sound speed) and/or
fetch it from one or more **sources**
(``ssp_sources`` / ``bathymetry_sources`` / ``bottom_sources`` /
``surface_sources``). If both are given for an axis the source is fetched first
and the literal is the **fallback** when the fetch yields nothing (no coverage,
service down, no installed cache), announced by a ``FallbackWarning`` naming
why. A source asked for without an argument it needs (``date=``,
``transect_to=``) is refused whether or not a literal is given. ``*_sources`` are ordered fallback lists (a bare string is a
1-element list, ``'auto'`` the best-available preset, ``'local'`` the
best-available *cached* source — local data only, no network); bathymetry and
SSP default to fetching ``'gebco'`` / ``'woa23'`` when neither form is given,
while bottom and surface are optional (fetched only when asked). Altimetry
(sea-state roughness) is fetched only via ``altimetry_sources`` (needs a
transect and date), else literal-only. Fetching is **cache-first within each
source**: a source with a locally installed twin (GEBCO, WOA23) samples it
before its own live backend. The chain order itself is quality-first, so
``'auto'`` tries the better live sources (Argo/Copernicus for SSP,
EMODnet-DTM/GMRT for bathymetry) *before* the installed global grids — an
``'auto'`` run can hit the network before any cache. ``*_sources='local'``
skips the network entirely (failing fast with an install hint), so an
air-gapped or reproducible run sets ``'local'`` on the axes it wants pinned to
local data (see ``install.sh --data``).
"""

import datetime as _dt
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

from uacpy._log import log_message
from uacpy.core.absorption import FrancoisGarrison
from uacpy.core.acoustics import density
from uacpy.core.environment import (
    Bathymetry, Bottom, BoundaryProperties, Environment, SeabedColumn,
    SoundSpeedProfile,
)
from uacpy.core.exceptions import (
    ConfigurationError, DataFetchError, FallbackWarning,
)
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.geo import Coordinate, as_coordinate, great_circle_km
from uacpy.data._geo import (
    checked_explicit_n_points, checked_max_days, checked_max_distance,
    checked_max_points, checked_n_points, mid_depth,
    point_along_track,
)
from uacpy.core.acoustics.seawater import pressure_dbar_to_depth
from uacpy.data._chain import (CACHE_BACKEND, MissingPrerequisiteError,
                               SourceChain, SourceProvider, first_answer,
                               lower_source_spec, source_tuple,
                               unused_setting_notices)
from uacpy.data.bathymetry import (
    _refuse_a_dry_point,
    _BATHY_CHAIN, transect_length,
)
from uacpy.core.sediment import (DEFAULT_GRAIN_SIZE_MODEL,
                                 _canonical_grain_size_selection)
from uacpy.data.sediment import transect_fetcher
from uacpy.data.seaice_local import SEA_ICE_EDGE_CONCENTRATION
from uacpy.data.sound_speed import (
    DEFAULT_SOUND_SPEED_FORMULA, _DEFAULT_RESOLUTION, _SSP_CHAIN,
    _checked_optional_resolution, _fetch_ts_profile_backend,
    extend_ssp_below_data, fetch_ssp, fetch_ts_profile,
)
from uacpy.data.sources import SOURCES, DataProvenance
from uacpy.data._provenance_notice import (licence_notice_text,
                                           one_provenance_notice,
                                           provenance_fallback,
                                           provenance_notice)

__all__ = ['fetch_environment', 'fetch_bottom', 'fetch_bottom_transect']


# Default fetch chains for the two mandatory axes (used when neither a literal
# nor an explicit ``*_sources`` is given). Both are global and need no login.
_DEFAULT_SSP_SOURCES = ('woa23',)
_DEFAULT_BATHY_SOURCES = ('gebco',)


def _resolve_cached(call, order, *, axis):
    """``(call(token), token)`` for the first ``(source, backend)`` token of
    ``order`` that answers, by the chain rule of
    :func:`~uacpy.data._chain.first_answer` (a cached twin that read an answer
    ends its source; the most substantive error is raised when none answers).

    An empty ``order`` (``*_sources='local'`` with no cached source) is
    refused by name, before any fetch.
    """
    if not order:
        raise ConfigurationError(
            f"fetch_environment: {axis}_sources='local' but no cached {axis} "
            f"source is available.",
            remediation="Install a cached dataset (see install.sh --data), or "
                        "use 'auto' / a live source.",
        )
    return first_answer(order, lambda src, backend: call((src, backend)))


def _require_nonempty_sources(spec, *, axis):
    """Reject an explicit empty ``*_sources`` sequence: it selects no source,
    so nothing could ever be fetched for that axis (the pre-fetch twin of
    :func:`~uacpy.data._chain.raise_substantive`'s empty-chain diagnostic).
    ``None`` (axis not requested) and strings (``'auto'`` / ``'local'`` / a
    source name) pass through."""
    if spec is None or isinstance(spec, str):
        return
    if len(source_tuple(spec)) == 0:
        raise ConfigurationError(
            f"fetch_environment: {axis}_sources={spec!r} selects no source.",
            remediation="Pass at least one source: an empty sequence "
                        f"({axis}_sources=()) selects none. Use 'auto', "
                        f"'local', or omit {axis}_sources=.",
        )


@one_provenance_notice(subject="this environment's data",
                       record='uacpy.data.citations(env)')
def fetch_environment(
    point: Coordinate,
    *,
    date: Union[str, _dt.date, None] = None,
    name: Optional[str] = None,
    ssp=None,
    ssp_sources: Union[str, Sequence[str], None] = None,
    bathymetry=None,
    bathymetry_sources: Union[str, Sequence[str], None] = None,
    bottom: Union[float, str, BoundaryProperties, SeabedColumn, Bottom,
                  None] = None,
    bottom_sources: Union[str, Sequence[str], None] = None,
    bottom_model: Optional[str] = None,
    bottom_hamilton_fit: Optional[str] = None,
    surface: Optional[BoundaryProperties] = None,
    surface_sources: Union[str, Sequence[str], None] = None,
    surface_threshold: Optional[float] = None,
    altimetry=None,
    altimetry_sources: Optional[str] = None,
    altimetry_n_points: Optional[int] = None,
    altimetry_rng: Optional[np.random.Generator] = None,
    transect_to: Optional[Coordinate] = None,
    n_points: Union[int, str, None] = None,
    max_points: Optional[int] = None,
    range_dependent_ssp: Optional[bool] = None,
    ssp_n_points: Union[int, str, None] = None,
    range_dependent_bottom: Optional[bool] = None,
    bottom_n_points: Union[int, str, None] = None,
    range_dependent_surface: Optional[bool] = None,
    surface_n_points: Union[int, str, None] = None,
    with_absorption: bool = False,
    max_distance_km: Optional[float] = None,
    max_days: Optional[int] = None,
    formula: str = DEFAULT_SOUND_SPEED_FORMULA,
    resolution: Optional[str] = None,
    timeout: float = 120.0,
    verbose: Union[bool, str] = False,
) -> Environment:
    """Fetch and assemble an :class:`Environment` for a ``(lat, lon)`` point.

    Parameters
    ----------
    point : (lat, lon)
        Latitude/longitude in decimal degrees (WGS84).
    date : str or datetime.date, optional
        Calendar date. Selects the climatological month for WOA23, or the
        time step for Copernicus (required when ``ssp_sources='copernicus'``).
    name : str, optional
        Environment name. Defaults to the coordinate string.
    ssp : SoundSpeedProfile or float or sequence, optional
        A **literal** sound-speed profile supplied directly (same forms as
        :class:`Environment`'s ``ssp=``: a ``SoundSpeedProfile``, a scalar c in
        m/s → isovelocity, or ``(depth, c_m_per_s)`` pairs). If ``ssp_sources``
        is *also* given, the source is fetched first and this literal is the
        **fallback** when the fetch yields nothing; on its own, SSP is not
        fetched at all. A literal is not deepened: one that stops above the
        seafloor is held at its deepest speed by ``Environment``, which warns
        (``FallbackWarning``) once the hold is long enough to matter, whereas
        a fetched profile is continued below its data. Extend a cast yourself
        before passing it (:func:`extend_ssp_below_data`, valid from the deep
        isothermal layer down).
    ssp_sources : str or sequence of str, optional
        Sound-speed source(s) to **fetch**, tried in order with the next as
        fallback (a bare string is a 1-element list), or a preset: ``'auto'``
        (best-available: ``argo`` → ``copernicus`` → ``woa23``, i.e. real float
        → model → climatology) or ``'local'`` (the cached WOA23 climatology
        only — no network). Choices: ``'woa23'`` (climatology, global),
        ``'copernicus'`` (operational model) and ``'argo'`` (nearest real
        float) — the latter two need ``date=`` and the network (``'auto'``
        falls through to WOA23 without them). Default ``None`` → fetch
        ``'woa23'``. E.g. ``ssp_sources=('copernicus', 'woa23')`` = Copernicus,
        else WOA23.
    bathymetry : float or array, optional
        A **literal** depth (m, scalar) or range-dependent ``(N, 2)``
        ``(range, depth)`` array — depth positive down — supplied directly.
        If ``bathymetry_sources`` is *also* given, the source is fetched first
        and this literal is the **fallback**; on its own, bathymetry is not
        fetched.
    bathymetry_sources : str or sequence of str, optional
        Bathymetry source(s) to **fetch**, tried in order, or a preset:
        ``'auto'`` (best-available: ``emodnet_dtm`` → ``gmrt`` → ``gebco``, i.e.
        the ~115 m regional DTM where it covers, else high-res multibeam where
        surveyed, else the global grid) or ``'local'`` (the cached GEBCO grid
        only — no network). Choices: ``'gebco'`` (global), ``'gmrt'`` (multibeam,
        higher-res, CC-BY) or ``'emodnet_dtm'`` (EMODnet DTM ~115 m, European
        seas + Caribbean, CC-BY). Default ``None`` → fetch ``'gebco'``.
    bottom : float or str or BoundaryProperties or SeabedColumn or Bottom, optional
        A **literal** seafloor supplied directly (no fetch): a mean grain size
        (ϕ, float), a :data:`~uacpy.materials.MATERIALS` class name, a ready
        ``BoundaryProperties``, the layered ``SeabedColumn`` or range-dependent
        ``Bottom`` that :func:`fetch_bottom` / :func:`fetch_bottom_transect`
        return, or ``None`` (default). A carrier passes unchanged with its own
        provenance. If ``bottom_sources`` is *also* given, the source is
        fetched first and this literal is the **fallback**. (A bottom *string*
        is a material name here, never a source — source keywords go in
        ``bottom_sources`` — which is why the two are separate args.)

        A class name is the material's **fluid** preset, as ``Environment``
        reads it (pass ``BoundaryProperties.from_preset(name, elastic=True)``
        for the elastic one). A
        number is a grain size, not the half-space sound speed
        ``Environment(bottom=1800)`` reads: a number with ``|ϕ| > 20`` (a
        grain coarser than 1 km or finer than 1 nm) is refused as a likely
        sound speed; pass a sound speed as a ``BoundaryProperties``.

        There is no ``bottom_roughness=`` knob, and because the literal is only
        a fallback, a ``BoundaryProperties(roughness=...)`` passed alongside a
        ``bottom_sources`` that resolves is discarded along with the rest of the
        literal. Fetched geoacoustics *and* a site roughness come from fetching
        the seabed yourself and handing it over as the literal — the direct
        analogue of the ``surface=`` remedy below — with the same source
        chains ``bottom_sources`` takes::

            bp  = uacpy.data.fetch_bottom(pt, source='local', roughness=0.5)
            env = fetch_environment(pt, bathymetry_sources='local',
                                    ssp_sources='local', bottom=bp)

        On a transect, pass the ``Bottom`` from
        ``uacpy.data.fetch_bottom_transect(pt, transect_to, ...)``, whose
        ranges are measured from ``pt`` as the transect's are.
    bottom_sources : str or sequence of str, optional
        Seafloor source(s) to **fetch**, tried in order, or a preset: ``'auto'``
        (best-available, cached-first: EMODnet → grain-size → Diesing → MARS
        → pelagic — measured samples precede the modelled maps, and the
        installed Diesing raster is consulted before the live AusSeabed service,
        which then covers the Australian shelf Diesing's deep-sea map misses) or
        ``'local'`` (network-free: EMODnet-local → grain-size → Diesing →
        pelagic, cached backends only). Per-source choices: ``'emodnet'``
        (European seas, high-res, CC-BY), ``'mars'`` (AusSeabed MARS samples,
        Australian margin, CC-BY — live), ``'grainsize'`` (NCEI grain-size
        samples, worldwide, public-domain — cached), ``'diesing'`` (global
        deep-sea lithology map, water deeper than 500 m, CC-BY — cached),
        ``'crust1'`` (CRUST1.0 + GlobSed → a layered *elastic* bottom for
        low-frequency work — cached), ``'graw'`` (measured seabed-density grid
        — cached), ``'pelagic'`` (first-principles, never fails). Default
        ``None`` — bottom is optional, so it is only fetched when you ask.
        Most sources permit commercial use;
        CRUST1.0 does not without verification — a non-commercial source is
        named in the call's provenance notice. See ``uacpy.data.citations(env)``.
    bottom_model : {'hamilton', 'apl-uw'}, optional
        The grain-size → geoacoustics relations behind every seabed that
        arrives as a mean grain size ϕ — the fetched sources above (all but
        ``'crust1'``) and a ϕ ``bottom=`` literal alike. ``'hamilton'``
        (``None``, the default, is :data:`~uacpy.core.sediment.DEFAULT_GRAIN_SIZE_MODEL`,
        ``'hamilton'``) is the low-frequency Hamilton & Bachman table;
        ``'apl-uw'`` the APL-UW TR 9407 relations for 10-100 kHz work, the
        same seabed the high-frequency scattering models in
        :mod:`uacpy.sonar` are built on (``BottomParameters.from_environment``
        reads it back). Both scale the sediment SOUND SPEED to the in-situ
        water at the seafloor and carry the DENSITY ratio against the water
        density the deck carries: the site's IES-80 value when
        ``with_absorption`` fetches the T/S column, else
        ``DEFAULT_WATER_DENSITY`` (1.027 g/cm³). A
        class-name literal, a ``BoundaryProperties`` and the hard-substrate
        presets some sources return (EMODnet rock, DECK41 ``'rock'``) carry
        their own numbers and are not affected. A value set when no grain
        size is converted (the default seabed, a literal carrier or class
        name, a CRUST1.0 answer) gives a ``FallbackWarning`` that it was not
        used, as does ``bottom_hamilton_fit``. See
        :func:`uacpy.core.sediment.grain_size_to_geoacoustics`.
    transect_to : (lat, lon), optional
        If given, bathymetry is sampled along the great-circle path from
        ``(lat, lon)`` to here (range-dependent); otherwise a single depth.
    n_points : int or 'auto', optional
        Bathymetry transect sample count (used only with ``transect_to``).
        An int is used as given; ``None`` (default) is 50 (uacpy guidance),
        within ``max_points`` when that is set. ``'auto'`` targets GEBCO
        native resolution (bathymetry is continuous, so it is not
        duplicate-collapsed), bounded by ``max_points``.
    max_points : int, optional
        The sampling budget of the bathymetry, SSP, bottom and surface
        transects: the points an ``'auto'`` count samples *before* its
        reduction (the reduced grid is never larger), and the ceiling of an
        unset count. ``None`` (default) is 1000 for ``'auto'`` (uacpy
        guidance). An explicit count (``n_points``, ``ssp_n_points``,
        ``bottom_n_points``, ``surface_n_points``) is used as given, and
        refused when it exceeds an explicit ``max_points`` (the two
        contradict each other).
    range_dependent_ssp : bool, optional
        Whether the SSP varies along the transect. **Default (``None``): a
        transect makes the SSP range-dependent** (a single point is always
        range-independent — there is nothing to vary). Pass ``False`` to force a
        single profile at the start point even along a transect; ``True`` on a
        point raises. Built from ``'woa23'`` or ``'copernicus'``; an ``'argo'``
        cast is a single profile, so a chain containing it falls through to the
        next source on a transect.
    ssp_n_points : int or 'auto', optional
        SSP columns along the transect when ``range_dependent_ssp``.
        ``None`` (default) is ``'auto'``: the transect is sampled at the
        **distinct grid cells** it
        crosses (one column per crossed cell — the source's native range
        resolution; a dry WOA23 cell takes its nearest wet cell, which two
        crossed cells may share), probing ``max_points`` waypoints:
        the WOA23 cells found analytically, the Copernicus 1/12° cells from
        the dataset's own axes (each column one remote read). Pass an int for
        exactly that many evenly-spaced columns.
    range_dependent_bottom : bool, optional
        Whether a fetched seafloor varies along the transect. A bottom is fetched
        only when requested (``bottom_sources=`` given, or this flag ``True``).
        **Default (``None``): a requested bottom is range-dependent on a
        transect**, range-independent at a point. Pass ``False`` to force a
        single representative bottom even along a transect; ``True`` on a point
        raises (and, with no ``bottom_sources``, implies ``'auto'``). A chain
        of sources (``'auto'``, ``'local'``, or a sequence) is resolved at
        every waypoint: each takes the first source in the chain that covers
        it, and each column cites that source. A single named source has no
        next source to ask, so its uncovered waypoints take the nearest
        covered waypoint's seabed, with a warning.
    bottom_n_points : int or 'auto', optional
        Seabed samples along the transect when ``range_dependent_bottom``.
        ``None`` (default) is 6 (uacpy guidance), within ``max_points`` when
        that is set — unlike SSP, the bottom sources expose no cheap
        sample identity, so ``'auto'`` must *fetch* at every probe point (cheap
        for the local sample DBs, but up to ``max_points`` live calls for the
        EMODnet WFS). Opt into ``'auto'`` for native resolution on local sources.
    range_dependent_surface : bool, optional
        Whether a fetched sea-ice surface varies along the transect. A surface
        is fetched only when requested (``surface_sources='seaice'`` given, or
        this flag ``True``) — a bare transect does **not** auto-fetch ice.
        **Default (``None``): a requested surface is range-dependent on a
        transect** (a marginal ice zone: open water ↔ pack ice), range-
        independent at a point. ``False`` forces a single surface; ``True`` on a
        point raises (and implies the ``'seaice'`` source). The solvers carry a
        single global top boundary, so every model collapses it to one (with a
        warning) — the range-dependent surface is for inspecting/plotting.
    surface_n_points : int or 'auto', optional
        Sea-ice samples along the transect when ``range_dependent_surface``.
        ``None`` (default) is ``'auto'``: probe the local NSIDC grid (cheap)
        and collapse
        each run of identical ice/open-water zones to the samples bracketing
        its edges (the marginal ice zone at native scale, each edge within one
        probe step, no staircase), probing ``max_points`` waypoints. Pass an
        int for exactly that many waypoints.
    surface : BoundaryProperties, optional
        A **literal** top-boundary override supplied directly (e.g. a custom ice
        canopy). If ``surface_sources`` is *also* given, the source is fetched
        first and this is the **fallback**; on its own, the surface is not
        fetched. Default ``None`` (free surface).
    surface_sources : str or sequence of str, optional
        Top-boundary source(s) to **fetch**, or ``'auto'`` / ``'local'``. The
        only source is the cached ``'seaice'`` climatology (so ``'auto'`` and
        ``'local'`` both == ``('seaice',)``): requires ``date=`` and
        the cached ``seaice`` climatology (``install.sh --data seaice``), and
        sets the surface from the NSIDC concentration at the point for
        ``date``'s month — an ice-covered point (≥ ``surface_threshold``) gets
        a homogeneous elastic ice canopy (cp 3500 m/s, cs 1800 m/s, ρ 0.9 g/cm³,
        αp/αs 0.4/1.0 dB/λ — *Computational Ocean Acoustics*); open water keeps
        the default free surface (no provenance). The canopy is an elastic
        **half-space**, not a plate of finite thickness: the model writers emit
        the top boundary as an upper half-space line, with no thickness field to
        give it, so concentration decides *whether* there is ice, never how
        thick. Default ``None`` (no surface fetch). At a point, one
        classification; on a transect, one per waypoint (see
        ``range_dependent_surface``), which the solvers collapse to their one
        top boundary. The half-space is the
        thick-ice limit of the 4 m plate the parameters are tabulated for, so
        it holds well above c_s/h, about 450 Hz for 4 m of ice. The fetched
        canopy is smooth (roughness 0), which
        under-predicts the loss over real deformed pack ice (see
        :func:`uacpy.data.sea_ice_surface`); a site roughness needs an
        explicit ``surface=`` literal, e.g. built with
        ``sea_ice_surface(conc, roughness=...)``.
    surface_threshold : float, optional
        The lowest sea-ice concentration (a fraction 0-1) a fetched surface
        takes as ice cover, at the point and at every transect waypoint.
        ``None`` (default) is 0.15: NSIDC's ice edge for its extent
        statistics (Sea Ice Index G02135 user guide), used as the acoustic
        switch as uacpy guidance — no acoustic source sets a concentration
        bound (Etter §2.4.2 separates pack ice, the marginal ice zone and open
        ocean with none). A set value with no sea-ice surface fetched gives a
        ``FallbackWarning`` that it was not used.
    altimetry : array-like, optional
        A **literal** rough-surface wave profile ``[(range, height), …]``
        (height positive up; same as :class:`Environment`'s ``altimetry=``).
        If ``altimetry_sources`` is *also* given, the source is fetched first and
        this is the **fallback**. Default ``None`` (flat).
    altimetry_sources : str or sequence of str, optional
        Sea-state sources to **fetch** into a Pierson-Moskowitz sea-surface
        realization, as catalogue ids tried in order: ``'waverys'`` /
        ``'ww3'`` (observed significant wave height → surface), ``'nbs'``
        (live 10 m wind → surface, fully-developed assumption), ``'local'``
        (the cached NBS monthly wind climatology — network-free, but a mean
        state rather than the day's) or ``'auto'`` (WAVERYS, WaveWatch III,
        live wind, the climatology). **Requires ``transect_to`` and ``date``**
        (the realization spans the transect range and sea state is time-
        specific); a single point has no range, so point altimetry is not
        fetched. Default ``None`` (flat surface unless ``altimetry=`` is given).
    altimetry_n_points : int, optional
        Range samples in a fetched sea-surface realization. Default ``None``:
        :func:`~uacpy.data.fetch_sea_surface` sizes it from the sea state, so
        the realization resolves the wave spectrum's peak over the whole
        transect (a fixed count aliases the waves away on a long one).
    altimetry_rng : numpy.random.Generator, optional
        Random generator a fetched sea-surface realization draws from
        (:func:`~uacpy.data.fetch_sea_surface`'s ``rng``); pass one for a
        reproducible surface, ``None`` draws a fresh one.
    bottom_hamilton_fit : str, optional
        Which of Hamilton & Bachman's three fits ``bottom_model=DEFAULT_GRAIN_SIZE_MODEL``
        evaluates: ``'continental-terrace'`` (the default when ``None``, shelf
        and slope, 1 to 9 ϕ), ``'abyssal-hill'`` or ``'abyssal-plain'`` (both
        7 to 10 ϕ). It reaches the ϕ ``bottom=`` literal and every fetched
        source that converts a grain size. **A deep-ocean site wants an abyssal
        fit** — at 9 ϕ the terrace fit gives ρ = 1.45 against the 1.35 and 1.41
        Hamilton & Bachman measured for abyssal clay — but nothing infers it:
        the paper states no rule for choosing, so only the caller knows the
        site. Pairing it with ``bottom_model='apl-uw'`` raises, that report
        publishing one set of relations rather than one per fit. The
        per-layer fetchers and converters take the same choice as
        ``hamilton_fit=``.
    with_absorption : bool, optional
        If ``True``, attach a Francois-Garrison absorption built from the
        site's fetched temperature/salinity column (costs one extra T/S
        request): the whole column, as a profile every engine evaluates at
        each depth with the water there
        (:meth:`~uacpy.core.absorption.FrancoisGarrison.from_temperature_salinity`).
        pH is the cached GLODAP column, as ``(depth, pH)`` pairs on its own
        levels, when installed (``install.sh --data glodap``) — at a coastal
        point whose own GLODAP cell holds none, the column of the cell the
        T/S came from; one value at
        the column's mid-depth from Copernicus BGC on that branch; else the
        reference-water default (8.0). The water density is the mid-depth
        row's, and every grain-size seabed (fetched or a ϕ literal) is
        converted against it. The column is read at ``point``; on a
        transect whose deepest seafloor sample lies below that column's
        deepest level, the column read at that sample (same dataset, one
        more request) continues it below — its T/S rows, and its GLODAP pH
        levels when the start's pH is a GLODAP column — so deep water is
        not given the start's deepest row. The added column is in
        ``env.data_sources``; the water density stays the start column's.
        Default ``False``: ``env.absorption`` is ``None``, lossless water.
    max_distance_km : float, optional
        Largest offset (km) between ``point`` (each waypoint, on a transect)
        and the data point a source read — the Argo cast, the grain-size or
        MARS sample, the centre of a gridded source's cell — applied to
        every fetched axis. A source whose data point is farther raises
        ``DataFetchError`` and the next source in the chain is tried (so a
        tight value makes ``'auto'`` fall through from Argo to WOA23). A
        gridded cell's centre stands up to half its diagonal from the point
        (about 79 km on the 1° WOA23 grid at the equator), so a value below
        that refuses ordinary cells too. ``None`` (default) sets no limit:
        a source past its own offset threshold is named in the call's
        provenance notice instead, and the nearest-sample sources search their own
        radius (Argo and grain-size 250 km, MARS 100 km).
    max_days : int, optional
        Maximum days the data used may differ from ``date`` for every
        **time-specific** read: the SSP sources ``'argo'`` (nearest float
        profile) and ``'copernicus'`` (nearest model time step), the
        ``with_absorption`` T/S column and Copernicus BGC pH read on those
        branches, and the WAVERYS sea state ``altimetry_sources`` reads. A
        source whose nearest match is staler raises ``DataFetchError`` and
        the chain falls through (e.g. ``'auto'`` → WOA23; a refused BGC pH →
        GLODAP). A monthly Copernicus product stamps its step once per
        month, so a value shorter than the date's distance to that stamp
        refuses the date's own month. Ignored by WOA23, GLODAP and the
        NBS wind climatology (climatologies keyed on month only), by
        WaveWatch III and the live NBS wind (which take no time tolerance),
        and by the bottom/bathymetry sources. Default ``None`` → each source's own
        default (Argo 15, Copernicus 31).
    formula, resolution, timeout, verbose
        Forwarded to the sound-speed / bathymetry fetchers. ``formula`` is the
        sound-speed equation, ``{'teos10', 'unesco', 'delgrosso', 'mackenzie'}``; ``resolution`` is the
        WOA23 grid spacing in degrees, ``{'1.00', '0.25'}`` (``None``, the
        default, is ``'1.00'``), and also selects the
        grid the ``with_absorption`` T/S column is drawn from, so the SSP and the
        absorption come from one cell; ``timeout`` is the network timeout in
        seconds of each attempt of each request (a failed attempt may be
        retried; a host that does not accept the connection within it is
        not) and does not reach the Copernicus fetchers — the
        ``copernicusmarine`` session owns its own (see
        :mod:`uacpy.data.copernicus`); ``verbose`` is the ``log_message`` gate,
        ``False``/``'off'``/``'silent'`` (warnings only), ``True``/``'info'``,
        or ``'debug'``.

    Returns
    -------
    Environment

    Warns
    -----
    FallbackWarning
        For each setting given that the run did not use: a transect count
        (``n_points``, ``ssp_n_points``, ``bottom_n_points``,
        ``surface_n_points``, ``max_points``) with no transect along that
        axis, ``altimetry_n_points`` / ``altimetry_rng`` with no sea surface
        fetched, ``bottom_model`` / ``bottom_hamilton_fit`` with no grain size
        converted, ``resolution`` with no WOA23 grid read, ``max_days`` with
        no time-specific source read (a fetched sea surface reports its own).
    ProvenanceWarning
        Once per call, when any fetched value is not the requested point's
        own measurement: one notice listing each source that stands far or
        stale from the point, each transect waypoint filled from a
        neighbour, each profile extended to the seafloor, and each licence
        that restricts use. A direct call to one fetcher (``fetch_ssp``,
        ``fetch_bottom``, ...) gives its own notices.
    """
    from uacpy.core.acoustics.seawater import canonical_formula
    formula = canonical_formula(formula, 'fetch_environment')
    _checked_optional_resolution(resolution, 'fetch_environment')
    woa_resolution = resolution or _DEFAULT_RESOLUTION
    lat, lon = as_coordinate(point)
    max_distance_km = checked_max_distance(max_distance_km, 'fetch_environment')
    if max_days is not None:
        max_days = checked_max_days(max_days, 'fetch_environment')
    if transect_to is not None:
        _check_transect_counts(max_points, n_points=n_points,
                               ssp_n_points=ssp_n_points,
                               bottom_n_points=bottom_n_points,
                               surface_n_points=surface_n_points)
    ssp_sources, bathymetry_sources, bottom_sources, surface_sources = (
        _checked_source_specs(ssp=ssp_sources, bathymetry=bathymetry_sources,
                              bottom=bottom_sources, surface=surface_sources))
    # The values as given, for the notice when no grain size is converted.
    given_model, given_fit = bottom_model, bottom_hamilton_fit
    bottom_model, bottom_hamilton_fit = _canonical_grain_size_selection(
        DEFAULT_GRAIN_SIZE_MODEL if bottom_model is None else bottom_model,
        bottom_hamilton_fit, who='fetch_environment',
        model_argument='bottom_model',
        environment_argument='bottom_hamilton_fit')
    _check_bottom_literal(bottom)
    surface_threshold = _checked_surface_threshold(
        surface_threshold, surface_sources, range_dependent_surface)

    # Each axis is a literal (ssp=/bathymetry=/bottom=) and/or fetched from
    # source(s) (*_sources). When both are given the source is fetched first and
    # the literal is the fallback if the fetch yields nothing (no coverage,
    # service down). Bathy/SSP are mandatory: with neither, fetch the default
    # chain.
    #
    # The axes below resolve in a fixed order set by their data dependencies:
    # bathymetry → SSP (reconciled to the fetched seafloor) → bottom (scaled to
    # the SSP at that seafloor) → surface → altimetry → assembly. The
    # range_dependent_* flags are settled in one block between bathymetry and
    # SSP, since they pick the point vs transect fetcher for the SSP, bottom and
    # surface alike; bathymetry takes no flag — transect_to alone decides it.

    # ── Bathymetry ──
    bathy = _fetch_chain_axis(
        _BATHY_CHAIN, bathymetry_sources, bathymetry,
        default=_DEFAULT_BATHY_SOURCES,
        where=(point,) if transect_to is None else (point, transect_to),
        request=dict(n_points=n_points, max_points=max_points,
                     max_distance_km=max_distance_km, timeout=timeout,
                     verbose=verbose),
        fallback_warning=(
            "fetch_environment: bathymetry fetch failed ({exc}); falling "
            "back to the supplied bathymetry= literal." + (
                "" if transect_to is None else
                " A range-independent literal reduces the transect to a "
                "single depth.")))

    plan = _resolve_range_dependence(
        transect_to, ssp=range_dependent_ssp, bottom=range_dependent_bottom,
        surface=range_dependent_surface, bottom_sources=bottom_sources,
        surface_sources=surface_sources)

    # ── SSP ──
    # The seafloor the SSP transect columns extend to, and later the carrier
    # the environment gets (a fetched one stamped by its fetcher).
    seafloor = Bathymetry.coerce(bathy.value)
    ssp_asked = (ssp_sources is not None or ssp is None
                 or range_dependent_ssp is True)
    fetched_ssp = _fetch_chain_axis(
        _SSP_CHAIN, ssp_sources, ssp,
        default=_DEFAULT_SSP_SOURCES,
        force_default=range_dependent_ssp is True,
        where=(point, transect_to) if plan.rd_ssp else (point,),
        request=dict(date=date, formula=formula, resolution=woa_resolution,
                     max_distance_km=max_distance_km, max_days=max_days,
                     n_points='auto' if ssp_n_points is None else ssp_n_points,
                     max_points=max_points,
                     seafloor=seafloor, timeout=timeout, verbose=verbose),
        fallback_warning=(
            "fetch_environment: SSP fetch failed ({exc}); falling back to the "
            "supplied ssp= literal." + (
                "" if not plan.rd_ssp else
                " A single-profile literal reduces the transect to "
                "range-independent.")))
    ssp = _reconcile_ssp(
        fetched_ssp, seafloor, point, date=date, formula=formula,
        resolution=woa_resolution, timeout=timeout, verbose=verbose)

    # ── Absorption (optional): the site's T/S column, whose water density
    # the grain-size seabed is converted against, so it precedes the bottom ──
    law, absorption_sources, water_density = (_fetch_absorption(
        point, date=date, ssp_source=fetched_ssp.source,
        ssp_backend=fetched_ssp.backend,
        cache_only=fetched_ssp.cache_only, resolution=woa_resolution,
        max_distance_km=max_distance_km, max_days=max_days,
        timeout=timeout, verbose=verbose, transect_to=transect_to,
        seafloor=seafloor) if with_absorption else (None, (), None))

    # ── Bottom (optional): fetch from bottom_sources / 'auto', else literal ──
    bottom_props, bottom_kw = _fetch_bottom_axis(
        plan, bottom, bottom_sources, ssp, seafloor, point, transect_to,
        model=bottom_model, hamilton_fit=bottom_hamilton_fit,
        water_density=water_density,
        max_distance_km=max_distance_km, n_points=bottom_n_points,
        max_points=max_points, timeout=timeout, verbose=verbose)

    # ── Surface (top boundary, optional): fetch sea ice, else literal ──
    surface_props, surface_src = _fetch_surface_axis(
        plan, surface, surface_sources, point, transect_to, date=date,
        n_points='auto' if surface_n_points is None else surface_n_points,
        max_points=max_points,
        threshold=surface_threshold, max_distance_km=max_distance_km)

    # ── Altimetry (sea surface, optional): fetch a wave/wind-driven surface ──
    altimetry_result = _fetch_altimetry_axis(
        altimetry, altimetry_sources, (lat, lon), transect_to, date=date,
        n_points=altimetry_n_points, rng=altimetry_rng, max_days=max_days,
        max_distance_km=max_distance_km, timeout=timeout, verbose=verbose)

    unused_setting_notices('fetch_environment', _unused_environment_settings(
        transect_to=transect_to, plan=plan,
        bathy_fetched=bathymetry_sources is not None or bathymetry is None,
        ssp_fetched=ssp_asked, ssp=ssp, fetched_ssp=fetched_ssp,
        absorption_sources=absorption_sources, bottom=bottom,
        bottom_props=bottom_props, bottom_kw=bottom_kw,
        altimetry_fetched=altimetry_sources is not None,
        n_points=n_points, max_points=max_points, ssp_n_points=ssp_n_points,
        bottom_n_points=bottom_n_points, surface_n_points=surface_n_points,
        altimetry_n_points=altimetry_n_points, altimetry_rng=altimetry_rng,
        bottom_model=given_model, bottom_hamilton_fit=given_fit,
        resolution=resolution, max_days=max_days))
    return _assemble(
        point, transect_to, name=name, date=date, seafloor=seafloor, ssp=ssp,
        bottom_props=bottom_props, bottom_kw=bottom_kw,
        surface_props=surface_props, surface_src=surface_src,
        altimetry=altimetry_result, bathy_src=bathy.source,
        ssp_src=fetched_ssp.source, absorption=law,
        absorption_sources=absorption_sources, water_density=water_density)


def _checked_surface_threshold(threshold, surface_sources,
                               range_dependent_surface):
    """``surface_threshold`` as the fetched surface takes it: ``None`` →
    :data:`~uacpy.data.seaice_local.SEA_ICE_EDGE_CONCENTRATION`; a fraction
    outside [0, 1] refused; a set value with no sea-ice surface fetched
    announced as unused."""
    if threshold is None:
        return SEA_ICE_EDGE_CONCENTRATION
    if not (isinstance(threshold, (int, float, np.integer, np.floating))
            and not isinstance(threshold, bool)
            and np.isfinite(threshold) and 0.0 <= threshold <= 1.0):
        raise ConfigurationError(
            f"fetch_environment: surface_threshold={threshold!r} is not a "
            f"concentration fraction in [0, 1].",
            remediation="Pass the lowest ice concentration taken as ice "
                        "cover as a fraction, e.g. 0.15 for 15 %.")
    if surface_sources is None and range_dependent_surface is not True:
        warnings.warn(
            f"fetch_environment: surface_threshold={threshold} was not used: "
            f"no sea-ice surface is fetched. Pass surface_sources='seaice' to "
            f"fetch one, or drop the argument.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
    return float(threshold)


def _unused_environment_settings(
        *, transect_to, plan, bathy_fetched, ssp_fetched, ssp, fetched_ssp,
        absorption_sources, bottom, bottom_props, bottom_kw,
        altimetry_fetched, n_points, max_points, ssp_n_points,
        bottom_n_points, surface_n_points, altimetry_n_points, altimetry_rng,
        bottom_model, bottom_hamilton_fit, resolution, max_days):
    """The ``(name, value, why)`` of each ``fetch_environment`` setting the
    run did not use, for :func:`~uacpy.data._chain.unused_setting_notices`
    (a value left unset gives no notice). A fetched sea surface reports an
    unused ``max_days`` itself, so none is listed when one was asked for."""
    unused = []
    if transect_to is None:
        why = "no transect_to is given, so nothing is sampled along a transect"
        unused += [(name, value, why) for name, value in (
            ('n_points', n_points), ('max_points', max_points),
            ('ssp_n_points', ssp_n_points),
            ('bottom_n_points', bottom_n_points),
            ('surface_n_points', surface_n_points))]
    else:
        if not bathy_fetched:
            unused.append(('n_points', n_points,
                           "the bathymetry= literal is used as given, so no "
                           "bathymetry transect is fetched"))
        if not (plan.rd_ssp and ssp_fetched):
            unused.append(('ssp_n_points', ssp_n_points,
                           "no range-dependent SSP is fetched"))
        if not plan.rd_bottom:
            unused.append(('bottom_n_points', bottom_n_points,
                           "no range-dependent seabed is fetched"))
        if not plan.rd_surface:
            unused.append(('surface_n_points', surface_n_points,
                           "no range-dependent sea-ice surface is fetched"))
    if not altimetry_fetched:
        why = "no altimetry_sources is given, so no sea surface is fetched"
        unused += [('altimetry_n_points', altimetry_n_points, why),
                   ('altimetry_rng', altimetry_rng, why)]
    if bottom_kw is not None:
        converted = _converts_a_grain_size(bottom_props)
        why = (f"source {bottom_kw!r} answered, and it carries measured "
               f"layer properties rather than converting a grain size")
    elif bottom_props is None:
        converted = False
        why = ("no seabed is fetched or given, so the generic default "
               "half-space stands and no grain size is converted")
    else:
        converted = not isinstance(bottom, (str, BoundaryProperties,
                                            SeabedColumn, Bottom))
        why = ("the bottom= literal carries its own numbers, so no grain "
               "size is converted")
    if not converted:
        unused += [('bottom_model', bottom_model, why),
                   ('bottom_hamilton_fit', bottom_hamilton_fit, why)]
    absorption_ids = {getattr(getattr(s, 'source', None), 'id', s)
                      for s in absorption_sources}
    ssp_ids = ({p.source.id for p in getattr(ssp, 'data_sources', ())}
               if fetched_ssp.fetched else set())
    if fetched_ssp.source != 'woa23' and 'woa23' not in ssp_ids | absorption_ids:
        unused.append(('resolution', resolution, "no WOA23 grid is read"))
    if (fetched_ssp.source not in ('argo', 'copernicus')
            and not altimetry_fetched):
        came = ('the ssp= literal is used' if fetched_ssp.source is None
                else 'the SSP came from WOA23, a climatology keyed on month')
        unused.append(('max_days', max_days,
                       f"{came}, and no other time-specific source is read"))
    return unused


def _check_transect_counts(max_points, **counts):
    """Refuse an invalid transect count, or one above an explicit
    ``max_points``, before any axis fetches — a ``*=`` literal stands in for
    data a valid request could not get, never for a refused argument."""
    max_points = checked_max_points(max_points, 'fetch_environment')
    for name, count in counts.items():
        if count is None:
            continue                 # unset: the fetcher's default, budgeted
        count = checked_n_points(count, 'fetch_environment', allow_auto=True,
                                 name=name)
        if count != 'auto':
            checked_explicit_n_points(count, max_points, 'fetch_environment',
                                      name=name)


def _checked_source_specs(**specs):
    """The four ``*_sources`` values, each lower-cased in its own shape,
    after refusing an explicit empty sequence on any of them — before any
    axis resolves or fetches, since it selects no source."""
    for axis, spec in specs.items():
        _require_nonempty_sources(spec, axis=axis)
    # Source names and presets are matched lower-case on every axis.
    return tuple(lower_source_spec(spec) for spec in specs.values())


class _AxisFetch(NamedTuple):
    """What a chained axis resolved to: the value, the catalogue id and
    backend that answered (``None`` for a literal), whether the chain was
    cache-only, and whether the value was fetched."""

    value: object
    source: Optional[str]
    backend: Optional[str]
    cache_only: bool
    fetched: bool


def _chain_call(chain, where, request):
    """``call(token)`` fetching one ``(source, backend)`` of ``chain``: the
    point fetch at ``where = (point,)``, or the transect when ``where`` holds
    both ends (:meth:`~uacpy.data._chain.SourceChain.fetch`)."""
    def call(token):
        src, backend = token
        return chain.fetch(src, backend, where, request,
                           who='fetch_environment',
                           keyword=f'{chain.axis}_sources=')
    return call


def _fetch_chain_axis(chain, spec, literal, *, default, where, request,
                      fallback_warning, force_default=False):
    """Resolve a mandatory chained axis (bathymetry, SSP).

    ``spec`` (a ``*_sources`` value) is fetched through ``chain``; without
    one, ``default`` is fetched when there is no ``literal`` (or when
    ``force_default``), else the literal is used as given. A failed fetch
    falls back to the literal with a ``FallbackWarning``,
    ``fallback_warning`` formatted with the failure as ``exc``, so the
    literal never stands in for requested data unannounced; with no literal,
    or when the failure is a :class:`MissingPrerequisiteError`, the failure
    propagates.
    """
    cache_only = False
    if spec is not None:
        sources, cache_only = chain.resolve(spec)
    elif literal is None or force_default:
        sources = default
    else:
        return _AxisFetch(literal, None, None, False, False)
    order = chain.attempts(sources, cache_only=cache_only)
    try:
        value, (src, backend) = _resolve_cached(
            _chain_call(chain, where, request), order, axis=chain.axis)
    except (DataFetchError, ConfigurationError) as exc:
        if literal is None or isinstance(exc, MissingPrerequisiteError):
            raise
        warnings.warn(fallback_warning.format(exc=exc),
                      FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
        return _AxisFetch(literal, None, None, cache_only, False)
    return _AxisFetch(value, src, backend, cache_only, True)


class _RangeDependence(NamedTuple):
    """Which axes are fetched range-dependent, and which optional axes are
    fetched at all."""

    rd_ssp: bool
    want_bottom: bool
    rd_bottom: bool
    want_surface: bool
    rd_surface: bool


def _resolve_range_dependence(transect_to, *, ssp, bottom, surface,
                              bottom_sources, surface_sources):
    """The ``range_dependent_*`` flags resolved against the transect.

    A transect makes each *fetched* axis range-dependent by default; a single
    point is always range-independent. ``range_dependent_*=True`` on a point
    is an error; ``=False`` forces a single representative sample even along a
    transect. SSP is always fetched (so a transect makes it range-dependent);
    bottom and surface are fetched only on request (their ``*_sources`` or an
    explicit ``True``), and become range-dependent when fetched on a transect.
    """
    for flag, axis in ((ssp, 'ssp'), (bottom, 'bottom'),
                       (surface, 'surface')):
        if flag is True and transect_to is None:
            raise ConfigurationError(
                f"fetch_environment: range_dependent_{axis}=True requires "
                f"transect_to=.",
                remediation="Pass transect_to=(lat, lon), or leave it unset "
                            "for a single point.")
    on_transect = transect_to is not None
    want_bottom = bottom_sources is not None or bottom is True
    want_surface = surface_sources is not None or surface is True
    return _RangeDependence(
        rd_ssp=on_transect and ssp is not False,
        want_bottom=want_bottom,
        rd_bottom=want_bottom and on_transect and bottom is not False,
        want_surface=want_surface,
        rd_surface=want_surface and on_transect and surface is not False,
    )


def _reconcile_ssp(fetched_ssp, seafloor, point, *, date, formula,
                   resolution, timeout, verbose):
    """The SSP reconciled to the seafloor.

    Bathymetry (GEBCO) and SSP (WOA/Copernicus) come from independent
    products, so their deepest points rarely coincide. A fetched profile is
    reconciled to span exactly the fetched water column with the carrier's
    own method (extend short profiles to the seafloor; trim points below it).
    It is not resampled onto a uniform grid — the native levels carry the
    real sampling, and each model owns SSP interpolation via its
    ``interp_ssp`` scheme. This precedes the bottom: grain-size geoacoustics
    are a velocity ratio against the water *at the interface*, so the
    reference sound speed has to come from the reconciled profile, not from
    the deepest analysed level. A literal ``ssp=`` passes straight to
    ``Environment``, which coerces a scalar / pairs / SoundSpeedProfile and
    reconciles its depth to the bathymetry.
    """
    ssp = fetched_ssp.value
    if not fetched_ssp.fetched:
        return ssp
    # Deepest point anywhere along the transect: the profile columns share one
    # depth axis, so it has to reach the deepest seafloor the run touches.
    depth_max = float(np.max(seafloor.depths))
    if fetched_ssp.source == 'argo':
        ssp = _deepen_cast_with_woa23(
            ssp, depth_max, point, date=date, formula=formula,
            resolution=resolution, cache_only=fetched_ssp.cache_only,
            timeout=timeout, verbose=verbose)
    lat, _lon = as_coordinate(point)
    return extend_ssp_below_data(ssp, depth_max, latitude=lat)


def _fetch_bottom_axis(plan, bottom, bottom_sources, ssp, seafloor, point,
                       transect_to, *, model, hamilton_fit, water_density,
                       max_distance_km, n_points, max_points, timeout,
                       verbose):
    """``(bottom, source id)``: fetched from ``bottom_sources`` / ``'auto'``
    when asked, else the ``bottom=`` literal, else ``(None, None)`` with a
    warning (the hamilton_fit then takes its generic half-space). A grain
    size is converted against ``water_density`` (``None``: the package's
    one water)."""
    lat, lon = as_coordinate(point)
    if not plan.want_bottom:
        if bottom is not None:
            return _resolve_bottom(
                bottom, water_sound_speed=_seabed_sound_speed(ssp, seafloor),
                water_density=water_density,
                model=model, hamilton_fit=hamilton_fit), None
        # The seabed is opt-in; without either argument the Environment takes
        # its generic half-space, which is no data and carries no provenance.
        warnings.warn(
            "fetch_environment: no bottom= or bottom_sources= given, so the "
            "seabed is uacpy's generic default half-space, not data about "
            "this site, and it appears in no provenance record. Pass "
            "bottom_sources='local' (cached data only) or 'auto' to fetch "
            "one, or bottom= a grain size (phi), a material name or a "
            "BoundaryProperties.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
        return None, None
    order, cache_only = _bottom_order(
        bottom_sources if bottom_sources is not None else 'auto')
    # Scale grain-size geoacoustics to the in-situ sound speed at the
    # seafloor (the conversion is a velocity ratio; the nominal 1500 m/s
    # default can be ~100 m/s off on a warm shelf / cold deep site).
    water_c = _seabed_sound_speed(ssp, seafloor)
    try:
        if plan.rd_bottom:
            return _fetch_bottom(
                order, point, transect_to, transect=True,
                cache_only=cache_only,
                water_sound_speed=_seabed_sound_speed_along(
                    ssp, seafloor, (lat, lon)),
                depth=_seabed_depth_along(seafloor, (lat, lon)),
                water_density=water_density,
                model=model, hamilton_fit=hamilton_fit,
                max_distance_km=max_distance_km,
                n_points=n_points, max_points=max_points,
                timeout=timeout, verbose=verbose,
            )
        return _fetch_bottom(
            order, point, transect=False, cache_only=cache_only,
            water_sound_speed=water_c, water_density=water_density,
            depth=float(seafloor.eval(range=0.0)),
            model=model, hamilton_fit=hamilton_fit,
            max_distance_km=max_distance_km,
            timeout=timeout, verbose=verbose,
        )
    except (DataFetchError, ConfigurationError) as exc:
        if bottom is None:
            raise
        uniform = ("" if not plan.rd_bottom or isinstance(bottom, Bottom)
                   else " A uniform literal makes the bottom "
                        "range-independent.")
        warnings.warn(
            f"fetch_environment: bottom fetch failed ({exc}); falling back "
            f"to the supplied bottom= literal.{uniform}",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
        return _resolve_bottom(  # fall back to the literal
            bottom, water_sound_speed=water_c, water_density=water_density,
            model=model, hamilton_fit=hamilton_fit), None


def _seaice_surface_point(_backend, point, *, date, threshold,
                          max_distance_km, **_request):
    """A :data:`_SURFACE_CHAIN` point fetch: the NSIDC ice canopy at
    ``point``, or ``None`` over open water."""
    from uacpy.data.seaice_local import fetch_sea_ice_surface
    return fetch_sea_ice_surface(point, date=date, threshold=threshold,
                                 max_distance_km=max_distance_km)


def _seaice_surface_transect(_backend, start, end, *, date, n_points,
                             max_points, threshold, max_distance_km,
                             **_request):
    """A :data:`_SURFACE_CHAIN` transect fetch: the range-dependent NSIDC
    sea-ice surface along ``start`` → ``end``."""
    from uacpy.data.seaice_local import sea_ice_surface_transect
    return sea_ice_surface_transect(start, end, date=date, n_points=n_points,
                                    max_points=max_points,
                                    threshold=threshold,
                                    max_distance_km=max_distance_km)


#: The top-boundary sources: the cached NSIDC sea-ice climatology alone.
_SURFACE_CHAIN = SourceChain(
    'surface',
    providers=(SourceProvider('seaice', (CACHE_BACKEND,),
                              _seaice_surface_point,
                              _seaice_surface_transect),),
    auto=('seaice',))


def _fetch_surface_axis(plan, surface, surface_sources, point, transect_to,
                        *, date, n_points, max_points, threshold,
                        max_distance_km=None):
    """``(surface, source id)``: NSIDC sea ice when asked, else the
    ``surface=`` literal.

    The only fetchable surface is NSIDC sea ice; a point classified as open
    water returns no boundary (the default free surface), with no
    provenance. ``range_dependent_surface`` implies the sea-ice source,
    mirroring how ``range_dependent_ssp`` / ``_bottom`` auto-fetch their axis
    along the transect.
    """
    if not plan.want_surface:
        return surface, None
    # The only surface source is the cached sea-ice climatology, so 'auto'
    # and 'local' (and the implied default) are identical here. An unknown
    # name is refused before the date is asked for.
    spec = 'auto' if surface_sources is None else surface_sources
    sources, cache_only = _SURFACE_CHAIN.resolve(spec)
    _SURFACE_CHAIN.attempts(sources, cache_only=cache_only,
                            who='fetch_environment',
                            keyword='surface_sources=')
    # A missing date is the caller's input, not a failed fetch: no
    # surface= literal stands in for it.
    if date is None:
        raise MissingPrerequisiteError(
            "fetch_environment: surface_sources='seaice' needs date= to "
            "pick the climatological sea-ice month.",
            remediation="Pass date='YYYY-MM-DD', or drop surface_sources= "
                        "(a surface= literal is then used as given).",
        )
    where = (point, transect_to) if plan.rd_surface else (point,)
    request = dict(date=date, n_points=n_points, max_points=max_points,
                   threshold=threshold, max_distance_km=max_distance_km)
    try:
        fetched, (src, _backend) = _SURFACE_CHAIN.answer(
            spec,
            lambda src, backend: _SURFACE_CHAIN.fetch(
                src, backend, where, request, who='fetch_environment',
                keyword='surface_sources='),
            who='fetch_environment', keyword='surface_sources=')
        # Open water (a point's None, a transect with no ice anywhere)
        # leaves the default free surface, with no provenance.
        if fetched is not None and (not plan.rd_surface
                                    or fetched.is_elastic):
            return fetched, src
    except (DataFetchError, ConfigurationError) as exc:
        if surface is None:
            raise
        warnings.warn(
            f"fetch_environment: sea-ice surface fetch failed ({exc}); "
            f"falling back to the supplied surface= literal.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
        return surface, None
    return None, None


def _fetch_altimetry_axis(altimetry, altimetry_sources, site, transect_to,
                          *, date, n_points, rng, max_days, timeout,
                          verbose, max_distance_km=None):
    """A wave/wind-driven sea surface when ``altimetry_sources`` asks for
    one, else the ``altimetry=`` literal.

    Sea state is live and time-specific, and the realization spans a range,
    so a fetched altimetry needs both a transect (for its extent) and a date.
    """
    if altimetry_sources is None:
        return altimetry
    from uacpy.data.sea_surface import fetch_sea_surface
    # A missing transect or date is the caller's input, not a failed fetch:
    # no altimetry= literal stands in for it.
    if transect_to is None:
        raise MissingPrerequisiteError(
            "fetch_environment: altimetry_sources requires transect_to= "
            "(the sea-surface realization spans the transect range).",
            remediation="Pass transect_to=(lat, lon), or drop "
                        "altimetry_sources= (an altimetry= literal is then "
                        "used as given).")
    if date is None:
        raise MissingPrerequisiteError(
            "fetch_environment: altimetry_sources needs date= (sea "
            "state is time-specific).",
            remediation="Pass date='YYYY-MM-DD'.")
    try:
        return fetch_sea_surface(
            site, date=date,
            rmax=transect_length(site, transect_to),
            n_points=n_points, rng=rng,
            source=altimetry_sources, max_days=max_days,
            max_distance_km=max_distance_km, timeout=timeout, verbose=verbose)
    except (DataFetchError, ConfigurationError) as exc:
        if altimetry is None:
            raise
        warnings.warn(
            f"fetch_environment: sea-surface fetch failed ({exc}); falling "
            f"back to the supplied altimetry= literal.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
        return altimetry


def _site_water_density(absorption):
    """The one water density (g/cm³) the decks carry for a fetched
    absorption: IES-80 of the T/S row nearest the column's mid-depth;
    ``None`` without a fetched Francois-Garrison law."""
    if not isinstance(absorption, FrancoisGarrison):
        return None
    column = absorption.profile_depths
    mid = mid_depth(column) if column is not None else 0.0
    return float(density(*_water_at(absorption, mid)))


def _assemble(point, transect_to, *, name, date, seafloor, ssp, bottom_props,
              bottom_kw, surface_props, surface_src, altimetry, bathy_src,
              ssp_src, absorption, absorption_sources, water_density):
    """The ``Environment`` of the resolved axes, with the site's absorption
    law, the ``absorption_sources`` its T/S and pH came from and its
    ``water_density`` when fetched, and its provenance."""
    lat, lon = as_coordinate(point)
    kwargs = dict(
        name=name or f"{lat:.3f}, {lon:.3f}",
        bathymetry=seafloor,          # the coerced carrier, provenance stamped
        ssp=ssp,
        # Stamp the geolocation + time so the fetched env carries its
        # provenance (survives env.copy()): the great-circle transect endpoints
        # when one was requested (``location`` then defaults to the midpoint),
        # else the single site point.
        location=(lat, lon) if transect_to is None else None,
        transect=((lat, lon), as_coordinate(transect_to))
        if transect_to is not None else None,
        date=date,
    )
    if bottom_props is not None:
        kwargs['bottom'] = bottom_props
    if surface_props is not None:
        kwargs['surface'] = surface_props
    if altimetry is not None:
        kwargs['altimetry'] = altimetry
    if absorption is not None:
        kwargs['absorption'] = absorption
    if water_density is not None:
        kwargs['water_density'] = water_density
    env = Environment(**kwargs)

    _record_provenance(env, bathy_src, ssp_src, bottom_kw,
                       bottom_props, surface_src, absorption_sources)
    return env


def _water_at(absorption, depth: float):
    """The ``(temperature, salinity)`` of a Francois-Garrison law at each
    property's sample nearest ``depth`` (its number when it has no
    profile)."""
    def nearest(value):
        if isinstance(value, float):                    # one water row
            return float(value)
        return float(value.values[int(np.argmin(np.abs(
            value.depths - float(depth))))])
    return nearest(absorption.temperature), nearest(absorption.salinity)


def _record_provenance(env, bathy_src, ssp_src, bottom_kw, bottom_props,
                       surface_src, absorption_sources=()):
    """Complete the provenance ``env.data_sources`` reads, so it lists the
    datasets in axis order
    (bathymetry → ssp → bottom → surface → absorption T/S → pH → altimetry),
    exact repeats removed. The ``*_src`` are catalogue ids; the fetched
    altimetry carries its own record, as every fetched carrier does.
    ``absorption_sources`` are the datasets the absorption's T/S column (WOA23
    beside a literal ``ssp=``) and pH came from, as :func:`_fetch_absorption`
    returns them.

    Each fetched carrier carries its own ``data_sources`` — a tuple of
    :class:`~uacpy.data.sources.DataProvenance` records holding the dataset plus
    the **actual** date/coordinates it returned (which can differ from what was
    requested). A carrier that wasn't stamped — a literal axis, or a fetcher
    that reports no date/coords — is stamped here with the bare catalogue id,
    so attribution is never lost. The T/S column and the pH are the
    absorption law's water: they go to ``env.absorption.data_sources``, so
    the law and its tables carry them. Warns on any non-commercial licence
    used."""
    def catalogue(ids):
        # The bare catalogue id in a DataProvenance, so env.data_sources is
        # uniformly DataProvenance — no date/coords, just the source, for a
        # carrier that arrives unstamped (the sea-ice canopy arrives stamped
        # with its climatology period by its own fetcher). A record the
        # fetcher already built (the pH node read) is kept as it is.
        return tuple(i if isinstance(i, DataProvenance) else
                     DataProvenance(source=SOURCES[i])
                     for i in ids if i is not None)

    if not env.bathymetry.data_sources and bathy_src is not None:
        env.bathymetry.data_sources = catalogue([bathy_src])
    if not env.ssp.data_sources and ssp_src is not None:
        env.ssp.data_sources = catalogue([ssp_src])
    bottom_ids = [bottom_kw] + (
        ['globsed']
        if getattr(bottom_props, 'sediment_thickness_source', None) == 'globsed'
        else [])
    if not env.bottom.data_sources and catalogue(bottom_ids):
        # A seabed's provenance is its half-spaces'.
        for column in env.bottom.columns:
            column.halfspace.data_sources = catalogue(bottom_ids)
    if not env.surface.data_sources and surface_src is not None:
        for node in env.surface.nodes:
            node.data_sources = catalogue([surface_src])
    if absorption_sources:
        env.absorption.data_sources = catalogue(absorption_sources)
    # A licence-restricted source must never enter a result silently: warn at
    # fetch time for any non-commercial dataset used (e.g. CRUST1.0). Driven off
    # the catalogue flag so future non-commercial sources are covered too.
    for src in {prov.source.id: prov.source
                for prov in env.data_sources}.values():
        if not src.commercial_use:
            provenance_notice(licence_notice_text(src))


def _fetch_absorption(point, *, date, ssp_source, ssp_backend, cache_only,
                      resolution, max_distance_km=None, max_days=None,
                      timeout, verbose, transect_to=None, seafloor=None):
    """Francois-Garrison absorption from the site's fetched T/S column.

    Returns ``(absorption, sources, water_density)``: ``sources`` names the
    datasets the T/S and pH came from, the catalogue id of the T/S dataset
    first (``'woa23'``, ``'copernicus'`` or ``'argo'``), then the record of
    each other column or pH node read; ``water_density`` is the IES-80
    density (g/cm³) of the start column's row nearest its mid-depth
    (:func:`_site_water_density`). Reuses the backend the SSP resolved to
    (``ssp_backend``) for the WOA23 T/S column, so a cache-resolved SSP draws
    its absorption from the same cached grid rather than re-fetching it live.
    ``None`` (literal SSP) reads the WOA23 column cache-first, through the
    public :func:`~uacpy.data.fetch_ts_profile`: the installed grid, then NCEI
    THREDDS.
    ``resolution`` is the same WOA23 grid the SSP was drawn from, so both come
    from one cell. ``cache_only`` (a ``*_sources='local'`` run) forces the local
    WOA23 grid so the T/S column never hits the network either — including when
    a cache-pinned SSP fell back to a literal. pH comes from the cached GLODAP
    grid when installed (``ph_source='glodap'``), else the model default. A
    fetched pH is declared ``ph_scale='total'`` so the absorption converts
    it to the NBS scale the formula was fitted on.

    On a transect (``transect_to``, with the fetched ``seafloor``), a start
    column that ends above the track's deepest seafloor sample is continued
    below its deepest level by the column read at that sample, from the same
    dataset: its T/S rows, and its GLODAP pH levels when the start's pH is a
    GLODAP column too. The law holds a column's end rows beyond its depths,
    so without this every depth below the start's seafloor would take the
    start's deepest water. When that column cannot be read, the start column
    stands alone and the call's notice says so.
    """
    read = dict(date=date, ssp_source=ssp_source, ssp_backend=ssp_backend,
                cache_only=cache_only, resolution=resolution,
                max_distance_km=max_distance_km, max_days=max_days,
                timeout=timeout, verbose=verbose)
    depths, temp, sal, ts_src, ts_record = _ts_column(point, **read)
    # GLODAP gives its pH column, (depth, pH) pairs on its own levels;
    # Copernicus BGC one value, read at the T/S column's mid-depth.
    z_arr = np.asarray(depths, dtype=float)
    ref_depth = mid_depth(z_arr) if z_arr.size else None
    pH, ph_src = _fetch_ph(point, date=date, ssp_source=ssp_source,
                           cache_only=cache_only, timeout=timeout,
                           verbose=verbose, reference_depth=ref_depth,
                           max_distance_km=max_distance_km, max_days=max_days,
                           ts_cell=getattr(ts_record, 'data_point', None))
    # GLODAP (pHtsinsitutp) and the Copernicus BGC ``ph`` field are on the
    # total scale; the formula was fitted on NBS. The model default that
    # stands in when neither is available is on the formula's own scale.
    scale = 'nbs' if ph_src is None else 'total'
    law = FrancoisGarrison.from_temperature_salinity(
        depths, temp, sal, pH=pH, ph_scale=scale)
    water_density = _site_water_density(law)
    ts_sources, ph_sources = [ts_src], [ph_src]

    deepest = _deepest_waypoint(point, transect_to, seafloor)
    if deepest is not None and z_arr.size and deepest.depth > z_arr.max():
        z_end = float(z_arr.max())
        where = (f"the track's {deepest.depth:.0f} m seafloor, "
                 f"{deepest.range / 1000.0:.1f} km along it")
        try:
            z_d, t_d, s_d, _, ts_record = _ts_column(deepest.point, **read)
        except (DataFetchError, ConfigurationError) as exc:
            provenance_fallback(
                f"fetch_environment: the absorption holds the start "
                f"column's deepest water below {z_end:.0f} m, because the "
                f"column at {where} could not be read ({exc.message}).")
            z_d = t_d = s_d = np.empty(0)
        below = np.asarray(z_d, dtype=float) > z_end
        if below.any():
            depths = np.concatenate([z_arr, np.asarray(z_d, float)[below]])
            temp = np.concatenate([np.asarray(temp, float),
                                   np.asarray(t_d, float)[below]])
            sal = np.concatenate([np.asarray(sal, float),
                                  np.asarray(s_d, float)[below]])
            ts_sources.append(ts_record)
            if isinstance(pH, np.ndarray):
                pH, ph_record = _deepened_ph(pH, deepest.point, where,
                                             max_distance_km=max_distance_km)
                ph_sources.append(ph_record)
            log_message('environment', f"absorption column continued below "
                        f"{z_end:.0f} m by the column at {where}",
                        verbose=verbose)
            law = FrancoisGarrison.from_temperature_salinity(
                depths, temp, sal, pH=pH, ph_scale=scale)
    sources = tuple(s for s in ts_sources + ph_sources if s is not None)
    return law, sources, water_density


def _ts_column(point, *, date, ssp_source, ssp_backend, cache_only,
               resolution, max_distance_km, max_days, timeout, verbose):
    """``(depths, temperature, salinity, source, record)`` of the T/S column
    at ``point`` that :func:`_fetch_absorption` builds the law from:
    ``source`` is the catalogue id of its dataset (``'woa23'``,
    ``'copernicus'`` or ``'argo'``) and ``record`` the provenance of the
    column read."""
    if cache_only:
        ts = _fetch_ts_profile_backend(
            point, date=date, backend='local', resolution=resolution,
            timeout=timeout, verbose=verbose, max_distance_km=max_distance_km,
            who='fetch_environment')
    elif ssp_source == 'copernicus':
        from uacpy.data.copernicus import fetch_ts_profile_operational
        extra = {} if max_days is None else {'max_days': max_days}
        # In-situ temperature, as Francois-Garrison takes it.
        ts = fetch_ts_profile_operational(
            point, date=date, verbose=verbose,
            max_distance_km=max_distance_km, **extra)
        return (ts.depths, ts.temperature, ts.salinity, 'copernicus',
                ts.provenance)
    elif ssp_source == 'argo':
        from uacpy.data.argo import fetch_argo_profile
        extra = {} if max_days is None else {'max_days': max_days}
        prof = fetch_argo_profile(point, date=date, timeout=timeout,
                                  verbose=verbose,
                                  max_distance_km=max_distance_km, **extra)
        return (pressure_dbar_to_depth(prof.pressure_dbar, prof.lat),
                prof.temperature, prof.salinity, 'argo', prof.provenance)
    elif ssp_backend is None:
        # Literal SSP: the public WOA23 fetcher is cache-first (the installed
        # grid, then NCEI THREDDS), as every fetched axis is.
        ts = fetch_ts_profile(
            point, date=date, resolution=resolution, source='woa23',
            timeout=timeout, verbose=verbose, max_distance_km=max_distance_km)
    else:
        ts = _fetch_ts_profile_backend(
            point, date=date, resolution=resolution, timeout=timeout,
            verbose=verbose, backend=ssp_backend,
            max_distance_km=max_distance_km, who='fetch_environment')
    return ts.depths, ts.temperature, ts.salinity, 'woa23', ts.provenance


class _Waypoint(NamedTuple):
    """A transect seafloor sample: its ``(lat, lon)``, its range (m) from
    the start and its depth (m)."""
    point: Coordinate
    range: float
    depth: float


def _deepest_waypoint(point, transect_to, seafloor):
    """The deepest sample of a transect's ``seafloor``; ``None`` at a point,
    and when the start is the deepest."""
    if transect_to is None or seafloor is None:
        return None
    ranges = np.asarray(seafloor.ranges, dtype=float)
    depths = np.asarray(seafloor.depths, dtype=float)
    k = int(np.argmax(depths))
    if ranges[k] <= 0.0:
        return None
    return _Waypoint(point_along_track(point, transect_to, ranges[k]),
                     float(ranges[k]), float(depths[k]))


def _deepened_ph(pairs, point, where, *, max_distance_km):
    """GLODAP ``(depth, pH)`` ``pairs`` continued below their deepest level
    by GLODAP's column at ``point`` (described by ``where``), and the record
    of the node read; ``pairs`` unchanged, and ``None``, when that column
    cannot be read or ends no deeper (the call's notice says so for the
    first)."""
    z_end = float(pairs[:, 0].max())
    try:
        deep, record = _glodap_ph_column(point,
                                         max_distance_km=max_distance_km)
    except (DataFetchError, ConfigurationError) as exc:
        provenance_fallback(
            f"fetch_environment: the absorption holds the start column's "
            f"deepest pH below {z_end:.0f} m, because GLODAP has no column "
            f"at {where} ({exc.message}).")
        return pairs, None
    below = deep[deep[:, 0] > z_end]
    if not below.size:
        return pairs, None
    return np.vstack([pairs, below]), record


def _glodap_ph_column(point, *, max_distance_km, read_at=None):
    """GLODAP's pH column at ``point`` as ``(depth, pH)`` pairs on its finite
    levels (an ``(N, 2)`` array), with the record of the node read — read at
    ``read_at`` instead when given, the record still measured from
    ``point``; ``DataFetchError`` / ``ConfigurationError`` when there is
    none."""
    from uacpy.data import glodap_local
    if read_at is None:
        profile = glodap_local.fetch_ph_profile(
            point, max_distance_km=max_distance_km)
    else:
        profile = glodap_local._ph_profile(
            point, max_distance_km=checked_max_distance(
                max_distance_km, 'fetch_environment'),
            who='fetch_environment', read_at=read_at)
    z = np.asarray(profile.depths, dtype=float)
    ph = np.asarray(profile.ph, dtype=float)
    keep = np.isfinite(z) & np.isfinite(ph)
    z_u, first = np.unique(z[keep], return_index=True)
    if z_u.size == 0:
        raise DataFetchError("no finite pH level in the column read.")
    return np.column_stack([z_u, ph[keep][first]]), profile.provenance


def _fetch_ph(point, *, date=None, ssp_source=None, cache_only=False,
              timeout=120.0, verbose=False, reference_depth=None,
              max_distance_km=None, max_days=None, ts_cell=None):
    """Representative seawater pH at ``point``, pH-source-aware and cache-first.

    Returns ``(pH, source)``: one value, or GLODAP's column as ``(depth,
    pH)`` pairs (an ``(N, 2)`` array on its finite levels; one value when it
    has a single level). On the Copernicus SSP branch (``ssp_source ==
    'copernicus'`` and not ``cache_only``) the time-varying BGC ``ph`` field is
    preferred (``source='copernicus_bgc'``), riding the same Copernicus
    session as the SSP, under the caller's ``max_days`` (``None``: the
    fetcher's own). Otherwise — or if the BGC fetch fails — it falls back to
    the cached GLODAP climatology, whose ``source`` is the
    :class:`DataProvenance` record of the node read (its point and offset),
    and finally to the model default (``REFERENCE_PH``, ``None``). Where
    GLODAP has no column at ``point`` (a coastal cell), the column at
    ``ts_cell`` — the cell the absorption's T/S column came from, which the
    WOA23 search may have taken from a wet neighbour — is read instead, so
    the T, S and pH share one location; its record measures the offset from
    ``point``. pH is cache/best-effort like every
    other axis: a run without any pH source keeps the default rather than
    failing, and so does one whose pH cells stand past ``max_distance_km`` —
    with a ``provenance_fallback`` item naming why, so the substituted pH is
    announced in the call's notice.
    """
    from uacpy.core.constants import REFERENCE_PH
    lat, lon = as_coordinate(point)
    failures = []

    def reference_ph():
        provenance_fallback(
            f"fetch_environment: the absorption takes the reference pH "
            f"{REFERENCE_PH:g} at ({lat:.4f}, {lon:.4f}), because no pH source "
            f"answered ({'; '.join(failures)}).")
        return REFERENCE_PH, None

    if ssp_source == 'copernicus' and not cache_only and date is not None:
        from uacpy.data.copernicus import fetch_ph_operational
        extra = {} if max_days is None else {'max_days': max_days}
        try:
            pH = fetch_ph_operational(point, date=date, verbose=verbose,
                                      reference_depth=reference_depth,
                                      max_distance_km=max_distance_km,
                                      **extra)
            log_message('copernicus', f"pH {pH:.3f} from Copernicus BGC at "
                        f"{lat:.3f}, {lon:.3f}", verbose=verbose)
            return pH, 'copernicus_bgc'
        except (DataFetchError, ConfigurationError) as exc:
            failures.append(f"Copernicus BGC: {exc.message}")  # then GLODAP
    try:
        pairs, record = _glodap_ph_column(point,
                                          max_distance_km=max_distance_km)
    except ConfigurationError as exc:                # no installed grid
        failures.append(f"GLODAP: {exc.message}")
        return reference_ph()
    except DataFetchError as exc:
        failures.append(f"GLODAP: {exc.message}")
        if ts_cell is None:
            return reference_ph()
        try:
            pairs, record = _glodap_ph_column(
                point, max_distance_km=max_distance_km, read_at=ts_cell)
        except (DataFetchError, ConfigurationError) as exc:
            failures.append(f"GLODAP at the T/S column's cell: {exc.message}")
            return reference_ph()
    log_message('glodap', f"pH {pairs[:, 1].min():.3f}-{pairs[:, 1].max():.3f} "
                f"over {len(pairs)} levels from GLODAP at {lat:.3f}, "
                f"{lon:.3f}", verbose=verbose)
    return (float(pairs[0, 1]) if len(pairs) == 1 else pairs), record


def _deepen_cast_with_woa23(cast, depth_max, point, *, date, formula,
                            resolution, cache_only, timeout, verbose):
    """An Argo cast continued below its deepest level by the WOA23 column,
    blended as :func:`~uacpy.data.deepen_with_climatology` blends it.

    A core Argo float profiles to about 2000 dbar. Below that the deep water
    keeps cooling, which the fixed-temperature extrapolation of
    :func:`~uacpy.data.extend_ssp_below_data` cannot follow; the WOA23 column
    at the same point and month has the analysed deep water.

    Returned unchanged when the cast already reaches the seafloor (to within
    the extension's own 50 m tolerance), when WOA23 cannot be read at the
    point, or when WOA23 ends no deeper than the cast; the caller's
    extrapolation then covers any remaining depth, with its warning.
    """
    from uacpy.data.sound_speed import (_EXTRAPOLATION_WARN_M,
                                        _blend_cast_onto_column)
    z_last = float(np.asarray(cast.depths, dtype=float)[-1])
    if depth_max - z_last <= _EXTRAPOLATION_WARN_M:
        return cast
    try:
        woa = fetch_ssp(point, date=date, formula=formula,
                        resolution=resolution,
                        source='local' if cache_only else 'woa23',
                        timeout=timeout, verbose=verbose)
    except (ConfigurationError, DataFetchError) as exc:
        log_message('sound_speed', f"Argo cast ends at {z_last:.0f} m and "
                    f"WOA23 is unavailable below it ({exc.message}); "
                    f"extrapolating instead", verbose=verbose)
        return cast
    return _blend_cast_onto_column(cast, woa, verbose=verbose)


# Bottom data sources — single declarative registry. To add a sediment source:
# write its module's point fetcher, then add one ``_BottomProvider`` row below.
# Everything else (the accepted source keywords, the 'auto' fallback order, the
# fetcher lookup and the provenance id) derives from this list.
#
# ``resolve`` returns the source's ``(point_fetcher, transect_fetcher)`` pair,
# imported lazily to keep optional deps optional and avoid import cycles. A
# half-space source's transect is its point fetcher sampled along the track
# (:func:`~uacpy.data.sediment.transect_fetcher`); CRUST1.0, whose columns are
# layered, has its own. Providers with a cached twin and a live backend
# (``has_cached_variant``: EMODnet's local polygons vs the live WFS, pelagic's
# local-GEBCO vs API-allowed depth lookup) take a ``cached`` flag and are tried
# cache-first; the rest take no argument. The rows become the providers of
# ``_BOTTOM_CHAIN``, the :class:`~uacpy.data._chain.SourceChain` every bottom
# fetch resolves through.


@dataclass(frozen=True)
class _BottomProvider:
    """One bottom data source. ``id`` doubles as the source keyword and the
    provenance catalogue id. ``backends`` are its backend tokens, its cached
    twin (:data:`~uacpy.data._chain.CACHE_BACKEND`) first; ``'local'`` keeps
    the cached ones alone. ``in_auto`` puts it in the 'auto' fallback chain
    (the network-free 'local' chain is that chain's cached backends)."""

    id: str
    resolve: Callable                  # () -> (point_fn, transect_fn); takes
                                       # (cached: bool) iff has_cached_variant
    backends: Tuple[str, ...] = (CACHE_BACKEND,)
    in_auto: bool = False
    accepts_depth: bool = False         # depth-driven source: takes the fetched water depth
    accepts_grain_size_model: bool = False  # converts a grain size: takes model=

    @property
    def has_cached_variant(self):
        """Whether it has a cached twin and a live backend, so ``resolve``
        takes ``cached``."""
        return len(self.backends) > 1


def _along_track(point_fn, label, preload=None):
    """``(point_fn, its transect)``: the point fetcher sampled along the
    track under ``label``."""
    return point_fn, transect_fetcher(point_fn, label, preload=preload)


def _emodnet_pair(cached):
    if cached:
        from uacpy.data import emodnet_local
        # The polygon index loads before any waypoint, so an absent or
        # unreadable cache raises its own error rather than a gap at each.
        return _along_track(emodnet_local.fetch_bottom_emodnet_local,
                            'EMODnet (offline)', preload=emodnet_local._index)
    from uacpy.data import seabed
    return _along_track(seabed.fetch_bottom_emodnet, 'EMODnet')


def _grainsize_pair():
    from uacpy.data import sediment_db as m
    # The sample index loads before any waypoint: its schema errors are plain
    # DataFetchErrors, which the sampler would otherwise count as a gap at
    # every waypoint and report as "no seabed anywhere".
    return _along_track(m.fetch_bottom_grainsize, 'local sediment DB',
                        preload=m._samples)


def _mars_pair():
    from uacpy.data import mars as m
    return _along_track(m.fetch_bottom_mars, 'AusSeabed MARS')


def _crust1_pair():
    from uacpy.data import crust1_local as m
    return (m.fetch_bottom_crust1, m.fetch_bottom_crust1_transect)


def _diesing_pair():
    from uacpy.data import diesing_local as m
    return _along_track(m.fetch_bottom_diesing, 'Diesing 2020')


def _graw_pair():
    from uacpy.data import graw_local as m
    return _along_track(m.fetch_bottom_graw, 'Graw density grid')


def _pelagic_pair(cached):
    from uacpy.data import pelagic as m
    # ``cached`` forbids the GEBCO live-API fallback in the depth lookup, so the
    # cached attempt (local GEBCO only) precedes the API-allowed one.
    return _along_track(partial(m.fetch_bottom_pelagic, cache_only=cached),
                        'pelagic model')


_BOTTOM_PROVIDERS = (
    _BottomProvider('emodnet', _emodnet_pair, (CACHE_BACKEND, 'live'),
                    in_auto=True,
                    accepts_grain_size_model=True),
    # A measured sample beats a modelled or interpolated map, so the
    # grain-size database sits directly behind EMODnet's polygons in 'auto'
    # too, not only in 'local': at 36 N 75 W it answers with a sand sample
    # 124 km away (rho*c 3608) where the pelagic rule gives ooze (2245).
    _BottomProvider('grainsize', _grainsize_pair, in_auto=True,
                    accepts_grain_size_model=True),
    _BottomProvider('crust1', _crust1_pair),
    _BottomProvider('graw', _graw_pair, accepts_grain_size_model=True),
    _BottomProvider('diesing', _diesing_pair, in_auto=True,
                    accepts_grain_size_model=True),
    # MARS is live-only, so it sits *after* the offline global Diesing map:
    # 'auto' consults the installed raster before any AusSeabed request, and
    # MARS then covers the Australian shelf Diesing (deep sea only) misses.
    _BottomProvider('mars', _mars_pair, ('live',), in_auto=True,
                    accepts_grain_size_model=True),
    _BottomProvider('pelagic', _pelagic_pair, (CACHE_BACKEND, 'live'),
                    in_auto=True,
                    accepts_depth=True,  # never fails (last resort)
                    accepts_grain_size_model=True),
)
_BOTTOM_BY_ID = {p.id: p for p in _BOTTOM_PROVIDERS}
_AUTO_BOTTOM_ORDER = tuple(p.id for p in _BOTTOM_PROVIDERS if p.in_auto)


def _bottom_fetch(source, transect, backend, *where, max_distance_km=None,
                  depth=None, model=None, hamilton_fit=None,
                  source_options=None, **kwargs):
    """One attempt of :data:`_BOTTOM_CHAIN`: ``source``'s point fetcher at
    ``where = (point,)`` or its transect fetcher (``transect``), its cached
    twin for ``backend`` :data:`~uacpy.data._chain.CACHE_BACKEND`. The
    keywords the chain supplies reach a fetcher only where it takes them (an
    offline backend takes no timeout); ``depth`` reaches the depth-driven
    sources and ``model`` / ``hamilton_fit`` the ones that convert a grain
    size; ``source_options`` were checked against every fetcher of the chain
    before any fetch."""
    provider = _BOTTOM_BY_ID[source]
    # Resolution happens per attempt, so a live backend's module is only
    # imported when the cached attempt has failed.
    point_fn, transect_fn = (provider.resolve(backend == CACHE_BACKEND)
                             if provider.has_cached_variant
                             else provider.resolve())
    fn = transect_fn if transect else point_fn
    call_kwargs = dict(_taken_by(fn, kwargs), **(source_options or {}))
    if max_distance_km is not None:
        call_kwargs['max_distance_km'] = max_distance_km
    if provider.accepts_depth and depth is not None:
        call_kwargs['depth'] = depth
    if provider.accepts_grain_size_model and model is not None:
        call_kwargs['model'] = model
        # Every provider that converts a grain size takes the
        # hamilton_fit with it; test_data_environment pins that the
        # two travel together, so a non-default hamilton_fit cannot
        # be dropped on the way to a fetcher.
        if hamilton_fit is not None:
            call_kwargs['hamilton_fit'] = hamilton_fit
    return fn(*where, **call_kwargs)


#: The bottom sources as one :class:`~uacpy.data._chain.SourceChain`: the
#: ``'auto'`` order of the rows above, ``'local'`` its cached backends.
_BOTTOM_CHAIN = SourceChain(
    'bottom',
    providers=tuple(SourceProvider(p.id, p.backends,
                                   partial(_bottom_fetch, p.id, False),
                                   partial(_bottom_fetch, p.id, True))
                    for p in _BOTTOM_PROVIDERS),
    auto=_AUTO_BOTTOM_ORDER)


def _bottom_order(bottom_source, who='fetch_environment',
                  keyword='bottom_sources='):
    """The bottom sources a user spec tries, in order, + a ``cache_only``
    flag. ``'auto'`` → the best-available chain (EMODnet → grain-size →
    Diesing → MARS → pelagic); ``'local'`` → the network-free chain (EMODnet
    local → grain-size → Diesing → pelagic, cached backends only); a
    str/sequence of keywords is used as-is. An unknown keyword is refused
    in the name of ``who``, the public function the user called, and the
    ``keyword`` the spec was passed as."""
    sources, cache_only = _BOTTOM_CHAIN.resolve(bottom_source)
    attempts = _BOTTOM_CHAIN.attempts(sources, cache_only=cache_only, who=who,
                                      keyword=keyword)
    return tuple(dict.fromkeys(src for src, _backend in attempts)), cache_only


#: Keywords ``_fetch_bottom`` supplies to the providers itself, so a
#: provider option of the same name is refused rather than overriding them.
_SUPPLIED_BOTTOM_KEYWORDS = frozenset({
    'cache_only', 'max_distance_km', 'depth', 'model',
    'water_sound_speed', 'water_density', 'roughness', 'n_points',
    'max_points', 'timeout', 'verbose'})


def _check_source_options(order, source_options, *, transect, who):
    """Refuse a provider option that a fetcher in ``order`` does not take.

    Every fetcher the chain could call (each cached and live twin) must
    accept every option, so an option is never dropped on the way to the
    source that answers."""
    import inspect
    for key in source_options:
        if key in _SUPPLIED_BOTTOM_KEYWORDS:
            raise ConfigurationError(
                f"{who}: {key!r} is set by {who} itself, not passed "
                f"through as a source option.",
                remediation=f"Pass {key}= through {who}'s own parameter, "
                            f"where it has one.")
    for name in order:
        provider = _BOTTOM_BY_ID[name]
        variants = (((True,), (False,)) if provider.has_cached_variant
                    else ((),))
        for resolve_args in variants:
            fn = provider.resolve(*resolve_args)[1 if transect else 0]
            # A sampled transect passes its options to the point fetcher.
            fn = getattr(fn, 'point_fetcher', fn)
            params = inspect.signature(fn).parameters
            unknown = sorted(k for k in source_options if k not in params)
            if unknown:
                accepted = sorted(
                    k for k, prm in params.items()
                    if prm.kind is prm.KEYWORD_ONLY
                    and k not in _SUPPLIED_BOTTOM_KEYWORDS)
                raise ConfigurationError(
                    f"{who}: bottom source {name!r} takes no option "
                    f"{', '.join(map(repr, unknown))}.",
                    remediation=(f"The options {name!r} takes: {accepted}. "
                                 f"Name only the sources that take the "
                                 f"option in source=.")
                    if accepted else
                    f"{name!r} takes no source options; drop them or name "
                    f"another source.")


def _fetch_bottom(order, *args, transect, cache_only=False,
                  max_distance_km=None, depth=None, model=None,
                  hamilton_fit=None, source_options=None,
                  who='fetch_environment', **kwargs):
    """Fetch a bottom from the first source in ``order`` that yields data.

    ``transect`` selects the point (``False``) or transect (``True``) fetcher.
    Cache-first within each source (EMODnet tries its local polygons before the
    live WFS); ``cache_only`` keeps only cached backends. ``args``/``kwargs``
    are forwarded; ``max_distance_km`` reaches every source, each refusing a
    data point farther than it from the point (the offset rule). ``depth`` is the water depth already fetched for this site, handed
    to the depth-driven sources (``accepts_depth``) so they classify off the
    same bathymetry the hamilton_fit uses rather than re-fetching their own.
    ``model`` names the grain-size relations and reaches the sources that
    convert a grain size (``accepts_grain_size_model``) — CRUST1.0 carries
    measured layer properties and takes none.
    ``source_options`` are provider options (e.g. CRUST1.0's attenuations),
    passed to every fetcher the chain calls; one that a fetcher in the chain
    does not take is refused in the name of ``who`` before any fetch.
    Returns ``(bottom, source_keyword)``; a source with no coverage (or no
    installed cache) falls through to the next.

    A transect over a chain of several half-space sources resolves **per
    waypoint** (:func:`_fetch_bottom_per_waypoint`), so a waypoint one source
    does not cover takes the next source's seabed there rather than a covered
    waypoint's seabed from elsewhere on the track.
    """
    per_waypoint = transect and len(order) > 1 and 'crust1' not in order
    if source_options:
        _check_source_options(order, source_options,
                              transect=transect and not per_waypoint,
                              who=who)
    if per_waypoint:
        return _fetch_bottom_per_waypoint(
            order, *args, cache_only=cache_only,
            max_distance_km=max_distance_km, depth=depth, model=model,
            hamilton_fit=hamilton_fit, source_options=source_options,
            who=who, **kwargs)

    request = dict(kwargs, max_distance_km=max_distance_km, depth=depth,
                   model=model, hamilton_fit=hamilton_fit,
                   source_options=source_options)
    # Cache-first within each source; a cached twin that read an answer ends
    # its provider, and only an absent or unreadable cache is worth asking
    # the live twin.
    bottom, (name, _backend) = first_answer(
        _BOTTOM_CHAIN.attempts(order, cache_only=cache_only, who=who),
        lambda src, backend: _BOTTOM_CHAIN.fetch(
            src, backend, args, request, who=who, keyword='source='))
    return bottom, name


def _taken_by(fn, kwargs):
    """``kwargs`` restricted to the keywords ``fn`` takes (all of them when
    it takes ``**kwargs``); a transect sampled from a point fetcher
    (:func:`~uacpy.data.sediment.transect_fetcher`) takes its own named
    ones and whatever its point fetcher takes."""
    import inspect
    names = set(inspect.signature(fn).parameters)
    reader = getattr(fn, 'point_fetcher', fn)
    params = inspect.signature(reader).parameters
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    names |= set(params)
    return {k: v for k, v in kwargs.items() if k in names}


def _fetch_bottom_per_waypoint(order, start, end, *, cache_only,
                               max_distance_km, depth, model, hamilton_fit,
                               water_sound_speed=None, n_points=None,
                               max_points=None, source_options=None,
                               who='fetch_environment', **kwargs):
    """Range-dependent bottom from a source chain, resolved at each waypoint.

    Every waypoint asks the chain in order and takes the first source that
    covers that waypoint, so the seabed at each range is the chain's own
    answer there, and each column carries the provenance of the source that
    supplied it (the transect's ``data_sources`` names every source used).
    Only a waypoint that *no* source in the chain covers is filled from the
    nearest covered waypoint, with the warning
    :func:`~uacpy.data.sediment.range_dependent_bottom_along` gives. CRUST1.0
    returns a layered column rather than a half-space and is not mixed into a
    chain this way.

    Returns ``(bottom, first_source)``, ``first_source`` being the chain's
    first source that answered anywhere on the track.
    """
    from uacpy.data.sediment import (range_dependent_bottom_along,
                                     water_sound_speed_at)
    used = []

    def point_bottom(lat, lon):
        bottom, name = _fetch_bottom(
            order, (lat, lon), transect=False, cache_only=cache_only,
            max_distance_km=max_distance_km,
            depth=depth(lat, lon) if callable(depth) else depth,
            model=model, hamilton_fit=hamilton_fit,
            water_sound_speed=water_sound_speed_at(water_sound_speed,
                                                   lat, lon),
            source_options=source_options, who=who, **kwargs)
        if name not in used:
            used.append(name)
        return bottom

    bottom = range_dependent_bottom_along(
        point_bottom, start, end, n_points,
        source_label=f"bottom chain {' -> '.join(order)}",
        max_points=max_points)
    return bottom, used[0]


def _bottom_request(source, who):
    """``(order, cache_only)`` for a public bottom fetcher's ``source``."""
    if not isinstance(source, str) and len(source_tuple(source)) == 0:
        raise ConfigurationError(
            f"{who}: source={source!r} selects no source.",
            remediation="Pass 'auto', 'local', a source id, or a non-empty "
                        "sequence of source ids.")
    return _bottom_order(lower_source_spec(source), who, keyword='source=')


@one_provenance_notice(subject="the seabed's data",
                       record='uacpy.data.citations(bottom)')
def fetch_bottom(
    point: Coordinate,
    *,
    source: Union[str, Sequence[str]] = 'auto',
    model: Optional[str] = None,
    hamilton_fit: Optional[str] = None,
    water_sound_speed: Optional[float] = None,
    depth: Optional[float] = None,
    roughness: float = 0.0,
    max_distance_km: Optional[float] = None,
    timeout: float = 60.0,
    verbose: Union[bool, str] = False,
    **source_options,
):
    """Model-ready seabed at a ``(lat, lon)`` point from one source or a chain.

    Parameters
    ----------
    point : (lat, lon)
        Latitude/longitude in decimal degrees (WGS84).
    source : str or sequence of str, optional
        ``'auto'`` (default: the best-available chain, cached first —
        EMODnet → grain-size → Diesing → MARS → pelagic), ``'local'`` (the
        network-free chain: EMODnet cached → grain-size → Diesing → pelagic),
        one source id — ``'emodnet'``, ``'grainsize'``, ``'diesing'``,
        ``'mars'``, ``'graw'``, ``'crust1'``, ``'pelagic'`` — or a sequence of
        them tried in order. The same values ``fetch_environment``'s
        ``bottom_sources=`` takes, resolved the same way.
    model, hamilton_fit, water_sound_speed
        The grain-size relations (``None``: :data:`~uacpy.core.sediment.DEFAULT_GRAIN_SIZE_MODEL`,
        ``'hamilton'``), Hamilton & Bachman fit and in-situ water
        sound speed (m/s) at the seabed, for every source that converts a
        grain size (see :func:`uacpy.core.sediment.grain_size_to_geoacoustics`);
        ``'crust1'`` carries measured layer properties, so any of the three
        set when it answers gives a ``FallbackWarning`` that it was not used.
    depth : float, optional
        Water depth (m) for the depth-driven ``'pelagic'`` source; fetched
        from GEBCO when ``None``. A given depth also stands in for the check
        that the point is wet (no GEBCO lookup), whichever source answers;
        set when another source answers, it gives a ``FallbackWarning``.
    roughness : float, optional
        RMS seabed roughness (m). Default 0.
    max_distance_km : float, optional
        Refuse a source whose data point (sample, or cell centre) stands
        farther than this (km) from ``point``; the next source in the chain
        is tried. ``None`` (default) refuses nothing beyond each source's
        ``ProvenanceWarning``.
    timeout, verbose
        Per-request network timeout (s) and the logging gate.
    **source_options
        Options of the source's own fetcher, passed through unchanged, e.g.
        ``source='crust1', sediment_attenuation=0.5, use_globsed=False``
        (CRUST1.0 carries no attenuation, so its nominal values are the ones
        to vary). Every source in the chain must take every option, else
        ``ConfigurationError`` before any fetch.

    Returns
    -------
    BoundaryProperties
        A half-space carrying the answering source's provenance on
        ``.data_sources`` (``'crust1'`` returns a layered
        :class:`~uacpy.core.environment.SeabedColumn`).

    Raises
    ------
    DataFetchError
        No source in the chain covers the point.
    ConfigurationError
        An unknown source, or no source whose cache is installed.
    """
    given_model = model
    model, hamilton_fit = _canonical_grain_size_selection(
        DEFAULT_GRAIN_SIZE_MODEL if model is None else model, hamilton_fit,
        who='fetch_bottom')
    order, cache_only = _bottom_request(source, 'fetch_bottom')
    max_distance_km = checked_max_distance(max_distance_km, 'fetch_bottom')
    if depth is None:
        _refuse_a_dry_point(point, who='fetch_bottom', cache_only=cache_only,
                            timeout=timeout, verbose=verbose)
    bottom, _source = _fetch_bottom(
        order, as_coordinate(point), transect=False, cache_only=cache_only,
        max_distance_km=max_distance_km, depth=depth, model=model,
        hamilton_fit=hamilton_fit, water_sound_speed=water_sound_speed,
        roughness=roughness, timeout=timeout, verbose=verbose,
        source_options=source_options, who='fetch_bottom')
    _unused_bottom_settings_notices(
        bottom, 'fetch_bottom', water_sound_speed=water_sound_speed,
        model=given_model, hamilton_fit=hamilton_fit, depth=depth)
    return bottom


@one_provenance_notice(subject="the seabed's data",
                       record='uacpy.data.citations(bottom)')
def fetch_bottom_transect(
    start: Coordinate,
    end: Coordinate,
    *,
    source: Union[str, Sequence[str]] = 'auto',
    n_points: Union[int, str, None] = None,
    max_points: Optional[int] = None,
    model: Optional[str] = None,
    hamilton_fit: Optional[str] = None,
    water_sound_speed=None,
    depth=None,
    roughness: float = 0.0,
    max_distance_km: Optional[float] = None,
    timeout: float = 60.0,
    verbose: Union[bool, str] = False,
    **source_options,
):
    """Range-dependent seabed along the great circle ``start`` → ``end``.

    ``source`` takes the values :func:`fetch_bottom` does. A chain is resolved
    at every waypoint — each takes the first source that covers it, and each
    column cites that source; a single source fills a waypoint it does not
    cover from the nearest covered one, with a warning. ``n_points`` is the
    waypoint count, used as given, or ``'auto'`` to probe ``max_points``
    waypoints and keep those bracketing each change of seabed.
    ``water_sound_speed`` and ``depth`` also take ``(lat, lon) -> value``
    callables, so each column scales to the water over its own seafloor. The
    remaining parameters are :func:`fetch_bottom`'s, ``**source_options``
    included.

    Parameters
    ----------
    start, end : (lat, lon)
        Transect endpoints in decimal degrees.
    source : str or sequence of str, optional
        As on :func:`fetch_bottom`. Default ``'auto'``.
    n_points : int or 'auto', optional
        Waypoints, or ``'auto'`` (see above). ``None`` (default) is 6 (uacpy
        guidance), within ``max_points`` when that is set.
    max_points : int, optional
        The sampling budget: waypoints ``'auto'`` probes, and the ceiling of
        an unset ``n_points``; ``None`` (default) is 1000 for ``'auto'``
        (uacpy guidance). An explicit ``n_points`` above an explicit
        ``max_points`` is refused.
    model, hamilton_fit : optional
        The grain-size relations and their Hamilton & Bachman fit, as on
        :func:`fetch_bottom`.
    water_sound_speed, depth : float or callable, optional
        As on :func:`fetch_bottom`, or ``(lat, lon) -> value`` callables
        evaluated per column.
    roughness : float, optional
        RMS seabed roughness (m). Default 0.
    max_distance_km : float, optional
        As in :func:`fetch_bottom`, at every waypoint.
    timeout : float, optional
        Per-request network timeout in seconds.
    verbose : bool or str, optional
        Logging gate passed through to ``log_message``.
    **source_options
        Options of the source's own fetcher, as on :func:`fetch_bottom`.

    Returns
    -------
    Bottom
        Ranges measured from ``start`` (``'crust1'`` returns its layered
        columns).
    """
    given_model = model
    model, hamilton_fit = _canonical_grain_size_selection(
        DEFAULT_GRAIN_SIZE_MODEL if model is None else model, hamilton_fit,
        who='fetch_bottom_transect')
    order, cache_only = _bottom_request(source, 'fetch_bottom_transect')
    max_distance_km = checked_max_distance(max_distance_km, 'fetch_bottom_transect')
    bottom, _source = _fetch_bottom(
        order, as_coordinate(start), as_coordinate(end), transect=True,
        cache_only=cache_only, max_distance_km=max_distance_km, depth=depth,
        model=model, hamilton_fit=hamilton_fit,
        water_sound_speed=water_sound_speed, roughness=roughness,
        n_points=n_points, max_points=max_points, timeout=timeout,
        verbose=verbose, source_options=source_options,
        who='fetch_bottom_transect')
    _unused_bottom_settings_notices(
        bottom, 'fetch_bottom_transect', water_sound_speed=water_sound_speed,
        model=given_model, hamilton_fit=hamilton_fit, depth=depth)
    return bottom


def _answering_ids(bottom):
    """The bottom-source ids a fetched seabed's records cite: one for a
    point, every source a chain resolved to along a transect."""
    if isinstance(bottom, Bottom):
        parts = [column.halfspace for column in bottom.columns]
    elif isinstance(bottom, SeabedColumn):
        parts = [bottom.halfspace]
    else:
        parts = [bottom]
    return sorted({p.source.id for part in parts for p in part.data_sources
                   if p.source.id in _BOTTOM_BY_ID})


def _converts_a_grain_size(bottom):
    """Whether a source that converts a grain size answered for some column
    of a fetched seabed (CRUST1.0 carries measured layer properties)."""
    return any(_BOTTOM_BY_ID[i].accepts_grain_size_model
               for i in _answering_ids(bottom))


def _unused_bottom_settings_notices(bottom, who, *, water_sound_speed,
                                    model, hamilton_fit, depth):
    """A ``FallbackWarning`` for each setting the caller gave that the
    answering sources do not read: the grain-size conversion's
    ``water_sound_speed``, ``model`` and ``hamilton_fit`` when none of them
    converts a grain size, ``depth`` when none is depth-driven (a given depth
    still stands in for the dry-point lookup)."""
    ids = _answering_ids(bottom)
    answered = ', '.join(repr(i) for i in ids)
    unused = []
    if not _converts_a_grain_size(bottom):
        why = (f"source {answered} answered, and it carries measured layer "
               f"properties rather than converting a grain size. Drop the "
               f"argument, or choose a source that converts a grain size")
        unused += [('water_sound_speed', water_sound_speed, why),
                   ('model', model, why), ('hamilton_fit', hamilton_fit, why)]
    if not any(_BOTTOM_BY_ID[i].accepts_depth for i in ids):
        unused.append(('depth', depth,
                       f"source {answered} answered, and only the "
                       f"depth-driven 'pelagic' source reads a depth (a "
                       f"given depth still stands in for the dry-point "
                       f"lookup)"))
    unused_setting_notices(who, unused)


def _seabed_sound_speed(ssp, seafloor, range_m=0.0):
    """In-water sound speed (m/s) at the seafloor under ``range_m``, or ``None``.

    Scales grain-size geoacoustics to the in-situ water *at the interface*
    rather than the nominal 1500 m/s default. ``ssp`` takes any form
    :class:`Environment` accepts (profile, scalar, ``(depth, c)`` pairs);
    ``seafloor`` is a :class:`Bathymetry`. A profile shallower than the seafloor
    holds its deepest value (the carrier's constant extrapolation).
    """
    if ssp is None:
        return None
    depth = float(seafloor.eval(range=float(range_m)))
    profile = SoundSpeedProfile.coerce(ssp, depth_max=depth)
    return float(profile.eval(depth=depth, range=float(range_m)).value)


def _seabed_sound_speed_along(ssp, seafloor, start):
    """``(lat, lon) -> seabed sound speed`` for the transect bottom fetchers.

    They sample at their own waypoints, so each seabed column is scaled to the
    water speed at *its* seafloor depth instead of one value from the start
    point. ``None`` when there is no usable SSP.
    """
    if ssp is None:
        return None
    lat0, lon0 = start

    def at(lat, lon):
        return _seabed_sound_speed(
            ssp, seafloor,
            range_m=great_circle_km(lat0, lon0, lat, lon) * 1000.0)
    return at


def _seabed_depth_along(seafloor, start):
    """``(lat, lon) -> water depth (m)`` for the depth-driven bottom fetchers.

    Reads the bathymetry already fetched for this environment, so the pelagic
    model classifies off the same seafloor the run uses instead of issuing its
    own GEBCO lookup per waypoint.
    """
    lat0, lon0 = start

    def at(lat, lon):
        return float(seafloor.eval(
            range=great_circle_km(lat0, lon0, lat, lon) * 1000.0))
    return at


# ϕ = -log2(d / 1 mm) (Ainslie, Principles of Sonar Performance Modelling,
# App. B and §4.4.1.2), so |ϕ| = 20 is a grain of about 1 km or 1 nm. The
# Udden-Wentworth classes of Ainslie Table 4.16 run from boulder gravel (below
# -8 ϕ) to fine clay (10 to 11 ϕ); a ``bottom=`` number past this limit is no
# grain size, and is most likely the half-space sound speed ``Environment``
# reads a number as.
_BOTTOM_PHI_LIMIT = 20.0


def _check_bottom_literal(bottom):
    """Refuse a ``bottom=`` literal of the wrong type, or a number that is no
    grain size (``|ϕ| > _BOTTOM_PHI_LIMIT`` or not finite)."""
    if bottom is None or isinstance(bottom, (str, BoundaryProperties,
                                             SeabedColumn, Bottom)):
        return
    # A NumPy scalar (a ϕ read from a NetCDF, a float32) is a number too;
    # a bool (an int subclass) is not, and NumPy's bool is no number type.
    if (not isinstance(bottom, (int, float, np.integer, np.floating))
            or isinstance(bottom, bool)):
        raise ConfigurationError(
            f"fetch_environment: bottom must be a ϕ float, a class name, a "
            f"BoundaryProperties, a SeabedColumn, a Bottom, or None; got "
            f"{type(bottom).__name__}.",
        )
    if not abs(float(bottom)) <= _BOTTOM_PHI_LIMIT:
        raise ConfigurationError(
            f"fetch_environment: a bottom= number is a mean grain size in ϕ, "
            f"and ϕ={bottom!r} is none (|ϕ| > {_BOTTOM_PHI_LIMIT:g} is a grain "
            f"coarser than 1 km or finer than 1 nm); it reads as a sound "
            f"speed, which fetch_environment does not take as a number.",
            remediation=(
                "Pass a sound-speed half-space as bottom=BoundaryProperties("
                "acoustic_type='half-space', sound_speed=..., density=...), "
                "or give the grain size in ϕ (e.g. bottom=2.0 for fine sand)."),
        )


def _resolve_bottom(bottom, *, water_sound_speed=None, water_density=None,
                    model=DEFAULT_GRAIN_SIZE_MODEL, hamilton_fit=None):
    """The ``bottom=`` literal as a seabed: a ready ``BoundaryProperties``,
    ``SeabedColumn`` or ``Bottom`` passes unchanged with its own provenance
    (``Environment`` coerces it), a class name takes its fluid preset (as
    ``Environment`` reads a name) and a ϕ float goes through the grain-size
    relations."""
    _check_bottom_literal(bottom)
    if bottom is None or isinstance(bottom, (BoundaryProperties, SeabedColumn,
                                             Bottom)):
        return bottom
    if isinstance(bottom, str):
        return BoundaryProperties.from_preset(bottom)
    return BoundaryProperties.from_grain_size(
        float(bottom), model=model, hamilton_fit=hamilton_fit,
        water_sound_speed=water_sound_speed, water_density=water_density)
