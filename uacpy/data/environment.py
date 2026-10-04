"""``fetch_environment`` — GPS (+ date) → a ready-to-run ``Environment``.

The capstone of the on-demand data layer: assemble bathymetry, sound speed
and (optionally) a bottom into a single :class:`~uacpy.core.environment.\
Environment` that drops straight into any propagation model.

    env = uacpy.data.fetch_environment((43.2, 7.5), date='2026-06-14',
                                       bottom='sand')

Each axis is handled the same way: supply it as a **literal** (``ssp=`` /
``bathymetry=`` / ``bottom=`` / ``surface=`` / ``altimetry=``, exactly as
:class:`Environment` takes them) and/or fetch it from one or more **sources**
(``ssp_sources`` / ``bathymetry_sources`` / ``bottom_sources`` /
``surface_sources``). If both are given for an axis the source is fetched first
and the literal is the **fallback** when the fetch yields nothing (no coverage,
service down). ``*_sources`` are ordered fallback lists (a bare string is a
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
from typing import Callable, NamedTuple, Optional, Sequence, Union

import numpy as np

from uacpy._log import log_message
from uacpy.core.absorption import FrancoisGarrison
from uacpy.core.acoustics import density
from uacpy.core.environment import (
    Bathymetry, Bottom, BoundaryProperties, Environment, SeabedColumn,
    SoundSpeedProfile,
)
from uacpy.core.exceptions import (
    ConfigurationError, DataFetchError, FallbackWarning, ProvenanceWarning,
)
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.geo import Coordinate, as_coordinate, great_circle_km
from uacpy.data._geo import (
    DEFAULT_MAX_TRANSECT_POINTS, checked_max_distance, checked_offset,
)
from uacpy.core.acoustics.seawater import pressure_dbar_to_depth
from uacpy.data._chain import first_answer, source_tuple
from uacpy.data.bathymetry import (
    _refuse_a_dry_point,
    DEFAULT_DATASET, _BATHY_CHAIN, transect_length,
)
from uacpy.core.sediment import (DEFAULT_GRAIN_SIZE_MODEL,
                                 canonical_grain_size_selection)
from uacpy.data.sediment import (bottom_from_class, bottom_from_grain_size,
                                  transect_fetcher)
from uacpy.data.sound_speed import (
    DEFAULT_SOUND_SPEED_FORMULA, _SSP_CHAIN, _fetch_ts_profile_backend,
    extend_ssp_below_data, fetch_ssp, fetch_ts_profile,
)
from uacpy.data.sources import SOURCES, DataProvenance
from uacpy.data._provenance_notice import one_provenance_notice

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


def _lower_source_spec(spec):
    """Lower-case a ``*_sources`` value in place of its own shape: ``None``
    stays ``None``, a preset or single name stays a ``str``, a sequence
    becomes a tuple of names — so the preset checks and the per-axis
    resolvers downstream all compare against one spelling."""
    if spec is None:
        return None
    if isinstance(spec, str):
        return spec.lower()
    return source_tuple(spec)


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
    bottom_model: str = DEFAULT_GRAIN_SIZE_MODEL,
    bottom_environment: Optional[str] = None,
    surface: Optional[BoundaryProperties] = None,
    surface_sources: Union[str, Sequence[str], None] = None,
    altimetry=None,
    altimetry_sources: Optional[str] = None,
    altimetry_n_points: Optional[int] = None,
    altimetry_rng: Optional[np.random.Generator] = None,
    transect_to: Optional[Coordinate] = None,
    n_points: Union[int, str] = 50,
    max_points: int = DEFAULT_MAX_TRANSECT_POINTS,
    range_dependent_ssp: Optional[bool] = None,
    ssp_n_points: Union[int, str] = 'auto',
    range_dependent_bottom: Optional[bool] = None,
    bottom_n_points: Union[int, str] = 6,
    range_dependent_surface: Optional[bool] = None,
    surface_n_points: Union[int, str] = 'auto',
    with_absorption: bool = False,
    max_distance_km: Optional[float] = None,
    max_days: Optional[int] = None,
    formula: str = DEFAULT_SOUND_SPEED_FORMULA,
    resolution: str = '1.00',
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
        m/s → isovelocity, or ``(depth_m, c_m_per_s)`` pairs). If ``ssp_sources``
        is *also* given, the source is fetched first and this literal is the
        **fallback** when the fetch yields nothing; on its own, SSP is not
        fetched at all.
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
        ``(range_m, depth_m)`` array — depth positive down — supplied directly.
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
        (default) is the low-frequency Hamilton & Bachman table;
        ``'apl-uw'`` the APL-UW TR 9407 relations for 10-100 kHz work, the
        same seabed the high-frequency scattering models in
        :mod:`uacpy.sonar` are built on (``BottomParameters.from_environment``
        reads it back). Both scale the sediment SOUND SPEED to the in-situ
        water at the seafloor; the DENSITY ratio is carried against
        ``DEFAULT_WATER_DENSITY_G_CM3`` (1.027 g/cm³), while the deck's water
        density is the fetched IES-80 value when absorption is fetched, so the
        seabed/water density ratio the deck sees is off the published one by
        1.027/rho_w (+0.4 % in 25 degC surface water, about 0.03 dB of bottom
        loss). A
        class-name literal, a ``BoundaryProperties`` and the hard-substrate
        presets some sources return (EMODnet rock, DECK41 ``'rock'``) carry
        their own numbers and are not affected. See
        :func:`uacpy.core.sediment.grain_size_to_geoacoustics`.
    transect_to : (lat, lon), optional
        If given, bathymetry is sampled along the great-circle path from
        ``(lat, lon)`` to here (range-dependent); otherwise a single depth.
    n_points : int or 'auto', optional
        Bathymetry transect sample count (used only with ``transect_to``).
        Default 50. ``'auto'`` targets GEBCO native resolution (bathymetry is
        continuous, so it is not duplicate-collapsed), bounded by ``max_points``.
    max_points : int, optional
        Ceiling on the points sampled along a transect *before* the ``'auto'``
        reduction — the fetch budget; the reduced grid is never larger. Default
        1000. Applies to bathymetry, SSP, and bottom transects.
    range_dependent_ssp : bool, optional
        Whether the SSP varies along the transect. **Default (``None``): a
        transect makes the SSP range-dependent** (a single point is always
        range-independent — there is nothing to vary). Pass ``False`` to force a
        single profile at the start point even along a transect; ``True`` on a
        point raises. Built from ``'woa23'`` or ``'copernicus'``; an ``'argo'``
        cast is a single profile, so a chain containing it falls through to the
        next source on a transect.
    ssp_n_points : int or 'auto', optional
        SSP columns along the transect when ``range_dependent_ssp``. Default
        ``'auto'``: the transect is sampled at the **distinct WOA23 cells** it
        crosses (one column per cell — WOA's native range resolution, found
        analytically so no duplicate column is fetched), capped at
        ``max_points``. For the Copernicus source, which exposes no cheap cell
        identity, ``'auto'`` maps to 6 evenly-spaced columns (capped at
        ``max_points``). Pass an int for exactly that many evenly-spaced
        columns.
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
        Default 6 (explicit) — unlike SSP, the bottom sources expose no cheap
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
        Default ``'auto'``: probe the local NSIDC grid (cheap) and collapse
        each run of identical ice/open-water zones to the samples bracketing
        its edges (the marginal ice zone at native scale, each edge within one
        probe step, no staircase), capped at ``max_points``. Pass an int for
        exactly that many waypoints.
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
        ``date``'s month — an ice-covered point (≥15 %, the NSIDC ice-edge) gets
        a homogeneous elastic ice canopy (cp 3500 m/s, cs 1800 m/s, ρ 0.9 g/cm³,
        αp/αs 0.4/1.0 dB/λ — *Computational Ocean Acoustics*); open water keeps
        the default free surface (no provenance). The canopy is an elastic
        **half-space**, not a plate of finite thickness: the model writers emit
        the top boundary as an upper half-space line, with no thickness field to
        give it, so concentration decides *whether* there is ice, never how
        thick. Default ``None`` (no surface fetch). Point classification only
        (the carrier's surface is one boundary). The fetched canopy is the
        smooth plate the tabulated parameters describe (roughness 0), which
        under-predicts the loss over real deformed pack ice (see
        :func:`uacpy.data.sea_ice_surface`); a site roughness needs an
        explicit ``surface=`` literal, e.g. built with
        ``sea_ice_surface(conc, roughness=...)``.
    altimetry : array-like, optional
        A **literal** rough-surface wave profile ``[(range_m, height_m), …]``
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
    bottom_environment : str, optional
        Which of Hamilton & Bachman's three fits ``bottom_model=DEFAULT_GRAIN_SIZE_MODEL``
        evaluates: ``'continental-terrace'`` (the default when ``None``, shelf
        and slope, 1 to 9 ϕ), ``'abyssal-hill'`` or ``'abyssal-plain'`` (both
        7 to 10 ϕ). It reaches the ϕ ``bottom=`` literal and every fetched
        source that converts a grain size. **A deep-ocean site wants an abyssal
        fit** — at 9 ϕ the terrace fit gives ρ = 1.45 against the 1.35 and 1.41
        Hamilton & Bachman measured for abyssal clay — but nothing infers it:
        the paper states no rule for choosing, so only the caller knows the
        site. Pairing it with ``bottom_model='apl-uw'`` raises, that report
        publishing one set of relations rather than one per environment.
    with_absorption : bool, optional
        If ``True``, attach a Francois-Garrison absorption built from the
        site's fetched temperature/salinity column (costs one extra T/S
        request): the whole column, as a profile every engine evaluates at
        each depth with the water there
        (:meth:`~uacpy.core.absorption.FrancoisGarrison.from_temperature_salinity`).
        pH is the cached GLODAP column, as ``(depth, pH)`` pairs on its own
        levels, when installed (``install.sh --data glodap``); one value at
        the column's mid-depth from Copernicus BGC on that branch; else the
        reference-water default (8.0). The water density is the mid-depth
        row's.
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
        Maximum days the data used may differ from ``date`` for the **time-
        specific** SSP sources — ``ssp_sources`` ``'argo'`` (nearest float
        profile) and ``'copernicus'`` (nearest model time step). A source whose
        nearest match is staler raises ``DataFetchError`` and the chain falls
        through (e.g. ``'auto'`` → WOA23). Ignored by WOA23 (a climatology
        keyed on month only) and by the bottom/bathymetry sources. Default
        ``None`` → each source's own default (Argo 15, Copernicus 31).
    formula, resolution, timeout, verbose
        Forwarded to the sound-speed / bathymetry fetchers. ``formula`` is the
        sound-speed equation, ``{'teos10', 'unesco', 'delgrosso', 'mackenzie'}``; ``resolution`` is the
        WOA23 grid spacing in degrees, ``{'1.00', '0.25'}``, and also selects the
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
    lat, lon = as_coordinate(point)
    max_distance_km = checked_max_distance(max_distance_km, 'fetch_environment')
    ssp_sources, bathymetry_sources, bottom_sources, surface_sources = (
        _checked_source_specs(ssp=ssp_sources, bathymetry=bathymetry_sources,
                              bottom=bottom_sources, surface=surface_sources))
    bottom_model, bottom_environment = canonical_grain_size_selection(
        bottom_model, bottom_environment, who='fetch_environment',
        model_argument='bottom_model',
        environment_argument='bottom_environment')

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
                     timeout=timeout, verbose=verbose),
        fallback_warning=None if transect_to is None else (
            "fetch_environment: range-dependent bathymetry fetch failed "
            "({exc}); falling back to the supplied bathymetry= literal. A "
            "range-independent literal reduces the transect to a single "
            "depth."))

    plan = _resolve_range_dependence(
        transect_to, ssp=range_dependent_ssp, bottom=range_dependent_bottom,
        surface=range_dependent_surface, bottom_sources=bottom_sources,
        surface_sources=surface_sources)

    # ── SSP ──
    # The seafloor the SSP transect columns extend to, and later the carrier
    # the environment gets, provenance stamped.
    seafloor = Bathymetry.coerce(bathy.value)
    fetched_ssp = _fetch_chain_axis(
        _SSP_CHAIN, ssp_sources, ssp,
        default=_DEFAULT_SSP_SOURCES,
        force_default=range_dependent_ssp is True,
        where=(point, transect_to) if plan.rd_ssp else (point,),
        request=dict(date=date, formula=formula, resolution=resolution,
                     max_distance_km=max_distance_km, max_days=max_days,
                     n_points=ssp_n_points, max_points=max_points,
                     seafloor=seafloor, timeout=timeout, verbose=verbose),
        fallback_warning=None if not plan.rd_ssp else (
            "fetch_environment: range-dependent SSP fetch failed ({exc}); "
            "falling back to the supplied ssp= literal. A single-profile "
            "literal reduces the transect to range-independent."))
    ssp = _reconcile_ssp(
        fetched_ssp, seafloor, bathy, point, date=date, formula=formula,
        resolution=resolution, max_distance_km=max_distance_km,
        timeout=timeout, verbose=verbose)

    # ── Bottom (optional): fetch from bottom_sources / 'auto', else literal ──
    bottom_props, bottom_kw = _fetch_bottom_axis(
        plan, bottom, bottom_sources, ssp, seafloor, point, transect_to,
        model=bottom_model, hamilton_fit=bottom_environment,
        max_distance_km=max_distance_km, n_points=bottom_n_points,
        max_points=max_points, timeout=timeout, verbose=verbose)

    # ── Surface (top boundary, optional): fetch sea ice, else literal ──
    surface_props, surface_src = _fetch_surface_axis(
        plan, surface, surface_sources, point, transect_to, date=date,
        n_points=surface_n_points, max_points=max_points,
        max_distance_km=max_distance_km)

    # ── Altimetry (sea surface, optional): fetch a wave/wind-driven surface ──
    altimetry_result = _fetch_altimetry_axis(
        altimetry, altimetry_sources, (lat, lon), transect_to, date=date,
        n_points=altimetry_n_points, rng=altimetry_rng, max_days=max_days,
        max_distance_km=max_distance_km, timeout=timeout, verbose=verbose)

    return _assemble(
        point, transect_to, name=name, date=date, seafloor=seafloor, ssp=ssp,
        bottom_props=bottom_props, bottom_kw=bottom_kw,
        surface_props=surface_props, surface_src=surface_src,
        altimetry=altimetry_result, bathy_src=bathy.source,
        fetched_ssp=fetched_ssp, with_absorption=with_absorption,
        resolution=resolution, max_distance_km=max_distance_km,
        max_days=max_days, timeout=timeout, verbose=verbose)


def _checked_source_specs(**specs):
    """The four ``*_sources`` values, each lower-cased in its own shape,
    after refusing an explicit empty sequence on any of them — before any
    axis resolves or fetches, since it selects no source."""
    for axis, spec in specs.items():
        _require_nonempty_sources(spec, axis=axis)
    # Source names and presets are matched lower-case on every axis.
    return tuple(_lower_source_spec(spec) for spec in specs.values())


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
    both ends."""
    def call(token):
        src, backend = token
        provider = chain.provider(src)
        if len(where) == 1:
            return provider.point(backend, *where, **request)
        if provider.transect is None:
            usable = ' or '.join(repr(p.id) for p in chain.providers
                                 if p.transect is not None)
            raise ConfigurationError(
                f"fetch_environment: range_dependent_{chain.axis} not "
                f"supported for {chain.axis}_sources={src!r}.",
                remediation=f"Use {usable}.",
            )
        return provider.transect(backend, *where, **request)
    return call


def _fetch_chain_axis(chain, spec, literal, *, default, where, request,
                      force_default=False, fallback_warning=None):
    """Resolve a mandatory chained axis (bathymetry, SSP).

    ``spec`` (a ``*_sources`` value) is fetched through ``chain``; without
    one, ``default`` is fetched when there is no ``literal`` (or when
    ``force_default``), else the literal is used as given. A failed fetch
    falls back to the literal, warning ``fallback_warning`` (formatted with
    the failure as ``exc``) when one is given; with no literal the failure
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
        if literal is None:
            raise
        if fallback_warning is not None:
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


def _reconcile_ssp(fetched_ssp, seafloor, bathy, point, *, date, formula,
                   resolution, timeout, verbose, max_distance_km=None):
    """The SSP reconciled to the seafloor, with the seafloor's provenance
    stamped.

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
    if bathy.source is not None and not seafloor.data_sources:
        seafloor.data_sources = (_bathymetry_provenance(
            bathy, point, max_distance_km=max_distance_km),)
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
                       transect_to, *, model, hamilton_fit, max_distance_km,
                       n_points, max_points, timeout, verbose):
    """``(bottom, source id)``: fetched from ``bottom_sources`` / ``'auto'``
    when asked, else the ``bottom=`` literal, else ``(None, None)`` with a
    warning (the hamilton_fit then takes its generic half-space)."""
    lat, lon = as_coordinate(point)
    if not plan.want_bottom:
        if bottom is not None:
            return _resolve_bottom(
                bottom, water_sound_speed=_seabed_sound_speed(ssp, seafloor),
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
                model=model, hamilton_fit=hamilton_fit,
                max_distance_km=max_distance_km,
                n_points=n_points, max_points=max_points,
                timeout=timeout, verbose=verbose,
            )
        return _fetch_bottom(
            order, point, transect=False, cache_only=cache_only,
            water_sound_speed=water_c,
            depth=float(seafloor.eval(range=0.0)),
            model=model, hamilton_fit=hamilton_fit,
            max_distance_km=max_distance_km,
            timeout=timeout, verbose=verbose,
        )
    except (DataFetchError, ConfigurationError) as exc:
        if bottom is None:
            raise
        if plan.rd_bottom:
            uniform = ("" if isinstance(bottom, Bottom) else
                       " A uniform literal makes the bottom "
                       "range-independent.")
            warnings.warn(
                f"fetch_environment: range-dependent bottom fetch failed "
                f"({exc}); falling back to the supplied bottom= "
                f"literal.{uniform}",
                FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
        return _resolve_bottom(  # fall back to the literal
            bottom, water_sound_speed=water_c, model=model,
            hamilton_fit=hamilton_fit), None


def _fetch_surface_axis(plan, surface, surface_sources, point, transect_to,
                        *, date, n_points, max_points, max_distance_km=None):
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
    # and 'local' (and the implied default) are identical here.
    srcs = (('seaice',) if surface_sources in ('auto', 'local', None)
            else source_tuple(surface_sources))
    for s in srcs:
        if s != 'seaice':
            raise ConfigurationError(
                f"fetch_environment: unknown surface source {s!r}.",
                remediation="Use 'seaice' or 'auto'.",
            )
    # The date guard sits inside the try so a supplied surface= literal is
    # the fallback for a missing date, like any other fetch failure.
    try:
        if date is None:
            raise ConfigurationError(
                "fetch_environment: surface_sources='seaice' needs date= to "
                "pick the climatological sea-ice month.",
                remediation="Pass date='YYYY-MM-DD', or supply surface= / "
                            "drop it.",
            )
        if plan.rd_surface:
            from uacpy.data.seaice_local import sea_ice_surface_transect
            fetched = sea_ice_surface_transect(
                point, transect_to, date=date,
                n_points=n_points, max_points=max_points,
                max_distance_km=max_distance_km)
            # A transect that is open water everywhere → leave the default
            # free surface (no provenance), matching the point case.
            if fetched.is_elastic:
                return fetched, 'seaice'
        else:
            from uacpy.data.seaice_local import fetch_sea_ice_surface
            fetched = fetch_sea_ice_surface(point, date=date,
                                            max_distance_km=max_distance_km)
            if fetched is not None:                 # None = open water
                return fetched, 'seaice'
    except (DataFetchError, ConfigurationError):
        if surface is None:
            raise
        return surface, None                        # literal fallback
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
    # The transect/date guards sit inside the try so a supplied altimetry=
    # literal is the fallback for a missing prerequisite, like any other
    # fetch failure.
    try:
        if transect_to is None:
            raise ConfigurationError(
                "fetch_environment: altimetry_sources requires transect_to= "
                "(the sea-surface realization spans the transect range).",
                remediation="Pass transect_to=(lat, lon), or supply "
                            "altimetry= directly for a single point.")
        if date is None:
            raise ConfigurationError(
                "fetch_environment: altimetry_sources needs date= (sea "
                "state is time-specific).",
                remediation="Pass date='YYYY-MM-DD'.")
        return fetch_sea_surface(
            site, date=date,
            rmax_m=transect_length(site, transect_to),
            n_points=n_points, rng=rng,
            source=altimetry_sources, max_days=max_days,
            max_distance_km=max_distance_km, timeout=timeout, verbose=verbose)
    except (DataFetchError, ConfigurationError):
        if altimetry is None:
            raise
        return altimetry                            # literal fallback


def _assemble(point, transect_to, *, name, date, seafloor, ssp, bottom_props,
              bottom_kw, surface_props, surface_src, altimetry, bathy_src,
              fetched_ssp, with_absorption, resolution, max_distance_km,
              max_days, timeout, verbose):
    """The ``Environment`` of the resolved axes, with the site's absorption
    and water density when ``with_absorption``, and its provenance."""
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
    ph_src, ts_src = None, None
    if with_absorption:
        kwargs['absorption'], ts_src, ph_src = _fetch_absorption(
            point, date=date, ssp_source=fetched_ssp.source,
            ssp_backend=fetched_ssp.backend,
            cache_only=fetched_ssp.cache_only, resolution=resolution,
            max_distance_km=max_distance_km, max_days=max_days,
            timeout=timeout, verbose=verbose,
        )
    absorption = kwargs.get('absorption')
    if isinstance(absorption, FrancoisGarrison):
        # The T/S row nearest the column's mid-depth sets the one water
        # density the decks carry (IES-80, kg/m³ -> g/cm³).
        column = absorption.profile_depths
        mid = (0.5 * (float(column.min()) + float(column.max()))
               if column is not None else 0.0)
        kwargs['water_density'] = float(density(
            *_water_at(absorption, mid))) / 1000.0
    env = Environment(**kwargs)

    _record_provenance(env, bathy_src, fetched_ssp.source, bottom_kw,
                       bottom_props, surface_src, ph_src, ts_src)
    return env


def _water_at(absorption, depth: float):
    """The ``(temperature, salinity)`` of a Francois-Garrison law at each
    property's sample nearest ``depth`` (its number when it has no
    profile)."""
    def nearest(value):
        if np.ndim(value) == 0:
            return float(value)
        return float(value[int(np.argmin(np.abs(value[:, 0] - float(depth)))),
                           1])
    return nearest(absorption.temperature), nearest(absorption.salinity)


def _record_provenance(env, bathy_src, ssp_src, bottom_kw, bottom_props,
                       surface_src, ph_src=None, ts_src=None):
    """Complete the provenance ``env.data_sources`` reads, so it lists the
    datasets in axis order
    (bathymetry → ssp → bottom → surface → absorption T/S → pH → altimetry),
    exact repeats removed. The ``*_src`` are catalogue ids; the fetched
    altimetry carries its own record, as every fetched carrier does.
    ``ts_src`` is the dataset the absorption's T/S row came from, which is
    WOA23 beside a literal ``ssp=``.

    Each fetched carrier carries its own ``data_sources`` — a tuple of
    :class:`~uacpy.data.sources.DataProvenance` records holding the dataset plus
    the **actual** date/coordinates it returned (which can differ from what was
    requested). A carrier that wasn't stamped — a literal axis, or a fetcher
    that reports no date/coords — is stamped here with the bare catalogue id,
    so attribution is never lost. The T/S row and the pH describe no carrier:
    they go to ``env.extra_data_sources``. Warns on any non-commercial
    licence used."""
    def catalogue(ids):
        # The bare catalogue id in a DataProvenance, so env.data_sources is
        # uniformly DataProvenance — no date/coords, just the source (plus the
        # climatology vintage where the cache records it). A record the
        # fetcher already built (the pH node read) is kept as it is.
        return tuple(i if isinstance(i, DataProvenance) else
                     DataProvenance(source=SOURCES[i],
                                    data_date=_climatology_vintage(i))
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
    extra = catalogue([ts_src, ph_src])
    if extra:
        env.extra_data_sources = extra
    # A licence-restricted source must never enter a result silently: warn at
    # fetch time for any non-commercial dataset used (e.g. CRUST1.0). Driven off
    # the catalogue flag so future non-commercial sources are covered too.
    for src in {prov.source.id: prov.source
                for prov in env.data_sources}.values():
        if not src.commercial_use:
            warnings.warn(
                f"fetch_environment: data source {src.id!r} ({src.name}) does "
                f"not permit commercial use without verification — see "
                f"uacpy.data.citations(env) for its licence/attribution.",
                ProvenanceWarning, skip_file_prefixes=USER_FRAME_SKIP)


#: Cached monthly climatologies whose reference period is recorded in the
#: ``.npz`` itself (see each module's ``climatology_period``). Everything else
#: is either dated per fetch or has no period to state.
_CLIMATOLOGY_PERIOD_SOURCES = {
    'seaice': 'uacpy.data.seaice_local',
}


def _climatology_vintage(src_id):
    """Reference period of a cached climatology, for the provenance
    ``data_date``, or ``None``.

    The sea-ice cache derives its default ``years`` from ``date.today()`` at
    build time, so the vintage differs per build; the cache records it and
    this reads it back for an un-stamped sea-ice layer (the NBS wind
    climatology is dated by :mod:`uacpy.data.sea_surface` itself). ``None``
    covers every other source, a cache written without the key, and any read
    failure — an unstated vintage must never turn a working fetch into an
    error.
    """
    module_name = _CLIMATOLOGY_PERIOD_SOURCES.get(src_id)
    if module_name is None:
        return None
    import importlib
    try:
        return importlib.import_module(module_name).climatology_period()
    except Exception:              # no cache installed, or unreadable
        return None


def _bathymetry_provenance(bathy, point, *, max_distance_km=None):
    """The ``DataProvenance`` of the depth read at ``point``, through the
    offset rule. It names the backend and vintage that answered (the GEBCO
    DOI is per release, and a local GEBCO_2025.nc is not the OpenTopoData
    service). The local GEBCO grid records the centre of the cell read; the
    live services are point queries that report no node, so they record
    the point itself."""
    lat, lon = (float(v) for v in as_coordinate(point))
    vintage = _bathymetry_vintage(bathy.source, bathy.backend)
    if bathy.backend == 'local' and bathy.source == 'gebco':
        from uacpy.data import gebco_local
        try:
            grid = gebco_local._grid()
        except (ConfigurationError, DataFetchError, FileNotFoundError):
            grid = None             # a stubbed backend: no grid to read
        if grid is not None:
            return grid.node_provenance(
                'gebco', lat, lon, who='fetch_environment',
                max_distance_km=max_distance_km, data_date=vintage)
        data_point = None
    else:
        data_point = (lat, lon)
    return checked_offset(
        DataProvenance(source=SOURCES[bathy.source], data_date=vintage,
                       data_point=data_point, requested_point=(lat, lon)),
        who='fetch_environment', warn_km=0.0, max_distance_km=max_distance_km)


def _bathymetry_vintage(src_id, backend):
    """What a bathymetry provenance record cites for its ``data_date``: the
    local grid's release (its file name), or the live service and dataset."""
    if backend == 'local' and src_id == 'gebco':
        from uacpy.data import gebco_local
        try:
            return gebco_local.grid_name()
        except (ConfigurationError, DataFetchError, FileNotFoundError):
            return 'local'          # a stubbed backend: no grid to name
    if backend == 'api' and src_id == 'gebco':
        return f"{DEFAULT_DATASET} via OpenTopoData"
    return f"{backend} (live)" if backend else None


def _fetch_absorption(point, *, date, ssp_source, ssp_backend, cache_only,
                      resolution, max_distance_km=None, max_days=None,
                      timeout, verbose):
    """Francois-Garrison absorption from the site's fetched T/S column.

    Returns ``(absorption, ts_source, ph_source)``: ``ts_source`` is the
    catalogue id of the dataset the T/S column came from (``'woa23'``,
    ``'copernicus'`` or ``'argo'``). Reuses the backend the SSP resolved to
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
    """
    ts_src = 'woa23'
    if cache_only:
        ts = _fetch_ts_profile_backend(
            point, date=date, backend='local', resolution=resolution,
            timeout=timeout, verbose=verbose, max_distance_km=max_distance_km)
        depths, temp, sal = ts.depths, ts.temperature, ts.salinity
    elif ssp_source == 'copernicus':
        from uacpy.data.copernicus import fetch_ts_profile_operational
        extra = {} if max_days is None else {'max_days': max_days}
        # In-situ temperature, as Francois-Garrison takes it.
        ts = fetch_ts_profile_operational(
            point, date=date, verbose=verbose,
            max_distance_km=max_distance_km, **extra)
        depths, temp, sal = ts.depths, ts.temperature, ts.salinity
        ts_src = 'copernicus'
    elif ssp_source == 'argo':
        from uacpy.data.argo import fetch_argo_profile
        extra = {} if max_days is None else {'max_days': max_days}
        prof = fetch_argo_profile(point, date=date, timeout=timeout,
                                  verbose=verbose,
                                  max_distance_km=max_distance_km, **extra)
        depths = pressure_dbar_to_depth(prof.pressure_dbar, prof.lat)
        temp, sal = prof.temperature, prof.salinity
        ts_src = 'argo'
    elif ssp_backend is None:
        # Literal SSP: the public WOA23 fetcher is cache-first (the installed
        # grid, then NCEI THREDDS), as every fetched axis is.
        ts = fetch_ts_profile(
            point, date=date, resolution=resolution, source='woa23',
            timeout=timeout, verbose=verbose, max_distance_km=max_distance_km)
        depths, temp, sal = ts.depths, ts.temperature, ts.salinity
    else:
        ts = _fetch_ts_profile_backend(
            point, date=date, resolution=resolution, timeout=timeout,
            verbose=verbose, backend=ssp_backend,
            max_distance_km=max_distance_km)
        depths, temp, sal = ts.depths, ts.temperature, ts.salinity
    # GLODAP gives its pH column, (depth, pH) pairs on its own levels;
    # Copernicus BGC one value, read at the T/S column's mid-depth.
    z_arr = np.asarray(depths, dtype=float)
    ref_depth = (0.5 * (float(z_arr.min()) + float(z_arr.max()))
                 if z_arr.size else None)
    pH, ph_src = _fetch_ph(point, date=date, ssp_source=ssp_source,
                           cache_only=cache_only, timeout=timeout,
                           verbose=verbose, reference_depth=ref_depth,
                           max_distance_km=max_distance_km)
    # GLODAP (pHtsinsitutp) and the Copernicus BGC ``ph`` field are on the
    # total scale; the formula was fitted on NBS. The model default that
    # stands in when neither is available is on the formula's own scale.
    return FrancoisGarrison.from_temperature_salinity(
        depths, temp, sal, pH=pH,
        ph_scale='nbs' if ph_src is None else 'total'), ts_src, ph_src


def _fetch_ph(point, *, date=None, ssp_source=None, cache_only=False,
              timeout=120.0, verbose=False, reference_depth=None,
              max_distance_km=None):
    """Representative seawater pH at ``point``, pH-source-aware and cache-first.

    Returns ``(pH, source)``: one value, or GLODAP's column as ``(depth,
    pH)`` pairs (an ``(N, 2)`` array on its finite levels; one value when it
    has a single level). On the Copernicus SSP branch (``ssp_source ==
    'copernicus'`` and not ``cache_only``) the time-varying BGC ``ph`` field is
    preferred (``source='copernicus_bgc'``), riding the same Copernicus
    session as the SSP. Otherwise — or if the BGC fetch fails — it falls back to
    the cached GLODAP climatology, whose ``source`` is the
    :class:`DataProvenance` record of the node read (its point and offset),
    and finally to the model default (``REFERENCE_PH``, ``None``). pH is cache/best-effort like every
    other axis: a run without any pH source silently keeps the default rather
    than failing, and so does one whose pH cells stand past
    ``max_distance_km``.
    """
    from uacpy.core.constants import REFERENCE_PH
    from uacpy.data.glodap_local import fetch_ph_profile
    lat, lon = as_coordinate(point)
    if ssp_source == 'copernicus' and not cache_only and date is not None:
        from uacpy.data.copernicus import fetch_ph_operational
        try:
            pH = fetch_ph_operational(point, date=date, verbose=verbose,
                                      reference_depth=reference_depth,
                                      max_distance_km=max_distance_km)
            log_message('copernicus', f"pH {pH:.3f} from Copernicus BGC at "
                        f"{lat:.3f}, {lon:.3f}", verbose=verbose)
            return pH, 'copernicus_bgc'
        except (DataFetchError, ConfigurationError):
            pass                                    # fall through to GLODAP
    try:
        profile = fetch_ph_profile(point, max_distance_km=max_distance_km)
    except (DataFetchError, ConfigurationError):
        return REFERENCE_PH, None
    z = np.asarray(profile.depths, dtype=float)
    ph = np.asarray(profile.ph, dtype=float)
    keep = np.isfinite(z) & np.isfinite(ph)
    z_u, first = np.unique(z[keep], return_index=True)
    if z_u.size == 0:
        return REFERENCE_PH, None
    pairs = np.column_stack([z_u, ph[keep][first]])
    log_message('glodap', f"pH {pairs[:, 1].min():.3f}-{pairs[:, 1].max():.3f} "
                f"over {len(pairs)} levels from GLODAP at {lat:.3f}, "
                f"{lon:.3f}", verbose=verbose)
    return (float(pairs[0, 1]) if len(pairs) == 1 else pairs), \
        profile.provenance


#: Depth span (m) over which an Argo cast hands over to the WOA23 column below
#: it: the cast's deepest this-many metres are a raised-cosine blend from the
#: cast to the climatology, so the joined profile carries no step and no
#: gradient kink at the join (a step would read as a reflector and a kink as a
#: false turning depth to a ray model). Core floats stop near 2000 dbar, so the
#: default blends over 1500-2000 m.
CAST_BLEND_M = 500.0


def _deepen_cast_with_woa23(cast, depth_max, point, *, date, formula,
                            resolution, cache_only, timeout, verbose):
    """An Argo cast continued below its deepest level by the WOA23 column.

    A core Argo float profiles to about 2000 dbar. Below that the deep water
    keeps cooling, which the fixed-temperature extrapolation of
    :func:`~uacpy.data.extend_ssp_below_data` cannot follow; the WOA23 column
    at the same point and month has the analysed deep water. Over the cast's
    deepest :data:`CAST_BLEND_M` the profile blends from the cast to WOA23
    with a raised-cosine weight, and below the cast it is WOA23. The result
    carries both provenance records.

    Returned unchanged when the cast already reaches the seafloor (to within
    the extension's own 50 m tolerance), when WOA23 cannot be read at the
    point, or when WOA23 ends no deeper than the cast; the caller's
    extrapolation then covers any remaining depth, with its warning.
    """
    from uacpy.data.sound_speed import _EXTRAPOLATION_WARN_M
    z_cast = np.asarray(cast.depths, dtype=float)
    c_cast = np.asarray(cast.sound_speed, dtype=float)[:, 0]
    z_last = float(z_cast[-1])
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
    z_woa = np.asarray(woa.depths, dtype=float)
    c_woa = np.asarray(woa.sound_speed, dtype=float)[:, 0]
    if z_woa[-1] <= z_last:
        return cast
    top = max(float(z_cast[0]), z_last - CAST_BLEND_M)
    above = z_cast < top
    blend_z = np.union1d(z_cast[(z_cast >= top) & (z_cast <= z_last)],
                         z_woa[(z_woa >= top) & (z_woa <= z_last)])
    span = z_last - top
    weight = (0.5 * (1.0 - np.cos(np.pi * (blend_z - top) / span))
              if span > 0.0 else np.ones_like(blend_z))
    blend_c = ((1.0 - weight) * np.interp(blend_z, z_cast, c_cast)
               + weight * np.interp(blend_z, z_woa, c_woa))
    below = z_woa > z_last
    log_message('sound_speed', f"Argo cast ends at {z_last:.0f} m; WOA23 "
                f"continues it to {z_woa[-1]:.0f} m (blended over "
                f"{top:.0f}-{z_last:.0f} m)", verbose=verbose)
    return SoundSpeedProfile(
        depths=np.concatenate([z_cast[above], blend_z, z_woa[below]]),
        sound_speed=np.concatenate([c_cast[above], blend_c, c_woa[below]]),
        kind='measured',
        data_sources=tuple(cast.data_sources) + tuple(woa.data_sources),
        formula=cast.formula)


# Bottom data sources — single declarative registry. To add a sediment source:
# write its module's point fetcher, then add one ``_BottomProvider`` row below.
# Everything else (the accepted source keywords, the 'auto' fallback order, the
# fetcher lookup and the provenance id) derives from this list.
#
# ``resolve`` returns the source's ``(point_fetcher, transect_fetcher)`` pair,
# imported lazily to keep optional deps optional and avoid import cycles. A
# half-space source's transect is its point fetcher sampled along the track
# (:func:`~uacpy.data.sediment.transect_fetcher`); CRUST1.0, whose columns are
# layered, has its own. Providers with a cached twin (``has_cached_variant``:
# EMODnet's local polygons vs the live WFS, pelagic's local-GEBCO vs
# API-allowed depth lookup) take a ``cached`` flag and are tried cache-first;
# the rest take no argument.


@dataclass(frozen=True)
class _BottomProvider:
    """One bottom data source. ``id`` doubles as the source keyword and the
    provenance catalogue id. ``in_auto`` puts it in the 'auto' fallback chain,
    ``in_cache_auto`` in the network-free 'local' chain. ``has_cached_variant``
    means it has a local twin tried before its live/compute backend
    (cache-first); under ``cache_only`` only the local twin is used."""

    id: str
    resolve: Callable                  # () -> (point_fn, transect_fn); takes
                                       # (cached: bool) iff has_cached_variant
    has_cached_variant: bool = False
    in_auto: bool = False
    in_cache_auto: bool = False
    accepts_depth: bool = False         # depth-driven source: takes the fetched water depth
    accepts_grain_size_model: bool = False  # converts a grain size: takes model=


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
    from functools import partial
    from uacpy.data import pelagic as m
    # ``cached`` forbids the GEBCO live-API fallback in the depth lookup, so the
    # cached attempt (local GEBCO only) precedes the API-allowed one.
    return _along_track(partial(m.fetch_bottom_pelagic, cache_only=cached),
                        'pelagic model')


_BOTTOM_PROVIDERS = (
    _BottomProvider('emodnet', _emodnet_pair, has_cached_variant=True,
                    in_auto=True, in_cache_auto=True,
                    accepts_grain_size_model=True),
    # A measured sample beats a modelled or interpolated map, so the
    # grain-size database sits directly behind EMODnet's polygons in 'auto'
    # too, not only in 'local': at 36 N 75 W it answers with a sand sample
    # 124 km away (rho*c 3608) where the pelagic rule gives ooze (2245).
    _BottomProvider('grainsize', _grainsize_pair, in_auto=True,
                    in_cache_auto=True, accepts_grain_size_model=True),
    _BottomProvider('crust1', _crust1_pair),
    _BottomProvider('graw', _graw_pair, accepts_grain_size_model=True),
    _BottomProvider('diesing', _diesing_pair, in_auto=True, in_cache_auto=True,
                    accepts_grain_size_model=True),
    # MARS is live-only, so it sits *after* the offline global Diesing map:
    # 'auto' consults the installed raster before any AusSeabed request, and
    # MARS then covers the Australian shelf Diesing (deep sea only) misses.
    _BottomProvider('mars', _mars_pair, in_auto=True,
                    accepts_grain_size_model=True),
    _BottomProvider('pelagic', _pelagic_pair, has_cached_variant=True,
                    in_auto=True, in_cache_auto=True,
                    accepts_depth=True,  # never fails (last resort)
                    accepts_grain_size_model=True),
)
_BOTTOM_BY_ID = {p.id: p for p in _BOTTOM_PROVIDERS}
_AUTO_BOTTOM_ORDER = tuple(p.id for p in _BOTTOM_PROVIDERS if p.in_auto)
_CACHE_BOTTOM_ORDER = tuple(p.id for p in _BOTTOM_PROVIDERS if p.in_cache_auto)


def _bottom_order(bottom_source, who='fetch_environment'):
    """Ordered bottom source keywords + a ``cache_only`` flag from the user
    spec. ``'auto'`` → the best-available chain (EMODnet → grain-size →
    Diesing → MARS → pelagic); ``'local'`` → the network-free chain (EMODnet local → grain-size →
    Diesing → pelagic, cached backends only); a str/sequence of keywords is used
    as-is. Validates each keyword; an unknown one is refused in the name of
    ``who``, the public function the user called."""
    if bottom_source == 'local':
        return _CACHE_BOTTOM_ORDER, True
    if bottom_source == 'auto':
        return _AUTO_BOTTOM_ORDER, False
    order = source_tuple(bottom_source)
    for name in order:
        if name not in _BOTTOM_BY_ID:
            raise ConfigurationError(
                f"{who}: unknown bottom source {name!r}.",
                remediation=f"Use 'auto', 'local' or one of {sorted(_BOTTOM_BY_ID)}.",
            )
    return order, False


#: Keywords ``_fetch_bottom`` supplies to the providers itself, so a
#: provider option of the same name is refused rather than overriding them.
_SUPPLIED_BOTTOM_KEYWORDS = frozenset({
    'cache_only', 'max_distance_km', 'depth', 'model', 'environment',
    'water_sound_speed', 'roughness', 'n_points', 'max_points', 'timeout',
    'verbose'})


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

    def attempts():
        # Cache-first: the local twin (variant True) before the live backend
        # (False), where one exists; cache_only drops the live attempt.
        # Providers without a cached twin resolve one fetcher pair (None).
        for name in order:
            if not _BOTTOM_BY_ID[name].has_cached_variant:
                yield name, None
            elif cache_only:
                yield name, True
            else:
                yield from ((name, True), (name, False))

    def call(name, cached):
        provider = _BOTTOM_BY_ID[name]
        call_kwargs = dict(kwargs, **(source_options or {}))
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
        # Resolution happens per attempt, so a live backend's module is only
        # imported when the cached attempt has failed.
        point_fn, transect_fn = provider.resolve(
            *(() if cached is None else (cached,)))
        fn = transect_fn if transect else point_fn
        return fn(*args, **call_kwargs)

    # A cached twin that read an answer ends its provider; only an absent or
    # unreadable cache is worth asking the live twin.
    bottom, (name, _cached) = first_answer(
        attempts(), call, cached=lambda variant: variant is True)
    return bottom, name


def _fetch_bottom_per_waypoint(order, start, end, *, cache_only,
                               max_distance_km, depth, model, hamilton_fit,
                               water_sound_speed=None, n_points=6,
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
    return _bottom_order(_lower_source_spec(source), who)


def fetch_bottom(
    point: Coordinate,
    *,
    source: Union[str, Sequence[str]] = 'auto',
    model: str = DEFAULT_GRAIN_SIZE_MODEL,
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
        The grain-size relations, Hamilton & Bachman fit and in-situ water
        sound speed (m/s) at the seabed, for every source that converts a
        grain size (see :func:`uacpy.core.sediment.grain_size_to_geoacoustics`).
    depth : float, optional
        Water depth (m) for the depth-driven ``'pelagic'`` source; fetched
        from GEBCO when ``None``.
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
    model, hamilton_fit = canonical_grain_size_selection(
        model, hamilton_fit, who='fetch_bottom')
    order, cache_only = _bottom_request(source, 'fetch_bottom')
    max_distance_km = checked_max_distance(max_distance_km, 'fetch_bottom')
    if depth is None:
        _refuse_a_dry_point(point, who='fetch_bottom', cache_only=cache_only,
                            timeout=timeout, verbose=verbose)
    bottom, _name = _fetch_bottom(
        order, as_coordinate(point), transect=False, cache_only=cache_only,
        max_distance_km=max_distance_km, depth=depth, model=model,
        hamilton_fit=hamilton_fit, water_sound_speed=water_sound_speed,
        roughness=roughness, timeout=timeout, verbose=verbose,
        source_options=source_options, who='fetch_bottom')
    return bottom


@one_provenance_notice(subject="the seabed's data",
                       record='uacpy.data.citations(bottom)')
def fetch_bottom_transect(
    start: Coordinate,
    end: Coordinate,
    *,
    source: Union[str, Sequence[str]] = 'auto',
    n_points: Union[int, str] = 6,
    max_points: int = DEFAULT_MAX_TRANSECT_POINTS,
    model: str = DEFAULT_GRAIN_SIZE_MODEL,
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
    waypoint count (default 6), or ``'auto'`` to probe ``max_points``
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
        Waypoints, or ``'auto'`` (see above). Default 6.
    max_points : int, optional
        Waypoints ``'auto'`` probes. Default 1000.
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
    model, hamilton_fit = canonical_grain_size_selection(
        model, hamilton_fit, who='fetch_bottom_transect')
    order, cache_only = _bottom_request(source, 'fetch_bottom_transect')
    max_distance_km = checked_max_distance(max_distance_km, 'fetch_bottom_transect')
    bottom, _name = _fetch_bottom(
        order, as_coordinate(start), as_coordinate(end), transect=True,
        cache_only=cache_only, max_distance_km=max_distance_km, depth=depth,
        model=model, hamilton_fit=hamilton_fit,
        water_sound_speed=water_sound_speed, roughness=roughness,
        n_points=n_points, max_points=max_points, timeout=timeout,
        verbose=verbose, source_options=source_options,
        who='fetch_bottom_transect')
    return bottom


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


def _resolve_bottom(bottom, *, water_sound_speed=None, model=DEFAULT_GRAIN_SIZE_MODEL,
                    hamilton_fit=None):
    """The ``bottom=`` literal as a seabed: a ready ``BoundaryProperties``,
    ``SeabedColumn`` or ``Bottom`` passes unchanged with its own provenance
    (``Environment`` coerces it), a class name takes its preset and a ϕ float
    goes through the grain-size relations."""
    if bottom is None or isinstance(bottom, (BoundaryProperties, SeabedColumn,
                                             Bottom)):
        return bottom
    if isinstance(bottom, str):
        return bottom_from_class(bottom)
    if isinstance(bottom, (int, float)) and not isinstance(bottom, bool):
        return bottom_from_grain_size(
            float(bottom), model=model, hamilton_fit=hamilton_fit,
            water_sound_speed=water_sound_speed)
    raise ConfigurationError(
        f"fetch_environment: bottom must be a ϕ float, a class name, a "
        f"BoundaryProperties, a SeabedColumn, a Bottom, or None; got "
        f"{type(bottom).__name__}.",
    )
