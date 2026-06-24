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
(sea-state roughness) has no fetch source, so it is literal-only. Fetching is
**cache-first**: a locally installed dataset is sampled before any network call,
and ``*_sources='local'`` skips the network entirely (failing fast with an
install hint), so an air-gapped or reproducible run sets ``'local'`` on the axes
it wants pinned to local data (see ``install.sh --data``).
"""

import datetime as _dt
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

import numpy as np

from uacpy.core.environment import BoundaryProperties, Environment
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data._geo import (
    Coordinate, as_coordinate, DEFAULT_MAX_TRANSECT_POINTS,
)
from uacpy.data.bathymetry import fetch_bathy, fetch_bathy_transect
from uacpy.data.sediment import bottom_from_class, bottom_from_grain_size
from uacpy.data.sound_speed import fetch_ssp, fetch_ssp_transect
from uacpy.data.sources import SOURCES, DataProvenance

__all__ = ['fetch_environment']


# User-facing source → ordered backends, cache-first. ``'local'`` is the cached
# twin (a static global grid/climatology); the other token is the live backend.
# Sources with no cached twin list only their live backend.
_SSP_BACKENDS = {
    'woa23': ('local', 'opendap'),
    'copernicus': ('opendap',),
    'argo': ('opendap',),
}
_BATHY_BACKENDS = {
    'gebco': ('local', 'api'),
    'gmrt': ('gmrt',),
}
_CACHE_BACKEND = 'local'   # the cached-twin backend token (``*_sources='local'``)
# Default fetch chains for the two mandatory axes (used when neither a literal
# nor an explicit ``*_sources`` is given). Both are global and need no login.
_DEFAULT_SSP_SOURCES = ('woa23',)
_DEFAULT_BATHY_SOURCES = ('gebco',)
# ``'auto'`` best-available presets (used when ``*_sources='auto'``):
# SSP prefers real → model → climatology (the first two need ``date=`` / a
# Copernicus login, else they fall through to WOA23); bathymetry prefers the
# higher-res multibeam synthesis, falling back to the global grid.
_AUTO_SSP_SOURCES = ('argo', 'copernicus', 'woa23')
_AUTO_BATHY_SOURCES = ('gmrt', 'gebco')


def _resolve_axis_sources(spec, *, auto):
    """Expand a ``*_sources`` spec to ``(sources, cache_only)``: ``'auto'`` →
    the axis's best-available chain; ``'local'`` → the same chain but cached
    backends only (no network); else the explicit list. (Bottom has its own
    ``_bottom_order``.)"""
    if spec == 'local':
        return auto, True
    if spec == 'auto':
        return auto, False
    return _as_source_tuple(spec), False


def _axis_attempts(sources, backends_map, *, axis, cache_only):
    """Expand an ordered user source list into ``(source, backend)`` attempts.

    Cache-first within each source (``'local'`` before the live backend), so a
    cached twin is sampled before any network call. ``cache_only`` keeps only
    the cached backend, so a source with no cached twin contributes no attempts
    (and falls through to the next). Validates source names against
    ``backends_map``."""
    attempts = []
    for src in sources:
        if src not in backends_map:
            raise ConfigurationError(
                f"fetch_environment: unknown {axis} source {src!r}.",
                remediation=f"Use one of {sorted(backends_map)}.",
            )
        backends = backends_map[src]
        if cache_only:
            backends = tuple(b for b in backends if b == _CACHE_BACKEND)
        attempts.extend((src, b) for b in backends)
    return attempts


def _resolve_cached(call, order, *, axis):
    """Call ``call(token)`` for each token in ``order``, falling back to the
    next on any fetch failure (cache absent → ``ConfigurationError``; live
    failure / no coverage → ``DataFetchError``). Returns ``(result, token)``.

    If ``order`` is empty (``*_sources='local'`` with no cached source) or
    every attempt fails, raise the most *substantive* error — a
    ``DataFetchError`` (no coverage / on land / live failure) over a bare
    ``ConfigurationError`` ("cache not installed") — so the user sees the real
    cause.
    """
    if not order:
        raise ConfigurationError(
            f"fetch_environment: {axis}_sources='local' but no cached {axis} "
            f"source is available.",
            remediation="Install a cached dataset (see install.sh --data), or "
                        "use 'auto' / a live source.",
        )
    errors = []
    for token in order:
        try:
            return call(token), token
        except (ConfigurationError, DataFetchError) as exc:
            errors.append(exc)
    data_errs = [e for e in errors if isinstance(e, DataFetchError)]
    raise (data_errs[0] if data_errs else errors[-1])


def _as_source_tuple(value):
    """Normalize a ``str``/sequence source spec to a tuple of source names."""
    return (value,) if isinstance(value, str) else tuple(value)


def fetch_environment(
    point: Coordinate,
    *,
    date: Union[str, _dt.date, None] = None,
    name: Optional[str] = None,
    ssp=None,
    ssp_sources: Union[str, Sequence[str], None] = None,
    bathymetry=None,
    bathymetry_sources: Union[str, Sequence[str], None] = None,
    bottom: Union[float, str, BoundaryProperties, None] = None,
    bottom_sources: Union[str, Sequence[str], None] = None,
    surface: Optional[BoundaryProperties] = None,
    surface_sources: Union[str, Sequence[str], None] = None,
    altimetry=None,
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
    formula: str = 'unesco',
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
        :class:`Environment`'s ``ssp=``: a ``SoundSpeedProfile``, a scalar c →
        isovelocity, or ``(depth, c)`` pairs). If ``ssp_sources`` is *also*
        given, the source is fetched first and this literal is the **fallback**
        when the fetch yields nothing; on its own, SSP is not fetched at all.
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
        A **literal** depth (m, scalar) or range-dependent ``(N, 2)`` ``(range,
        depth)`` array, supplied directly. If ``bathymetry_sources`` is *also*
        given, the source is fetched first and this literal is the **fallback**;
        on its own, bathymetry is not fetched.
    bathymetry_sources : str or sequence of str, optional
        Bathymetry source(s) to **fetch**, tried in order, or a preset:
        ``'auto'`` (best-available: ``gmrt`` → ``gebco``, i.e. high-res multibeam
        where surveyed, else the global grid) or ``'local'`` (the cached GEBCO
        grid only — no network). Choices: ``'gebco'`` (global) or ``'gmrt'``
        (multibeam, higher-res, CC-BY). Default ``None`` → fetch ``'gebco'``.
    bottom : float or str or BoundaryProperties, optional
        A **literal** seafloor supplied directly (no fetch): a mean grain size
        (ϕ, float), a :data:`~uacpy.materials.MATERIALS` class name, a ready
        ``BoundaryProperties``, or ``None`` (default). If ``bottom_sources`` is
        *also* given, the source is fetched first and this literal is the
        **fallback**. (A bottom *string* is a material name here, never a
        source — source keywords go in ``bottom_sources`` — which is why the two
        are separate args.)
    bottom_sources : str or sequence of str, optional
        Seafloor source(s) to **fetch**, tried in order, or a preset: ``'auto'``
        (best-available: EMODnet → Diesing → pelagic) or ``'local'`` (network-
        free: EMODnet-local → grain-size → Diesing → pelagic, cached backends
        only). Per-source choices: ``'emodnet'`` (European seas, high-res,
        CC-BY), ``'grainsize'`` (NCEI grain-size samples, worldwide, public-
        domain — cached), ``'crust1'``, ``'diesing'``, ``'pelagic'`` (first-
        principles, never fails). Default ``None`` — bottom is optional, so it
        is only fetched when you ask. Most sources permit commercial use;
        CRUST1.0 does not without verification — a non-commercial source emits a
        ``UserWarning`` when fetched. See ``uacpy.data.citations(env)``.
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
        point raises. WOA23 only.
    ssp_n_points : int or 'auto', optional
        SSP columns along the transect when ``range_dependent_ssp``. Default
        ``'auto'``: the transect is sampled at the **distinct WOA23 cells** it
        crosses (one column per cell — WOA's native range resolution, found
        analytically so no duplicate column is fetched), capped at
        ``max_points``. Pass an int for exactly that many evenly-spaced columns.
    range_dependent_bottom : bool, optional
        Whether a fetched seafloor varies along the transect. A bottom is fetched
        only when requested (``bottom_sources=`` given, or this flag ``True``).
        **Default (``None``): a requested bottom is range-dependent on a
        transect**, range-independent at a point. Pass ``False`` to force a
        single representative bottom even along a transect; ``True`` on a point
        raises (and, with no ``bottom_sources``, implies ``'auto'``). A single
        source spans the transect, so gaps forward-fill the nearest covered
        value — ``bottom_sources='grainsize'`` is a uniform worldwide source.
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
        consecutive identical ice/open-water zones to one boundary each (the
        marginal ice zone at native scale, no staircase), capped at
        ``max_points``. Pass an int for exactly that many waypoints.
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
        a homogeneous elastic ice canopy (cp 3500 m/s, cs 1800 m/s, ρ 0.9,
        αp/αs 0.5/1.0 dB/λ — *Computational Ocean Acoustics*); open water keeps
        the default free surface (no provenance). Default ``None`` (no surface
        fetch). Point classification only (the carrier's surface is one boundary).
    altimetry : array-like, optional
        A **literal** rough-surface wave profile ``[(range, height_m), …]``
        (height positive up; same as :class:`Environment`'s ``altimetry=``).
        There is no fetch source for sea state, so this is literal-only — build
        one with :func:`uacpy.generate_sea_surface`. Default ``None`` (flat).
    with_absorption : bool, optional
        If ``True``, attach a Francois-Garrison absorption built from the
        site's fetched temperature/salinity column (costs one extra T/S
        request). Default ``False`` (model-default Thorp absorption).
    formula, resolution, timeout, verbose
        Forwarded to the sound-speed / bathymetry fetchers.

    Returns
    -------
    Environment
    """
    lat, lon = as_coordinate(point)

    # Each axis is a literal (ssp=/bathymetry=/bottom=) and/or fetched from
    # source(s) (*_sources). When both are given the source is fetched first and
    # the literal is the fallback if the fetch yields nothing (no coverage,
    # service down). Bathy/SSP are mandatory: with neither, fetch the default
    # chain.

    # ── Bathymetry ──
    bathy_cache_only = False
    if bathymetry_sources is not None:
        bathy_srcs, bathy_cache_only = _resolve_axis_sources(
            bathymetry_sources, auto=_AUTO_BATHY_SOURCES)
    elif bathymetry is None:
        bathy_srcs = _DEFAULT_BATHY_SOURCES         # mandatory axis, none given
    else:
        bathy_srcs = None                           # literal only
    bathy_src = None
    if bathy_srcs is not None:
        bathy_order = _axis_attempts(bathy_srcs, _BATHY_BACKENDS,
                                     axis='bathymetry',
                                     cache_only=bathy_cache_only)

        def _bathy_call(token):
            _src, backend = token
            if transect_to is None:
                return fetch_bathy(point, source=backend, timeout=timeout,
                                   verbose=verbose)
            return fetch_bathy_transect(point, transect_to, n_points=n_points,
                                        max_points=max_points,
                                        source=backend, timeout=timeout,
                                        verbose=verbose)

        try:
            bathymetry, (bathy_src, _) = _resolve_cached(
                _bathy_call, bathy_order, axis='bathymetry')
        except (DataFetchError, ConfigurationError) as exc:
            if bathymetry is None:
                raise
            if transect_to is not None:
                warnings.warn(
                    f"fetch_environment: range-dependent bathymetry fetch "
                    f"failed ({exc}); falling back to the supplied bathymetry= "
                    f"literal. A range-independent literal reduces the transect "
                    f"to a single depth.",
                    UserWarning, stacklevel=2)
            bathy_src = None                        # fall back to the literal

    # ── Range-dependence resolution ──
    # A transect makes each *fetched* axis range-dependent by default; a single
    # point is always range-independent. ``range_dependent_*=True`` on a point
    # is an error; ``=False`` forces a single representative sample even along a
    # transect. SSP is always fetched (so a transect makes it range-dependent);
    # bottom and surface are fetched only on request (their ``*_sources`` or an
    # explicit ``True``), and become range-dependent when fetched on a transect.
    for _flag, _ax in ((range_dependent_ssp, 'ssp'),
                       (range_dependent_bottom, 'bottom'),
                       (range_dependent_surface, 'surface')):
        if _flag is True and transect_to is None:
            raise ConfigurationError(
                f"fetch_environment: range_dependent_{_ax}=True requires "
                f"transect_to=.",
                remediation="Pass transect_to=(lat, lon), or leave it unset "
                            "for a single point.")
    _on_transect = transect_to is not None
    rd_ssp = _on_transect and range_dependent_ssp is not False
    want_bottom = bottom_sources is not None or range_dependent_bottom is True
    rd_bottom = (want_bottom and _on_transect
                 and range_dependent_bottom is not False)
    want_surface = surface_sources is not None or range_dependent_surface is True
    rd_surface = (want_surface and _on_transect
                  and range_dependent_surface is not False)

    # ── SSP ──
    ssp_cache_only = False
    if ssp_sources is not None:
        ssp_srcs, ssp_cache_only = _resolve_axis_sources(
            ssp_sources, auto=_AUTO_SSP_SOURCES)
    elif ssp is None or range_dependent_ssp is True:
        ssp_srcs = _DEFAULT_SSP_SOURCES             # mandatory default / rd-fetch
    else:
        ssp_srcs = None                             # literal only
    ssp_src = None
    ssp_backend = None
    ssp_fetched = False
    if ssp_srcs is not None:
        ssp_order = _axis_attempts(ssp_srcs, _SSP_BACKENDS, axis='ssp',
                                   cache_only=ssp_cache_only)

        def _ssp_call(token):
            src, backend = token
            if not rd_ssp:
                return _fetch_ssp(
                    point, date=date, ssp_source=src, formula=formula,
                    resolution=resolution, source=backend, timeout=timeout,
                    verbose=verbose)
            if src == 'woa23':
                return fetch_ssp_transect(
                    point, transect_to, n_points=ssp_n_points,
                    max_points=max_points, date=date,
                    formula=formula, resolution=resolution, source=backend,
                    timeout=timeout, verbose=verbose)
            if src == 'copernicus':
                if date is None:
                    raise ConfigurationError(
                        "fetch_environment: ssp_sources='copernicus' requires date=.",
                    )
                from uacpy.data.copernicus import fetch_ssp_transect_operational
                # Copernicus has no 'auto' resolver (no cheap cell identity
                # exposed here); fall back to a fixed column count, capped.
                cop_n = ssp_n_points if isinstance(ssp_n_points, int) else 6
                return fetch_ssp_transect_operational(
                    point, transect_to, date=date, n_points=min(cop_n, max_points),
                    formula=formula, timeout=timeout, verbose=verbose)
            raise ConfigurationError(
                f"fetch_environment: range_dependent_ssp not supported for "
                f"ssp_sources={src!r}.",
                remediation="Use 'woa23' or 'copernicus'.",
            )

        try:
            ssp, (ssp_src, ssp_backend) = _resolve_cached(
                _ssp_call, ssp_order, axis='ssp')
            ssp_fetched = True
        except (DataFetchError, ConfigurationError) as exc:
            if ssp is None:
                raise
            if rd_ssp:
                warnings.warn(
                    f"fetch_environment: range-dependent SSP fetch failed "
                    f"({exc}); falling back to the supplied ssp= literal. A "
                    f"single-profile literal reduces the transect to "
                    f"range-independent.",
                    UserWarning, stacklevel=2)
            ssp_src = None                          # fall back to the literal ssp

    # ── Bottom (optional): fetch from bottom_sources / 'auto', else literal ──
    bottom_props, bottom_kw = None, None
    if want_bottom:
        order, bottom_cache_only = _bottom_order(
            bottom_sources if bottom_sources is not None else 'auto')
        # Scale grain-size geoacoustics to the in-situ near-seabed water sound
        # speed (the conversion is a velocity ratio; the Hamilton 1510 m/s
        # reference can be ~100 m/s off on a warm shelf / cold deep site).
        water_c = _near_seabed_sound_speed(ssp)
        try:
            if rd_bottom:
                bottom_props, bottom_kw = _fetch_bottom(
                    order, point, transect_to, transect=True,
                    cache_only=bottom_cache_only, water_sound_speed=water_c,
                    n_points=bottom_n_points, max_points=max_points,
                    timeout=timeout, verbose=verbose,
                )
            else:
                bottom_props, bottom_kw = _fetch_bottom(
                    order, point, transect=False, cache_only=bottom_cache_only,
                    water_sound_speed=water_c, timeout=timeout, verbose=verbose,
                )
        except (DataFetchError, ConfigurationError) as exc:
            if bottom is None:
                raise
            if rd_bottom:
                warnings.warn(
                    f"fetch_environment: range-dependent bottom fetch failed "
                    f"({exc}); falling back to the supplied bottom= literal. A "
                    f"uniform literal makes the bottom range-independent.",
                    UserWarning, stacklevel=2)
            bottom_props = _resolve_bottom(  # fall back to the literal
                bottom, water_sound_speed=water_c)
    elif bottom is not None:
        bottom_props = _resolve_bottom(
            bottom, water_sound_speed=_near_seabed_sound_speed(ssp))

    # ── Surface (top boundary, optional): fetch sea ice, else literal ──
    # The only fetchable surface is NSIDC sea ice; a point classified as open
    # water returns no boundary (the default free surface), with no provenance.
    surface_props, surface_src = None, None
    # range_dependent_surface implies the sea-ice source, mirroring how
    # range_dependent_ssp / _bottom auto-fetch their axis along the transect.
    if want_surface:
        # The only surface source is the cached sea-ice climatology, so 'auto'
        # and 'local' (and the implied default) are identical here.
        srcs = (('seaice',) if surface_sources in ('auto', 'local', None)
                else _as_source_tuple(surface_sources))
        for s in srcs:
            if s != 'seaice':
                raise ConfigurationError(
                    f"fetch_environment: unknown surface source {s!r}.",
                    remediation="Use 'seaice' or 'auto'.",
                )
        if date is None:
            raise ConfigurationError(
                "fetch_environment: surface_sources='seaice' needs date= to "
                "pick the climatological sea-ice month.",
                remediation="Pass date='YYYY-MM-DD', or supply surface= / drop it.",
            )
        try:
            if rd_surface:
                from uacpy.data.seaice_local import sea_ice_surface_transect
                fetched = sea_ice_surface_transect(
                    point, transect_to, date=date,
                    n_points=surface_n_points, max_points=max_points)
                # A transect that is open water everywhere → leave the default
                # free surface (no provenance), matching the point case.
                if fetched.is_elastic:
                    surface_props, surface_src = fetched, 'seaice'
            else:
                from uacpy.data.seaice_local import fetch_sea_ice_surface
                fetched = fetch_sea_ice_surface(point, date=date)
                if fetched is not None:                 # None = open water
                    surface_props, surface_src = fetched, 'seaice'
        except (DataFetchError, ConfigurationError):
            if surface is None:
                raise
            surface_props = surface                     # literal fallback
    elif surface is not None:
        surface_props = surface

    # Bathymetry (GEBCO) and SSP (WOA/Copernicus) come from independent
    # products, so their deepest points rarely coincide. Reconcile the SSP to
    # span exactly the fetched water column with the carrier's own method
    # (extend short profiles to the seafloor; trim points below it). We do NOT
    # resample onto a uniform grid — the native levels carry the real sampling,
    # and each model owns SSP interpolation via its ``interp_ssp`` scheme.
    depth_max = (float(bathymetry) if np.ndim(bathymetry) == 0
                 else float(np.max(np.asarray(bathymetry)[:, 1])))
    if ssp_fetched:
        ssp = ssp.extend_to(depth_max)
    # A literal ssp= passes straight to Environment, which coerces a scalar /
    # pairs / SoundSpeedProfile and reconciles its depth to the bathymetry.

    kwargs = dict(
        name=name or f"{lat:.3f}, {lon:.3f}",
        bathymetry=bathymetry,
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
    if with_absorption:
        kwargs['absorption'] = _fetch_absorption(
            point, date=date, ssp_source=ssp_src, ssp_backend=ssp_backend,
            cache_only=ssp_cache_only, timeout=timeout, verbose=verbose,
        )
    env = Environment(**kwargs)

    _record_provenance(env, bathy_src, ssp_src, bottom_kw, bottom_props, surface_src)
    return env


def _record_provenance(env, bathy_src, ssp_src, bottom_kw, bottom_props, surface_src):
    """Stamp ``env.data_sources`` by aggregating the per-layer provenance the
    fetchers attached to each carrier, in axis order
    (bathymetry → ssp → bottom → surface), de-duplicated by source id.

    Each fetched carrier carries its own ``data_sources`` — a tuple of
    :class:`~uacpy.data.sources.DataProvenance` records holding the dataset plus
    the **actual** date/coordinates it returned (which can differ from what was
    requested). Where a carrier wasn't stamped (older path, or a literal axis),
    fall back to the bare catalogue id so attribution is never lost. Warns on
    any non-commercial licence used."""
    def layer(carrier, src_id, extra_ids=()):
        prov = tuple(getattr(carrier, 'data_sources', ()) or ())
        if prov:
            return prov
        # Un-stamped layer (older path / literal axis): wrap the bare catalogue
        # id in a DataProvenance so env.data_sources is uniformly DataProvenance
        # — no date/coords, just the source. Keeps one record type in the tuple.
        ids = ([src_id] if src_id is not None else []) + list(extra_ids)
        return tuple(DataProvenance(source=SOURCES[i]) for i in ids)

    extra = (('globsed',)
             if getattr(bottom_props, 'sediment_thickness_source', None) == 'globsed'
             else ())
    records = (layer(env.bathymetry, bathy_src)
               + layer(env.ssp, ssp_src)
               + layer(bottom_props, bottom_kw, extra)
               + layer(getattr(env, 'surface', None), surface_src))
    seen, dedup = set(), []
    for r in records:
        if r.source.id not in seen:
            seen.add(r.source.id)
            dedup.append(r)
    env.data_sources = tuple(dedup)
    # A licence-restricted source must never enter a result silently: warn at
    # fetch time for any non-commercial dataset used (e.g. CRUST1.0). Driven off
    # the catalogue flag so future non-commercial sources are covered too.
    for prov in env.data_sources:
        src = prov.source
        if not src.commercial_use:
            warnings.warn(
                f"fetch_environment: data source {src.id!r} ({src.name}) does "
                f"not permit commercial use without verification — see "
                f"uacpy.data.citations(env) for its licence/attribution.",
                UserWarning, stacklevel=2)


def _fetch_absorption(point, *, date, ssp_source, ssp_backend, cache_only,
                      timeout, verbose):
    """Francois-Garrison absorption from the site's fetched T/S column.

    Reuses the backend the SSP resolved to (``ssp_backend``) for the WOA23 T/S
    column, so a cache-resolved SSP draws its absorption from the same cached
    grid rather than re-fetching it live. ``None`` (literal SSP) leaves the
    WOA fetcher on its own default. ``cache_only`` (a ``*_sources='local'``
    run) forces the local WOA23 grid so the T/S column never hits the network
    either — including when a cache-pinned SSP fell back to a literal.
    """
    from uacpy.data.absorption import build_francois_garrison
    if cache_only:
        from uacpy.data.sound_speed import fetch_ts_profile
        depths, temp, sal = fetch_ts_profile(
            point, date=date, source='local', timeout=timeout, verbose=verbose)
        return build_francois_garrison(depths, temp, sal)
    if ssp_source == 'copernicus':
        from uacpy.data.copernicus import fetch_ts_profile_operational
        depths, temp, sal = fetch_ts_profile_operational(
            point, date=date, timeout=timeout, verbose=verbose)
    elif ssp_source == 'argo':
        from uacpy.data.argo import fetch_argo_profile, _pressure_dbar_to_depth
        prof = fetch_argo_profile(point, date=date, timeout=timeout, verbose=verbose)
        depths = _pressure_dbar_to_depth(prof['pres'], prof['lat'])
        temp, sal = prof['temp'], prof['psal']
    else:
        from uacpy.data.sound_speed import fetch_ts_profile
        ts_kwargs = {} if ssp_backend is None else {'source': ssp_backend}
        depths, temp, sal = fetch_ts_profile(
            point, date=date, timeout=timeout, verbose=verbose, **ts_kwargs)
    return build_francois_garrison(depths, temp, sal)


def _fetch_ssp(point, *, date, ssp_source, formula, resolution, source,
               timeout, verbose):
    if ssp_source == 'woa23':
        return fetch_ssp(
            point, date=date, formula=formula, resolution=resolution,
            source=source, timeout=timeout, verbose=verbose,
        )
    if ssp_source == 'copernicus':
        if date is None:
            raise ConfigurationError(
                "fetch_environment: ssp_sources='copernicus' requires date=.",
                remediation="Pass a date, or use ssp_sources='woa23'.",
            )
        from uacpy.data.copernicus import fetch_ssp_operational
        return fetch_ssp_operational(
            point, date=date, formula=formula, timeout=timeout, verbose=verbose,
        )
    if ssp_source == 'argo':
        if date is None:
            raise ConfigurationError(
                "fetch_environment: ssp_sources='argo' requires date=.",
                remediation="Pass a date, or use ssp_sources='woa23'.",
            )
        from uacpy.data.argo import fetch_ssp_argo
        return fetch_ssp_argo(
            point, date=date, formula=formula, timeout=timeout, verbose=verbose,
        )
    raise ConfigurationError(
        f"fetch_environment: unknown ssp_source={ssp_source!r}.",
        remediation="Use 'woa23', 'copernicus' or 'argo'.",
    )


# Bottom data sources — single declarative registry. To add a sediment source:
# write its module, then add one ``_BottomProvider`` row below. Everything else
# (the accepted source keywords, the 'auto' fallback order, the fetcher lookup
# and the provenance id) derives from this list.
#
# ``resolve(cached)`` returns the source's ``(point_fetcher, transect_fetcher)``
# pair, imported lazily to keep optional deps optional and avoid import cycles.
# Only EMODnet has a distinct cached backend (local polygons vs the live WFS),
# tried cache-first; the rest are cache-or-compute and ignore the flag.


@dataclass(frozen=True)
class _BottomProvider:
    """One bottom data source. ``id`` doubles as the source keyword and the
    provenance catalogue id. ``in_auto`` puts it in the 'auto' fallback chain,
    ``in_cache_auto`` in the network-free 'local' chain. ``has_cached_variant``
    means it has a local twin tried before its live/compute backend
    (cache-first); under ``cache_only`` only the local twin is used."""

    id: str
    resolve: Callable                  # (cached: bool) -> (point_fn, transect_fn)
    has_cached_variant: bool = False
    in_auto: bool = False
    in_cache_auto: bool = False


def _emodnet_pair(cached):
    if cached:
        from uacpy.data import emodnet_local as m
        return (m.fetch_bottom_local, m.fetch_bottom_local_transect)
    from uacpy.data import seabed as m
    return (m.fetch_bottom, m.fetch_bottom_transect)


def _grainsize_pair(cached):
    from uacpy.data import sediment_db as m
    return (m.fetch_bottom_local, m.fetch_bottom_local_transect)


def _crust1_pair(cached):
    from uacpy.data import crust1_local as m
    return (m.fetch_bottom_crust1, m.fetch_bottom_crust1_transect)


def _diesing_pair(cached):
    from uacpy.data import diesing_local as m
    return (m.fetch_bottom_diesing, m.fetch_bottom_diesing_transect)


def _pelagic_pair(cached):
    from functools import partial
    from uacpy.data import pelagic as m
    # ``cached`` forbids the GEBCO live-API fallback in the depth lookup, so the
    # cached attempt (local GEBCO only) precedes the API-allowed one.
    return (partial(m.fetch_bottom_pelagic, cache_only=cached),
            partial(m.fetch_bottom_pelagic_transect, cache_only=cached))


_BOTTOM_PROVIDERS = (
    _BottomProvider('emodnet', _emodnet_pair, has_cached_variant=True,
                    in_auto=True, in_cache_auto=True),
    _BottomProvider('grainsize', _grainsize_pair, in_cache_auto=True),
    _BottomProvider('crust1', _crust1_pair),
    _BottomProvider('diesing', _diesing_pair, in_auto=True, in_cache_auto=True),
    _BottomProvider('pelagic', _pelagic_pair, has_cached_variant=True,
                    in_auto=True, in_cache_auto=True),  # never fails (last resort)
)
_BOTTOM_BY_ID = {p.id: p for p in _BOTTOM_PROVIDERS}
_AUTO_BOTTOM_ORDER = tuple(p.id for p in _BOTTOM_PROVIDERS if p.in_auto)
_CACHE_BOTTOM_ORDER = tuple(p.id for p in _BOTTOM_PROVIDERS if p.in_cache_auto)


def _bottom_order(bottom_source):
    """Ordered bottom source keywords + a ``cache_only`` flag from the user
    spec. ``'auto'`` → the best-available chain (EMODnet → Diesing → pelagic);
    ``'local'`` → the network-free chain (EMODnet local → grain-size → Diesing →
    pelagic, cached backends only); a str/sequence of keywords is used as-is.
    Validates each keyword."""
    if bottom_source == 'local':
        return _CACHE_BOTTOM_ORDER, True
    if bottom_source == 'auto':
        return _AUTO_BOTTOM_ORDER, False
    order = tuple(s.lower() for s in _as_source_tuple(bottom_source))
    for name in order:
        if name not in _BOTTOM_BY_ID:
            raise ConfigurationError(
                f"fetch_environment: unknown bottom source {name!r}.",
                remediation=f"Use 'auto', 'local' or one of {sorted(_BOTTOM_BY_ID)}.",
            )
    return order, False


def _fetch_bottom(order, *args, transect, cache_only=False, **kwargs):
    """Fetch a bottom from the first source in ``order`` that yields data.

    ``transect`` selects the point (``False``) or transect (``True``) fetcher.
    Cache-first within each source (EMODnet tries its local polygons before the
    live WFS); ``cache_only`` keeps only cached backends. ``args``/``kwargs``
    are forwarded. Returns ``(bottom, source_keyword)``; a source with no
    coverage (or no installed cache) falls through to the next.
    """
    idx = 1 if transect else 0
    errors = []
    for name in order:
        provider = _BOTTOM_BY_ID[name]
        # Cache-first: the local twin before the live backend, where one exists;
        # cache_only drops the live attempt.
        if provider.has_cached_variant:
            cached_flags = (True,) if cache_only else (True, False)
        else:
            cached_flags = (False,)
        for cached in cached_flags:
            fn = provider.resolve(cached)[idx]
            try:
                return fn(*args, **kwargs), name
            except (DataFetchError, ConfigurationError) as exc:
                errors.append(exc)
    data_errs = [e for e in errors if isinstance(e, DataFetchError)]
    raise (data_errs[0] if data_errs else errors[-1])


def _near_seabed_sound_speed(ssp):
    """Deepest in-water sound speed (m/s) from a resolved SSP, or ``None``.

    Used to scale grain-size geoacoustics to the in-situ near-seabed water
    instead of the Hamilton 1510 m/s reference. ``ssp`` may be a
    :class:`SoundSpeedProfile`, a scalar, an array of ``(depth, speed)`` pairs,
    or ``None``. Returns ``None`` when no usable value can be derived.
    """
    if ssp is None:
        return None
    if isinstance(ssp, (int, float)):
        return float(ssp)
    data = getattr(ssp, 'data', None)
    if data is not None and getattr(data, 'size', 0) > 0:
        # SoundSpeedProfile.depths is strictly increasing → deepest row;
        # take the r = 0 column for range-dependent profiles.
        return float(np.asarray(data, dtype=float)[-1, 0])
    try:
        arr = np.asarray(ssp, dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return float(arr[np.argmax(arr[:, 0]), 1])
    except (ValueError, TypeError):
        pass
    return None


def _resolve_bottom(bottom, *, water_sound_speed=None):
    if bottom is None or isinstance(bottom, BoundaryProperties):
        return bottom
    if isinstance(bottom, str):
        return bottom_from_class(bottom)
    if isinstance(bottom, (int, float)):
        return bottom_from_grain_size(
            float(bottom), water_sound_speed=water_sound_speed)
    raise ConfigurationError(
        f"fetch_environment: bottom must be a ϕ float, a class name, a "
        f"BoundaryProperties, or None; got {type(bottom).__name__}.",
    )
