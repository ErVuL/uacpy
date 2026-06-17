"""``fetch_environment`` — GPS (+ date) → a ready-to-run ``Environment``.

The capstone of the on-demand data layer: assemble bathymetry, sound speed
and (optionally) a bottom into a single :class:`~uacpy.core.environment.\
Environment` that drops straight into any propagation model.

    env = uacpy.data.fetch_environment((43.2, 7.5), date='2026-06-14',
                                       bottom='sand')

Each axis is sourced independently:

* **bathymetry** — GEBCO point depth, or a range-dependent transect when
  ``transect_to`` is given (:mod:`uacpy.data.bathymetry`);
* **sound speed** — WOA23 climatology (default) or Copernicus operational
  (``ssp_source='copernicus'``, needs the optional toolbox + login);
* **bottom** — a ϕ grain size (float), a sediment-class name (str), a ready
  ``BoundaryProperties``, or a fetch keyword (``'emodnet'`` / ``'grainsize'`` /
  ``'auto'``). Omitted when ``bottom=None``.
"""

import datetime as _dt
from typing import Optional, Union

import numpy as np

from uacpy.core.environment import BoundaryProperties, Environment
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data._geo import Coordinate, as_coordinate
from uacpy.data.bathymetry import fetch_bathy, fetch_bathy_transect
from uacpy.data.sediment import bottom_from_class, bottom_from_grain_size
from uacpy.data.sound_speed import fetch_ssp, fetch_ssp_transect
from uacpy.data.sources import SOURCES

__all__ = ['fetch_environment']


def _source_order(prefer_cache, online, local='local'):
    """Access order for one dataset: cache→live (``prefer_cache=True``) or
    live→cache (``prefer_cache=False``). Whichever is tried first, the other is
    the fallback."""
    return (local, online) if prefer_cache else (online, local)


def _resolve_cached(call, order):
    """Call ``call(source)`` for each source in ``order``, falling back to the
    next on **any** fetch failure (cache absent → ``ConfigurationError``; live
    failure / no coverage → ``DataFetchError``). Returns ``(result,
    source_used)``; re-raises the last error if every source fails."""
    last = None
    for src in order:
        try:
            return call(src), src
        except (ConfigurationError, DataFetchError) as exc:
            last = exc
    raise last


def fetch_environment(
    point: Coordinate,
    *,
    date: Union[str, _dt.date, None] = None,
    name: Optional[str] = None,
    ssp_source: str = 'woa23',
    bottom: Union[float, str, BoundaryProperties, None] = None,
    transect_to: Optional[Coordinate] = None,
    n_points: int = 50,
    range_dependent_ssp: bool = False,
    ssp_n_points: int = 6,
    range_dependent_bottom: bool = False,
    bottom_n_points: int = 6,
    sea_ice: bool = False,
    with_absorption: bool = False,
    formula: str = 'unesco',
    resolution: str = '1.00',
    bathymetry_source: str = 'api',
    prefer_cache: bool = False,
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
        time step for Copernicus (required when ``ssp_source='copernicus'``).
    name : str, optional
        Environment name. Defaults to the coordinate string.
    ssp_source : {'woa23', 'copernicus', 'argo'}, optional
        Sound-speed backend. Default ``'woa23'`` (climatology); ``'copernicus'``
        (operational model) and ``'argo'`` (nearest real float profile) both need
        ``date=`` and network.
    bottom : float or str or BoundaryProperties, optional
        Seafloor spec — one of: a mean grain size (ϕ, float); a
        :data:`~uacpy.materials.MATERIALS` class name; a fetch keyword
        ``'emodnet'`` (EMODnet, European seas, high-res, CC-BY), ``'grainsize'``
        (NCEI grain-size samples, worldwide, public-domain — needs the local
        cache), or ``'auto'`` (EMODnet online, or the local grain-size samples
        when ``offline``); a ready ``BoundaryProperties``; or ``None`` (default,
        bottom unset). With ``range_dependent_bottom`` the same keyword selects
        the source sampled along the transect. All sources permit commercial use.
    transect_to : (lat, lon), optional
        If given, bathymetry is sampled along the great-circle path from
        ``(lat, lon)`` to here (range-dependent); otherwise a single depth.
    n_points : int, optional
        Bathymetry transect sample count (used only with ``transect_to``).
    range_dependent_ssp : bool, optional
        If ``True`` (requires ``transect_to``), the sound-speed profile is also
        sampled along the transect → a 2-D, range-dependent SSP sharing the
        transect's range axis. Default ``False`` (single profile at the start
        point). WOA23 only. Each column is several requests, hence the separate
        ``ssp_n_points``.
    ssp_n_points : int, optional
        Number of SSP columns along the transect when ``range_dependent_ssp``.
        Default 6.
    range_dependent_bottom : bool, optional
        If ``True`` (requires ``transect_to``), the seafloor is sampled along
        the transect → a ``RangeDependentBottom`` sharing the range axis. The
        source follows the ``bottom`` keyword (default ``'auto'``). Default
        ``False``. Note: a single source is used for the whole transect, so gaps
        outside its coverage forward-fill the nearest covered value — pass
        ``bottom='grainsize'`` for a uniform worldwide (public-domain) source.
    bottom_n_points : int, optional
        Number of EMODnet samples along the transect when
        ``range_dependent_bottom``. Default 6.
    sea_ice : bool, optional
        If ``True`` (requires ``date=`` and the cached ``seaice`` climatology —
        ``install.sh --data seaice``), set the surface boundary from the NSIDC
        sea-ice concentration at the point for ``date``'s month: an ice-covered
        point (≥15 %, the NSIDC ice-edge) gets a homogeneous elastic ice canopy
        (cp 3500 m/s, cs 1800 m/s, ρ 0.9, αp/αs 0.5/1.0 dB/λ — *Computational
        Ocean Acoustics*); open water keeps the default free surface. Point
        classification only (the carrier's surface is a single boundary).
        Default ``False``.
    with_absorption : bool, optional
        If ``True``, attach a Francois-Garrison absorption built from the
        site's fetched temperature/salinity column (costs one extra T/S
        request). Default ``False`` (model-default Thorp absorption).
    bathymetry_source : {'api', 'gmrt'}, optional
        Online bathymetry backend: ``'api'`` (default, OpenTopoData/GEBCO) or
        ``'gmrt'`` (GMRT multibeam synthesis — higher resolution where surveyed,
        CC-BY). The local GEBCO grid is the cache twin (see ``prefer_cache``).
    prefer_cache : bool, optional
        Per-axis access order, with the *other* access as a fallback if the first
        fails. ``False`` (default) → **live first, fall back to the local cache**
        (e.g. when the network is down); ``True`` → **local cache first, fall back
        to the live service** (e.g. when a dataset isn't installed). Either way,
        bathymetry/WOA23-SSP/bottom each try one access then the other. Sources
        with no local twin (GMRT, Copernicus, Argo) have no fallback. For a
        guaranteed no-network run, call the low-level fetchers with
        ``source='local'`` (which raise if a dataset isn't cached).
    formula, resolution, timeout, verbose
        Forwarded to the sound-speed / bathymetry fetchers.

    Returns
    -------
    Environment
    """
    lat, lon = as_coordinate(point)
    bathy_order = _source_order(prefer_cache, bathymetry_source)
    woa_order = (_source_order(prefer_cache, 'opendap')
                 if ssp_source == 'woa23' else ('opendap',))

    def _fetch_bathy(src):
        if transect_to is None:
            return fetch_bathy(point, source=src, timeout=timeout,
                                     verbose=verbose)
        return fetch_bathy_transect(point, transect_to, n_points=n_points, source=src,
                              timeout=timeout, verbose=verbose)
    bathymetry, bathy_source = _resolve_cached(_fetch_bathy, bathy_order)

    if range_dependent_ssp:
        if transect_to is None:
            raise ConfigurationError(
                "fetch_environment: range_dependent_ssp=True requires transect_to=.",
                remediation="Pass transect_to=(lat, lon), or leave "
                            "range_dependent_ssp=False for a single profile.",
            )
        if ssp_source == 'copernicus':
            if date is None:
                raise ConfigurationError(
                    "fetch_environment: ssp_source='copernicus' requires date=.",
                )
            from uacpy.data.copernicus import fetch_ssp_transect_operational
            ssp = fetch_ssp_transect_operational(
                point, transect_to, date, n_points=ssp_n_points,
                formula=formula, timeout=timeout, verbose=verbose,
            )
        elif ssp_source == 'woa23':
            ssp, _ = _resolve_cached(
                lambda src: fetch_ssp_transect(
                    point, transect_to, n_points=ssp_n_points, date=date,
                    formula=formula, resolution=resolution, source=src,
                    timeout=timeout, verbose=verbose),
                woa_order)
        else:
            raise ConfigurationError(
                f"fetch_environment: unknown ssp_source={ssp_source!r}.",
                remediation="Use 'woa23' or 'copernicus'.",
            )
    else:
        ssp, _ = _resolve_cached(
            lambda src: _fetch_ssp(
                point, date=date, ssp_source=ssp_source, formula=formula,
                resolution=resolution, source=src, timeout=timeout,
                verbose=verbose),
            woa_order)
    if range_dependent_bottom:
        if transect_to is None:
            raise ConfigurationError(
                "fetch_environment: range_dependent_bottom=True requires transect_to=.",
                remediation="Pass transect_to=(lat, lon), or leave "
                            "range_dependent_bottom=False for a uniform bottom.",
            )
        bottom_props, bottom_kw = _fetch_bottom(
            bottom, point, transect_to, transect=True, prefer_cache=prefer_cache,
            n_points=bottom_n_points, timeout=timeout, verbose=verbose,
        )
    elif _is_bottom_source(bottom):
        bottom_props, bottom_kw = _fetch_bottom(
            bottom, point, transect=False,
            prefer_cache=prefer_cache, timeout=timeout, verbose=verbose,
        )
    else:
        bottom_props, bottom_kw = _resolve_bottom(bottom), None

    surface_props = None
    if sea_ice:
        if date is None:
            raise ConfigurationError(
                "fetch_environment: sea_ice=True needs date= to pick the "
                "climatological sea-ice month.",
                remediation="Pass date='YYYY-MM-DD', or leave sea_ice=False.",
            )
        from uacpy.data.seaice_local import fetch_sea_ice_surface
        surface_props = fetch_sea_ice_surface(point, date)

    # Bathymetry (GEBCO) and SSP (WOA/Copernicus) come from independent
    # products, so their deepest points rarely coincide. Reconcile the SSP to
    # span exactly the fetched water column with the carrier's own method
    # (extend short profiles to the seafloor; trim points below it). We do NOT
    # resample onto a uniform grid — the native levels carry the real sampling,
    # and each model owns SSP interpolation via its ``interp_ssp`` scheme.
    depth_max = (float(bathymetry) if np.ndim(bathymetry) == 0
                 else float(np.max(np.asarray(bathymetry)[:, 1])))
    ssp = ssp.extend_to(depth_max)

    kwargs = dict(
        name=name or f"{lat:.3f}, {lon:.3f}",
        bathymetry=bathymetry,
        ssp=ssp,
    )
    if bottom_props is not None:
        kwargs['bottom'] = bottom_props
    if surface_props is not None:
        kwargs['surface'] = surface_props
    if with_absorption:
        kwargs['absorption'] = _fetch_absorption(
            point, date=date, ssp_source=ssp_source,
            timeout=timeout, verbose=verbose,
        )
    env = Environment(**kwargs)

    # Provenance: which catalogue sources this environment was built from.
    used = [_BATHY_SOURCE_ID[bathy_source], ssp_source]
    if bottom_kw is not None:
        used.append(_BOTTOM_SOURCE_ID[bottom_kw])
        # CRUST1.0 rescales its sediment column with GlobSed thickness by
        # default; record GlobSed too whenever it was actually applied.
        if getattr(bottom_props, 'sediment_thickness_source', None) == 'globsed':
            used.append('globsed')
    if surface_props is not None:
        used.append('seaice')
    env.data_sources = tuple(SOURCES[i] for i in dict.fromkeys(used))
    return env


def _fetch_absorption(point, *, date, ssp_source, timeout, verbose):
    """Francois-Garrison absorption from the site's fetched T/S column."""
    from uacpy.data.absorption import build_francois_garrison
    if ssp_source == 'copernicus':
        from uacpy.data.copernicus import fetch_ts_profile_operational
        depths, temp, sal = fetch_ts_profile_operational(
            point, date, timeout=timeout, verbose=verbose)
    elif ssp_source == 'argo':
        from uacpy.data.argo import fetch_argo_profile, _pressure_dbar_to_depth
        prof = fetch_argo_profile(point, date, timeout=timeout, verbose=verbose)
        depths = _pressure_dbar_to_depth(prof['pres'], prof['lat'])
        temp, sal = prof['temp'], prof['psal']
    else:
        from uacpy.data.sound_speed import fetch_ts_profile
        depths, temp, sal = fetch_ts_profile(
            point, date=date, timeout=timeout, verbose=verbose)
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
                "fetch_environment: ssp_source='copernicus' requires date=.",
                remediation="Pass a date, or use ssp_source='woa23'.",
            )
        from uacpy.data.copernicus import fetch_ssp_operational
        return fetch_ssp_operational(
            point, date, formula=formula, timeout=timeout, verbose=verbose,
        )
    if ssp_source == 'argo':
        if date is None:
            raise ConfigurationError(
                "fetch_environment: ssp_source='argo' requires date=.",
                remediation="Pass a date, or use ssp_source='woa23'.",
            )
        from uacpy.data.argo import fetch_ssp_argo
        return fetch_ssp_argo(
            point, date, formula=formula, timeout=timeout, verbose=verbose,
        )
    raise ConfigurationError(
        f"fetch_environment: unknown ssp_source={ssp_source!r}.",
        remediation="Use 'woa23', 'copernicus' or 'argo'.",
    )


# Bottom data sources. ``'auto'`` tries the order below, falling through where a
# source has no coverage: online → EMODnet WFS (European seas); offline → the
# local EMODnet polygons (European seas) then the global grain-size DB. Each
# source has the right backend selected by ``offline`` in ``_bottom_registry``.
# To add a sediment source: write its module, add one row to ``_bottom_registry``
# and (if it joins 'auto') to the order tuples and ``_BOTTOM_SOURCE_KEYWORDS``.
_BOTTOM_SOURCE_KEYWORDS = ('emodnet', 'grainsize', 'crust1', 'diesing',
                           'pelagic', 'auto')
# 'auto' resolves best-available, falling through where a source has no coverage
# or isn't installed: measured/regional first, then the global Diesing deep-sea
# map, then the first-principles 'pelagic' rule (never fails) as the last resort.
_AUTO_BOTTOM_ORDER = ('emodnet', 'diesing', 'pelagic')
_OFFLINE_AUTO_BOTTOM_ORDER = ('emodnet', 'grainsize', 'diesing', 'pelagic')


def _is_bottom_source(bottom):
    return isinstance(bottom, str) and bottom.lower() in _BOTTOM_SOURCE_KEYWORDS


def _bottom_registry(offline=False):
    """``source -> (point_fetcher, transect_fetcher)``. Lazily imported to keep
    optional dependencies optional and avoid import cycles. ``offline`` swaps the
    EMODnet backend from the live WFS to the local polygon index."""
    from uacpy.data import crust1_local, diesing_local, pelagic, sediment_db
    grainsize = (sediment_db.fetch_bottom_local,
                 sediment_db.fetch_bottom_local_transect)
    crust1 = (crust1_local.fetch_bottom_crust1,
              crust1_local.fetch_bottom_crust1_transect)
    diesing = (diesing_local.fetch_bottom_diesing,
               diesing_local.fetch_bottom_diesing_transect)
    pelagic_pair = (pelagic.fetch_bottom_pelagic,
                    pelagic.fetch_bottom_pelagic_transect)
    common = {'grainsize': grainsize, 'crust1': crust1, 'diesing': diesing,
              'pelagic': pelagic_pair}
    if offline:
        from uacpy.data import emodnet_local
        return {'emodnet': (emodnet_local.fetch_bottom_local,
                            emodnet_local.fetch_bottom_local_transect), **common}
    from uacpy.data import seabed
    return {'emodnet': (seabed.fetch_bottom, seabed.fetch_bottom_transect),
            **common}


# Bottom source keyword → data-source catalogue id (for provenance).
_BOTTOM_SOURCE_ID = {'emodnet': 'emodnet', 'grainsize': 'grainsize',
                     'crust1': 'crust1', 'diesing': 'diesing',
                     'pelagic': 'pelagic'}

# Bathymetry source → catalogue id ('local' is still the GEBCO product).
_BATHY_SOURCE_ID = {'api': 'gebco', 'gmrt': 'gmrt', 'local': 'gebco'}


def _fetch_bottom(bottom, *args, transect, prefer_cache=False, **kwargs):
    """Fetch a bottom from a source keyword, with 'auto' fallback.

    ``transect`` selects the point (``False``) or transect (``True``) fetcher.
    ``prefer_cache`` sets the per-source access order: local-then-online when
    ``True``, online-then-local when ``False`` (only ``emodnet`` actually differs —
    the others are cache-or-compute either way). ``args``/``kwargs`` are forwarded.
    Returns ``(bottom, source_keyword)``. In 'auto', a source with no coverage (or
    no installed cache) falls through to the next.
    """
    idx = 1 if transect else 0
    source = bottom.lower() if _is_bottom_source(bottom) else 'auto'
    if source == 'auto':
        order = _OFFLINE_AUTO_BOTTOM_ORDER if prefer_cache else _AUTO_BOTTOM_ORDER
    else:
        order = (source,)
    local_reg = _bottom_registry(offline=True)
    online_reg = _bottom_registry(offline=False)
    regs = (local_reg, online_reg) if prefer_cache else (online_reg, local_reg)
    last_error = None
    for name in order:
        tried = []
        for reg in regs:                              # preferred access, then fallback
            fn = reg[name][idx] if name in reg else None
            if fn is None or fn in tried:             # skip the cache-or-compute dups
                continue
            tried.append(fn)
            try:
                return fn(*args, **kwargs), name
            except (DataFetchError, ConfigurationError) as exc:
                last_error = exc
    raise last_error


def _resolve_bottom(bottom):
    if bottom is None or isinstance(bottom, BoundaryProperties):
        return bottom
    if isinstance(bottom, str):
        return bottom_from_class(bottom)
    if isinstance(bottom, (int, float)):
        return bottom_from_grain_size(float(bottom))
    raise ConfigurationError(
        f"fetch_environment: bottom must be a ϕ float, a class name, a "
        f"BoundaryProperties, or None; got {type(bottom).__name__}.",
    )
