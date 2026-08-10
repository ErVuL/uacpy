"""Offline CRUST1.0 layered crustal model (UCSD) → layered elastic bottom.

``install.sh --data crust1`` downloads CRUST1.0 (Laske et al. 2013): a global
1°×1° model giving, per cell, **Vp, Vs, density and boundary depth** for 8
layers (water, ice, 3 sediment, 3 crystalline-crust) plus the sub-Moho mantle.
This module turns the column at a point into a uacpy layered **elastic** bottom —
the description the *low-frequency* field actually needs (it penetrates the whole
sediment column down to basement, and shear matters), unlike a surficial
grain-size half-space (:mod:`uacpy.data.sediment`).

The bottom is the **sediment stack over the crystalline-crust half-space**
(``Vs`` retained → elastic). CRUST1.0 carries no attenuation, so ``α`` is
assigned from nominal defaults (overridable). It is a 1° crustal-scale average —
excellent for the deep layered/elastic structure, coarse for fine surficial
detail; the coarse sediment column is therefore rescaled to the
higher-resolution **GlobSed** total thickness (:mod:`uacpy.data.globsed_local`)
**by default**, falling back to CRUST1.0's own column where GlobSed is
unavailable.

Licensing: CRUST1.0 ships with **no formal licence** — the only stated obligation
is to cite Laske et al. 2013; commercial terms are unspecified, so verify before
commercial use. Downloaded at install time, never bundled.
"""

import io
import tarfile
from pathlib import Path

import re
import warnings

import numpy as np

from uacpy._log import log_message
from uacpy.core.environment import (
    BoundaryProperties, SeabedColumn, Bottom, SedimentLayer,
)
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data import _cache
from uacpy.data._geo import as_coordinate, normalize_lon
from uacpy.data._http import http_get, checked_member_size

__all__ = ['download_crust1_db', 'fetch_crust1_profile', 'fetch_bottom_crust1',
           'fetch_bottom_crust1_transect']

CRUST1_URL = 'https://igppweb.ucsd.edu/~gabi/crust1/crust1.0.tar.gz'
_FILES = ('crust1.bnds', 'crust1.vp', 'crust1.vs', 'crust1.rho')
_NLAT, _NLON, _NLAYER = 180, 360, 9
# 9 columns per cell: water, ice, upper/middle/lower sediment,
# upper/middle/lower crystalline crust, mantle (below Moho).
_UPPER_SED, _LOW_SED = 2, 4
_UPPER_CRYST, _MID_CRYST, _LOW_CRYST = 5, 6, 7

# CRUST1.0 has no Q; nominal compressional attenuation (dB/λ) by default.
DEFAULT_SEDIMENT_ATTENUATION = 0.5
DEFAULT_BASEMENT_ATTENUATION = 0.1

# CRUST1.0 is the catalogue's only commercial_use=False source and ships with no
# formal licence. Direct callers of the low-level fetcher bypass the orchestrator
# warning in fetch_environment, so warn here too — a non-commercial dataset must
# never enter a result silently (DEV.md §7.1).
_COMMERCIAL_WARNING = (
    "CRUST1.0 has no formal licence and does not permit commercial use without "
    "verification — cite Laske et al. 2013 and verify terms before commercial "
    "use. See uacpy.data.citations() for attribution."
)


def _warn_non_commercial():
    warnings.warn(_COMMERCIAL_WARNING, UserWarning, stacklevel=3)

# Below this total sediment thickness (m) the seabed is treated as bare rock —
# a thinner column (e.g. at a spreading-ridge crest, or after a near-zero GlobSed
# rescale) is acoustically negligible and would only yield a sub-resolution
# sediment medium downstream.
_MIN_SEDIMENT_M = 1.0

_MODEL = {}   # cache_root -> dict(bnds=, vp=, vs=, rho=) each (64800, 9)
_cache.register_cache(_MODEL.clear)


def download_crust1_db(cache_dir=None, *, timeout=180.0, verbose=False):
    """Download + extract the CRUST1.0 grids (``crust1.bnds/vp/vs/rho``).

    Writes the four ASCII grids into ``<cache>/crust1/`` and returns that dir.
    """
    dest = Path(cache_dir) if cache_dir else _cache.dataset_root('crust1')
    dest.mkdir(parents=True, exist_ok=True)
    log_message('crust1', "downloading CRUST1.0 (Laske et al. 2013, ~1 MB)",
                verbose=verbose)
    blob = http_get(CRUST1_URL, timeout=timeout, verbose=verbose, source='crust1')
    tf = tarfile.open(fileobj=io.BytesIO(blob))
    written = 0
    for member in tf.getmembers():
        name = Path(member.name).name
        if name in _FILES and not Path(member.name).name.startswith('._'):
            checked_member_size(member.size, name)
            (dest / name).write_bytes(tf.extractfile(member).read())
            written += 1
    if written < len(_FILES):
        raise DataFetchError(
            "CRUST1.0 archive did not contain the expected grids.",
            remediation="Retry; the upstream archive layout may have changed.",
        )
    _MODEL.clear()
    log_message('crust1', f"CRUST1.0 grids cached → {dest}", verbose=verbose)
    return dest


def _model():
    """Load (or reuse) the four CRUST1.0 grids as ``(64800, 9)`` arrays."""
    root = str(_cache.cache_root())
    if root in _MODEL:
        return _MODEL[root]
    grids = {}
    for fname in _FILES:
        path = _cache.require('crust1', fname)
        arr = np.loadtxt(path)
        if arr.shape != (_NLAT * _NLON, _NLAYER):
            raise DataFetchError(
                f"CRUST1.0 file {fname} has unexpected shape {arr.shape}.",
                remediation="Re-run ./install.sh --data crust1.",
            )
        grids[fname.split('.')[-1]] = arr      # 'bnds' / 'vp' / 'vs' / 'rho'
    _MODEL[root] = grids
    return grids


def _column(lat, lon):
    """The 9-layer (bnds, vp, vs, rho) vectors at a point (nearest 1° cell).

    ``bnds[i]`` is the **elevation of the top of layer i in km, positive up** —
    so it is negative below sea level, and a layer thickness is the *downward*
    difference ``bnds[i] - bnds[i+1]``. Vp/Vs are km/s and rho is g/cm³.
    """
    # Grid: row 0 = 89.5°N → row 179 = −89.5°N; col 0 = −179.5° → col 359 = 179.5°.
    row = int(np.clip(89 - np.floor(lat), 0, _NLAT - 1))
    col = int(np.clip(np.floor(normalize_lon(lon)) + 180, 0, _NLON - 1))
    idx = row * _NLON + col
    g = _model()
    return g['bnds'][idx], g['vp'][idx], g['vs'][idx], g['rho'][idx]


def _layer(i, bnds, vp, vs, rho, atten, elastic):
    """A :class:`SedimentLayer` for CRUST1.0 layer ``i`` (km/s → m/s, km → m)."""
    cs = vs[i] * 1000.0 if elastic else 0.0
    return SedimentLayer(
        thickness=(bnds[i] - bnds[i + 1]) * 1000.0,
        sound_speed=vp[i] * 1000.0, density=float(rho[i]), attenuation=atten,
        shear_speed=cs, shear_attenuation=(atten if cs > 0 else 0.0),
    )


def _halfspace(i, vp, vs, rho, atten, elastic):
    cs = vs[i] * 1000.0 if elastic else 0.0
    return BoundaryProperties(
        acoustic_type='half-space', sound_speed=vp[i] * 1000.0,
        density=float(rho[i]), attenuation=atten,
        shear_speed=cs, shear_attenuation=(atten if cs > 0 else 0.0),
    )


def _layered_from_column(bnds, vp, vs, rho, *, sediment_attenuation,
                         basement_attenuation, elastic, sediment_thickness):
    """Build a :class:`SeabedColumn`: sediment stack over crystalline basement."""
    # CRUST1.0 quantises every boundary to 0.01 km, so the thinnest layer it can
    # express is 10 m and any sub-metre threshold separates "absent" from
    # "present" identically — this one is in km, the one in
    # fetch_crust1_profile is in m.
    sed = [i for i in range(_UPPER_SED, _LOW_SED + 1)
           if bnds[i] - bnds[i + 1] > 1e-3]
    # Effective sediment thickness (the GlobSed override when given, else the
    # CRUST1.0 column). Below ``_MIN_SEDIMENT_M`` the column is negligible and a
    # real layer would only produce a sub-resolution medium, so emit bare rock.
    if sed:
        crust1_total = sum((bnds[i] - bnds[i + 1]) * 1000.0 for i in sed)
        eff_sed = (sediment_thickness
                   if (sediment_thickness is not None and sediment_thickness > 0)
                   else crust1_total)
    else:
        eff_sed = 0.0
    if sed and eff_sed >= _MIN_SEDIMENT_M:
        layers = [_layer(i, bnds, vp, vs, rho, sediment_attenuation, elastic)
                  for i in sed]
        if sediment_thickness is not None and sediment_thickness > 0:
            total = sum(layer.thickness for layer in layers)
            if total > 0:
                scale = sediment_thickness / total
                for layer in layers:
                    layer.thickness *= scale
        halfspace = _halfspace(_UPPER_CRYST, vp, vs, rho, basement_attenuation,
                               elastic)
    else:           # bare rock / negligible sediment: crust over crust
        layers = [_layer(_UPPER_CRYST, bnds, vp, vs, rho, basement_attenuation,
                         elastic)]
        halfspace = _halfspace(_MID_CRYST, vp, vs, rho, basement_attenuation,
                               elastic)
    return SeabedColumn(layers=layers, halfspace=halfspace)


def _globsed_thickness(point, *, verbose):
    """GlobSed total sediment thickness (m) at ``point`` for CRUST1.0 rescaling.

    Returns ``None`` when GlobSed is unavailable (dataset not cached) or has no
    value at the point (land / unmapped) so the caller keeps CRUST1.0's own
    1°-scale sediment column. GlobSed is the higher-resolution total thickness,
    used by default to rescale the coarse CRUST1.0 sediment stack.
    """
    from uacpy.data.globsed_local import fetch_sediment_thickness
    try:
        return fetch_sediment_thickness(point)
    except (DataFetchError, ConfigurationError):
        log_message('crust1', f"GlobSed thickness unavailable at {point}; "
                    "keeping CRUST1.0 native sediment column", verbose=verbose)
        return None


def fetch_crust1_profile(point):
    """Inspect the CRUST1.0 column at a point.

    Returns ``{'water_depth_m', 'sediment_thickness_m', 'layers'}`` where each
    layer is ``{'name', 'thickness_m', 'vp', 'vs', 'rho'}`` (m, m/s, m/s, g/cm³).

    ``water_depth_m`` is the elevation of the base of the water layer, negated —
    so on land or under an ice sheet, where CRUST1.0 puts no water, it comes back
    **negative** and equals minus the ground/ice surface elevation.
    """
    lat, lon = as_coordinate(point)
    bnds, vp, vs, rho = _column(lat, lon)
    names = ['water', 'ice', 'upper_sed', 'mid_sed', 'low_sed',
             'upper_cryst', 'mid_cryst', 'low_cryst']
    layers = []
    for i in range(_UPPER_SED, _LOW_CRYST + 1):
        thk = (bnds[i] - bnds[i + 1]) * 1000.0
        if thk > 1e-3:
            layers.append({'name': names[i], 'thickness_m': float(thk),
                           'vp': float(vp[i] * 1000), 'vs': float(vs[i] * 1000),
                           'rho': float(rho[i])})
    sed_thk = sum(layer['thickness_m'] for layer in layers
                  if layer['name'].endswith('sed'))
    return {'water_depth_m': float(-bnds[1] * 1000.0),
            'sediment_thickness_m': float(sed_thk), 'layers': layers}


def fetch_bottom_crust1(point, *, sediment_attenuation=DEFAULT_SEDIMENT_ATTENUATION,
                        basement_attenuation=DEFAULT_BASEMENT_ATTENUATION,
                        elastic=True, sediment_thickness=None, use_globsed=True,
                        water_sound_speed=None,
                        timeout=None, verbose=False):
    """Layered **elastic** bottom from CRUST1.0 at a ``(lat, lon)`` point.

    The CRUST1.0 sediment layers over the crystalline-crust half-space, with
    ``Vs`` retained (``elastic=True``). ``sediment_attenuation`` /
    ``basement_attenuation`` set the (nominal) dB/λ losses CRUST1.0 lacks.

    The coarse 1° CRUST1.0 sediment column is rescaled to a higher-resolution
    **GlobSed** total thickness by default (``use_globsed=True``); pass an
    explicit ``sediment_thickness`` (m) to override, or ``use_globsed=False`` to
    keep CRUST1.0's own column. Where GlobSed is not cached or has no value at
    the point, CRUST1.0's native thickness is kept. The returned bottom carries
    ``.sediment_thickness_source`` (``'globsed'`` or ``None``) recording which
    was used. ``timeout`` is ignored (offline) for signature parity with the
    network bottom fetchers; ``water_sound_speed`` is likewise accepted and
    ignored (CRUST1.0 yields absolute Vp/Vs/ρ, not water-referenced ratios).
    """
    _warn_non_commercial()
    lat, lon = as_coordinate(point)
    bnds, vp, vs, rho = _column(lat, lon)
    globsed_applied = False
    if sediment_thickness is None and use_globsed:
        sediment_thickness = _globsed_thickness(point, verbose=verbose)
        globsed_applied = sediment_thickness is not None
    bottom = _layered_from_column(
        bnds, vp, vs, rho, sediment_attenuation=sediment_attenuation,
        basement_attenuation=basement_attenuation, elastic=elastic,
        sediment_thickness=sediment_thickness)
    bottom.sediment_thickness_source = 'globsed' if globsed_applied else None
    log_message('crust1', f"CRUST1.0 at {lat:.2f}, {lon:.2f} → {bottom!r}",
                verbose=verbose)
    return bottom


def fetch_bottom_crust1_transect(start, end, *, n_points=6, max_points=None,
                                 sediment_attenuation=DEFAULT_SEDIMENT_ATTENUATION,
                                 basement_attenuation=DEFAULT_BASEMENT_ATTENUATION,
                                 elastic=True, use_globsed=True,
                                 water_sound_speed=None,
                                 timeout=None, verbose=False):
    """Range-dependent layered bottom from CRUST1.0 along ``start`` → ``end``.

    Each profile rescales its CRUST1.0 sediment column to the local GlobSed
    thickness by default (``use_globsed``); the result carries
    ``.sediment_thickness_source`` (``'globsed'`` if any waypoint used it).
    ``max_points`` caps the waypoint count (signature parity with the other
    transect bottom fetchers); CRUST1.0 is a cached 1° grid, so the sole effect
    is clamping ``n_points``.
    """
    from uacpy.data._geo import geodesic_waypoints
    if max_points is not None:
        n_points = max(2, min(int(n_points), int(max_points)))
    _warn_non_commercial()
    lats, lons, ranges_m = geodesic_waypoints(start, end, n_points)
    with warnings.catch_warnings():
        # The per-waypoint fetches would each re-warn; emit once at the transect
        # level above instead (filter only the commercial notice, not others).
        warnings.filterwarnings('ignore', message=re.escape(_COMMERCIAL_WARNING),
                                category=UserWarning)
        profiles = [
            fetch_bottom_crust1((la, lo), sediment_attenuation=sediment_attenuation,
                                basement_attenuation=basement_attenuation,
                                elastic=elastic, use_globsed=use_globsed)
            for la, lo in zip(lats, lons)
        ]
    rdl = Bottom.from_columns(profiles, ranges=np.asarray(ranges_m))
    rdl.sediment_thickness_source = (
        'globsed' if any(getattr(p, 'sediment_thickness_source', None) == 'globsed'
                         for p in profiles) else None)
    return rdl
