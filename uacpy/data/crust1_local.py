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

import copy
import hashlib
import io
import tarfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import warnings

import numpy as np

from uacpy._log import log_message
from uacpy.core._export import ExportRecord
from uacpy.core.environment import (
    BoundaryProperties, SeabedColumn, Bottom, SedimentLayer,
)
from uacpy.core.exceptions import (
    ConfigurationError, DataFetchError, FallbackWarning, ProvenanceWarning,
)
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.data import _cache
from uacpy.core.geo import (
    as_coordinate, normalize_lon, central_angle, geodesic_waypoints,
)
from uacpy.data._geo import (
    checked_max_points, checked_n_points, capped_n_points,
    checked_max_distance, checked_offset, cell_half_diagonal_km,
)
from uacpy.data.globsed_local import globsed_node
from uacpy.data._http import http_get, checked_member_size
from uacpy.data.sediment import _fill_gaps_from_nearest, _warn_filled_gaps
from uacpy.data.sources import SOURCES, DataProvenance

__all__ = ['Crust1Profile', 'download_crust1_db', 'fetch_crust1_profile', 'fetch_bottom_crust1',
           'fetch_bottom_crust1_transect']

#: Where the model is published: Laske et al.'s page at IGPP. Tried first
#: although it does not answer (below), so a restored server is preferred to a
#: mirror without an edit here.
CRUST1_URL = 'https://igppweb.ucsd.edu/~gabi/crust1/crust1.0.tar.gz'
#: Tried in order after :data:`CRUST1_URL`, which no longer serves the model:
#: every path under ``igppweb.ucsd.edu`` answers a plain Apache 403 while
#: ``/`` redirects to ``igpp.ucsd.edu``, the institute's current site. That is
#: the whole legacy document tree rather than this page in particular --
#: ``/~definitely-no-such-user/`` is 403 as well, where a live ``mod_userdir``
#: answers 404 -- and it holds for every client, curl, a browser agent, a
#: Python agent and none alike.
#:
#: A mirror is safe here because :data:`CRUST1_MD5` decides whether a download
#: is the model, not the address it came from. This one is the Internet
#: Archive's snapshot of the publisher's own tarball; ``id_`` asks it for the
#: original bytes rather than a rewritten page.
CRUST1_MIRROR_URLS = (
    'https://web.archive.org/web/20260217164003id_/'
    'https://igppweb.ucsd.edu/~gabi/crust1/crust1.0.tar.gz',
)
_FILES = ('crust1.bnds', 'crust1.vp', 'crust1.vs', 'crust1.rho')
#: MD5 of each grid as published, checked on every download before any of the
#: four is written. Measured on two independently obtained copies that agree
#: to the byte: a cache built from the IGPP page while it served, and the
#: Internet Archive's snapshot of the same tarball. What identifies CRUST1.0
#: here is therefore the data, leaving the address free to change.
CRUST1_MD5 = {
    'crust1.bnds': '2b472a8d99d1c8d2ca35ecb8b79e36e2',
    'crust1.vp': '8d2bab6fb836ab88407deed8c2084cf1',
    'crust1.vs': '34f29ce5f8d4846d0508d4e709b8191b',
    'crust1.rho': '6202a5339b98c9662caa45c4e6382c8a',
}
_NLAT, _NLON, _NLAYER = 180, 360, 9
# 9 columns per cell: water, ice, upper/middle/lower sediment,
# upper/middle/lower crystalline crust, mantle (below Moho).
_UPPER_SED, _LOW_SED = 2, 4
_UPPER_CRYST, _MID_CRYST, _LOW_CRYST = 5, 6, 7

# CRUST1.0 has no Q; nominal compressional attenuation (dB/λ) by default.
DEFAULT_SEDIMENT_ATTENUATION = 0.5
DEFAULT_BASEMENT_ATTENUATION = 0.1
# Shear loss is NOT the compressional value. Computational Ocean Acoustics
# (Jensen, Kuperman, Porter & Schmidt) Table 1.3 tabulates α_p / α_s in dB per
# wavelength for the continental shelf and slope materials: clay 0.2 / 1.0,
# silt 1.0 / 1.5, sand 0.8 / 2.5, gravel 0.6 / 1.5, moraine 0.4 / 1.0 — shear
# loss runs 1.5 to 5 times the compressional value in every unconsolidated
# sediment. 1.5 is the median of that column (and silt's and gravel's own
# value). For the crystalline basement the table gives basalt 0.1 / 0.2 and
# limestone 0.1 / 0.2, hence 0.2.
#
# COA §1.6.2 notes that α_s "can be shown to have negligible effect on bottom
# loss for low-shear-speed sediments (c_s < c_w)", so on a soft column these
# defaults barely move the answer; they matter where CRUST1.0's Vs exceeds the
# water speed, which is most of the crystalline crust and the stiffer
# sediments, and there a sediment 0.5 / basement 0.1 would understate the
# loss by factors of 3 and 2.
DEFAULT_SEDIMENT_SHEAR_ATTENUATION = 1.5
DEFAULT_BASEMENT_SHEAR_ATTENUATION = 0.2

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
    warnings.warn(_COMMERCIAL_WARNING, ProvenanceWarning,
                  skip_file_prefixes=USER_FRAME_SKIP)

# Below this total sediment thickness (m) the seabed is treated as bare rock —
# a thinner column (e.g. at a spreading-ridge crest, or after a near-zero GlobSed
# rescale) is acoustically negligible and would only yield a sub-resolution
# sediment medium downstream.
_MIN_SEDIMENT_M = 1.0


def download_crust1_db(cache_dir=None, *, url: Optional[str] = None,
                       timeout=180.0, verbose=False):
    """Download + extract the CRUST1.0 grids (``crust1.bnds/vp/vs/rho``).

    Writes the four ASCII grids into ``<cache>/crust1/`` and returns that dir.

    Every grid is checked against :data:`CRUST1_MD5` before it is written, so
    what decides whether a download is CRUST1.0 is the data, not the host it
    came from. A mismatch raises and leaves the cache untouched.

    ``url`` fetches one address and only that one. Given nothing, the
    published address :data:`CRUST1_URL` is tried first and
    :data:`CRUST1_MIRROR_URLS` after it, so a restored publisher is preferred
    to a mirror (those constants carry the state of each).

    A copy of an existing ``<cache>/crust1/`` directory also works:
    :func:`uacpy.data._cache.require` reads what is in the cache without
    asking where it came from.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Directory to write into; ``None`` is the dataset's own directory under
        the cache root (:func:`dataset_root`).
    url : str, optional
        The one address to fetch; ``None`` tries :data:`CRUST1_URL`, then
        :data:`CRUST1_MIRROR_URLS`.
    timeout : float, optional
        Network timeout in seconds. Default 180.
    verbose : bool or str, optional
        Logging gate passed through to ``log_message``.
    """
    dest = _cache.prepare_download(
        'crust1', "downloading CRUST1.0 (Laske et al. 2013, ~1 MB)",
        cache_dir=cache_dir, verbose=verbose)
    addresses = (url,) if url else (CRUST1_URL,) + CRUST1_MIRROR_URLS
    blob, failures = None, []
    for address in addresses:
        try:
            blob = http_get(address, timeout=timeout, verbose=verbose,
                            source='crust1')
            break
        except DataFetchError as exc:
            failures.append(f"{address}: {exc}")
            log_message('crust1', f"{address} did not answer; trying the next"
                        if address is not addresses[-1] else f"{address} did "
                        f"not answer", verbose=verbose, level='warning')
    if blob is None:
        raise DataFetchError(
            "CRUST1.0 could not be fetched from any known address.\n  "
            + "\n  ".join(failures),
            remediation="Pass url= for a mirror, or copy an existing "
                        "<cache>/crust1/ directory (crust1.bnds, .vp, .vs, "
                        ".rho) into this cache; the reader does not care "
                        "where the files came from.",
        )
    tf = tarfile.open(fileobj=io.BytesIO(blob))
    # Digest every grid BEFORE any of them is written: a half-written cache of
    # verified files beside unverified ones is the state that would be hardest
    # to notice later.
    bodies = {}
    for member in tf.getmembers():
        name = Path(member.name).name
        if name in _FILES and not name.startswith('._'):
            checked_member_size(member.size, name)
            bodies[name] = tf.extractfile(member).read()
    missing = [name for name in _FILES if name not in bodies]
    if missing:
        raise DataFetchError(
            f"CRUST1.0 archive did not contain {', '.join(missing)}.",
            remediation="Retry; the upstream archive layout may have changed.",
        )
    for name, body in bodies.items():
        digest = hashlib.md5(body).hexdigest()
        if digest != CRUST1_MD5[name]:
            raise DataFetchError(
                f"{name} from {addresses[0] if not url else url} is not the "
                f"published grid: md5 {digest}, expected "
                f"{CRUST1_MD5[name]}.",
                remediation="The archive at that address is not CRUST1.0 as "
                            "this package was validated against it. Use "
                            "another mirror; do not relax the digest to make "
                            "a download pass.",
            )
    for name, body in bodies.items():
        with _cache.atomic_write(dest / name) as part:
            part.write_bytes(body)
    _cache.invalidate_grids()
    log_message('crust1', f"CRUST1.0 grids cached → {dest}", verbose=verbose)
    return dest


def _build_model():
    """Read the four CRUST1.0 grids into ``(64800, 9)`` arrays."""
    grids = {}
    for fname in _FILES:
        path = _cache.require('crust1', fname)
        with _cache.reading('crust1', path):
            arr = np.loadtxt(path)
        if arr.shape != (_NLAT * _NLON, _NLAYER):
            raise DataFetchError(
                f"CRUST1.0 file {fname} has unexpected shape {arr.shape}.",
                remediation="Re-run ./install.sh --data crust1.",
            )
        grids[fname.split('.')[-1]] = arr      # 'bnds' / 'vp' / 'vs' / 'rho'
    return grids


@_cache.per_root_memo
def _model():
    """Load (or reuse) the four CRUST1.0 grids as ``(64800, 9)`` arrays.

    Built through :func:`uacpy.data._cache.per_root_memo`, so threads racing a cold
    memo run np.loadtxt once between them rather than once each.
    """
    return _build_model()


def _cell(lat, lon):
    """``(row, col)`` of the 1° cell holding ``(lat, lon)``."""
    return (int(np.clip(89 - np.floor(lat), 0, _NLAT - 1)),
            int(np.clip(np.floor(normalize_lon(lon)) + 180, 0, _NLON - 1)))


def _crust1_provenance(lat, lon, *, who, max_distance_km=None):
    """The ``'crust1'`` record of the cell a ``(lat, lon)`` read lands on
    (its centre as ``data_point``), through the offset rule."""
    row, col = _cell(lat, lon)
    prov = DataProvenance(source=SOURCES['crust1'],
                          data_point=(89.5 - row, col - 179.5),
                          requested_point=(float(lat), float(lon)),
                          point_kind='cell', cell_size_deg=1.0,
                          from_neighbour_cell=False)
    return checked_offset(prov, who=who, max_distance_km=max_distance_km,
                          warn_km=cell_half_diagonal_km(89.5 - row, 1.0, 1.0))


def _column(lat, lon):
    """The 9-layer (bnds, vp, vs, rho) vectors at a point (nearest 1° cell).

    ``bnds[i]`` is the **elevation of the top of layer i in km, positive up** —
    so it is negative below sea level, and a layer thickness is the *downward*
    difference ``bnds[i] - bnds[i+1]``. Vp/Vs are km/s and rho is g/cm³.
    """
    # Grid: row 0 = 89.5°N → row 179 = −89.5°N; col 0 = −179.5° → col 359 = 179.5°.
    row, col = _cell(lat, lon)
    idx = row * _NLON + col
    g = _model()
    return g['bnds'][idx], g['vp'][idx], g['vs'][idx], g['rho'][idx]


def _layer(i, bnds, vp, vs, rho, atten, shear_atten, elastic):
    """A :class:`SedimentLayer` for CRUST1.0 layer ``i`` (km/s → m/s, km → m)."""
    cs = vs[i] * 1000.0 if elastic else 0.0
    return SedimentLayer(
        thickness=(bnds[i] - bnds[i + 1]) * 1000.0,
        sound_speed=vp[i] * 1000.0, density=float(rho[i]), attenuation=atten,
        shear_speed=cs, shear_attenuation=(shear_atten if cs > 0 else 0.0),
    )


def _halfspace(i, vp, vs, rho, atten, shear_atten, elastic):
    cs = vs[i] * 1000.0 if elastic else 0.0
    return BoundaryProperties(
        acoustic_type='half-space', sound_speed=vp[i] * 1000.0,
        density=float(rho[i]), attenuation=atten,
        shear_speed=cs, shear_attenuation=(shear_atten if cs > 0 else 0.0),
    )


def _sediment_layer_indices(bnds):
    """Indices of the CRUST1.0 sediment layers with non-zero thickness.

    CRUST1.0 quantises every boundary to 0.01 km, so the thinnest layer it can
    express is 10 m and any sub-metre threshold separates "absent" from
    "present" identically — this one is in km, the one in
    fetch_crust1_profile is in m."""
    return [i for i in range(_UPPER_SED, _LOW_SED + 1)
            if bnds[i] - bnds[i + 1] > 1e-3]


def _layered_from_column(bnds, vp, vs, rho, *, sediment_attenuation,
                         basement_attenuation, elastic, sediment_thickness,
                         sediment_shear_attenuation=(
                             DEFAULT_SEDIMENT_SHEAR_ATTENUATION),
                         basement_shear_attenuation=(
                             DEFAULT_BASEMENT_SHEAR_ATTENUATION),
                         roughness=0.0):
    """Build a :class:`SeabedColumn`: sediment stack over crystalline basement.

    ``roughness`` lands on the first layer, whose top interface is the seafloor
    (:class:`~uacpy.core.boundary.SedimentLayer`), whichever of the two column
    shapes below is built.

    A ``sediment_thickness`` can only **rescale** sediment layers the column
    already has: on a zero-sediment column there are no sediment Vp/Vs/ρ to
    build a layer from, so the thickness is discarded and the bare-rock shape
    is emitted regardless — the caller (``_bottom_at_point``) owns saying so."""
    sed = _sediment_layer_indices(bnds)
    # Effective sediment thickness (the GlobSed override when given — a real
    # 0.0 means bare basement, not "no data" — else the CRUST1.0 column).
    # Below ``_MIN_SEDIMENT_M`` the column is negligible and a real layer would
    # only produce a sub-resolution medium, so emit bare rock.
    if sed:
        eff_sed = (sum((bnds[i] - bnds[i + 1]) * 1000.0 for i in sed)
                   if sediment_thickness is None else sediment_thickness)
    else:
        eff_sed = 0.0
    if sed and eff_sed >= _MIN_SEDIMENT_M:
        layers = [_layer(i, bnds, vp, vs, rho, sediment_attenuation,
                         sediment_shear_attenuation, elastic)
                  for i in sed]
        if sediment_thickness is not None:
            total = sum(layer.thickness for layer in layers)
            if total > 0:
                scale = sediment_thickness / total
                for layer in layers:
                    layer.thickness *= scale
        halfspace = _halfspace(_UPPER_CRYST, vp, vs, rho, basement_attenuation,
                               basement_shear_attenuation, elastic)
    else:           # bare rock / negligible sediment: crust over crust
        layers = [_layer(_UPPER_CRYST, bnds, vp, vs, rho, basement_attenuation,
                         basement_shear_attenuation, elastic)]
        halfspace = _halfspace(_MID_CRYST, vp, vs, rho, basement_attenuation,
                               basement_shear_attenuation, elastic)
    layers[0].roughness = float(roughness)
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


@dataclass(frozen=True, eq=False)
class Crust1Profile(ExportRecord):
    """The CRUST1.0 column at a point, as :func:`fetch_crust1_profile`
    returns it: the sediment and crystalline layers present (thicker than
    1 mm), top down.

    Attributes
    ----------
    water_depth : float
        The base of the water layer (m), negated elevation — so on land or
        under an ice sheet, where CRUST1.0 puts no water, it is **negative**
        and equals minus the ground/ice surface elevation.
    sediment_thickness : float
        The summed thickness of the sediment layers (m).
    layer_names : tuple of str
        CRUST1.0's layer names (``'upper_sed'`` ... ``'low_cryst'``).
    thickness, sound_speed, shear_speed, density : ndarray
        Per layer: m, m/s, m/s, g/cm³.
    provenance : DataProvenance
        The ``'crust1'`` record with the requested point.
    """

    water_depth: float
    sediment_thickness: float
    layer_names: Tuple[str, ...]
    thickness: np.ndarray
    sound_speed: np.ndarray
    shear_speed: np.ndarray
    density: np.ndarray
    provenance: DataProvenance

    _REPR_UNITS = {'water_depth': 'm', 'sediment_thickness': 'm',
                   'thickness': 'm', 'sound_speed': 'm/s',
                   'shear_speed': 'm/s', 'density': 'g/cm³'}

    _ARRAY_FIELDS = ('thickness', 'sound_speed', 'shear_speed', 'density')

    def _table(self):
        """One row per layer: ``layer`` and the four per-layer columns."""
        return {'layer': list(self.layer_names),
                **{name: np.asarray(getattr(self, name))
                   for name in self._ARRAY_FIELDS}}


def fetch_crust1_profile(point, *, max_distance_km=None) -> Crust1Profile:
    """Inspect the CRUST1.0 column at a point.

    Returns a :class:`Crust1Profile`: the water depth, the sediment
    thickness, and per layer its name, thickness (m), Vp, Vs (m/s) and
    density (g/cm³), with the ``'crust1'`` provenance; ``to_dataframe()``
    gives one row per layer.

    Parameters
    ----------
    point : (lat, lon)
        Site coordinates in decimal degrees.
    max_distance_km : float, optional
        Refuse a cell whose centre stands farther than this (km) from
        ``point`` (the offset rule); ``None`` (default) sets no limit.
    """
    lat, lon = as_coordinate(point)
    prov = _crust1_provenance(lat, lon, who='fetch_crust1_profile',
                              max_distance_km=checked_max_distance(
                                  max_distance_km, 'fetch_crust1_profile'))
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
    return Crust1Profile(
        water_depth=float(-bnds[1] * 1000.0),
        sediment_thickness=float(sed_thk),
        layer_names=tuple(layer['name'] for layer in layers),
        thickness=np.array([layer['thickness_m'] for layer in layers]),
        sound_speed=np.array([layer['vp'] for layer in layers]),
        shear_speed=np.array([layer['vs'] for layer in layers]),
        density=np.array([layer['rho'] for layer in layers]),
        provenance=prov)


def fetch_bottom_crust1(point, *, roughness=0.0,
                        sediment_attenuation=DEFAULT_SEDIMENT_ATTENUATION,
                        basement_attenuation=DEFAULT_BASEMENT_ATTENUATION,
                        sediment_shear_attenuation=(
                            DEFAULT_SEDIMENT_SHEAR_ATTENUATION),
                        basement_shear_attenuation=(
                            DEFAULT_BASEMENT_SHEAR_ATTENUATION),
                        elastic=True, sediment_thickness=None, use_globsed=True,
                        water_sound_speed=None, max_distance_km=None,
                        timeout=None, verbose=False):
    """Layered **elastic** bottom from CRUST1.0 at a ``(lat, lon)`` point.

    The CRUST1.0 sediment layers over the crystalline-crust half-space, with
    ``Vs`` retained (``elastic=True``). ``sediment_attenuation`` /
    ``basement_attenuation`` set the (nominal) dB/λ compressional losses
    CRUST1.0 lacks, and ``sediment_shear_attenuation`` /
    ``basement_shear_attenuation`` the shear ones. Shear loss is not the
    compressional value: Computational Ocean Acoustics Table 1.3 puts it at
    1.5 to 5 times the compressional figure in unconsolidated sediments (clay
    0.2/1.0, silt 1.0/1.5, sand 0.8/2.5, gravel 0.6/1.5 dB/λ), hence the
    defaults 1.5 (sediment) and 0.2 (basalt/limestone basement). COA §1.6.2
    notes shear attenuation has little effect on bottom loss while the shear
    speed stays below the water speed, so on a soft column the choice barely
    moves the answer.

    The coarse 1° CRUST1.0 sediment column is rescaled to a higher-resolution
    **GlobSed** total thickness by default (``use_globsed=True``); pass an
    explicit ``sediment_thickness`` (m) to override, or ``use_globsed=False`` to
    keep CRUST1.0's own column. Where GlobSed is not cached or has no value at
    the point, CRUST1.0's native thickness is kept; a genuine GlobSed 0.0
    (bare basement) is honoured and yields the bare-rock column. The returned
    bottom carries ``.sediment_thickness_source`` (``'globsed'``,
    ``'globsed-ignored'`` or ``None``) recording which was used.

    The rescaling is **asymmetric**: a thickness can shrink or stretch
    sediment layers CRUST1.0 already has, but on a column with zero sediment
    layers (4.4% of ocean cells) CRUST1.0 carries no sediment Vp/Vs/density,
    so there is nothing to build a layer from — a positive GlobSed or explicit
    ``sediment_thickness`` is then discarded with a ``FallbackWarning``, the
    bottom is bare rock, and the stamp reads ``'globsed-ignored'`` (GlobSed
    consulted but unusable) or ``None`` (explicit value).

    ``timeout`` is ignored (offline) for signature parity with the
    network bottom fetchers; ``water_sound_speed`` is likewise accepted and
    ignored (CRUST1.0 yields absolute Vp/Vs/ρ, not water-referenced ratios).

    ``roughness`` is the RMS roughness (m) of the seafloor interface — the top
    of the first returned layer, which is where
    :class:`~uacpy.core.boundary.SedimentLayer` puts the water/seabed interface.
    CRUST1.0 tabulates no roughness, so the default 0.0 is a smooth seafloor;
    give a site value to model interface scattering.

    Raises :class:`~uacpy.core.exceptions.DataFetchError` on a cell with no
    water layer (land or grounded ice, where :func:`fetch_crust1_profile`
    reports ``water_depth <= 0``), as the bathymetry fetchers do on land.
    ``max_distance_km`` refuses a cell (or GlobSed node) standing farther
    than that (km) from ``point``; ``None`` (default) sets no limit.
    """
    _warn_non_commercial()
    return _bottom_at_point(
        point, sediment_thickness=sediment_thickness, use_globsed=use_globsed,
        verbose=verbose, max_distance_km=checked_max_distance(
            max_distance_km, 'fetch_bottom_crust1'), layer_kw=dict(
            roughness=roughness, elastic=elastic,
            sediment_attenuation=sediment_attenuation,
            basement_attenuation=basement_attenuation,
            sediment_shear_attenuation=sediment_shear_attenuation,
            basement_shear_attenuation=basement_shear_attenuation))


class _NoWaterLayer(DataFetchError):
    """A CRUST1.0 cell with no water layer (land or grounded ice): the one
    per-waypoint condition a transect fills from its neighbour. Every other
    failure propagates."""


def _bottom_at_point(point, *, layer_kw, sediment_thickness=None,
                     use_globsed=True, verbose=False, max_distance_km=None,
                     who='fetch_bottom_crust1'):
    """One CRUST1.0 column, without the commercial notice.

    ``layer_kw`` carries the per-layer keywords of :func:`_layered_from_column`
    (roughness, the four attenuations, ``elastic``), built once per public
    fetch and applied unchanged at every waypoint of a transect.

    The notice belongs to a whole fetch, not to each point of one, so the
    transect emits it once and builds its waypoints through here. Suppressing
    it instead by wrapping the loop in ``warnings.catch_warnings()`` would edit
    the process-global filter list, and so also swallow an identical notice
    another thread raised while the loop was open.
    """
    lat, lon = as_coordinate(point)
    bnds, vp, vs, rho = _column(lat, lon)
    # ``bnds[1]`` is the base of the water layer (km, positive up); a cell
    # with no water — land, or an ice sheet resting on the ground — has it
    # at or above sea level, and carries no seabed to build a bottom from.
    water_depth_m = -bnds[1] * 1000.0
    if water_depth_m <= 0.0:
        raise _NoWaterLayer(
            f"CRUST1.0 at ({lat:.2f}, {lon:.2f}) has no water layer: the "
            f"cell is land or grounded ice (water_depth_m = "
            f"{water_depth_m:+.0f}), so there is no seabed there.",
            remediation="Pick an offshore point, or supply a bottom "
                        "directly.",
        )
    globsed_applied = False
    if sediment_thickness is None and use_globsed:
        sediment_thickness = _globsed_thickness(point, verbose=verbose)
        globsed_applied = sediment_thickness is not None
    # A thickness can only rescale sediment layers CRUST1.0 already has; on a
    # zero-sediment column (4.4% of ocean cells) there are no sediment Vp/Vs/ρ
    # to build a layer from, so a positive requested thickness is discarded and
    # the bottom is bare rock. Below _MIN_SEDIMENT_M the outcome is bare rock
    # on any column, so nothing is discarded and the notice stays quiet.
    thickness_discarded = (
        not _sediment_layer_indices(bnds) and sediment_thickness is not None
        and sediment_thickness >= _MIN_SEDIMENT_M)
    if thickness_discarded:
        origin = ("the GlobSed total thickness" if globsed_applied
                  else "the requested sediment_thickness")
        warnings.warn(
            f"CRUST1.0 at ({lat:.2f}, {lon:.2f}) has no sediment layers, so "
            f"there are no sediment Vp/Vs/density values to build a column "
            f"from: {origin} of {sediment_thickness:g} m is discarded and the "
            f"bottom is bare crystalline rock. If the sediment column matters "
            f"here, build the layers from another source (e.g. "
            f"uacpy.data.sediment's grain-size backends).",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
    bottom = _layered_from_column(
        bnds, vp, vs, rho, sediment_thickness=sediment_thickness, **layer_kw)
    # 'globsed' only when the GlobSed value shaped the column; a consulted but
    # discarded value is stamped 'globsed-ignored' so provenance never lists a
    # dataset the result does not contain.
    bottom.sediment_thickness_source = (
        'globsed-ignored' if (globsed_applied and thickness_discarded)
        else 'globsed' if globsed_applied else None)
    ids = (('crust1', 'globsed')
           if bottom.sediment_thickness_source == 'globsed' else ('crust1',))
    bottom.halfspace.data_sources = (
        _crust1_provenance(lat, lon, who=who, max_distance_km=max_distance_km),
        *((globsed_node(lat, lon, who=who, max_distance_km=max_distance_km),)
          if 'globsed' in ids else ()))
    log_message('crust1', f"CRUST1.0 at {lat:.2f}, {lon:.2f} → {bottom!r}",
                verbose=verbose)
    return bottom


def fetch_bottom_crust1_transect(start, end, *, n_points=6, max_points=None,
                                 roughness=0.0,
                                 sediment_attenuation=DEFAULT_SEDIMENT_ATTENUATION,
                                 basement_attenuation=DEFAULT_BASEMENT_ATTENUATION,
                                 sediment_shear_attenuation=(
                                     DEFAULT_SEDIMENT_SHEAR_ATTENUATION),
                                 basement_shear_attenuation=(
                                     DEFAULT_BASEMENT_SHEAR_ATTENUATION),
                                 elastic=True, use_globsed=True,
                                 water_sound_speed=None, max_distance_km=None,
                                 timeout=None, verbose=False):
    """Range-dependent layered bottom from CRUST1.0 along ``start`` → ``end``.

    Each profile rescales its CRUST1.0 sediment column to the local GlobSed
    thickness by default (``use_globsed``); the result carries
    ``.sediment_thickness_source`` (``'globsed'`` if any waypoint used it).
    ``n_points='auto'`` targets roughly one waypoint per degree of arc —
    CRUST1.0's native 1° resolution — unlike the probe-and-collapse ``'auto'``
    of the sample-database bottom fetchers.
    ``max_points`` caps the waypoint count (signature parity with the other
    transect bottom fetchers); CRUST1.0 is a cached 1° grid, so the sole effect
    is clamping ``n_points``.

    ``timeout``/``verbose`` are accepted (and ignored — this backend is
    offline) for signature parity with the network bottom fetchers;
    ``water_sound_speed`` is likewise accepted and ignored (CRUST1.0 yields
    absolute Vp/Vs/ρ, not water-referenced ratios).

    ``roughness`` and the four attenuation keywords are as in
    :func:`fetch_bottom_crust1`, applied at every waypoint. A waypoint whose
    1° cell has no water layer (land or grounded ice — a coastal point GEBCO
    puts in water can sit in one) takes the column of the nearest waypoint
    that has one, with a ``FallbackWarning``, as every other bottom transect
    does; the call raises only when no waypoint has water.
    ``max_distance_km`` applies the offset rule at every waypoint, as in
    :func:`fetch_bottom_crust1`.
    """
    limit = checked_max_distance(max_distance_km, 'fetch_bottom_crust1_transect')
    n_points = checked_n_points(n_points, 'fetch_bottom_crust1_transect',
                                allow_auto=True)
    # 'auto': CRUST1.0 is a 1-degree cached grid, so target roughly one
    # waypoint per degree of arc, clamped like the siblings.
    if n_points == 'auto':
        n_points = max(2, int(np.degrees(central_angle(start, end))) + 1)
    if max_points is not None:
        max_points = checked_max_points(max_points,
                                        'fetch_bottom_crust1_transect')
    n_points = capped_n_points(n_points, max_points,
                               'fetch_bottom_crust1_transect')
    _warn_non_commercial()
    lats, lons, ranges_m = geodesic_waypoints(start, end, n_points)
    # The notice is emitted once above, for the transect as a whole; the
    # waypoints go through the builder that does not raise it, so no filter
    # window has to be opened over the loop to keep it quiet.
    layer_kw = dict(
        roughness=roughness, elastic=elastic,
        sediment_attenuation=sediment_attenuation,
        basement_attenuation=basement_attenuation,
        sediment_shear_attenuation=sediment_shear_attenuation,
        basement_shear_attenuation=basement_shear_attenuation)
    # Read the grids once up front, so an unreadable cache raises its own
    # error here rather than as a "no water" gap at every waypoint.
    _model()
    profiles = []
    for la, lo in zip(lats, lons):
        try:
            profiles.append(_bottom_at_point(
                (la, lo), layer_kw=layer_kw, use_globsed=use_globsed,
                max_distance_km=limit, who='fetch_bottom_crust1_transect'))
        except _NoWaterLayer:
            profiles.append(None)       # no water layer; filled below
    if all(p is None for p in profiles):
        raise DataFetchError(
            "CRUST1.0 has no water layer anywhere along the transect: every "
            "waypoint's 1° cell is land or grounded ice.",
            remediation="Use an offshore transect, or pick another "
                        "bottom_sources.",
        )
    profiles, filled = _fill_gaps_from_nearest(profiles, ranges_m)
    _warn_filled_gaps('CRUST1.0', filled, len(profiles))
    # A filled waypoint shares its neighbour's column object; give each range
    # its own copy so editing one column cannot change another.
    seen = set()
    for i, p in enumerate(profiles):
        if id(p) in seen:
            profiles[i] = copy.deepcopy(p)
        seen.add(id(p))
    rdl = Bottom.from_columns(profiles, ranges=np.asarray(ranges_m))
    rdl.sediment_thickness_source = (
        'globsed' if any(getattr(p, 'sediment_thickness_source', None) == 'globsed'
                         for p in profiles) else None)
    return rdl
