"""Local seafloor-sediment samples (DECK41 + NGDC grain-size) → bottom.

The offline, *public-domain*, **global** sediment backend (commercial-clean).
It reads up to two normalized CSVs from the cache:

* ``grainsize.csv`` — NGDC Seafloor Sediment Grain-Size DB (quantitative mean ϕ).
  ``install.sh --data sediment`` downloads and normalizes this one
  (:func:`download_sediment_db`).
* ``deck41.csv``    — DECK41 surficial descriptions (dominant lithology text).
  Optional, and no installer step produces it: drop it in by hand
  (``latitude, longitude, lithology``) to widen coverage where grain-size
  samples are sparse.

A nearest-neighbour lookup (great-circle, via a unit-sphere KD-tree) returns the
closest sample. The two files hold **separate** trees, compared rather than
merged: a grain-size sample is preferred over a lithology description at
comparable range, where its precision is the only thing separating them, and
the nearer sample wins once they are far enough apart to be describing
different seabed. Because these are sparse point samples (not a continuous
map), the nearest sample can be far offshore — a ``max_distance_km`` guard
raises when no sample is close enough.
"""

import csv
import dataclasses
import io
import tarfile
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from uacpy._log import log_message
from uacpy.core.exceptions import DataFetchError
from uacpy.data import _cache
from uacpy.data._geo import as_coordinate, normalize_lon, EARTH_RADIUS_KM
from uacpy.data._http import http_get, checked_member_size
from uacpy.data.sources import SOURCES, DataProvenance
from uacpy.data.sediment import (
    bottom_from_class, bottom_from_grain_size, range_dependent_bottom_along,
    water_sound_speed_at,
)

__all__ = ['download_sediment_db', 'fetch_sediment_sample', 'fetch_bottom_local',
           'fetch_bottom_local_transect']

DEFAULT_MAX_DISTANCE_KM = 250.0

# NCEI Seafloor Sediment Grain-Size Database (G00127, public domain): a ~3 MB
# tarball of TSV tables. We join the per-sample lat/lon with a weighted-mean ϕ
# from the grain-size distribution to build the normalized grainsize.csv.
GRAINSIZE_TARBALL_URL = (
    'https://www.ngdc.noaa.gov/mgg/geology/data/g00127/g00127.tar.gz')
# Clamp the open-ended catch-all ϕ bins to the real Wentworth range before
# taking bin centres (the raw data uses sentinels like -50 / 51).
_PHI_CLAMP = (-5.0, 13.0)

# DECK41 dominant-lithology terms → representative Wentworth grain size (ϕ).
_DECK41_LITHOLOGY_TO_PHI = {
    'gravel': -2.0, 'sand': 1.5, 'silt': 5.5, 'clay': 9.0, 'mud': 7.5,
    'ooze': 7.5, 'calcareous ooze': 7.5, 'siliceous ooze': 8.5,
    'diatom ooze': 9.0, 'radiolarian ooze': 8.0, 'foraminiferal ooze': 6.5,
    # 'rock' has no honest phi (the Wentworth scale describes loose
    # sediment): it carries the sentinel below through the phi index and
    # fetch_sediment_sample translates it to the 'limestone' material —
    # the same route the EMODnet substrate path uses for hard substrata
    # (seabed.py _FOLK5_TO_BOTTOM class 5).
    'rock': -99.0, 'gravel and coarser': -3.0,
}

# Candidate column names (lower-cased) for the tolerant CSV reader.
_LAT_COLS = ('latitude', 'lat', 'y')
_LON_COLS = ('longitude', 'lon', 'long', 'x')
_PHI_COLS = ('phi', 'mean_phi', 'mean', 'mean_grain_size', 'grain_size_phi')
_LITH_COLS = ('lithology', 'dominant_lithology', 'dominant', 'description')

_SAMPLES = {}   # cache_root -> (grainsize_index, lithology_index), either None
_cache.register_cache(_SAMPLES.clear)


def _tar_rows(tf, suffix):
    """All TSV rows from the member of ``tf`` whose name ends with ``suffix``."""
    try:
        member = next(m for m in tf.getmembers()
                      if Path(m.name).name.endswith(suffix))
    except StopIteration as exc:
        raise DataFetchError(
            f"NCEI grain-size archive did not contain a *{suffix} member.",
            remediation="Retry; the upstream archive layout may have changed.",
        ) from exc
    checked_member_size(member.size, member.name)
    stream = io.TextIOWrapper(tf.extractfile(member), encoding='latin-1')
    return list(csv.DictReader(stream, delimiter='\t'))


def download_sediment_db(cache_dir=None, *, timeout=180.0, verbose=False):
    """Download + normalize the NCEI grain-size database into ``grainsize.csv``.

    Fetches the public-domain G00127 tarball, joins each sample's lat/lon with a
    weighted-mean ϕ from its (shallowest-interval) grain-size distribution, and
    writes ``<cache>/sediment/grainsize.csv`` (``latitude, longitude, mean_phi``)
    — the file the local sediment backend reads. Returns the written path.

    ``cache_dir`` defaults to the offline cache's ``sediment`` directory.
    """
    dest = _cache.prepare_download(
        'sediment', "downloading NCEI grain-size DB (G00127, ~3 MB)",
        cache_dir=cache_dir, verbose=verbose)
    blob = http_get(GRAINSIZE_TARBALL_URL, timeout=timeout, verbose=verbose,
                    source='sediment')
    tf = tarfile.open(fileobj=io.BytesIO(blob))

    loc = {}
    for r in _tar_rows(tf, 'sample.txt'):
        try:
            loc[(r['mggid'], r['sample'])] = (float(r['lat']), float(r['lon']))
        except (KeyError, ValueError):
            continue

    lo_c, hi_c = _PHI_CLAMP
    best = {}     # (mggid, sample) -> [min_interval, sum_w, sum_w_phi]
    for r in _tar_rows(tf, 'phi.txt'):
        key = (r.get('mggid'), r.get('sample'))
        try:
            interval = int(r['interval'] or 0)
            lo = max(float(r['lower_phi_limit']), lo_c)
            hi = min(float(r['upper_phi_limit']), hi_c)
            weight = float(r['weight_percent'])
        except (KeyError, ValueError):
            continue
        if hi < lo or weight <= 0.0:
            continue
        centre = 0.5 * (lo + hi)
        if key not in best or interval < best[key][0]:
            # Surficial sediment only: a shallower interval restarts the sum.
            best[key] = [interval, 0.0, 0.0]
        if interval == best[key][0]:
            best[key][1] += weight
            best[key][2] += centre * weight

    out = dest / 'grainsize.csv'
    n = 0
    with _cache.atomic_write(out) as part:
        with open(part, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(['latitude', 'longitude', 'mean_phi'])
            for key, (_interval, sum_w, sum_w_phi) in best.items():
                if key in loc and sum_w > 0.0:
                    lat, lon = loc[key]
                    writer.writerow([lat, lon, round(sum_w_phi / sum_w, 3)])
                    n += 1
        # Inside the staging block, so a header-only file is discarded rather
        # than left in the cache for the reader to accept.
        if n == 0:
            raise DataFetchError(
                "NCEI grain-size DB parsed to zero usable samples.",
                remediation="Retry; the upstream tarball layout may have changed.",
            )
    _SAMPLES.clear()                          # force rebuild of the KD-tree
    log_message('sediment', f"grain-size DB normalized: {n} samples → {out}",
                verbose=verbose)
    return out


def _pick(header, candidates):
    lut = {h.lower().strip(): h for h in header}
    for c in candidates:
        if c in lut:
            return lut[c]
    return None


def _unit_vectors(lat, lon):
    """``(lat, lon)`` degrees → unit vectors on the sphere, the KD-tree's space.

    The tree indexes 3-D points rather than the raw degree pair for two reasons:
    a degree-space tree has a false seam at ±180° (and would depend on which
    longitude convention each CSV used), and a degree of longitude is not a
    degree of distance away from the equator. Chord length between unit vectors
    is ``2·sin(θ/2)`` in the central angle θ, which increases monotonically over
    ``θ ∈ [0, π]`` — so nearest-in-chord is exactly nearest-in-great-circle, and
    the chord the query returns converts back to an arc length in
    :func:`fetch_sediment_sample`.
    """
    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(la) * np.cos(lo),
                            np.cos(la) * np.sin(lo),
                            np.sin(la)])


def _read_csv(path, value_cols, transform):
    """Return ``(lats, lons, phis)`` from a normalized sediment CSV.

    ``transform`` maps a raw cell (ϕ string or lithology text) to a float ϕ, or
    ``None`` to skip the row. A row that is empty, short, or carries an
    unreadable value is skipped; a row whose coordinates are not numbers raises
    :class:`DataFetchError` naming the line — these files are hand-writable
    (``deck41.csv``), so a typo has to be reported rather than swallowed.

    Both raises are :class:`DataFetchError` because that is the data layer's
    exception whatever the reason: the caller asks *can I get a sediment value
    here?* and acts the same way on every no. Neither file is one the user
    supplied — ``./install.sh`` wrote both — so neither unreadable cache is a
    ``ConfigurationError``; ``gebco_local`` types the identical failure (a
    cached dataset missing an expected variable) the same way.
    """
    lats, lons, phis = [], [], []
    with open(path, newline='') as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return lats, lons, phis
        lat_c = _pick(header, _LAT_COLS)
        lon_c = _pick(header, _LON_COLS)
        val_c = _pick(header, value_cols)
        if lat_c is None or lon_c is None or val_c is None:
            raise DataFetchError(
                f"Sediment file {path.name} is missing expected columns "
                f"(need lat, lon and one of {value_cols}); got {header}.",
                remediation="Re-run ./install.sh --data sediment to refresh it.",
            )
        idx = {h: i for i, h in enumerate(header)}
        li, oi, vi = idx[lat_c], idx[lon_c], idx[val_c]
        for line_no, row in enumerate(reader, start=2):     # 1 is the header
            if not row or len(row) <= max(li, oi, vi):
                continue
            phi = transform(row[vi])
            if phi is None:
                continue
            # Parse both coordinates before appending either: appending as they
            # convert leaves lats one longer than lons when the second cell is
            # malformed, and the three lists then reach np.asarray/cKDTree at
            # different lengths — an untyped "operands could not be broadcast"
            # far from the row that caused it.
            try:
                lat = float(row[li])
                lon = normalize_lon(float(row[oi]))
            except ValueError as exc:
                raise DataFetchError(
                    f"Sediment file {path.name} line {line_no} has a "
                    f"non-numeric coordinate ({lat_c}={row[li]!r}, "
                    f"{lon_c}={row[oi]!r}).",
                    remediation="Fix or delete that row; for grainsize.csv, "
                                "re-run ./install.sh --data sediment to "
                                "regenerate it.",
                ) from exc
            lats.append(lat)
            lons.append(lon)
            phis.append(phi)
    return lats, lons, phis


def _phi_from_float(cell):
    try:
        return float(cell)
    except ValueError:
        return None


def _phi_from_lithology(cell):
    return _DECK41_LITHOLOGY_TO_PHI.get(cell.lower().strip())


def _index(path, value_cols, transform):
    """``(tree, phis, lats, lons)`` for one CSV, or ``None`` if it has no rows."""
    if not path.exists():
        return None
    lats, lons, phis = _read_csv(path, value_cols, transform)
    if not phis:
        return None
    lats, lons = np.array(lats), np.array(lons)
    return cKDTree(_unit_vectors(lats, lons)), np.array(phis), lats, lons


def _samples():
    """Build (or reuse) the nearest-neighbour indices of ϕ samples.

    Returns ``(grainsize_index, lithology_index)``, either of which may be
    ``None``. The two files stay in **separate** KD-trees because they are not
    peers: ``grainsize.csv`` carries a measured mean ϕ, ``deck41.csv`` a
    lithology word mapped to a class-representative ϕ. Concatenated into one
    tree, a marginally nearer lithology class wins on distance alone and the
    measured sample beside it is never returned — the opposite of the
    preference this module documents.

    Built through :func:`uacpy.data._cache.memoize`, so threads racing a cold
    memo build the trees once between them rather than once each.
    """
    return _cache.memoize(_SAMPLES, str(_cache.cache_root()), _build_samples)


def _build_samples():
    """Read both sample files into their nearest-neighbour indices."""
    _cache.require('sediment')                       # raises if not installed
    dataset = _cache.dataset_root('sediment')
    result = (_index(dataset / 'grainsize.csv', _PHI_COLS, _phi_from_float),
              _index(dataset / 'deck41.csv', _LITH_COLS, _phi_from_lithology))
    if not any(result):
        raise DataFetchError(
            "No usable sediment samples found in the local cache.",
            remediation="Re-run ./install.sh --data sediment.",
        )
    return result


def _nearest(index, lat, lon):
    """``(distance_km, phi, sample_lat, sample_lon)`` from one index."""
    tree, phis, samp_lats, samp_lons = index
    chord, idx = tree.query(_unit_vectors(np.array([lat]), np.array([lon]))[0])
    # chord length on the unit sphere → great-circle distance.
    dist_km = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.clip(chord / 2.0, 0, 1))
    return (float(dist_km), float(phis[idx]),
            float(samp_lats[idx]), float(samp_lons[idx]))


#: Any stored phi at or below this is a lithology-class sentinel, not a
#: grain size ('rock' -> -99.0 above).
_PHI_CLASS_SENTINEL_MAX = -90.0

#: Separation (km) within which a grain-size sample and a lithology
#: description are treated as describing the same patch of seabed, so the
#: quantitative one wins on precision alone. Both numbers here are a judgement
#: about how fast the seabed decorrelates with distance, not a measurement:
#: they are deliberately round, and nothing published pins them. The scale they
#: are set against is the one the geoacoustic literature works at — Jensen,
#: Kuperman, Porter & Schmidt §1.6 requires "information on the variation of
#: all of these parameters with geographical position", and shelf sediment
#: provinces turn over in tens of kilometres — so a few km is inside one
#: province and a hundred is not.
_SAME_SEABED_KM = 5.0

#: How much farther than the nearest lithology description a grain-size sample
#: may sit and still be preferred, once both are past :data:`_SAME_SEABED_KM`.
#: A measured mean ϕ is worth perhaps ±0.2 against ±1 for a class-derived one,
#: which buys some distance but nowhere near an order of magnitude: past this
#: the sample is describing different seabed, and being quantitative about the
#: wrong place is worse than being approximate about the right one.
_QUANTITATIVE_REACH_FACTOR = 3.0


def _prefers_grain_size(grainsize, lithology) -> bool:
    """Whether the quantitative sample beats the lithology class here.

    Both arguments are :func:`_nearest` hits already known to be in reach, or
    ``None``. Preference is *distance-aware*, not absolute: an unconditional
    preference hands back a measurement 138.7 km away over a class 1.4 km away
    (a 101x hop) whenever a transect strays off the grain-size coverage, which
    is a larger error than merging the two trees and taking the nearest hit of
    either makes.
    """
    if lithology is None:
        return True
    return grainsize[0] <= max(_SAME_SEABED_KM,
                               _QUANTITATIVE_REACH_FACTOR * lithology[0])


def fetch_sediment_sample(point, *, max_distance_km=DEFAULT_MAX_DISTANCE_KM):
    """Nearest local sediment sample to ``point``.

    Returns ``{'phi', 'material', 'distance_km', 'latitude', 'longitude'}`` —
    the sample's own coordinates travel with it so a caller can record where
    the value actually came from. A grain-size sample carries ``phi``
    (float) and ``material=None``; a hard-substrate sample ('rock' in
    DECK41) carries ``phi=None`` and ``material='limestone'``, the same
    material preset the EMODnet substrate route uses for hard substrata.

    The two files hold separate indices and are compared, not merged: a
    quantitative grain-size sample is preferred over a lithology description
    at *comparable* range (see :func:`_prefers_grain_size` and the constants
    above it), and the nearer sample wins once they are far enough apart to be
    describing different seabed. Merging them into one tree let a marginally
    nearer class beat a better sample; preferring the sample unconditionally
    let one a hundred times farther away beat a class next door.

    Raises ``DataFetchError`` if the closest sample is farther than
    ``max_distance_km`` (sparse point data — no nearby ground truth).
    """
    lat, lon = as_coordinate(point)
    hits = [_nearest(index, lat, lon) if index is not None else None
            for index in _samples()]
    grainsize, lithology = [
        h if h is not None and (max_distance_km is None
                                or h[0] <= max_distance_km) else None
        for h in hits
    ]
    if grainsize is not None and _prefers_grain_size(grainsize, lithology):
        hit, dataset = grainsize, 'grainsize'
    else:
        hit, dataset = lithology, 'deck41'
    if hit is None:
        raise DataFetchError(
            f"Nearest sediment sample is "
            f"{min(h[0] for h in hits if h is not None):.0f} km "
            f"away (> max_distance_km={max_distance_km:.0f}).",
            remediation="Raise max_distance_km, or supply a bottom directly.",
        )
    dist_km, phi, samp_lat, samp_lon = hit
    if phi <= _PHI_CLASS_SENTINEL_MAX:
        # A lithology-class sentinel ('rock'): no grain size exists; the
        # material preset carries the geoacoustics instead.
        return {'phi': None, 'material': 'limestone', 'distance_km': dist_km,
                'latitude': samp_lat, 'longitude': samp_lon,
                'dataset': dataset}
    return {'phi': phi, 'material': None, 'distance_km': dist_km,
            'latitude': samp_lat, 'longitude': samp_lon, 'dataset': dataset}


def fetch_bottom_local(point, *, roughness=0.0, water_sound_speed=None,
                       max_distance_km=DEFAULT_MAX_DISTANCE_KM,
                       timeout=None, verbose=False):
    """Model-ready bottom from the nearest local sediment sample.

    This is the NCEI grain-size / DECK41 sample-database provider of the
    ``fetch_bottom_local`` protocol name that ``fetch_environment`` resolves
    per provider module (``bottom_sources='grainsize'``), and it is the one
    exported at package level as ``uacpy.data.fetch_bottom_local``;
    :mod:`uacpy.data.emodnet_local` exposes the same name for its EMODnet
    substrate provider.

    ``timeout``/``verbose`` are accepted (and ignored — this backend is offline)
    for signature uniformity with the network bottom fetchers.
    ``water_sound_speed`` (m/s) scales the grain-size velocity ratio to the
    in-situ near-seabed water; ``None`` uses the Hamilton reference.
    """
    lat, lon = as_coordinate(point)
    sample = fetch_sediment_sample(point, max_distance_km=max_distance_km)
    if sample['material'] is not None:
        # Hard substrata route through the material preset (~3000 m/s
        # limestone), exactly as the EMODnet substrate path does — a
        # grain-size relation cannot describe rock.
        bottom = bottom_from_class(sample['material'], roughness=roughness)
    else:
        bottom = bottom_from_grain_size(
            sample['phi'], roughness=roughness,
            water_sound_speed=water_sound_speed)
    # Point samples are sparse, so the nearest one can be far from the
    # requested position; record where it actually came from so
    # ``citations(env)`` reports the hop and ``prov.offset_km`` measures it.
    # Cite the index the sample actually came from. Stamping 'grainsize'
    # unconditionally reported a DECK41 lithology description under the NCEI
    # grain-size database's name, licence and DOI (10.7289/V5G44N6W) — a
    # citation for a dataset the value never touched.
    prov = DataProvenance(
        source=SOURCES[sample.get('dataset', 'grainsize')],
        data_point=(sample['latitude'], sample['longitude']),
        requested_point=(lat, lon),
    )
    return dataclasses.replace(bottom, data_sources=(prov,))


def fetch_bottom_local_transect(start, end, *, n_points=6, max_points=None,
                                roughness=0.0,
                                water_sound_speed=None,
                                max_distance_km=DEFAULT_MAX_DISTANCE_KM,
                                timeout=None, verbose=False):
    """Range-dependent bottom from local samples along ``start`` → ``end``.

    The grain-size provider of the ``fetch_bottom_local_transect`` protocol
    name, and the one exported at package level as
    ``uacpy.data.fetch_bottom_local_transect``. ``water_sound_speed`` also
    takes a ``(lat, lon) -> m/s`` callable, so each column scales to the
    water over its own seafloor. ``timeout``/``verbose`` are
    accepted (and ignored — this backend is offline) for signature
    uniformity with the network bottom fetchers.

    An unreadable cache is reported as itself: the sample indices are built
    once here, before any waypoint, so a corrupt or absent CSV raises its own
    message rather than the transect's coverage message.
    """
    # ``range_dependent_bottom_along`` treats a waypoint's ``DataFetchError``
    # as "this point is not covered" and fills it from the nearest covered
    # neighbour. A cache that cannot be read raises that same exception at
    # *every* waypoint, so all of them become gaps and the all-gaps guard
    # reports the transect as uncovered — discarding the one remediation that
    # would fix it ("Re-run ./install.sh --data sediment"). A broken source is
    # not a property of any waypoint, so read it once up front and let the
    # per-waypoint exception keep its single meaning. Discriminating on the
    # message instead would be the wrong tool. ``_samples`` is memoized and
    # the first waypoint would build it anyway, so a healthy cache pays a
    # dict lookup.
    _samples()
    return range_dependent_bottom_along(
        lambda la, lo: fetch_bottom_local(
            (la, lo), roughness=roughness,
            water_sound_speed=water_sound_speed_at(water_sound_speed, la, lo),
            max_distance_km=max_distance_km),
        start, end, n_points, source_label='local sediment DB',
        max_points=max_points,
    )
