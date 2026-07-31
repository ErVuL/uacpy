"""Local seafloor-sediment samples (DECK41 + NGDC grain-size) → bottom.

The offline, *public-domain*, **global** sediment backend (commercial-clean).
``install.sh --data sediment`` downloads two NCEI point databases into the cache
as normalized CSVs:

* ``grainsize.csv`` — NGDC Seafloor Sediment Grain-Size DB (quantitative mean ϕ),
* ``deck41.csv``    — DECK41 surficial descriptions (dominant lithology text).

A nearest-neighbour lookup (great-circle, via a unit-sphere KD-tree) returns the
closest sample; quantitative grain-size samples are preferred over lithology
descriptions. Because these are sparse point samples (not a continuous map), the
nearest sample can be far offshore — a ``max_distance_km`` guard raises when no
sample is close enough.
"""

import csv
import io
import tarfile
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from uacpy._log import log_message
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data import _cache
from uacpy.data._geo import as_coordinate, normalize_lon, EARTH_RADIUS_KM
from uacpy.data._http import http_get, checked_member_size
from uacpy.data.sediment import (
    bottom_from_grain_size, range_dependent_bottom_along, water_sound_speed_at,
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
    'rock': -5.0, 'gravel and coarser': -3.0,
}

# Candidate column names (lower-cased) for the tolerant CSV reader.
_LAT_COLS = ('latitude', 'lat', 'y')
_LON_COLS = ('longitude', 'lon', 'long', 'x')
_PHI_COLS = ('phi', 'mean_phi', 'mean', 'mean_grain_size', 'grain_size_phi')
_LITH_COLS = ('lithology', 'dominant_lithology', 'dominant', 'description')

_SAMPLES = {}   # cache_root -> (tree, phi_array)


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
    dest = Path(cache_dir) if cache_dir else _cache.dataset_root('sediment')
    dest.mkdir(parents=True, exist_ok=True)
    log_message('sediment', "downloading NCEI grain-size DB (G00127, ~3 MB)",
                verbose=verbose)
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
            iv = int(r['interval'] or 0)
            lo = max(float(r['lower_phi_limit']), lo_c)
            hi = min(float(r['upper_phi_limit']), hi_c)
            w = float(r['weight_percent'])
        except (KeyError, ValueError):
            continue
        if hi < lo or w <= 0.0:
            continue
        centre = 0.5 * (lo + hi)
        if key not in best or iv < best[key][0]:
            best[key] = [iv, 0.0, 0.0]       # surficial: keep the top interval
        if iv == best[key][0]:
            best[key][1] += w
            best[key][2] += centre * w

    out = dest / 'grainsize.csv'
    n = 0
    with open(out, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['latitude', 'longitude', 'mean_phi'])
        for key, (iv, sum_w, sum_w_phi) in best.items():
            if key in loc and sum_w > 0.0:
                lat, lon = loc[key]
                writer.writerow([lat, lon, round(sum_w_phi / sum_w, 3)])
                n += 1
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
    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(la) * np.cos(lo),
                            np.cos(la) * np.sin(lo),
                            np.sin(la)])


def _read_csv(path, value_cols, transform):
    """Return ``(lats, lons, phis)`` from a normalized sediment CSV.

    ``transform`` maps a raw cell (ϕ string or lithology text) to a float ϕ, or
    ``None`` to skip the row.
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
            raise ConfigurationError(
                f"Sediment file {path.name} is missing expected columns "
                f"(need lat, lon and one of {value_cols}); got {header}.",
                remediation="Re-run ./install.sh --data sediment to refresh it.",
            )
        idx = {h: i for i, h in enumerate(header)}
        li, oi, vi = idx[lat_c], idx[lon_c], idx[val_c]
        for row in reader:
            if not row or len(row) <= max(li, oi, vi):
                continue
            phi = transform(row[vi])
            if phi is None:
                continue
            try:
                lats.append(float(row[li]))
                lons.append(normalize_lon(float(row[oi])))
            except ValueError:
                continue
            phis.append(phi)
    return lats, lons, phis


def _phi_from_float(cell):
    try:
        return float(cell)
    except ValueError:
        return None


def _phi_from_lithology(cell):
    return _DECK41_LITHOLOGY_TO_PHI.get(cell.lower().strip())


def _samples():
    """Build (or reuse) the combined nearest-neighbour index of ϕ samples."""
    root = str(_cache.cache_root())
    if root in _SAMPLES:
        return _SAMPLES[root]
    _cache.require('sediment')                       # raises if not installed
    lats, lons, phis = [], [], []
    gs = _cache.dataset_root('sediment') / 'grainsize.csv'
    if gs.exists():                                  # quantitative, preferred
        la, lo, ph = _read_csv(gs, _PHI_COLS, _phi_from_float)
        lats += la; lons += lo; phis += ph
    d41 = _cache.dataset_root('sediment') / 'deck41.csv'
    if d41.exists():
        la, lo, ph = _read_csv(d41, _LITH_COLS, _phi_from_lithology)
        lats += la; lons += lo; phis += ph
    if not phis:
        raise DataFetchError(
            "No usable sediment samples found in the local cache.",
            remediation="Re-run ./install.sh --data sediment.",
        )
    tree = cKDTree(_unit_vectors(np.array(lats), np.array(lons)))
    result = (tree, np.array(phis))
    _SAMPLES[root] = result
    return result


def fetch_sediment_sample(point, *, max_distance_km=DEFAULT_MAX_DISTANCE_KM):
    """Nearest local sediment sample to ``point``: ``{'phi', 'distance_km'}``.

    Raises ``DataFetchError`` if the closest sample is farther than
    ``max_distance_km`` (sparse point data — no nearby ground truth).
    """
    lat, lon = as_coordinate(point)
    tree, phis = _samples()
    chord, idx = tree.query(_unit_vectors(np.array([lat]), np.array([lon]))[0])
    # chord length on the unit sphere → great-circle distance.
    dist_km = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.clip(chord / 2.0, 0, 1))
    if max_distance_km is not None and dist_km > max_distance_km:
        raise DataFetchError(
            f"Nearest sediment sample is {dist_km:.0f} km away "
            f"(> max_distance_km={max_distance_km:.0f}).",
            remediation="Raise max_distance_km, or supply a bottom directly.",
        )
    return {'phi': float(phis[idx]), 'distance_km': float(dist_km)}


def fetch_bottom_local(point, *, roughness=0.0, water_sound_speed=None,
                       max_distance_km=DEFAULT_MAX_DISTANCE_KM,
                       timeout=None, verbose=False):
    """Model-ready bottom from the nearest local sediment sample.

    ``timeout``/``verbose`` are accepted (and ignored — this backend is offline)
    for signature uniformity with the network bottom fetchers.
    ``water_sound_speed`` (m/s) scales the grain-size velocity ratio to the
    in-situ near-seabed water; ``None`` uses the Hamilton reference.
    """
    phi = fetch_sediment_sample(point, max_distance_km=max_distance_km)['phi']
    return bottom_from_grain_size(
        phi, roughness=roughness, water_sound_speed=water_sound_speed)


def fetch_bottom_local_transect(start, end, *, n_points=6, max_points=None,
                                roughness=0.0,
                                water_sound_speed=None,
                                max_distance_km=DEFAULT_MAX_DISTANCE_KM,
                                timeout=None, verbose=False):
    """Range-dependent bottom from local samples along ``start`` → ``end``.

    ``water_sound_speed`` also takes a ``(lat, lon) -> m/s`` callable, so each column scales to the water over its own seafloor.
    """
    return range_dependent_bottom_along(
        lambda la, lo: fetch_bottom_local(
            (la, lo), roughness=roughness,
            water_sound_speed=water_sound_speed_at(water_sound_speed, la, lo),
            max_distance_km=max_distance_km),
        start, end, n_points, source_label='local sediment DB',
        max_points=max_points,
    )
