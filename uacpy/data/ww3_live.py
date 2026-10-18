"""NOAA WaveWatch III live significant wave height (public domain).

WaveWatch III is NOAA's operational spectral wave model; its global grid is
served live (no auth) from the PacIOOS **ERDDAP** griddap mirror. The operational
feed is a rolling recent window, so arbitrary historical dates are not covered —
:func:`uacpy.data.fetch_waves` prefers the Copernicus WAVERYS reanalysis for
history and falls back here.

The backend of ``fetch_waves(source='ww3')``, which returns the significant
wave height (m) with its provenance; WaveWatch III output is **public
domain**.
"""

import numpy as np

from uacpy.core.exceptions import DataFetchError
from uacpy.core.geo import as_coordinate
from uacpy.data._geo import checked_offset
from uacpy.data.sources import SOURCES, DataProvenance
from uacpy.data._http import (erddap_griddap_url, erddap_point,
                              http_get)
from uacpy.core.geo import parse_date

__all__ = ['ERDDAP_URL', 'DATASET']

ERDDAP_URL = 'https://pae-paha.pacioos.hawaii.edu/erddap/griddap'
DATASET = 'ww3_global'
#: Significant-wave-height variable candidates (the PacIOOS mirror names it
#: ``Thgt``; other ERDDAP hosts use ``hs`` / ``htsgwsfc``).
_HS_VARS = ('Thgt', 'hs', 'htsgwsfc', 'significant_wave_height')
_USER_AGENT = 'uacpy (+https://github.com/ErVuL/uacpy)'


def _griddap_url(var, when, lat, lon):
    return erddap_griddap_url(ERDDAP_URL, DATASET, var, when, lat, lon,
                              level=0.0)


def hs_at(point, *, date, timeout=60.0, verbose=False, max_distance_km=None,
          who='fetch_waves'):
    """``(Hs in m, the 'ww3' DataProvenance of the grid node read)``.

    The field is read at 00:00 UTC of ``date`` (a time of day is ignored);
    the provenance names the step read in its ``data_date``. The host is the
    module constant :data:`ERDDAP_URL` (a third-party PacIOOS mirror);
    reassign it to read another. The griddap selectors snap to the node
    nearest the query, so the node is the point's own cell by construction
    and only ``max_distance_km`` (checked by the caller) can refuse it.
    Raises ``DataFetchError`` where WW3 has no value (land, or a date outside
    the served window), naming what each variable request answered, so a
    network failure reads differently from a miss.
    """
    lat, lon = as_coordinate(point)
    outcomes = []
    for var in _HS_VARS:
        try:
            body = http_get(_griddap_url(var, date, lat, lon), timeout=timeout,
                            verbose=verbose, source='waves',
                            user_agent=_USER_AGENT).decode('utf-8', 'replace')
        except DataFetchError as exc:
            outcomes.append(f"{var}: {exc.message}")  # variable / range miss — next
            continue
        node_lat, node_lon, step, hs = erddap_point(body)
        # A wave height cannot be negative: a negative number is a fill or a
        # schema change, and reads as no value rather than as its magnitude.
        if np.isfinite(hs) and hs >= 0.0:
            prov = checked_offset(
                DataProvenance(
                    source=SOURCES['ww3'],
                    data_point=(None if node_lat is None else (node_lat, node_lon)),
                    data_date=step,
                    requested_point=(lat, lon),
                    requested_date=parse_date(date).isoformat(),
                    point_kind='cell'),
                who=who, warn_km=float('inf'), max_distance_km=max_distance_km)
            return float(hs), prov
        outcomes.append(f"{var}: no value at the nearest cell")
    raise DataFetchError(
        f"WaveWatch III has no wave height at ({lat:.4f}, {lon:.4f}) on "
        f"{parse_date(date)} — {'; '.join(outcomes)}.",
        remediation="Land or a date outside the served window: use a recent "
                    "date / ocean point, or waves via Copernicus (fetch_waves "
                    "with the copernicusmarine login). A transport failure "
                    "above is the network, not the point.",
    )
