"""NOAA WaveWatch III live significant wave height (public domain).

WaveWatch III is NOAA's operational spectral wave model; its global grid is
served live (no auth) from the PacIOOS **ERDDAP** griddap mirror. The operational
feed is a rolling recent window, so arbitrary historical dates are not covered —
:func:`uacpy.data.fetch_waves` prefers the Copernicus WAVERYS reanalysis for
history and falls back here.

Returns significant wave height (m); WaveWatch III output is **public domain**.
"""

import numpy as np

from uacpy.core.exceptions import DataFetchError
from uacpy.data._geo import as_coordinate
from uacpy.data._http import http_get
from uacpy.data._time import parse_date

__all__ = ['fetch_hs', 'ERDDAP_URL', 'DATASET']

ERDDAP_URL = 'https://pae-paha.pacioos.hawaii.edu/erddap/griddap'
DATASET = 'ww3_global'
#: Significant-wave-height variable candidates (the PacIOOS mirror names it
#: ``Thgt``; other ERDDAP hosts use ``hs`` / ``htsgwsfc``).
_HS_VARS = ('Thgt', 'hs', 'htsgwsfc', 'significant_wave_height')
_USER_AGENT = 'uacpy (+https://github.com/ErVuL/uacpy)'


def _griddap_url(var, when, lat, lon):
    """``ww3_global`` axes are [time][depth][latitude][longitude] with a
    singleton surface depth node and a [0, 360) longitude axis."""
    import urllib.parse
    iso = f"{parse_date(when)}T00:00:00Z"
    constraint = f"{var}[({iso})][(0.0)][({lat})][({lon % 360.0})]"
    query = urllib.parse.quote(constraint, safe='[]():.,-TZ')
    return f"{ERDDAP_URL}/{DATASET}.csv?{query}"


def _last_value(body):
    rows = [ln for ln in body.splitlines() if ln.strip()]
    if len(rows) < 3:
        return np.nan
    try:
        return float(rows[-1].split(',')[-1])
    except (ValueError, IndexError):
        return np.nan


def fetch_hs(point, *, date, timeout=60.0, verbose=False):
    """Significant wave height (m) at a ``(lat, lon)`` point and date from WW3.

    Raises ``DataFetchError`` where WW3 has no value (land, or a date outside the
    served window).
    """
    lat, lon = as_coordinate(point)
    for var in _HS_VARS:
        try:
            body = http_get(_griddap_url(var, date, lat, lon), timeout=timeout,
                            verbose=verbose, source='waves',
                            user_agent=_USER_AGENT).decode('utf-8', 'replace')
        except DataFetchError:
            continue                              # variable / range miss — next
        hs = _last_value(body)
        if np.isfinite(hs):
            return abs(hs)
    raise DataFetchError(
        f"WaveWatch III has no wave height at ({lat:.4f}, {lon:.4f}) on "
        f"{parse_date(date)} (land, or outside the served window).",
        remediation="Use a recent date / ocean point, or waves via Copernicus "
                    "(fetch_waves with the copernicusmarine login).",
    )
