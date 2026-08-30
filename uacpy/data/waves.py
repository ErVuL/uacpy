"""Significant wave height dispatcher — Copernicus WAVERYS / WaveWatch III.

Sea state is live-only (a mean-state climatology would smooth away the day-to-day
wave field it is meant to capture). Two sources, tried in order by ``'auto'``:
the Copernicus WAVERYS reanalysis (full history, 1980→present, needs the
``copernicusmarine`` login) and NOAA WaveWatch III (no auth, recent window).

The wave height drives the Pierson-Moskowitz sea surface in
:func:`uacpy.data.fetch_sea_surface`.
"""

from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data._geo import as_coordinate
from uacpy.data._http import raise_substantive

__all__ = ['fetch_waves', 'WAVE_SOURCES']

#: Valid wave sources, in the order ``'auto'`` tries them: Copernicus WAVERYS
#: (full history, needs the login) before WaveWatch III (no auth, recent).
WAVE_SOURCES = ('copernicus', 'ww3')
_SOURCE_ID = {'copernicus': 'waverys', 'ww3': 'ww3'}


def _resolve_order(source):
    if source == 'auto':
        return WAVE_SOURCES
    order = (source,) if isinstance(source, str) else tuple(source)
    for name in order:
        if name not in WAVE_SOURCES:
            raise ConfigurationError(
                f"fetch_waves: unknown wave source {name!r}.",
                remediation=f"Use 'auto' or one of {sorted(WAVE_SOURCES)}.",
            )
    return order


def fetch_waves(point, *, date, source='auto', max_days=None, timeout=120.0,
                verbose=False):
    """Significant wave height (m) at a ``(lat, lon)`` point and date.

    Returns ``{'hs', 'tp', 'source'}`` (``tp`` may be ``None``; ``source`` is the
    catalogue id that answered). ``source='auto'`` tries Copernicus WAVERYS
    (full history) then WaveWatch III (recent). Raises ``DataFetchError`` when no
    source yields a value. ``timeout`` bounds the WaveWatch III request; the
    Copernicus session owns its own (see :mod:`uacpy.data.copernicus`).
    """
    as_coordinate(point)                       # validate before any request
    order = _resolve_order(source)
    errors = []
    for name in order:
        try:
            if name == 'copernicus':
                from uacpy.data.copernicus import fetch_waves_operational
                extra = {} if max_days is None else {'max_days': max_days}
                out = fetch_waves_operational(point, date=date,
                                              verbose=verbose, **extra)
            else:
                from uacpy.data.ww3_live import fetch_hs
                out = {'hs': fetch_hs(point, date=date, timeout=timeout,
                                     verbose=verbose), 'tp': None}
        except (DataFetchError, ConfigurationError) as exc:
            errors.append(exc)
            continue
        out['source'] = _SOURCE_ID[name]
        return out
    raise_substantive(errors)
