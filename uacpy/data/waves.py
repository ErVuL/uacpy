"""Significant wave height dispatcher — Copernicus WAVERYS / WaveWatch III.

Sea state is live-only (a mean-state climatology would smooth away the day-to-day
wave field it is meant to capture). Two sources, tried in order by ``'auto'``:
the Copernicus WAVERYS reanalysis (full history, 1980→present, needs the
``copernicusmarine`` login) and NOAA WaveWatch III (no auth, recent window).

The wave height drives the Pierson-Moskowitz sea surface in
:func:`uacpy.data.fetch_sea_surface`.
"""

from dataclasses import dataclass
from typing import Optional

from uacpy.core._export import ExportRecord
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.geo import as_coordinate
from uacpy.data._chain import first_answer
from uacpy.data._geo import checked_max_distance
from uacpy.data.sources import DataProvenance

__all__ = ['SeaStateRecord', 'fetch_waves', 'WAVE_SOURCES']


@dataclass(frozen=True, eq=False)
class SeaStateRecord(ExportRecord):
    """The sea state at one point and date, as :func:`fetch_waves` and
    :func:`~uacpy.data.copernicus.fetch_waves_operational` return it.

    Attributes
    ----------
    hs : float
        Significant wave height (m).
    tp : float or None
        Peak wave period (s); ``None`` where the source does not provide one
        (WaveWatch III, or a WAVERYS dataset without the variable).
    provenance : DataProvenance
        The source that answered — ``provenance.source.id`` is its catalogue
        id — with the requested point and date and, for WAVERYS, the snapped
        date and cell.
    """

    hs: float
    tp: Optional[float]
    provenance: DataProvenance

    _REPR_UNITS = {'hs': 'm', 'tp': 's'}

#: Valid wave sources — their catalogue ids — in the order ``'auto'`` tries
#: them: Copernicus WAVERYS (full history, needs the login) before WaveWatch
#: III (no auth, recent).
WAVE_SOURCES = ('waverys', 'ww3')


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
                verbose=False, max_distance_km=None) -> SeaStateRecord:
    """Significant wave height (m) at a ``(lat, lon)`` point and date.

    ``source`` is ``'waverys'`` (the Copernicus WAVERYS reanalysis), ``'ww3'``
    (NOAA WaveWatch III), a sequence of them, or ``'auto'``; each is its
    catalogue id in :data:`uacpy.data.SOURCES`. Returns a
    :class:`SeaStateRecord` (``tp`` may be ``None``); its ``provenance`` is
    the :class:`~uacpy.data.DataProvenance` of the source that answered —
    ``provenance.source.id`` is its id, with the requested point and date,
    and for WAVERYS the snapped date and cell.
    ``source='auto'`` tries Copernicus WAVERYS
    (full history) then WaveWatch III (recent). Raises ``DataFetchError`` when no
    source yields a value. ``timeout`` bounds the WaveWatch III request; the
    Copernicus session owns its own (see :mod:`uacpy.data.copernicus`).

    Parameters
    ----------
    point : (lat, lon)
        Site coordinates in decimal degrees.
    date : str or datetime.date
        Calendar date of the sea state.
    source : str or sequence of str, optional
        ``'waverys'``, ``'ww3'``, a sequence of them, or ``'auto'`` (default).
    max_days : int, optional
        Time tolerance forwarded to the wave source; ``None`` keeps its own.
    timeout : float, optional
        WaveWatch III request timeout in seconds. Default 120.
    verbose : bool or str, optional
        Logging gate passed through to ``log_message``.
    max_distance_km : float, optional
        Refuse a grid node standing farther than this (km) from ``point`` (the
        offset rule), whichever source answers; ``None`` (default) sets no
        limit beyond its warning.
    """
    as_coordinate(point)                       # validate before any request
    limit = checked_max_distance(max_distance_km, 'fetch_waves')
    order = _resolve_order(source)

    def call(name, _backend):
        if name == 'waverys':
            from uacpy.data.copernicus import fetch_waves_operational
            extra = {} if max_days is None else {'max_days': max_days}
            return fetch_waves_operational(point, date=date, verbose=verbose,
                                           max_distance_km=limit, **extra)
        from uacpy.data.ww3_live import hs_at
        hs, prov = hs_at(point, date=date, timeout=timeout, verbose=verbose,
                         max_distance_km=limit, who='fetch_waves')
        return SeaStateRecord(hs=hs, tp=None, provenance=prov)

    out, _attempt = first_answer(((name, None) for name in order), call)
    return out
