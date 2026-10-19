"""Significant wave height dispatcher — Copernicus WAVERYS / WaveWatch III.

Sea state is live-only (a mean-state climatology would smooth away the day-to-day
wave field it is meant to capture). Two sources, tried in order by ``'auto'``:
the Copernicus WAVERYS reanalysis (full history, 1980→present, needs the
``copernicusmarine`` login) and NOAA WaveWatch III (no auth, recent window).

The wave height drives the Pierson-Moskowitz sea surface in
:func:`uacpy.data.fetch_sea_surface`.
"""

import warnings
from dataclasses import dataclass
from typing import Optional

from uacpy.core._export import ExportRecord
from uacpy.core.exceptions import ConfigurationError, FallbackWarning
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.geo import as_coordinate
from uacpy.data._chain import (SourceChain, SourceProvider,
                               unused_setting_notices)
from uacpy.data._geo import checked_max_days, checked_max_distance
from uacpy.data.sources import DataProvenance
from uacpy.data._provenance_notice import one_provenance_notice

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


def _waverys_point(_backend, point, *, date, verbose, max_distance_km,
                   max_days, **_request):
    """A :data:`_WAVE_CHAIN` point fetch: the Copernicus WAVERYS record."""
    from uacpy.data.copernicus import fetch_waves_operational
    extra = {} if max_days is None else {'max_days': max_days}
    return fetch_waves_operational(point, date=date, verbose=verbose,
                                   max_distance_km=max_distance_km, **extra)


def _ww3_point(_backend, point, *, date, timeout, verbose, max_distance_km,
               base_url=None, **_request):
    """A :data:`_WAVE_CHAIN` point fetch: the WaveWatch III record."""
    from uacpy.data.ww3_live import hs_at
    hs, prov = hs_at(point, date=date, timeout=timeout, verbose=verbose,
                     max_distance_km=max_distance_km, who='fetch_waves',
                     base_url=base_url)
    return SeaStateRecord(hs=hs, tp=None, provenance=prov)


#: The wave sources, both live (sea state has no cached twin, so ``'auto'``
#: is the one preset).
_WAVE_CHAIN = SourceChain(
    'wave',
    providers=(SourceProvider('waverys', ('live',), _waverys_point),
               SourceProvider('ww3', ('live',), _ww3_point)),
    auto=WAVE_SOURCES, presets=('auto',))


@one_provenance_notice(subject="the sea state's data",
                       record="the result's .provenance")
def fetch_waves(point, *, date, source='auto', max_days=None, timeout=120.0,
                verbose=False, max_distance_km=None,
                base_url=None) -> SeaStateRecord:
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
        Time tolerance of the WAVERYS read; ``None`` keeps its own. WaveWatch
        III reads ``date`` itself, so a ``max_days`` it answers gives a
        ``FallbackWarning`` that the value was not used.
    timeout : float, optional
        WaveWatch III request timeout in seconds. Default 120.
    verbose : bool or str, optional
        Logging gate passed through to ``log_message``.
    max_distance_km : float, optional
        Refuse a grid node standing farther than this (km) from ``point`` (the
        offset rule), whichever source answers; ``None`` (default) sets no
        limit beyond its warning.
    base_url : str, optional
        The ERDDAP griddap root WaveWatch III is read from (a mirror serving
        ``ww3_global``); ``None`` (default) reads the module constant
        ``uacpy.data.ww3_live.ERDDAP_URL``. WAVERYS reads its own Copernicus
        session, so ``base_url`` with a source list that holds no ``'ww3'``
        is refused, and a ``FallbackWarning`` says it was not used when
        WAVERYS answered.
    """
    as_coordinate(point)                       # validate before any request
    limit = checked_max_distance(max_distance_km, 'fetch_waves')
    if max_days is not None:
        max_days = checked_max_days(max_days, 'fetch_waves')
    if base_url is not None and 'ww3' not in _WAVE_CHAIN.resolve(source)[0]:
        raise ConfigurationError(
            f"fetch_waves: base_url= is the WaveWatch III ERDDAP griddap "
            f"root, which only source 'ww3' reads; source={source!r} does "
            f"not read it.",
            remediation="Drop base_url=, or include 'ww3' in source=.")
    request = dict(date=date, timeout=timeout, verbose=verbose,
                   max_distance_km=limit, max_days=max_days,
                   base_url=base_url)
    out, (name, _backend) = _WAVE_CHAIN.answer(
        source,
        lambda src, backend: _WAVE_CHAIN.fetch(
            src, backend, (point,), request, who='fetch_waves',
            keyword='source='),
        who='fetch_waves', keyword='source=')
    if max_days is not None and name == 'ww3':
        warnings.warn(
            f"fetch_waves: max_days={max_days} was not used: WaveWatch III "
            f"answered, and it reads the field at 00:00 UTC of date= itself, "
            f"with no time tolerance (only WAVERYS reads max_days).",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
    if name != 'ww3':
        unused_setting_notices('fetch_waves', [(
            'base_url', base_url,
            "WAVERYS answered, and only WaveWatch III reads base_url")])
    return out
