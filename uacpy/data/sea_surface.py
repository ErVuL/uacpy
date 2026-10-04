"""Fetched sea-state → a Pierson-Moskowitz sea-surface altimetry realization.

Turns a fetched sea state into the :class:`~uacpy.core.altimetry.Altimetry`
carrier ``Environment(altimetry=...)`` consumes for rough-surface scattering,
with the sea state's provenance on its ``data_sources``. The
observed significant wave height ``Hs`` is inverted to the effective
Pierson-Moskowitz wind ``U = √(Hs / 0.0214)`` (see :data:`_PM_HS_COEFF` for
the sources) and handed to
:func:`uacpy.core.altimetry.generate_sea_surface`, so the realization reproduces the
observed ``Hs`` regardless of whether the sea is fully developed. When no wave
source is available it falls back to the fetched 10 m wind, scaled to the
19.5 m PM reference height (a fully-developed assumption) — the live NBS field
first, then the cached NBS monthly climatology.
"""

import dataclasses

import numpy as np

from uacpy.core.altimetry import Altimetry
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.core.altimetry import SEA_SURFACE_CALM_WIND_MPS, generate_sea_surface
from uacpy.core.units import knots_to_ms, ms_to_knots
from uacpy._log import log_message
from uacpy.core.geo import as_coordinate
from uacpy.data._geo import checked_max_distance
from uacpy.data._chain import first_answer
from uacpy.core.geo import parse_date
from uacpy.data.sources import SOURCES, DataProvenance

__all__ = ['fetch_sea_surface', 'hs_to_pm_wind', 'SEA_SURFACE_SOURCES']

SEA_SURFACE_SOURCES = ('waverys', 'ww3', 'nbs', 'local', 'auto')
#: The rung each source token tries: ``(rung, backend)``. ``'waverys'`` and
#: ``'ww3'`` build from the wave height, ``'nbs'`` from the live 10 m wind;
#: ``'local'`` is the cached NBS monthly climatology (``install.sh --data
#: wind``) — a mean state, not the day's, so it is the last rung of
#: ``'auto'`` rather than the cache-first rung the other axes use: for sea
#: state the date-specific product is the better answer.
_RUNGS = {'waverys': ('waves', 'waverys'), 'ww3': ('waves', 'ww3'),
          'nbs': ('nbs', 'nbs'), 'local': ('nbs', 'local')}
_AUTO_ORDER = ('waverys', 'ww3', 'nbs', 'local')
#: Pierson-Moskowitz fully-developed significant wave height Hs = 0.21·U²/g
#: with U at the 19.5 m reference height, the relation used by
#: :func:`generate_sea_surface`. Etter, *Underwater Acoustic Modeling and
#: Simulation*, gives it as H(1/3) = 0.566e-2·V² for V in knots, which is
#: 0.02139 in m/s; integrating Medwin & Clay's Pierson-Moskowitz spectrum
#: (alpha 8.1e-3, beta 0.74) and taking Hs = 4·h_rms gives 0.02133.
_PM_HS_COEFF = 0.0214
#: Wind scaling from the 10 m observation height to the 19.5 m
#: Pierson-Moskowitz reference height. A conventional factor: a neutral log
#: profile over any plausible sea roughness gives 1.05-1.06 instead, so this is
#: not one, and no source in the corpus derives it. Kept because it is the
#: value in common use; treat a wave height built from a 10 m wind as
#: approximate.
_U10_TO_U195 = 1.026


def hs_to_pm_wind(hs):
    """Effective Pierson-Moskowitz wind speed (m/s) reproducing wave height ``hs``.

    Parameters
    ----------
    hs : float
        Significant wave height (m); a negative value counts as 0.
    """
    return float(np.sqrt(max(float(hs), 0.0) / _PM_HS_COEFF))


def fetch_sea_surface(point, *, date, rmax_m, n_points=None, rng=None,
                      source='auto', max_days=None, timeout=120.0,
                      verbose=False, max_distance_km=None):
    """Sea-surface altimetry and the provenance of the sea state it came from.

    Parameters
    ----------
    point : (lat, lon)
        Site coordinates in decimal degrees.
    date : str or datetime.date
        Calendar date of the sea state (required; sea state is time-specific).
    rmax_m : float
        Range extent (m) of the realization — set to the transect length.
    n_points : int, optional
        Range samples in the returned altimetry. Default ``None``: size the
        realization from the sea state by
        :func:`uacpy.core.altimetry.sea_surface_n_points` — the rule
        :func:`generate_sea_surface` uses: 8 samples per Pierson-Moskowitz
        peak wavelength, between 500 and 200 000 (capped with a warning), 500
        for a calm sea — so the realized wave height tracks the fetched one at
        any transect length. A pinned count too coarse to resolve the peak
        warns and returns the aliased (flatter than requested) surface.
    rng : numpy.random.Generator, optional
        Random generator the surface realization draws from; pass one (e.g.
        ``np.random.default_rng(1)``) for a reproducible surface. ``None``
        draws a fresh realization on every call.
    source : str or sequence of str, optional
        Catalogue ids, tried in order: ``'waverys'`` (Copernicus WAVERYS) and
        ``'ww3'`` (WaveWatch III) build from the fetched significant wave
        height; ``'nbs'`` from the live 10 m wind (fully-developed
        assumption); ``'local'`` from the cached NBS monthly wind climatology
        (no network — a mean state, so it understates day-to-day sea state).
        ``'auto'`` (default) is ``('waverys', 'ww3', 'nbs', 'local')``.
    max_days : int, optional
        Time tolerance forwarded to the wave source.
    timeout : float, optional
        Per-request network timeout in seconds. Default 120.
    verbose : bool or str, optional
        Logging gate passed through to ``log_message``.
    max_distance_km : float, optional
        Refuse a sea state read farther than this (km) from ``point`` (the
        offset rule), whichever source answers; ``None`` (default) sets no
        limit beyond its warning.

    Returns
    -------
    Altimetry
        The realization (``.ranges`` 0 → ``rmax_m``, ``.heights`` in m),
        carrying the sea state's provenance like every fetched carrier:
        ``.data_sources`` is one :class:`~uacpy.data.DataProvenance` whose
        ``source.id`` is the catalogue id that supplied it (``'waverys'`` /
        ``'ww3'`` / ``'nbs'``). Its ``data_date`` separates the backends that
        share an id: the cached NBS climatology reads ``'month MM, <period>'``
        (the period the cache recorded); a live fetch leaves it ``None`` and
        carries the ``requested_date``.
    """
    lat, lon = as_coordinate(point)            # validate before any request
    limit = checked_max_distance(max_distance_km, 'fetch_sea_surface')
    order = (_AUTO_ORDER if source == 'auto'
             else (source,) if isinstance(source, str) else tuple(source))
    unknown = [s for s in order if s not in _RUNGS]
    if not order or unknown:
        raise ConfigurationError(
            f"fetch_sea_surface: unknown source {source!r}.",
            remediation=f"Use one of {sorted(SEA_SURFACE_SOURCES)}, or a "
                        f"sequence of {sorted(_RUNGS)}.",
        )
    # A wave rung answers with the finished surface; a wind rung answers
    # with the 10 m wind, from which the surface is built below.
    attempts = [_RUNGS[s] for s in order]

    def call(rung, backend):
        if rung == 'waves':
            from uacpy.data.waves import fetch_waves
            w = fetch_waves(point, date=date, source=backend,
                            max_days=max_days, timeout=timeout,
                            verbose=verbose, max_distance_km=limit)
            u = hs_to_pm_wind(w.hs)
            log_message('waves', f"Hs {w.hs:.2f} m "
                        f"({w.provenance.source.id}) → "
                        f"PM wind {float(ms_to_knots(u)):.1f} kn", verbose=verbose)
            return _altimetry(_surface(rmax_m, u, n_points, rng),
                              w.provenance)
        from uacpy.data.wind_live import wind_at
        return wind_at(point, date=date, source=backend, timeout=timeout,
                       verbose=verbose, max_distance_km=limit,
                       who='fetch_sea_surface')

    answer, (rung, backend) = first_answer(attempts, call)
    if rung == 'waves':
        return answer
    answer, wind_prov = answer
    u10 = float(knots_to_ms(answer))    # wind_at answers in knots
    u = u10 * _U10_TO_U195
    kind = 'climatology' if backend == 'local' else 'live'
    log_message('wind', f"U10 {answer:.1f} kn (nbs {kind}) → PM wind "
                f"{float(ms_to_knots(u)):.1f} kn", verbose=verbose)
    return _altimetry(_surface(rmax_m, u, n_points, rng),
                      dataclasses.replace(
                          _provenance('nbs', lat, lon, date,
                                      climatology=(backend == 'local')),
                          data_point=wind_prov.data_point))


def _provenance(src_id, lat, lon, date, *, climatology=False):
    """Provenance of a sea-state fetch: a live product is dated by the
    request; the cached wind climatology by its month and reference period."""
    day = parse_date(date)
    data_date = None
    if climatology:
        from uacpy.data.wind_local import climatology_period
        try:
            period = climatology_period()
        except (ConfigurationError, DataFetchError):
            period = None
        data_date = (f"month {day.month:02d}, {period}" if period
                     else f"month {day.month:02d} (climatology)")
    return DataProvenance(source=SOURCES[src_id], data_date=data_date,
                          requested_point=(lat, lon),
                          requested_date=day.isoformat())


def _altimetry(pairs, prov):
    """``(n, 2)`` ``[range_m, height_m]`` pairs → an ``Altimetry`` stamped
    with ``prov``."""
    pairs = np.asarray(pairs, dtype=float)
    return Altimetry(ranges=pairs[:, 0], heights=pairs[:, 1],
                     data_sources=(prov,))


def _surface(rmax_m, wind_ms, n_points, rng):
    # A calm sea (near-zero wind / wave) would trip generate_sea_surface's
    # positive-wind guard; floor it to the calm threshold, which the core
    # sizing rule answers with its fixed count and no warning — the
    # realization stands in for a flat surface either way. Sizing, cap and
    # the under-resolution warning are generate_sea_surface's.
    wind_ms = max(wind_ms, SEA_SURFACE_CALM_WIND_MPS)
    return generate_sea_surface(rmax_m, ms_to_knots(wind_ms), n_points=n_points,
                                rng=rng)
