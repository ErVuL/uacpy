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

import numpy as np

from uacpy.core.altimetry import Altimetry
from uacpy.core.altimetry import SEA_SURFACE_CALM_WIND, generate_sea_surface
from uacpy.core.units import knots_to_ms, ms_to_knots
from uacpy._log import log_message
from uacpy.core.geo import as_coordinate
from uacpy.data._geo import checked_max_days, checked_max_distance
from uacpy.data._chain import (SourceChain, SourceProvider,
                               unused_setting_notices)
from uacpy.data._provenance_notice import one_provenance_notice
from uacpy.data.wind_live import WIND_SOURCES

__all__ = ['fetch_sea_surface', 'hs_to_pm_wind', 'SEA_SURFACE_SOURCES']

SEA_SURFACE_SOURCES = ('waverys', 'ww3', 'nbs', 'local', 'auto')
#: Pierson-Moskowitz fully-developed significant wave height Hs = 0.21·U²/g
#: with U at the 19.5 m reference height, the relation used by
#: :func:`generate_sea_surface`. Etter, *Underwater Acoustic Modeling and
#: Simulation*, gives it as H(1/3) = 0.566e-2·V² for V in knots, which is
#: 0.02139 in m/s; integrating Medwin & Clay's Pierson-Moskowitz spectrum
#: (alpha 8.1e-3, beta 0.74) and taking Hs = 4·h_rms gives 0.02133.
_PM_HS_COEFF = 0.0214
#: Wind scaling from the 10 m observation height to the 19.5 m
#: Pierson-Moskowitz reference height: the inverse of the approximate ratio
#: v10/v20 ≈ 0.94 (Dobson 1981) of Ainslie, *Principles of Sonar Performance
#: Modelling*, §8.1.1.1.2 Eq. (8.12), whose 20 m is the PM measurement height
#: of 19.5 m rounded (§4.3.2.1, footnote 23). A neutral log profile over a
#: plausible sea roughness gives 1.05-1.06, close to it; the ratio depends on
#: the air-sea stability (Ainslie Fig. 4.19), so it is a neutral-stability
#: value, and a wave height built from a 10 m wind is approximate.
_U10_TO_U195 = 1.0 / 0.94


def _pm_wind_ms(hs):
    """The Pierson-Moskowitz wind (m/s, 19.5 m) of significant wave height
    ``hs`` (m): ``U = √(Hs / _PM_HS_COEFF)``, a negative ``hs`` taken as 0."""
    return float(np.sqrt(max(float(hs), 0.0) / _PM_HS_COEFF))


def hs_to_pm_wind(hs):
    """Pierson-Moskowitz wind (knots, at 19.5 m) that raises wave height ``hs``.

    Inverts the fully developed ``Hs = 0.0214·U²`` (``U`` in m/s at the 19.5 m
    PM reference height, not a 10 m wind) and returns ``U`` in knots, the unit
    of every wind in uacpy and of
    :func:`~uacpy.core.altimetry.generate_sea_surface`'s ``wind_speed_19_5m_kn``, so
    the answer feeds it directly::

        from uacpy.core.altimetry import generate_sea_surface
        wind_kn = uacpy.data.hs_to_pm_wind(2.0)            # 18.8 kn
        pairs = generate_sea_surface(20000.0, wind_kn)     # Hs ≈ 2.0 m

    Parameters
    ----------
    hs : float
        Significant wave height (m); a negative value counts as 0.

    Returns
    -------
    float
        The PM reference wind in knots.
    """
    return float(ms_to_knots(_pm_wind_ms(hs)))


@one_provenance_notice(subject="the surface's data",
                       record='uacpy.data.citations(surface)')
def fetch_sea_surface(point, *, date, rmax, n_points=None, rng=None,
                      source='auto', max_days=None, timeout=120.0,
                      verbose=False, max_distance_km=None):
    """Sea-surface altimetry and the provenance of the sea state it came from.

    Parameters
    ----------
    point : (lat, lon)
        Site coordinates in decimal degrees.
    date : str or datetime.date
        Calendar date of the sea state (required; sea state is time-specific).
    rmax : float
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
        Time tolerance forwarded to the wave source (only WAVERYS reads it);
        set when a wind source answers, it gives a ``FallbackWarning`` that
        it was not used.
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
        The realization (``.ranges`` 0 → ``rmax``, ``.heights`` in m),
        carrying the sea state's provenance like every fetched carrier:
        ``.data_sources`` is one :class:`~uacpy.data.DataProvenance` whose
        ``source.id`` is the catalogue id that supplied it (``'waverys'`` /
        ``'ww3'`` / ``'nbs'``). Its ``data_date`` separates the backends that
        share an id: the cached NBS climatology reads ``'month MM, <period>'``
        (the period the cache recorded); a live fetch records the time step
        it read (the live NBS and WaveWatch III fields are read at 00:00 UTC
        of ``date``), beside the ``requested_date``.
    """
    as_coordinate(point)                       # validate before any request
    limit = checked_max_distance(max_distance_km, 'fetch_sea_surface')
    if max_days is not None:
        max_days = checked_max_days(max_days, 'fetch_sea_surface')
    request = dict(date=date, max_days=max_days, timeout=timeout,
                   verbose=verbose, max_distance_km=limit, rmax=rmax,
                   n_points=n_points, rng=rng)
    surface, (answered, _backend) = _SEA_SURFACE_CHAIN.answer(
        source,
        lambda src, backend: _SEA_SURFACE_CHAIN.fetch(
            src, backend, (point,), request, who='fetch_sea_surface',
            keyword='source='),
        who='fetch_sea_surface', keyword='source=')
    if answered in WIND_SOURCES:
        unused_setting_notices('fetch_sea_surface', [(
            'max_days', max_days,
            "the NBS wind answered, which takes no time tolerance (only "
            "WAVERYS reads max_days)")])
    return surface


def _wave_surface(backend, point, *, date, max_days, timeout, verbose,
                  max_distance_km, rmax, n_points, rng, **_request):
    """A :data:`_SEA_SURFACE_CHAIN` fetch from a wave source (``backend`` is
    its :func:`~uacpy.data.fetch_waves` source): the realization of the
    Pierson-Moskowitz wind of the significant wave height read."""
    from uacpy.data.waves import fetch_waves
    w = fetch_waves(point, date=date, source=backend, max_days=max_days,
                    timeout=timeout, verbose=verbose,
                    max_distance_km=max_distance_km)
    u = _pm_wind_ms(w.hs)
    log_message('waves', f"Hs {w.hs:.2f} m ({w.provenance.source.id}) → "
                f"PM wind {float(ms_to_knots(u)):.1f} kn", verbose=verbose)
    return _altimetry(_surface(rmax, u, n_points, rng), w.provenance)


def _wind_surface(backend, point, *, date, timeout, verbose, max_distance_km,
                  rmax, n_points, rng, **_request):
    """A :data:`_SEA_SURFACE_CHAIN` fetch from the NBS wind (``backend`` is
    the live ``'nbs'`` or the cached ``'local'`` climatology): the
    realization of the 10 m wind scaled to the 19.5 m reference height."""
    from uacpy.data.wind_live import wind_at
    knots, wind_prov = wind_at(point, date=date, source=backend,
                               timeout=timeout, verbose=verbose,
                               max_distance_km=max_distance_km,
                               who='fetch_sea_surface')
    u = float(knots_to_ms(knots)) * _U10_TO_U195
    kind = 'climatology' if backend == 'local' else 'live'
    log_message('wind', f"U10 {knots:.1f} kn (nbs {kind}) → PM wind "
                f"{float(ms_to_knots(u)):.1f} kn", verbose=verbose)
    # The wind reading's own record: the 0.25° cell read and the live step
    # (or the cached climatology's month and reference period), through the
    # offset rule already.
    return _altimetry(_surface(rmax, u, n_points, rng), wind_prov)


#: The sea-surface sources: the wave heights, then the live 10 m wind, then
#: the cached NBS monthly climatology (``install.sh --data wind``) — a mean
#: state, not the day's, so it is the last source of ``'auto'`` rather than
#: the cache-first twin the other axes try first: for sea state the
#: date-specific product is the better answer. ``'local'`` (the cached
#: sources alone) is that climatology.
_SEA_SURFACE_CHAIN = SourceChain(
    'sea surface',
    providers=(SourceProvider('waverys', ('waverys',), _wave_surface),
               SourceProvider('ww3', ('ww3',), _wave_surface),
               SourceProvider('nbs', ('nbs',), _wind_surface),
               SourceProvider('local', ('local',), _wind_surface)),
    auto=('waverys', 'ww3', 'nbs', 'local'))


def _altimetry(pairs, prov):
    """``(n, 2)`` ``[range, height]`` pairs → an ``Altimetry`` stamped
    with ``prov``."""
    pairs = np.asarray(pairs, dtype=float)
    return Altimetry(ranges=pairs[:, 0], heights=pairs[:, 1],
                     data_sources=(prov,))


def _surface(rmax, wind_ms, n_points, rng):
    # A calm sea (near-zero wind / wave) would trip generate_sea_surface's
    # positive-wind guard; floor it to the calm threshold, which the core
    # sizing rule answers with its fixed count and no warning — the
    # realization stands in for a flat surface either way. Sizing, cap and
    # the under-resolution warning are generate_sea_surface's.
    wind_ms = max(wind_ms, SEA_SURFACE_CALM_WIND)
    return generate_sea_surface(rmax, ms_to_knots(wind_ms), n_points=n_points,
                                rng=rng)
