"""Fetched sea-state → a Pierson-Moskowitz sea-surface altimetry realization.

Turns a fetched sea state into the ``(range, height)`` altimetry array that
``Environment(altimetry=...)`` consumes for rough-surface scattering. The
observed significant wave height ``Hs`` is inverted to the effective
Pierson-Moskowitz wind ``U = √(Hs / 0.0214)`` (see :data:`_PM_HS_COEFF` for
the sources) and handed to
:func:`uacpy.core.ssp.generate_sea_surface`, so the realization reproduces the
observed ``Hs`` regardless of whether the sea is fully developed. When no wave
source is available it falls back to the fetched 10 m wind, scaled to the
19.5 m PM reference height (a fully-developed assumption) — the live NBS field
first, then the cached NBS monthly climatology.
"""

import numpy as np

from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.core.ssp import generate_sea_surface
from uacpy._log import log_message
from uacpy.data._geo import as_coordinate
from uacpy.data._http import raise_substantive

__all__ = ['fetch_sea_surface', 'hs_to_pm_wind', 'SEA_SURFACE_SOURCES']

SEA_SURFACE_SOURCES = ('waves', 'wind', 'local', 'auto')
#: Wind backends tried per source token. ``'local'`` is the cached NBS monthly
#: climatology (``install.sh --data wind``) — a mean state, not the day's, so
#: it is the last rung of ``'auto'`` rather than the cache-first rung the other
#: axes use: for sea state the date-specific product is the better answer.
_WIND_BACKENDS = {'wind': ('erddap',), 'local': ('local',),
                  'auto': ('erddap', 'local')}
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
    """Effective Pierson-Moskowitz wind speed (m/s) reproducing wave height ``hs``."""
    return float(np.sqrt(max(float(hs), 0.0) / _PM_HS_COEFF))


def fetch_sea_surface(point, *, date, max_range, n_points=500, seed=None,
                      source='auto', max_days=None, timeout=120.0,
                      verbose=False):
    """Sea-surface altimetry ``(n_points, 2)`` from the fetched sea state.

    Parameters
    ----------
    point : (lat, lon)
        Site coordinates in decimal degrees.
    date : str or datetime.date
        Calendar date of the sea state (required; sea state is time-specific).
    max_range : float
        Range extent (m) of the realization — set to the transect length.
    n_points : int, optional
        Range samples in the returned altimetry. Default 500.
    seed : int, optional
        Random seed for the surface realization (reproducibility).
    source : {'auto', 'waves', 'wind', 'local'}, optional
        ``'waves'`` builds from the fetched significant wave height; ``'wind'``
        from the live 10 m wind (fully-developed assumption); ``'local'`` from
        the cached NBS monthly wind climatology (no network — a mean state, so
        it understates day-to-day sea state); ``'auto'`` (default) tries waves,
        then live wind, then the climatology.
    max_days : int, optional
        Time tolerance forwarded to the wave source.

    Returns
    -------
    (altimetry, source_id)
        ``altimetry`` is an ``(n_points, 2)`` ``[range_m, height_m]`` array;
        ``source_id`` is the catalogue id that supplied the sea state
        (``'waverys'`` / ``'ww3'`` / ``'nbs'``).
    """
    as_coordinate(point)                       # validate before any request
    if source not in SEA_SURFACE_SOURCES:
        raise ConfigurationError(
            f"fetch_sea_surface: unknown source {source!r}.",
            remediation=f"Use one of {sorted(SEA_SURFACE_SOURCES)}.",
        )
    errors = []
    if source in ('waves', 'auto'):
        try:
            from uacpy.data.waves import fetch_waves
            w = fetch_waves(point, date=date, max_days=max_days,
                            timeout=timeout, verbose=verbose)
            u = hs_to_pm_wind(w['hs'])
            log_message('waves', f"Hs {w['hs']:.2f} m ({w['source']}) → "
                        f"PM wind {u:.1f} m/s", verbose=verbose)
            return _surface(max_range, u, n_points, seed), w['source']
        except (DataFetchError, ConfigurationError) as exc:
            if source == 'waves':
                raise
            errors.append(exc)
    # Wind-driven (explicit, or the 'auto' fallback when waves are unavailable).
    from uacpy.data.wind_live import fetch_wind
    for backend in _WIND_BACKENDS[source]:
        try:
            u10 = fetch_wind(point, date=date, source=backend, timeout=timeout,
                             verbose=verbose)
        except (DataFetchError, ConfigurationError) as exc:
            errors.append(exc)
            continue
        u = u10 * _U10_TO_U195
        log_message('wind', f"U10 {u10:.1f} m/s (nbs {backend}) → PM wind "
                    f"{u:.1f} m/s", verbose=verbose)
        return _surface(max_range, u, n_points, seed), 'nbs'
    raise_substantive(errors)


def _surface(max_range, wind_ms, n_points, seed):
    # A calm sea (near-zero wind / wave) would trip generate_sea_surface's
    # positive-wind guard; floor it to a light breeze so a flat-ish surface is
    # still returned rather than raising.
    return generate_sea_surface(max_range, max(wind_ms, 0.5), n_points=n_points,
                                seed=seed)
