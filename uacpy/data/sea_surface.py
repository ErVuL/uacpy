"""Fetched sea-state → a Pierson-Moskowitz sea-surface altimetry realization.

Turns a fetched sea state into the ``(range, height)`` altimetry array that
``Environment(altimetry=...)`` consumes for rough-surface scattering. The
observed significant wave height ``Hs`` is inverted to the effective
Pierson-Moskowitz wind ``U = √(Hs / 0.021)`` and handed to
:func:`uacpy.core.ssp.generate_sea_surface`, so the realization reproduces the
observed ``Hs`` regardless of whether the sea is fully developed. When no wave
source is available it falls back to the fetched 10 m wind (treated as the PM
wind — a fully-developed assumption).
"""

import numpy as np

from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.core.ssp import generate_sea_surface
from uacpy._log import log_message
from uacpy.data._geo import as_coordinate

__all__ = ['fetch_sea_surface', 'hs_to_pm_wind', 'SEA_SURFACE_SOURCES']

SEA_SURFACE_SOURCES = ('waves', 'wind', 'auto')
#: Pierson-Moskowitz fully-developed significant wave height Hs = 0.021·U²
#: (U at the 19.5 m reference height), the relation used by
#: :func:`generate_sea_surface`.
_PM_HS_COEFF = 0.021


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
    source : {'auto', 'waves', 'wind'}, optional
        ``'waves'`` builds from the fetched significant wave height; ``'wind'``
        from the fetched 10 m wind (fully-developed assumption); ``'auto'``
        (default) tries waves, then wind.
    max_days : int, optional
        Time tolerance forwarded to the wave source.

    Returns
    -------
    (altimetry, source_id)
        ``altimetry`` is an ``(n_points, 2)`` ``[range_m, height_m]`` array;
        ``source_id`` is the catalogue id that supplied the sea state
        (``'waverys'`` / ``'ww3'`` / ``'nbs'``).
    """
    lat, lon = as_coordinate(point)
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
    try:
        from uacpy.data.wind_live import fetch_wind
        u = fetch_wind(point, date=date, timeout=timeout, verbose=verbose)
        log_message('wind', f"U10 {u:.1f} m/s (nbs) → sea surface",
                    verbose=verbose)
        return _surface(max_range, u, n_points, seed), 'nbs'
    except (DataFetchError, ConfigurationError) as exc:
        errors.append(exc)
    data_errs = [e for e in errors if isinstance(e, DataFetchError)]
    raise (data_errs[0] if data_errs else errors[-1])


def _surface(max_range, wind_ms, n_points, seed):
    # A calm sea (near-zero wind / wave) would trip generate_sea_surface's
    # positive-wind guard; floor it to a light breeze so a flat-ish surface is
    # still returned rather than raising.
    return generate_sea_surface(max_range, max(wind_ms, 0.5), n_points=n_points,
                                seed=seed)
