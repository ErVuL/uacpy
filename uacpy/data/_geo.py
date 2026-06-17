"""Small shared geographic helpers for the data layer."""

from typing import Tuple

from uacpy.core.exceptions import ConfigurationError

__all__ = ['Coordinate', 'as_coordinate', 'normalize_lon']

Coordinate = Tuple[float, float]


def as_coordinate(point) -> Coordinate:
    """Validate and unpack a ``(lat, lon)`` coordinate pair as floats.

    The data layer takes a single ``point`` tuple everywhere, so this is the
    shared guard against the easy mistake of passing two bare scalars or a
    single number — it turns a cryptic unpack ``TypeError`` into a typed,
    actionable :class:`ConfigurationError`.
    """
    try:
        lat, lon = point
        return float(lat), float(lon)
    except (TypeError, ValueError):
        raise ConfigurationError(
            f"expected a (lat, lon) coordinate pair; got {point!r}.",
            remediation="Pass a 2-tuple of degrees, e.g. (43.2, 7.5).",
        ) from None


def normalize_lon(lon: float) -> float:
    """Wrap a longitude (degrees) into ``[-180, 180)``.

    Callers may pass longitude in either ``[-180, 180]`` or ``[0, 360]``;
    every source normalizes through here so the same physical point yields the
    same result regardless of convention (and dateline values stay in range).
    """
    return ((float(lon) + 180.0) % 360.0) - 180.0
