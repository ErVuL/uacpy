"""How far, in space and in time, a fetched value may stand from the point and
the date asked for before uacpy says so: the offset rule's thresholds, in one
place.

Distances are judged by :func:`uacpy.data._geo.checked_offset`, dates by
:func:`checked_date_offset`. Past a threshold the value is returned with a
``ProvenanceWarning`` naming the cause; only ``max_distance_km`` /
``max_days`` refuse. A gridded source's distance threshold is its own cell
(:func:`uacpy.data._geo.cell_half_diagonal_km`), not a number here.
"""

import warnings
from typing import Optional

from uacpy.core.exceptions import ProvenanceWarning
from uacpy.core._warn_frames import USER_FRAME_SKIP

__all__ = ['ARGO_OFFSET_WARN_KM', 'SAMPLE_OFFSET_WARN_KM',
           'MARS_OFFSET_WARN_KM', 'ARGO_OFFSET_WARN_DAYS',
           'COPERNICUS_DAILY_OFFSET_WARN_DAYS', 'date_offset_breach',
           'checked_date_offset']

#: Past this distance (km) a returned cast warns (the offset rule): uacpy's
#: stated policy for "far" for a single in-situ profile, not a physical bound.
ARGO_OFFSET_WARN_KM = 50.0
#: Past this distance (km) a returned sample warns (the offset rule): uacpy's
#: stated policy for "far" for a seabed sample, whose type varies on km
#: scales, not a physical bound. ``max_distance_km`` refuses.
SAMPLE_OFFSET_WARN_KM = 10.0
#: Past this distance (km) a returned sample warns (the offset rule): uacpy's
#: stated policy for "far" for a seabed sample, as for the grain-size DB.
MARS_OFFSET_WARN_KM = 10.0

#: Past this many days from the requested date an Argo cast warns: one cast
#: is a single in-situ snapshot of a field that decorrelates within weeks.
ARGO_OFFSET_WARN_DAYS = 5
#: Past this many days a Copernicus daily (or finer) field warns: a daily
#: model mean is smooth, but a few days is already another synoptic state.
#: A monthly field warns instead when it is another calendar month.
COPERNICUS_DAILY_OFFSET_WARN_DAYS = 3

#: The sources whose dated fields follow the Copernicus rule.
_COPERNICUS_SOURCES = ('copernicus', 'copernicus_bgc', 'waverys')


def _is_monthly(product: Optional[str]) -> bool:
    """Whether a Copernicus dataset id is a monthly-mean product."""
    return product is not None and 'P1M' in product


def date_offset_breach(prov) -> Optional[str]:
    """Why ``prov``'s date stands too far from the requested one for its
    source, or ``None`` when it does not (or either date is unknown)."""
    days = prov.offset_days
    if days is None:
        return None
    source = prov.source.id
    if source == 'argo' and abs(days) > ARGO_OFFSET_WARN_DAYS:
        return (f"beyond the {ARGO_OFFSET_WARN_DAYS} days an Argo cast "
                f"stands for")
    if source in _COPERNICUS_SOURCES:
        if _is_monthly(prov.product):
            if (str(prov.data_date)[:7] != str(prov.requested_date)[:7]):
                return ("a monthly mean of another calendar month than the "
                        "one requested")
            return None
        if abs(days) > COPERNICUS_DAILY_OFFSET_WARN_DAYS:
            return (f"beyond the {COPERNICUS_DAILY_OFFSET_WARN_DAYS} days a "
                    f"daily field stands for")
    return None


def checked_date_offset(prov, *, who: str):
    """``prov`` once the date rule has passed it: a ``ProvenanceWarning``
    naming the cause when :func:`date_offset_breach` finds one. The refusal
    past ``max_days`` is each fetcher's own, applied where it picks the date
    (Argo's query window, Copernicus's nearest time step)."""
    why = date_offset_breach(prov)
    if why is not None:
        days = abs(prov.offset_days)
        warnings.warn(
            f"{who}: {prov.source.name}: the data are dated {prov.data_date}, "
            f"{days} day{'s' if days != 1 else ''} from the requested "
            f"{prov.requested_date}: {why}. Pass max_days= to refuse data "
            f"that far in time.",
            ProvenanceWarning, skip_file_prefixes=USER_FRAME_SKIP)
    return prov
