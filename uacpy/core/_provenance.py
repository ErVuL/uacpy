"""Provenance on the carriers: a ``data_sources`` argument checked into a
tuple of records, and the records of several carriers merged.
"""

import dataclasses

from uacpy.core.exceptions import ConfigurationError


def coerce_data_sources(value, label: str) -> tuple:
    """Validate and freeze a carrier's ``data_sources`` into a tuple of
    provenance records, enforcing the harmonised invariant that every element
    is a :class:`~uacpy.core.provenance.DataProvenance` (carries a ``.source``).

    Duck-typed so ``core`` keeps no import dependency on ``data``: a record is
    accepted iff it exposes ``.source`` with an ``.id``. A bare ``DataSource``
    (no ``.source``) or any other object is rejected with a typed error rather
    than leaking downstream to crash ``env.data_sources`` aggregation or
    ``citations()`` on ``r.source.id``. ``None`` means no provenance and
    coerces to ``()``.
    """
    if value is None:
        return ()
    records = tuple(value)
    for r in records:
        if not (hasattr(r, 'source') and hasattr(getattr(r, 'source'), 'id')):
            raise ConfigurationError(
                f"{label}: data_sources elements must be DataProvenance "
                f"records (each carrying a .source); got {type(r).__name__}. "
                f"Wrap a catalogue DataSource via "
                f"uacpy.data.DataProvenance(source=...)."
            )
    return records


def dedupe_records(records) -> tuple:
    """``records`` with exact repeats removed, kept in first-seen order.

    A repeat is the same source read at the same point for the same request
    (the WOA23 cell an SSP and the absorption's T/S row both read); two
    columns of a transect reading different cells, or the same cell for
    different requested points, are two records, so every column's offset
    survives. A bare record (the source alone) is dropped where another
    record of its source says what was read."""
    seen, out = set(), []
    for record in records:
        if record not in seen:
            seen.add(record)
            out.append(record)
    # A bare record (the catalogue entry alone, nothing fetched) adds only
    # the attribution, which a record of the same source that says what was
    # read already carries.
    described = {r.source.id for r in out if not _is_bare(r)}
    return tuple(r for r in out
                 if not (_is_bare(r) and r.source.id in described))


def _is_bare(record) -> bool:
    """Whether ``record`` names its source and nothing about the fetch."""
    return all(getattr(record, name, None) is None for name in (
        'data_date', 'data_point', 'requested_point', 'requested_date',
        'product'))


def dedupe_provenance(carriers, ranges=None) -> tuple:
    """Union of ``carrier.data_sources`` over ``carriers`` with exact repeats
    removed (:func:`dedupe_records`), in first-seen order.

    The single home for the aggregation ``Bottom`` (over its columns),
    ``Surface`` (over its nodes) and ``Environment`` (over its carriers)
    each expose. ``ranges``, one along-track range (m) per carrier, stamps
    each record that has no ``range_m`` with its carrier's, so the records of
    a range-dependent carrier say which column they came from. A carrier
    that is ``None``, or carries no ``data_sources``, contributes nothing.
    """
    def stamped(record, r):
        if r is None or getattr(record, 'range_m', 0.0) is not None:
            return record
        return dataclasses.replace(record, range_m=float(r))

    rs = [None] * len(carriers) if ranges is None else list(ranges)
    return dedupe_records(
        stamped(record, r) for carrier, r in zip(carriers, rs)
        for record in getattr(carrier, 'data_sources', ()) or ())
