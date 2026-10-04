"""Catalogue of the external data sources, their licences and citations.

The record types and the catalogue are defined in
:mod:`uacpy.core.provenance`, beside the carriers that hold them, and are
these same objects here; this module adds :func:`citations`.

Two levels, one renderer — deliberately not three parallel classes:

* :class:`DataSource` — a **catalogue entry**: one immutable record per dataset
  (GEBCO, WOA23, …) holding its identity, licence, **attribution and citation
  text**. (So "citation" is a *field* here, not its own class.) :data:`SOURCES`
  is the single source of truth; the README licensing table mirrors it. One
  ``DataSource`` is shared by every fetch that used that dataset.
* :class:`DataProvenance` — one **fetch instance**: references a ``DataSource``
  (``.source``) and adds the *actual* date/coordinates that fetch returned. The
  two levels stay distinct — read the dataset's identity/licence/citation
  through ``prov.source`` and this fetch's specifics off ``prov`` directly.

Carriers and environments carry provenance **uniformly as a tuple of
``DataProvenance``** (``carrier.data_sources`` / ``env.data_sources``); an
un-stamped/literal layer is a bare ``DataProvenance(source=…)`` with no
date/coords. :func:`uacpy.data.fetch_environment` aggregates the union across a
built env's carriers; :func:`citations` renders the licence/attribution/citation
(plus the fetched date/coords when present) for an environment, a carrier, a
list of ids/sources, or the whole catalogue.
"""

from typing import List

from uacpy.core._export import require_extra
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.provenance import (DataProvenance, DataSource, SOURCES,
                                   fetch_summary_line)

__all__ = ['DataSource', 'DataProvenance', 'SOURCES', 'citations',
           'provenance_table']


def _resolve(obj) -> List[DataProvenance]:
    """Normalise any accepted input to a uniform list of :class:`DataProvenance`.

    A bare :class:`DataSource` or a catalogue id is wrapped in a
    ``DataProvenance`` with no fetch specifics, so every consumer reads the
    same type (``.source`` for the dataset, the rest for the fetch)."""
    def wrap(x):
        if isinstance(x, DataProvenance):
            return x
        if isinstance(x, DataSource):
            return DataProvenance(source=x)
        if x in SOURCES:
            return DataProvenance(source=SOURCES[x])
        raise ConfigurationError(
            f"citations: unknown source id {x!r}; valid ids: {sorted(SOURCES)}."
        )

    if obj is None:
        return [DataProvenance(source=s) for s in SOURCES.values()]
    if hasattr(obj, 'data_sources'):       # an Environment / carrier
        return list(obj.data_sources)
    if isinstance(obj, (str, DataSource, DataProvenance)):  # a single id/source
        obj = [obj]
    return [wrap(s) for s in obj]


def citations(obj=None) -> str:
    """Render the required attribution / citation text.

    Parameters
    ----------
    obj : optional
        ``None`` → the whole catalogue; an ``Environment`` / carrier → the
        provenance it carries (``.data_sources``); or an iterable of source
        ids / :class:`DataSource` / :class:`DataProvenance`.

    Returns
    -------
    str
        One block per source (name, licence, attribution, citation, plus what
        was fetched when known), with a flag where commercial use is not
        confirmed. A source read at several points — the columns of a
        transect — gets one block whose ``Fetched:`` line gives the number of
        points, the span of their offsets and where the largest one is.
    """
    by_source = {}
    for prov in _resolve(obj):
        by_source.setdefault(prov.source.id, []).append(prov)
    blocks = []
    for records in by_source.values():
        src = records[0].source
        lines = [f"{src.name}  [{src.license}]",
                 f"  Attribution: {src.attribution}",
                 f"  Cite:        {src.citation}"]
        fetched = fetch_summary_line(records)
        if fetched is not None:
            lines.append(fetched)
        if not src.commercial_use:
            lines.append("  ⚠ Commercial use not confirmed — verify licence terms")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def provenance_table(*objs):
    """Every dataset and engine behind ``objs`` as a ``pandas.DataFrame``
    (optional extra ``uacpy[xarray]``), one row each.

    Parameters
    ----------
    *objs
        Anything :func:`citations` takes — an ``Environment`` or carrier
        (its ``.data_sources``), source ids, :class:`DataSource` or
        :class:`DataProvenance` records — and results: a result adds a row
        for the engine that produced it (its ``model_source``). Pass the
        environment and the result together for data and engine in one
        table. No argument tabulates the whole data catalogue.

    Returns
    -------
    pandas.DataFrame
        Columns ``kind`` (``'data'`` or ``'engine'``), ``id``, ``name``,
        ``license``, ``commercial_use``, ``citation``, then the fetch's
        ``product``, ``data_date``, ``requested_date``, ``data_point``,
        ``requested_point``, ``offset_km``, ``point_kind``,
        ``cell_size_deg``, ``from_neighbour_cell`` and ``range_m`` (``None``
        for an engine). A transect has one row per column read.
    """
    pandas = require_extra('pandas', 'provenance_table')
    rows = []
    for obj in (objs or (None,)):
        model_source = getattr(obj, 'model_source', None)
        if model_source is not None:
            rows.append({
                'kind': 'engine', 'id': model_source.id,
                'name': model_source.name, 'license': model_source.license,
                'commercial_use': model_source.commercial_use,
                'citation': model_source.citation_for(obj.model),
                'product': None, 'data_date': None, 'requested_date': None,
                'data_point': None, 'requested_point': None,
                'offset_km': None, 'point_kind': None, 'cell_size_deg': None,
                'from_neighbour_cell': None, 'range_m': None})
            continue
        for prov in _resolve(obj):
            src = prov.source
            rows.append({
                'kind': 'data', 'id': src.id, 'name': src.name,
                'license': src.license,
                'commercial_use': src.commercial_use,
                'citation': src.citation, 'product': prov.product,
                'data_date': prov.data_date,
                'requested_date': prov.requested_date,
                'data_point': prov.data_point,
                'requested_point': prov.requested_point,
                'offset_km': prov.offset_km,
                'point_kind': prov.point_kind,
                'cell_size_deg': prov.cell_size_deg,
                'from_neighbour_cell': prov.from_neighbour_cell,
                'range_m': prov.range_m})
    return pandas.DataFrame(rows)

