"""Ray/eigenray and arrival result types."""

from __future__ import annotations

import json
import operator
import warnings
import numpy as np
from typing import (Optional, Dict, Any, List, Tuple, TYPE_CHECKING,
                    Union)

from uacpy.core.exceptions import ConfigurationError, FallbackWarning
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core._validate import require_positive
from uacpy.core.acoustics.ray_geometry import polyline_miss_distance
from uacpy.core._export import (FrozenDict, FrozenList, _from_json,
                                _to_json, read_only)

from uacpy.core.results._base import (PhaseReference, Result, _count,
                                      _window_pair, axis_match_tolerance,
                                      coordinate_axis)
from uacpy.core.results.quantities import coordinate_unit
from uacpy.core._repr import count


if TYPE_CHECKING:                      # the runtime imports are deferred
    from uacpy.acoustic_signal.delay_profile import ChannelRegime
    from uacpy.comms.channel import ChannelTaps
    from uacpy.core.results.field import Field


def _arrival_kind(n_top: int, n_bot: int) -> str:
    if n_top >= 1 and n_bot >= 1:
        return 'both'
    if n_bot >= 1:
        return 'bottom'
    if n_top >= 1:
        return 'surface'
    return 'direct'


_BOUNCE_KINDS = ('direct', 'surface', 'bottom', 'both')


def _bounce_in_bounds(value: int, spec) -> bool:
    """Match ``value`` against an integer (exact; a numpy integer, such as
    a bounce column's own ``max()``, included) or ``(lo, hi)`` tuple
    (closed range, ``None`` = unbounded). Shared with :class:`Rays`."""
    if spec is None:
        return True
    try:
        return value == operator.index(spec)
    except TypeError:
        pass
    try:
        lo, hi = spec
    except (TypeError, ValueError):
        raise ConfigurationError(
            f"bounce filter: a bounce count is an integer or a (lo, hi) "
            f"pair, either end None; got {spec!r}.") from None
    if lo is not None and value < lo:
        return False
    if hi is not None and value > hi:
        return False
    return True


def _bounce_predicate(kind, top, bot):
    """Build a predicate ``(n_top, n_bot) -> bool`` for bounce filtering.

    Shared by :class:`Arrivals` and :class:`Rays`. ``kind`` is one of
    :data:`_BOUNCE_KINDS` (or ``None``); ``top`` / ``bot`` are
    int / (lo, hi) / None specs matching :func:`_bounce_in_bounds`.
    """
    if kind is not None and kind not in _BOUNCE_KINDS:
        raise ConfigurationError(
            f"bounce filter: kind={kind!r} not in {_BOUNCE_KINDS}."
        )

    def predicate(n_top: int, n_bot: int) -> bool:
        if kind is not None and _arrival_kind(n_top, n_bot) != kind:
            return False
        return (_bounce_in_bounds(n_top, top)
                and _bounce_in_bounds(n_bot, bot))

    return predicate


#: The per-arrival table of :class:`Arrivals`: record key -> (column name,
#: dtype). The eight ``.arr`` columns keep the dtype ``read_arr_file`` gives
#: its cells, so the derived :attr:`Arrivals.by_receiver` cells are the
#: reader's, dtype for dtype; the cell indices are integers.
_ARRIVAL_COLUMNS = {
    'delay': ('delays', 'float64'), 'delay_imag': ('delays_imag', 'float64'),
    'amplitude': ('amplitudes', 'float64'), 'phase': ('phases', 'float64'),
    'n_top_bounces': ('n_top_bounces', 'int32'),
    'n_bot_bounces': ('n_bot_bounces', 'int32'),
    'source_angle': ('source_angles_deg', 'float64'),
    'receiver_angle': ('receiver_angles_deg', 'float64'),
    'src_idx': ('src_idx', 'int64'), 'depth_idx': ('depth_idx', 'int64'),
    'range_idx': ('range_idx', 'int64'),
}

#: The keys of one ``read_arr_file`` cell, in the reader's order.
_CELL_KEYS = ('amplitudes', 'phases', 'delays', 'delays_imag', 'source_angles_deg',
              'receiver_angles_deg', 'n_top_bounces', 'n_bot_bounces')

#: A record key a cell may omit, and the value an arrival takes then.
_CELL_DEFAULTS = {'delays_imag': 0.0, 'source_angles_deg': 0.0, 'receiver_angles_deg': 0.0,
                  'n_top_bounces': 0, 'n_bot_bounces': 0}


def _columns_from_records(records) -> Dict[str, np.ndarray]:
    """The table of a record list: one column per :data:`_ARRIVAL_COLUMNS`
    key every record carries (``kind`` is derived from the bounce counts,
    so it is not stored)."""
    records = list(records)
    columns = {}
    for key, (name, dtype) in _ARRIVAL_COLUMNS.items():
        if records and all(key in a for a in records):
            columns[name] = np.asarray([a[key] for a in records],
                                       dtype=dtype)
    return columns


def _grid_shape_of(by_receiver):
    """``(n_src, n_depth, n_range)`` of a nested receiver grid, or ``None``."""
    if by_receiver is None:
        return None
    return (len(by_receiver),
            len(by_receiver[0]) if by_receiver else 0,
            len(by_receiver[0][0]) if by_receiver and by_receiver[0] else 0)


def _columns_from_cells(by_receiver):
    """``(table, grid_shape)`` of Bellhop's ``[src][depth][range] -> cell``
    nesting, the cells in source / depth / range order and each cell's
    arrivals in its own order. A cell without arrivals adds no row."""
    shape = _grid_shape_of(by_receiver)
    parts = {name: [] for name in (*_CELL_KEYS, 'src_idx', 'depth_idx',
                                   'range_idx')}
    for s_idx, by_src in enumerate(by_receiver if isinstance(by_receiver, list) else []):
        for d_idx, by_depth in enumerate(by_src if isinstance(by_src, list) else []):
            for r_idx, cell in enumerate(by_depth if isinstance(by_depth, list) else []):
                if not isinstance(cell, dict):
                    continue
                delays = np.asarray(cell.get('delays', []))
                n = len(delays)
                if n == 0:
                    continue
                for name in _CELL_KEYS:
                    if name in cell:
                        parts[name].append(np.asarray(cell[name]))
                    else:
                        fill = (np.zeros_like(delays) if name in ('amplitudes', 'phases')
                                else np.full(n, _CELL_DEFAULTS[name]))
                        parts[name].append(fill)
                parts['src_idx'].append(np.full(n, s_idx))
                parts['depth_idx'].append(np.full(n, d_idx))
                parts['range_idx'].append(np.full(n, r_idx))
    dtypes = dict(_ARRIVAL_COLUMNS.values())
    columns = {name: (np.concatenate(chunks).astype(dtypes[name], copy=False)
                      if chunks else np.zeros(0, dtype=dtypes[name]))
               for name, chunks in parts.items()}
    return columns, shape


def _records_from_columns(columns) -> List[Dict[str, Any]]:
    """One record per table row, each key a Python scalar: floats, the
    bounce counts and cell indices as ints, and ``kind`` from the counts."""
    keys = [(key, name, dtype) for key, (name, dtype) in _ARRIVAL_COLUMNS.items()
            if name in columns]
    if not keys:
        return []
    lists = {key: np.asarray(columns[name]).tolist() for key, name, _ in keys}
    n = len(next(iter(lists.values())))
    records = []
    for i in range(n):
        record = {}
        for key, _name, dtype in keys:
            value = lists[key][i]
            record[key] = int(value) if dtype.startswith('int') else float(value)
        if 'n_top_bounces' in record and 'n_bot_bounces' in record:
            record['kind'] = _arrival_kind(record['n_top_bounces'],
                                           record['n_bot_bounces'])
        records.append(record)
    return records


def _cells_from_columns(columns, shape):
    """The ``[src][depth][range] -> cell`` nesting of a table on a grid of
    ``shape``, each cell the record ``read_arr_file`` builds: the eight
    columns in their dtypes and ``n_arrivals`` as an int. Rows whose cell
    lies outside the grid are not placed."""
    n_src, n_depth, n_range = shape
    n = len(columns.get('delays', ()))
    src = np.asarray(columns.get('src_idx', np.zeros(n, int)))
    dep = np.asarray(columns.get('depth_idx', np.zeros(n, int)))
    rng = np.asarray(columns.get('range_idx', np.zeros(n, int)))
    dtypes = dict(_ARRIVAL_COLUMNS.values())
    full = {name: (np.asarray(columns[name]) if name in columns
                   else np.full(n, _CELL_DEFAULTS.get(name, 0.0),
                                dtype=dtypes[name]))
            for name in _CELL_KEYS}
    flat = (src * n_depth + dep) * n_range + rng
    inside = (src < n_src) & (dep < n_depth) & (rng < n_range)
    order = np.argsort(np.where(inside, flat, -1), kind='stable')
    bounds = np.searchsorted(np.where(inside, flat, -1)[order],
                             np.arange(n_src * n_depth * n_range + 1))
    cells = []
    for s in range(n_src):
        by_depth = []
        for d in range(n_depth):
            row = []
            for r in range(n_range):
                k = (s * n_depth + d) * n_range + r
                rows = order[bounds[k]:bounds[k + 1]]
                cell = {name: read_only(full[name][rows])
                        for name in _CELL_KEYS}
                cell['n_arrivals'] = int(len(rows))
                row.append(FrozenDict(cell))
            by_depth.append(FrozenList(row))
        cells.append(FrozenList(by_depth))
    return FrozenList(cells)


def _absorption_from(saved):
    """The water absorption ``saved`` names, a law or a measured table as
    the traced environment held it: itself when it is one, the one its
    ``to_dict`` mapping rebuilds when it is that, ``None`` for ``None``."""
    from uacpy.core.absorption import Absorption, AbsorptionCoefficient
    from uacpy.core._export import CarrierExport, _resolve_class
    if saved is None or isinstance(saved,
                                   (Absorption, AbsorptionCoefficient)):
        return saved
    path = saved.get('__class__')
    table = path is not None and issubclass(
        _resolve_class(path, CarrierExport), AbsorptionCoefficient)
    return (AbsorptionCoefficient if table else Absorption).from_dict(saved)


class Arrivals(Result):
    """Ray arrivals from Bellhop — one table, one row per arrival event.

    The table is the single store; :attr:`arrivals` (one record dict per
    arrival), :attr:`by_receiver` (the nested per-receiver cells) and the
    bulk accessors (:attr:`delays`, :attr:`amplitudes`, ...) are read from
    it, and :meth:`to_dataframe` returns it. Each arrival record has:
    ``delay`` (s), ``delay_imag`` (s),
    ``amplitude``, ``phase`` (**radians** — the ``.arr`` reader converts
    the file's degree column once; :attr:`phases` returns it as stored),
    ``n_top_bounces``, ``n_bot_bounces``, ``source_angle``, ``receiver_angle``,
    ``kind`` ('direct' / 'surface' / 'bottom' / 'both'), plus the cell of
    origin (``src_idx``, ``depth_idx``, ``range_idx``) so multi-cell runs
    can be filtered back to one cell if needed.

    ``delay_imag`` is Bellhop's volume-absorption term, the imaginary part
    of the travel time (``ArrMod.f90:118-125`` writes it as its own field).
    :attr:`received_amplitudes` applies it as ``exp(omega * Im tau)`` and
    defaults it to 0 when it is absent, so an object assembled by hand
    without the key carries the LOSSLESS amplitude — frequency-dependently
    too loud, silently.

    :attr:`absorption` is the water-column absorption (a law, or a
    measured table) the ``Im tau`` column was traced with, an attribute of
    the arrivals (a ``metadata`` carrying ``absorption`` is refused).

    Mirrors the :class:`Rays` API surface: filter / chain / sort.
    """

    #: The plotter that draws this result (``.plot()``).
    _plotter = 'plot_arrivals'

    def __init__(
        self,
        *,
        arrivals: Optional[List[Dict[str, Any]]] = None,
        by_receiver: Any = None,
        receiver_depths: np.ndarray,
        receiver_ranges: np.ndarray,
        absorption=None,
        **kwargs,
    ):
        # The law is the arrivals' own attribute; a metadata entry naming
        # it would be a second decider.
        if 'absorption' in (kwargs.get('metadata') or {}):
            raise ConfigurationError(
                "Arrivals: metadata carries 'absorption', which is an "
                "attribute of the arrivals, not metadata.",
                remediation="Pass it as a keyword: Arrivals(..., "
                            "absorption=...).")
        super().__init__(**kwargs)
        self._absorption = absorption
        self.receiver_depths = np.atleast_1d(np.asarray(receiver_depths, dtype=float))
        self.receiver_ranges = np.atleast_1d(np.asarray(receiver_ranges, dtype=float))
        # One table, one row per arrival, is the single store; the record
        # list and the nested ``[src][depth][range] -> cell`` form Bellhop's
        # IO produces are both derived from it.
        if arrivals is not None:
            self._columns = _columns_from_records(arrivals)
            self._grid_shape = _grid_shape_of(by_receiver)
        else:
            self._columns, self._grid_shape = _columns_from_cells(by_receiver)

    @classmethod
    def _from_columns(cls, columns: Dict[str, np.ndarray], grid_shape,
                      **kwargs) -> 'Arrivals':
        """An :class:`Arrivals` holding ``columns`` (as :attr:`_COLUMNS`
        names them) on a receiver grid of ``grid_shape`` (``None`` for a
        list built without one), the other keywords as the constructor's."""
        out = cls(arrivals=[], **kwargs)
        out._columns = {name: np.asarray(values)
                        for name, values in columns.items()}
        out._grid_shape = (None if grid_shape is None
                           else tuple(int(n) for n in grid_shape))
        return out

    def _with_columns(self, columns: Dict[str, np.ndarray],
                      grid_shape=None, **changes) -> 'Arrivals':
        """These arrivals with their table replaced by ``columns`` (and the
        grid by ``grid_shape`` when given), keeping the receiver axes and the
        identity unless ``changes`` names them."""
        kwargs = dict(receiver_depths=self.receiver_depths,
                      receiver_ranges=self.receiver_ranges,
                      absorption=self._absorption,
                      **self.id_kwargs())
        kwargs.update(changes)
        return Arrivals._from_columns(
            columns, self._grid_shape if grid_shape is None else grid_shape,
            **kwargs)

    # The producer's in-place edits of a freshly read table: Bellhop trims
    # its padded range columns, scales a line source to the package level and
    # empties the cells below the seabed before the result is handed out.

    def _record_absorption(self, absorption) -> None:
        """Record ``absorption``, the law the table's ``Im tau`` was
        traced with, as :attr:`absorption`."""
        self._absorption = absorption

    def _trim_ranges(self, lo: int, hi: int) -> None:
        """Keep range cells ``lo`` to ``hi - 1``, renumbered from 0, and
        the receiver ranges with them."""
        index = self._cell_index('range_idx')
        keep = (index >= lo) & (index < hi)
        self._columns = {name: values[keep]
                         for name, values in self._columns.items()}
        if 'range_idx' in self._columns:
            self._columns['range_idx'] = self._columns['range_idx'] - lo
        if self._grid_shape is not None:
            self._grid_shape = (*self._grid_shape[:2], hi - lo)
        self.receiver_ranges = np.asarray(self.receiver_ranges)[lo:hi]

    def _scale_paths(self, amplitude: float, phase: float) -> None:
        """Multiply every amplitude by ``amplitude`` and add ``phase`` (rad)
        to every phase."""
        self._columns['amplitudes'] = (
            np.asarray(self._columns['amplitudes'], dtype=float) * amplitude)
        self._columns['phases'] = (
            np.asarray(self._columns['phases'], dtype=float) + phase)

    def _empty_cells(self, empty, *, by: str) -> None:
        """Drop the arrivals of every cell whose ``by`` index
        (``'depth_idx'`` or ``'range_idx'``) ``empty`` flags."""
        drop = np.asarray(empty, dtype=bool)[self._cell_index(by)]
        self._columns = {name: values[~drop]
                         for name, values in self._columns.items()}

    @property
    def arrivals(self) -> List[Dict[str, Any]]:
        """The arrivals as a list of records, one dict per arrival, built
        from the table: ``delay`` (s), ``delay_imag`` (s), ``amplitude``,
        ``phase`` (rad), ``n_top_bounces``, ``n_bot_bounces``,
        ``source_angle`` / ``receiver_angle`` (deg), ``kind`` and the cell of origin
        ``src_idx`` / ``depth_idx`` / ``range_idx`` — each key the table
        holds. Built on every call and read-only: the list and its records
        refuse an edit, which could change nothing here."""
        return FrozenList(FrozenDict(record) for record in
                          _records_from_columns(self._columns))

    @property
    def by_receiver(self) -> Any:
        """The nested ``[src][depth][range] -> cell`` view, each cell the
        record ``read_arr_file`` builds (``amplitudes``, ``phases``,
        ``delays``, ``delays_imag``, ``source_angles_deg``, ``receiver_angles_deg``,
        ``n_top_bounces``, ``n_bot_bounces`` and the count ``n_arrivals``),
        regrouped from the table on every call and read-only (the nesting,
        the cells and their arrays refuse an edit); ``None`` for arrivals
        built without a receiver grid."""
        if self._grid_shape is None:
            return None
        return _cells_from_columns(self._columns, self._grid_shape)

    def __len__(self) -> int:
        return self.n_arrivals

    def __iter__(self):
        return iter(self.arrivals)

    @property
    def n_arrivals(self) -> int:
        """How many arrivals the table holds."""
        return int(len(self._columns.get('delays', ())))

    def _repr_bits(self) -> list:
        return [count(self.n_arrivals, 'arrival'),
                coordinate_axis('receiver_depth', self.receiver_depths),
                coordinate_axis('receiver_range', self.receiver_ranges)]

    # Per-field bulk views ---------------------------------------------------

    def _column(self, name: str, default=None) -> np.ndarray:
        """Column ``name`` as a read-only view; ``default`` (a fill value)
        for a table that holds no such column."""
        values = self._columns.get(name)
        if values is None:
            values = np.full(self.n_arrivals, default, dtype=float)
        return read_only(values)

    @property
    def delays(self) -> np.ndarray:
        """Travel times (s) of every arrival in the list."""
        return read_only(np.asarray(self._column('delays'), dtype=float))

    @property
    def delays_imag(self) -> np.ndarray:
        """Imaginary travel times (s), Bellhop's volume-absorption term
        (``ArrMod.f90:118-125``); 0 for arrivals built without it."""
        return self._column('delays_imag', 0.0)

    @property
    def amplitudes(self) -> np.ndarray:
        """Amplitudes (linear) of every arrival in the list.

        The GEOMETRIC amplitude, with no volume absorption in it. Use
        :attr:`received_amplitudes` for what each path actually delivers.
        """
        return read_only(np.asarray(self._column('amplitudes'), dtype=float))

    @property
    def n_top_bounces(self) -> np.ndarray:
        """Surface reflections of every arrival."""
        return self._bounce_column('n_top_bounces')

    @property
    def n_bot_bounces(self) -> np.ndarray:
        """Bottom reflections of every arrival."""
        return self._bounce_column('n_bot_bounces')

    def _bounce_column(self, name: str) -> np.ndarray:
        if name not in self._columns:
            raise AttributeError(
                f"Arrivals.{name}: these arrivals carry no '{name}'; an "
                f"Arrivals built by hand without the column.")
        return self._column(name)

    @property
    def kinds(self) -> np.ndarray:
        """The multipath class of every arrival, from its bounce counts:
        ``'direct'``, ``'surface'``, ``'bottom'`` or ``'both'``."""
        top, bot = self.n_top_bounces, self.n_bot_bounces
        return read_only(np.array(
            [_arrival_kind(int(t), int(b)) for t, b in zip(top, bot)],
            dtype=object))

    @property
    def receiver_depth(self) -> np.ndarray:
        """The receiver depth (m) each arrival reaches, from its cell. On
        an irregular grid, whose one depth block pairs ``receiver_depths[i]``
        with ``receiver_ranges[i]`` (``bellhop.f90:202-206``), the depth is
        the range cell's partner."""
        index = self._cell_index('range_idx' if self._paired_grid
                                 else 'depth_idx')
        return read_only(self.receiver_depths[index])

    @property
    def receiver_range(self) -> np.ndarray:
        """The receiver range (m) each arrival reaches, from its cell."""
        return read_only(self.receiver_ranges[self._cell_index('range_idx')])

    def _cell_index(self, name: str) -> np.ndarray:
        index = self._columns.get(name)
        if index is None:
            return np.zeros(self._n_rows, dtype=int)
        return np.asarray(index, dtype=int)

    @property
    def _n_rows(self) -> int:
        """Rows of the table, counted on its first column as
        :attr:`arrivals` counts them: :attr:`n_arrivals` on a well-formed
        table, and the rows of a hand-built one that lacks the ``delays``
        column, which the consumers then refuse as invalid input rather
        than as an empty cell."""
        return int(len(next(iter(self._columns.values()), ())))

    @property
    def received_amplitudes(self) -> np.ndarray:
        """Complex amplitude each arrival delivers to the receiver.

        Use this, not :attr:`amplitudes`, whenever the numbers are going to be
        compared or summed. ``amplitudes`` is the GEOMETRIC amplitude and
        carries no volume absorption — Bellhop keeps that in the imaginary
        travel time — so on that column a long, heavily absorbed path stands
        at its lossless height. The error is not a detail: on a 1 km
        near-bottom link at 40 kHz, a 3161 m surface-bounce path reads 20 dB
        **stronger** than a 1000 m bottom bounce on ``amplitudes`` and 7.5 dB
        **weaker** once the 41 dB it loses to absorption is applied, so the
        two paths change places.

        The value is ``A * exp(omega * Im tau) * exp(1j * phase)``, the
        convention ``read_arr_file`` documents for ``delays_imag`` and
        ``arrival_grid_transfer_function`` applies, so it drops straight into a
        coherent sum or into
        :func:`~uacpy.acoustic_signal.impulse_response` (whose fractional-delay
        taps are flat only below ``0.35·fs``; :meth:`impulse_response` holds
        the band up to the Nyquist frequency).
        :meth:`_arrival_power` is its squared magnitude. On plain columns
        this is :func:`~uacpy.acoustic_signal.received_amplitudes`.
        """
        return self._received_amplitudes_at(self.f0 or 0.0)

    @property
    def absorption(self):
        """The water-column absorption the arrivals' ``Im tau`` carries —
        the traced environment's ``absorption`` as it holds it (a law, or a
        measured :class:`~uacpy.core.absorption.AbsorptionCoefficient`),
        which :class:`Bellhop` records — or ``None`` (lossless water, or a
        list built by hand without ``absorption=``). With :attr:`f0` it sets
        how the absorption scales to another frequency
        (:func:`~uacpy.core.absorption.arrival_absorption_exponent`)."""
        return self._absorption

    def _received_amplitudes_at(self, frequency: float) -> np.ndarray:
        """``A * exp(e) * exp(1j * phase)`` of every arrival at
        ``frequency`` (Hz), ``e`` the absorption exponent scaled from
        :attr:`f0` by :attr:`absorption` — :attr:`received_amplitudes` at a
        frequency other than the result's own, which the channel-tap builder
        needs at its carrier."""
        amplitude = self.amplitudes
        # Both derived columns are read tolerantly, unlike the strict
        # ``phases`` accessor. A magnitude does not depend on phase, so an
        # arrival set assembled without one — Bellhop always writes it, but
        # callers building Arrivals by hand need not — must still be able to
        # ask what reaches the receiver, and :meth:`_arrival_power` goes
        # through here. The stored phase is radians.
        delays_imag = self._column('delays_imag', 0.0)
        phase = self._column('phases', 0.0)
        # Deferred: acoustic_signal pulls scipy, and uacpy's public
        # surface is imported without it (test_lazy_imports).
        from uacpy.acoustic_signal.channel import received_amplitudes
        return received_amplitudes(amplitude, delays_imag, phase, frequency,
                                   trace_frequency=self.f0,
                                   absorption=self.absorption)

    @property
    def phases(self) -> np.ndarray:
        """Phases (rad) of every arrival in the list.

        The Bellhop ``.arr`` file stores phase in degrees (``ArrMod.f90:120``
        writes ``RadDeg * Phase``); ``read_arr_file`` converts once, so the
        per-arrival ``'phase'`` is already **radians** and drops straight
        into ``exp(1j * phase)`` for phase-coherent synthesis."""
        return self._required_column('phase', 'phases')

    def _required_column(self, key: str, name: str) -> np.ndarray:
        """Column ``name`` (record key ``key``) as a read-only view, with an
        error that names the cause when the table holds no such column."""
        if name not in self._columns:
            raise AttributeError(
                f"Arrivals.{name}: these arrivals carry no '{key}'. Bellhop "
                f"always writes it, so this is an Arrivals built by hand or "
                f"read from a source that dropped the column; supply "
                f"'{key}s' in each receiver cell to use this accessor.")
        return read_only(np.asarray(self._columns[name], dtype=float))

    @property
    def source_angles_deg(self) -> np.ndarray:
        """Declination angle each arrival LEFT the source at, in **degrees**.

        Degrees, not radians — unlike :attr:`phases`, which is radians because
        its consumer is ``exp(1j * phase)``. These angles are reported for
        reading and for geometry (a Doppler projection, a grazing-angle
        filter), so they keep the unit ``ArrMod.f90:55`` writes and
        ``read_arr_file`` documents. Call ``np.deg2rad`` yourself before
        feeding a trigonometric function.

        Sign follows Bellhop's convention: positive is downward-declined.
        """
        return self._required_column('source_angle', 'source_angles_deg')

    @property
    def receiver_angles_deg(self) -> np.ndarray:
        """Declination angle each arrival ARRIVED at the receiver at, in **degrees**.

        The companion to :attr:`source_angles_deg`, and the one a Doppler
        calculation wants: a platform closing at speed ``v`` shifts each path
        by ``f * v * cos(theta) / c`` with ``theta`` the arrival angle, so the
        SPREAD of this column across the arrivals is the channel's Doppler
        spread. Degrees, as ``ArrMod.f90:56`` writes them.
        """
        return self._required_column('receiver_angle', 'receiver_angles_deg')

    # Filter / chain / sort --------------------------------------------------

    #: Per-arrival record key -> the column name :meth:`to_dict` writes it
    #: under: the bulk-accessor / ``.arr`` reader-cell name.
    _COLUMNS = {
        **{key: name for key, (name, _dtype) in _ARRIVAL_COLUMNS.items()},
        'kind': 'kinds',
    }

    def _table(self):
        """One row per arrival, each column named for one arrival as the
        record keys are: the stored columns (``delay`` s, ``delay_imag`` s,
        ``amplitude``, ``phase`` rad, the bounce counts, ``source_angle`` /
        ``receiver_angle`` deg, the cell indices), ``kind``, the complex
        ``received_amplitude`` (:attr:`received_amplitudes`, the volume
        absorption applied), and the receiver each arrival reaches,
        ``receiver_depth`` and ``receiver_range`` (m). The bulk accessors
        and :meth:`to_dict` keep the plural column names."""
        record_key = {name: key for key, name in self._COLUMNS.items()}
        table = {record_key[name]: np.asarray(values).copy()
                 for name, values in self._columns.items()}
        if 'n_top_bounces' in table and 'n_bot_bounces' in table:
            table['kind'] = np.asarray(self.kinds).copy()
        if 'amplitude' in table and 'phase' in table:
            table['received_amplitude'] = np.asarray(
                self.received_amplitudes).copy()
        table['receiver_depth'] = np.asarray(self.receiver_depth).copy()
        table['receiver_range'] = np.asarray(self.receiver_range).copy()
        return table

    def _table_units(self):
        """The stored columns' units under the table's one-arrival names,
        and the receiver's position in metres."""
        units = {key: self._COLUMN_UNITS.get(name, '')
                 for key, name in self._COLUMNS.items()}
        units.update(receiver_depth='m', receiver_range='m')
        return units

    #: The unit of each stored column.
    _COLUMN_UNITS = {'delays': 's', 'delays_imag': 's', 'phases': 'rad',
                     'source_angles_deg': 'deg', 'receiver_angles_deg': 'deg'}

    def _payload(self):
        return {name: (values, ('arrival',),
                       self._COLUMN_UNITS.get(name, ''))
                for name, values in self._columns.items()}

    def _coords(self):
        return {'depth': (self.receiver_depths, coordinate_unit('depth')),
                'range': (self.receiver_ranges, coordinate_unit('range'))}

    def _export_attrs(self):
        attrs = super()._export_attrs()
        if self._grid_shape is not None:
            attrs['grid_shape'] = np.asarray(self._grid_shape, dtype=int)
        if self._absorption is not None:
            # The law or table as the JSON of its own to_dict.
            attrs['absorption'] = json.dumps(
                _to_json(self._absorption.to_dict()))
        return attrs

    @classmethod
    def _from_export(cls, arrays, attrs):
        columns = {name: np.asarray(arrays[name], dtype=dtype)
                   for name, dtype in _ARRIVAL_COLUMNS.values()
                   if name in arrays}
        grid_shape = attrs.get('grid_shape')
        law = attrs.get('absorption')
        return cls._from_columns(
            columns, None if grid_shape is None else tuple(grid_shape),
            receiver_depths=arrays['depth'], receiver_ranges=arrays['range'],
            absorption=_absorption_from(
                None if law is None else _from_json(json.loads(law))),
            **cls._identity_from_attrs(attrs,
                                       ('grid_shape', 'absorption')))

    def to_dict(self) -> Dict[str, Any]:
        """Serialise these arrivals to plain arrays, one column per record key.

        Every column the table holds is one 1-D array named as in
        :attr:`_COLUMNS` (``delays`` s, ``phases`` rad, ``amplitudes``, the
        bounce counts, ``source_angles_deg`` / ``receiver_angles_deg``, ``kinds``, and the
        cell indices ``src_idx`` / ``depth_idx`` / ``range_idx``), so the
        columns load straight into a table or a CSV. The receiver grid, the
        shape of the nested :attr:`by_receiver` view (``None`` without one),
        the :attr:`absorption` (law or table) as its own ``to_dict``
        (``None`` without one) and the identity follow, the enums as their string
        values.
        ``np.savez(f, **d)`` stores it; read it back with
        ``np.load(f, allow_pickle=True)`` into :meth:`from_dict`. That load
        unpickles, so use it only on files you wrote; :meth:`to_netcdf` /
        :meth:`from_netcdf` is the route for a file from elsewhere.
        """
        d: Dict[str, Any] = {name: np.asarray(values).copy()
                             for name, values in self._columns.items()}
        if 'n_top_bounces' in d and 'n_bot_bounces' in d and len(self):
            d['kinds'] = np.asarray(self.kinds).astype(str)
        d.update({
            'receiver_depths': self.receiver_depths.copy(),
            'receiver_ranges': self.receiver_ranges.copy(),
            'by_receiver_shape': self._grid_shape,
            'absorption': (None if self._absorption is None
                           else self._absorption.to_dict()),
            **self._identity_dict(),
        })
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Arrivals':
        """Reconstruct :class:`Arrivals` from :meth:`to_dict` output, or from
        the mapping ``np.load(f, allow_pickle=True)`` returns for a file
        written with ``np.savez(f, **arrivals.to_dict())`` (its 0-d entries
        are unwrapped; load such a file only if you wrote it:
        ``allow_pickle=True`` unpickles). The nested :attr:`by_receiver` view
        is regrouped from the cell indices when the source carried one.

        Parameters
        ----------
        d : mapping
            :meth:`to_dict` output, or the mapping ``np.load`` returns for it.
        """
        d = cls._unwrap_saved(
            d, required=('receiver_depths', 'receiver_ranges',
                         'by_receiver_shape', 'absorption'))
        columns = {name: np.atleast_1d(np.asarray(d[name], dtype=dtype))
                   for name, dtype in _ARRIVAL_COLUMNS.values() if name in d}
        # An empty set writes no columns at all; a set with arrivals writes
        # at least their delays and amplitudes (a hand-built set may hold no
        # other column).
        absent = sorted({'delays', 'amplitudes'} - set(columns))
        if columns and absent:
            raise ConfigurationError(
                f"Arrivals.from_dict: the mapping holds arrival columns but "
                f"no {absent}; it reads what Arrivals.to_dict writes.",
                remediation="Pass the dict Arrivals.to_dict returned (or "
                            "np.load of the .npz it was saved to).")
        identity = cls._identity_from_dict(d)
        return cls._from_columns(
            columns, d.get('by_receiver_shape'),
            receiver_depths=d['receiver_depths'],
            receiver_ranges=d['receiver_ranges'],
            absorption=_absorption_from(d.get('absorption')),
            **identity)

    def filter(self, predicate) -> 'Arrivals':
        """Return a new ``Arrivals`` keeping arrivals for which
        ``predicate(arrival_dict)`` returns true.

        Parameters
        ----------
        predicate : callable
            ``predicate(arrival_dict) -> bool``.
        """
        return self._rows(np.asarray(
            [i for i, a in enumerate(self.arrivals) if predicate(a)],
            dtype=int))

    def filter_by_bounces(
        self,
        kind: Optional[str] = None,
        top: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None,
        bot: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None,
    ) -> 'Arrivals':
        """Subset by multipath component — same semantics as
        :meth:`Rays.filter_by_bounces`. ``kind`` ∈
        ``{'direct', 'surface', 'bottom', 'both'}``; ``top`` / ``bot`` are
        an int (exact) or ``(lo, hi)`` tuple (closed range, ``None`` =
        unbounded).

        Parameters
        ----------
        kind : {'direct', 'surface', 'bottom', 'both'}, optional
            Bounce class to keep; ``None`` keeps every class.
        top, bot : int or (int, int), optional
            Surface / bottom bounce count: exact, or a closed ``(lo, hi)`` range
            with ``None`` for an open end; ``None`` is any count.
        """
        pred = _bounce_predicate(kind, top, bot)
        return self._rows(np.flatnonzero(np.asarray(
            [pred(int(t), int(b)) for t, b in zip(self.n_top_bounces,
                                                  self.n_bot_bounces)],
            dtype=bool)))

    def window(self, *, delay) -> 'Arrivals':
        """Keep the arrivals whose ``delay`` (s) falls inside the inclusive
        ``(lo, hi)`` pair, either end ``None`` to leave it open — the window
        rule of :meth:`Field.window`. An empty window keeps no arrival: a
        delay span with no path in it is an answer about the channel.

        Parameters
        ----------
        delay : (float, float)
            Inclusive delay window (s); ``None`` leaves that end open.
        """
        low, high = _window_pair('Arrivals.window', 'delay', delay)
        delays = self.delays
        keep = np.ones(delays.size, dtype=bool)
        if low is not None:
            keep &= delays >= low
        if high is not None:
            keep &= delays <= high
        return self._rows(np.flatnonzero(keep))

    def sorted_by_amplitude(self, descending: bool = True) -> 'Arrivals':
        """Return a copy sorted by received amplitude (descending by default).

        Ranked on what reaches the receiver, not on the ``amplitude`` column
        alone: Bellhop keeps volume absorption in the imaginary travel time
        (see :meth:`_arrival_power`), so a long path can carry the larger
        geometric amplitude and still arrive far quieter. At 40 kHz a 6 km
        bounce path with three times the direct path's amplitude lands 55 dB
        below it, and ranking on the column alone puts it first.

        With no frequency on the result the absorption factor is 1 and this
        is the column order, unchanged.

        Parameters
        ----------
        descending : bool, optional
            Loudest first. Default True.
        """
        # argsort on power: monotone in received amplitude, so it orders the
        # same way without the square root, and it is the same quantity
        # ``rms_delay_spread`` and ``energy_support`` weigh by.
        order = np.argsort(self._arrival_power(), kind='stable')
        if descending:
            order = order[::-1]
        return self._rows(order)

    def top_n_by_amplitude(self, n: int) -> 'Arrivals':
        """Keep the ``n`` arrivals that reach the receiver loudest.

        "Loudest" is the received level, absorption included — see
        :meth:`sorted_by_amplitude` for why the amplitude column alone
        answers a different question.

        Parameters
        ----------
        n : int
            Arrivals to keep.
        """
        n = _count(n, 'Arrivals.top_n_by_amplitude')
        order = np.argsort(self._arrival_power(), kind='stable')[::-1]
        return self._rows(order[:n])

    def _rows(self, index) -> 'Arrivals':
        """These arrivals reduced to the table rows ``index``, in that
        order, on the parent's receiver grid: one fancy index per column,
        not a record per row, so every column survives (with no row when
        ``index`` is empty) and the derived :attr:`by_receiver` holds
        exactly the rows kept."""
        return self._with_columns({name: np.asarray(values)[index]
                                   for name, values in self._columns.items()})

    def _arrival_power(self) -> np.ndarray:
        """Power each arrival delivers to the receiver, absorption included.

        Volume absorption does not live in the amplitude column: Bellhop
        carries it in the IMAGINARY travel time, so the received amplitude is
        ``A * exp(w * Im tau)`` — the convention ``read_arr_file`` documents
        for ``delays_imag`` and ``arrival_grid_transfer_function`` applies. Scoring
        on ``A`` alone treats a late, heavily absorbed path as though the
        water were lossless, and the late paths are the ones every caller of
        this is weighing.
        """
        with np.errstate(over='ignore'):
            return np.abs(self.received_amplitudes) ** 2

    def _cell_profile(self, receiver, who: str):
        """``(delays, powers)`` of one receiver cell's power delay profile
        (:meth:`_one_cell`), absorption included."""
        cell = self._one_cell(receiver, who)
        return cell.delays, cell._arrival_power()

    def rms_delay_spread(self, *, receiver=None) -> float:
        """Energy-weighted spread of the arrival delays, in seconds.

        The second central moment of the power delay profile: delays weighted
        by the power each arrival delivers, absorption included (the received
        amplitude squared, not the amplitude column's — see
        :meth:`sorted_by_amplitude`), about their weighted mean. It measures how much
        the arrival pattern smears a pulse in time — the width the multipath
        gives an impulse — so it bounds the time resolution any processing of
        this channel can have, whatever the processing is for: the smearing
        of a transmitted pulse, the length a replica or matched filter has to
        cover, the interval a symbol would have to exceed to avoid
        overlapping its neighbour.

        Prefer it to the peak-to-peak spread ``ptp(delays)``, which is set by
        whichever ray arrives last no matter how faint: on a 1 km
        bottom-to-bottom path in 1000 m of water at 40 kHz the two differ by
        more than two orders of magnitude, because a path tens of dB down
        lands seconds late while almost all the energy arrives within a
        millisecond of the first.

        Returns ``0.0`` for a single arrival, and for arrivals carrying no
        energy at all. A non-finite delay or amplitude propagates: the result
        is ``nan``, not a spread computed from whatever else was finite.

        Parameters
        ----------
        receiver : (float, float), optional
            ``(depth, range)`` of the cell whose channel this is. A
            channel is one receiver's: arrivals spanning several cells are
            refused without it, as :meth:`transfer_function` refuses them,
            because pooling them would read the travel-time differences
            between receivers as multipath spread.

        Notes
        -----
        Its reciprocal is the frequency scale over which the transfer
        function decorrelates: a widely dispersed arrival pattern fades over
        a narrow band, which is why a broadband view of such a channel shows
        structure far finer than the band. The constant relating the two is a
        correlation-threshold convention rather than a law, so it is left to
        the caller to state.

        On a power delay profile from anywhere else this is
        :func:`~uacpy.acoustic_signal.rms_delay_spread(delays, powers)`.
        """
        # Deferred: acoustic_signal pulls scipy, and uacpy's public
        # surface is imported without it (test_lazy_imports).
        from uacpy.acoustic_signal.delay_profile import _rms_delay_spread
        who = "Arrivals.rms_delay_spread"
        delays, powers = self._cell_profile(receiver, who)
        return _rms_delay_spread(delays, powers, who=who)

    def energy_support(self, fraction: float = 0.999, *,
                       receiver=None) -> float:
        """Delay span holding ``fraction`` of the arrival energy, in seconds.

        Measured from the first arrival to the one by which ``fraction`` of
        the received energy has arrived. It answers the question a synthesis
        window asks — how long does the response have to be? — which neither
        of the other two measures does: ``ptp(delays)`` is an extremum, moved
        by one faint straggler however little it carries, and
        :meth:`rms_delay_spread` is a second moment, a width rather than a
        span the energy fits inside.

        Parameters
        ----------
        fraction : float, default 0.999
            Share of the total energy the span must hold, in ``(0, 1]``.
            ``1.0`` is the peak-to-peak span. The default leaves a thousandth
            of the energy — 30 dB down — outside.
        receiver : (float, float), optional
            ``(depth, range)`` of the cell whose channel this is. A
            channel is one receiver's: arrivals spanning several cells are
            refused without it, as :meth:`transfer_function` refuses them,
            because pooling them would read the travel-time differences
            between receivers as multipath spread.

        Returns
        -------
        float
            Seconds. ``0.0`` for a single arrival and for arrivals carrying
            no energy at all; ``nan`` if a delay or amplitude is non-finite,
            rather than a span computed from whatever else was finite.

        On a power delay profile from anywhere else this is
        :func:`~uacpy.acoustic_signal.energy_support`.
        """
        # Deferred: acoustic_signal pulls scipy, and uacpy's public
        # surface is imported without it (test_lazy_imports).
        from uacpy.acoustic_signal.delay_profile import _energy_support
        who = "Arrivals.energy_support"
        delays, powers = self._cell_profile(receiver, who)
        return _energy_support(delays, powers, fraction, who=who)

    def synthesis_band(
        self,
        *,
        bandwidth: float,
        fc: Optional[float] = None,
        output_duration: Optional[float] = None,
        energy_fraction: Optional[float] = None,
        margin: float = 1.2,
    ) -> np.ndarray:
        """Frequency grid wide enough to synthesise these arrivals un-aliased.

        A record built by an inverse FFT is ``1/df`` long, so the frequency
        spacing — not the bandwidth — decides how much multipath the trace
        can hold. Choose it without looking at the arrivals and the late
        paths wrap onto the early ones, which reads as extra arrivals rather
        than as a mistake.

        The reasoning, the literature and the measurements behind this method
        are in ``docs/theory/broadband_products.md``, section "The synthesis
        band of an arrival set".

        **This budgets the ARRIVALS only.** A transmitted pulse has length
        too, and it is not visible here — a 20 ms waveform through a 2.7 ms
        channel can be handed a 5 ms record by the default call, which stays
        silent because the arrivals do fit. Pass ``output_duration`` with
        the whole budget instead, remembering that it bypasses ``margin`` so
        the headroom is the caller's::

            span = arrivals.energy_support(0.999) + len(waveform) / fs
            f = arrivals.synthesis_band(bandwidth=B, fc=f0,
                                        output_duration=6.0 * span)

        Several times over rather than a token factor: the pulse has to fit
        and the band edge's precursor needs somewhere to sit. Measured on
        one geometry (example_45's), energy arriving before the first path
        could ran 1.9 % at 1.5x that span and 0.18 % from 4x onwards.

        Parameters
        ----------
        bandwidth : float
            Width of the band to synthesise (Hz), centred on ``fc``.
        fc : float, optional
            Band centre frequency (Hz); defaults to this result's own
            frequency.
        output_duration : float, optional
            Length of the record to synthesise (s), stated rather than
            derived. Passing it with ``energy_fraction`` is refused — they
            are two answers to one question.
        energy_fraction : float, optional
            Share of the arrival energy the derived window must hold
            (default 0.999). Lower it for a shorter record and a coarser
            grid; whatever falls outside wraps, at a level this reports.
        margin : float, optional
            Headroom on the derived span (default 1.2), so the last arrival
            inside it lands within the record rather than on its final
            sample. It is also the number of frequency samples per
            interference fringe: two paths ``dtau`` apart beat in ``|H(f)|``
            with period ``1/dtau``, and a record ``margin * dtau`` long
            samples that at ``df = 1/(margin * dtau)`` — ``margin`` points
            per fringe. 1.2 is enough for the inverse FFT, which needs only
            that the record hold the arrivals; it is not enough to LOOK at
            the transfer function, whose curve between samples is then the
            plotter's, not the model's. Raise it to 8 or so for a drawn
            ``|H(f)|``. Not used when ``output_duration`` is given.

        Returns
        -------
        ndarray
            Ascending frequencies to hand to a broadband run.

        Warns
        -----
        NumericsWarning
            When the record does not reach the last arrival, giving how many
            arrivals fold back and the level they fold in at.

        Notes
        -----
        Sizing the window to the last arrival however faint is what makes
        this expensive, because the peak-to-peak span is an extremum: on a
        1 km bottom-mounted link at 40 kHz it asks for some 440 000 bins to
        hold one ray around 200 dB down, where 0.999 of the energy is spanned
        by a few milliseconds. Trading that tail for a shorter record is a
        choice rather than an approximation, so it is made explicitly and its
        cost is reported instead of absorbed. Filtering the arrivals first
        (:meth:`window`, :meth:`top_n_by_amplitude`) drops the tail
        outright rather than folding it."""
        if fc is None:
            fc = self.f0
            if fc is None:
                raise ConfigurationError(
                    "Arrivals.synthesis_band: this result carries no "
                    "frequency, so the band has no centre. Pass fc= (Hz).")
        # POOLED over every cell, deliberately: the synthesis shares one
        # record across the grid, so it has to hold every cell's arrivals
        # at once, not one receiver's channel. Deferred: acoustic_signal is
        # imported on first use.
        from uacpy.acoustic_signal.delay_profile import _synthesis_band
        return _synthesis_band(
            self.delays, self._arrival_power(), bandwidth=bandwidth,
            fc=fc, output_duration=output_duration,
            energy_fraction=energy_fraction,
            margin=margin, who="Arrivals.synthesis_band")

    # Communications view -------------------------------------------------

    def at_receiver(self, receiver=None) -> 'Arrivals':
        """The arrivals of one receiver cell, as ``Arrivals``.

        ``receiver=(depth, range)`` picks the cell by its coordinates on
        the result's receiver axes; ``None`` is accepted only when every
        arrival already sits in one cell. A multi-cell set without
        ``receiver=``, a multi-source set and an empty cell are refused, as
        the channel methods refuse them.

        Parameters
        ----------
        receiver : (float, float), optional
            ``(depth, range)`` of the cell.
        """
        return self._at_receiver(receiver, 'Arrivals.at_receiver')

    def _at_receiver(self, receiver, who: str) -> 'Arrivals':
        """:meth:`at_receiver` reporting its refusals as ``who``."""
        return self._one_cell(receiver, who)

    @property
    def _paired_grid(self) -> bool:
        """Whether the receivers form an irregular grid: one depth block
        whose cell ``i`` is the receiver ``(receiver_depths[i],
        receiver_ranges[i])`` (``bellhop.f90:202-206``)."""
        return (self._grid_shape is not None and self._grid_shape[1] == 1
                and self.receiver_depths.size > 1
                and self.receiver_depths.size == self.receiver_ranges.size)

    def _one_cell(self, receiver, who: str) -> 'Arrivals':
        """The arrivals of one receiver cell.

        ``receiver=None`` is accepted only when every arrival sits in one
        cell; otherwise ``receiver=(depth, range)`` picks a cell by the
        coordinates on the result's receiver axes (on an irregular grid,
        the one receiver at that depth AND range), and an empty cell — a
        shadow zone, where the channel is undefined — is refused too.
        """
        src = self._cell_index('src_idx')
        depth_index = self._cell_index('depth_idx')
        range_index = self._cell_index('range_idx')
        cells = sorted(set(zip(src.tolist(), depth_index.tolist(),
                               range_index.tolist())))
        if receiver is None:
            if len(cells) > 1:
                raise ConfigurationError(
                    f"{who}: these arrivals span {len(cells)} receiver "
                    f"cells ({self.receiver_depths.size} depths x "
                    f"{self.receiver_ranges.size} ranges), and a channel "
                    f"is one receiver's. Pass receiver=(depth, range) "
                    f"to choose the cell, or filter to one first.")
            if not cells:
                raise ConfigurationError(
                    f"{who}: no arrivals — the channel of a cell nothing "
                    f"reaches is undefined, not empty.")
            return self._rows(np.arange(self._n_rows))
        try:
            depth, rng = (float(v) for v in receiver)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"{who}: receiver= must be a (depth, range) pair; got "
                f"{receiver!r}.") from exc
        if self._paired_grid:
            d_idx = 0
            r_idx = _pair_index(self.receiver_depths, self.receiver_ranges,
                                depth, rng, who)
        else:
            d_idx = _axis_index(self.receiver_depths, depth, who, "depth")
            r_idx = _axis_index(self.receiver_ranges, rng, who, "range")
        sources = sorted({c[0] for c in cells})
        if len(sources) > 1:
            raise ConfigurationError(
                f"{who}: these arrivals come from {len(sources)} source "
                f"depths, and a channel is one source's too. Filter on "
                f"'src_idx' first: arr.filter(lambda a: a['src_idx'] == 0).")
        picked = np.flatnonzero((depth_index == d_idx)
                                & (range_index == r_idx))
        if not picked.size:
            raise ConfigurationError(
                f"{who}: no arrivals at receiver depth {depth:g} m, range "
                f"{rng:g} m — the channel of a cell nothing reaches is "
                f"undefined, not empty.")
        return self._rows(picked)

    def transfer_function(self, frequencies, *, receiver=None) -> "Field":
        """``H(f)`` of these arrivals, as a single-cell broadband
        :class:`~uacpy.core.results.Field`.

        The channel frequency response is "the weighted sum over the paths of
        a complex phase term depending on each path's delay" (Abraham,
        *Underwater Acoustic Signal Processing*, sect. 3.2.3.2)::

            H(f) = sum_i a_i(f) exp(i phi_i) exp(-i 2 pi f tau_i)

        ``a_i(f)`` is :attr:`received_amplitudes` evaluated at ``f``: the
        volume absorption Bellhop keeps in ``Im tau`` is exact at :attr:`f0`
        and scaled to each ``f`` by :attr:`absorption`
        (:func:`~uacpy.core.absorption.arrival_absorption_exponent`) — as
        ``alpha(f)/alpha(f0)`` for Thorp and Francois-Garrison, which is
        exact; linearly in ``f`` for a constant dB/wavelength law (also
        exact), for a Biological layer (an approximation) and for a list
        with no law. It is the same expression ``arrival_grid_transfer_function``
        uses, and a ``RunMode.BROADBAND`` run on the same grid reproduces
        this to floating-point.

        The reasoning, the literature and the measurements behind this method
        are in ``docs/theory/broadband_products.md``, section "H(f) of an
        arrival set".

        Parameters
        ----------
        frequencies : array_like
            Frequency grid (Hz), finite and non-negative: a 0 Hz bin (the
            first of ``np.fft.rfftfreq``) carries no absorption, as
            :func:`~uacpy.acoustic_signal.arrival_transfer_function` reads
            it.
        receiver : (float, float), optional
            ``(depth, range)`` of the cell to take, on the result's
            receiver axes. ``None`` takes every cell of the per-receiver
            grid (``by_receiver``), zero where a cell's arrival list is
            empty (the empty sum); a list with
            no such grid must hold a single cell.

        Returns
        -------
        Field
            Complex ``H`` with canonical ``['depth', 'range', 'frequency']``
            coords — a single depth and range for ``receiver=``, the
            receiver axes for every cell (a paired grid's receivers on
            ``'range'``, their depths on ``aux_coords['receiver_depth']``) —
            so
            :meth:`~uacpy.core.results.Field.remove_delay`,
            :meth:`~uacpy.core.results.Field.synthesize_time_series` and
            :meth:`~uacpy.core.results.Field.plot_transfer_function` all
            take it. Its :attr:`~uacpy.core.results.Field.speeds` state as
            ``water_max`` the fastest mean horizontal speed ``r/tau`` of a
            cell's first arrival, so the synthesis windows anchored on
            ``r/water_max`` open at or before every cell's first arrival.
            An arrival list carries no sound-speed profile; no path's mean
            horizontal speed exceeds the water's fastest speed, so this
            stands in for it from below.

        Raises
        ------
        ConfigurationError
            An empty, non-finite or negative grid; every cell of arrivals
            from several source depths; several cells of a list with no
            per-receiver grid without ``receiver=``; a picked cell nothing
            reaches.

        On a plain arrival list this is
        :func:`~uacpy.acoustic_signal.arrival_transfer_function`.
        """
        who = "Arrivals.transfer_function"
        freqs = np.atleast_1d(np.asarray(frequencies, dtype=float)).ravel()
        if freqs.size == 0:
            raise ConfigurationError(
                f"{who}: frequencies is empty; H(f) needs a grid to be "
                f"evaluated on.")
        if not np.all(np.isfinite(freqs)):
            raise ConfigurationError(
                f"{who}: frequencies must all be finite.")
        if np.any(freqs < 0.0):
            raise ConfigurationError(
                f"{who}: frequencies must be non-negative (Hz); got a "
                f"minimum of {freqs.min():g}.")
        if receiver is None and self.by_receiver is not None:
            return self._grid_transfer_function(freqs, who)
        cell = self._one_cell(receiver, who)
        delays = cell.delays
        if not np.all(np.isfinite(delays)):
            raise ConfigurationError(
                f"{who}: an arrival has a non-finite delay, so its phase "
                f"term is undefined at every frequency.")
        # The amplitude is re-evaluated at each frequency rather than taken
        # once at the result's own: the absorption lives in Im(tau), traced
        # at f0, and is scaled to each frequency by the traced law
        # (arrival_absorption_exponent: alpha(f)/alpha(f0) for Thorp and
        # Francois-Garrison, linear in f otherwise). The phase term is the
        # outer product of delays and frequencies.
        # Deferred: acoustic_signal pulls scipy, and uacpy's public
        # surface is imported without it (test_lazy_imports).
        from uacpy.acoustic_signal.channel import _arrival_transfer_function
        H = _arrival_transfer_function(
            freqs, cell.amplitudes, delays,
            delays_imag=cell.delays_imag,
            phases=cell._column('phases', 0.0),
            trace_frequency=self.f0, absorption=self.absorption,
            who=who)
        depth, rng = self._cell_coordinates(cell, receiver)
        # Deferred: ``field`` imports ``core.environment``, and importing it
        # at this module's scope pulls that chain into every ``Arrivals``.
        from uacpy.core.results.field import Field
        # The identity and run settings of the arrivals run carry over; the
        # arrival-list metadata describes the records, not H(f).
        id_kwargs = self.id_kwargs()
        id_kwargs.update(phase_reference=PhaseReference.TRAVELLING_WAVE,
                         frequencies=freqs, metadata=None)
        return Field(
            data=H.reshape(1, 1, freqs.size),
            coords={'depth': np.array([depth]),
                    'range': np.array([rng]),
                    'frequency': freqs},
            speeds=_first_arrival_speeds([rng], [delays.min()]),
            **id_kwargs,
        )


    def _receiver_cells(self, who: str):
        """``by_receiver[0]``, the ``[depth][range]`` arrival records of the
        one source, refusing a result with no per-receiver form or with
        several sources."""
        if self.by_receiver is None:
            raise ConfigurationError(
                f"{who}: these arrivals carry no per-receiver grid "
                f"(by_receiver); pass receiver=(depth, range) for one "
                f"cell.")
        if len(self.by_receiver) > 1:
            raise ConfigurationError(
                f"{who}: these arrivals come from {len(self.by_receiver)} "
                f"source depths, and a channel is one source's. Filter on "
                f"'src_idx' first: arr.filter(lambda a: a['src_idx'] == 0).")
        return self.by_receiver[0]

    def _grid_transfer_function(self, freqs, who: str) -> "Field":
        """:meth:`transfer_function` over every cell: a Field on the
        receiver axes, zero where a cell's arrival list is empty."""
        from uacpy.acoustic_signal.channel import (
            _arrival_grid_transfer_function,
        )
        from uacpy.core.results.field import Field
        cells = self._receiver_cells(who)
        H = _arrival_grid_transfer_function(
            freqs, cells, trace_frequency=self.f0,
            absorption=self.absorption, who=who)
        depths = np.atleast_1d(np.asarray(self.receiver_depths, dtype=float))
        ranges = np.atleast_1d(np.asarray(self.receiver_ranges, dtype=float))
        first = [[np.asarray(cell.get('delays', ()), dtype=float)
                  for cell in row] for row in cells]
        filled = [(ranges[j], float(delays.min()))
                  for row in first for j, delays in enumerate(row)
                  if delays.size]
        id_kwargs = self.id_kwargs()
        id_kwargs.update(phase_reference=PhaseReference.TRAVELLING_WAVE,
                         frequencies=freqs, metadata=None,
                         speeds=_first_arrival_speeds(
                             [r for r, _ in filled], [t for _, t in filled]))
        if self._paired_grid:
            # A paired grid: one block whose cells are the receivers
            # (depths[i], ranges[i]), the depths labelling the range axis.
            return Field(data=H[0], coords={'range': ranges,
                                            'frequency': freqs},
                         aux_coords={'receiver_depth': ('range', depths)},
                         **id_kwargs)
        return Field(data=H, coords={'depth': depths, 'range': ranges,
                                     'frequency': freqs},
                     **id_kwargs)

    def to_time_series(self, *, source_waveform, sample_rate: float,
                       output_duration: Optional[float] = None,
                       t_start: Optional[float] = None) -> "Field":
        """The waveform every receiver of these arrivals receives, on one
        clock, as a time-domain :class:`~uacpy.core.results.Field`
        (:func:`~uacpy.acoustic_signal.simulate_arrival_grid`, at
        :attr:`f0` with :attr:`absorption`).

        Each arrival delivers a phase-rotated copy of the waveform at its
        delay, filtered by its absorption at every frequency, exactly at
        ``sample_rate``: each cell's own reception
        (:func:`~uacpy.acoustic_signal.simulate_arrival_reception` — the
        waveform and its Hilbert transform through the cell's in-phase and
        quadrature taps, by FFT convolution), placed on the one window every
        cell shares. The phase rotation is taken on the waveform, so it
        holds down to the waveform's lowest frequency, where the taps of
        :meth:`impulse_response` cannot rotate content below about one over
        their margin. An echo a stated window omits is left out whole, the
        precursor and tail of its Hilbert transform with it.

        Parameters
        ----------
        source_waveform : ndarray
            Source waveform (1-D, real).
        sample_rate : float
            Sample rate in Hz.
        output_duration, t_start : float, optional
            The shared window (s); left ``None``, the window spans every
            cell's arrivals (the earliest less a tenth of the pulse, the
            latest plus one pulse length and a quiet tail of one pulse
            length capped at the delay span).

        Returns
        -------
        Field
            ``p(t)`` on ``(depth, range, time)`` over ``by_receiver``'s
            grid — a paired grid's receivers on ``'range'``, their depths
            on ``aux_coords['receiver_depth']`` — at ``sample_rate``. A cell
            whose arrival list is empty is a silent trace.
        """
        t_vec, data = self._time_series(
            source_waveform, sample_rate, output_duration=output_duration,
            t_start=t_start, who="Arrivals.to_time_series")
        return self._receiver_field(data, 'time', t_vec, kind='pressure',
                                    unit='Pa')

    def _time_series(self, source_waveform, sample_rate, *, output_duration,
                     t_start, who: str):
        """``(time, traces)`` of :meth:`to_time_series`, the traces on
        ``by_receiver``'s ``(depth block, range)`` grid, its refusals and
        window notice reported as ``who``."""
        from uacpy.acoustic_signal.channel import _simulate_arrival_grid
        return _simulate_arrival_grid(
            self._receiver_cells(who), source_waveform=source_waveform,
            sample_rate=sample_rate, trace_frequency=self.f0,
            output_duration=output_duration, t_start=t_start,
            absorption=self.absorption, who=who)

    def impulse_response(self, sample_rate: float, *,
                         t_start='reduced',
                         output_duration: Optional[float] = None,
                         band=None, receiver=None) -> "Field":
        """The impulse response ``h(t)`` of every receiver these arrivals
        reach, at exactly ``sample_rate``: the channel a signal of any
        length is played through by :meth:`Field.apply
        <uacpy.core.results.Field.apply>`.

        Real passband taps of :meth:`transfer_function`
        (:func:`~uacpy.acoustic_signal.arrival_impulse_response` per cell):
        each path's :attr:`received_amplitudes` — its phase, and its volume
        absorption scaled to every frequency by :attr:`absorption`, never
        the lossless geometric :attr:`amplitudes` — as a band-limited impulse
        at its delay, up to the Nyquist frequency: the in-phase part of every
        path built on a record of its own about it, the quadrature part
        rotated once on the whole record. Stored as the
        density in 1/s (``taps·sample_rate``), ``kind='impulse_response'``,
        as :meth:`Field.impulse_response
        <uacpy.core.results.Field.impulse_response>` returns it.

        The record of each range opens at ``|r|/c`` less a margin, ``c``
        the fastest mean horizontal speed of any cell's first arrival
        (``r/τ`` — no path's exceeds the water's fastest speed), so every
        first arrival lies inside it; the ``time`` axis is the reduced time
        ``t − |r|/c`` (Jensen et al., *Computational Ocean Acoustics*,
        sect. 8.2.1.3, eq. 8.15) and ``c`` the result's
        :attr:`~uacpy.core.results.Field.reduction_speed`. It closes after
        every cell's last arrival plus the same margin: the arrivals' delay
        span either side, and at least 256 samples, for the tail of a
        phase-rotated arrival (a band-limited impulse decaying as
        ``1/(π n)`` over ``n`` samples) to fall away before the record
        ends. That tail is also the response's limit: a record of finite
        length cannot rotate the phase of content below about one over its
        margin (188 Hz at 256 samples at 48 kHz) — a 4-cycle 50 Hz pulse
        played through a lone quarter-turn arrival this way is 0.55 of its
        peak off (:func:`~uacpy.acoustic_signal.arrival_impulse_response`).
        For a waveform with content that low, :meth:`to_time_series` gives
        the reception with the rotation taken on the waveform itself.

        Parameters
        ----------
        sample_rate : float
            The rate (Hz) of ``h``.
        t_start : 'reduced', float or None, default 'reduced'
            ``'reduced'``: each range its own record, as above (one shared
            record when no cell has a positive range and first delay to
            define ``c``). A number: one shared record starting there (s),
            on absolute time. ``None``: one shared record from the earliest
            arrival less the margin.
        output_duration : float, optional
            Length (s) of each record, from the same start; ``None`` sizes
            it from the arrivals as above. A record that does not reach an
            arrival leaves that arrival out — on the record it would leave
            only a cut fragment of its impulse, its phase rotation wrapped
            round the record — and a ``NumericsWarning`` names the share of
            the received energy left out. A record that this length, or a
            number ``t_start``, brings closer to its arrivals than the
            margin is built padded to the margin and cut back, so its taps
            are the response's own samples there; it then costs the padded
            record, up to the record sized from the arrivals.
        band : (float, float), optional
            ``(low, high)`` in Hz: the response keeps ``H`` there and is
            zero outside it. ``None``: 0 Hz to the Nyquist frequency.
        receiver : (float, float), optional
            ``(depth, range)`` of one cell; ``None`` takes every cell of
            the per-receiver grid, and a list with no such grid must hold
            a single cell.

        Returns
        -------
        Field
            ``kind='impulse_response'`` in 1/s on ``(depth, range, time)``
            (a paired grid's receivers on ``'range'``), zeros in a cell
            whose arrival list is empty.

        Raises
        ------
        ConfigurationError
            A non-positive ``sample_rate``, a ``band`` outside ``[0,
            sample_rate/2]`` or keeping no bin, a non-positive
            ``output_duration``, an unknown ``t_start``, or the refusals of
            :meth:`transfer_function`.

        On a plain arrival list this is
        :func:`~uacpy.acoustic_signal.arrival_impulse_response` (and
        :func:`~uacpy.acoustic_signal.arrival_grid_impulse_response` over a
        grid); what the method adds is the reduced-time placement from the
        receiver ranges and the Field.
        """
        who = "Arrivals.impulse_response"
        from uacpy.core._validate import (real_scalar_or_none,
                                          require_positive_finite_scalar)
        from uacpy.acoustic_signal._synthesis import arrival_ir_margin
        from uacpy.acoustic_signal.channel import (
            _arrival_cells_impulse_response,
        )
        fs = require_positive_finite_scalar(sample_rate, who, "sample_rate",
                                            " Hz")
        if receiver is None and self.by_receiver is not None:
            cells = self._receiver_cells(who)
            ranges = np.atleast_1d(np.asarray(self.receiver_ranges,
                                              dtype=float))
            single = None
        else:
            cell = self._one_cell(receiver, who)
            depth, rng = self._cell_coordinates(cell, receiver)
            cells = [[{'n_arrivals': cell.n_arrivals,
                       'amplitudes': cell.amplitudes,
                       'delays': cell.delays,
                       'delays_imag': cell.delays_imag,
                       'phases': cell._column('phases', 0.0)}]]
            ranges, single = np.array([rng]), (depth, rng)
        n_i, n_j = len(cells), len(cells[0])
        cell_range = np.broadcast_to(ranges[:n_j], (n_i, n_j))
        first = np.full((n_i, n_j), np.nan)
        last = np.full((n_i, n_j), np.nan)
        for i in range(n_i):
            for j in range(n_j):
                delays = np.asarray(cells[i][j].get('delays', ()),
                                    dtype=float)
                if int(cells[i][j].get('n_arrivals', delays.size)):
                    if not np.all(np.isfinite(delays)):
                        raise ConfigurationError(
                            f"{who}: an arrival has a non-finite delay, so "
                            f"its place on the record is undefined.")
                    first[i, j], last[i, j] = delays.min(), delays.max()
        reached = np.isfinite(first)
        if not reached.any():
            raise ConfigurationError(
                f"{who}: no cell holds an arrival — the channel of a cell "
                f"nothing reaches is undefined, not empty.")
        n_samples = None
        if output_duration is not None:
            record = require_positive_finite_scalar(
                output_duration, who, "output_duration", " s",
                why=" Pass each record's length, or None to size it from "
                    "the arrivals.")
            # The relative 1e-12 keeps floating-point dust from adding a
            # sample (1.1 s at 48 kHz is 52800.00000000001 samples).
            n_samples = int(np.ceil(record * fs * (1.0 - 1e-12)))
        # The margin either side of the arrivals is the plain function's;
        # the widest serves every range.
        margin = max(arrival_ir_margin(span, fs)
                     for span in (last - first)[reached])
        speeds = _first_arrival_speeds(cell_range[reached], first[reached])
        reduction = None
        stated = None
        if isinstance(t_start, str):
            if t_start != 'reduced':
                raise ConfigurationError(
                    f"{who}: t_start must be 'reduced' (each range its own "
                    f"record, on the reduced time), a time in seconds or "
                    f"None (one shared record from the earliest arrival); "
                    f"got {t_start!r}.",
                    remediation="Pass one time (s) for a record every cell "
                                "shares, 'reduced' for each range its own "
                                "record, or None to start at the earliest "
                                "arrival less the margin.")
        elif t_start is not None:
            stated = real_scalar_or_none(t_start)
            if stated is None or not np.isfinite(stated):
                raise ConfigurationError(
                    f"{who}: t_start must be 'reduced', one finite time in "
                    f"seconds or None; got {t_start!r}.",
                    remediation="Pass one time (s) for a record every cell "
                                "shares, 'reduced' for each range its own "
                                "record, or None to start at the earliest "
                                "arrival less the margin.")
        if t_start == 'reduced' and speeds is not None:
            reduction = float(speeds.water_max)
            starts = np.abs(cell_range) / reduction - margin
            origin = -margin
        elif stated is None:
            origin = float(np.nanmin(first)) - margin
            starts = np.full((n_i, n_j), origin)
        else:
            origin = stated
            starts = np.full((n_i, n_j), origin)
        _, taps = _arrival_cells_impulse_response(
            cells, sample_rate=fs, trace_frequency=self.f0,
            absorption=self.absorption, t_start=starts,
            n_samples=n_samples, band=band, who=who,
            remedy="Lengthen output_duration= or move t_start=, or leave "
                   "both at their defaults to size each record from its "
                   "arrivals.", length_knob="output_duration=")
        time_vec = origin + np.arange(taps.shape[-1]) / fs
        if single is not None:
            from uacpy.core.results.field import Field
            id_kwargs = self.id_kwargs()
            id_kwargs.update(phase_reference=PhaseReference.TIME_DOMAIN_NATIVE,
                             metadata=None, reduction_speed=reduction)
            return Field(data=taps * fs,
                         coords={'depth': np.array([single[0]]),
                                 'range': np.array([single[1]]),
                                 'time': time_vec},
                         kind='impulse_response', unit='1/s', **id_kwargs)
        return self._receiver_field(taps * fs, 'time', time_vec,
                                    kind='impulse_response', unit='1/s',
                                    reduction_speed=reduction)

    def _receiver_field(self, data, axis_name: str, axis_values, *, kind,
                        unit, reduction_speed=None) -> "Field":
        """``data`` ``(n_depth_blocks, n_ranges, n)`` over ``by_receiver``'s
        grid as a Field on ``(depth, range, axis_name)`` — a paired grid's
        one block on ``(range, axis_name)``, its depths on
        ``aux_coords['receiver_depth']`` — with this result's identity and
        ``phase_reference='time_domain_native'``."""
        from uacpy.core.results.field import Field
        depths = np.atleast_1d(np.asarray(self.receiver_depths, dtype=float))
        ranges = np.atleast_1d(np.asarray(self.receiver_ranges, dtype=float))
        id_kwargs = self.id_kwargs()
        id_kwargs.update(phase_reference=PhaseReference.TIME_DOMAIN_NATIVE,
                         metadata=None, reduction_speed=reduction_speed)
        if self._paired_grid:
            return Field(data=data[0], coords={'range': ranges,
                                               axis_name: axis_values},
                         aux_coords={'receiver_depth': ('range', depths)},
                         kind=kind, unit=unit, **id_kwargs)
        return Field(data=data, coords={'depth': depths, 'range': ranges,
                                        axis_name: axis_values},
                     kind=kind, unit=unit, **id_kwargs)

    def _cell_coordinates(self, cell: 'Arrivals', receiver):
        """``(depth, range)`` of the one-cell arrivals ``cell``.

        The arrivals carry indices into the receiver axes, so the labels come
        from the axes when they are populated. An ``Arrivals`` assembled by
        hand may have neither, and a transfer function is still defined for
        it — the coordinates are then the caller's ``receiver=`` or zero,
        which is a placement on the Field's axes and not a claim about
        geometry.
        """
        if receiver is not None:
            return float(receiver[0]), float(receiver[1])
        d_idx = int(cell._cell_index('depth_idx')[0])
        r_idx = int(cell._cell_index('range_idx')[0])
        depths = np.atleast_1d(np.asarray(self.receiver_depths, dtype=float))
        ranges = np.atleast_1d(np.asarray(self.receiver_ranges, dtype=float))
        depth = float(depths[d_idx]) if d_idx < depths.size else 0.0
        rng = float(ranges[r_idx]) if r_idx < ranges.size else 0.0
        return depth, rng

    def channel_taps(
        self,
        symbol_rate: float,
        *,
        fc: float,
        sps: int = 1,
        pulse: Optional[str] = None,
        rolloff: float = 0.25,
        span: int = 8,
        receiver=None,
        normalize: bool = False,
    ) -> ChannelTaps:
        """Baseband channel taps of these arrivals at a symbol rate.

        The discrete-time channel a modem at carrier ``fc`` sees between its
        pulse shaper and its receiver, at ``sps`` samples per symbol
        (``T = 1 / (sps * symbol_rate)``)::

            h[k] = sum_i a_i exp(i phi_i) exp(-i 2 pi f_c tau_i) g(k T - tau_i)

        ``a_i`` is the received amplitude at the carrier, absorption included
        (:attr:`received_amplitudes` evaluated at ``fc`` rather than at
        the result's own frequency, the absorption scaled from :attr:`f0` by
        :attr:`absorption`), ``phi_i`` the arrival phase, ``tau_i``
        the travel time, and ``g`` the transmit pulse. The tap grid starts
        at the earliest arrival (``first_arrival`` records it), but the
        rotation keeps the absolute ``tau_i``: the taps are what a receiver
        mixing with the transmitter's carrier sees, common phase
        ``exp(-i 2 pi f_c tau_first)`` included, and that phase is what
        makes the passband test in ``test_comms.py`` reproducible.

        ``pulse`` names ``g``, and ``None`` (the default) infers it from
        ``sps``:

        - ``'rc'`` — the raised cosine (:func:`~uacpy.comms.rc_pulse`, unit
          peak): transmit root-raised-cosine times its matched filter, so
          the taps are the channel at the receiver's decision instants —
          what :func:`~uacpy.comms.simulate_link` and a symbol-spaced
          equaliser consume. The default at ``sps=1``. Measured against the
          ``sps=16`` root-raised-cosine taps matched-filtered and decimated,
          it agrees to an NMSE of 1e-4 on- and off-grid; the
          root-raised-cosine half alone is 1.7e-2 to 2.4e-2 off, so it is
          not offered as a symbol-spaced default.
        - ``'rrc'`` — the transmit root-raised-cosine alone
          (:func:`~uacpy.comms.rrc_pulse`), normalised as
          :func:`~uacpy.comms.rrc_filter` is (unit energy on the ``sps``
          grid), so an arrival on a sample instant reproduces that filter's
          taps and ``comms.apply_channel(upsampled_symbols, h)`` is the
          pulse-shaped waveform through the channel, for a receiver that
          applies its own matched filter. The default at ``sps > 1``.
        - ``'nearest'`` — no pulse; each arrival on its nearest sample,
          which is :func:`~uacpy.comms.multipath_channel` on
          ``received_amplitudes`` and the re-referenced delays, tap for tap.

        **Sign of the carrier rotation.** The package's time convention is
        ``exp(+i omega t)``: ``arrival_grid_transfer_function`` writes each arrival
        into ``H(f)`` as ``A exp(i(phi - 2 pi f tau))``, and
        :func:`~uacpy.acoustic_signal.delayandsum` places ``a Re{x_a(t - tau)
        exp(i phi)}`` with ``x_a`` the analytic source signal. For a
        passband burst ``x(t) = Re{b(t) exp(i 2 pi f_c t)}`` the analytic
        signal is ``b(t) exp(i 2 pi f_c t)``, so the received passband is
        ``Re{sum_i a_i exp(i phi_i) b(t - tau_i) exp(i 2 pi f_c (t - tau_i))}``
        and demodulating with the same carrier leaves
        ``sum_i a_i exp(i phi_i) exp(-i 2 pi f_c tau_i) b(t - tau_i)``: the
        rotation is ``exp(-i 2 pi f_c tau_i)``, as written above. Under the
        opposite convention it would be the conjugate.

        Parameters
        ----------
        symbol_rate : float
            Symbol rate (Bd).
        fc : float
            Carrier frequency (Hz) the modem mixes with. It sets both the
            per-path rotation and the absorption applied to the amplitudes.
        sps : int, default 1
            Samples per symbol the taps are spaced at. ``1`` is the
            symbol-spaced channel :func:`~uacpy.comms.simulate_link` takes.
        pulse : {'rc', 'rrc', 'nearest', None}, default None
            The pulse ``g``, as listed above; ``None`` takes ``'rc'`` at
            ``sps=1`` and ``'rrc'`` above it.
        rolloff, span : float, int
            The pulse's excess bandwidth and length in symbols. Ignored
            for ``pulse='nearest'``.
        receiver : (float, float), optional
            ``(depth, range)`` of the cell to take, on the result's
            receiver axes. Required when the arrivals span several cells.
        normalize : bool, default False
            Scale the taps to unit energy (``sum |h|^2 = 1``), so the
            received symbols keep the constellation's scale and a slicer
            with no equalizer in front of it decides QAM on the right
            rings. :func:`~uacpy.comms.simulate_link` sets its noise from
            the received power, so the SNR does not depend on this.

        Returns
        -------
        ChannelTaps
            ``(taps, delays, symbol_rate, fc, sps, first_arrival)``.

        Raises
        ------
        ConfigurationError
            Non-positive ``symbol_rate`` or ``fc``, ``sps`` below 1, an
            unknown ``pulse``, several cells without ``receiver=``, or a
            non-finite arrival.

        References
        ----------
        Stojanovic, M. and Preisig, J., "Underwater acoustic communication
        channels: propagation models and statistical characterization",
        IEEE Commun. Mag. 47(1), 2009, sect. II — the sparse tap-delay
        line with per-path gains and delays this samples.
        Proakis, J. G., *Digital Communications*, 4th ed., sect. 14.5 —
        the discrete-time model of a frequency-selective channel as taps
        at the symbol (or fractional-symbol) spacing.

        The pulse-shaped placement on plain arrays is
        :func:`~uacpy.comms.pulse_shaped_taps`; ``pulse='nearest'`` is
        :func:`~uacpy.comms.multipath_channel`.
        """
        who = "Arrivals.channel_taps"
        symbol_rate = float(symbol_rate)
        require_positive(symbol_rate, f"{who} symbol_rate", hint="Bd")
        fc = float(fc)
        require_positive(fc, f"{who} fc (carrier frequency)", hint="Hz")
        if int(sps) != sps or int(sps) < 1:
            raise ConfigurationError(
                f"{who}: sps must be a whole number of samples per symbol, "
                f">= 1; got {sps!r}.")
        sps = int(sps)
        if pulse is None:
            pulse = 'rc' if sps == 1 else 'rrc'
        if pulse not in ('rc', 'rrc', 'nearest'):
            raise ConfigurationError(
                f"{who}: pulse must be 'rc' (raised cosine, the channel at "
                f"the decision instants), 'rrc' (transmit root-raised-"
                f"cosine alone), 'nearest' (no pulse) or None to infer "
                f"from sps; got {pulse!r}.")
        cell = self._one_cell(receiver, who)
        delays = cell.delays
        gains = cell._received_amplitudes_at(fc)
        if not (np.all(np.isfinite(delays)) and np.all(np.isfinite(gains))):
            raise ConfigurationError(
                f"{who}: an arrival carries a non-finite delay or amplitude, "
                f"so the channel is undefined.")
        first = float(delays.min())
        rel = delays - first
        # Baseband rotation of each path by its own carrier delay; the sign
        # is derived in the docstring from the package's exp(+i omega t)
        # convention.
        gains = gains * np.exp(-2j * np.pi * fc * delays)
        fs = sps * symbol_rate
        from uacpy.comms.channel import (
            ChannelTaps, multipath_channel, _pulse_shaped_taps,
        )
        if pulse == 'nearest':
            taps = multipath_channel(gains, rel, sample_rate=fs)
            times = np.arange(taps.size) / fs
        else:
            # The placement is pulse_shaped_taps'; what this branch adds
            # is the carrier rotation and absorption already in `gains`.
            shaped = _pulse_shaped_taps(
                gains, rel, symbol_rate, pulse=pulse, rolloff=rolloff,
                sps=sps, span=span, who=who)
            times, taps = shaped.delays, shaped.taps
        if normalize:
            energy = float(np.sum(np.abs(taps) ** 2))
            if energy <= 0.0:
                raise ConfigurationError(
                    f"{who}: every tap is zero, so there is no energy to "
                    f"normalise to.")
            taps = taps / np.sqrt(energy)
        return ChannelTaps(taps=taps, delays=times, symbol_rate=symbol_rate,
                           fc=fc, sps=sps, first_arrival=first)

    def coherence_bandwidth(self, *, convention: str = 'inverse_spread',
                            factor: Optional[float] = None,
                            receiver=None) -> float:
        """Bandwidth over which the channel's transfer function stays
        correlated, in Hz, as ``1 / (k * tau_rms)`` with ``tau_rms`` the
        :meth:`rms_delay_spread`.

        The default ``k = 1`` is the convention APL-UW and Abraham state:
        "the inverse [of the elongation time] in hertz is a measure of the
        coherence bandwidth of the channel" (APL-UW TR 9407, sect. II.7.b,
        p. II-32) and ``W < 1 / sigma_t = W_c`` (Abraham, *Underwater
        Acoustic Signal Processing*, sect. 8.8.1, Fig. 8.34; 1/(33 ms) =
        30 Hz). The named options ``'rappaport_0.5'``
        (``k = 5``) and ``'rappaport_0.9'`` (``k = 50``) are the
        0.5- and 0.9-correlation rules of Rappaport, *Wireless
        Communications*, 2nd ed., sect. 5.4.3, eqs 5.39-5.40.
        ``factor=k`` sets any other ``k > 0``. ``inf`` for a single arrival
        (no spread, a flat channel), ``nan`` when the spread is.

        Parameters
        ----------
        convention : {'inverse_spread', 'rappaport_0.5', 'rappaport_0.9'}
            Which ``k`` to use. Default ``'inverse_spread'``.
        factor : float, optional
            Explicit ``k > 0``; overrides ``convention``.
        receiver : (float, float), optional
            ``(depth, range)`` of the cell whose channel this is. A
            channel is one receiver's: arrivals spanning several cells are
            refused without it, as :meth:`transfer_function` refuses them,
            because pooling them would read the travel-time differences
            between receivers as multipath spread.

        On a power delay profile from anywhere else this is
        :func:`~uacpy.acoustic_signal.coherence_bandwidth`.
        """
        # Deferred: acoustic_signal pulls scipy, and uacpy's public
        # surface is imported without it (test_lazy_imports).
        from uacpy.acoustic_signal.delay_profile import _coherence_bandwidth
        who = "Arrivals.coherence_bandwidth"
        delays, powers = self._cell_profile(receiver, who)
        return _coherence_bandwidth(
            delays, powers, convention=convention, factor=factor, who=who)

    def channel_regime(self, symbol_rate: float, *,
                       convention: str = 'inverse_spread',
                       factor: Optional[float] = None,
                       rolloff: float = 0.0,
                       receiver=None) -> ChannelRegime:
        """Whether a modem at ``symbol_rate`` sees this channel as flat or
        frequency-selective.

        Compares the signal bandwidth ``(1 + rolloff) * symbol_rate`` with
        :meth:`coherence_bandwidth` under ``convention`` / ``factor``:
        selective when the signal is the wider (Proakis, *Digital
        Communications*, 4th ed., sect. 14.1.2; Stojanovic and Preisig
        2009, sect. II). ``isi_symbols`` is the rms delay spread in symbol
        periods, the number of neighbours each symbol overlaps. The result
        records the convention it was judged under (``factor=k`` is
        recorded as ``'factor=k'``).

        Parameters
        ----------
        symbol_rate : float
            Symbol rate (Bd).
        convention, factor
            As on :meth:`coherence_bandwidth`.
        rolloff : float, default 0.0
            Excess bandwidth of the pulse; ``0`` takes the Nyquist bandwidth
            equal to the symbol rate.
        receiver : (float, float), optional
            ``(depth, range)`` of the cell whose channel this is. A
            channel is one receiver's: arrivals spanning several cells are
            refused without it, as :meth:`transfer_function` refuses them,
            because pooling them would read the travel-time differences
            between receivers as multipath spread.

        On a power delay profile from anywhere else this is
        :func:`~uacpy.acoustic_signal.channel_regime`.
        """
        # Deferred: acoustic_signal pulls scipy, and uacpy's public
        # surface is imported without it (test_lazy_imports).
        from uacpy.acoustic_signal.delay_profile import _channel_regime
        who = "Arrivals.channel_regime"
        delays, powers = self._cell_profile(receiver, who)
        return _channel_regime(
            delays, powers, symbol_rate, convention=convention,
            factor=factor, rolloff=rolloff, who=who)


def _axis_index(axis: np.ndarray, value: float, who: str, name: str) -> int:
    """Index of ``value`` on ``axis``, matched within
    :func:`~uacpy.core.results._base.axis_match_tolerance`."""
    axis = np.asarray(axis, dtype=float).ravel()
    hits = np.flatnonzero(np.abs(axis - value)
                          <= axis_match_tolerance(axis, value))
    if hits.size != 1:
        raise ConfigurationError(
            f"{who}: receiver {name} {value:g} m is not on this result's "
            f"{name} axis {np.array2string(axis, max_line_width=60)}; "
            f"pass one of its values.")
    return int(hits[0])


def _first_arrival_speeds(ranges, first_delays):
    """The :class:`~uacpy.core.results.SoundSpeeds` a transfer function
    built from arrivals states: as ``water_max``, the fastest mean
    horizontal speed ``r / tau`` of a cell's first arrival (cells at a
    positive range with a positive first delay), so a synthesis window
    anchored on ``r / water_max`` opens at or before every cell's first
    arrival; ``None`` when no cell qualifies."""
    from uacpy.core.results.speeds import SoundSpeeds
    r = np.abs(np.asarray(ranges, dtype=float))
    tau = np.asarray(first_delays, dtype=float)
    usable = (r > 0.0) & (tau > 0.0) & np.isfinite(tau)
    if not np.any(usable):
        return None
    return SoundSpeeds(water_max=float(np.max(r[usable] / tau[usable])))


def _pair_index(depths: np.ndarray, ranges: np.ndarray, depth: float,
                rng: float, who: str) -> int:
    """Index ``i`` of the irregular-grid receiver ``(depths[i], ranges[i])``
    at ``(depth, rng)``, each matched within
    :func:`~uacpy.core.results._base.axis_match_tolerance`."""
    depths = np.asarray(depths, dtype=float).ravel()
    ranges = np.asarray(ranges, dtype=float).ravel()
    hits = np.flatnonzero(
        (np.abs(depths - depth) <= axis_match_tolerance(depths, depth))
        & (np.abs(ranges - rng) <= axis_match_tolerance(ranges, rng)))
    if hits.size != 1:
        raise ConfigurationError(
            f"{who}: receiver (depth {depth:g} m, range {rng:g} m) is not "
            f"one of this irregular grid's {depths.size} receivers, whose "
            f"i-th receiver is (receiver_depths[i], receiver_ranges[i]); "
            f"pass one of those pairs.")
    return int(hits[0])


#: The unit of each per-ray scalar a :class:`Rays` export carries.
_RAY_COLUMN_UNITS = {'launch_angle': 'deg', 'miss_distance': 'm'}


class Rays(Result):
    """Ray paths from Bellhop (any backend).

    Pure data container: a list of ray polylines plus the geometric
    context of the run. Filtering helpers return new ``Rays`` objects;
    none of them call back into a solver. To compute "rays at a
    receiver" use :meth:`uacpy.models.PropagationModel.compute_eigenrays`, which
    runs Bellhop's eigenray solver (``RunType='E'``).

    Attributes
    ----------
    rays : list
        Ray dicts with ``r``, ``z``, ``launch_angle``, ``n_top_bounces``,
        ``n_bot_bounces``. **Polyline coordinates ``r`` (range) and
        ``z`` (depth) are in metres**; ``launch_angle`` is the launch angle
        in degrees. Bellhop writes the polyline as ``ray2D%x`` in metres
        (``Bellhop/WriteRay.f90:45``) behind a take-off angle already
        converted to degrees (``Bellhop/bellhop.f90:263``), and the reader
        (:func:`uacpy.io.oalib_reader.read_ray_file`) passes both through
        unconverted — so downstream helpers such as
        :meth:`filter_by_miss_distance` work in metres without any
        unit detection.
    is_eigen : bool
        ``True`` for output of Bellhop's eigenray solver (``RunType='E'``),
        ``False`` for a regular ray fan (``RunType='R'``). Set by the
        wrapper from the run type, not by post-processing.
    receiver_depths, receiver_ranges : ndarray or None
        Receiver geometry the run targeted, when available. ``None``
        when the ``Rays`` came from a standalone reader call without
        receiver context.
    """

    #: The plotter that draws this result (``.plot()``).
    _plotter = 'plot_rays'

    #: Its view takes ``env=`` and draws the environment under the rays.
    _draws_env = True

    def __init__(
        self,
        *,
        rays: List[Any],
        is_eigen: bool = False,
        receiver_depths: Optional[np.ndarray] = None,
        receiver_ranges: Optional[np.ndarray] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rays = list(rays)
        self.is_eigen = bool(is_eigen)
        self.receiver_depths = (
            np.atleast_1d(np.asarray(receiver_depths, dtype=float))
            if receiver_depths is not None else None
        )
        self.receiver_ranges = (
            np.atleast_1d(np.asarray(receiver_ranges, dtype=float))
            if receiver_ranges is not None else None
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise these rays to plain arrays.

        The polylines are ragged, so they are concatenated: ``r`` and ``z``
        (metres) hold every vertex of every ray in order, and
        ``ray_lengths`` says how many belong to each. Every scalar key that
        EVERY ray carries (``launch_angle``, ``n_top_bounces``, ``n_bot_bounces``,
        and ``miss_distance`` after a miss sort) becomes one column in
        ``columns``. Then ``is_eigen``, the receiver geometry and the
        identity, as :meth:`Field.to_dict` writes it. ``np.savez(f, **d)``
        stores it; read it back with ``np.load(f, allow_pickle=True)`` into
        :meth:`from_dict`. That load unpickles, so use it only on files you
        wrote; :meth:`to_netcdf` / :meth:`from_netcdf` is the route for a
        file from elsewhere.
        """
        rays = self.rays
        common = [key for key in (rays[0] if rays else {})
                  if key not in ('r', 'z')
                  and all(key in ray and np.ndim(ray[key]) == 0
                          for ray in rays)]

        def vertices(key):
            if not rays:
                return np.zeros(0)
            return np.concatenate([np.asarray(ray[key], dtype=float).ravel()
                                   for ray in rays])

        return {
            'r': vertices('r'),
            'z': vertices('z'),
            'ray_lengths': np.array([np.size(ray['r']) for ray in rays],
                                    dtype=int),
            'columns': {key: np.asarray([ray[key] for ray in rays])
                        for key in common},
            'is_eigen': self.is_eigen,
            'receiver_depths': (None if self.receiver_depths is None
                                else self.receiver_depths.copy()),
            'receiver_ranges': (None if self.receiver_ranges is None
                                else self.receiver_ranges.copy()),
            **self._identity_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Rays':
        """Reconstruct :class:`Rays` from :meth:`to_dict` output, or from the
        mapping ``np.load(f, allow_pickle=True)`` returns for a file written
        with ``np.savez(f, **rays.to_dict())``. Load such a file only if you
        wrote it: ``allow_pickle=True`` unpickles.

        Parameters
        ----------
        d : mapping
            :meth:`to_dict` output, or the mapping ``np.load`` returns for it.
        """
        d = cls._unwrap_saved(d, payload=('r', 'z', 'ray_lengths'),
                              required=('columns', 'is_eigen',
                                        'receiver_depths', 'receiver_ranges'))
        return cls._from_vertices(
            d['r'], d['z'], d['ray_lengths'], d['columns'],
            is_eigen=bool(d['is_eigen']),
            receiver_depths=d['receiver_depths'],
            receiver_ranges=d['receiver_ranges'],
            **cls._identity_from_dict(d))

    @classmethod
    def _from_vertices(cls, r, z, ray_lengths, columns, **kwargs) -> 'Rays':
        """The rays held as one vertex list ``r``, ``z`` cut at
        ``ray_lengths``, each ray's scalar entries from its row of the
        per-ray ``columns``; ``kwargs`` go to the constructor."""
        bounds = np.concatenate([[0], np.cumsum(np.asarray(
            ray_lengths, dtype=int))])
        r, z = np.asarray(r, dtype=float), np.asarray(z, dtype=float)
        rays = []
        for i in range(bounds.size - 1):
            ray = {'r': r[bounds[i]:bounds[i + 1]].copy(),
                   'z': z[bounds[i]:bounds[i + 1]].copy()}
            ray.update({key: np.asarray(col)[i].item()
                        for key, col in (columns or {}).items()})
            rays.append(ray)
        return cls(rays=rays, **kwargs)

    def _repr_bits(self) -> list:
        bits = [count(len(self.rays), 'eigenray' if self.is_eigen else 'ray')]
        for name in ('receiver_depth', 'receiver_range'):
            values = getattr(self, name + 's')
            if values is not None:
                bits.append(coordinate_axis(name, values))
        return bits

    def __len__(self) -> int:
        return self.n_rays

    @property
    def n_rays(self) -> int:
        """How many rays (or eigenrays) these are."""
        return len(self.rays)

    # Per-ray bulk views ----------------------------------------------------

    def _ray_column(self, key: str, default=np.nan, dtype=float) -> np.ndarray:
        return read_only(np.array([ray.get(key, default) for ray in self.rays],
                                  dtype=dtype))

    @property
    def launch_angles_deg(self) -> np.ndarray:
        """Launch angle (deg) of every ray; NaN for a ray without one."""
        return self._ray_column('launch_angle')

    @property
    def n_top_bounces(self) -> np.ndarray:
        """Surface reflections of every ray."""
        return self._ray_column('n_top_bounces', 0, int)

    @property
    def n_bot_bounces(self) -> np.ndarray:
        """Bottom reflections of every ray."""
        return self._ray_column('n_bot_bounces', 0, int)

    @property
    def kinds(self) -> np.ndarray:
        """The multipath class of every ray, from its bounce counts:
        ``'direct'``, ``'surface'``, ``'bottom'`` or ``'both'``, the rule
        :attr:`Arrivals.kinds` and :meth:`filter_by_bounces` use."""
        top, bot = self.n_top_bounces, self.n_bot_bounces
        return read_only(np.array(
            [_arrival_kind(int(t), int(b)) for t, b in zip(top, bot)],
            dtype=object))

    @property
    def lengths(self) -> np.ndarray:
        """Path length (m) of every ray: the summed lengths of its polyline
        segments."""
        return read_only(np.array([
            float(np.sum(np.hypot(np.diff(np.asarray(ray['r'], dtype=float)),
                                  np.diff(np.asarray(ray['z'], dtype=float)))))
            for ray in self.rays], dtype=float))

    @property
    def miss_distances(self) -> np.ndarray:
        """Each ray's miss distance (m) to the target a miss filter or
        sort measured it against; NaN for a ray none has measured."""
        return self._ray_column('miss_distance')

    def _table(self):
        """One row per ray: ``ray`` (its index), ``launch_angle`` (deg),
        ``n_top_bounces``, ``n_bot_bounces``, ``length`` (m),
        ``miss_distance`` (m, NaN unmeasured) and ``n_vertices``. The
        polylines themselves are in :meth:`to_xarray`."""
        return {
            'ray': np.arange(self.n_rays),
            'launch_angle': np.asarray(self.launch_angles_deg).copy(),
            'n_top_bounces': np.asarray(self.n_top_bounces).copy(),
            'n_bot_bounces': np.asarray(self.n_bot_bounces).copy(),
            'length': np.asarray(self.lengths).copy(),
            'miss_distance': np.asarray(self.miss_distances).copy(),
            'n_vertices': np.array([np.size(ray['r']) for ray in self.rays],
                                   dtype=int),
        }

    def _table_units(self):
        return {'ray': '', 'launch_angle': 'deg', 'n_top_bounces': '',
                'n_bot_bounces': '', 'length': 'm', 'miss_distance': 'm',
                'n_vertices': ''}

    def _payload(self):
        d = self.to_dict()
        ray_of_vertex = np.repeat(np.arange(self.n_rays), d['ray_lengths'])
        return {
            'r': (d['r'], ('vertex',), 'm'),
            'z': (d['z'], ('vertex',), 'm'),
            'ray_index': (ray_of_vertex, ('vertex',), ''),
            **{key: (column, ('ray',), _RAY_COLUMN_UNITS.get(key, ''))
               for key, column in d['columns'].items()},
        }

    def _export_attrs(self):
        attrs = super()._export_attrs()
        attrs['is_eigen'] = int(self.is_eigen)
        return attrs

    def _coords(self):
        coords = {}
        if self.receiver_depths is not None:
            coords['receiver_depth'] = (self.receiver_depths, 'm')
        if self.receiver_ranges is not None:
            coords['receiver_range'] = (self.receiver_ranges, 'm')
        return coords

    @classmethod
    def _from_export(cls, arrays, attrs):
        index = np.asarray(arrays['ray_index'], dtype=int)
        columns = {key: values for key, values in arrays.items()
                   if key not in ('r', 'z', 'ray_index', 'receiver_depth',
                                  'receiver_range')}
        # A ray of one vertex still has its row in every per-ray column.
        n_rays = max([(int(index.max()) + 1) if index.size else 0]
                     + [len(values) for values in columns.values()])
        return cls._from_vertices(
            arrays['r'], arrays['z'], np.bincount(index, minlength=n_rays),
            columns, is_eigen=bool(attrs.get('is_eigen', 0)),
            receiver_depths=arrays.get('receiver_depth'),
            receiver_ranges=arrays.get('receiver_range'),
            **cls._identity_from_attrs(attrs, ('is_eigen',)))

    # ------------------------------------------------------------------
    # Filtering helpers — pure data subsets. ``is_eigen`` is preserved
    # (a subset of a fan stays a fan; a subset of eigenrays stays
    # eigenrays). None of these accept receiver coordinates: geometric
    # "rays at a receiver" is what ``PropagationModel.compute_eigenrays`` is for.
    # ------------------------------------------------------------------

    def filter(self, predicate) -> 'Rays':
        """Return a new ``Rays`` keeping rays for which ``predicate(ray)`` is true.

        Parameters
        ----------
        predicate : callable
            ``predicate(ray) -> bool``.
        """
        kept = [r for r in self.rays if predicate(r)]
        return self._spawn(kept)

    def filter_by_bounces(
        self,
        kind: Optional[str] = None,
        top: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None,
        bot: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None,
    ) -> 'Rays':
        """Subset by multipath component.

        ``kind`` ∈ ``{'direct', 'surface', 'bottom', 'both'}`` keeps a
        qualitative bounce class.

        ``top`` / ``bot`` further constrain the exact bounce count on
        each boundary:

        * ``None``       — any count
        * ``int``        — exact match (e.g. ``top=2``)
        * ``(lo, hi)``   — closed range; ``None`` on either end is
                           unbounded. ``bot=(1, None)`` keeps rays with
                           at least one bottom bounce; ``top=(0, 1)``
                           keeps 0–1 surface bounces.

        Parameters
        ----------
        kind : {'direct', 'surface', 'bottom', 'both'}, optional
            Bounce class to keep; ``None`` keeps every class.
        top, bot : int or (int, int), optional
            Surface / bottom bounce count: exact, or a closed ``(lo, hi)`` range
            with ``None`` for an open end; ``None`` is any count.
        """
        if self.rays and not any(
            'n_top_bounces' in r or 'n_bot_bounces' in r for r in self.rays
        ):
            warnings.warn(
                "Rays.filter_by_bounces: rays carry no bounce counts, so "
                "every ray classifies as 'direct'. A .ray file read through "
                "uacpy.io.read_ray_file always supplies them; a hand-built "
                "Rays must set 'n_top_bounces' / 'n_bot_bounces' per ray.",
                FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
        pred = _bounce_predicate(kind, top, bot)
        return self.filter(
            lambda r: pred(
                int(r.get('n_top_bounces', 0) or 0),
                int(r.get('n_bot_bounces', 0) or 0),
            )
        )

    def window(self, *, launch_angle_deg) -> 'Rays':
        """Keep the rays whose launch angle ``launch_angle`` (degrees) falls inside
        the inclusive ``(lo, hi)`` pair, either end ``None`` to leave it
        open — the window rule of :meth:`Field.window`.

        Parameters
        ----------
        launch_angle_deg : (float, float)
            Inclusive launch-angle window (deg); ``None`` leaves that end open.
        """
        low, high = _window_pair('Rays.window', 'launch_angle_deg',
                                 launch_angle_deg)
        if self.rays and not any('launch_angle' in r for r in self.rays):
            warnings.warn(
                "Rays.window: rays carry no launch angles, "
                "so the filter drops every ray. A .ray file read through "
                "uacpy.io.read_ray_file always supplies 'launch_angle'; a "
                "hand-built Rays must set it per ray.",
                FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
        def pred(ray):
            a = ray.get('launch_angle')
            return (a is not None and (low is None or a >= low)
                    and (high is None or a <= high))
        return self.filter(pred)

    def first_n(
        self,
        n: int = 10
    ) -> 'Rays':
        """Keep only the first ``n`` rays, in the order currently held.

        Meaningful after a sort: ``sorted_by_miss(...).first_n(n)`` is
        the ``n`` closest rays, which is what :meth:`top_n_by_miss` wraps.

        On an untouched fan the order is launch angle, so this keeps one EDGE
        of the fan rather than a spread across it — measured, 41 rays of a
        5001-ray ±76.6° fan span 1.2°, which plots as a narrow beam aimed one
        way rather than as the fan. For a slice of the fan use
        :meth:`window`; for the ``n`` nearest a receiver,
        :meth:`top_n_by_miss`.

        Parameters
        ----------
        n : int, optional
            Rays to keep. Default 10.
        """
        return self._spawn(self.rays[:_count(n, 'Rays.first_n')])

    def _miss_distance_to(
        self, ray, target_range: float, target_depth: float,
    ) -> Tuple[float, int]:
        """Closest approach of the ray to a point, and the vertex index there.

        Measured to the polyline's SEGMENTS, not to its vertices
        (:func:`~uacpy.core.acoustics.polyline_miss_distance`). A ray that
        passes exactly through the receiver still has its nearest stored point
        half a step away, so a vertex-only distance reports the ray step
        rather than the miss. Measured on a flat 1500 m case at 40 kHz with a
        0.5 m step, the single-surface-bounce eigenray came back 0.188 m from
        a receiver it passes through, and a 10 cm
        :meth:`filter_by_miss_distance` therefore discarded it as a miss while
        keeping other classes whose vertices happened to fall closer. The
        floor tracked the step (1.0 m at the 30 m default), which is the
        signature of a sampling artefact rather than a real miss.

        The returned index is still a VERTEX index — the one nearest the
        closest approach — because :meth:`truncate_at_receiver` clips the
        polyline there.

        Ray polylines are required to carry ``r`` / ``z`` in **metres**
        (see :class:`Rays` docstring). The Bellhop reader in
        :mod:`uacpy.io.oalib_reader` already preserves Bellhop's native
        metres, so no unit-detection heuristic is needed here.
        """
        return polyline_miss_distance(ray.get('r', []), ray.get('z', []),
                                      target_range, target_depth)

    def distinct_paths(
        self,
        *,
        target_depth: Optional[float] = None,
        target_range: Optional[float] = None,
    ) -> 'Rays':
        """One ray per physical path: the closest-approaching of each.

        With the default beam (geometric, hat-shaped), Bellhop writes both
        bracketing rays of the ray tube that encloses the receiver: "When we
        use the default beam type, the rays that are written will be only
        the bracketing rays for the receiver location" (Bellhop User Guide,
        §3 "Eigenray plots"; other beam types write "lots of additional
        rays") — so a dense fan reports one path once per bracketing ray. On
        a 1 km near-bottom link at 40 kHz a 2.56M-beam fan returns 40 rays
        for 20 paths, never more than 2 per path. For the ARRIVALS list the
        same beam gives "arrivals [that] come in pairs", and "there is also
        logic inside BELLHOP to try and combine such pairs into a single
        arrival" (ibid., §9.3 "Generating a receiver timeseries"); the ray
        output keeps both, because a picture wants the tube while a count
        wants the path.

        A path is keyed by its bounce counts and the direction it left the
        source: over a flat seabed those identify it, and the four members of
        one surface order — ``(n, n-1, down)``, ``(n, n, down)``,
        ``(n, n, up)``, ``(n, n+1, up)`` — differ only by whether a 1 m detour
        to the bed happens at each end. The survivor of each group is the ray
        passing closest to the target, so the kept geometry is the best the
        fan resolved.

        ``target_range`` / ``target_depth`` default to the single-point
        receiver this ``Rays`` was built for, as in :meth:`sorted_by_miss`.

        Parameters
        ----------
        target_depth, target_range : float, optional
            The target point (m); ``None`` is the receiver this result was built
            for.

        Raises
        ------
        ConfigurationError
            For a ray fan (``is_eigen`` false). A fan's rays are samples of a
            continuum rather than paths that reach a receiver, so grouping
            them by bounce count would collapse the picture to a handful of
            rays; use :meth:`window` to subset one instead.
        """
        if not self.is_eigen:
            raise ConfigurationError(
                "Rays.distinct_paths: this Rays is a ray fan, not an "
                "eigenray set — its rays sample a continuum rather than "
                "reaching the receiver, so there are no paths to collapse "
                "to. Run RunMode.EIGENRAYS, or subset the fan with "
                "window(launch_angle_deg=...)."
            )
        seen = set()
        kept = []
        for ray in self.sorted_by_miss(target_range=target_range,
                                       target_depth=target_depth).rays:
            key = (ray['n_top_bounces'], ray['n_bot_bounces'],
                   ray['launch_angle'] >= 0.0)
            if key not in seen:
                seen.add(key)
                kept.append(ray)
        return self._spawn(kept)

    def _resolve_target(
        self,
        target_range: Optional[float],
        target_depth: Optional[float],
    ) -> Tuple[float, float]:
        """Default target to the receiver context when this Rays was built
        from a single-point eigenray query."""
        if target_range is None:
            if self.receiver_ranges is None or len(self.receiver_ranges) != 1:
                raise ConfigurationError(
                    "Rays.miss-distance helpers: target_range must be "
                    "supplied unless this Rays carries a single-point "
                    "receiver context."
                )
            target_range = float(self.receiver_ranges[0])
        if target_depth is None:
            if self.receiver_depths is None or len(self.receiver_depths) != 1:
                raise ConfigurationError(
                    "Rays.miss-distance helpers: target_depth must be "
                    "supplied unless this Rays carries a single-point "
                    "receiver context."
                )
            target_depth = float(self.receiver_depths[0])
        return target_range, target_depth

    def filter_by_miss_distance(
        self,
        max_miss: float,
        *,
        target_depth: Optional[float] = None,
        target_range: Optional[float] = None,
    ) -> 'Rays':
        """Keep rays whose closest approach to the target is ``≤ max_miss``.

        Each kept ray gets a ``miss_distance`` entry attached. Target
        defaults to the single-point receiver this ``Rays`` was built for.
        The target is keyword-only on every miss-distance helper, so a
        range and a depth cannot be swapped by position.

        Parameters
        ----------
        max_miss : float
            Largest miss distance (m) kept.
        target_depth, target_range : float, optional
            The target point (m); ``None`` is the receiver this result was built
            for.
        """
        tr, td = self._resolve_target(target_range, target_depth)
        kept = []
        for ray in self.rays:
            miss, _ = self._miss_distance_to(ray, tr, td)
            if miss <= max_miss:
                ray = dict(ray)
                ray['miss_distance'] = miss
                kept.append(ray)
        return self._spawn(kept)

    def sorted_by_miss(
        self,
        *,
        target_depth: Optional[float] = None,
        target_range: Optional[float] = None,
    ) -> 'Rays':
        """Return rays sorted by ascending miss-distance to the target.

        Each ray gets ``miss_distance`` attached. Target defaults to
        the single-point receiver this ``Rays`` was built for. Compose
        with ``first_n`` to cap, or ``truncate_at_receiver`` to
        clip polylines.

        Parameters
        ----------
        target_depth, target_range : float, optional
            The target point (m); ``None`` is the receiver this result was built
            for.
        """
        tr, td = self._resolve_target(target_range, target_depth)
        scored = []
        for ray in self.rays:
            miss, _ = self._miss_distance_to(ray, tr, td)
            ray = dict(ray)
            ray['miss_distance'] = miss
            scored.append((miss, ray))
        scored.sort(key=lambda t: t[0])
        return self._spawn([r for _, r in scored])

    def top_n_by_miss(
        self,
        n: int,
        *,
        target_depth: Optional[float] = None,
        target_range: Optional[float] = None,
    ) -> 'Rays':
        """Return the ``n`` rays with smallest miss-distance to the target.

        Equivalent to ``self.sorted_by_miss(...).first_n(n)``.
        Target defaults to the single-point receiver this ``Rays`` was
        built for.

        Parameters
        ----------
        n : int
            Rays to keep.
        target_depth, target_range : float, optional
            The target point (m); ``None`` is the receiver this result was built
            for.
        """
        n = _count(n, 'Rays.top_n_by_miss')
        return self.sorted_by_miss(
            target_range=target_range,
            target_depth=target_depth).first_n(n)

    def truncate_at_receiver(
        self,
        *,
        target_depth: Optional[float] = None,
        target_range: Optional[float] = None,
    ) -> 'Rays':
        """Clip each ray polyline at its closest-approach index.

        Target defaults to the single-point receiver this ``Rays`` was
        built for. Useful before plotting eigenrays so each path stops
        at the receiver instead of running off to its full extent.

        Parameters
        ----------
        target_depth, target_range : float, optional
            The target point (m); ``None`` is the receiver this result was built
            for.
        """
        tr, td = self._resolve_target(target_range, target_depth)
        clipped = []
        for ray in self.rays:
            miss, k = self._miss_distance_to(ray, tr, td)
            ray = dict(ray)
            ray['miss_distance'] = miss
            r = np.asarray(ray.get('r', []))
            z = np.asarray(ray.get('z', []))
            if k + 1 < len(r):
                ray['r'] = r[:k + 1]
                ray['z'] = z[:k + 1]
            clipped.append(ray)
        return self._spawn(clipped)

    def _spawn(self, rays: List[Any]) -> 'Rays':
        """Build a new ``Rays`` from a subset, preserving identification."""
        return Rays(
            rays=rays,
            is_eigen=self.is_eigen,
            receiver_depths=self.receiver_depths,
            receiver_ranges=self.receiver_ranges,
            **self.id_kwargs(),
        )
