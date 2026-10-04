"""The run modes, and the settings a model run resolves once, before
anything is launched.

:class:`RunMode` names what a run computes.
:meth:`~uacpy.models.base.PropagationModel.run` decides every fact about a
call in one place — the run mode, the frequency grid, the source-depth loop,
the source weights, the time-series request — and records them in a
:class:`RunSettings`. :meth:`~uacpy.models.base.PropagationModel.run_settings`
returns that record without launching anything, and every result a run
returns carries the one it was produced with, as the read-only
``result.run_settings``.

Every object here is immutable: dataclasses are frozen and their arrays are
read-only copies. None of them holds a carrier (an ``Environment``,
``Source`` or ``Receiver``): the settings travel on results, which carry no
carriers.

This module imports nothing above ``core``, so ``uacpy.core.results``
rebuilds a stamped result's settings from its ``to_dict`` form without
loading a model.
"""

from __future__ import annotations

import dataclasses
import importlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, NamedTuple, Optional, Tuple

import numpy as np

from uacpy.core import exceptions as _exceptions
from uacpy.core._export import require_extra, saved_class_path
from uacpy.core._records import FrozenMapping, FrozenRecord, array_summary
from uacpy.core._repr import build, extent
from uacpy.core.exceptions import ConfigurationError, UACPYWarning

__all__ = [
    'RunMode', 'RunSettings', 'TimeSettings', 'WaveguideSpeeds', 'OutputSpec',
    'EngineSettings', 'Notice', 'DEPTH_LOOPS',
]


class RunMode(str, Enum):
    """
    Standard run modes for acoustic propagation models.

    Models may support a subset of these modes. Inherits from ``str`` so a
    member equals its value (``RunMode.COHERENT_TL == 'coherent_tl'``): a
    ``Result.run_mode`` read back from a file as the plain string still
    compares equal to the member.
    """
    COHERENT_TL = 'coherent_tl'          # Coherent transmission loss
    INCOHERENT_TL = 'incoherent_tl'      # Incoherent (averaged) TL
    # Incoherent beam sum (Bellhop/influence.f90:139-141, same as 'I') with a
    # Lloyd-mirror source directivity folded into the launch amplitude
    # (Bellhop/bellhop.f90:276-278).
    SEMICOHERENT_TL = 'semicoherent_tl'

    RAYS = 'rays'                        # Ray paths only
    EIGENRAYS = 'eigenrays'              # Eigenrays (specific paths)
    ARRIVALS = 'arrivals'                # Arrival structure

    MODES = 'modes'                      # Normal modes (Kraken depth eigenfunctions)

    # Frequency-domain array products. COVARIANCE → C(f, i, j) hydrophone ×
    # hydrophone matrix, declared by two OASES sub-models: OASN builds it from
    # the noise sources, OASS from the reverberant field (REVCOV, product
    # letter 'a'); OASES.for_mode picks between them on reverberation=.
    # REPLICA → OASN alone: Green's-function samples at the array elements per
    # candidate source position. See core/results.Covariance and
    # core/results.Replicas.
    COVARIANCE = 'covariance'
    REPLICA = 'replica'

    # Time-domain pressure p(t) at the receiver(s). Models that compute a
    # broadband transfer function natively (Bellhop, RAM, Scooter,
    # Kraken, OASES) require ``source_waveform=`` + ``sample_rate=``;
    # SPARC marches p(t) directly: ``source_waveform`` itself when its
    # ``pulse_type`` is unpinned, else its own pulse.
    TIME_SERIES = 'time_series'

    # Broadband complex transfer function H(f).
    BROADBAND = 'broadband'

    REFLECTION = 'reflection'            # Plane-wave reflection coefficients (Bounce, OASR)

    # Scattered / reverberant field from rough interfaces (OASS; OASSP emits
    # its scattered field through BROADBAND / TIME_SERIES instead). Unlike
    # every other mode this is a two-stage run: the scattering kernel is a
    # post-processor over a .rhs written by a preceding OAST/OASR mean-field
    # run with option 's'.
    REVERBERATION = 'reverberation'


#: How ``RunSettings.depth_loop`` names the three ways a run treats the
#: source depths: one depth; one engine run per depth, stacked by the base
#: class; every depth handed to the engine, which stacks them itself (one
#: deck, or its own loop).
DEPTH_LOOPS = ('single', 'per_depth', 'engine')


@dataclass(frozen=True, eq=False)
class TimeSettings(FrozenRecord):
    """The TIME_SERIES request of a run.

    Attributes
    ----------
    source_waveform : ndarray or None
        The source pulse (Pa), zero-padded to ``output_duration`` when that
        is longer, as the synthesis uses it; ``None`` when the run was given
        none (SPARC then marches its own ``pulse_type``).
    sample_rate : float or None
        Sample rate (Hz) of ``source_waveform``.
    output_duration : float or None
        The record length (s) the caller asked for.
    t_start : float or None
        The time (s after emission) of the record's first sample; ``None``
        opens the record just before the earliest arrival.
    """

    source_waveform: Optional[np.ndarray] = None
    sample_rate: Optional[float] = None
    output_duration: Optional[float] = None
    t_start: Optional[float] = None

    _ARRAY_FIELDS = ('source_waveform',)

    def __post_init__(self):
        self._freeze_arrays()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TimeSettings':
        return cls(**d)

    def summary(self) -> str:
        bits = []
        if self.source_waveform is not None:
            bits.append(f"pulse {self.source_waveform.size} samples")
        if self.sample_rate is not None:
            bits.append(f"at {self.sample_rate:g} Hz")
        if self.output_duration is not None:
            bits.append(f"record {self.output_duration:g} s")
        bits.append('t_start auto' if self.t_start is None
                    else f"t_start {self.t_start:g} s")
        return ', '.join(bits)

    def __repr__(self) -> str:
        return build('TimeSettings', [self.summary()])


@dataclass(frozen=True, eq=False)
class WaveguideSpeeds(FrozenRecord):
    """Slowest and fastest compressional speeds (m/s) of the environment
    the engine ran on: water column plus every geoacoustic bottom layer."""

    c_min: float
    c_max: float

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'WaveguideSpeeds':
        return cls(**d)

    def summary(self) -> str:
        return f"c {self.c_min:g}-{self.c_max:g} m/s"

    def __repr__(self) -> str:
        return build('WaveguideSpeeds',
                     [f"c={extent([self.c_min, self.c_max], 'm/s')}"])


@dataclass(frozen=True, eq=False)
class OutputSpec(FrozenRecord):
    """What a run returns, fixed before the engine is launched.

    Attributes
    ----------
    result_type : str
        Name of the result class (``'Field'``, ``'Modes'``, ``'Rays'``, …).
    kind : str or None
        The quantity (``'pressure'``, …) for a ``Field``.
    unit : str or None
        ``'Pa'`` or ``'dB'`` for a ``Field``.
    phase_reference : str or None
        The phase convention of complex pressure
        (:class:`~uacpy.core.results.PhaseReference` value).
    coherent : bool or None
        Whether a pressure field is a coherent sum; ``None`` for anything
        that is not a pressure field.
    """

    result_type: str
    kind: Optional[str] = None
    unit: Optional[str] = None
    phase_reference: Optional[str] = None
    coherent: Optional[bool] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'OutputSpec':
        return cls(**d)

    def summary(self) -> str:
        bits = [self.result_type]
        for name in ('kind', 'unit', 'phase_reference'):
            value = getattr(self, name)
            if value is not None:
                bits.append(str(value))
        if self.coherent is not None:
            bits.append('coherent' if self.coherent else 'incoherent')
        return ' '.join(bits)

    def __repr__(self) -> str:
        return build('OutputSpec', [self.summary()])


class Notice(NamedTuple):
    """One condition of a run's settings, as its settings record it
    (``note``, a short line of ``print(settings)``, or ``None``) and as the
    run warns of it (``message``, announced once by ``run`` and
    ``run_settings``, as a ``category`` warning). A stage-3 resolver returns
    it, or ``None`` when there is nothing to say."""
    note: Optional[str]
    message: str
    category: type = UACPYWarning

    def __repr__(self) -> str:
        return build('Notice', [None if self.note is None else repr(self.note),
                                repr(self.message), self.category.__name__])


def _saved_notice(item) -> Notice:
    """``item`` as a :class:`Notice`: one already, the ``{note, message,
    category}`` dict :meth:`EngineSettings.to_dict` writes (``category`` by
    class name; a name that is not a uacpy warning class reads as
    :class:`UACPYWarning`), or a plain message string, as a record saved
    before notices carried their class wrote them (:class:`UACPYWarning`, no
    note)."""
    if isinstance(item, Notice):
        return item
    if isinstance(item, str):
        return Notice(None, item, UACPYWarning)
    category = getattr(_exceptions, str(item.get('category')), None)
    if not (isinstance(category, type) and issubclass(category, UACPYWarning)):
        category = UACPYWarning
    return Notice(item.get('note'), item['message'], category)


def _with_saved_notes(notices, notes):
    """The notices of a record saved with a separate ``notes`` tuple. Its
    writer kept one note per notice that had one, in order, so equal
    counts pair them one to one; otherwise which message a note belonged to
    was not saved, and each note is kept as a notice of its own."""
    notices = [_saved_notice(n) for n in notices]
    notes = list(notes)
    if not notes:
        return notices
    if len(notes) == len(notices):
        return [n._replace(note=note) for n, note in zip(notices, notes)]
    return notices + [Notice(note, note, UACPYWarning) for note in notes]


def _block(name: str, lines) -> str:
    """The multi-line repr of a run record: ``name(``, one aligned
    ``label  text`` line per entry, ``)``."""
    lines = list(lines)
    width = max((len(label) for label, _ in lines), default=0)
    body = '\n'.join(f"  {label:<{width}}  {text}" for label, text in lines)
    return f"{name}(\n{body}\n)"


@dataclass(frozen=True, eq=False)
class EngineSettings(FrozenRecord):
    """Base of the per-engine part of :class:`RunSettings`.

    An engine that resolves its own settings (grids, phase-speed windows,
    beams, the binary it launches) subclasses this as a frozen dataclass
    and returns it from :meth:`PropagationModel._resolve_engine_settings`.
    Array fields are listed in ``_ARRAY_FIELDS`` so they are stored as
    read-only copies.

    Attributes
    ----------
    notices : tuple of Notice
        Each condition of these settings the run will warn about, decided
        while the settings are resolved: its short ``note`` (or ``None``),
        and its ``message``, announced once as a ``category`` warning by
        ``run`` and ``run_settings`` (``validate_inputs`` announces
        nothing). Keyword-only.
    knobs : mapping of str to value
        Every knob of the model that ran, by its constructor name, as given
        — ``None`` where the run derived it, the derived value being the
        field of the same name — in plain types (read-only). Host knobs that
        cannot change a result (scratch location, logging, timeout, binary
        path) are not recorded. Keyword-only.
    """

    notices: Tuple[Notice, ...] = dataclasses.field(default=(), kw_only=True)
    knobs: Mapping[str, Any] = dataclasses.field(
        default_factory=FrozenMapping, kw_only=True)

    #: The field holding one record per binary launch, which
    #: :meth:`to_dataframe` tabulates; ``None`` for an engine that resolves
    #: no per-launch records.
    _TABLE_FIELD = None

    def __post_init__(self):
        object.__setattr__(self, 'notices',
                           tuple(_saved_notice(n) for n in self.notices))
        object.__setattr__(self, 'knobs', FrozenMapping(self.knobs))
        self._freeze_arrays()

    def to_dataframe(self):
        """One row per binary launch as a ``pandas.DataFrame`` (optional
        extra ``uacpy[xarray]``): one column per field of the launch record,
        in field order; an array or a tuple (a band, a per-profile ``c_high``)
        is one cell.

        Raises
        ------
        ConfigurationError
            The engine resolves no per-launch records.
        """
        who = f"{type(self).__name__}.to_dataframe"
        if self._TABLE_FIELD is None:
            raise ConfigurationError(
                f"{who}: a {type(self).__name__} holds no per-launch "
                f"records.",
                remediation="Use to_dict() for the settings as plain types.")
        pandas = require_extra('pandas', who)
        records = getattr(self, self._TABLE_FIELD)
        names = [f.name for f in dataclasses.fields(records[0])]
        return pandas.DataFrame(
            [[getattr(r, name) for name in names] for r in records],
            columns=names)

    def to_dict(self) -> Dict[str, Any]:
        """The record as plain types: each notice as ``{note, message,
        category}``, the category by class name; ``knobs`` is left out when
        it is empty, as on a record saved before the knobs were recorded, so
        such a record writes back what was saved."""
        d = super().to_dict()
        d['notices'] = [{'note': n.note, 'message': n.message,
                         'category': n.category.__name__}
                        for n in self.notices]
        if not d['knobs']:
            del d['knobs']
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'EngineSettings':
        """Rebuild the record from :meth:`to_dict` output, or from a record
        saved with the notes beside the notices (see :func:`_with_saved_notes`)."""
        d = dict(d)
        if 'notes' in d:
            d['notices'] = _with_saved_notes(d.get('notices', ()), d.pop('notes'))
        return cls(**d)

    def summary_lines(self):
        """One ``(name, text)`` pair per field, for the multi-line repr,
        the engine's own fields first and ``notices`` last."""
        out = []
        for f in sorted(dataclasses.fields(self),
                        key=lambda f: f.name in ('knobs', 'notices')):
            value = getattr(self, f.name)
            if f.name == 'notices':
                text = repr(tuple(n.message for n in value))
            elif isinstance(value, Mapping):
                text = repr(dict(value))
            elif isinstance(value, np.ndarray):
                text = array_summary(value, '')
            elif dataclasses.is_dataclass(value) and hasattr(value, 'summary'):
                text = value.summary()
            else:
                text = repr(value)
            out.append((f.name, text))
        out.extend(('note', n.note) for n in self.notices
                   if n.note is not None)
        return out

    def __repr__(self) -> str:
        return _block(type(self).__name__, self.summary_lines())


@dataclass(frozen=True, eq=False, repr=False)
class RunSettings(FrozenRecord):
    """Every setting one ``run()`` call resolved, decided once.

    Returned by :meth:`PropagationModel.run_settings` and carried by every
    result as ``result.run_settings``. ``print(settings)`` shows one line per
    setting; :meth:`summary` is the one-line form a NetCDF attribute holds.

    Attributes
    ----------
    model : str
        The model class name.
    mode : RunMode
        The run mode, resolved once (``run_mode=None`` is the model's
        default, which on Kraken depends on ``frequencies=``).
    frequencies : ndarray or None
        The frequency grid (Hz) the run propagates.
    source_depths : ndarray
        The source depths (m) of the call.
    source_type : str
        ``'point'``, ``'line'`` or ``'scaled'``.
    source_level_dB : float or None
        The source level (dB re 1 µPa at 1 m) the result is referenced to.
    depth_loop : str
        ``'single'`` (one source depth), ``'per_depth'`` (one engine run
        per depth, stacked into a ``ResultStack``) or ``'engine'`` (every
        depth handed to the engine, which stacks them itself).
    source_weights : ndarray or None
        The source's complex weights; ``None`` for unit weights.
    weights_applied : bool
        Whether this mode applies ``source_weights`` (a field mode). A
        one-depth run scales the returned field; a stack records them for
        ``ResultStack.superpose``.
    time : TimeSettings or None
        The TIME_SERIES request; ``None`` in every other mode.
    waveguide : WaveguideSpeeds or None
        Speed bounds of the environment the engine ran on (every run
        records them; ``None`` only on a record built without them).
    output : OutputSpec or None
        What the run returns; ``None`` when the engine declares no output
        table.
    engine : EngineSettings or None
        The engine's own resolved settings; ``None`` for an engine with no
        settings of its own.
    notes : tuple of str
        Every derivation worth reading back, in the order it was made.
    """

    model: str
    mode: Any
    frequencies: Optional[np.ndarray]
    source_depths: np.ndarray
    source_type: str = 'point'
    source_level_dB: Optional[float] = None
    depth_loop: str = 'single'
    source_weights: Optional[np.ndarray] = None
    weights_applied: bool = False
    time: Optional[TimeSettings] = None
    waveguide: Optional[WaveguideSpeeds] = None
    output: Optional[OutputSpec] = None
    engine: Optional[EngineSettings] = None
    notes: Tuple[str, ...] = ()

    _ARRAY_FIELDS = ('frequencies', 'source_depths', 'source_weights')

    def __post_init__(self):
        if self.depth_loop not in DEPTH_LOOPS:
            raise ConfigurationError(f"RunSettings.depth_loop must be one of "
                                     f"{DEPTH_LOOPS}; got {self.depth_loop!r}.")
        object.__setattr__(self, 'notes', tuple(self.notes))
        self._freeze_arrays()

    def _replace(self, **changes) -> 'RunSettings':
        """A copy with ``changes`` applied, for the stages that build a
        record (the record itself is frozen). Not public: a record is
        changed by re-running, through
        :meth:`~uacpy.models.base.PropagationModel.from_run_settings`."""
        return dataclasses.replace(self, **changes)

    # ── text forms ──────────────────────────────────────────────────────

    def _lines(self):
        mode = getattr(self.mode, 'name', self.mode)
        lines = [('model', self.model), ('mode', str(mode)),
                 ('frequencies', array_summary(self.frequencies, 'Hz')),
                 ('source', f"{array_summary(self.source_depths, 'm')}, "
                            f"{self.source_type}"
                            + ('' if self.source_level_dB is None else
                               f", {self.source_level_dB:g} dB")),
                 ('depth loop', self.depth_loop)]
        if self.source_weights is not None:
            lines.append(('weights',
                          f"{[complex(w) for w in self.source_weights]} "
                          f"({'applied' if self.weights_applied else 'not applied'})"))
        if self.time is not None:
            lines.append(('time', self.time.summary()))
        if self.waveguide is not None:
            lines.append(('waveguide', self.waveguide.summary()))
        if self.output is not None:
            lines.append(('output', self.output.summary()))
        if self.engine is not None:
            lines.append(('engine', type(self.engine).__name__))
            lines.extend((f"  {name}", text)
                         for name, text in self.engine.summary_lines())
        for note in self.notes:
            lines.append(('note', note))
        return lines

    def __repr__(self) -> str:
        return _block('RunSettings', self._lines())

    def summary(self) -> str:
        """The settings on one line (what a NetCDF attribute stores)."""
        mode = getattr(self.mode, 'value', self.mode)
        bits = [self.model, str(mode),
                f"f {array_summary(self.frequencies, 'Hz')}",
                f"source {array_summary(self.source_depths, 'm')} "
                f"{self.source_type}",
                f"depth loop {self.depth_loop}"]
        if self.source_weights is not None:
            bits.append('weighted' if self.weights_applied
                        else 'weights not applied')
        if self.time is not None:
            bits.append(self.time.summary())
        if self.waveguide is not None:
            bits.append(self.waveguide.summary())
        if self.output is not None:
            bits.append(self.output.summary())
        if self.engine is not None:
            bits.append(type(self.engine).__name__)
        return '; '.join(bits)

    # ── plain-type round trip ──────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """The settings as plain Python types (lists, floats, strings,
        nested dicts), for logging, provenance and ``Result.to_dict``.
        :meth:`from_dict` rebuilds an equal record."""
        d = super().to_dict()
        d['notes'] = list(self.notes)
        if self.engine is not None:
            d['engine'] = dict(self.engine.to_dict(),
                               __class__=saved_class_path(
                                   type(self.engine)))
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RunSettings':
        """Rebuild a :class:`RunSettings` from :meth:`to_dict` output."""
        d = dict(d)
        weights = d.get('source_weights')
        if weights is not None:
            d['source_weights'] = np.array(
                [complex(re, im) for re, im in weights])
        d['mode'] = RunMode(d['mode'])
        for key, klass in (('time', TimeSettings),
                           ('waveguide', WaveguideSpeeds),
                           ('output', OutputSpec)):
            if d.get(key) is not None:
                d[key] = klass.from_dict(d[key])
        engine = d.get('engine')
        if engine is not None:
            engine = dict(engine)
            module, _, name = engine.pop('__class__').rpartition('.')
            d['engine'] = getattr(importlib.import_module(module),
                                  name).from_dict(engine)
        d['notes'] = tuple(d.get('notes', ()))
        return cls(**d)
