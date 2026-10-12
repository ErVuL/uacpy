"""Which engine binaries this install holds: :func:`engine_status` and the
:class:`EngineStatus` table it returns."""

from dataclasses import dataclass
from typing import Tuple

from uacpy.core._export import ExportRecord
from uacpy.core.exceptions import ExecutableNotFoundError
from uacpy.models._registry import ENGINES

__all__ = ['EngineStatus', 'engine_status']


@dataclass(frozen=True, eq=False, repr=False)
class EngineStatus(ExportRecord):
    """Which engine binaries this install holds, one row per binary a
    registered engine can launch, as :func:`engine_status` found them.
    ``print(status)`` shows the table; :meth:`to_dataframe` gives one row
    per binary.

    Attributes
    ----------
    engine : tuple of str
        The engine's class name (``'Bellhop'``).
    backend : tuple of str
        The ``backend=`` value that runs the binary (``'cuda'``,
        ``'krakenc'``, ``'ramgeo'``); the engine's name in lower case for a
        one-binary engine, as ``result.backend`` reads; ``'field'`` for
        Kraken's ``field.exe``, which every Kraken backend launches to sum
        its modes into a field.
    binary : tuple of str
        The binary's file name. An OASES binary runs through its
        ``<name>_bash`` wrapper when one is installed.
    install : tuple of str
        ``'oases'`` for the separately licensed OASES engines
        (``./install.sh --oases yes``), ``'base'`` for every other engine.
    found : tuple of bool
        Whether a file this process can execute was found where the engine
        looks for it. The binary is not run.
    path : tuple of str
        Where it was found; ``''`` when it was not.
    default : tuple of bool
        The backend a construction with no ``backend=`` resolves: Bellhop's
        first installed of ``cuda``, ``cxx``, ``fortran``; Kraken's
        ``kraken`` and RAM's ``mpirams``, from which a run switches by
        environment (``select_backend``); the one binary of every other
        engine.
    """

    engine: Tuple[str, ...]
    backend: Tuple[str, ...]
    binary: Tuple[str, ...]
    install: Tuple[str, ...]
    found: Tuple[bool, ...]
    path: Tuple[str, ...]
    default: Tuple[bool, ...]

    _TABLE_FIELDS = ('engine', 'backend', 'binary', 'install', 'found',
                     'path', 'default')

    def missing_engines(self, *, optional_installs: bool = False
                        ) -> Tuple[str, ...]:
        """The engines none of whose binaries was found, in table order.

        An optional install (``install`` other than ``'base'``) none of
        whose binaries was found is an install not made, and its engines
        are left out unless ``optional_installs=True``; once one of its
        binaries is found, each of its engines counts.

        Parameters
        ----------
        optional_installs : bool, default False
            Also name the engines of an optional install not made.
        """
        made = {'base'} | {install for install, found
                           in zip(self.install, self.found)
                           if found or optional_installs}
        has_binary = {}
        for engine, install, found in zip(self.engine, self.install,
                                          self.found):
            if install in made:
                has_binary[engine] = has_binary.get(engine, False) or found
        return tuple(engine for engine, ok in has_binary.items() if not ok)

    def __repr__(self) -> str:
        rows = [('engine', 'backend', 'default', 'found', 'path')]
        for engine, backend, binary, found, path, default in zip(
                self.engine, self.backend, self.binary, self.found,
                self.path, self.default):
            rows.append((engine, backend, 'yes' if default else '',
                         'yes' if found else 'no',
                         path if found else f"not found: {binary}"))
        widths = [max(len(row[i]) for row in rows) for i in range(4)]
        body = '\n'.join(
            '  ' + '  '.join(cell.ljust(width)
                             for cell, width in zip(row, widths))
            + '  ' + row[4] for row in rows)
        return f"{type(self).__name__}(\n{body}\n)"


def engine_status() -> EngineStatus:
    """Which binaries of the registered engines this install holds.

    One row per binary each engine can launch — every backend, and
    Kraken's ``field.exe`` — looked for exactly where the engine looks when
    it is constructed or run (its declared search: ``uacpy/bin``, then the
    development tree ``uacpy/third_party``, then ``PATH``): whether it was
    found and where, and which backend a construction with no
    ``backend=`` resolves. ``python -m uacpy`` prints this table.

    Found means on disk with its execute bit set; no binary is run, so a
    found build can still fail on this machine (bellhopcuda on a host with
    no CUDA device, which a ``backend=None`` run then replaces with the
    next installed build).

    Returns
    -------
    EngineStatus
        The table; :meth:`EngineStatus.missing_engines` names the engines
        with no binary found.

    Examples
    --------
    >>> from uacpy.models import engine_status
    >>> status = engine_status()
    >>> print(status)                      # doctest: +SKIP
    >>> status.missing_engines()           # doctest: +SKIP
    ()
    """
    columns = {name: [] for name in EngineStatus._TABLE_FIELDS}
    for entry in ENGINES.values():
        cls = entry.load()
        default = cls._default_backend()
        install = next((name for name in entry.requires if name != 'binary'),
                       'base')
        for backend, search in cls._BINARIES.items():
            try:
                path = str(search.find(cls.__name__))
            except ExecutableNotFoundError:
                path = ''
            for name, value in (('engine', cls.__name__),
                                ('backend', backend),
                                ('binary', search.names[-1]),
                                ('install', install), ('found', bool(path)),
                                ('path', path),
                                ('default', backend == default)):
                columns[name].append(value)
    return EngineStatus(**{name: tuple(values)
                           for name, values in columns.items()})
