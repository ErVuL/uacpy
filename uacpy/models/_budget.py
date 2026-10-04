"""One memory policy for what a run is about to hold in memory (M-14).

An engine supplies only its byte estimate and the words that describe it;
:func:`memory_budget` weighs the estimate against what the host reports free
(:func:`~uacpy.core._host.available_memory_bytes`) and decides:

- over all of it: :class:`~uacpy.core.exceptions.ConfigurationError` before
  any file is written, since waiting cannot make it fit (an estimate that is
  only an upper bound is announced instead);
- over half of it: a :class:`~uacpy.models._notices.Notice`, since the run
  should complete but leaves little headroom;
- with the host's free memory unreadable: a notice above
  :data:`UNREADABLE_HOST_BYTES`, never a refusal, since nothing measured says
  the run cannot fit.

An output file counts as memory only when the run's work directory is on a
memory-backed filesystem (:func:`work_dir_is_memory_backed`); on disk it is a
large-file matter the engine states on its own.
"""
import platform
import re
import tempfile
from pathlib import Path
from typing import Optional

from uacpy.core._host import available_memory_bytes
from uacpy.core.exceptions import ConfigurationError, NumericsWarning
from uacpy.core.run_settings import Notice
from uacpy.models._workspace import FileManager, ScratchPolicy

#: Fraction of the free memory above which an estimate is announced.
HEADROOM_FRACTION = 0.5

#: Estimate above which a run on a host whose free memory cannot be read is
#: announced (2 GiB).
UNREADABLE_HOST_BYTES = 2 * 1024 ** 3

#: Filesystem types whose files live in memory.
_MEMORY_FILESYSTEMS = frozenset({'tmpfs', 'ramfs'})

_GIB = 1024 ** 3

_OCTAL_ESCAPE = re.compile(r'\\([0-7]{3})')


def memory_budget(n_bytes: int, *, model_name: str, what: str, detail: str,
                  remediation: str, upper_bound: bool = False
                  ) -> Optional[Notice]:
    """Weigh ``n_bytes`` against the host's free memory.

    Parameters
    ----------
    n_bytes : int
        The engine's estimate of what the run holds at its peak.
    model_name : str
        The engine the messages name.
    what : str
        A short name of the allocation (``"Green's-function cube"``), for
        the settings note.
    detail : str
        One or more sentences stating the estimate and how it is made up.
    remediation : str
        The knobs that shrink it: the refusal's remediation, and the
        notices' last sentence.
    upper_bound : bool, optional
        Whether ``n_bytes`` is a bound the run may stay well under rather
        than an estimate of what it holds; an upper bound over the free
        memory is announced, not refused.

    Returns
    -------
    Notice or None
        The notice of an estimate over :data:`HEADROOM_FRACTION` of the free
        memory, or over :data:`UNREADABLE_HOST_BYTES` when the free memory
        cannot be read; ``None`` otherwise.

    Raises
    ------
    ConfigurationError
        When the estimate is more than the host reports free (never for an
        ``upper_bound``).
    """
    gib = n_bytes / _GIB
    available = available_memory_bytes()
    if available is None:
        if n_bytes <= UNREADABLE_HOST_BYTES:
            return None
        return Notice(
            f"{what} {gib:.1f} GiB, free memory unreadable",
            f"{model_name}: {detail} The free memory of this "
            f"{platform.system() or 'unknown'} host cannot be read "
            f"(uacpy reads /proc/meminfo MemAvailable, else sysconf "
            f"SC_AVPHYS_PAGES), and that is over the "
            f"{UNREADABLE_HOST_BYTES / _GIB:.0f} GiB announced in its "
            f"place; the run is not refused, but may run out of memory. "
            f"{remediation}", NumericsWarning)
    free = available / _GIB
    if n_bytes > available and upper_bound:
        return Notice(
            f"{what} up to {gib:.1f} GiB, over the {free:.1f} GiB free",
            f"{model_name}: {detail} That bound is more than the "
            f"{free:.1f} GiB this host reports free; the run is not refused, "
            f"since it may stay well under it, but may run out of memory. "
            f"{remediation}", NumericsWarning)
    if n_bytes > available:
        raise ConfigurationError(
            f"{model_name}: {detail} That is more than the {free:.1f} GiB "
            f"this host reports free.",
            remediation=remediation)
    if n_bytes > HEADROOM_FRACTION * available:
        return Notice(
            f"{what} {gib:.1f} GiB, over half the {free:.1f} GiB free",
            f"{model_name}: {detail} That is over half the {free:.1f} GiB "
            f"this host reports free; the run should complete but leaves "
            f"little headroom. {remediation}", NumericsWarning)
    return None


def work_dir_is_memory_backed(policy: ScratchPolicy) -> bool:
    """Whether the directory a run with this scratch ``policy`` writes into
    sits on a memory-backed filesystem (tmpfs, ramfs): the pinned
    ``work_dir``, else :attr:`FileManager.DEV_SHM_ROOT` for ``use_tmpfs=True``
    where it is available, else the system temporary directory — which is
    itself a tmpfs on many Linux systems. ``False`` where the mount table
    cannot be read."""
    if policy.pinned_dir is not None:
        base = Path(policy.pinned_dir)
    elif policy.on_tmpfs and FileManager.dev_shm_usable():
        base = FileManager.DEV_SHM_ROOT
    else:
        base = Path(tempfile.gettempdir())
    return filesystem_type(base) in _MEMORY_FILESYSTEMS


def filesystem_type(path, *, mount_table='/proc/self/mounts'
                    ) -> Optional[str]:
    """The type of the filesystem ``path`` is on, from the longest mount
    point of ``mount_table`` (Linux's ``/proc/self/mounts``) containing it,
    or ``None`` where the table cannot be read."""
    try:
        with open(mount_table, encoding='utf-8') as fh:
            rows = [line.split() for line in fh]
    except OSError:
        return None
    target = Path(path).resolve()
    best, kind = -1, None
    for row in rows:
        if len(row) < 3:
            continue
        # The table writes a space, tab, newline or backslash in a mount
        # point as a three-digit octal escape; of two mounts on one point,
        # the later one is the one in effect.
        mount = Path(_OCTAL_ESCAPE.sub(
            lambda m: chr(int(m.group(1), 8)), row[1]))
        if (target == mount or mount in target.parents) and \
                len(mount.parts) >= best:
            best, kind = len(mount.parts), row[2]
    return kind
