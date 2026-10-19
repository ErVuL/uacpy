"""What the host machine can give a run.

One reader of available memory for every solver that sizes an allocation
against it (the ``.grn`` transform, Scooter, SPARC, RAM), so the fallback
chain and its guard live in one place.
"""

import os
from pathlib import Path
from typing import Optional

#: A cgroup v1 limit at or above this is the kernel's "unlimited" (it is
#: stored as the largest page-aligned signed 64-bit value).
_CGROUP_V1_UNLIMITED = 1 << 60


def _read_int(path: Path) -> Optional[int]:
    """The integer in a cgroup file, or ``None`` for ``max`` / unreadable."""
    try:
        text = path.read_text(encoding='ascii').strip()
    except OSError:
        return None
    if text == 'max':
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _read_stat(path: Path, key: str) -> int:
    """The value of ``key`` in a cgroup ``memory.stat`` file, or 0 when the
    file or the key is absent or unreadable."""
    try:
        lines = path.read_text(encoding='ascii').splitlines()
    except OSError:
        return 0
    for line in lines:
        parts = line.split()
        if len(parts) == 2 and parts[0] == key:
            try:
                return int(parts[1])
            except ValueError:
                return 0
    return 0


def _cgroup_memory_headroom(cgroup_file: str = '/proc/self/cgroup',
                            cgroup_root: str = '/sys/fs/cgroup',
                            ) -> Optional[int]:
    """Bytes this process's memory cgroup still admits, or ``None`` when no
    cgroup limits it (or the host has no cgroup files).

    A container (``docker --memory``), a Kubernetes pod or a SLURM job caps
    memory through its cgroup, while ``/proc/meminfo`` keeps reporting the
    HOST, so an allocation sized against ``MemAvailable`` alone is admitted
    and then OOM-killed. cgroup v2 (the ``0::<path>`` line): the tightest
    ``memory.max - (memory.current - inactive_file)`` over the process's
    cgroup and every ancestor, since a parent's limit binds its children.
    cgroup v1 (the ``memory`` controller line): ``memory.limit_in_bytes -
    (memory.usage_in_bytes - total_inactive_file)``, a limit of 2**60 or
    more meaning none.

    The usage a cgroup is charged includes its page cache ("Userland
    memory - page cache and anonymous memory", external: Linux kernel
    documentation, admin-guide/cgroup-v2, "Memory"), which the kernel
    reclaims for a new allocation as it does on the host, where
    ``MemAvailable`` counts "the size of the file LRU lists" (external:
    filesystems/proc, ``/proc/meminfo``). The inactive file list
    (``memory.stat``) is taken off the usage for that reason, as
    :func:`available_memory_bytes` reads ``MemAvailable`` rather than
    ``MemFree``; the active file list and reclaimable slab stay counted, so
    the figure errs low. A cgroup with no ``memory.stat`` counts none.
    """
    try:
        lines = Path(cgroup_file).read_text(encoding='ascii').splitlines()
    except OSError:
        return None
    root = Path(cgroup_root)
    headroom = None
    for line in lines:
        parts = line.split(':', 2)
        if len(parts) != 3:
            continue
        _, controllers, rel = parts
        if controllers == '':                      # cgroup v2
            node = root / rel.lstrip('/')
            while True:
                limit = _read_int(node / 'memory.max')
                used = _read_int(node / 'memory.current')
                if limit is not None and used is not None:
                    used -= _read_stat(node / 'memory.stat', 'inactive_file')
                    free = max(limit - max(used, 0), 0)
                    headroom = free if headroom is None else min(headroom, free)
                if node == root or root not in node.parents:
                    break
                node = node.parent
        elif 'memory' in controllers.split(','):   # cgroup v1
            node = root / 'memory' / rel.lstrip('/')
            limit = _read_int(node / 'memory.limit_in_bytes')
            used = _read_int(node / 'memory.usage_in_bytes')
            if (limit is not None and used is not None
                    and limit < _CGROUP_V1_UNLIMITED):
                used -= _read_stat(node / 'memory.stat',
                                   'total_inactive_file')
                free = max(limit - max(used, 0), 0)
                headroom = free if headroom is None else min(headroom, free)
    return headroom


def available_memory_bytes() -> Optional[int]:
    """Memory the host says it can still hand out, in bytes, or ``None`` if
    it cannot say.

    Linux's ``/proc/meminfo`` ``MemAvailable`` first: it counts the page
    cache the kernel would reclaim for a new allocation, which
    ``SC_AVPHYS_PAGES`` (``MemFree``) leaves out, so the sysconf pair
    under-reports by whatever the cache holds — most of what reading a
    large ``.grn`` has just consumed, and 10 GB of a 31 GB box measured.
    The sysconf pair is the fallback for hosts without the file; it
    reports ``-1`` for a value the host does not know, which is ``None``
    here rather than a negative size. Callers size their allocations
    against this rather than a fixed constant — the same cube is nothing
    on a workstation and fatal on a laptop.

    Inside a memory cgroup (a container, a pod, a batch job) the answer is
    the smaller of the host's figure and what the cgroup still admits
    (:func:`_cgroup_memory_headroom`).
    """
    host = _host_available_memory_bytes()
    cgroup = _cgroup_memory_headroom()
    if cgroup is None:
        return host
    return cgroup if host is None else min(host, cgroup)


def _host_available_memory_bytes() -> Optional[int]:
    """The host's own figure: ``MemAvailable``, else the sysconf pair."""
    try:
        with open('/proc/meminfo', encoding='ascii') as fh:
            for line in fh:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        pages = os.sysconf('SC_AVPHYS_PAGES')
        page = os.sysconf('SC_PAGE_SIZE')
    except (AttributeError, OSError, ValueError):
        return None
    if pages < 0 or page < 0:
        return None
    return int(pages) * int(page)
