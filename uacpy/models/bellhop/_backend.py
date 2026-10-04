"""Which Bellhop binary runs, and how: the engine a binary's name says it
is, the search for one, its argv and ARRIVALS memory budget, whether its
``.arr`` still needs the ``AddArr`` pair-merge, the process-wide finding that
bellhopcuda has no CUDA device, and the diagnoses the C++ / CUDA ports print
to stdout."""

import os
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.exceptions import (
    ExecutableNotFoundError, FallbackWarning, IOWarning,
)
from uacpy.models._launch import _launch_single_threaded
from uacpy.models.bellhop._tables import grid_is_paired

# One ``BHC_WARN_<NAME>: ...`` / ``BHC_ERR_<NAME>: ...`` line of the exit
# report bellhopcxx / bellhopcuda print to stdout (``util/errors.cpp``).
_BHC_DIAGNOSTIC_LINE = re.compile(
    r'^BHC_(WARN|ERR)_[A-Z0-9_]+:'
    # Capacity notices the ports print bare through EXTWARN
    # (src/mode/arr.hpp:96, src/mode/eigen.cpp:73).
    r'|^Only enough memory to allocate up to \d+ arrivals'
    r'|^Would have had \d+ eigenrays')

#: What bellhopcuda prints when the host has no usable CUDA device —
#: ``cudaGetDeviceCount`` failing with ``cudaErrorNoDevice`` (code 100) in
#: ``bhc::setup`` (``bellhopcuda/src/api.cpp:148``), after which the CLI
#: exits 1 (``src/cmdline.cpp:27``). Measured with
#: ``CUDA_VISIBLE_DEVICES=`` on a GPU host.
_CUDA_NO_DEVICE = re.compile(r'cudaErrorNoDevice|code=100\b')

#: Set once per process when an auto-selected bellhopcuda has found no CUDA
#: device; auto selection then starts at bellhopcxx.
_cuda_found_no_device = False

_BELLHOP_OUTPUT_SUFFIXES = ('.shd', '.arr', '.ray')

#: bellhopcxx / bellhopcuda size an ARRIVALS table from the memory budget,
#: not from the grid: ``MaxNArr = remaining budget / (sources x receivers x
#: sizeof(Arrival))``, then the whole table is allocated and zero-filled
#: (``bellhopcuda/src/mode/arr.hpp:83-105``). With no ``-mem=`` the budget is
#: 4 GiB (``include/bhc/structs.hpp:494``), so every ARRIVALS run touched
#: ~4 GiB (measured peak RSS: 4.17 GiB bellhopcxx, 4.27 GiB bellhopcuda, on a
#: 10 x 20 grid that Fortran ran in 18 MB). uacpy passes a budget sized from
#: the grid instead.
_BHC_ARRIVAL_BYTES = 40            # structs.hpp:376-380: 2 int32, 6 float, cpxf
_BHC_ARRIVALS_PER_RECEIVER = 10000
_FORTRAN_ARRIVALS_STORAGE = 20_000_000   # bellhop.f90:158 ArrivalsStorage
_BHC_ARRIVALS_OVERHEAD_BYTES = 16 * 1024 ** 2
_BHC_ARRIVALS_MEMORY_FLOOR = 64 * 1024 ** 2
_BHC_ARRIVALS_MEMORY_CAP = 1024 ** 3


def _ports_arrivals_memory_bytes(n_sources: int, n_receivers: int) -> int:
    """``-mem=`` budget for a bellhopcxx / bellhopcuda ARRIVALS run.

    Room for the per-receiver capacity the Fortran engine gives the same
    grid, ``max(ArrivalsStorage / receivers, 10)`` (``bellhop.f90:222``),
    held to at most ``_BHC_ARRIVALS_PER_RECEIVER`` on small grids, plus the
    ports' other allocations; clamped to [64 MiB, 1 GiB].
    """
    n_sources = max(int(n_sources), 1)
    n_receivers = max(int(n_receivers), 1)
    per_receiver = min(max(_FORTRAN_ARRIVALS_STORAGE // n_receivers, 10),
                       _BHC_ARRIVALS_PER_RECEIVER)
    table = n_sources * n_receivers * (per_receiver * _BHC_ARRIVAL_BYTES + 4)
    budget = table + _BHC_ARRIVALS_OVERHEAD_BYTES
    return int(min(max(budget, _BHC_ARRIVALS_MEMORY_FLOOR),
                   _BHC_ARRIVALS_MEMORY_CAP))


def _bellhop_variant(exe) -> Optional[str]:
    """Which engine a Bellhop binary is, read off its basename: ``'cuda'``
    (bellhopcuda), ``'cxx'`` (bellhopcxx) or ``'fortran'`` (bellhop, the
    name every AT build and ``install.sh`` produce). ``None`` for a name
    carrying none of the three, which only a user-pinned ``executable=``
    can be."""
    lower = Path(exe).name.lower()
    if 'cuda' in lower:
        return 'cuda'
    if 'cxx' in lower:
        return 'cxx'
    if 'bellhop' in lower:
        return 'fortran'
    return None


def arrivals_memory_bytes(source, deck_receiver, *, grid_type) -> int:
    """The ports' ``-mem=`` budget for an ARRIVALS deck carrying
    ``deck_receiver``: :func:`_ports_arrivals_memory_bytes` of its source
    depths and its receivers, one per range on a paired grid."""
    n_rz = len(np.atleast_1d(deck_receiver.depths))
    n_rr = len(np.atleast_1d(deck_receiver.ranges))
    return _ports_arrivals_memory_bytes(
        len(np.atleast_1d(source.depths)),
        n_rr if grid_is_paired(grid_type) else n_rz * n_rr)


def find_bellhop_executable(backend, *, find) -> Path:
    """Locate the Bellhop binary, keyed on ``backend``.

    ``None`` auto-selects in preference order CUDA > C++ > Fortran among
    whatever ``install.sh`` built. An explicitly requested variant is
    the only candidate: when it is not installed the search raises
    :class:`ExecutableNotFoundError` naming the ``install.sh`` flag that
    builds it, rather than running another engine. ``find`` is the
    model's path search
    (:meth:`~uacpy.models.base.PropagationModel._find_executable_in_paths`).
    """
    backend_names = {
        'fortran': ['bellhop'],
        'cxx': ['bellhopcxx'],
        'cuda': ['bellhopcuda'],
    }
    auto_names = (['bellhopcxx', 'bellhop'] if _cuda_found_no_device
                  else ['bellhopcuda', 'bellhopcxx', 'bellhop'])
    names = backend_names.get(backend, auto_names)
    try:
        path = find(
            names,
            bin_subdirs=['bellhopcuda', 'oalib', 'bellhop'],
            dev_subdir='Acoustics-Toolbox/Bellhop',
        )
    except ExecutableNotFoundError as exc:
        if backend in ('cxx', 'cuda'):
            exc.remediation = (
                f"Build it with ./install.sh --bellhop {backend}, or "
                f"pass backend=None to use the best Bellhop binary that "
                f"is installed.\n\n{exc.remediation}")
        raise
    return path


def has_no_device_signature(exc) -> bool:
    """Whether the failed launch ``exc`` printed bellhopcuda's no-device
    signature (:data:`_CUDA_NO_DEVICE`)."""
    text = f"{getattr(exc, 'stdout', '') or ''}\n{exc}"
    return bool(_CUDA_NO_DEVICE.search(text))


def record_cuda_without_a_device() -> bool:
    """Keep for the process the finding that bellhopcuda has no CUDA
    device, so auto selection starts at bellhopcxx from then on, and say
    whether this is the first time it was found."""
    global _cuda_found_no_device
    first = not _cuda_found_no_device
    _cuda_found_no_device = True
    return first


def build_command(exe, base_name: str, *, backend, dimensionality,
                  memory_bytes: Optional[int] = None) -> list:
    """Build the argv used to launch the binary.

    The bellhopcxx / bellhopcuda CLIs accept a ``--<dim>`` flag (they
    assume 2D without one, ``cmdline.cpp:186-191``); the Fortran binary
    takes none. ``memory_bytes`` becomes the ports' ``-mem=`` budget
    (``cmdline.cpp:126-152``), written in KiB because the parser reads
    the number with ``std::stoi``. Inside a ``run_parallel`` worker
    bellhopcxx is launched with ``-1`` (one thread, ``cmdline.cpp:96-97``):
    the pool supplies the parallelism.
    """
    if backend in ('cuda', 'cxx'):
        cmd = [str(exe), f'--{dimensionality}']
        if backend == 'cxx' and _launch_single_threaded():
            cmd.append('-1')
        if memory_bytes is not None:
            cmd.append(f'-mem={int(memory_bytes) // 1024}KiB')
        return cmd + [base_name]
    return [str(exe), base_name]


def arrivals_need_merge(*, backend, exe, model_name) -> bool:
    """Whether the ``.arr`` this run wrote still needs Bellhop's
    ``AddArr`` pair-merge applied on reading.

    The Fortran engine merges as it accumulates, so its file is read as
    written; bellhopcxx merges only when it runs single-threaded
    (``bellhopcuda/src/mode/arr.hpp:79``; the CLI takes no thread count,
    only ``-1`` / ``--singlethread``, and defaults to every core), so its
    file is unmerged on any multi-core host unless it was launched with
    ``-1`` (inside a ``run_parallel`` worker, see :func:`build_command`);
    bellhopcuda's GPU run is always unmerged. Re-merging a merged file is not a no-op (see
    :func:`uacpy.io.oalib_reader.read_arr_file`), so the two cases must
    not be confused. A ``'custom'`` binary (a pinned ``executable=``
    whose name says nothing) is read as written, with a warning that
    this may be the unmerged multi-thread file.
    """
    if backend == 'cuda':
        return True
    if backend == 'cxx':
        return (os.cpu_count() or 1) > 1 and not _launch_single_threaded()
    if backend == 'custom':
        warnings.warn(
            f"{model_name}: executable={str(exe)!r} names "
            f"none of bellhop / bellhopcxx / bellhopcuda, so whether its "
            f".arr needs the AddArr pair-merge is unknown; it is read as "
            f"written. Rename the binary after its engine, or pass "
            f"backend= and let the package locate it.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
    return False


def warn_on_engine_stdout_warnings(stdout: str, *, model_name,
                                   backend) -> None:
    """Re-emit the run-time diagnoses bellhopcxx / bellhopcuda print
    to stdout as one ``IOWarning``.

    The ports keep a bitmask of run-time conditions and report it at
    exit through ``printf`` (``bellhopcuda/src/util/errors.cpp:26-36``,
    ``113-137``): a header ``N warning(s) thrown of the following
    type(s):`` followed by one ``BHC_WARN_<NAME>: <description>`` line
    per set bit, and the same shape with ``error(s)`` / ``BHC_ERR_``
    for non-fatal error bits. Nothing of it reaches the ``.prt``, so
    the Fortran-only ``Warning in ...`` scan sees a silent run — measured,
    ``n_beams=3`` warns ``Too few beams`` on ``backend='fortran'`` and
    nothing on ``backend='cxx'`` for the same field.

    Captured stdout of that ``cxx`` run, verbatim::

        setup: 0.624820 ms
        Preprocess: 0.140358 ms
        1 warning(s) thrown of the following type(s):
        BHC_WARN_TOO_FEW_BEAMS: Nalpha is too small; there may be gaps between the beams
        Run: 0.379620 ms

    The engine's lines are passed through verbatim, as the ``.prt``
    path does.
    """
    lines = [line.strip() for line in stdout.splitlines()
             if _BHC_DIAGNOSTIC_LINE.match(line.strip())]
    if not lines:
        return
    joined = "\n  ".join(dict.fromkeys(lines))
    warnings.warn(
        f"{model_name} ({backend}) reported {len(lines)} "
        f"non-fatal warning(s):\n  {joined}",
        IOWarning, skip_file_prefixes=USER_FRAME_SKIP)
