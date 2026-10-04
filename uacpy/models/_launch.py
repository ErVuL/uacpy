"""Launching an engine binary: where the package keeps its builds, finding
one, the :class:`Launch` record of one launch and its runner
(:func:`run_launch`, through :func:`run_subprocess`), reading what a binary
reports on failure — the fatal and warning lines of its ``.prt``, the tail
of a captured stream — and the one thread rule: whether this process
launches its engines single-threaded (:func:`_launch_single_threaded`) and
the thread count an OpenMP binary gets (:func:`openmp_thread_env`)."""

import errno
import os
import re
import shutil
import signal
import subprocess
import warnings
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Callable, Optional

from uacpy._log import log_message
from uacpy._stack import parent_death_prefix, stack_limit_prefix
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.exceptions import (
    ExecutableNotFoundError, IOWarning, ModelExecutionError,
)
from uacpy.io.oalib_reader import read_prt


#: The installed ``uacpy`` package directory, which holds ``bin/`` and
#: ``third_party/``.
_PACKAGE_DIR = Path(str(files('uacpy')))

#: How much of a captured child stream to quote back in an error. The
#: binaries echo their whole deck and every progress block to stdout, so only
#: the tail is useful; the abort reason is always last.
_STREAM_TAIL = 2000


def _tail(text: Optional[str], n_chars: int = _STREAM_TAIL) -> Optional[str]:
    """Last ``n_chars`` of a captured child stream, or ``None`` if empty."""
    if not text:
        return None
    return text if len(text) <= n_chars else '…' + text[-n_chars:]


def _stream_block(label: str, text: Optional[str]) -> str:
    """Render a captured child stream as a labelled block, or nothing."""
    trimmed = _tail(text)
    return f"\n\n{label}:\n{trimmed}" if trimmed else ''


def _is_runnable(path: Path) -> bool:
    """Whether ``path`` is a file this process can actually exec.

    ``exists()`` alone also accepts a directory, the text file an unexpanded
    LFS pointer leaves behind, and a build whose exec bit was lost in transit
    — the realistic broken installs. Each of those reaches ``Popen`` and comes
    back as a raw ``PermissionError`` or ``OSError [Errno 8]`` instead of a
    typed install error, so the executability test belongs at resolve time.
    """
    return path.is_file() and os.access(path, os.X_OK)


# Leading bytes of the file formats the kernel execs directly: ELF, the
# ``#!`` interpreter line, and the Mach-O / fat-binary magics.
_EXEC_MAGICS = (b'\x7fELF', b'#!', b'\xfe\xed\xfa\xce', b'\xfe\xed\xfa\xcf',
                b'\xce\xfa\xed\xfe', b'\xcf\xfa\xed\xfe', b'\xca\xfe\xba\xbe')


def _check_executable(program: str, env: Optional[dict] = None) -> None:
    """Raise the ``OSError`` a direct ``exec`` of ``program`` would raise.

    ``FileNotFoundError`` for a missing file, ``PermissionError`` for a
    directory or a file without the execute bit, and ``OSError(ENOEXEC)``
    for a file in no format the kernel runs (an unexpanded LFS pointer).
    A name without a separator is looked up on ``PATH`` as ``exec`` does.
    """
    path = program
    if os.sep not in program:
        search = (env if env is not None else os.environ).get('PATH')
        found = shutil.which(program, path=search)
        if found is None:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    program)
        path = found
    if not os.path.exists(path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                program)
    if os.path.isdir(path) or not os.access(path, os.X_OK):
        raise PermissionError(errno.EACCES, os.strerror(errno.EACCES),
                              program)
    with open(path, 'rb') as fh:
        head = fh.read(4)
    if not head.startswith(_EXEC_MAGICS):
        raise OSError(errno.ENOEXEC, os.strerror(errno.ENOEXEC), program)


#: Whether engines launch their binaries single-threaded in this process.
#: ``run_parallel`` sets it in each pool worker: the pool supplies the
#: parallelism, and an engine that also starts one thread per core
#: (bellhopcxx by default, ``timing.hpp:31-35``) would run N x N threads on
#: N cores. Outside a pool the engines use the cores.
_single_threaded_launches = False


def _launch_single_threaded() -> bool:
    """Whether this process's engines launch single-threaded (a
    ``run_parallel`` worker)."""
    return _single_threaded_launches


def set_launch_threads(single: bool) -> None:
    """Make this process's engines launch single-threaded (``single=True``)
    or use the cores (``False``, the default). ``run_parallel``'s pool
    initializer sets it in every worker; :func:`_launch_single_threaded`
    reads it."""
    global _single_threaded_launches
    _single_threaded_launches = bool(single)


def openmp_thread_env(env: Optional[dict] = None):
    """``(env, threads)``: the environment an OpenMP binary (RAM's mpiramS)
    is launched with, a copy of ``env`` (this process's by default), and a
    line saying how its thread count was decided.

    One rule with Bellhop's ``-1`` (:func:`_launch_single_threaded`): an
    exported ``OMP_NUM_THREADS`` is passed through as given; inside a
    ``run_parallel`` worker it is set to 1, the pool supplying the
    parallelism; otherwise it is left unset, and libgomp starts one thread
    per CPU."""
    env = dict(os.environ if env is None else env)
    if 'OMP_NUM_THREADS' in env:
        return env, f"OMP_NUM_THREADS={env['OMP_NUM_THREADS']} (inherited)"
    if _launch_single_threaded():
        env['OMP_NUM_THREADS'] = '1'
        return env, 'OMP_NUM_THREADS=1 (a run_parallel worker)'
    return env, 'OMP_NUM_THREADS unset (one thread per CPU)'


def find_executable_in_paths(
    names,
    bin_subdirs=None,
    dev_subdir: Optional[str] = None,
    *,
    model_name: str,
) -> Path:
    """
    Find a model executable by searching standard locations.

    Search order (each name is also tried with a ``.exe`` suffix):
        1. uacpy/bin/<bin_subdir>/<name>[+.exe] for each combination
        2. uacpy/third_party/<dev_subdir>/bin (development location)
        3. System PATH

    Parameters
    ----------
    names : str or list of str
        Executable name(s) to try, in preference order.
    bin_subdirs : str or list of str, optional
        Subdirectory/ies under uacpy/bin/. Default 'oalib'.
    dev_subdir : str, optional
        Subdirectory under uacpy/third_party/ (e.g. 'Acoustics-Toolbox/Kraken',
        'oases'). If given, also checks <dev_subdir>/bin and <dev_subdir>/.

    Raises
    ------
    ExecutableNotFoundError
    """
    if isinstance(names, str):
        names = [names]
    if bin_subdirs is None:
        bin_subdirs = ['oalib']
    elif isinstance(bin_subdirs, str):
        bin_subdirs = [bin_subdirs]

    base_dir = _PACKAGE_DIR
    candidates = []
    for name in names:
        variants = [name]
        if not name.endswith('.exe'):
            variants.append(name + '.exe')
        for v in variants:
            for sd in bin_subdirs:
                candidates.append(base_dir / 'bin' / sd / v)
            if dev_subdir:
                candidates.append(base_dir / 'third_party' / dev_subdir / 'bin' / v)
                candidates.append(base_dir / 'third_party' / dev_subdir / v)

    # A dud candidate is skipped rather than selected: an earlier search
    # location holding an unexpanded LFS pointer or a half-extracted file
    # must not shadow the working build further down the list.
    for path in candidates:
        if _is_runnable(path):
            return path

    for name in names:
        variants = [name]
        if not name.endswith('.exe'):
            variants.append(name + '.exe')
        for v in variants:
            found = shutil.which(v)
            if found:
                return Path(found)

    raise ExecutableNotFoundError(
        model_name,
        names[0],
        search_paths=[str(p) for p in candidates],
    )


def run_subprocess(
    model_name: str,
    cmd,
    cwd,
    *,
    timeout: float,
    verbose,
    env: Optional[dict] = None,
):
    """
    Run an external binary and raise ModelExecutionError on failure.

    All Fortran acoustic binaries are spawned through this helper so that
    failures surface as ``ModelExecutionError`` with stdout/stderr
    attached, and so every child runs with its soft ``RLIMIT_STACK``
    raised to the hard limit (:func:`uacpy._stack.stack_limit_prefix`;
    the calling process keeps its own limit). Several Acoustics-Toolbox
    binaries (notably SPARC, whose ``MARCH`` declares twelve automatic
    ``COMPLEX(NTot1)`` arrays at ``Scooter/sparc.f90:354-355``) put large
    working arrays on the stack; the default 8 MB Linux soft stack
    segfaults them on first use.

    Parameters
    ----------
    model_name : str
        The model the errors name.
    cmd : list
        Command argv (str-able elements).
    cwd : path-like
        Working directory for the subprocess.
    timeout : float
        Max seconds this one launch may take before it is killed and
        raises. A run that launches several binaries (Kraken's modes
        and field steps, RAM's per-frequency Collins runs, the OASES
        two-stage runs) gives each launch this budget.
    verbose : bool or str
        The model's status-output gate (the command line is logged at
        ``debug``).
    env : dict, optional
        Environment variables for the subprocess.

    Returns
    -------
    subprocess.CompletedProcess
    """
    cmd_str = ' '.join(str(c) for c in cmd)
    log_message(model_name, f"Running: {cmd_str}", verbose=verbose,
                level='debug')

    # start_new_session puts the child in its own process group so a
    # timeout can SIGTERM the whole tree, not just the direct child.
    proc = None
    argv = [str(c) for c in cmd]
    # setpriv sets the parent-death signal, then the stack-limit shell
    # raises the stack, then the binary is exec'd (see uacpy._stack).
    prefix = parent_death_prefix() + stack_limit_prefix()
    try:
        if prefix:
            # The binary is exec'd by the wrappers, so a missing,
            # non-executable or unrecognised file would surface as a
            # wrapper's exit status; raise the OSError a direct exec gives.
            _check_executable(argv[0], env)
        proc = subprocess.Popen(
            prefix + argv,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # The engines echo byte-cut titles and other raw bytes; a
            # stray invalid byte must not turn a run into a decode error.
            encoding='utf-8',
            errors='replace',
            env=env,
            start_new_session=(os.name == 'posix'),
        )
        stdout, stderr = proc.communicate(timeout=timeout)
        result = subprocess.CompletedProcess(
            proc.args, proc.returncode, stdout, stderr,
        )
    except OSError as e:
        # ``Popen`` reports a failed chdir with ``filename`` set to the
        # cwd, so a work directory deleted mid-run arrives here naming that
        # directory. Matching on it keeps the failure from being announced
        # as "Executable not found" quoting a path that was never a binary.
        if e.filename is not None and Path(e.filename) == Path(cwd):
            raise ModelExecutionError(
                model_name, return_code=-1, stdout=None,
                stderr=(f"Work directory is no longer usable: {e}. It was "
                        f"deleted, moved or made inaccessible while the "
                        f"run was in progress."),
            ) from e
        if isinstance(e, FileNotFoundError):
            raise ModelExecutionError(
                model_name, return_code=-1,
                stdout=None, stderr=f"Executable not found: {e}.",
            ) from e
        # A binary that is present but will not exec — permission denied,
        # or ENOEXEC from an unexpanded LFS pointer, a wrong-architecture
        # build or a half-extracted archive — is a broken install, and its
        # remediation is the install script's, so it joins the
        # missing-binary case instead of escaping as a bare OSError.
        if isinstance(e, PermissionError) or e.errno == errno.ENOEXEC:
            raise ExecutableNotFoundError(
                model_name, str(cmd[0]), reason=e.strerror,
            ) from e
        raise
    except subprocess.TimeoutExpired as e:
        # Kill the whole process group, not just the direct child.
        if proc is not None:
            terminate_process_group(proc)
        # ``TimeoutExpired.stdout`` holds the unread pipe chunks as raw
        # bytes even under ``encoding=``, so it is decoded here with the
        # same ``errors='replace'`` as the pipe itself.
        raise ModelExecutionError(
            model_name, return_code=-1,
            stdout=(e.stdout.decode('utf-8', errors='replace')
                    if isinstance(e.stdout, bytes) else e.stdout),
            stderr=f"Timed out after {timeout}s.",
            timed_out=True,
        ) from e
    except BaseException:
        # start_new_session detaches the child from the terminal's signals,
        # so an interrupt reaches this process alone and the binary would
        # keep running. The wrapper owns the whole tree, so the group is
        # reaped before the exception continues on its way.
        if proc is not None and proc.poll() is None:
            terminate_process_group(proc)
        raise

    if result.returncode != 0:
        raise ModelExecutionError(
            model_name,
            return_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    return result


@dataclass(frozen=True)
class Launch:
    """One launch of an engine binary, as :func:`run_launch` runs it.

    Attributes
    ----------
    argv : tuple
        The command line.
    cwd : Path
        The work directory the binary runs in.
    env : dict or None
        The child's environment; ``None`` is this process's.
    timeout : float or None
        Seconds before the launch is killed; ``None`` is the model's.
    stale_outputs : tuple of str
        Names of the files under ``cwd`` the binary writes, removed before
        it starts, so a pinned work directory cannot hand an earlier run's
        output back as this run's answer.
    prt_root : str or None
        The root of the ``.prt`` log the binary writes: removed with the
        stale outputs, and its tail attached to a failure.
    tolerate_exit : callable or None
        ``tolerate_exit(exc)`` on a failed launch: ``True`` reads the run
        anyway (Kraken's field.exe teardown), ``False`` raises. ``None``
        raises every failure.
    checks : tuple of callables
        Called in order with the ``CompletedProcess`` (``None`` after a
        tolerated exit): the engine's fatal scans and warning passes over
        its marker tables, and the outputs the run must leave.
    stdout_label : str or None
        The name the child's stdout is logged under at the ``debug`` level;
        ``None`` logs nothing.
    """
    argv: tuple
    cwd: Path
    env: Optional[dict] = None
    timeout: Optional[float] = None
    stale_outputs: tuple = ()
    prt_root: Optional[str] = None
    tolerate_exit: Optional[Callable] = None
    checks: tuple = ()
    stdout_label: Optional[str] = None


def run_launch(launch: Launch, *, run, log=None):
    """Run ``launch`` and return its ``CompletedProcess`` (``None`` after a
    tolerated exit): the stale outputs and the ``.prt`` removed, the binary
    run through ``run`` (a model's ``_run_subprocess``, the one seam every
    launch passes), a failure raised with the ``.prt`` tail attached unless
    ``tolerate_exit`` accepts it, the stdout logged through ``log`` (a
    ``log(message, level=...)`` callable, passed only when the ``debug``
    level prints), then the checks in order."""
    cwd = Path(launch.cwd)
    prt = () if launch.prt_root is None else (f'{launch.prt_root}.prt',)
    for name in tuple(launch.stale_outputs) + prt:
        cwd.joinpath(name).unlink(missing_ok=True)
    try:
        result = run(list(launch.argv), cwd=launch.cwd,
                     timeout=launch.timeout, env=launch.env)
    except ModelExecutionError as exc:
        if launch.tolerate_exit is None or not launch.tolerate_exit(exc):
            if launch.prt_root is not None:
                attach_prt_tail(exc, launch.cwd, launch.prt_root)
            raise
        result = None
    if (log is not None and launch.stdout_label is not None
            and result is not None and result.stdout):
        log(f"{launch.stdout_label} output:\n{result.stdout}", level='debug')
    for check in launch.checks:
        check(result)
    return result


def require_output(model_name: str, candidates, *, what: str, process=None,
                   hint: str = '', prt_base: Optional[str] = None,
                   work_dir=None) -> Path:
    """Return the first of ``candidates`` that exists and is non-empty.

    None of them means the binary failed without a usable exit status —
    the OALIB/OASES/RAM engines routinely exit 0 after writing nothing.
    The raised :class:`ModelExecutionError` names every checked path and
    carries whatever diagnostics the engine family has:

    * ``process`` — the binary's :class:`subprocess.CompletedProcess`;
      its stdout/stderr tails are quoted (OASES and the RAM family write
      no print file, so the streams are the only record of the failure).
    * ``prt_base`` + ``work_dir`` — appends the tail of
      ``<work_dir>/<prt_base>.prt`` (the Acoustics-Toolbox models log
      fatal errors there rather than on their streams).
    """
    for candidate in candidates:
        path = Path(candidate)
        if path.exists() and path.stat().st_size > 0:
            return path
    checked = ', '.join(str(c) for c in candidates)
    exc = ModelExecutionError(
        model_name,
        return_code=getattr(process, 'returncode', 0),
        stdout=_tail(getattr(process, 'stdout', None)),
        stderr=(
            f"{model_name} did not produce {what}. Checked: "
            f"{checked}." + (f" {hint}" if hint else "")
            + _stream_block('binary stderr',
                            getattr(process, 'stderr', None))
        ),
    )
    if prt_base is not None and work_dir is not None:
        attach_prt_tail(exc, work_dir, prt_base)
    raise exc


def attach_prt_tail(exc, work_dir, base_name, tail_bytes: int = 2000):
    """Append the tail of the binary's ``<base>.prt`` log to ``exc``.

    Acoustics-Toolbox binaries dump fatal errors (``*** FATAL ERROR ***``)
    to ``.prt`` instead of stderr; surface that tail on the raised
    ``ModelExecutionError`` so the user sees the actual cause, not just a
    "check the .prt file" pointer.

    Updates ``exc.message`` (what ``UACPYError.__str__`` renders) **and**
    ``exc.stderr`` (a constructor arg, so the tail survives the pickle
    round-trip that ``run_parallel`` relies on — the rebuilt message is
    re-derived from ``stderr``), keeping ``exc.args`` in sync.
    """
    tail = read_prt(Path(work_dir) / f"{base_name}.prt",
                    tail_bytes=tail_bytes)
    if tail is None:
        return
    block = f"\n\n.prt tail:\n{tail}"
    if getattr(exc, 'message', None) is not None:
        exc.message += block
    if hasattr(exc, 'stderr'):
        exc.stderr = (exc.stderr + block) if exc.stderr else f".prt tail:\n{tail}"
    head = getattr(exc, 'message', None) or (
        f"{exc.args[0]}{block}" if exc.args else f"{exc}{block}")
    exc.args = (head,) + exc.args[1:]


def raise_on_fortran_fatal(model_name: str, result, work_dir, base_name, *,
                           benign=()):
    """Raise when a binary reported a fatal error but exited 0.

    Acoustics-Toolbox errors funnel through ``ERROUT``
    (``misc/FatalError.f90:18,30``), which writes ``*** FATAL ERROR ***``
    to the ``.prt`` and then ends in ``STOP '<string>'`` — a *character*
    stop code, which gfortran exits **0** for. A return-code test therefore
    never fires, and the binary leaves its output file untouched: with a
    pinned ``work_dir`` the previous run's ``.mod`` / ``.shd`` is still on
    disk and would be read as this run's answer.

    stderr is the authoritative signal because it belongs to this process;
    the ``.prt`` is checked too since it names the actual cause.

    The test is on the *form* of the stop, not on any banner text. A
    character stop code is how both toolchains report an abnormal end, and
    their banners do not share a marker: AT ends at
    ``STOP 'Fatal Error: …'`` via ``ERROUT`` but also stops directly with
    ``STOP 'ERROR IN KRAKENC: …'`` and ``STOP 'FATAL ERROR in BandPass: …'``,
    while OASES uses 46 distinct banners of which only 19 carry ``***`` —
    26 use ``>>> … <<<`` and one is bare ``'INVALID INPATCH'``. Matching
    ``***`` caught well under half of them, and a missed stop is read as
    success off whatever output file happens to be on disk.
    """
    stderr = result.stderr or ''
    prt = read_prt(Path(work_dir) / f"{base_name}.prt")
    fatal_stderr = bool(re.search(r'^\s*STOP\s+\S', stderr, re.MULTILINE))
    if not fatal_stderr and (
            prt is None or '*** FATAL ERROR ***' not in prt):
        return
    if prt and any(m in prt for m in benign):
        return
    exc = ModelExecutionError(
        model_name, return_code=0, stdout=result.stdout,
        stderr=stderr or "binary reported a fatal error and exited 0",
    )
    attach_prt_tail(exc, work_dir, base_name)
    raise exc


def terminate_process_group(proc) -> None:
    """Reap ``proc`` and everything it spawned, then close its pipes.

    Binaries are launched with ``start_new_session``, so they sit in their
    own process group and survive a signal delivered only to this process.
    SIGTERM to the group first, escalating to SIGKILL if it does not exit.
    Nothing reads the pipes of a reaped process again, and left open their
    file objects outlive it and are reported as a ``ResourceWarning``
    wherever the garbage collector next runs.
    """
    try:
        if os.name == 'posix':
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(pgid, signal.SIGKILL)
                    proc.wait()
                return
            except (ProcessLookupError, PermissionError):
                pass
        proc.kill()
        proc.wait()
    finally:
        for pipe in (proc.stdin, proc.stdout, proc.stderr):
            if pipe is not None:
                pipe.close()


def warn_on_prt_warnings(model_name: str, work_dir, base_name) -> None:
    """Surface the binary's own non-fatal ``Warning in ...`` lines.

    The AT binaries write both fatals and *non-fatal* diagnoses to the
    ``.prt``. Only the fatals were read (``_attach_prt_tail``, on the
    exception path), so a run the solver itself diagnosed came back at
    exit 0 with a full-size result and nothing said — measured, BELLHOP
    writes ``Warning in BELLHOP : Too few beams`` and uacpy emitted zero
    warnings while returning a TL field the binary had just called
    under-sampled.

    These are the solver's words, not uacpy's, so they are passed through
    verbatim rather than reinterpreted.
    """
    text = read_prt(Path(work_dir) / f"{base_name}.prt", tail_bytes=200000)
    if not text:
        return
    seen, lines = set(), []
    for raw in text.splitlines():
        line = raw.strip()
        if line.lower().startswith('warning in') and line not in seen:
            seen.add(line)
            lines.append(line)
    if lines:
        joined = "\n  ".join(lines)
        warnings.warn(
            f"{model_name} reported {len(lines)} non-fatal "
            f"warning(s) in its .prt log:\n  {joined}",
            IOWarning, skip_file_prefixes=USER_FRAME_SKIP)
