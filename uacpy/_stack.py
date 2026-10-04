"""Stack limit for the Fortran model binaries, raised in the child only.

SPARC-class binaries can blow an 8 MiB default stack on the first large
allocation (``Scooter/sparc.f90:354-355`` declares twelve automatic
``COMPLEX(NTot1)`` arrays), and ``mpiramS/README.OMP`` asks for
``ulimit -s unlimited``. :func:`stack_limit_prefix` returns an argv prefix
that raises the soft ``RLIMIT_STACK`` to the hard limit inside the spawned
shell and then ``exec``\\ s the binary, so the limit of the Python process
that imports uacpy is never touched. A ``preexec_fn`` would do the same
without the extra hop, but Python documents it as unsafe once threads exist,
and uacpy runs models from threads.

The prefix is empty — the binary inherits the caller's limit — where there
is nothing to raise (soft already at the hard limit), no POSIX shell or
``resource`` module (Windows), or ``UACPY_NO_STACK_RAISE=1`` is set;
SPARC-class models may then segfault on large problems.

:func:`parent_death_prefix` is the second launch wrapper: on Linux it has
``setpriv --pdeathsig TERM`` set the child's parent-death signal before the
stack-limit shell and the binary are exec'd, so the binary is terminated when
the Python process that launched it dies — including by SIGKILL, which no
Python-side handler can see.
"""

import functools
import os
import shutil
import subprocess
import sys
from typing import List


def _opted_out() -> bool:
    # Truthy opt-out only: '0'/'false'/'no' keep the default behaviour
    # (raising), since someone setting 0 means "do not disable".
    return os.environ.get('UACPY_NO_STACK_RAISE', '').strip().lower() not in (
        '', '0', 'false', 'no')


def stack_limit_prefix() -> List[str]:
    """Argv prefix that raises the child's soft stack limit to its hard limit.

    ``['/bin/sh', '-c', 'ulimit -s <hard> 2>/dev/null; exec "$0" "$@"']``:
    the command follows as ``$0 $@``, so ``exec`` replaces the shell with the
    binary (same pid, same process group). A refused ``ulimit`` (a hardened
    container) leaves the inherited limit and the binary still runs.
    """
    if os.name != 'posix' or _opted_out():
        return []
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    except (ImportError, ValueError, OSError):
        return []
    if soft == hard:
        return []
    sh = shutil.which('sh') or ('/bin/sh' if os.path.exists('/bin/sh')
                                else None)
    if sh is None:
        return []
    # ``ulimit -s`` counts KiB.
    value = ('unlimited' if hard == resource.RLIM_INFINITY
             else str(int(hard) // 1024))
    return [sh, '-c', f'ulimit -s {value} 2>/dev/null; exec "$0" "$@"']


@functools.lru_cache(maxsize=1)
def parent_death_prefix() -> List[str]:
    """Argv prefix that terminates the binary when its launching process dies.

    ``['setpriv', '--pdeathsig', 'TERM', '--']``: util-linux ``setpriv``
    (2.33 and later) sets ``PR_SET_PDEATHSIG`` and execs the rest of the argv;
    the setting survives each later ``execve`` (the stack-limit shell's
    ``exec`` included), so it lands on the binary. Without it a binary started
    in its own session (``start_new_session``) outlives a Python process ended
    by SIGTERM or SIGKILL — an outer ``timeout``, a batch scheduler, a
    notebook kernel restart — and keeps running under init: three KRAKENC
    runs were found at 98 % CPU after 2.5-3 h. The signal fires when the
    forking thread exits, and that thread is blocked in ``communicate`` until
    the binary ends, so it does not fire early.

    Empty where it cannot be provided — not Linux, no ``setpriv``, or one too
    old to know ``--pdeathsig`` — and the binary is then launched exactly as
    before. Checked once per process.
    """
    if not sys.platform.startswith('linux'):
        return []
    setpriv = shutil.which('setpriv')
    if setpriv is None:
        return []
    try:
        probe = subprocess.run([setpriv, '--help'], capture_output=True,
                               text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    if '--pdeathsig' not in (probe.stdout or '') + (probe.stderr or ''):
        return []
    return [setpriv, '--pdeathsig', 'TERM', '--']
