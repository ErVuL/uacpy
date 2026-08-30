"""Process-wide stack-limit setup — imported once before the heavy
Fortran-binary models load.

SPARC-class binaries can blow an 8 MiB default stack on the first large
allocation; raising RLIMIT_STACK to the hard limit at import time means
every subprocess spawned later inherits the larger value.

The change is process-global (an unlimited RLIMIT_STACK also switches
Linux to the legacy bottom-up mmap layout) — set ``UACPY_NO_STACK_RAISE=1``
to keep the inherited limit; SPARC-class models may then segfault on
large problems.
"""

import os


def raise_stack_limit() -> None:
    # Truthy opt-out only: '0'/'false'/'no' keep the default behaviour
    # (raising), since someone setting 0 means "do not disable".
    if os.environ.get('UACPY_NO_STACK_RAISE', '').strip().lower() not in (
            '', '0', 'false', 'no'):
        return
    try:
        import resource
        _soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        target = (
            resource.RLIM_INFINITY
            if hard == resource.RLIM_INFINITY else hard
        )
        resource.setrlimit(resource.RLIMIT_STACK, (target, hard))
    except (ImportError, ValueError, OSError) as exc:
        # A hardened container may forbid raising RLIMIT_STACK; leave the
        # default in place but leave a breadcrumb, otherwise a later SPARC
        # stack overflow surfaces as an opaque subprocess segfault.
        from uacpy._log import log_message
        log_message('_stack',
                    f"could not raise RLIMIT_STACK ({exc!r}); "
                    "SPARC-class models may segfault on large allocations",
                    level='warning')


raise_stack_limit()
