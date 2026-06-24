"""Process-wide stack-limit setup — imported once before the heavy
Fortran-binary models load.

SPARC-class binaries can blow an 8 MiB default stack on the first large
allocation; raising RLIMIT_STACK to the hard limit at import time means
every subprocess spawned later inherits the larger value.
"""


def raise_stack_limit() -> None:
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
