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
    except (ImportError, ValueError, OSError):
        pass


raise_stack_limit()
