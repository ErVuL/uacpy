"""``python -m uacpy``: print which engine binaries this install holds
(:func:`uacpy.models.engine_status`) and exit 0 when every engine has a
binary found, 1 otherwise. An optional install none of whose binaries is
found (OASES, ``./install.sh --oases no``) is an install not made and does
not fail the check. No binary is run."""

import sys

from uacpy import __version__
from uacpy.models import engine_status


def main() -> int:
    """Print the engine status table and a verdict line; return the exit
    status."""
    status = engine_status()
    print(f"uacpy {__version__}")
    print(status)
    missing = status.missing_engines()
    absent = [engine for engine
              in status.missing_engines(optional_installs=True)
              if engine not in missing]
    if absent:
        print(f"Optional install not made: {', '.join(absent)}.")
    if missing:
        print(f"No binary found for: {', '.join(missing)}. Build them with "
              f"./install.sh (./install.sh --help lists the options).")
        return 1
    print("Every installed engine has a binary.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
