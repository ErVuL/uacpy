"""The prefix set that points a ``warnings.warn`` at the caller's own code.

``warnings.warn(..., skip_file_prefixes=USER_FRAME_SKIP)`` reports the first
stack frame whose file lies outside the shipped package, however many model /
io / core layers sit in between — so a warning raised deep in a reader still
names the user's ``run()`` or constructor call, and two different call sites
get two different dedup keys instead of collapsing onto one library line.

The entries are the package's own subdirectories and top-level modules. A
directory carries a trailing ``os.sep`` and a module drops its ``.py``, and
both endings are load-bearing in opposite directions.

The separator bounds the match: without it the prefix is a bare string match
that also swallows every sibling path sharing the package directory's name —
``<pkg>_venv/lib/...``, a ``<pkg>.egg-info`` — and attribution then walks past
the frame it wanted.

Dropping the suffix is what makes a module entry match at all. CPython skips a
frame whose filename *starts with* an entry and is longer than it; an entry
equal to the filename in full skips nothing. Listed by its whole path, a
top-level module is inert, and a warning raised two frames deep inside one
(``run_parallel`` calling its own reaper, say) lands on the outer of those two
frames rather than on the caller. Both endings are pinned by tests, in both
directions.

``tests`` and ``examples`` are excluded because their files play the caller
role the attribution points at.

Pass this set *instead of* ``stacklevel``, never alongside it. The walk already
lands on the first frame outside the package, so a ``stacklevel`` of 3 or more
carries it that many frames further and names the caller's own caller — over-
skipping, the mirror of the frame counts this set exists to replace, and
silent in exactly the same way. Levels 1 and 2 are indistinguishable here
(CPython raises the level to 2 before walking), so only 3 and above bite. No
call site in the package combines them today; a test asserts that.

This module sits in ``core`` so every layer can import it without importing a
model. ``uacpy.models.base`` re-exports the tuple built here under the same
name, so the two names are one object rather than two definitions free to
drift; a test asserts that identity, which a re-introduced second definition
would fail.
"""

import os
from pathlib import Path

_PACKAGE_DIR = Path(__file__).parent.parent

#: Suffix dropped from a top-level module's path to keep its entry a *strict*
#: prefix of that module's filename. CPython skips a frame whose filename
#: starts with an entry and is longer than it; an entry equal to the filename
#: in full does not skip. A module listed by its whole path is therefore inert,
#: and a warning raised two frames deep inside one lands on the outer of those
#: two frames instead of on the caller (measured, both ways).
_MODULE_SUFFIX = '.py'

USER_FRAME_SKIP = tuple(
    str(entry) + os.sep if entry.is_dir() else str(entry)[:-len(_MODULE_SUFFIX)]
    for entry in sorted(_PACKAGE_DIR.iterdir())
    if entry.name not in ('tests', 'examples')
    and (entry.is_dir() or entry.suffix == _MODULE_SUFFIX)
)
