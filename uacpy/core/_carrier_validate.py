"""Shared input validators for the core environment carriers
(:mod:`uacpy.core.bottom`, :mod:`uacpy.core.ssp`, :mod:`uacpy.core.environment`).
"""

import numpy as np

from uacpy.core.exceptions import ConfigurationError


def _validate_acoustic_type(value, label: str) -> None:
    """Reject unrecognized ``acoustic_type`` strings up front, so a typo
    like ``'halfspace'`` (vs. ``'half-space'``) fails at construction
    instead of producing a wrong Acoustics-Toolbox bottom-type code
    deep inside a writer.
    """
    from uacpy.core.constants import BoundaryType
    try:
        BoundaryType.from_string(value)
    except (ConfigurationError, ValueError, KeyError, AttributeError) as exc:
        valid = sorted({bt.value for bt in BoundaryType})
        raise ConfigurationError(
            f"{label}: acoustic_type={value!r} is not recognized. "
            f"Valid values (plus the aliases handled by "
            f"BoundaryType.from_string): {valid}"
        ) from exc


def _require_strictly_increasing(values: np.ndarray, label: str) -> None:
    """Raise ``ConfigurationError`` if ``values`` is not strictly
    monotonically increasing. Used to guard every range / depth axis that
    feeds into ``np.interp``, which silently produces garbage on unsorted
    ``xp``.
    """
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size <= 1:
        return
    diffs = np.diff(arr)
    if not np.all(diffs > 0):
        bad = int(np.argmin(diffs))
        raise ConfigurationError(
            f"{label} must be strictly increasing; "
            f"got {arr[bad]} >= {arr[bad + 1]} at index {bad + 1} "
            f"(full axis: {arr.tolist()})"
        )


def _sanitize_title(name: str) -> str:
    """Strip newlines/control chars and escape single quotes in a Fortran
    title field. Acoustics-Toolbox `.env` titles are quote-delimited and
    column-sensitive; an unsanitized name with a newline silently corrupts
    the file and the binary parses garbage downstream.
    """
    if name is None:
        return 'unnamed'
    s = str(name)
    s = ''.join(ch if (ord(ch) >= 32 and ch != '\x7f') else ' ' for ch in s)
    s = s.replace("'", "''")
    return s.strip() or 'unnamed'
