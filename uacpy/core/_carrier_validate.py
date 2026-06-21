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


def _hint(hint: str) -> str:
    """Render an optional unit/context note appended after the rule clause.

    Kept separate from ``label`` so ``label`` ends immediately before
    ``"must be"`` — callers (and tests) can match ``"<noun> must be"`` as a
    contiguous phrase while the unit context still survives in the message.
    """
    return f" ({hint})" if hint else ""


def _require_finite(values, label: str, *, hint: str = "") -> None:
    """Raise ``ConfigurationError`` if any element is NaN or inf.

    Accepts a scalar or any array-like. Shared by the carriers so the
    "must be finite" guard reads identically everywhere instead of being
    re-inlined per attribute.
    """
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ConfigurationError(
            f"{label} must be finite (no NaN/inf){_hint(hint)}; "
            f"got {arr.ravel().tolist()}")


def _require_positive(values, label: str, *, hint: str = "") -> None:
    """Raise ``ConfigurationError`` unless every element is finite and ``> 0``.

    Finiteness is checked first (NaN/inf pass every plain ``<= 0`` test) and
    reported separately, so the sign message stays the contiguous phrase
    ``"<label> must be positive"`` that callers relied on.
    """
    arr = np.asarray(values, dtype=float)
    _require_finite(arr, label, hint=hint)
    if np.any(arr <= 0):
        raise ConfigurationError(
            f"{label} must be positive, > 0{_hint(hint)}; "
            f"got {arr.ravel().tolist()}")


def _require_non_negative(values, label: str, *, hint: str = "") -> None:
    """Raise ``ConfigurationError`` unless every element is finite and ``>= 0``.

    Finiteness is reported separately so the sign message stays the contiguous
    phrase ``"<label> must be non-negative"``.
    """
    arr = np.asarray(values, dtype=float)
    _require_finite(arr, label, hint=hint)
    if np.any(arr < 0):
        raise ConfigurationError(
            f"{label} must be non-negative, >= 0{_hint(hint)}; "
            f"got {arr.ravel().tolist()}")


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
