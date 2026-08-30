"""
Utility functions for IO operations
"""

import numpy as np

from uacpy.core.exceptions import ConfigurationError


def equally_spaced(x: np.ndarray, tol: float = 1e-9) -> bool:
    """
    Test whether vector x is composed of equally-spaced values.

    Parameters
    ----------
    x : ndarray
        Vector to test
    tol : float, optional
        Tolerance for equality test. Default is 1e-9.

    Returns
    -------
    is_equal : bool
        True if x is equally spaced within tolerance

    Notes
    -----
    Compares the input vector against a linearly spaced vector with
    the same start, end, and number of points. Returns True if the
    maximum absolute difference is less than the tolerance.

    This is useful for determining if a vector can be represented
    compactly (e.g., "N points from x0 to x1") rather than storing
    all values explicitly.

    Translated from OALIB equally_spaced.m

    Examples
    --------
    >>> # Equally spaced
    >>> x = np.linspace(0, 10, 11)
    >>> equally_spaced(x)
    True

    >>> # Not equally spaced
    >>> x = np.array([0, 1, 3, 7, 10])
    >>> equally_spaced(x)
    False

    >>> # Jitter on one interior sample beyond tolerance
    >>> x = np.linspace(0, 10, 11); x[5] += 1e-6
    >>> equally_spaced(x)
    False
    """
    x = np.asarray(x).ravel()
    n = len(x)

    if n <= 1:
        return True

    # Generate equally spaced vector
    x_linspace = np.linspace(x[0], x[-1], n)

    # Compute maximum deviation
    delta = np.abs(x - x_linspace)

    # bool(), not the bare comparison: ``np.max(...) < tol`` is a numpy scalar,
    # and np.bool_ is NOT a Python bool — ``isinstance(r, bool)`` and
    # ``r is True`` are both False, and json.dumps raises TypeError on it. The
    # ``n <= 1`` branch above already returns a real bool, so without this the
    # return type depended on the input length.
    return bool(np.max(delta) < tol)


def reject_unknown_kwargs(writer: str, kwargs: dict, known) -> None:
    """Raise on a writer knob no block of this deck reads.

    Every deck writer that accepts ``**kwargs`` calls this first, so a
    misspelled option fails loudly instead of being silently dropped and
    leaving the deck subtly different from what the caller asked for.
    """
    unknown = sorted(set(kwargs) - set(known))
    if unknown:
        raise ConfigurationError(
            f"{writer}: parameter(s) {unknown} are not read by this deck.",
            remediation=f"Drop them, or check they belong to this writer's "
                        f"program; it reads {sorted(known)}.",
        )


def _collapsed_pair_index(written, *, raw=None, min_step=None):
    """Index ``i`` of the adjacent pair ``(i, i+1)`` a deck column loses,
    or ``None`` when every pair survives.

    ``written`` is the axis as the file will hold it — formatted tokens
    (accepted directly) or values rounded to the column's resolution. One
    rule per way a writer loses an axis, selected by the keyword:

    * default — the written column must be strictly increasing; when it is
      not (NaN pairs included, since NaN fails every comparison), the index
      of the smallest step comes back.
    * ``raw=`` — only a pair whose ``raw`` step is positive and whose
      written step is zero counts: a value the caller repeated on purpose
      is not a collision. The first such pair comes back.
    * ``min_step=`` — steps of ``written`` below ``min_step`` fail even
      where the tokens stay distinct; the index of the smallest step comes
      back. Callers pass the unrounded axis here: the rule bounds spacing,
      not token identity.

    Callers raise their own :class:`ConfigurationError` naming the pair,
    the column's resolution and the engine consequence — the six deck
    writers that lose axes this way share the detection, not the message.
    """
    written = np.asarray(written, dtype=float)
    if written.size <= 1:
        return None
    steps = np.diff(written)
    if min_step is not None:
        if steps.min() < min_step:
            return int(np.argmin(steps))
        return None
    if raw is not None:
        collision = ((np.diff(np.asarray(raw, dtype=float)) > 0.0)
                     & (steps == 0.0))
        if collision.any():
            return int(np.argmax(collision))
        return None
    if not np.all(steps > 0):
        return int(np.argmin(steps))
    return None
