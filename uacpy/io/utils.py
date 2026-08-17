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

    return np.max(delta) < tol


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
