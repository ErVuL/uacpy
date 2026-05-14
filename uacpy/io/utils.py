"""
Utility functions for IO operations
"""

import numpy as np


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

    >>> # Nearly equally spaced (within tolerance)
    >>> x = np.linspace(0, 10, 11) + 1e-12
    >>> equally_spaced(x)
    True

    >>> # Beyond tolerance
    >>> x = np.linspace(0, 10, 11) + 1e-6
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
