"""Utility functions for uacpy."""

import numpy as np


def db_to_linear(db: np.ndarray) -> np.ndarray:
    """
    Convert dB to linear scale.

    Parameters
    ----------
    db : ndarray
        Values in dB.

    Returns
    -------
    linear : ndarray
        Values in linear scale.
    """
    return 10 ** (db / 20)


def linear_to_db(linear: np.ndarray) -> np.ndarray:
    """
    Convert linear scale to dB.

    Parameters
    ----------
    linear : ndarray
        Values in linear scale.

    Returns
    -------
    db : ndarray
        Values in dB.
    """
    return 20 * np.log10(np.abs(linear) + 1e-20)


def wavelength(frequency: float, sound_speed: float = 1500.0) -> float:
    """
    Compute acoustic wavelength.

    Parameters
    ----------
    frequency : float
        Frequency in Hz.
    sound_speed : float, optional
        Sound speed in m/s. Default is 1500 m/s.

    Returns
    -------
    wavelength : float
        Wavelength in meters.
    """
    return sound_speed / frequency


def range_dependent_grid(
    ranges: np.ndarray,
    depths: np.ndarray
) -> tuple:
    """
    Build a 2-D meshgrid for range-dependent environments.

    Parameters
    ----------
    ranges : ndarray
        Range vector.
    depths : ndarray
        Depth vector.

    Returns
    -------
    R : ndarray
        Range meshgrid.
    Z : ndarray
        Depth meshgrid.
    """
    return np.meshgrid(ranges, depths)


def equally_spaced(x: np.ndarray, tol: float = 1e-9) -> bool:
    """
    Test whether the elements of ``x`` are equally spaced.

    Parameters
    ----------
    x : ndarray
        Vector to test.
    tol : float, optional
        Tolerance for equality. Default is 1e-9.

    Returns
    -------
    is_equal : bool
        True if equally spaced, False otherwise.

    Notes
    -----
    Used by file writers to decide whether a vector can be written as a
    ``"min max"`` pair instead of listing all values.

    Examples
    --------
    >>> x = np.array([0, 1, 2, 3])
    >>> equally_spaced(x)
    True

    >>> x = np.array([0, 1, 2, 4])
    >>> equally_spaced(x)
    False
    """
    if len(x) < 2:
        return True

    dx = np.diff(x)
    return np.all(np.abs(dx - dx[0]) < tol)


def pekeris_root(gamma2: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Return the Pekeris branch of the complex square root.

    Selects the branch of sqrt(gamma2) that enforces the radiation
    condition in a halfspace (outgoing waves for propagating modes,
    exponential decay for evanescent modes).

    Parameters
    ----------
    gamma2 : ndarray, complex
        Squared vertical wavenumber, ``gamma^2 = k^2 - k_halfspace^2``.
    tol : float, optional
        Tolerance for branch-cut detection (default: 1e-10). Avoids
        numerical instabilities near the critical angle.

    Returns
    -------
    gamma : ndarray, complex
        Vertical wavenumber on the correct branch.

    Notes
    -----
    Branch selection rule (Pekeris, 1948):

    - ``Re(gamma2) > tol``: propagating, choose ``Re(gamma) > 0``.
    - ``Re(gamma2) < -tol``: evanescent, choose ``Im(gamma) < 0``.
    - ``|Re(gamma2)| <= tol``: critical angle; use ``Im(gamma2)`` to pick
      the branch.

    This guarantees exponential decay into the halfspace for trapped
    modes and outgoing radiation for leaky modes.

    References
    ----------
    Pekeris, C.L., "Theory of propagation of explosive sound in shallow
    water," Geol. Soc. Am. Mem. 27 (1948).

    Examples
    --------
    >>> gamma2 = np.array([1.0 + 0j, -1.0 + 0j, 1e-15 + 0j])
    >>> gamma = pekeris_root(gamma2)
    """
    gamma2 = np.asarray(gamma2, dtype=complex)
    gamma = np.sqrt(gamma2)

    re_gamma2 = np.real(gamma2)
    im_gamma2 = np.imag(gamma2)

    needs_flip = np.zeros(gamma2.shape, dtype=bool)

    # Propagating: outgoing wave, choose Re(gamma) > 0.
    mask_propagating = re_gamma2 > tol
    needs_flip |= mask_propagating & (np.real(gamma) < 0)

    # Evanescent: exponential decay, choose Im(gamma) < 0.
    mask_evanescent = re_gamma2 < -tol
    needs_flip |= mask_evanescent & (np.imag(gamma) > 0)

    # Near the branch cut: fall back to the sign of Im(gamma2).
    mask_critical = np.abs(re_gamma2) <= tol
    needs_flip |= mask_critical & (im_gamma2 >= 0) & (np.imag(gamma) > 0)
    needs_flip |= mask_critical & (im_gamma2 < 0) & (np.imag(gamma) < 0)

    gamma[needs_flip] = -gamma[needs_flip]

    return gamma


__all__ = [
    'db_to_linear',
    'linear_to_db',
    'wavelength',
    'range_dependent_grid',
    'equally_spaced',
    'pekeris_root',
]
