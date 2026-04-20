"""
Utility functions for uacpy
"""

import numpy as np


def db_to_linear(db: np.ndarray) -> np.ndarray:
    """
    Convert dB to linear scale

    Parameters
    ----------
    db : ndarray
        Values in dB

    Returns
    -------
    linear : ndarray
        Values in linear scale
    """
    return 10 ** (db / 20)


def linear_to_db(linear: np.ndarray) -> np.ndarray:
    """
    Convert linear scale to dB

    Parameters
    ----------
    linear : ndarray
        Values in linear scale

    Returns
    -------
    db : ndarray
        Values in dB
    """
    return 20 * np.log10(np.abs(linear) + 1e-20)


def wavelength(frequency: float, sound_speed: float = 1500.0) -> float:
    """
    Calculate acoustic wavelength

    Parameters
    ----------
    frequency : float
        Frequency in Hz
    sound_speed : float, optional
        Sound speed in m/s. Default is 1500 m/s.

    Returns
    -------
    wavelength : float
        Wavelength in meters
    """
    return sound_speed / frequency


def range_dependent_grid(
    ranges: np.ndarray,
    depths: np.ndarray
) -> tuple:
    """
    Create 2D meshgrid for range-dependent environments

    Parameters
    ----------
    ranges : ndarray
        Range vector
    depths : ndarray
        Depth vector

    Returns
    -------
    R : ndarray
        Range meshgrid
    Z : ndarray
        Depth meshgrid
    """
    return np.meshgrid(ranges, depths)


def equally_spaced(x: np.ndarray, tol: float = 1e-9) -> bool:
    """
    Test if vector elements are equally spaced

    Parameters
    ----------
    x : ndarray
        Vector to test
    tol : float, optional
        Tolerance for equality. Default is 1e-9.

    Returns
    -------
    is_equal : bool
        True if equally spaced, False otherwise

    Notes
    -----
    Checks if all differences between consecutive elements are equal
    within the specified tolerance.

    Used by file writers to determine if vectors can be written as
    "min max" pairs instead of listing all values.

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
    Compute Pekeris branch of square root for halfspace wavenumbers

    The Pekeris root selects the appropriate branch of sqrt(gamma2) to
    ensure outgoing waves in the halfspace (proper radiation condition).

    Parameters
    ----------
    gamma2 : ndarray (complex)
        Squared vertical wavenumber: gamma^2 = k^2 - k_halfspace^2
    tol : float, optional
        Tolerance for branch cut detection (default: 1e-10)
        Used to avoid numerical instabilities near critical angle

    Returns
    -------
    gamma : ndarray (complex)
        Vertical wavenumber with correct branch selection

    Notes
    -----
    Branch selection rule (Pekeris, 1948):
    - If Re(gamma2) > tol: propagating in halfspace, choose Re(gamma) > 0 (outgoing wave)
    - If Re(gamma2) < -tol: evanescent in halfspace, choose Im(gamma) < 0 (exponential decay)
    - If |Re(gamma2)| <= tol: critical angle region, use Im(gamma2) to determine branch

    This ensures exponential decay into the halfspace for trapped modes
    and outgoing radiation for leaky modes.

    The tolerance parameter prevents numerical instabilities near the branch cut
    (critical angle) where floating-point errors can cause incorrect branch selection.

    References
    ----------
    Pekeris, C.L., "Theory of propagation of explosive sound in shallow
    water," Geol. Soc. Am. Mem. 27 (1948).

    Examples
    --------
    >>> gamma2 = np.array([1.0 + 0j, -1.0 + 0j, 1e-15 + 0j])
    >>> gamma = pekeris_root(gamma2)
    >>> # For gamma2 = 1: Re(gamma) > 0 (propagating)
    >>> # For gamma2 = -1: Im(gamma) < 0 (evanescent)
    >>> # For gamma2 ≈ 0: handled as critical angle case
    """
    # Ensure input is complex
    gamma2 = np.asarray(gamma2, dtype=complex)

    # Compute principal square root
    gamma = np.sqrt(gamma2)

    # Extract real and imaginary parts of gamma2 for branch selection
    re_gamma2 = np.real(gamma2)
    im_gamma2 = np.imag(gamma2)

    # Initialize mask for values that need sign flip
    needs_flip = np.zeros(gamma2.shape, dtype=bool)

    # Region 1: Re(gamma2) > tol (propagating in halfspace)
    # Physics: outgoing wave, choose Re(gamma) > 0
    mask_propagating = re_gamma2 > tol
    needs_flip_prop = mask_propagating & (np.real(gamma) < 0)
    needs_flip |= needs_flip_prop

    # Region 2: Re(gamma2) < -tol (evanescent in halfspace)
    # Physics: exponentially decaying wave, choose Im(gamma) < 0
    mask_evanescent = re_gamma2 < -tol
    needs_flip_evan = mask_evanescent & (np.imag(gamma) > 0)
    needs_flip |= needs_flip_evan

    # Region 3: |Re(gamma2)| <= tol (critical angle region)
    # Near branch cut - use Im(gamma2) to determine branch
    mask_critical = np.abs(re_gamma2) <= tol

    # Standard convention: choose Im(gamma) < 0 for energy flowing upward (Im(gamma2) >= 0)
    needs_flip_crit = mask_critical & (im_gamma2 >= 0) & (np.imag(gamma) > 0)
    needs_flip |= needs_flip_crit

    # Also flip if Im(gamma2) < 0 (energy downward) and Im(gamma) < 0
    needs_flip_crit2 = mask_critical & (im_gamma2 < 0) & (np.imag(gamma) < 0)
    needs_flip |= needs_flip_crit2

    # Apply sign flip where needed
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
