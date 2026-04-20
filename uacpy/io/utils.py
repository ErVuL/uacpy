"""
Utility functions for IO operations
"""

import numpy as np
from typing import Union


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


def crci(c_real: float, c_imag: float, freq: float, atten_unit: str) -> complex:
    """
    Compute complex sound speed from real part, imaginary part, and attenuation units.

    Parameters
    ----------
    c_real : float
        Real part of sound speed (m/s)
    c_imag : float
        Imaginary part or attenuation value
    freq : float
        Frequency in Hz
    atten_unit : str
        Attenuation units:
        - 'N': Nepers/m
        - 'F': dB/m-kHz
        - 'M': dB/m
        - 'W': dB/wavelength
        - 'Q': Quality factor Q
        - 'L': Loss tangent

    Returns
    -------
    c_complex : complex
        Complex sound speed

    Notes
    -----
    Converts attenuation specified in various units to the imaginary part
    of the complex sound speed. The complex sound speed is used in the
    Helmholtz equation to represent absorption.

    The relationship between attenuation α and imaginary sound speed:
        c = c_real + i*c_imag
        α = ω * c_imag / c_real^2

    Examples
    --------
    >>> # Water with 0.1 dB/wavelength attenuation at 100 Hz
    >>> c = crci(1500, 0.1, 100, 'W')
    >>> print(f"c = {c.real:.2f} + {c.imag:.4f}i m/s")

    >>> # No attenuation
    >>> c = crci(1500, 0, 100, 'W')
    >>> print(f"c = {c:.2f}")
    """
    if atten_unit == 'N':
        # Nepers/m - already in correct form
        # c_imag is directly the imaginary part
        return complex(c_real, c_imag)

    elif atten_unit == 'F':
        # dB/m-kHz
        # Convert to nepers/m: α = c_imag * freq * log(10) / (40 * 1000)
        omega = 2 * np.pi * freq
        alpha_nepers = c_imag * freq * np.log(10) / (40.0 * 1000.0)
        c_imag_calc = alpha_nepers * c_real**2 / omega
        return complex(c_real, c_imag_calc)

    elif atten_unit == 'M':
        # dB/m
        # Convert to nepers/m: α = c_imag * log(10) / 40
        omega = 2 * np.pi * freq
        alpha_nepers = c_imag * np.log(10) / 40.0
        c_imag_calc = alpha_nepers * c_real**2 / omega
        return complex(c_real, c_imag_calc)

    elif atten_unit == 'W':
        # dB/wavelength
        # Convert to nepers/wavelength then to nepers/m
        wavelength = c_real / freq
        alpha_nepers_per_wavelength = c_imag * np.log(10) / 40.0
        alpha_nepers = alpha_nepers_per_wavelength / wavelength
        omega = 2 * np.pi * freq
        c_imag_calc = alpha_nepers * c_real**2 / omega
        return complex(c_real, c_imag_calc)

    elif atten_unit == 'Q':
        # Quality factor Q
        # α = ω / (2 * Q * c)
        # c_imag = α * c_real^2 / ω = c_real / (2 * Q)
        if c_imag != 0:
            c_imag_calc = c_real / (2.0 * c_imag)
        else:
            c_imag_calc = 0.0
        return complex(c_real, c_imag_calc)

    elif atten_unit == 'L':
        # Loss tangent
        # tan(δ) = c_imag / c_real
        c_imag_calc = c_imag * c_real
        return complex(c_real, c_imag_calc)

    else:
        raise ValueError(f"Unknown attenuation unit: {atten_unit}")


def complex_ssp(alphaR: np.ndarray, alphaI: np.ndarray, freq: float,
                atten_unit: str = 'W') -> np.ndarray:
    """
    Create complex sound speed profile from real and imaginary parts.

    Parameters
    ----------
    alphaR : ndarray
        Real part of sound speeds (m/s)
    alphaI : ndarray
        Imaginary part or attenuation values
    freq : float
        Frequency in Hz
    atten_unit : str, optional
        Attenuation units (default: 'W' for dB/wavelength)

    Returns
    -------
    c_complex : ndarray
        Complex sound speed profile

    Notes
    -----
    Applies crci() to each point in the profile.

    Examples
    --------
    >>> z = np.linspace(0, 100, 11)
    >>> c_real = 1500 - 0.1 * z  # Linear gradient
    >>> c_imag = 0.1 * np.ones_like(z)  # Constant attenuation
    >>> c_complex = complex_ssp(c_real, c_imag, 100, 'W')
    """
    alphaR = np.asarray(alphaR)
    alphaI = np.asarray(alphaI)

    if alphaR.shape != alphaI.shape:
        raise ValueError("alphaR and alphaI must have the same shape")

    c_complex = np.zeros(alphaR.shape, dtype=complex)
    for i, (cr, ci) in enumerate(zip(alphaR.ravel(), alphaI.ravel())):
        c_complex.ravel()[i] = crci(cr, ci, freq, atten_unit)

    return c_complex
