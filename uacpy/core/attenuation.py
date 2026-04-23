"""
Attenuation models for underwater acoustics

Implements various attenuation formulas including Thorp, Francois-Garrison,
and custom models.

Note
----
This module is a **user/example helper**.  The core uacpy library wires
frequency-dependent volume attenuation inside the individual model writers
(see ``uacpy.io.at_env_writer`` and model ``volume_attenuation`` kwargs).
Nothing in ``uacpy.core`` or ``uacpy.models`` imports from this module;
it exists primarily for example_17_attenuation_models.py and for users
who want to compute attenuation curves directly.
"""

import numpy as np
from typing import Union, Optional
from dataclasses import dataclass


@dataclass
class AttenuationParameters:
    """Parameters for frequency-dependent attenuation calculations.

    Attributes
    ----------
    temperature : float
        Water temperature in degrees Celsius.
    salinity : float
        Salinity in parts per thousand (ppt).
    pH : float
        Water pH value.
    depth : float
        Water depth in meters.
    """
    temperature: float = 10.0
    salinity: float = 35.0
    pH: float = 8.0
    depth: float = 1000.0


def thorp_attenuation(frequency: Union[float, np.ndarray]) -> np.ndarray:
    """
    Thorp attenuation formula

    Parameters
    ----------
    frequency : float or array
        Frequency in Hz

    Returns
    -------
    alpha : float or array
        Attenuation in dB/km

    References
    ----------
    Thorp, W. H. (1967). "Analytic Description of the Low-Frequency
    Attenuation Coefficient". JASA, 42(1), 270.

    Examples
    --------
    >>> alpha = thorp_attenuation(1000.0)  # 1 kHz
    >>> print(f"Attenuation: {alpha:.4f} dB/km")
    """
    f = np.atleast_1d(frequency) / 1000.0  # Convert to kHz

    # Thorp formula
    f2 = f**2
    alpha = 0.1 * f2 / (1 + f2) + 40 * f2 / (4100 + f2) + 2.75e-4 * f2 + 0.003

    return np.squeeze(alpha)


def francois_garrison(
    frequency: Union[float, np.ndarray],
    temperature: float = 10.0,
    salinity: float = 35.0,
    pH: float = 8.0,
    depth: float = 1000.0
) -> np.ndarray:
    """
    Francois-Garrison attenuation formula (more accurate than Thorp)

    Parameters
    ----------
    frequency : float or array
        Frequency in Hz
    temperature : float, optional
        Temperature in °C. Default is 10.0.
    salinity : float, optional
        Salinity in ppt. Default is 35.0.
    pH : float, optional
        pH value. Default is 8.0.
    depth : float, optional
        Depth in meters. Default is 1000.0.

    Returns
    -------
    alpha : float or array
        Attenuation in dB/km

    References
    ----------
    Francois, R. E. and Garrison, G. R. (1982). "Sound absorption based
    on ocean measurements. Part II: Boric acid contribution and equation
    for total absorption". JASA, 72(6), 1879-1890.

    Notes
    -----
    Implementation based on Acoustics Toolbox AttenMod.f90 by M. Porter.
    Verified using Francois-Garrison Table IV.

    Examples
    --------
    >>> alpha = francois_garrison(1000.0, temperature=15, salinity=35)
    >>> print(f"Attenuation: {alpha:.4f} dB/km")
    """
    f = np.atleast_1d(frequency) / 1000.0  # Convert Hz to kHz

    T = temperature  # °C
    S = salinity  # ppt
    z_bar = depth  # meters

    # Sound speed (Francois-Garrison formula)
    c = 1412.0 + 3.21 * T + 1.19 * S + 0.0167 * z_bar

    # Boric acid contribution
    A1 = 8.86 / c * 10.0 ** (0.78 * pH - 5.0)
    P1 = 1.0
    f1 = 2.8 * np.sqrt(S / 35.0) * 10.0 ** (4.0 - 1245.0 / (T + 273.0))

    # Magnesium sulfate contribution
    A2 = 21.44 * S / c * (1.0 + 0.025 * T)
    P2 = 1.0 - 1.37e-4 * z_bar + 6.2e-9 * z_bar**2
    f2 = 8.17 * 10.0 ** (8.0 - 1990.0 / (T + 273.0)) / (1.0 + 0.0018 * (S - 35.0))

    # Pure water viscosity contribution
    P3 = 1.0 - 3.83e-5 * z_bar + 4.9e-10 * z_bar**2

    # Temperature-dependent coefficients for viscosity
    if np.isscalar(T):
        if T < 20:
            A3 = 4.937e-4 - 2.59e-5 * T + 9.11e-7 * T**2 - 1.5e-8 * T**3
        else:
            A3 = 3.964e-4 - 1.146e-5 * T + 1.45e-7 * T**2 - 6.5e-10 * T**3
    else:
        # Handle array input
        A3 = np.where(T < 20,
                      4.937e-4 - 2.59e-5 * T + 9.11e-7 * T**2 - 1.5e-8 * T**3,
                      3.964e-4 - 1.146e-5 * T + 1.45e-7 * T**2 - 6.5e-10 * T**3)

    # Total attenuation (dB/km)
    # Each term: A * P * (f_relax * f^2) / (f_relax^2 + f^2)
    alpha = (A1 * P1 * (f1 * f**2) / (f1**2 + f**2) +
             A2 * P2 * (f2 * f**2) / (f2**2 + f**2) +
             A3 * P3 * f**2)

    return np.squeeze(alpha)


def fisher_simmons(
    frequency: Union[float, np.ndarray],
    depth: float = 100.0
) -> np.ndarray:
    """
    Fisher-Simmons attenuation (simplified shallow water formula)

    Parameters
    ----------
    frequency : float or array
        Frequency in Hz
    depth : float, optional
        Water depth in meters. Default is 100.0.

    Returns
    -------
    alpha : float or array
        Attenuation in dB/km

    References
    ----------
    Fisher, F. H. and Simmons, V. P. (1977). "Sound absorption in sea water".
    JASA, 62(3), 558-564.
    """
    f = np.atleast_1d(frequency) / 1000.0  # kHz

    # Simplified formula for shallow water
    alpha = 0.11 * f**2 / (1.0 + f**2) + 44.0 * f**2 / (4100.0 + f**2) + 2.75e-4 * f**2 + 0.003

    # Depth correction
    depth_factor = 1.0 + depth / 5000.0
    alpha *= depth_factor

    return np.squeeze(alpha)


def ainslie_mccolm(
    frequency: Union[float, np.ndarray],
    temperature: float = 10.0,
    salinity: float = 35.0,
    pH: float = 8.0,
    depth: float = 1000.0
) -> np.ndarray:
    """
    Ainslie & McColm simplified formula

    Parameters
    ----------
    frequency : float or array
        Frequency in Hz
    temperature : float, optional
        Temperature in °C. Default is 10.0.
    salinity : float, optional
        Salinity in ppt. Default is 35.0.
    pH : float, optional
        pH value. Default is 8.0.
    depth : float, optional
        Depth in meters. Default is 1000.0.

    Returns
    -------
    alpha : float or array
        Attenuation in dB/km
    """
    f = np.atleast_1d(frequency) / 1000.0  # kHz

    # Simplified combination of Thorp-like formula with temperature dependence
    T = temperature

    # Relaxation frequency (temperature dependent)
    f_relax = 21.0 * np.exp(T / 15.0)

    # Attenuation
    alpha = 0.1 * f**2 / (1.0 + f**2) + 40.0 * f**2 / (f_relax**2 + f**2) + 2.5e-4 * f**2

    # Salinity correction
    alpha *= (1.0 + 0.01 * (salinity - 35.0))

    # Depth correction
    depth_km = depth / 1000.0
    alpha *= (1.0 + 0.05 * depth_km)

    return np.squeeze(alpha)


class AttenuationModel:
    """
    Attenuation model handler

    Provides unified interface to various attenuation models.

    Parameters
    ----------
    model_type : str
        Attenuation model: 'thorp', 'francois-garrison', 'fisher-simmons',
        'ainslie-mccolm', or 'custom'
    params : AttenuationParameters, optional
        Parameters for the model
    custom_function : callable, optional
        Custom attenuation function(frequency) -> attenuation_db_per_km

    Examples
    --------
    >>> model = AttenuationModel('francois-garrison',
    ...                           params=AttenuationParameters(temperature=15))
    >>> alpha = model.calculate(1000.0)  # At 1 kHz
    """

    def __init__(
        self,
        model_type: str = 'thorp',
        params: Optional[AttenuationParameters] = None,
        custom_function: Optional[callable] = None
    ):
        self.model_type = model_type.lower()
        self.params = params if params is not None else AttenuationParameters()
        self.custom_function = custom_function

        valid_models = ['thorp', 'francois-garrison', 'fisher-simmons',
                       'ainslie-mccolm', 'custom']
        if self.model_type not in valid_models:
            raise ValueError(f"model_type must be one of {valid_models}")

        if self.model_type == 'custom' and custom_function is None:
            raise ValueError("custom_function required for model_type='custom'")

    def calculate(self, frequency: Union[float, np.ndarray]) -> np.ndarray:
        """
        Calculate attenuation at given frequency/frequencies

        Parameters
        ----------
        frequency : float or array
            Frequency in Hz

        Returns
        -------
        alpha : float or array
            Attenuation in dB/km
        """
        if self.model_type == 'thorp':
            return thorp_attenuation(frequency)

        elif self.model_type == 'francois-garrison':
            return francois_garrison(
                frequency,
                temperature=self.params.temperature,
                salinity=self.params.salinity,
                pH=self.params.pH,
                depth=self.params.depth
            )

        elif self.model_type == 'fisher-simmons':
            return fisher_simmons(frequency, depth=self.params.depth)

        elif self.model_type == 'ainslie-mccolm':
            return ainslie_mccolm(
                frequency,
                temperature=self.params.temperature,
                salinity=self.params.salinity,
                pH=self.params.pH,
                depth=self.params.depth
            )

        elif self.model_type == 'custom':
            return self.custom_function(frequency)

    def __repr__(self) -> str:
        return f"AttenuationModel(type='{self.model_type}')"


def convert_attenuation_units(
    alpha: Union[float, np.ndarray],
    frequency: float,
    from_unit: str,
    to_unit: str,
    sound_speed: float = 1500.0
) -> np.ndarray:
    """
    Convert attenuation between different units

    Parameters
    ----------
    alpha : float or array
        Attenuation value
    frequency : float
        Frequency in Hz
    from_unit : str
        Source unit: 'dB/km', 'dB/m', 'dB/wavelength', 'Nepers/m', 'Q', 'L'
    to_unit : str
        Target unit (same options as from_unit)
    sound_speed : float, optional
        Sound speed in m/s. Default is 1500.0.

    Returns
    -------
    alpha_converted : float or array
        Converted attenuation

    Examples
    --------
    >>> # Convert dB/km to dB/m
    >>> alpha_m = convert_attenuation_units(0.1, 1000, 'dB/km', 'dB/m')
    """
    alpha = np.atleast_1d(alpha)

    # First convert to dB/m
    if from_unit == 'dB/km':
        alpha_db_m = alpha / 1000.0
    elif from_unit == 'dB/m':
        alpha_db_m = alpha
    elif from_unit == 'dB/wavelength':
        wavelength = sound_speed / frequency
        alpha_db_m = alpha / wavelength
    elif from_unit == 'Nepers/m':
        alpha_db_m = alpha * 8.686  # 1 Neper = 8.686 dB
    elif from_unit == 'Q':  # Quality factor
        # Q = 2π/α where α is in Nepers/wavelength
        # Convert to dB/m
        alpha_nepers_m = 2 * np.pi * frequency / (alpha * sound_speed)
        alpha_db_m = alpha_nepers_m * 8.686
    elif from_unit == 'L':  # Loss parameter (loss tangent)
        # L = α * λ / π where α is in Nepers/m
        alpha_nepers_m = alpha * np.pi * frequency / sound_speed
        alpha_db_m = alpha_nepers_m * 8.686
    else:
        raise ValueError(f"Unknown unit: {from_unit}")

    # Now convert from dB/m to target unit
    if to_unit == 'dB/km':
        result = alpha_db_m * 1000.0
    elif to_unit == 'dB/m':
        result = alpha_db_m
    elif to_unit == 'dB/wavelength':
        wavelength = sound_speed / frequency
        result = alpha_db_m * wavelength
    elif to_unit == 'Nepers/m':
        result = alpha_db_m / 8.686
    elif to_unit == 'Q':
        alpha_nepers_m = alpha_db_m / 8.686
        result = 2 * np.pi * frequency / (alpha_nepers_m * sound_speed)
    elif to_unit == 'L':
        alpha_nepers_m = alpha_db_m / 8.686
        result = alpha_nepers_m * sound_speed / (np.pi * frequency)
    else:
        raise ValueError(f"Unknown unit: {to_unit}")

    return np.squeeze(result)
