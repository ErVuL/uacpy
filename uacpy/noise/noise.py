"""
Comprehensive Ambient Noise Model (Wenz-based Framework + Knudsen 1948) Extended with Biological,
Seismic, and Explosion Sources.

Implements a modular composite model for underwater ambient noise, combining multiple empirical
submodels (wind, shipping, rain, turbulence, thermal, biological, seismic, explosion) in a
physically consistent manner.

Key Features:
- All noise sources treated as models with consistent interfaces
- Multiple instances of the same model type can be used simultaneously
- Well-known explosion models included
- Comprehensive reference system

References:
    - Wenz, G. M. (1962). Acoustic ambient noise in the ocean: spectra and sources. JASA, 34(12), 1936–1956.
    - Knudsen, V. O., Alford, R. S., & Emling, J. W. (1948). Underwater ambient noise. JASA, 20(2), 188–195.
    - Mellen, R. H. (1987). Thermal noise in the ocean.
    - Hasselmann, K. (1963). A statistical analysis of the generation of microseisms. Reviews of Geophysics, 1(2), 177–210.
    - Webb, S. C. (1998). Ocean microseisms. Science, 281(5384), 198–200.
    - Longuet-Higgins, M. S. (1950). A theory of the origin of microseisms. Philosophical Transactions A, 243(857), 1–35.
    - Duennebier, F. K., & Sutton, G. H. (1977). Measurements of microseisms in the North Pacific. JGR, 82(5), 717–732.
    - Cato, D. H. (1978). Acoustic characteristics of snapping shrimp. JASA, 64(5), 1522–1528.
    - Ma, G., & Nystuen, J. A. (2005). Rain-induced underwater noise. JASA, 117(6), 3617–3628.
    - Chapman, N. R. (1988). Source levels of shallow explosive charges. JASA, 84(2), 697–702.
    - McCauley, R. D. (2012). Fish choruses in the coastal waters of Australia: Links to ocean temperature and their potential use for ecosystem monitoring.
    - Parsons, M. J. G., et al. (2016). Fish choruses off Port Hedland, Western Australia.
    - Wenz, G. M. (1962). Biological sources of ambient noise (marine mammals).
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Callable, Optional, List, Union, Any
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for an individual noise model contribution.

    Attributes
    ----------
    name : str
        Model name (e.g., 'knudsen', 'wenz').
    model_type : str
        Category of noise model (e.g., 'wind', 'shipping', 'thermal').
    function : callable
        The noise model function to call.
    parameters : dict
        Parameters passed to the model function.
    label : str or None
        Display label for plotting. Auto-generated from type and name if None.
    """

    name: str
    model_type: str
    function: Callable
    parameters: Dict[str, Any]
    label: Optional[str] = None

    def __post_init__(self):
        if self.label is None:
            self.label = f"{self.model_type}_{self.name}"


# ============================================================
# Helper: Map Wind Speed to Sea State (Urick, 1984)
# ============================================================
def wind_speed_to_sea_state(wind_speed: float) -> float:
    """Map wind speed in m/s to sea state (Beaufort scale).

    Reference: Urick, R. J. (1984). Ambient Noise in the Sea.
    """
    if wind_speed < 0.3:
        return 0
    elif wind_speed < 1.6:
        return 0.5
    elif wind_speed < 3.4:
        return 1
    elif wind_speed < 5.5:
        return 2
    elif wind_speed < 8.0:
        return 3
    elif wind_speed < 10.8:
        return 4
    elif wind_speed < 13.9:
        return 5
    else:
        return 6


# ============================================================
# Thermal Noise (Mellen 1987)
# ============================================================
def thermal_noise_mellen(freq: np.ndarray, **kwargs) -> np.ndarray:
    """Thermal noise component (dB re 1 µPa²/Hz).

    Reference: Mellen, R. H. (1987). Thermal noise in the ocean.
    """
    return -75.0 + 20.0 * np.log10(freq)


# ============================================================
# Wind Noise Models
# ============================================================
def wind_noise_knudsen(freq: np.ndarray, wind_speed: float = 5, **kwargs) -> np.ndarray:
    """Knudsen et al. (1948) wind noise using Urick (1984) sea state mapping.

    Reference:
        - Knudsen et al. (1948). Underwater ambient noise.
        - Urick, R. J. (1984). Ambient Noise in the Sea.
    """
    f_min, f_max = 100, 25000
    sea_state = wind_speed_to_sea_state(wind_speed)

    # Knudsen amplitude table
    A_lookup = {0: 30, 0.5: 35, 1: 40, 2: 50, 3: 60, 4: 70, 5: 80, 6: 90}
    A = A_lookup.get(sea_state, 60)

    Lw = A - 17 * np.log10(freq / 1000)
    Lw = np.where((freq >= f_min) & (freq <= f_max), Lw, 0)
    return Lw


def wind_noise_piggot_merklinger(
    freq: np.ndarray, wind_speed: float = 5, water_depth: str = "deep", **kwargs
) -> np.ndarray:
    """Piggot & Merklinger formulation of Wenz (1962) wind noise.

    Reference: Wenz, G. M. (1962). Acoustic ambient noise in the ocean.
    """
    if wind_speed <= 0:
        return np.zeros_like(freq)
    f_wind = 2000
    s1w, s2w, a = 1.5, -5.0, -25
    cst = 45 if water_depth == "shallow" else 42
    f0w = 770 - 100 * np.log10(wind_speed)
    L0w = cst + 20 * np.log10(wind_speed) - 17 * np.log10(f0w / 770)
    i_wind = freq <= f_wind
    f_temp = freq[i_wind] if np.any(i_wind) else np.array([2000])
    L1w = L0w + (s1w / np.log10(2)) * np.log10(f_temp / f0w)
    L2w = L0w + (s2w / np.log10(2)) * np.log10(f_temp / f0w)
    Lw = L1w * (1 + (L1w / L2w) ** (-a)) ** (1 / a)
    NL = np.zeros_like(freq)
    NL[i_wind] = 10 ** (Lw / 10)
    slope = s2w * (0.1 / np.log10(2))
    if np.any(~i_wind):
        prop_const = NL[i_wind][-1] / f_temp[-1] ** slope
        NL[~i_wind] = prop_const * freq[~i_wind] ** slope
    return 10 * np.log10(NL)


def wind_noise_wilson(freq: np.ndarray, wind_speed: float = 5, **kwargs) -> np.ndarray:
    """Wilson wind noise model.

    Reference: Wilson, J. H. (1979). Very low frequency wind-generated noise.
    """
    Lw = 50 + 7.5 * np.sqrt(wind_speed) + 20 * np.log10(freq)
    Lw = np.where((freq >= 5) & (freq <= 50), Lw, 0)
    return np.maximum(Lw, 0)


def wind_noise_kewley(freq: np.ndarray, wind_speed: float = 5, **kwargs) -> np.ndarray:
    """Kewley et al. (1990) empirical wind noise model.

    Reference: Kewley, D. J., Browning, D. G., & Carey, W. M. (1990).
    """
    Lw = 44 + 21 * np.log10(wind_speed + 1e-3) - 17 * np.log10(freq / 1000)
    return np.maximum(Lw, 0)


def wind_noise_gsm(freq: np.ndarray, wind_speed: float = 5, **kwargs) -> np.ndarray:
    """Generic Shipping Model (GSM) wind noise component."""
    Lw = 50 + 10 * np.log10(wind_speed + 1e-3) - 18 * np.log10(freq / 1000)
    return np.maximum(Lw, 0)


# ============================================================
# Shipping Noise Models
# ============================================================
def shipping_noise_wenz(
    freq: np.ndarray,
    water_depth: str = "deep",
    shipping_level: str = "medium",
    **kwargs,
) -> np.ndarray:
    """Wenz shipping noise model.

    Reference: Wenz, G. M. (1962). Acoustic ambient noise in the ocean.
    """
    c1 = 30 if water_depth == "deep" else 65
    c2 = {"low": 1, "medium": 4, "high": 7}.get(shipping_level, 4)
    noise = 76 - 20 * (np.log10(freq / c1)) ** 2 + 5 * (c2 - 4)
    return np.maximum(noise, 1)


def shipping_noise_knudsen(
    freq: np.ndarray, shipping_activity: str = "medium", **kwargs
) -> np.ndarray:
    """Knudsen shipping noise model.

    Reference: Knudsen et al. (1948). Underwater ambient noise.
    """
    f_min, f_max = 10, 1000
    A_lookup = {"low": 60, "medium": 70, "high": 80}
    A = A_lookup.get(shipping_activity, 70)
    Ls = A - 20 * np.log10(freq / 1000)
    Ls = np.where((freq >= f_min) & (freq <= f_max), Ls, 0)
    return Ls


# ============================================================
# Rain Noise (Ma & Nystuen 2005)
# ============================================================
def rain_noise_ma_nystuen(
    freq: np.ndarray, rain_rate: str = "no", **kwargs
) -> np.ndarray:
    """Rain-induced underwater noise model.

    Reference: Ma, G., & Nystuen, J. A. (2005). Rain-induced underwater noise.
    """
    if rain_rate == "no":
        return np.zeros_like(freq)
    r0 = [0, 51.1, 61.5, 65.1, 74.3]
    r1 = [0, 1.47, 1.01, 0.82, 1.01]
    r2 = [0, -0.52, -0.43, -0.38, -0.43]
    r3 = [0, 0.033, 0.028, 0.025, 0.028]
    idx = {"light": 1, "moderate": 2, "heavy": 3, "veryheavy": 4}.get(rain_rate, 1)
    fk = freq / 1000
    noise = r0[idx] + r1[idx] * fk + r2[idx] * fk**2 + r3[idx] * fk**3
    slope = -5.0 * (0.1 / np.log10(2))
    cutoff = np.where(freq < 7000)[0][-1]
    temp_val = 10 ** (noise[cutoff] / 10)
    prop_const = temp_val / freq[cutoff] ** slope
    noise[freq > 7000] = 10 * np.log10(prop_const * freq[freq > 7000] ** slope)
    return noise


# ============================================================
# Turbulence Noise (Urick 1984)
# ============================================================
def turbulence_noise_urick(freq: np.ndarray, **kwargs) -> np.ndarray:
    """Turbulence-induced noise model.

    Reference: Urick, R. J. (1984). Ambient Noise in the Sea.
    """
    return np.maximum(108.5 - 32.5 * np.log10(freq), 1)


# ============================================================
# Biological Noise
# ============================================================
def biological_noise_snapping_shrimp_cato(
    freq: np.ndarray, shrimp_activity: str = "moderate", **kwargs
) -> np.ndarray:
    """Snapping shrimp biological noise model.

    Reference: Cato, D. H. (1978). Acoustic characteristics of snapping shrimp.
    """
    if shrimp_activity == "none":
        return np.zeros_like(freq)
    A_lookup = {"low": 50, "moderate": 60, "high": 70}
    A = A_lookup.get(shrimp_activity, 60)
    noise = A - 8 * np.log10(freq / 1000)
    noise = np.where((freq >= 1000) & (freq <= 100000), noise, 0)
    return np.maximum(noise, 0)


def biological_noise_marine_mammals_wenz(
    freq: np.ndarray, mammal_activity: str = "low", **kwargs
) -> np.ndarray:
    """Marine mammal vocalization noise model.

    Reference: Wenz, G. M. (1962). Acoustic ambient noise in the ocean: spectra and sources.
              Biological sources of ambient noise (marine mammals).

    Frequency range: 10 Hz - 10 kHz (whales: 10-1000 Hz, dolphins: 1-10 kHz)
    """
    if mammal_activity == "none":
        return np.zeros_like(freq)
    A_lookup = {"low": 65, "moderate": 75, "high": 85}
    A = A_lookup.get(mammal_activity, 65)
    f_center = 300
    bandwidth = 2.0
    noise = A - 10 * (np.log10(freq / f_center) / bandwidth) ** 2
    noise = np.where((freq >= 10) & (freq <= 10000), noise, 0)
    return np.maximum(noise, 0)


def biological_noise_fish_chorus_mccauley_parsons(
    freq: np.ndarray, fish_activity: str = "moderate", **kwargs
) -> np.ndarray:
    """Fish chorus biological noise model.

    References:
        - McCauley, R. D. (2012). Fish choruses in the coastal waters of Australia.
        - Parsons, M. J. G., et al. (2016). Fish choruses off Port Hedland, Western Australia.

    Frequency range: 50-2000 Hz, typically 100-500 Hz peak
    """
    if fish_activity == "none":
        return np.zeros_like(freq)
    A_lookup = {"low": 70, "moderate": 80, "high": 90}
    A = A_lookup.get(fish_activity, 80)
    f_center = 300
    bandwidth = 1.5
    noise = A - 15 * (np.log10(freq / f_center) / bandwidth) ** 2
    noise = np.where((freq >= 50) & (freq <= 2000), noise, 0)
    return np.maximum(noise, 0)


# ============================================================
# Seismic Noise
# ============================================================
def primary_microseism_longuet_higgins(
    freq: np.ndarray, level: str = "moderate", **kwargs
) -> np.ndarray:
    """Primary microseism noise component.

    Reference: Longuet-Higgins, M. S. (1950). A theory of the origin of microseisms.
    """
    A_lookup = {"low": 70, "moderate": 80, "high": 90}
    A = A_lookup.get(level, 80)
    L = np.zeros_like(freq)
    valid = (freq >= 0.05) & (freq <= 0.1)
    L[valid] = A
    return L


def secondary_microseism_hasselmann(
    freq: np.ndarray, level: str = "moderate", **kwargs
) -> np.ndarray:
    """Secondary microseism noise component.

    Reference: Hasselmann, K. (1963). A statistical analysis of the generation of microseisms.
    """
    A_lookup = {"low": 80, "moderate": 90, "high": 100}
    A = A_lookup.get(level, 90)
    f0 = 0.2
    bandwidth = 0.5
    L = A - 20 * ((np.log10(freq / f0) / bandwidth) ** 2)
    L = np.where((freq >= 0.1) & (freq <= 0.3), L, 0)
    return L


def high_frequency_microseism_webb(
    freq: np.ndarray, level: str = "moderate", **kwargs
) -> np.ndarray:
    """High-frequency microseism noise component.

    Reference: Webb, S. C. (1998). Ocean microseisms.
    """
    A_lookup = {"low": 70, "moderate": 80, "high": 90}
    A = A_lookup.get(level, 80)
    f0 = 0.3
    slope = 20
    L = A - slope * np.log10(freq / f0)
    L = np.where((freq >= 0.3) & (freq <= 10), L, 0)
    return L


def seismic_noise_composite(
    freq: np.ndarray, level: str = "moderate", **kwargs
) -> np.ndarray:
    """Composite seismic noise model combining primary, secondary, and HF microseisms.

    Sums contributions from ``primary_microseism_longuet_higgins``,
    ``secondary_microseism_hasselmann``, and ``high_frequency_microseism_webb``
    incoherently.
    """
    L_primary = 10 ** (primary_microseism_longuet_higgins(freq, level) / 10)
    L_secondary = 10 ** (secondary_microseism_hasselmann(freq, level) / 10)
    L_high = 10 ** (high_frequency_microseism_webb(freq, level) / 10)
    total = 10 * np.log10(L_primary + L_secondary + L_high)
    return total


# ============================================================
# Explosion Noise Models
# ============================================================
def explosion_noise_chapman(
    freq: np.ndarray, charge_weight: float = 1.0, distance: float = 1000.0, **kwargs
) -> np.ndarray:
    """Chapman explosion noise model for shallow explosive charges.

    Reference: Chapman, N. R. (1988). Source levels of shallow explosive charges.

    Parameters
    ----------
    freq : ndarray
        Frequency array in Hz.
    charge_weight : float
        Explosive charge weight in kg (TNT equivalent).
    distance : float
        Distance from explosion in meters.
    """
    # Reference level for 1 kg TNT at 1 m
    SL_ref = 262  # dB re 1 µPa²/Hz at 1 m for 1 kg TNT

    # Spreading and absorption
    spreading_loss = 20 * np.log10(distance)
    absorption = 0.036 * freq**0.5 * distance / 1000  # Approximate absorption in dB

    # Frequency dependence - explosion spectrum typically decays with frequency
    freq_dependence = -10 * np.log10(1 + (freq / 100) ** 2)

    # Weight scaling (logarithmic)
    weight_scaling = 10 * np.log10(charge_weight)

    # Total spectrum
    spectrum = SL_ref - spreading_loss - absorption + freq_dependence + weight_scaling

    return np.maximum(spectrum, 0)


def explosion_noise_arons(
    freq: np.ndarray, charge_weight: float = 1.0, distance: float = 1000.0, **kwargs
) -> np.ndarray:
    """Arons explosion noise model for underwater explosions.

    Reference: Arons, A. B. (1954). Underwater explosion shock wave parameters.

    Parameters
    ----------
    freq : ndarray
        Frequency array in Hz.
    charge_weight : float
        Explosive charge weight in kg (TNT equivalent).
    distance : float
        Distance from explosion in meters.
    """
    # Reference parameters
    P0 = 5.2e7  # Peak pressure for 1 kg TNT at 1 m in Pa
    theta = 92e-6  # Time constant for 1 kg TNT in seconds

    # Scaling with weight and distance
    scaled_theta = theta * charge_weight ** (1 / 3)
    scaled_distance = distance / charge_weight ** (1 / 3)

    # Pressure spectrum (Fourier transform of exponential decay)
    omega = 2 * np.pi * freq
    pressure_spectrum = P0 * scaled_theta / (1 + (omega * scaled_theta) ** 2) ** 0.5

    # Convert to dB re 1 µPa²/Hz
    spectrum_db = 20 * np.log10(pressure_spectrum / 1e-6)

    # Apply spherical spreading
    spreading_loss = 20 * np.log10(scaled_distance)
    spectrum_db -= spreading_loss

    return np.maximum(spectrum_db, 0)


def explosion_noise_broadband_generic(
    freq: np.ndarray,
    source_level: float = 180,
    center_freq: float = 100,
    bandwidth: float = 2.0,
    **kwargs,
) -> np.ndarray:
    """Simple broadband explosion noise model with Gaussian spectrum.

    Parameters:
        source_level: Peak source level in dB re 1 µPa²/Hz
        center_freq: Center frequency of the explosion spectrum in Hz
        bandwidth: Bandwidth in octaves
    """
    # Gaussian spectrum in log-frequency domain
    spectrum = source_level - 20 * (np.log2(freq / center_freq) / bandwidth) ** 2
    return np.maximum(spectrum, 0)


# ============================================================
# Comprehensive Model Registry
# ============================================================
MODEL_REGISTRY: Dict[str, Dict[str, Callable]] = {
    "wind": {
        "piggot_merklinger": wind_noise_piggot_merklinger,
        "wilson": wind_noise_wilson,
        "kewley": wind_noise_kewley,
        "gsm": wind_noise_gsm,
        "knudsen": wind_noise_knudsen,
    },
    "shipping": {
        "wenz": shipping_noise_wenz,
        "knudsen": shipping_noise_knudsen,
    },
    "rain": {
        "ma_nystuen": rain_noise_ma_nystuen,
    },
    "turbulence": {
        "urick": turbulence_noise_urick,
    },
    "thermal": {
        "mellen": thermal_noise_mellen,
    },
    "biological": {
        "snapping_shrimp_cato": biological_noise_snapping_shrimp_cato,
        "marine_mammals_wenz": biological_noise_marine_mammals_wenz,
        "fish_chorus_mccauley_parsons": biological_noise_fish_chorus_mccauley_parsons,
    },
    "seismic": {
        "composite": seismic_noise_composite,
        "primary_microseism_longuet_higgins": primary_microseism_longuet_higgins,
        "secondary_microseism_hasselmann": secondary_microseism_hasselmann,
        "high_frequency_microseism_webb": high_frequency_microseism_webb,
    },
    "explosion": {
        "chapman": explosion_noise_chapman,
        "arons": explosion_noise_arons,
        "broadband_generic": explosion_noise_broadband_generic,
    },
}


class AmbientNoiseSimulator:
    """Builder for simulating composite ambient noise from multiple models.

    Provides a fluent interface for adding noise source contributions
    (wind, shipping, thermal, biological, seismic, etc.) and computing
    or plotting the resulting composite spectrum.

    Parameters
    ----------
    freq : ndarray, optional
        Frequency array in Hz. Defaults to 1000 points log-spaced
        from 1 Hz to 100 kHz.

    Examples
    --------
    >>> sim = AmbientNoiseSimulator()
    >>> sim.add_wind('knudsen', wind_speed=10)
    >>> sim.add_shipping('wenz', shipping_level=5)
    >>> sim.add_thermal('mellen')
    >>> components, total = sim.compute()
    """

    def __init__(self, freq: Optional[np.ndarray] = None):
        if freq is None:
            freq = np.logspace(0, 5, 1000)
        self.freq = freq
        self.model_configs: List[ModelConfig] = []
        self.global_params: Dict[str, Any] = {}

    def add_model(
        self,
        model_type: str,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        label: Optional[str] = None,
    ):
        if parameters is None:
            parameters = {}
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}")
        if name not in MODEL_REGISTRY[model_type]:
            raise ValueError(f"Unknown model name '{name}' in type '{model_type}'")
        function = MODEL_REGISTRY[model_type][name]
        config = ModelConfig(name, model_type, function, parameters, label)
        self.model_configs.append(config)

    def add_wind(
        self,
        name: str,
        label: Optional[str] = None,
        **parameters,
    ):
        self.add_model("wind", name, parameters, label)

    def add_shipping(
        self,
        name: str,
        label: Optional[str] = None,
        **parameters,
    ):
        self.add_model("shipping", name, parameters, label)

    def add_rain(
        self,
        name: str,
        label: Optional[str] = None,
        **parameters,
    ):
        self.add_model("rain", name, parameters, label)

    def add_turbulence(
        self,
        name: str,
        label: Optional[str] = None,
        **parameters,
    ):
        self.add_model("turbulence", name, parameters, label)

    def add_thermal(
        self,
        name: str,
        label: Optional[str] = None,
        **parameters,
    ):
        self.add_model("thermal", name, parameters, label)

    def add_biological(
        self,
        name: str,
        label: Optional[str] = None,
        **parameters,
    ):
        self.add_model("biological", name, parameters, label)

    def add_seismic(
        self,
        name: str,
        label: Optional[str] = None,
        **parameters,
    ):
        self.add_model("seismic", name, parameters, label)

    def add_explosion(
        self,
        name: str,
        label: Optional[str] = None,
        **parameters,
    ):
        self.add_model("explosion", name, parameters, label)

    def set_global_params(self, **params):
        """Set parameters applied to all models (e.g., wind_speed)."""
        self.global_params.update(params)

    def compute(self) -> tuple[Dict[str, np.ndarray], np.ndarray]:
        """Compute noise spectra for all added models.

        Returns
        -------
        components : dict
            Per-model spectral levels keyed by model label.
        total : ndarray
            Incoherent sum of all components in dB re 1 uPa^2/Hz.
        """
        components, total, _ = compute_ambient_noise(
            self.freq, self.model_configs, **self.global_params
        )
        return components, total

    def plot(
        self,
        title: str = "Ambient Noise Spectrum",
        show_total: bool = True,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot composite ambient noise spectrum.

        Parameters
        ----------
        title : str
            Figure title.
        show_total : bool
            Whether to overlay the total composite level.

        Returns
        -------
        fig : Figure
            Matplotlib figure.
        ax : Axes
            Matplotlib axes.
        """
        return plot_ambient_noise(
            self.freq, self.model_configs, title, show_total, **self.global_params
        )


# ============================================================
# Composite Ambient Noise Model
# ============================================================
def compute_ambient_noise(
    freq: Optional[np.ndarray] = None,
    model_configs: Optional[List[ModelConfig]] = None,
    **params,
) -> tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Compute ambient noise using multiple configurable models.

    Parameters
    ----------
    freq : ndarray or None
        Frequency array in Hz. Defaults to 1000 points log-spaced 1 Hz - 100 kHz.
    model_configs : list of ModelConfig or None
        List of model configurations. Defaults to a standard set of models.
    **params
        Additional parameters passed to model functions (e.g., wind_speed).

    Returns
    -------
    components : dict
        Dictionary of noise component spectra keyed by model label.
    total : ndarray
        Total (incoherent sum) noise spectrum in dB re 1 uPa^2/Hz.
    freq : ndarray
        Frequency array used.
    """
    if freq is None:
        freq = np.logspace(0, 5, 1000)
    else:
        freq = np.asarray(freq)

    if model_configs is None:
        # Default configuration for backward compatibility
        model_configs = [
            ModelConfig("mellen", "thermal", thermal_noise_mellen, {}),
            ModelConfig("urick", "turbulence", turbulence_noise_urick, {}),
            ModelConfig(
                "piggot_merklinger",
                "wind",
                wind_noise_piggot_merklinger,
                {"wind_speed": params.get("wind_speed", 5)},
            ),
            ModelConfig(
                "wenz",
                "shipping",
                shipping_noise_wenz,
                {"shipping_level": params.get("shipping_level", "medium")},
            ),
        ]

    components = {}

    # Compute each model component
    for config in model_configs:
        # Merge default parameters with config-specific parameters
        all_params = {**config.parameters, **params}
        components[config.label] = config.function(freq, **all_params)

    # Compute total noise (power summation)
    if components:
        total = 10 * np.log10(sum(10 ** (v / 10) for v in components.values()))
    else:
        total = np.zeros_like(freq)

    return components, total, freq


# ============================================================
# Plotting Function
# ============================================================
def plot_ambient_noise(
    freq: Optional[np.ndarray] = None,
    model_configs: Optional[List[ModelConfig]] = None,
    title: str = "Ambient Noise Spectrum",
    show_total: bool = True,
    show_txtbox: bool = True,
    **params,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot ambient noise spectrum using configurable models.

    Parameters
    ----------
    freq : ndarray or None
        Frequency array in Hz.
    model_configs : list of ModelConfig or None
        List of model configurations.
    title : str
        Plot title.
    show_total : bool
        Whether to overlay the total composite level.
    show_txtbox : bool
        Whether to show model parameters text box.
    **params
        Additional parameters passed to model functions.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.
    """
    components, total, freq = compute_ambient_noise(freq, model_configs, **params)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot individual components
    for label, spec in components.items():
        ax.semilogx(freq, spec, "--", lw=1.5, label=label, alpha=0.8)

    # Plot total noise
    if show_total:
        ax.semilogx(freq, total, "k-", lw=3, label="Total Noise", alpha=0.9)

    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("Frequency [Hz]", fontsize=12)
    ax.set_ylabel("Noise Level [dB re 1 µPa²/Hz]", fontsize=12)
    ax.grid(True, which="both", alpha=0.4)
    ax.set_xlim((freq[0], freq[-1]))
    ax.set_ylim((20, 220))

    # Enhanced legend
    ax.legend(
        loc="upper right", frameon=True, fancybox=True, shadow=True, fontsize=10, ncol=2
    )

    # Settings text box
    if show_txtbox:
        settings_text = "Model Configuration\n============\n\n"
        for config in model_configs or []:
            settings_text += f"* {config.model_type}, {config.name}\n"
            for k, v in sorted(config.parameters.items()):
                settings_text += f"      - {k}: {v}\n"
            settings_text += "\n"
        if params:
            settings_text += "\nGlobal Parameters:\n"
            for k, v in sorted(params.items()):
                settings_text += f"      - {k}: {v}\n"
            settings_text += "\n"

        ax.text(
            0.02,
            0.98,
            settings_text[:-2],
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
        )

    fig.tight_layout()
    return fig, ax


# ============================================================
# Example Usage and Demonstration
# ============================================================
if __name__ == "__main__":
    # Example 1: Multiple wind models comparison
    sim = AmbientNoiseSimulator()
    sim.add_wind(
        "piggot_merklinger", wind_speed=10, water_depth="deep", label="Wind_PM"
    )
    sim.add_wind("wilson", wind_speed=10, label="Wind_Wilson")
    sim.add_wind("knudsen", wind_speed=10, label="Wind_Knudsen")

    fig, ax = sim.plot(
        title="Comparison of Wind Noise Models (Wind Speed = 10 m/s)", show_total=False
    )
    plt.savefig("wind_models_comparison.png", dpi=300, bbox_inches="tight")

    # Example 2: Multiple biological models
    sim = AmbientNoiseSimulator()
    sim.add_biological(
        "snapping_shrimp_cato", shrimp_activity="high", label="Snapping_Shrimp_Cato"
    )
    sim.add_biological(
        "marine_mammals_wenz", mammal_activity="moderate", label="Marine_Mammals_Wenz"
    )
    sim.add_biological(
        "fish_chorus_mccauley_parsons",
        fish_activity="high",
        label="Fish_Chorus_McCauley_Parsons",
    )

    fig, ax = sim.plot(title="Biological Noise Sources", show_total=False)
    plt.savefig("biological_models.png", dpi=300, bbox_inches="tight")

    # Example 3: Explosion models comparison
    sim = AmbientNoiseSimulator(freq=np.logspace(0, 4, 1000))  # 1 Hz to 10 kHz
    sim.add_explosion(
        "chapman", charge_weight=10, distance=1000, label="Explosion_Chapman"
    )
    sim.add_explosion("arons", charge_weight=10, distance=1000, label="Explosion_Arons")
    sim.add_explosion(
        "broadband_generic",
        source_level=160,
        center_freq=50,
        bandwidth=3,
        label="Explosion_Broadband",
    )

    fig, ax = sim.plot(
        title="Explosion Noise Models (10 kg TNT at 1 km)", show_total=False
    )
    plt.savefig("explosion_models.png", dpi=300, bbox_inches="tight")

    # Example 4: Complex scenario with multiple model types including rain
    sim = AmbientNoiseSimulator()
    sim.add_thermal("mellen", label="Thermal_Mellen")
    sim.add_turbulence("urick", label="Turbulence_Urick")
    sim.add_wind("piggot_merklinger", wind_speed=8, water_depth="deep", label="Wind_PM")
    sim.add_shipping(
        "wenz", shipping_level="medium", water_depth="deep", label="Shipping_Wenz"
    )
    sim.add_rain("ma_nystuen", rain_rate="moderate", label="Rain_Ma_Nystuen")
    sim.add_biological(
        "snapping_shrimp_cato", shrimp_activity="moderate", label="Shrimp_Cato"
    )

    fig, ax = sim.plot(title="Complex Ambient Noise Scenario with Rain")
    plt.savefig("complex_scenario.png", dpi=300, bbox_inches="tight")

    # Show all plots
    plt.show()
