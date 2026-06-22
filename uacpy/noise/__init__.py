"""
Ambient-noise model — Tollefsen / Pecknold packaging.

Compact Wenz-style noise spectrum (wind / shipping / rain / thermal /
turbulence), in dB re 1 µPa²/Hz.

Examples
--------
>>> import numpy as np
>>> from uacpy.noise import WenzNoise
>>> f = np.linspace(1, 1e5, 1000)
>>> wenz = WenzNoise(f, wind_speed=15,
...                  water_depth='deep', shipping_level='medium',
...                  rain_rate='moderate')
>>> from uacpy.visualization import plot_wenz
>>> plot_wenz(wenz)
>>> psd_pa2_per_hz = wenz.as_psd(ref=1)            # linear, Pa²/Hz
"""

from uacpy.noise.noise import (
    compute_windnoise, WenzNoise, NoiseComponents,
    WIND_MODELS, SHIPPING_MODELS, RAIN_MODELS, THERMAL_MODELS, TURBULENCE_MODELS,
)
from uacpy.noise.ship_radiated_noise import (
    RNL_UNCERTAINTY_DB,
    lloyd_mirror_correction,
    monopole_source_level,
    nominal_source_depth,
    radiated_noise_level,
)
from uacpy.noise.marine_mammal import (
    HEARING_GROUPS,
    WEIGHTING_PARAMS,
    apply_weighting,
    auditory_weighting,
    weighted_level,
)
from uacpy.noise import marine_mammal, ship_radiated_noise

__all__ = [
    'compute_windnoise',
    'WenzNoise',
    'NoiseComponents',
    'WIND_MODELS', 'SHIPPING_MODELS', 'RAIN_MODELS',
    'THERMAL_MODELS', 'TURBULENCE_MODELS',
    # ship radiated noise (ISO 17208)
    'radiated_noise_level',
    'nominal_source_depth',
    'lloyd_mirror_correction',
    'monopole_source_level',
    'RNL_UNCERTAINTY_DB',
    'ship_radiated_noise',
    # marine-mammal auditory weighting (Southall 2019)
    'auditory_weighting',
    'apply_weighting',
    'weighted_level',
    'WEIGHTING_PARAMS',
    'HEARING_GROUPS',
    'marine_mammal',
]
