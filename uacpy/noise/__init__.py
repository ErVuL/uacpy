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
>>> wenz.plot()
>>> psd_pa2_per_hz = wenz.as_psd(ref=1)            # linear, Pa²/Hz
"""

from uacpy.noise.noise import compute_windnoise, WenzNoise

__all__ = [
    'compute_windnoise',
    'WenzNoise',
]
