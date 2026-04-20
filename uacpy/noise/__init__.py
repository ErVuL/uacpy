"""
Ambient Noise Module for UACPY

This module provides comprehensive ambient noise modeling capabilities for
underwater acoustics, including:

- Multiple noise source models (wind, shipping, rain, biological, seismic, etc.)
- Composite noise simulation framework
- Wenz-based empirical models
- Knudsen 1948 historical reference models

Main Classes
------------
AmbientNoiseSimulator : Composite noise simulator supporting multiple sources
ModelConfig : Configuration container for individual noise models

Examples
--------
>>> from uacpy.noise import AmbientNoiseSimulator
>>> simulator = AmbientNoiseSimulator()
>>> simulator.add_wind_noise(wind_speed=10)
>>> simulator.add_shipping_noise(shipping_density=0.5)
>>> simulator.compute()
>>> fig, ax = simulator.plot()
"""

from uacpy.noise.noise import (
    AmbientNoiseSimulator,
    ModelConfig,
)

__all__ = [
    'AmbientNoiseSimulator',
    'ModelConfig',
]
