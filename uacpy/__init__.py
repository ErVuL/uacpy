"""
uacpy - Underwater Acoustics Python Library

A comprehensive library for underwater acoustics propagation modeling.

Conventions
-----------
Distances are in **metres** unless the attribute or argument name carries
an explicit suffix (``_km``, ``_cm``). Sound speeds are m/s, densities
g/cm³, attenuations dB/wavelength, frequencies Hz. Depth is positive
downward; sea-surface altimetry height is positive upward (z=0 at the
mean sea surface).
"""

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version('uacpy')
except Exception:
    __version__ = 'unknown'
__author__ = 'ErVuL'

from uacpy.core.source import Source
from uacpy.core.environment import (
    Environment, BoundaryProperties, RangeDependentBottom,
    SedimentLayer, LayeredBottom, RangeDependentLayeredBottom,
    SoundSpeedProfile, generate_sea_surface,
)
from uacpy.core.receiver import Receiver
from uacpy.core.results import (
    Result, PhaseReference,
    TLField, PressureField, TransferFunction,
    TimeSeriesField, TimeTrace,
    Arrivals, Rays, Modes,
    Covariance, Replicas,
    ReflectionCoefficient,
)
from uacpy.core.constants import (
    AttenuationUnits, VolumeAttenuation, BoundaryType,
)

from uacpy import models
from uacpy.models.base import RunMode
from uacpy.visualization import plots as plot
from uacpy import io

# Exposed as `uacpy.signal`; the package is named `acoustic_signal` to avoid
# colliding with Python's stdlib `signal` module.
from uacpy import acoustic_signal as signal

from uacpy import noise
from uacpy.core import acoustics

__all__ = [
    'Source',
    'Environment',
    'BoundaryProperties',
    'RangeDependentBottom',
    'SedimentLayer',
    'LayeredBottom',
    'RangeDependentLayeredBottom',
    'SoundSpeedProfile',
    'generate_sea_surface',
    'Receiver',
    'Result', 'PhaseReference',
    'TLField', 'PressureField', 'TransferFunction',
    'TimeSeriesField', 'TimeTrace',
    'Arrivals', 'Rays', 'Modes',
    'Covariance', 'Replicas',
    'ReflectionCoefficient',
    'AttenuationUnits', 'VolumeAttenuation', 'BoundaryType',
    'RunMode',
    'models',
    'plot',
    'io',
    'signal',
    'noise',
    'acoustics',
    '__version__',
]
