"""
uacpy - Underwater Acoustics Python Library

A comprehensive library for underwater acoustics propagation modeling.
"""

__version__ = '0.0.1'
__author__ = 'ErVuL'

from uacpy.core.source import Source
from uacpy.core.environment import (
    Environment, BoundaryProperties, RangeDependentBottom,
    SedimentLayer, LayeredBottom, RangeDependentLayeredBottom,
    generate_sea_surface,
)
from uacpy.core.receiver import Receiver
from uacpy.core.field import Field

from uacpy import models
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
    'generate_sea_surface',
    'Receiver',
    'Field',
    'models',
    'plot',
    'io',
    'signal',
    'noise',
    'acoustics',
    '__version__',
]
