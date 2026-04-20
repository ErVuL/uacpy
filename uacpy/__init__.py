"""
uacpy - Underwater Acoustics Python Library

A comprehensive library for underwater acoustics propagation modeling.
"""

__version__ = '0.0.1'
__author__ = 'ErVuL'

# Core classes
from uacpy.core.source import Source
from uacpy.core.environment import (
    Environment, BoundaryProperties, RangeDependentBottom,
    SedimentLayer, LayeredBottom, RangeDependentLayeredBottom,
    generate_sea_surface,
)
from uacpy.core.receiver import Receiver
from uacpy.core.field import Field

# Models
from uacpy import models

# Visualization
from uacpy.visualization import plots as plot

# I/O utilities
from uacpy import io

# Signal processing (renamed from 'signal' to 'acoustic_signal' to avoid conflict with Python's built-in signal module)
from uacpy import acoustic_signal as signal

# Ambient noise modeling
from uacpy import noise

# Core utilities (acoustics module)
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
