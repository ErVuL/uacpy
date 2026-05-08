"""
Core classes for underwater acoustics modeling
"""

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
from uacpy.core.exceptions import (
    UACPYError,
    ExecutableNotFoundError,
    ModelExecutionError,
    InvalidDepthError,
    UnsupportedFeatureError,
    ConfigurationError,
)

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
    'UACPYError',
    'ExecutableNotFoundError',
    'ModelExecutionError',
    'InvalidDepthError',
    'UnsupportedFeatureError',
    'ConfigurationError',
    'acoustics',
]
