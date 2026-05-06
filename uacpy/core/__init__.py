"""
Core classes for underwater acoustics modeling
"""

from uacpy.core.source import Source
from uacpy.core.environment import (
    Environment, BoundaryProperties, RangeDependentBottom,
    SedimentLayer, LayeredBottom, RangeDependentLayeredBottom
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

from uacpy.core import acoustics

__all__ = [
    'Source',
    'Environment',
    'BoundaryProperties',
    'RangeDependentBottom',
    'SedimentLayer',
    'LayeredBottom',
    'RangeDependentLayeredBottom',
    'Receiver',
    'Result', 'PhaseReference',
    'TLField', 'PressureField', 'TransferFunction',
    'TimeSeriesField', 'TimeTrace',
    'Arrivals', 'Rays', 'Modes',
    'Covariance', 'Replicas',
    'ReflectionCoefficient',
    'acoustics',
]
