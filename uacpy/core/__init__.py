"""
Core classes for underwater acoustics modeling
"""

from uacpy.core.source import Source
from uacpy.core.environment import (
    Environment, BoundaryProperties, RangeDependentBottom,
    SedimentLayer, LayeredBottom, RangeDependentLayeredBottom
)
from uacpy.core.receiver import Receiver
from uacpy.core.field import Field

# Import acoustics utilities
try:
    from uacpy.core import acoustics
except ImportError:
    pass

__all__ = [
    'Source',
    'Environment',
    'BoundaryProperties',
    'RangeDependentBottom',
    'SedimentLayer',
    'LayeredBottom',
    'RangeDependentLayeredBottom',
    'Receiver',
    'Field',
    'acoustics'
]
