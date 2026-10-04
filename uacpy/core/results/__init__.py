"""Result types produced by the propagation models.

A package split by result kind. All public names are re-exported here, so
``from uacpy.core.results import Field`` resolves regardless of submodule.
"""

from uacpy.core.results._base import (  # noqa: F401
    Result, PhaseReference,
    _UNIVERSAL_METADATA, _DOCUMENTED_METADATA,
)
from uacpy.core.results.field import Field
from uacpy.core.results.stack import ResultStack
from uacpy.core.results.rays import Arrivals, Rays
from uacpy.core.results.modes import MediaTable, Modes
from uacpy.core.results.array_products import (
    Covariance, Replicas, ambiguity_field,
)
from uacpy.core.results.reflection import ReflectionCoefficient
from uacpy.core.results.greens_function import GreensFunction
from uacpy.core.results.speeds import SoundSpeeds

__all__ = [
    'Result', 'PhaseReference', 'Field', 'ResultStack',
    'Arrivals', 'Rays', 'Modes',
    'Covariance', 'Replicas', 'ReflectionCoefficient', 'GreensFunction',
    'ambiguity_field', 'SoundSpeeds', 'MediaTable',
    # submodules
    'array_products', 'field', 'greens_function', 'modes', 'quantities', 'rays', 'reflection',
    'speeds', 'stack',
]
