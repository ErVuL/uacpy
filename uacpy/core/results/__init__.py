"""Result types produced by the propagation models.

A package split by result kind. All public names are re-exported here, so
``from uacpy.core.results import Field`` resolves regardless of submodule.
"""

from uacpy.core.results._base import (  # noqa: F401
    Result, PhaseReference,
    _UNIVERSAL_METADATA, _DOCUMENTED_METADATA,
)
from uacpy.core.results.field import Field, ResultStack
from uacpy.core.results.rays import Arrivals, Rays
from uacpy.core.results.modes import Modes
from uacpy.core.results.array_products import Covariance, Replicas
from uacpy.core.results.reflection import ReflectionCoefficient

__all__ = [
    'Result', 'PhaseReference', 'Field', 'ResultStack',
    'Arrivals', 'Rays', 'Modes',
    'Covariance', 'Replicas', 'ReflectionCoefficient',
]
