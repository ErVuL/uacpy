"""Result types produced by the propagation models.

Formerly one large ``core/results.py``; now a package split by result kind. All
public names (and the metadata registries the model layer reaches for) are
re-exported here, so ``from uacpy.core.results import Field`` etc. is unchanged.
"""

from uacpy.core.results._base import (
    Result, PhaseReference, _complex_to_db,
    _UNIVERSAL_METADATA, _DOCUMENTED_METADATA,
)
from uacpy.core.results.field import (
    Field, ResultStack, _CANONICAL_AXIS_ORDER,
    _ifft_to_trace, _synthesize_time_series,
)
from uacpy.core.results.rays import Arrivals, Rays
from uacpy.core.results.modes import Modes
from uacpy.core.results.array_products import Covariance, Replicas
from uacpy.core.results.reflection import ReflectionCoefficient

__all__ = [
    'Result', 'PhaseReference', 'Field', 'ResultStack',
    'Arrivals', 'Rays', 'Modes',
    'Covariance', 'Replicas', 'ReflectionCoefficient',
]
