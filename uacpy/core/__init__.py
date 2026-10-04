"""
Core classes for underwater acoustics modeling

Every name here resolves on first access (PEP 562), so importing one core
module — ``uacpy.core.source`` from the signal layer, say — loads that module
and what it imports, never the result types of ``uacpy.core.results``.
"""

import importlib as _importlib
from importlib.util import find_spec as _find_spec
import typing as _typing

# Public name -> the module it is read from.
_EXPORTS = {
    'Source': 'uacpy.core.source',
    **{name: 'uacpy.core.environment' for name in (
        'Environment', 'BoundaryProperties', 'SedimentLayer', 'SeabedColumn',
        'Bottom', 'SoundSpeedProfile', 'Bathymetry', 'Altimetry', 'Surface',
    )},
    'generate_sea_surface': 'uacpy.core.altimetry',
    **{name: 'uacpy.core.absorption' for name in (
        'Absorption', 'Thorp', 'FrancoisGarrison', 'Biological',
        'BiologicalLayer', 'ConstantAbsorption', 'AbsorptionCoefficient',
    )},
    'Receiver': 'uacpy.core.receiver',
    **{name: 'uacpy.core.results' for name in (
        'Result', 'PhaseReference', 'Field', 'ResultStack',
        'Arrivals', 'Rays', 'Modes',
        'Covariance', 'Replicas',
        'ReflectionCoefficient', 'GreensFunction',
    )},
    **{name: 'uacpy.core.exceptions' for name in (
        'UACPYError', 'ExecutableNotFoundError', 'ModelExecutionError',
        'InvalidDepthError', 'UnsupportedFeatureError', 'ConfigurationError',
        'DataFetchError', 'FileFormatError', 'OutputContractError',
        'UACPYWarning', 'NumericsWarning', 'ValidityWarning',
        'FallbackWarning', 'ProvenanceWarning', 'IOWarning',
    )},
    **{name: 'uacpy.core.materials' for name in (
        'MATERIALS', 'list_materials', 'get_material', 'materials_table',
    )},
    'BoundaryType': 'uacpy.core.boundary',
    # The package's one reference water: what every default describes.
    **{name: 'uacpy.core.constants' for name in (
        'DEFAULT_SOUND_SPEED', 'DEFAULT_WATER_DENSITY_G_CM3',
        'REFERENCE_TEMPERATURE_C', 'REFERENCE_SALINITY_PSU',
        'REFERENCE_DEPTH_M', 'REFERENCE_PH',
    )},
}

#: The submodules ``__all__`` lists beside the names above.
_LISTED_SUBMODULES = ('acoustics', 'materials', 'metrics', 'units')

__all__ = [*_EXPORTS, *_LISTED_SUBMODULES]


if _typing.TYPE_CHECKING:
    # Static mirror of ``_EXPORTS`` for PEP 561 consumers: a checker reads
    # these imports instead of inferring ``Any`` from ``__getattr__``. The
    # body never runs, so nothing is imported eagerly. Kept in step with the
    # table by ``tests/test_lazy_imports.py``.
    from uacpy.core import acoustics, materials, metrics, units  # noqa: F401
    from uacpy.core.constants import (  # noqa: F401
        DEFAULT_SOUND_SPEED, DEFAULT_WATER_DENSITY_G_CM3, REFERENCE_DEPTH_M,
        REFERENCE_PH, REFERENCE_SALINITY_PSU, REFERENCE_TEMPERATURE_C,
    )
    from uacpy.core.absorption import (  # noqa: F401
        Absorption, AbsorptionCoefficient, Biological, BiologicalLayer,
        ConstantAbsorption, FrancoisGarrison, Thorp,
    )
    from uacpy.core.boundary import BoundaryType  # noqa: F401
    from uacpy.core.altimetry import generate_sea_surface  # noqa: F401
    from uacpy.core.environment import (  # noqa: F401
        Altimetry, Bathymetry, Bottom, BoundaryProperties, Environment,
        SeabedColumn, SedimentLayer, SoundSpeedProfile, Surface,
    )
    from uacpy.core.exceptions import (  # noqa: F401
        ConfigurationError, DataFetchError, ExecutableNotFoundError,
        FileFormatError, InvalidDepthError, ModelExecutionError,
        OutputContractError, UACPYError,
        UnsupportedFeatureError, UACPYWarning, NumericsWarning,
        ValidityWarning, FallbackWarning, ProvenanceWarning, IOWarning,
    )
    from uacpy.core.materials import (  # noqa: F401
        MATERIALS, get_material, list_materials, materials_table,
    )
    from uacpy.core.receiver import Receiver  # noqa: F401
    from uacpy.core.results import (  # noqa: F401
        Arrivals, Covariance, Field, GreensFunction, Modes, PhaseReference,
        Rays, ReflectionCoefficient, Replicas, Result, ResultStack,
    )
    from uacpy.core.source import Source  # noqa: F401


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is not None:
        value = getattr(_importlib.import_module(module), name)
        globals()[name] = value       # cache: __getattr__ not hit again
        return value
    # A submodule is an attribute of its package once imported; importing
    # it here keeps ``uacpy.core.<submodule>`` reachable without a prior
    # ``import`` of it.
    if (not name.startswith('__')
            and _find_spec(f'{__name__}.{name}') is not None):
        return _importlib.import_module(f'{__name__}.{name}')
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}.")


def __dir__():
    # The public names, loaded or not, plus the module's dunders; the
    # private loader table and the ``_importlib`` alias stay out.
    return sorted(set(__all__) | {n for n in globals() if n.startswith('__')})
