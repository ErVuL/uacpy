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

Import behaviour
----------------
``import uacpy`` eagerly loads only :mod:`uacpy.core` (numpy-based
carriers, results, exceptions). The model wrappers, plotting, DSP and
data subpackages resolve lazily on first attribute access (PEP 562), so
matplotlib/scipy are paid for only by code that actually uses them.
"""

import importlib

from uacpy._version import __version__
__author__ = 'ErVuL'

from uacpy._log import install_warning_formatter as _install_warning_formatter

from uacpy.core.source import Source
from uacpy.core.environment import (
    Environment, BoundaryProperties, SedimentLayer, SeabedColumn, Bottom,
    SoundSpeedProfile, generate_sea_surface, Bathymetry, Altimetry, Surface,
)
from uacpy.core.absorption import (
    Absorption, Thorp, FrancoisGarrison, Biological, BiologicalLayer,
    ConstantAbsorption,
)
from uacpy.core.receiver import Receiver
from uacpy.core.results import (
    Result, PhaseReference, Field, ResultStack,
    Arrivals, Rays, Modes,
    Covariance, Replicas,
    ReflectionCoefficient,
)
from uacpy.core.constants import (
    AttenuationUnits, BoundaryType,
)
from uacpy.core.exceptions import (
    UACPYError,
    ExecutableNotFoundError,
    ModelExecutionError,
    InvalidDepthError,
    UnsupportedFeatureError,
    ConfigurationError,
    DataFetchError,
    FileFormatError,
)
from uacpy.core import acoustics
from uacpy.core import materials
from uacpy.core.materials import MATERIALS, list_materials, get_material

# Lazy surface (PEP 562). Everything below core is deferred to first use:
# ``uacpy.models`` pulls scipy (ram/bellhop), ``uacpy.visualization`` pulls
# matplotlib, ``uacpy.acoustic_signal`` pulls scipy.signal/scipy.stats.
# Submodule attributes resolve through __getattr__; ``import uacpy.models``
# still works as a normal import.
_LAZY_SUBMODULES = {
    'models': 'uacpy.models',
    'io': 'uacpy.io',
    'acoustic_signal': 'uacpy.acoustic_signal',
    'noise': 'uacpy.noise',
    'sonar': 'uacpy.sonar',
    'comms': 'uacpy.comms',
    'data': 'uacpy.data',
    'parallel': 'uacpy.parallel',
    # Cross-model comparison metrics (re-export module of uacpy.core.metrics).
    'metrics': 'uacpy.metrics',
    'visualization': 'uacpy.visualization',
    'plot': 'uacpy.visualization.plots',
}

_LAZY_ATTRS = {
    # model classes
    **{name: ('uacpy.models', name) for name in (
        'Bellhop', 'Kraken', 'RAM', 'Scooter', 'SPARC', 'Bounce',
        'OAST', 'OASN', 'OASR', 'OASP', 'OASS', 'OASSP', 'OASES',
    )},
    'PropagationModel': ('uacpy.models.base', 'PropagationModel'),
    'RunMode': ('uacpy.models.base', 'RunMode'),
    'ModelSpec': ('uacpy.models.base', 'ModelSpec'),
    # parallel execution
    'run_parallel': ('uacpy.parallel', 'run_parallel'),
    'Job': ('uacpy.parallel', 'Job'),
    'ParallelResult': ('uacpy.parallel', 'ParallelResult'),
    # plotting entry points
    'plot_result': ('uacpy.visualization.plots', 'plot_result'),
    'plot_field': ('uacpy.visualization.plots', 'plot_field'),
    'plot_overview': ('uacpy.visualization.plots', 'plot_overview'),
    'compare_models': ('uacpy.visualization.plots', 'compare_models'),
}


def __getattr__(name):
    target = _LAZY_SUBMODULES.get(name)
    if target is not None:
        module = importlib.import_module(target)
        globals()[name] = module          # cache: __getattr__ not hit again
        return module
    entry = _LAZY_ATTRS.get(name)
    if entry is not None:
        module_name, attr = entry
        value = getattr(importlib.import_module(module_name), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__) | set(globals()))


__all__ = [
    'Source',
    'Environment',
    'BoundaryProperties',
    'SedimentLayer',
    'SeabedColumn',
    'Bottom',
    'SoundSpeedProfile', 'generate_sea_surface', 'Bathymetry', 'Altimetry', 'Surface',
    'Absorption', 'Thorp', 'FrancoisGarrison',
    'Biological', 'BiologicalLayer', 'ConstantAbsorption',
    'Receiver',
    'Result', 'PhaseReference', 'Field', 'ResultStack',
    'Arrivals', 'Rays', 'Modes',
    'Covariance', 'Replicas',
    'ReflectionCoefficient',
    'AttenuationUnits', 'BoundaryType',
    'UACPYError',
    'ExecutableNotFoundError',
    'ModelExecutionError',
    'InvalidDepthError',
    'UnsupportedFeatureError',
    'ConfigurationError',
    'DataFetchError',
    'FileFormatError',
    'RunMode',
    'PropagationModel',
    'ModelSpec',
    'Bellhop', 'Kraken', 'RAM', 'Scooter', 'SPARC', 'Bounce',
    'OAST', 'OASN', 'OASR', 'OASP', 'OASS', 'OASSP', 'OASES',
    'run_parallel',
    'Job',
    'ParallelResult',
    'models',
    'plot',
    'plot_result', 'plot_field', 'plot_overview', 'compare_models',
    'io',
    'parallel',
    'visualization',
    'acoustic_signal',
    'noise',
    'sonar',
    'comms',
    'data',
    'acoustics',
    'materials', 'MATERIALS', 'list_materials', 'get_material',
    'metrics',
    '__version__',
]


_install_warning_formatter()
