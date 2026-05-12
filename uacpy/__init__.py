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
"""

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version('uacpy')
except Exception:
    __version__ = 'unknown'
__author__ = 'ErVuL'

from uacpy._log import install_warning_formatter as _install_warning_formatter

from uacpy.core.source import Source
from uacpy.core.environment import (
    Environment, BoundaryProperties, RangeDependentBottom,
    SedimentLayer, LayeredBottom, RangeDependentLayeredBottom,
    SoundSpeedProfile, generate_sea_surface,
)
from uacpy.core.absorption import (
    Absorption, Thorp, FrancoisGarrison, Biological, BiologicalLayer,
    ConstantAbsorption,
)
from uacpy.core.receiver import Receiver
from uacpy.core.results import (
    Result, PhaseReference,
    PressureField, TransferFunction, ResultStack,
    TimeSeriesField, TimeTrace,
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
)

from uacpy import models
from uacpy.models.base import PropagationModel, RunMode
from uacpy.visualization import plots as plot
from uacpy import io

# Exposed as `uacpy.signal`; the package is named `acoustic_signal` to avoid
# colliding with Python's stdlib `signal` module.
from uacpy import acoustic_signal as signal

from uacpy import noise
from uacpy.core import acoustics
from uacpy.core import materials
from uacpy.core.materials import MATERIALS, list_materials, get_material
from uacpy.core.metrics import tl_rmse

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
    'Absorption', 'Thorp', 'FrancoisGarrison',
    'Biological', 'BiologicalLayer', 'ConstantAbsorption',
    'Receiver',
    'Result', 'PhaseReference',
    'PressureField', 'TransferFunction', 'ResultStack',
    'TimeSeriesField', 'TimeTrace',
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
    'RunMode',
    'PropagationModel',
    'models',
    'plot',
    'io',
    'signal',
    'noise',
    'acoustics',
    'materials', 'MATERIALS', 'list_materials', 'get_material',
    'tl_rmse',
    '__version__',
]


_install_warning_formatter()
