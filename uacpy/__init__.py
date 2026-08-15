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

from uacpy import models
from uacpy.models.base import PropagationModel, RunMode, ModelSpec
from uacpy.models import (
    Bellhop, Kraken, RAM, Scooter, SPARC, Bounce,
    OAST, OASN, OASR, OASP, OASS, OASSP, OASES,
)
from uacpy.parallel import run_parallel, Job, ParallelResult
from uacpy.visualization import plots as plot
from uacpy.visualization.plots import (
    plot_result, plot_field, plot_overview, compare_models,
)
from uacpy import io
from uacpy import acoustic_signal
from uacpy import noise
from uacpy import sonar
from uacpy import comms
from uacpy import data
from uacpy.core import acoustics
from uacpy.core import materials
from uacpy.core.materials import MATERIALS, list_materials, get_material

# Cross-model comparison metrics (re-export module of uacpy.core.metrics).
from uacpy import metrics

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
