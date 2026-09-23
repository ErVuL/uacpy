"""
Core classes for underwater acoustics modeling
"""

from uacpy.core.source import Source
from uacpy.core.environment import (
    Environment, BoundaryProperties, SedimentLayer, SeabedColumn, Bottom,
    SoundSpeedProfile, generate_sea_surface, Bathymetry, Altimetry, Surface,
)
from uacpy.core.acoustics.boundaries import critical_angle
from uacpy.core.acoustics.seawater import (
    sound_speed_mackenzie, sound_speed_unesco, sound_speed_delgrosso,
    sound_speed_teos10, doppler,
)
from uacpy.core.absorption import (
    Absorption, Thorp, FrancoisGarrison, Biological, BiologicalLayer,
    ConstantAbsorption, AbsorptionCoefficient, absorption_thorp,
    absorption_francois_garrison, absorption_biological, absorption_constant,
)
from uacpy.core.receiver import Receiver
from uacpy.core.results import (
    Result, PhaseReference, Field, ResultStack,
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
    DataFetchError,
    FileFormatError,
)

from uacpy.core import acoustics
from uacpy.core import materials
from uacpy.core import metrics
from uacpy.core.materials import MATERIALS, list_materials, get_material
from uacpy.core.constants import (
    AttenuationUnits, BoundaryType,
)

__all__ = [
    'Source',
    'Environment',
    'BoundaryProperties',
    'SedimentLayer',
    'SeabedColumn',
    'Bottom',
    'SoundSpeedProfile',
    'generate_sea_surface',
    'Bathymetry', 'Altimetry', 'Surface',
    'Absorption', 'Thorp', 'FrancoisGarrison',
    'critical_angle',
    'sound_speed_mackenzie', 'sound_speed_unesco',
    'sound_speed_delgrosso', 'sound_speed_teos10', 'doppler',
    'AbsorptionCoefficient', 'absorption_thorp',
    'absorption_francois_garrison', 'absorption_biological',
    'absorption_constant',
    'Biological', 'BiologicalLayer', 'ConstantAbsorption',
    'Receiver',
    'Result', 'PhaseReference', 'Field', 'ResultStack',
    'Arrivals', 'Rays', 'Modes',
    'Covariance', 'Replicas',
    'ReflectionCoefficient',
    'UACPYError',
    'ExecutableNotFoundError',
    'ModelExecutionError',
    'InvalidDepthError',
    'UnsupportedFeatureError',
    'ConfigurationError',
    'DataFetchError',
    'FileFormatError',
    'acoustics',
    'materials', 'MATERIALS', 'list_materials', 'get_material',
    'metrics',
    'AttenuationUnits', 'BoundaryType',
]
