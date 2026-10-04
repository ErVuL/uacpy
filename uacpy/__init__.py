"""
uacpy - Underwater Acoustics Python Library

A comprehensive library for underwater acoustics propagation modeling.

Conventions
-----------
Distances are in **metres** unless the attribute or argument name carries
an explicit suffix (``_km``). Sound speeds are m/s, densities
g/cm³, attenuations dB/wavelength, frequencies Hz. Depth is positive
downward; sea-surface altimetry height is positive upward (z=0 at the
mean sea surface).

Import behaviour
----------------
Every public name resolves on first attribute access (PEP 562): ``import
uacpy`` loads the export tables and the warning formatter, and a carrier,
a result type, a model wrapper, a plotter or a subpackage is imported when it
is first named, so matplotlib/scipy are paid for only by code that uses them.

The top level lists nothing of its own. Its core names are
``uacpy.core.__all__``; its model names are the engines of
``uacpy.models._registry`` plus the OASES family base and the protocol types
(:class:`~uacpy.models.base.PropagationModel`,
:class:`~uacpy.models._spec.ModelSpec`, :class:`RunMode`); the rest are the
subpackages, the parallel runner and the plotting entry points.
"""

import importlib as _importlib
from importlib.util import find_spec as _find_spec
from typing import TYPE_CHECKING as _TYPE_CHECKING

from uacpy._version import __version__
__author__ = 'ErVuL'

from uacpy._log import install_warning_formatter as _install_warning_formatter
from uacpy import core as _core
from uacpy.models._registry import ENGINES as _ENGINES

# Lazy surface (PEP 562). ``uacpy.models`` pulls scipy (ram/bellhop),
# ``uacpy.visualization`` pulls matplotlib, ``uacpy.acoustic_signal`` pulls
# scipy.signal/scipy.stats. Submodule attributes resolve through __getattr__;
# ``import uacpy.models`` still works as a normal import.
_LAZY_SUBMODULES = {
    'models': 'uacpy.models',
    'io': 'uacpy.io',
    'acoustic_signal': 'uacpy.acoustic_signal',
    'noise': 'uacpy.noise',
    'sonar': 'uacpy.sonar',
    'comms': 'uacpy.comms',
    'data': 'uacpy.data',
    'parallel': 'uacpy.parallel',
    # The user-helper physics and the comparison metrics, as root modules
    # over uacpy.core.acoustics / uacpy.core.metrics.
    'acoustics': 'uacpy.acoustics',
    'metrics': 'uacpy.metrics',
    # The seabed catalogue and the knot, Beaufort and sea-state conversions,
    # as root modules over uacpy.core.materials / uacpy.core.units.
    'materials': 'uacpy.materials',
    'units': 'uacpy.units',
    # Closed-form reference fields (free field, image, ideal, Pekeris).
    'analytic': 'uacpy.analytic',
    'visualization': 'uacpy.visualization',
    'plot': 'uacpy.plot',
}

#: The model classes the top level offers: every registered engine, the base
#: and factory of the OASES family, and the protocol types.
_MODEL_NAMES = (tuple(entry.class_name for entry in _ENGINES.values())
                + ('OASES', 'PropagationModel', 'ModelSpec', 'RunMode'))

_LAZY_ATTRS = {
    # Every core name that is not a submodule, read through uacpy.core, so
    # importing a layer below the result types does not load them.
    **{name: ('uacpy.core', name) for name in _core.__all__
       if _find_spec(f'uacpy.core.{name}') is None},
    **{name: ('uacpy.models', name) for name in _MODEL_NAMES},
    # parallel execution
    'run_parallel': ('uacpy.parallel', 'run_parallel'),
    'Job': ('uacpy.parallel', 'Job'),
    'ParallelResult': ('uacpy.parallel', 'ParallelResult'),
}


if _TYPE_CHECKING:
    # Static mirror of the two tables above, for PEP 561 consumers. The body
    # never runs (``TYPE_CHECKING`` is False at runtime), so the PEP 562
    # deferral above is untouched and nothing here is imported eagerly; it
    # exists so a checker resolves ``uacpy.Bellhop`` to the class rather than
    # to the ``ModuleType | Any`` union it infers from ``__getattr__``'s two
    # return paths — which makes ``from uacpy import Bellhop; Bellhop()`` a
    # hard pyright error on a py.typed package. Kept in step with the tables
    # by ``test_every_lazy_name_is_statically_re_imported_for_type_checkers``.
    from uacpy import (  # noqa: F401
        acoustic_signal, acoustics, analytic, comms, data, io, materials,
        metrics, models, noise, parallel, plot, sonar, units, visualization,
    )
    from uacpy.core import (  # noqa: F401
        Absorption, AbsorptionCoefficient, Altimetry, Arrivals, Bathymetry,
        Biological, BiologicalLayer, Bottom, BoundaryProperties, BoundaryType,
        ConfigurationError, ConstantAbsorption, Covariance, DataFetchError,
        DEFAULT_SOUND_SPEED, DEFAULT_WATER_DENSITY_G_CM3, Environment,
        ExecutableNotFoundError, Field, FileFormatError, FrancoisGarrison,
        GreensFunction, InvalidDepthError, MATERIALS, ModelExecutionError,
        Modes, OutputContractError, PhaseReference, Rays, Receiver,
        REFERENCE_DEPTH_M, REFERENCE_PH, REFERENCE_SALINITY_PSU,
        REFERENCE_TEMPERATURE_C, ReflectionCoefficient, Replicas, Result,
        ResultStack, SeabedColumn, SedimentLayer, SoundSpeedProfile, Source,
        Surface, Thorp, UACPYError, UACPYWarning, NumericsWarning,
        ValidityWarning, FallbackWarning, ProvenanceWarning, IOWarning,
        UnsupportedFeatureError, generate_sea_surface, get_material,
        list_materials, materials_table,
    )
    from uacpy.models import (  # noqa: F401
        Bellhop, Bounce, Kraken, ModelSpec, OASES, OASN, OASP, OASR, OASS,
        OASSP, OAST, PropagationModel, RAM, RunMode, SPARC, Scooter,
    )
    from uacpy.parallel import Job, ParallelResult, run_parallel  # noqa: F401


def __getattr__(name):
    target = _LAZY_SUBMODULES.get(name)
    if target is not None:
        module = _importlib.import_module(target)
        globals()[name] = module          # cache: __getattr__ not hit again
        return module
    entry = _LAZY_ATTRS.get(name)
    if entry is not None:
        module_name, attr = entry
        value = getattr(_importlib.import_module(module_name), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}.")


def __dir__():
    return sorted(set(__all__) | set(globals()))


__all__ = [*_LAZY_ATTRS, *_LAZY_SUBMODULES, '__version__']


_install_warning_formatter()
