"""Acoustic propagation models.

Every name here resolves on first access (PEP 562): ``import uacpy.models``
loads no engine, and ``uacpy.models.Bellhop`` imports ``uacpy.models.bellhop``
alone. The engine classes are read from :data:`uacpy.models._registry.ENGINES`,
so a registered engine is exported here with no second list.
"""

import importlib as _importlib
from importlib.util import find_spec as _find_spec
import typing as _typing

from uacpy.models._registry import ENGINES as _ENGINES

# Public name -> the module that defines it.
_EXPORTS = {
    # PE grid quality, from models/pe_grid.py. These answer the
    # questions a RAM user has to ask before trusting a run — is this
    # (dr, dz) accurate here, what c0 should the PE expand about, what
    # step is the rotated Crank-Nicolson stable at, how much does the
    # seabed leak — and they took only plain numbers, so they are
    # reachable rather than reported as warning text.
    **{name: 'uacpy.models.pe_grid' for name in (
        'numerov_error',
        'combined_error',
        'optimal_c0',
        'optimize_grid',
        'grid_error',
        'rams_dz_shear_cap',
        'rotated_pade_coefficients',
        'rotated_cn_growth',
        'rotated_growth_floor',
        'seabed_leak_rate',
        'rams_growth_margin',
        'rams_stable_dr',
        'rams_stable_theta',
        'GridInfeasibleError',
    )},
    'PropagationModel': 'uacpy.models.base',
    'RunMode': 'uacpy.core.run_settings',
    'ModelSpec': 'uacpy.models._spec',
    'RunSettings': 'uacpy.core.run_settings',
    **{entry.class_name: entry.module for entry in _ENGINES.values()},
    # The abstract base and factory of the OASES family (not an engine).
    'OASES': 'uacpy.models.oases',
}

__all__ = list(_EXPORTS)


if _typing.TYPE_CHECKING:
    # Static mirror of ``_EXPORTS`` for PEP 561 consumers: a checker reads
    # these imports instead of inferring ``Any`` from ``__getattr__``. The
    # body never runs, so nothing is imported eagerly. Kept in step with the
    # table by ``tests/test_lazy_imports.py``.
    from uacpy.core.run_settings import RunMode, RunSettings  # noqa: F401
    from uacpy.models.base import PropagationModel  # noqa: F401
    from uacpy.models._spec import ModelSpec  # noqa: F401
    from uacpy.models.bellhop import Bellhop  # noqa: F401
    from uacpy.models.bounce import Bounce  # noqa: F401
    from uacpy.models.kraken import Kraken  # noqa: F401
    from uacpy.models.oases import (  # noqa: F401
        OASES, OASN, OASP, OASR, OASS, OASSP, OAST,
    )
    from uacpy.models.pe_grid import (  # noqa: F401
        GridInfeasibleError, combined_error, grid_error, numerov_error,
        optimal_c0, optimize_grid, rams_dz_shear_cap, rams_growth_margin,
        rams_stable_dr, rams_stable_theta, rotated_cn_growth,
        rotated_growth_floor, rotated_pade_coefficients, seabed_leak_rate,
    )
    from uacpy.models.ram import RAM  # noqa: F401
    from uacpy.models.scooter import Scooter  # noqa: F401
    from uacpy.models.sparc import SPARC  # noqa: F401


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is not None:
        value = getattr(_importlib.import_module(module), name)
        globals()[name] = value       # cache: __getattr__ not hit again
        return value
    # A submodule is an attribute of its package once imported; importing
    # it here keeps ``uacpy.models.<submodule>`` reachable without a prior
    # ``import`` of it.
    if (not name.startswith('__')
            and _find_spec(f'{__name__}.{name}') is not None):
        return _importlib.import_module(f'{__name__}.{name}')
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}.")


def __dir__():
    # The public names, loaded or not, plus the module's dunders; the
    # private loader table and the ``_importlib`` alias stay out.
    return sorted(set(__all__) | {n for n in globals() if n.startswith('__')})
