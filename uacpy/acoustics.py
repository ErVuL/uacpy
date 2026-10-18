"""Underwater acoustics utilities: the documented public path of
:mod:`uacpy.core.acoustics`.

``from uacpy.acoustics import spl`` works as well as ``uacpy.acoustics.spl``.
The subjects (seawater, boundaries, bubbles, levels, weighting, wavenumber,
modal, arrays, ray_geometry, attenuation) are described in
:mod:`uacpy.core.acoustics`; the absorption
formulas on plain arrays (``absorption_thorp``,
``absorption_francois_garrison``, ...) are among them. The absorption laws
(``Thorp``, ``FrancoisGarrison``, ...) are top-level names: one object per
law, which an ``Environment`` holds and whose ``.table(f)`` evaluates it.
:func:`bottom_loss_curve` (a seabed's plane-wave reflection as a
``ReflectionCoefficient``) is defined in :mod:`uacpy.core.results.reflection`,
and :func:`resample_depth_range` (a ``(depth, range)`` grid resampled onto
another, which ``Field.resample_to`` wraps) in the grid library
:mod:`uacpy.core._grid`; :func:`mask_below_seafloor` (an array's samples below
a seafloor set to NaN, which ``Field.mask_below_seafloor`` wraps) beside the
bathymetry carrier in :mod:`uacpy.core.bathymetry`.
"""

from uacpy.core.acoustics import *  # noqa: F401,F403
from uacpy.core.acoustics import __all__ as _CORE_ACOUSTICS
# The plane-wave bottom loss of a fluid seabed is a ReflectionCoefficient,
# so it is built where that result lives.
from uacpy.core.results.reflection import bottom_loss_curve
# The (depth, range) regrid the Field method and the RAM wrappers share.
from uacpy.core._grid import resample_depth_range
# The seafloor mask on plain arrays, beside the carrier it reads.
from uacpy.core.bathymetry import mask_below_seafloor

__all__ = [*_CORE_ACOUSTICS, 'bottom_loss_curve', 'resample_depth_range',
           'mask_below_seafloor']
