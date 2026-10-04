"""Underwater acoustics utilities: the documented public path of
:mod:`uacpy.core.acoustics`.

``from uacpy.acoustics import spl`` works as well as ``uacpy.acoustics.spl``.
The subjects (seawater, boundaries, bubbles, levels, wavenumber, modal,
attenuation) are described in :mod:`uacpy.core.acoustics`; the absorption
formulas on plain arrays (``absorption_thorp``,
``absorption_francois_garrison``, ...) are among them. The absorption laws
(``Thorp``, ``FrancoisGarrison``, ...) are top-level names: one object per
law, which an ``Environment`` holds and whose ``.table(f)`` evaluates it.
"""

from uacpy.core.acoustics import *  # noqa: F401,F403
from uacpy.core.acoustics import __all__ as _CORE_ACOUSTICS

__all__ = [*_CORE_ACOUSTICS]
