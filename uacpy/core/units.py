"""Scale conversions between the units the public uacpy API speaks (metres,
Hz, radians) and the ones other layers need.

Three pure array operations with no dependency beyond numpy, so every layer —
``io`` at the file-format boundary, ``visualization`` at the axis-label
boundary — can reach them without importing a sibling package for arithmetic.
The io writers and readers import them from here: every writer that emits
a km-on-disk axis goes through ``m_to_km``, every reader that returns a
metres-API axis through ``km_to_m``, and phase columns through
``deg_to_rad`` / ``rad_to_deg``, so the "did I convert?" question stays
grep-able.
"""

from __future__ import annotations

import numpy as np


def km_to_m(x):
    """Multiply a km axis by 1000 to get metres."""
    return np.asarray(x, dtype=float) * 1000.0


def m_to_km(x):
    """Divide a metres axis by 1000 to get km."""
    return np.asarray(x, dtype=float) / 1000.0


def deg_to_rad(x):
    """Convert degrees to radians."""
    return np.asarray(x, dtype=float) * (np.pi / 180.0)


def rad_to_deg(x):
    """Convert radians to degrees.

    The write direction of :func:`deg_to_rad`: uacpy carries phase in
    radians and the ``.brc``/``.trc`` reflection tables hold it in degrees,
    so :func:`~uacpy.io.refl_io.write_reflection_coefficient` converts here
    and :func:`~uacpy.io.refl_io.read_reflection_coefficient` converts back.
    """
    return np.asarray(x, dtype=float) * (180.0 / np.pi)


__all__ = ["km_to_m", "m_to_km", "deg_to_rad", "rad_to_deg"]
