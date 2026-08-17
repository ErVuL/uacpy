"""Unit-conversion helpers used at the boundary between the public uacpy
API (metres / Hz / radians) and the native Acoustics-Toolbox / OASES /
RAM file formats (which variously expect km, kHz, or degrees).

Every writer that emits a km-on-disk axis goes through ``m_to_km``;
every reader that returns a metres-API axis goes through ``km_to_m``.
Same for phase columns via ``deg_to_rad`` / ``rad_to_deg``. Centralising
the conversions makes the "did I convert?" question grep-able.
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
    """Convert radians to degrees."""
    return np.asarray(x, dtype=float) * (180.0 / np.pi)


__all__ = ["km_to_m", "m_to_km", "deg_to_rad", "rad_to_deg"]
