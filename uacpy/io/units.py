"""Unit-conversion helpers used at the boundary between the public uacpy
API (metres / Hz / radians) and the native Acoustics-Toolbox / OASES /
RAM file formats (which variously expect km, kHz, or degrees).

Every writer that emits a km-on-disk axis goes through ``m_to_km``;
every reader that returns a metres-API axis goes through ``km_to_m``.
Same for phase columns via ``deg_to_rad`` on the way in and ``rad_to_deg``
on the way out. Centralising the conversions makes the "did I convert?"
question grep-able.

The four functions are defined in :mod:`uacpy.core.units` — they are pure
arithmetic with no file format in them, and layers above ``io`` label axes
with the same conversions — and re-exported here under their own names, so
``from uacpy.io.units import m_to_km`` resolves to the one definition.
"""

from __future__ import annotations

from uacpy.core.units import deg_to_rad, km_to_m, m_to_km, rad_to_deg

__all__ = ["km_to_m", "m_to_km", "deg_to_rad", "rad_to_deg"]
