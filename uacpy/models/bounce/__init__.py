"""
BOUNCE - Reflection Coefficient Computation Module

BOUNCE computes reflection coefficients for a stack of acoustic/elastic layers.
Part of the Acoustics Toolbox (OALIB).

Outputs:
- .BRC file: Bottom Reflection Coefficient (BotOpt 'F')
  -> Read by BELLHOP (``Bellhop/bellhop.f90:136`` loads the table, ``:688-693``
     applies it in the ``Reflect2D`` contained in that same file;
     ``Bellhop/ReflectMod.f90`` holds a near-identical ``Reflect2D`` that only
     bellhop3D links, per ``Bellhop/Makefile:4,8``), SCOOTER, KRAKENC
- .IRC file: Internal Reflection Coefficient (BotOpt 'P')
  -> Read by KRAKENC (``Kraken/BCImpedancecMod.f90:105``), SCOOTER
     (``Scooter/scooter.f90:357``) and KRAKEL; BELLHOP has no 'P' branch at
     all. Real KRAKEN parses it (``Kraken/BCImpedanceMod.f90:118``) but its
     mode search runs with ``ComplexFlag = .FALSE.``, which discards the
     table for a rigid boundary (:121-125) — uacpy therefore routes an
     ``.irc``/``.brc`` environment to krakenc.exe.

Note: SPARC does not support reflection coefficient files.
"""

from uacpy.models.bounce._model import Bounce
from uacpy.models.bounce._settings import BounceSettings

__all__ = ['Bounce', 'BounceSettings']
