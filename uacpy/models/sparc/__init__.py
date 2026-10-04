"""
SPARC - Seismo-Acoustic Propagation in Realistic oCeans

SPARC is a time-domain FFP (Fast Field Program) model using the same wavenumber
integration approach as Scooter. The vacuum / rigid restriction is the BINARY's,
not the wrapper's: ``Scooter/sparc.f90:100-103`` refuses any other boundary
condition itself ("SPARC only allows Vacuum or Rigid boundary conditions"), and
although the mesh tabulator declares a shear array and asks ``EvaluateSSP`` to
fill it (``:193``, ``:211``), only ``cp`` ever enters the march — ``c2R`` /
``c2I`` at ``:213-224`` are built from the compressional speed alone and ``cs``
is read nowhere else in the file. So SPARC is a fluid model, and a half-space
bottom is refused with a ``ConfigurationError`` rather than replaced: a rigid
floor in its place is a different waveguide (7-10 dB louder over 1-9 km on
the default seabed, up to 34 dB on clay, measured with Scooter).
"""

from uacpy.models.sparc._extract import rts_to_pressure
from uacpy.models.sparc._model import SPARC
from uacpy.models.sparc._settings import SparcSettings

__all__ = ['SPARC', 'SparcSettings', 'rts_to_pressure']
