"""
Scooter finite-element FFP (Fast Field Program) model.

Computes the acoustic field in the frequency-wavenumber domain using a
finite-element discretization, then transforms ``.grn`` to a range-domain
TL field via the in-tree Python Hankel transform of
:class:`uacpy.core.results.GreensFunction`. Supports coherent TL, broadband ``H(f)``,
and broadband time-series output.
"""

from uacpy.models.scooter._model import Scooter
from uacpy.models.scooter._settings import ScooterSettings

__all__ = ['Scooter', 'ScooterSettings']
