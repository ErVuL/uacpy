"""
Acoustic propagation models
"""

from uacpy.models.base import PropagationModel, RunMode
from uacpy.models.bellhop import Bellhop, BellhopCUDA, Bellhop3D
from uacpy.models.ram import RAM  # mpiramS Fortran PE backend
from uacpy.models.kraken import Kraken, KrakenC, KrakenField
from uacpy.models.bounce import Bounce
from uacpy.models.scooter import Scooter
from uacpy.models.sparc import SPARC
from uacpy.models.oases import OASES, OAST, OASN, OASR, OASP

__all__ = [
    # Base types
    'PropagationModel',
    'RunMode',
    # Models
    'Bellhop',
    'BellhopCUDA',
    'Bellhop3D',  # stub — raises NotImplementedError on construction

    'RAM',
    'Kraken',      # Normal modes (real)
    'KrakenC',     # Complex modes (elastic)
    'KrakenField', # Field from modes (supports mode_coupling='adiabatic'/'coupled')
    'Bounce',      # Reflection coefficient computation
    'Scooter',
    'SPARC',
    'OASES',  # Unified interface
    'OAST',   # Individual models
    'OASN',
    'OASR',
    'OASP',
]
