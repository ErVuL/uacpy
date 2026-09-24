"""Acoustic propagation models."""

from uacpy.models.base import PropagationModel, RunMode, ModelSpec
from uacpy.models.bellhop import Bellhop
from uacpy.models.ram import RAM
from uacpy.models.kraken import Kraken
from uacpy.models.bounce import Bounce
from uacpy.models.scooter import Scooter
from uacpy.models.sparc import SPARC
from uacpy.models.oases import OASES, OAST, OASN, OASR, OASP, OASS, OASSP

# PE grid quality, from models/_pade_optimizer.py. These answer the
# questions a RAM user has to ask before trusting a run — is this
# (dr, dz) accurate here, what c0 should the PE expand about, what
# step is the rotated Crank-Nicolson stable at, how much does the
# seabed leak — and they took only plain numbers, so they are
# reachable rather than reported as warning text.
from uacpy.models._pade_optimizer import (
    numerov_error,
    combined_error,
    optimal_c0,
    optimize_grid,
    grid_error,
    rams_dz_shear_cap,
    rotated_pade_coefficients,
    rotated_cn_growth,
    rotated_growth_floor,
    seabed_leak_rate,
    rams_growth_margin,
    rams_stable_dr,
    rams_stable_theta,
)

__all__ = [
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
    'PropagationModel',
    'RunMode',
    'ModelSpec',
    'Bellhop',
    'RAM',
    'Kraken',
    'Bounce',
    'Scooter',
    'SPARC',
    'OASES',
    'OAST',
    'OASN',
    'OASR',
    'OASP',
    'OASSP',
    'OASS',
    # submodules
    'base',
    'bellhop',
    'bounce',
    'kraken',
    'oases',
    'ram',
    'scooter',
    'sources',
    'sparc',
]
