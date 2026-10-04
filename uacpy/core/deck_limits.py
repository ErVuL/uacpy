"""
Limits the Acoustics-Toolbox deck formats set on the values a carrier admits.

The carriers refuse a value the decks cannot carry faithfully (an attenuation
AT's ``CRCI`` would reject, two samples one printed column cannot tell apart),
so these limits are checked at construction, before any engine is chosen.
"""

import math

__all__ = [
    'AT_DB_PER_NEPER', 'MAX_ATTENUATION_DB_PER_WAVELENGTH',
    'AT_LAST_SSP_POINT_EPS_M',
    'DECK_AXIS_DECIMALS', 'DECK_DEPTH_FMT', 'DECK_DEPTH_RESOLUTION_M',
    'DECK_RANGE_RESOLUTION_M', 'SBP_ANGLE_RESOLUTION_DEG',
    'NO_RECEIVER_RANGE_FALLBACK_M',
]


# Every AT solver converts attenuation through ``misc/AttenMod.f90``'s ``CRCI``,
# and uacpy writes ``AttenUnit = 'W'`` (dB/wavelength) in every deck. Substituting
# that branch (:73, ``alphaT = alpha*freq/(8.6858896*c)``) into the conversion to an
# imaginary sound speed (:113, ``alphaT = alphaT*c*c/omega``) with
# ``omega = 2*pi*freq`` leaves ``alphaT_imag = alpha*c/(8.6858896*2*pi)``, so the
# fatal test at :116 (``alphaT > c``) reduces to a bound on alpha alone —
# independent of frequency and of sound speed. CRCI is reached for every water
# sample (``misc/sspMod.f90`` UpdateSSPLoss), both half-spaces (UpdateHSLoss) and
# Bellhop's own (``ReadEnvironmentBell.f90``), so the bound is package-wide.
#: The dB-per-neper factor as ``misc/AttenMod.f90`` writes it, truncated
#: (the exact one is :data:`~uacpy.core.constants.NEPER_TO_DB`).
AT_DB_PER_NEPER = 8.6858896
MAX_ATTENUATION_DB_PER_WAVELENGTH = AT_DB_PER_NEPER * 2.0 * math.pi   # 54.575

# ``misc/sspMod.f90:353`` ends a medium's SSP block at the first sample within
# ``100*EPSILON(1.0e0)`` of the declared medium depth — an absolute tolerance in
# metres, single precision, not scaled by depth. A second sample inside that window
# is never read as an SSP row: the next READ consumes it as the bottom-option
# record (``misc/ReadEnvironmentMod.f90``), so the boundary condition is taken from
# a sound speed.
AT_LAST_SSP_POINT_EPS_M = 100.0 * 2.0 ** -23          # 1.1920929e-05 m

# Resolution at which the AT decks print their axes: depths in metres and ranges in
# kilometres, both at ``%.6f``. Two samples closer than this collapse to one token.
# On a range axis the readers then reject it outright — ``misc/sspMod.f90:342``,
# ``Bellhop/bdryMod.f90:132``/``:231``, ``misc/SourceReceiverPositions.f90:163``,
# with ``misc/monotonicMod.f90`` strict (``<=`` fails). On a source/receiver *depth*
# axis the equivalent ERROUTs are commented out
# (``SourceReceiverPositions.f90:142``/``:146``), so the collapse is silent instead:
# the deck simply carries the same depth twice.
#: Decimals each axis is printed at. One number, so the carriers' admission
#: rule below and the writers' column format (``io/oalib_writer.py``, which
#: imports these) are the same statement rather than two that agree today.
DECK_AXIS_DECIMALS = 6
DECK_DEPTH_RESOLUTION_M = 10.0 ** -DECK_AXIS_DECIMALS            # metres at %.6f
DECK_RANGE_RESOLUTION_M = 1.0e3 * 10.0 ** -DECK_AXIS_DECIMALS    # km at %.6f
#: The format spec every deck writes a depth with.
DECK_DEPTH_FMT = f'.{DECK_AXIS_DECIMALS}f'

# The ``.sbp`` source beam pattern writes its angle column at ``%12.6f``
# (``io/refl_io.py``), so two angles closer than this land on the same token.
# Bellhop's load-time guard (``misc/beampattern.f90:56`` via
# ``misc/monotonicMod.f90``) is strict, so the collapsed pair aborts the run
# with ERROUT rather than degrading silently.
SBP_ANGLE_RESOLUTION_DEG = 1.0e-6

# m — the range a deck takes when no receiver range is positive:
# Bellhop's ray-box range and BOUNCE's RMax.
NO_RECEIVER_RANGE_FALLBACK_M = 10_000.0
