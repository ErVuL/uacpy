"""
The physics reference every layer of uacpy reads.

The reference sea water every default describes, the no-energy floor and its
dB marker, the dB reference pressures and the neper-to-dB factor. A constant
with a narrower owner lives with it: deck-format limits in
:mod:`uacpy.core.deck_limits`, the boundary types in :mod:`uacpy.core.boundary`,
the Earth radius in :mod:`uacpy.core.geo`,
model defaults in ``uacpy.models._defaults``, Acoustics-Toolbox letters in
:mod:`uacpy.io.at_codes`.
"""

import math

__all__ = [
    'REFERENCE_TEMPERATURE_C', 'REFERENCE_SALINITY_PSU', 'REFERENCE_DEPTH_M',
    'REFERENCE_PH', 'PH_MIN', 'PH_MAX', 'DEFAULT_SOUND_SPEED',
    'DEFAULT_WATER_DENSITY_G_CM3', 'WATER_DENSITY_MIN_G_CM3',
    'WATER_DENSITY_MAX_G_CM3',
    'PRESSURE_FLOOR', 'NO_ENERGY_DB',
    'REFERENCE_PRESSURE_WATER', 'REFERENCE_PRESSURE_AIR', 'NEPER_TO_DB',
    'STANDARD_GRAVITY_M_S2', 'STANDARD_ATMOSPHERE_PA',
]


# The reference sea water every seawater-formula default describes: 10 °C,
# 35 psu, at the surface (0 m, one atmosphere), pH 8.0. One state for every
# function, so a default density, sound speed, absorption and bubble
# resonance all describe the same water.
REFERENCE_TEMPERATURE_C = 10.0
REFERENCE_SALINITY_PSU = 35.0
REFERENCE_DEPTH_M = 0.0
REFERENCE_PH = 8.0

# The pH scale: a value outside it is no water's.
PH_MIN = 0.0
PH_MAX = 14.0

# m/s — the field's NOMINAL water sound speed (the Jensen et al. benchmarks,
# the Acoustics-Toolbox test decks). Deliberately not the reference water's
# own speed (TEOS-10 gives 1489.8 m/s there): it is the conventional value
# every function that needs "a" water sound speed defaults to.
DEFAULT_SOUND_SPEED = 1500.0

# g/cm³ — the reference sea water's density: IES-80 (Fofonoff 1985,
# ``core.acoustics.density``) gives 1026.95 kg/m³ at 10 °C, 35 psu, one
# atmosphere. The ONE water density of the package: what every deck writes
# for the water column unless ``Environment(water_density=...)`` says
# otherwise, what every standalone default reads, and the water a published
# seabed density RATIO (Jensen et al. Table 1.3, Hamilton, APL-UW) is
# multiplied by to become an absolute density — so a default run sees
# exactly the published ratio.
DEFAULT_WATER_DENSITY_G_CM3 = 1.027

# g/cm³ — the band a water density must lie in. Sea water is about
# 1.027; a value in kg/m³ lands a factor 1000 outside it.
WATER_DENSITY_MIN_G_CM3 = 0.9
WATER_DENSITY_MAX_G_CM3 = 1.1

# Floor applied whenever we take 20*log10(|p|), and the single no-energy level
# uacpy reports: 1e-30 lands at 600 dB of loss, far past anything a real
# field reaches, so a cell carrying no energy is never mistakable for one
# that does. Wrappers leave such a sample at zero and
# let this floor speak for it rather than writing a level of their own.
PRESSURE_FLOOR = 1e-30

# dB — where PRESSURE_FLOOR lands on a dB axis, 600. A level of this
# magnitude is the no-energy MARKER, not a level: nothing a model computes
# reaches it. Read by magnitude, because a loss view puts the marker at +600
# and a level view (-TL) at -600. ``core.acoustics.no_energy_mask`` is the
# one predicate every metric and plotter tests it with.
NO_ENERGY_DB = abs(20.0 * math.log10(PRESSURE_FLOOR))

# SPL reference pressures for dB conversion (levels are dB re ref²).
# Underwater acoustics references 1 µPa; in-air references 20 µPa.
REFERENCE_PRESSURE_WATER = 1e-6  # Pa (1 µPa)
REFERENCE_PRESSURE_AIR = 2e-5    # Pa (20 µPa)

# Exact nepers → dB conversion (20/ln10 ≈ 8.6858896).
NEPER_TO_DB = 20.0 / math.log(10.0)

# m/s² — standard gravity (CGPM 1901, exact).
STANDARD_GRAVITY_M_S2 = 9.80665

# Pa — the standard atmosphere (exact).
STANDARD_ATMOSPHERE_PA = 101325.0
