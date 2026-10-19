"""
Defaults the propagation-model wrappers resolve an unset knob to.

Read by the engines only; the carriers and io never consult them.
"""


# m/s — the cap on BOUNCE's derived c_low, min(DEFAULT_C_MIN, min(env.ssp)):
# doc/bounce.htm sets CMin to "the lowest speed in the problem (say
# 1400.0)", and its example deck writes 1400.0.
DEFAULT_C_MIN = 1400.0

# Broadband-mode auto-generated frequency grid: when the user runs a
# broadband-capable wrapper (Bellhop, Scooter, Kraken, RAM, OASP)
# without an explicit ``frequencies=`` override, the wrapper picks
# ``N`` bins linearly spaced over ``[fc·(1 - BW/2), fc·(1 + BW/2)]``
# where ``fc = source.frequencies[0]`` (BW below 2, so the band stays
# above 0 Hz); any other band is the run's ``frequencies=``.
# Default BW=0.5, uacpy guidance: the Bellhop User Guide §9 recommends
# dividing a wide band into sub-bands "so that the appropriate attenuation
# is used in each sub-band" and names no width.
# N=128, uacpy guidance (no value in the manuals): the bins are 1/T apart,
# T the alias-free time window (Jensen et al., *Computational Ocean
# Acoustics*, Sect. 8.2.1.2), so 128 bins over 0.5·fc give T = 254/fc.
DEFAULT_BROADBAND_N_FREQUENCIES = 128
DEFAULT_BROADBAND_BANDWIDTH_FACTOR = 0.5
