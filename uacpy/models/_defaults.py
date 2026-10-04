"""
Defaults the propagation-model wrappers resolve an unset knob to.

Read by the engines only; the carriers and io never consult them.
"""


DEFAULT_C_MIN = 1400.0   # below slowest expected water-column speed

# Broadband-mode auto-generated frequency grid: when the user runs a
# broadband-capable wrapper (Bellhop, Scooter, Kraken, RAM, OASP)
# without an explicit ``frequencies=`` override, the wrapper picks
# ``N`` bins linearly spaced over ``[fc·(1 - BW/2), fc·(1 + BW/2)]``
# (clipped to [1, ∞)) where ``fc = source.frequencies[0]``.
# Default BW=0.5 — Bellhop User Guide §9 recommends sub-banding for
# wide bandwidths because arrivals are computed at a single fc.
DEFAULT_BROADBAND_N_FREQS = 128
DEFAULT_BROADBAND_BANDWIDTH_FACTOR = 0.5
