"""
The knob defaults an engine and its deck writer share, each stated once.

An engine constructor defaults to the constant; its writer in
:mod:`uacpy.io` defaults to ``None`` and reads the same constant, so a deck
written by hand states what the engine would run. Below both
:mod:`uacpy.models` and :mod:`uacpy.io`, so either may import it. A knob
whose engine default is ``None`` (resolved from the inputs at run time) is
not listed: both sides call one resolver instead.
"""

__all__ = [
    'BELLHOP_BEAM_TYPE',
    'BELLHOP_N_BEAMS',
    'BELLHOP_LAUNCH_ANGLES',
    'BELLHOP_RAY_STEP',
    'BELLHOP_GRID_TYPE',
    'BELLHOP_INTERP_BATHYMETRY',
    'BELLHOP_INTERP_ALTIMETRY',
    'BELLHOP_BEAM_SHIFT',
    'KRAKEN_N_MESH',
    'SCOOTER_N_MESH',
    'SPARC_N_MESH',
    'SPARC_OUTPUT_MODE',
    'OAST_INTEGRATION_OFFSET',
    'OAST_RANGE_MIN',
    'OAST_VREC',
    'OASN_INTEGRATION_OFFSET',
    'OASP_INTEGRATION_OFFSET',
    'OASP_FREQ_MIN',
    'OASP_N_TIME_SAMPLES',
    'OASR_ANGLE_TYPE',
    'OASR_REFLECTION_TYPE',
    'OASSP_SPECTRAL_EXPONENT',
    'OASSP_REALIZATION',
    'OASSP_INTEGRATION_OFFSET',
]

#: Bellhop's beam type: geometric hat, Cartesian (BELLHOP's own default).
BELLHOP_BEAM_TYPE = 'G'

#: Bellhop's beam count: 0 lets Bellhop pick it.
BELLHOP_N_BEAMS = 0

#: Bellhop's launch-angle limits (deg).
BELLHOP_LAUNCH_ANGLES = (-80, 80)

#: Bellhop's ray step (m): 0 resolves from the depth.
BELLHOP_RAY_STEP = 0.0

#: Bellhop's receiver grid: rectilinear.
BELLHOP_GRID_TYPE = 'R'

#: Bellhop's ``.bty`` interpolation.
BELLHOP_INTERP_BATHYMETRY = 'linear'

#: Bellhop's ``.ati`` interpolation.
BELLHOP_INTERP_ALTIMETRY = 'linear'

#: Bellhop's beam shift on boundary reflection: off.
BELLHOP_BEAM_SHIFT = False

#: Kraken's mesh points per medium: 0 lets it size each.
KRAKEN_N_MESH = 0

#: Scooter's mesh points per medium: 0 lets it size each.
SCOOTER_N_MESH = 0

#: SPARC's mesh points per medium: 0 lets it size each.
SPARC_N_MESH = 0

#: SPARC's output: a horizontal array of receivers.
SPARC_OUTPUT_MODE = 'R'

#: OAST's wavenumber-contour offset (dB/wavelength). 0 is not "no offset":
#: under 'J', OASES applies its own default to any value below 1e-10.
OAST_INTEGRATION_OFFSET = 0.0

#: OAST's first output range (m).
OAST_RANGE_MIN = 0.0

#: OAST's receiver speed (m/s).
OAST_VREC = 0.0

#: OASN's wavenumber-contour offset (dB/wavelength). 0 is not "no offset":
#: under 'J', OASES applies its own default to any value below 1e-10.
OASN_INTEGRATION_OFFSET = 0.0

#: OASP's wavenumber-contour offset (dB/wavelength). 0 is not "no offset":
#: under 'J', OASES applies its own default to any value below 1e-10.
OASP_INTEGRATION_OFFSET = 0.0

#: OASP's lower band edge (Hz).
OASP_FREQ_MIN = 0.0

#: OASP's FFT length when neither ``n_time_samples`` nor a requested band sets it.
OASP_N_TIME_SAMPLES = 4096

#: OASR's angle convention.
OASR_ANGLE_TYPE = 'grazing'

#: OASR's coefficient: P-P, OASES' own default.
OASR_REFLECTION_TYPE = 'P-P'

#: OASSP's roughness power-spectrum exponent.
OASSP_SPECTRAL_EXPONENT = 2.0

#: OASSP's realization index (the seed offset).
OASSP_REALIZATION = 0

#: OASSP's wavenumber-contour offset (dB/wavelength). 0 is not "no offset":
#: under 'J', OASES applies its own default to any value below 1e-10.
OASSP_INTEGRATION_OFFSET = 0.0
