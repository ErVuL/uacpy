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
    'C_LOW_FACTOR',
    'C_LOW_FACTOR_ELASTIC',
    'default_c_low',
    'BELLHOP_BEAM_TYPE',
    'BELLHOP_N_BEAMS',
    'BELLHOP_LAUNCH_ANGLES_DEG',
    'BELLHOP_RAY_STEP',
    'BELLHOP_GRID_TYPE',
    'BELLHOP_INTERP_BATHYMETRY',
    'BELLHOP_INTERP_ALTIMETRY',
    'BELLHOP_BEAM_SHIFT',
    'KRAKEN_N_MESH',
    'SCOOTER_N_MESH',
    'SPARC_N_MESH',
    'SPARC_OUTPUT_MODE',
    'OASES_PRESSURE_LETTER',
    'OAST_INTEGRATION_OFFSET',
    'OAST_RANGE_MIN',
    'OAST_VREC',
    'OASN_INTEGRATION_OFFSET',
    'OASP_INTEGRATION_OFFSET',
    'OASP_FREQ_MIN',
    'OASP_N_TIME_SAMPLES',
    'OASP_INTEGRAND_PLOT_STEP',
    'OASR_ANGLE_TYPE',
    'OASR_REFLECTION_TYPE',
    'OASSP_SPECTRAL_EXPONENT',
    'OASSP_REALIZATION',
    'OASSP_INTEGRATION_OFFSET',
]

#: The slow phase-speed edge every wavenumber-integration engine derives when
#: ``c_low`` is unpinned (Scooter, SPARC, OAST, OASP, OASS, OASSP, OASN and
#: their deck writers) is a factor on the slowest speed of the whole problem
#: (:func:`default_c_low`): this factor, or :data:`C_LOW_FACTOR_ELASTIC` when
#: a half-space is elastic. ``scooter.f90:193-259`` takes cMin over every
#: acoustic medium's cP (:209-210), every elastic medium's cS (:216-217) and
#: the acoustic half-spaces (cS when elastic, :236-253), then ``cMin = 0.90 *
#: cMin   ! phase speed must be some 10% below slowest acoustic speed``
#: (:258), or 0.85 "a bit less than Scholte/Stoneley wave speed" when a
#: half-space is elastic (:255-256) — and does not apply it (:261-263,
#: ``!!!cLow = MAX( cLow, cMin )``). Jensen et al. (*Computational Ocean
#: Acoustics*, ch. 4 App. 1): "the highest significant spatial frequency in
#: the horizontal corresponds to the lowest wave speed in the problem. It is,
#: therefore, in general, conservative to choose k_max ≃ 1.1 ω/c_min"
#: (0.91·c_min). OASES states no factor; its worked decks pin CMIN by hand
#: (oast.tex:568, oasn.tex:479).
C_LOW_FACTOR = 0.90

#: The factor on the slowest speed when a half-space is elastic:
#: ``scooter.f90:255-256``, "phase speed a bit less than Scholte/Stoneley
#: wave speed" — the interface wave travels at "about 0.85 times the minimum
#: shear-wave speed" (Jensen et al., *Computational Ocean Acoustics*,
#: §5.10.5).
C_LOW_FACTOR_ELASTIC = 0.85


def default_c_low(env):
    """``(c_low, origin)``: the slow phase-speed edge (m/s) an unpinned
    ``c_low`` takes on ``env``, and the sentence stating where it came from.

    The rule of ``scooter.f90:193-259`` (see :data:`C_LOW_FACTOR`): the
    slowest speed of every medium — the water, each sediment layer, each
    'half-space' boundary at the bottom and the top, a shear speed counted
    wherever it is non-zero — times :data:`C_LOW_FACTOR_ELASTIC` when a
    half-space is elastic, :data:`C_LOW_FACTOR` otherwise. An elastic
    medium contributes ``min(cP, cS)``, which is scooter.f90's ``cS``
    whenever the shear speed is below the compressional one. Vacuum, rigid
    and reflection-table boundaries carry no medium (scooter.f90 reads only
    ``BC == 'A'``). Every seabed column counts, so on a range-dependent
    ``env`` it is the slowest speed along the track."""
    import numpy as np
    from uacpy.core.boundary import BoundaryType

    speeds = [(float(np.nanmin(np.asarray(env.ssp.sound_speed, dtype=float))),
               'the water')]
    elastic_halfspace = False

    def medium(props, name):
        cs = float(props.shear_speed or 0.0)
        if cs > 0.0:
            speeds.append((min(float(props.sound_speed), cs),
                           f"{name}'s shear speed"))
        else:
            speeds.append((float(props.sound_speed), name))
        return cs > 0.0

    for column in env.bottom.columns:
        for layer in column.layers:
            medium(layer, "a sediment layer")
        if BoundaryType.from_string(
                column.halfspace.acoustic_type).is_geoacoustic:
            elastic_halfspace |= medium(column.halfspace,
                                        "the bottom half-space")
    for top in env.surface.nodes:
        if BoundaryType.from_string(top.acoustic_type).is_geoacoustic:
            elastic_halfspace |= medium(top, "the top half-space")
    c_min, where = min(speeds, key=lambda s: s[0])
    factor = C_LOW_FACTOR_ELASTIC if elastic_halfspace else C_LOW_FACTOR
    return factor * c_min, (
        f"{factor:g} × {c_min:g} m/s, {where} — the slowest speed of every "
        f"medium{', over an elastic half-space' if elastic_halfspace else ''}"
        f" (scooter.f90:193-259)")

#: Bellhop's beam type: geometric hat, Cartesian (BELLHOP's own default:
#: doc/bellhop.htm OPTION(2:2) "'G' Geometric hat beams in Cartesian
#: coordinates (default)"; ReadEnvironmentBell.f90:393-395).
BELLHOP_BEAM_TYPE = 'G'

#: Bellhop's beam count: 0 lets Bellhop pick it (doc/bellhop.htm NBEAMS: "use
#: 0 to have the program calculate a value automatically, but
#: conservatively"; angleMod.f90:38-51).
BELLHOP_N_BEAMS = 0

#: Bellhop's launch-angle limits (deg). uacpy guidance (no value in the
#: manual): the Bellhop User Guide leaves the fan to "what part of the field
#: the user is interested in seeing"; a receiver whose direct path needs a
#: steeper launch is warned about (``fan_miss_notice``).
BELLHOP_LAUNCH_ANGLES_DEG = (-80, 80)

#: The ray step (m) a Bellhop deck carries when none is given: 0, with which
#: BELLHOP chooses its own depth/10 (bellhop.f90:170-174). ``Bellhop()``'s own
#: default is None, env.depth / 50.
BELLHOP_RAY_STEP = 0.0

#: Bellhop's receiver grid: rectilinear (doc/bellhop.htm OPTION(5:5): "'R'
#: rectilinear grid (default)").
BELLHOP_GRID_TYPE = 'R'

#: Bellhop's ``.bty`` interpolation. Piecewise linear, as doc/ATI_BTY_File.htm
#: advises for both boundary files: "Generally the piecewise linear option
#: should be selected."
BELLHOP_INTERP_BATHYMETRY = 'linear'

#: Bellhop's ``.ati`` interpolation (same advice as the ``.bty``).
BELLHOP_INTERP_ALTIMETRY = 'linear'

#: Bellhop's beam shift on boundary reflection: off, BELLHOP's own default
#: (ReadEnvironmentBell.f90:159-165 shifts only for RunType(7:7) = 'S').
BELLHOP_BEAM_SHIFT = False

#: Kraken's mesh points per medium: 0 lets it size each (doc/
#: EnvironmentalFile.htm: "If you type 0 for the number of mesh points, the
#: code will calculate NMESH automatically"; ReadEnvironmentMod.f90:103-105,
#: 20 points per wavelength, at least 10).
KRAKEN_N_MESH = 0

#: Scooter's mesh points per medium: 0 lets it size each (as Kraken's, then
#: scaled with frequency and floored at 100 points, scooter.f90:106-110).
SCOOTER_N_MESH = 0

#: SPARC's mesh points per medium: 0 lets it size each (sparc.f90:85 reads
#: the deck through ReadEnvironmentMod, whose ReadEnvironmentMod.f90:103-105
#: take 20 points per wavelength, at least 10).
SPARC_N_MESH = 0

#: SPARC's output: a horizontal array of receivers. uacpy guidance (no value
#: in the manual: doc/sparc.htm lists 'S', 'R' and 'D' and names no default).
SPARC_OUTPUT_MODE = 'R'

#: The output-parameter letter OAST, OASP and OASSP decks carry when no raw
#: option line is given: OASES' bulk pressure 'K' (GETOPT IOUT(6),
#: unoast31.f:1005-1010, unoasp22.f:909-914, unoassp30.f:922-927). In a fluid
#: the kernel sets it to minus the normal stress, the acoustic pressure
#: (oasaun31.f:1448-1449, :1471-1472); in a solid it is the bulk pressure of
#: the compressional potential alone (:1509-1513, "changed to pressure from
#: stress"), as oasp.tex:178-183 describes. oast.tex:161-165 and
#: oassp.tex:145-150 still call it the bulk *stress*, the negative of the
#: pressure; the vendored kernel is the pressure. The normal stress 'N' is
#: sigma_zz, -p in a fluid but built from both potentials in a solid
#: (oasaun31.f:1489-1491). OASN keeps 'N': its 'K' plots noise kernels.
OASES_PRESSURE_LETTER = 'K'

#: OAST's wavenumber-contour offset (dB/wavelength). 0 is not "no offset":
#: under 'J', OASES applies its own default to any value below 1e-10 — the
#: offset giving 60 dB at the longest FFT range, which oast.tex:293-303
#: recommends: "This value is the default which is applied if COFF is
#: specified to 0.0."
OAST_INTEGRATION_OFFSET = 0.0

#: OAST's first output range (m): 0, the manual's worked deck (oast.tex:489).
OAST_RANGE_MIN = 0.0

#: OAST's receiver speed (m/s). uacpy guidance (no value in the manual): a
#: stationary source and receiver; OAST reads the token only under option
#: 'd' (unoast31.f:125-127).
OAST_VREC = 0.0

#: OASN's wavenumber-contour offset (dB/wavelength). 0 is not "no offset":
#: under 'J', OASES applies its own default to any value below 1e-10.
OASN_INTEGRATION_OFFSET = 0.0

#: OASP's wavenumber-contour offset (dB/wavelength). 0 is not "no offset":
#: under 'J', OASES applies its own default to any value below 1e-10.
OASP_INTEGRATION_OFFSET = 0.0

#: OASP's lower band edge (Hz): 0, the manual's worked deck (oasp.tex:583).
OASP_FREQ_MIN = 0.0

#: OASP's FFT length when neither ``n_time_samples`` nor a requested band sets
#: it. uacpy guidance: OASES requires only ``NT = 2^M`` (oasp.tex:129); its
#: worked decks use 2048 (oasp.tex:583) and 1024 (oasp.tex:773).
OASP_N_TIME_SAMPLES = 4096

#: OASP's integrand-plot decimation ``INTF`` (Block VII): 0, the manual's
#: worked decks (oasp.tex:582; :772 writes 1). 0 turns the integrand plots and
#: the .plp/.plt plot files off (unoasp22.f:173); the .trf carries every bin
#: either way (:591).
OASP_INTEGRAND_PLOT_STEP = 0

#: OASR's angle convention: grazing, OASR's default axis (oasr.tex:151-152).
OASR_ANGLE_TYPE = 'grazing'

#: OASR's coefficient: P-P, OASES' own default (oasr.tex:138-140).
OASR_REFLECTION_TYPE = 'P-P'

#: The roughness power-spectrum exponent M an OASS / OASSP deck carries when
#: unset: OASES' own Goff-Jordan exponent when a roughness record carries none
#: (oaseun31.f:103-104 ``amod(m)=2.5``), the "very smooth variation" end of the
#: manual's realistic range 1.5 < M <= 2.5 (oassp.tex:356-358); inert for the
#: default Gaussian spectrum (oaseun31.f:3689-3692).
OASSP_SPECTRAL_EXPONENT = 2.5

#: OASSP's realization index (the seed offset): 0, as both of the manual's
#: worked OASSP decks (oassp.tex:604, :664).
OASSP_REALIZATION = 0

#: OASSP's wavenumber-contour offset (dB/wavelength). 0 is not "no offset":
#: under 'J', OASES applies its own default to any value below 1e-10.
OASSP_INTEGRATION_OFFSET = 0.0
