"""
The knob defaults an engine and its deck writer share, each stated once.

An engine constructor defaults to the constant; its writer in
:mod:`uacpy.io` defaults to ``None`` and reads the same constant, so a deck
written by hand states what the engine would run. Below both
:mod:`uacpy.models` and :mod:`uacpy.io`, so either may import it. A knob
whose engine default is ``None`` (resolved from the inputs at run time) is
not listed: both sides call one resolver instead.
"""

import math
from typing import NamedTuple

__all__ = [
    'C_LOW_FACTOR',
    'ScholteWave',
    'scholte_waves',
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
#: their deck writers) is this factor on the slowest wave speed of the whole
#: problem (:func:`default_c_low`): ``scooter.f90:258``, ``cMin = 0.90 * cMin
#: ! phase speed must be some 10% below slowest acoustic speed``. Jensen et
#: al. (*Computational Ocean Acoustics*, ch. 4 App. 1): "the highest
#: significant spatial frequency in the horizontal corresponds to the lowest
#: wave speed in the problem. It is, therefore, in general, conservative to
#: choose k_max ≃ 1.1 ω/c_min" (0.91·c_min). ``scooter.f90:193-259`` takes
#: cMin over every acoustic medium's cP (:209-210), every elastic medium's
#: cS (:216-217) and the acoustic half-spaces (cS when elastic, :236-253),
#: and over an elastic half-space writes 0.85 instead, "phase speed a bit
#: less than Scholte/Stoneley wave speed" (:255-256) — an estimate of the
#: Scholte wave, which falls short when the shear speed is close to the
#: water's (1240 m/s = 0.83 × 1500 m/s on the limestone preset). uacpy
#: solves that wave (:func:`scholte_waves`) and counts it among the
#: speeds, so the one factor applies to fluid and elastic problems alike.
#: scooter.f90 computes cMin and does not apply it (:261-263,
#: ``!!!cLow = MAX( cLow, cMin )``). OASES states no factor; its worked decks
#: pin CMIN by hand (oast.tex:568, oasn.tex:479).
C_LOW_FACTOR = 0.90


class ScholteWave(NamedTuple):
    """A Scholte wave of an environment: its phase speed
    (m/s), the depth (m) of the interface it travels along, and which two
    media meet there."""
    speed: float
    depth: float
    where: str

    def __repr__(self) -> str:
        from uacpy.core._repr import fields_repr
        return fields_repr(self, units={'speed': 'm/s'})


def scholte_waves(env):
    """The Scholte waves of ``env``, one
    :class:`ScholteWave` per liquid medium touching an elastic one: the
    water on an elastic seabed or under an elastic top half-space, a fluid
    sediment layer on an elastic layer or half-space, an elastic layer on a
    fluid one. Each speed is
    :func:`~uacpy.core.acoustics.boundaries.scholte_wave_speed` for the
    two media taken as half-spaces (Brekhovskikh & Lysanov §3.4.2,
    eq. 3.4.29), with the water's slowest speed at its deepest sample
    against the seabed and at its shallowest against the top. Two elastic
    media in contact are not
    a liquid-solid pair and give none. The seafloor is ``env.depth``; a
    sediment interface lies the thicknesses of the layers above it deeper.
    Every seabed column counts."""
    import numpy as np
    from uacpy.core.acoustics.boundaries import scholte_wave_speed
    from uacpy.core.boundary import BoundaryType

    ssp = np.atleast_2d(np.asarray(env.ssp.sound_speed, dtype=float))
    rho_w = float(env.water_density)
    waves = []

    def meet(upper, lower, depth, where):
        liquid, solid = (upper, lower) if lower[1] > 0.0 else (lower, upper)
        if liquid[1] > 0.0 or solid[1] <= 0.0:
            return
        speed = scholte_wave_speed(solid[0], solid[1], solid[2],
                                   water_sound_speed=liquid[0],
                                   water_density=liquid[2])
        if not math.isnan(speed):
            waves.append(ScholteWave(speed, depth, where))

    def medium(props):
        return (float(props.sound_speed), float(props.shear_speed or 0.0),
                float(props.density))

    for column in env.bottom.columns:
        upper = (float(np.nanmin(ssp[-1])), 0.0, rho_w)
        upper_name, depth = 'the water', float(env.depth)
        stack = [(layer, 'a sediment layer', float(layer.thickness))
                 for layer in column.layers]
        if BoundaryType.from_string(
                column.halfspace.acoustic_type).is_geoacoustic:
            stack.append((column.halfspace, 'the bottom half-space', 0.0))
        for props, name, thickness in stack:
            meet(upper, medium(props), depth,
                 f"between {upper_name} and {name}")
            upper, upper_name = medium(props), name
            depth += thickness
    for top in env.surface.nodes:
        if BoundaryType.from_string(top.acoustic_type).is_geoacoustic:
            meet(medium(top), (float(np.nanmin(ssp[0])), 0.0, rho_w), 0.0,
                 'between the top half-space and the water')
    return waves


def default_c_low(env):
    """``(c_low, origin)``: the slow phase-speed edge (m/s) an unpinned
    ``c_low`` takes on ``env``, and the sentence stating where it came from.

    :data:`C_LOW_FACTOR` × the slowest wave speed of the problem: the
    water's slowest sound speed; the compressional speed of each sediment
    layer and of each 'half-space' boundary at the bottom and the top, and
    its shear speed wherever that is non-zero; and every Scholte wave a liquid
    medium carries against a solid one
    (:func:`scholte_waves`, Brekhovskikh & Lysanov §3.4.2, eq. 3.4.29).
    Vacuum, rigid and reflection-table boundaries carry no medium
    (scooter.f90 reads only ``BC == 'A'``). Every seabed column counts, so
    on a range-dependent ``env`` it is the slowest wave along the track.
    A Stoneley wave (between two solids) is not solved and does
    not count."""
    import numpy as np
    from uacpy.core.boundary import BoundaryType

    speeds = [(float(np.nanmin(np.asarray(env.ssp.sound_speed, dtype=float))),
               'the water')]

    def medium(props, name):
        speeds.append((float(props.sound_speed), name))
        if float(props.shear_speed or 0.0) > 0.0:
            speeds.append((float(props.shear_speed), f"{name}'s shear speed"))

    for column in env.bottom.columns:
        for layer in column.layers:
            medium(layer, "a sediment layer")
        if BoundaryType.from_string(
                column.halfspace.acoustic_type).is_geoacoustic:
            medium(column.halfspace, "the bottom half-space")
    for top in env.surface.nodes:
        if BoundaryType.from_string(top.acoustic_type).is_geoacoustic:
            medium(top, "the top half-space")
    c_min, where = min(speeds, key=lambda s: s[0])
    rule = (f"{C_LOW_FACTOR:g} × {c_min:g} m/s, {where} — the slowest speed "
            f"of every medium (scooter.f90:193-259)")
    waves = scholte_waves(env)
    if waves:
        wave = min(waves, key=lambda w: w.speed)
        if wave.speed < c_min:
            return C_LOW_FACTOR * wave.speed, (
                f"{C_LOW_FACTOR:g} × {wave.speed:.1f} m/s, the Scholte "
                f"wave {wave.where} (Brekhovskikh & Lysanov "
                f"§3.4.2, eq. 3.4.29; scooter.f90:258), slower than every "
                f"medium (the slowest: {where}, {c_min:g} m/s)")
    return C_LOW_FACTOR * c_min, rule

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
