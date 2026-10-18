"""Published range-dependent benchmarks — uacpy's engines measured against
reference solutions computed by an independent code for the standard cases of
the ocean-acoustics literature.

The rest of the benchmark suite checks the engines against closed forms
(``test_benchmarks_analytic.py``), against a mode-matching solution written in
this repository (``test_benchmarks_coupled_mode.py``) and against each other.
This module checks them against the cases the field uses to judge a
range-dependent model, with references from a code uacpy does not wrap.

Reference solutions
-------------------
COUPLE07 (R. B. Evans, December 2007 version, OALIB ``Modes/couple/couple07.zip``)
is the stepwise coupled-mode code behind the published reference curves of the
ASA benchmark wedge (Jensen & Ferla, JASA 87, 1499-1510, 1990). It was built
outside this repository (gfortran ``-std=legacy`` plus stubs for the Lahey
``TIMER``/``DATE`` intrinsics) and run on the test inputs it ships. Each table
under ``tests/data/benchmarks/`` holds the TL it printed, with a header naming
the problem, the run settings and the source. COUPLE carries no licence, so
only those generated tables are shipped — never its source or its figures.

Every reference here was computed with water of density 1.0 g/cm^3 (COUPLE's
RHOW, the OASES Bucker deck), so every environment below is built with
``water_density=RHO_WATER`` (1.0) rather than uacpy's default sea water
(1.027): the engines then solve the problem the reference solved, and the
bottom density contrasts are the published ones.

Tolerances
----------
Every engine test states the measured agreement and sets its bound from that
measurement plus a margin, and every one runs a NULL CONTROL: the same engine
on the range-independent environment the benchmark starts from, which must
land outside the bound. A bound that the null run would also pass would
measure only that the engine produces a plausible waveguide field.

Agreement is an RMS TL difference over a range window chosen per receiver,
plus the positions of the interference nulls the published figures show. The
windows start at 250 m, inside which the near field depends on how each code
starts its field and not on the range dependence under test.
"""
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.benchmark]

from uacpy import (Environment, SoundSpeedProfile, BoundaryProperties, Bathymetry,
                   Bottom, SeabedColumn, SedimentLayer,
                   Source, Receiver, RAM, Bellhop, Kraken, Scooter, OAST)
from uacpy.tests.test_benchmarks_analytic import RHO_WATER

_DATA = Path(__file__).parent / 'data' / 'benchmarks'


def _load_tl_table(name):
    """``(ranges, tl)`` of a reference table: ``tl[i]`` is the TL (dB) at
    the table's ``i``-th receiver depth over ``ranges``."""
    table = np.loadtxt(_DATA / name)
    return table[:, 0] * 1000.0, table[:, 1:].T


def _rms_dB(tl, ref, ranges, r_min, r_max):
    """RMS of ``tl - ref`` (dB) over ``r_min <= range <= r_max``."""
    in_window = (ranges >= r_min) & (ranges <= r_max)
    return float(np.sqrt(np.mean((tl[in_window] - ref[in_window]) ** 2)))


def _null_range(tl, ranges, r_min, r_max):
    """Range (m) of the TL maximum — the interference null — inside
    ``[r_min, r_max]``."""
    in_window = (ranges >= r_min) & (ranges <= r_max)
    return float(ranges[in_window][np.argmax(tl[in_window])])


# ── ASA benchmark upslope wedge, 25 Hz ──────────────────────────────────────
#
# Jensen & Ferla 1990; COA (Jensen, Kuperman, Porter & Schmidt, 2nd ed.)
# Sect. 6.7, Fig. 6.8. Isovelocity water 200 m deep at r = 0 shoaling linearly
# to the apex at 4 km over a lossy fluid bottom; source 100 m, receivers 30 m
# and 150 m. The reference is COUPLE07's Test Case 2 (45 modes, 200 range
# steps, single-scatter matching), ``asa_wedge_25hz_couple07.txt``.

WEDGE_TABLE = 'asa_wedge_25hz_couple07.txt'
WEDGE_FREQ = 25.0
WEDGE_ZS = 100.0
WEDGE_ZR = (30.0, 150.0)
WEDGE_D0 = 200.0
WEDGE_APEX_R = 4000.0
# The apex is written 0.1 m deep: Environment refuses a zero depth, and the
# Kraken segmenter rounds a segment depth to 0.1 m, so an apex shallower than
# 0.05 m becomes a zero-depth segment it cannot build. COUPLE writes 0.01 m;
# both are far below the 25 Hz cutoff depth, where no mode is trapped.
WEDGE_APEX_DEPTH = 0.1
# The 30 m receiver is in the water out to 3.4 km and the 150 m receiver out
# to 1.0 km; RAM and Bellhop return no samples below the seafloor, so the
# comparison stops 50 m short of each crossing.
WEDGE_WINDOWS = ((250.0, 3400.0), (250.0, 950.0))
# The two deep nulls of the 30 m curve in the reference table (checked by eye
# against COUPLE's own Test Case 2 plot), and the window each is searched in.
WEDGE_NULL_WINDOWS = ((1000.0, 1300.0), (1700.0, 2050.0))
WEDGE_REF_NULLS_M = (1160.0, 1870.0)


def _wedge_env(slope=True):
    """The ASA wedge; ``slope=False`` keeps the 200 m depth throughout and is
    the null control."""
    bottom = BoundaryProperties(acoustic_type='half-space', sound_speed=1700.0,
                                density=1.5, attenuation=0.5)
    end_depth = WEDGE_APEX_DEPTH if slope else WEDGE_D0
    return Environment(
        water_density=RHO_WATER,
        bathymetry=Bathymetry(ranges=np.array([0.0, WEDGE_APEX_R]),
                              depths=np.array([WEDGE_D0, end_depth])),
        ssp=SoundSpeedProfile.from_pairs([(0.0, 1500.0), (WEDGE_D0, 1500.0)]),
        bottom=bottom)


def _wedge_agreement(model, slope=True):
    """Run ``model`` on the wedge (or its null control) at the reference
    table's receivers and return ``(rms_per_receiver, nulls_at_30m, tl)``."""
    ranges, ref = _load_tl_table(WEDGE_TABLE)
    tl = model.compute_field(_wedge_env(slope),
                          Source(depths=WEDGE_ZS, frequencies=WEDGE_FREQ),
                          Receiver(depths=list(WEDGE_ZR), ranges=ranges)).dB
    tl = np.asarray(tl, dtype=float).reshape(ref.shape)
    rms = [_rms_dB(tl[i], ref[i], ranges, *WEDGE_WINDOWS[i]) for i in range(len(WEDGE_ZR))]
    nulls = [_null_range(tl[0], ranges, *w) for w in WEDGE_NULL_WINDOWS]
    return rms, nulls, tl


def test_the_asa_wedge_reference_table_places_the_published_nulls():
    """The shipped COUPLE07 table is the wedge of COA Fig. 6.8: 401 ranges
    from 10 m to 4.01 km, the two deep nulls of the 30 m curve at 1.16 and
    1.87 km, the 50-60 dB plateau between 1 and 3 km, and the cutoff past
    3 km where the field drains into the bottom (78.4 dB at 4 km). A table
    that loses a column, a row or its range unit fails here before any engine
    is measured against it.
    """
    ranges, ref = _load_tl_table(WEDGE_TABLE)
    assert ref.shape == (2, 401)
    assert ranges[0] == pytest.approx(10.0) and ranges[-1] == pytest.approx(4010.0)
    nulls = [_null_range(ref[0], ranges, *w) for w in WEDGE_NULL_WINDOWS]
    assert nulls == pytest.approx(WEDGE_REF_NULLS_M)
    plateau = (ranges >= 1000.0) & (ranges <= 3000.0)
    assert 50.0 < np.median(ref[0, plateau]) < 60.0
    assert ref[0, ranges == 4000.0][0] == pytest.approx(78.4, abs=0.1)


@pytest.mark.requires_binary
@pytest.mark.parametrize('backend', ['mpirams', 'ramgeo'])
def test_ram_matches_the_coupled_mode_reference_in_the_asa_wedge(backend):
    """RAM's two range-dependent fluid PEs reproduce COUPLE07's wedge.

    Measured on the pinned grid below: RMS 0.110 dB at 30 m and 0.133 dB at
    150 m (mpirams), 0.106 and 0.134 dB (ramgeo), with both nulls of the 30 m
    curve on the reference sample. Refining to dr 2 m, dz 0.1 m, Pade 8 moves
    the mpirams figures to 0.109 and 0.099 dB, so they are the PE's agreement
    and not its discretisation. The bound is 0.3 dB, 2.2x the worst
    measurement; a null may move one 10 m sample either way.

    The grid is pinned because the automatic one is slower and looser here:
    8.9 s against 2.6 s, at 0.298 and 0.674 dB. COA Sect. 6.7 shows one-way
    PEs of 1990 losing about 2 dB on this wedge through their stair-step
    treatment of the slope; neither backend does.

    NULL CONTROL, asserted below: the same engine with the slope removed sits
    at 6.114 and 5.765 dB RMS, 19x outside the bound.
    """
    ram = RAM(backend=backend, dr=5.0, dz=0.25, n_pade=6, timeout=900)

    rms, nulls, _ = _wedge_agreement(ram)
    assert rms[0] < 0.3, f"30 m RMS {rms[0]:.3f} dB"
    assert rms[1] < 0.3, f"150 m RMS {rms[1]:.3f} dB"
    assert nulls == pytest.approx(WEDGE_REF_NULLS_M, abs=20.0)

    null_rms, _, _ = _wedge_agreement(ram, slope=False)
    assert min(null_rms) > 2.0, (
        f"the flat-bottom run is only {min(null_rms):.2f} dB RMS from the wedge "
        f"reference: this benchmark cannot tell the slope from its absence")


@pytest.mark.requires_binary
@pytest.mark.parametrize('mode_coupling, measured, bound', [
    ('coupled', (2.756, 2.392), 3.5),
    ('adiabatic', (2.221, 2.502), 3.3),
])
def test_kraken_mode_sums_follow_the_asa_wedge_within_their_method_error(
        mode_coupling, measured, bound):
    """Kraken's range-dependent mode sums place the wedge's nulls but sit
    2-3 dB RMS off COUPLE07's level, and that gap is the method's, not a
    transcription error.

    Measured with the automatic 15-profile decomposition: coupled 2.756 dB
    at 30 m and 2.392 dB at 150 m, adiabatic 2.221 and 2.502 dB; the 30 m
    nulls land at 1.18 / 1.88 km (coupled) and 1.16 / 1.89 km (adiabatic)
    against 1.16 / 1.87. The bounds sit 0.7-0.8 dB above the measurement, a
    null may move 40 m, and the null control below is 5.742 dB RMS or more.

    Two things separate Kraken from COUPLE here.

    * The mode set. Kraken sums the trapped modes; COUPLE carries 45 modes
      of a guide closed by an absorbing false bottom, which include the
      steep, leaky part of the spectrum. On the flat 200 m start of this
      wedge, over 250 m - 2 km, Kraken's trapped sum sits 1.125 / 2.088 dB RMS
      (30 / 150 m) from its ``leaky_modes=True`` sum, and the leaky sum sits
      0.102 / 0.185 dB from RAM. That remedy does not reach the apex: with
      ``leaky_modes=True`` the krakenc search finds no mode in the profiles
      of 28 m and less at 25 Hz, and Kraken refuses the run there (a wedge
      ending at 50 m solves).
    * The coupling. ``field.exe``'s coupled option projects the pressure
      only onto each new segment's modes, so its answer moves with the
      segment count rather than converging: 2.756 / 2.914 / 3.52 dB at
      30 m for 15 / 51 / 200 profiles. The bound is set at the automatic
      decomposition the default run uses.
    """
    kraken = Kraken(mode_coupling=mode_coupling, timeout=900)

    rms, nulls, _ = _wedge_agreement(kraken)
    assert max(rms) < bound, (
        f"RMS {rms[0]:.3f} / {rms[1]:.3f} dB at 30 / 150 m; measured {measured}")
    assert nulls == pytest.approx(WEDGE_REF_NULLS_M, abs=40.0)

    null_rms, _, _ = _wedge_agreement(kraken, slope=False)
    assert min(null_rms) > bound + 1.0, (
        f"the flat-bottom run is only {min(null_rms):.2f} dB RMS from the wedge "
        f"reference: this bound cannot tell the slope from its absence")


@pytest.mark.requires_binary
def test_bellhop_follows_the_asa_wedge_within_a_ray_models_error_at_25_hz():
    """Bellhop's coherent beam sum tracks the wedge's mean level and places
    both nulls, at the accuracy a ray model has at 25 Hz in a 200 m guide
    (3.3 wavelengths of water).

    Measured with the default (geometric hat) beams: 1.834 dB RMS at 30 m
    and 2.141 dB at 150 m with the default fan, 1.907 and 2.146 dB with 2000
    beams and 1.801 / 2.146 dB with 8000, so beam count is not what limits
    it; ``beam_type='B'`` gives 2.011 / 2.195 dB. The nulls land at 1.14 and
    1.86 km against 1.16 and 1.87, and the 30 m curve carries an extra null
    at 1.56 km that the reference does not have (63.65 dB against the
    reference's 57.06 there). The bound is 2.6 dB, 0.45 dB above the worst
    measurement; it is the ray method's error at this frequency, so it is
    not loosened to suit the PE and mode bounds and not tightened toward
    them either.

    NULL CONTROL, asserted below: slope removed, 5.473 and 5.726 dB RMS.
    """
    bellhop = Bellhop(timeout=900)

    rms, nulls, _ = _wedge_agreement(bellhop)
    assert max(rms) < 2.6, f"RMS {rms[0]:.3f} / {rms[1]:.3f} dB at 30 / 150 m"
    assert nulls == pytest.approx(WEDGE_REF_NULLS_M, abs=40.0)

    null_rms, _, _ = _wedge_agreement(bellhop, slope=False)
    assert min(null_rms) > 3.8, (
        f"the flat-bottom run is only {min(null_rms):.2f} dB RMS from the wedge "
        f"reference: this bound cannot tell the slope from its absence")


# ── NORDA PE Workshop I, Test Case 3b, 250 Hz ───────────────────────────────
#
# A range-independent Pekeris guide, 100 m of isovelocity water over a lossy
# fluid bottom, with source and receiver both 0.5 m above the seafloor. The
# reference is COUPLE07's Test Case 1 (28 modes), ``pe_workshop_3b_250hz_couple07.txt``.
# Agreement is the median and 90th percentile of |dTL| over the table's
# 4.9-10.1 km, plus the seven nulls deeper than 10 dB; an RMS would be set by
# the few samples at the bottom of those nulls.

PE3B_TABLE = 'pe_workshop_3b_250hz_couple07.txt'
PE3B_FREQ = 250.0
PE3B_Z = 99.5
# Ranges (m) of the seven reference nulls with more than 10 dB prominence;
# each engine's null is searched within 50 m of them.
PE3B_REF_NULLS_M = (5130.0, 5930.0, 6950.0, 8090.0, 8320.0, 9190.0, 9600.0)


def _pe3b_env(attenuation=0.5):
    """Test Case 3b; ``attenuation=0`` makes the bottom lossless and is the
    null control."""
    return Environment(
        water_density=RHO_WATER,
        bathymetry=100.0,
        ssp=SoundSpeedProfile.from_pairs([(0.0, 1500.0), (100.0, 1500.0)]),
        bottom=BoundaryProperties(acoustic_type='half-space', sound_speed=1590.0,
                                  density=1.2, attenuation=attenuation))


def _pe3b_tl(model, attenuation=0.5, depth=PE3B_Z):
    """``model``'s TL over the reference table's ranges, source and receiver
    both at ``depth``."""
    ranges, _ = _load_tl_table(PE3B_TABLE)
    tl = model.compute_field(_pe3b_env(attenuation),
                          Source(depths=depth, frequencies=PE3B_FREQ),
                          Receiver(depths=[depth], ranges=ranges)).dB
    return np.asarray(tl, dtype=float).ravel()


def _pe3b_agreement(model, attenuation=0.5):
    """``(median |dTL|, p90 |dTL|, null offsets in m)`` of ``model`` against
    the Test Case 3b reference."""
    ranges, ref = _load_tl_table(PE3B_TABLE)
    tl = _pe3b_tl(model, attenuation)
    diff = np.abs(tl - ref[0])
    offsets = [_null_range(tl, ranges, n - 50.0, n + 50.0) - n for n in PE3B_REF_NULLS_M]
    return float(np.median(diff)), float(np.percentile(diff, 90)), offsets


def test_the_pe_workshop_3b_reference_table_places_its_nulls():
    """The shipped COUPLE07 table covers 4.9-10.1 km in 10 m steps at the
    99.5 m receiver, with its seven deepest nulls where the engine tests
    look for them, over a 63-113 dB field."""
    ranges, ref = _load_tl_table(PE3B_TABLE)
    assert ref.shape == (1, 521)
    assert ranges[0] == pytest.approx(4900.0) and ranges[-1] == pytest.approx(10100.0)
    nulls = [_null_range(ref[0], ranges, n - 50.0, n + 50.0) for n in PE3B_REF_NULLS_M]
    assert nulls == pytest.approx(PE3B_REF_NULLS_M)
    assert 60.0 < ref[0].min() and ref[0].max() < 115.0


@pytest.mark.requires_binary
@pytest.mark.parametrize('engine, measured, bounds, null_bound', [
    ('ram', (0.160, 0.763), (0.35, 1.3), 20.0),
    ('kraken', (0.067, 0.555), (0.2, 1.1), 20.0),
    ('scooter', (0.192, 1.12), (0.4, 1.8), 60.0),
])
def test_wave_engines_match_the_pe_workshop_3b_reference(engine, measured,
                                                          bounds, null_bound):
    """RAM, Kraken and Scooter at their defaults reproduce COUPLE07's
    Test Case 3b, source and receiver half a metre above a lossy bottom.

    Measured median / p90 |dTL| over 4.9-10.1 km: RAM 0.160 / 0.763 dB,
    Kraken 0.067 / 0.555 dB, Scooter 0.192 / 1.12 dB. RAM and Kraken put six
    of the seven nulls deeper than 10 dB on the reference sample and the
    6.95 km one 10 m short; Scooter puts three on it, three 10 m short and
    the 6.95 km one 50 m short. The bounds sit 2-3x above the medians and
    0.5-0.7 dB above the p90s; a null may move 20 m, Scooter's 60. RAM
    pinned to dr 5 m, dz 0.1 m, Pade 8 lands at 0.223 / 0.866 dB and Kraken
    at ``n_mesh=2000`` at 0.065 / 0.53 dB, so the defaults are what is
    measured, not a loose setting of them. Scooter's 6.95 km null moves with
    the edges of its wavenumber window: 10 m short with c_low 0.95 × min
    SSP and c_high 1.05 × the seabed speed, 30 m at c_low 0.9, 40 m at an
    unbounded c_high, 50 m with both (the defaults), 20 m with
    ``taper=0.05``.

    NULL CONTROL, asserted below: the same engine over a lossless bottom
    sits 11.46-11.54 dB median from the reference.
    """
    model = {'ram': RAM, 'kraken': Kraken, 'scooter': Scooter}[engine](timeout=900)

    median, p90, offsets = _pe3b_agreement(model)
    assert median < bounds[0], f"median |dTL| {median:.3f} dB; measured {measured[0]}"
    assert p90 < bounds[1], f"p90 |dTL| {p90:.3f} dB; measured {measured[1]}"
    assert max(abs(o) for o in offsets) <= null_bound, (
        f"null offsets {offsets} m")

    null_median, _, _ = _pe3b_agreement(model, attenuation=0.0)
    assert null_median > 3.0 * bounds[0], (
        f"the lossless-bottom run is only {null_median:.2f} dB median from the "
        f"reference: this benchmark cannot see the bottom loss")


@pytest.mark.requires_binary
def test_bellhop_loses_the_near_bed_field_of_pe_workshop_3b():
    """Bellhop does not reproduce Test Case 3b, and the error is the
    geometry the case was chosen for, half a metre (0.08 wavelength) above
    the seafloor.

    Measured against the reference with the default (geometric hat) beams:
    4.41 dB median |dTL| and a -0.77 dB mean, so the error is in the
    interference pattern rather than in the mean level. Across beam types
    the median stays between 4.21 and 5.24 dB (5000 beams over +-89 deg
    4.21, ``'C'`` 4.97, ``'B'`` 5.24), so no beam setting rescues it; the
    mean depends on the beam type (``'B'`` +3.91 dB, ``'C'`` +4.58 dB), so it
    is not a correction to apply. Against Kraken, with source and receiver
    moved together, the gap closes as they leave the bed: 4.37 dB median at
    99.5 m, 1.64 at 97 m, 0.39 at 90 m, 0.48 at 50 m.

    Both sides are pinned. The upper bound catches a Bellhop that gets
    worse; the lower one catches a fix, which should move Bellhop into the
    wave-engine test above rather than leave this documenting a gap that
    no longer exists. The mid-column check shows that the case's other
    ingredients (250 Hz, the lossy bottom, 10 km) are within Bellhop's reach.
    """
    ranges, ref = _load_tl_table(PE3B_TABLE)
    bellhop = Bellhop(timeout=900)

    near_bed = np.median(np.abs(_pe3b_tl(bellhop) - ref[0]))
    assert 3.0 < near_bed < 7.0, f"near-bed median |dTL| {near_bed:.2f} dB; measured 4.41"

    mid_column = np.median(np.abs(_pe3b_tl(bellhop, depth=50.0)
                                  - _pe3b_tl(Kraken(timeout=900), depth=50.0)))
    assert mid_column < 1.0, f"mid-column median |Bellhop - Kraken| {mid_column:.2f} dB; measured 0.48"


# ── Square-wave corrugated seafloor, 25 Hz (Evans & Gilbert 1985) ───────────
#
# 100 m of water over a dense (2.5) fluid bottom, with a 10 m high, 100 m
# period square-wave corrugation of the seafloor from 5 to 10 km; source 18 m,
# receiver 50 m. The reference is COUPLE07's Test Case 3, the fully two-way
# solution (``corrugated_seafloor_25hz_couple07.txt``). The quantity compared
# is the mean TL offset beyond the corrugation, 10-18.09 km, where the
# corrugation has removed energy from the guide.

CORRUGATION_TABLE = 'corrugated_seafloor_25hz_couple07.txt'
# The same input with the bottom density 1.5 instead of 2.5.
CORRUGATION_RHO15_TABLE = 'corrugated_seafloor_rho15_25hz_couple07.txt'
CORRUGATION_FAR = (10000.0, 18090.0)


def _corrugated_env(corrugated=True, ramp=0.5, density=2.5):
    """The corrugated guide; ``corrugated=False`` is the flat 100 m guide,
    ``density`` the bottom's (g/cm^3).
    Each vertical face of the square wave is written as two control points
    ``2 * ramp`` apart, since the bathymetry is interpolated linearly
    between control points."""
    ranges, depths = [0.0], [100.0]
    if corrugated:
        for k in range(50):
            start = 5000.0 + 100.0 * k
            ranges += [start - ramp, start + ramp, start + 50.0 - ramp, start + 50.0 + ramp]
            depths += [100.0, 90.0, 90.0, 100.0]
    ranges.append(20000.0)
    depths.append(100.0)
    return Environment(
        water_density=RHO_WATER,
        bathymetry=Bathymetry(ranges=np.array(ranges), depths=np.array(depths)),
        ssp=SoundSpeedProfile.from_pairs([(0.0, 1500.0), (100.0, 1500.0)]),
        bottom=BoundaryProperties(acoustic_type='half-space', sound_speed=1704.5,
                                  density=density, attenuation=0.5))


def _corrugation_far_offset(model, corrugated=True, density=2.5):
    """Mean of ``model``'s TL minus the reference of that bottom density
    over ``CORRUGATION_FAR``."""
    ranges, ref = _load_tl_table(
        {2.5: CORRUGATION_TABLE, 1.5: CORRUGATION_RHO15_TABLE}[density])
    tl = model.compute_field(_corrugated_env(corrugated, density=density),
                          Source(depths=18.0, frequencies=25.0),
                          Receiver(depths=[50.0], ranges=ranges)).dB
    tl = np.asarray(tl, dtype=float).ravel()
    far = (ranges >= CORRUGATION_FAR[0]) & (ranges <= CORRUGATION_FAR[1])
    return float(np.mean(tl[far] - ref[0, far]))


def test_the_corrugated_seafloor_reference_table_is_the_published_run():
    """The shipped COUPLE07 table spans 10 m to 18.09 km in 10 m steps at
    the 50 m receiver, and beyond the corrugation the field sits at a
    72-78 dB plateau."""
    ranges, ref = _load_tl_table(CORRUGATION_TABLE)
    assert ref.shape == (1, 1809)
    assert ranges[0] == pytest.approx(10.0) and ranges[-1] == pytest.approx(18090.0)
    far = (ranges >= CORRUGATION_FAR[0]) & (ranges <= CORRUGATION_FAR[1])
    assert 72.0 < np.median(ref[0, far]) < 78.0


@pytest.mark.requires_binary
@pytest.mark.parametrize('engine, measured, bounds', [
    ('mpirams', 4.70, (3.4, 6.0)),
    ('ramgeo', 5.82, (4.4, 7.4)),
    ('kraken_coupled', 2.83, (1.7, 4.0)),
])
def test_one_way_engines_lose_more_to_the_corrugation_than_the_two_way_reference(
        engine, measured, bounds):
    """Beyond the corrugation every forward-marching engine in uacpy puts
    the field below COUPLE07's two-way solution: the mean TL offset over
    10-18.09 km is +4.70 dB for RAM's mpirams, +5.82 dB for ramgeo and
    +2.83 dB for Kraken's coupled modes. The offsets are pinned on both
    sides rather than bounded from above: a change in either direction is
    news.

    The cause is the one-way codes' condition at the vertical faces of the
    steps. Kraken's coupled answer moves with the segment count, as on the
    ASA wedge: +2.83 dB at its automatic decomposition (thinned to 200
    profiles) and +4.02 dB with the 209 profiles the quarter-wavelength
    criterion asks for, where COUPLE07's own one-way pressure-matching run
    of this input (IOUT = 1) lands at +1.13 dB. The PE offset follows the
    density contrast: with nothing changed but the bottom density, mpirams
    sits at +4.70 dB at 2.5, +0.69 at 1.5 and -0.03 at 1.0 against COUPLE's
    two-way run of each (density 1.0: COUPLE07 rerun, not shipped)
    (test_the_pe_offset_beyond_the_corrugation_grows_with_the_bottom_density).
    Collins & Siegmann, Parabolic Wave Equations with Applications, Sect. 2.6,
    describe exactly this: conserving the dependent variable across stair-step
    faces leaves an amplitude error that grows with the contrast, and an
    energy-conserving or single-scattering condition at the faces removes
    it. RAM 1.5's updat only re-forms the matrices when the bottom steps.

    What is known. Over the flat guide the engines agree with COUPLE
    (COUPLE07 run on this input with the corrugation removed: mpirams
    +0.02 dB mean beyond 10 km, Kraken -0.00 dB), and COUPLE's two-way
    answer is converged in mode count (120 to 200 modes moves it 0.11 dB),
    so the gap is the corrugation's. Grid and Pade order move the PE figure
    by under a decibel: over 36 grids, dr 1, 2, 5 and 10 m x dz 0.1, 0.25
    and 0.5 m x Pade 4, 6 and 8 with 1 m step faces, mpirams lands at +4.41
    to +5.31 dB; and RAMS, the elastic PE with its own
    energy-flux correction, run on its automatic grid over a 50 m/s shear
    bottom lands at +5.81 dB. Three PE codes agree within 1.4 dB and sit
    4.4-5.9 dB from the two-way field, where COA Sect. 6.7 places the one-way models'
    treatment of vertical interfaces as their known weakness on stepped
    bathymetry; COUPLE's own single-scatter run of this input
    (IOUT = -1) disagrees with its two-way run by 2.3 dB in the other
    direction, so the case is not a one-way benchmark in either code.

    NULL CONTROL, asserted below: the same engine on the flat guide sits
    near -2 dB (mpirams and ramgeo -1.94, Kraken -1.96; COUPLE's own flat
    run -1.96), the corrugation's own loss in the two-way reference,
    outside every bound.
    """
    model = {
        'mpirams': lambda: RAM(backend='mpirams', dr=5.0, dz=0.25, n_pade=6, timeout=900),
        # ramgeo takes one bathymetry node per step and the manual bounds dr by
        # the smallest region (ram.pdf p.8): the 1 m faces need dr <= 1 m.
        # Measured over dr: 0.25 m +5.49, 0.5 m +5.82, 0.9999 m +5.88 dB, and
        # +4.72 at exactly 1 m, where every step lands on a face's ends.
        'ramgeo': lambda: RAM(backend='ramgeo', dr=0.5, dz=0.25, n_pade=6, timeout=900),
        'kraken_coupled': lambda: Kraken(mode_coupling='coupled', timeout=900),
    }[engine]()

    offset = _corrugation_far_offset(model)
    assert bounds[0] < offset < bounds[1], (
        f"mean TL offset beyond the corrugation {offset:+.2f} dB; measured {measured:+.2f}")

    flat_offset = _corrugation_far_offset(model, corrugated=False)
    assert not bounds[0] < flat_offset < bounds[1], (
        f"the flat-guide run sits at {flat_offset:+.2f} dB, inside the bound: this "
        f"benchmark cannot tell the corrugation from its absence")



def test_the_density_1p5_corrugation_table_is_the_published_run_with_one_change():
    """The density-1.5 COUPLE07 table shares the density-2.5 table's range
    axis, and the lighter bottom leaves the far field 1.39 dB louder on
    mean over 10-18.09 km."""
    ranges, ref = _load_tl_table(CORRUGATION_RHO15_TABLE)
    ranges_25, ref_25 = _load_tl_table(CORRUGATION_TABLE)
    assert ref.shape == (1, 1809)
    np.testing.assert_allclose(ranges, ranges_25)
    far = (ranges >= CORRUGATION_FAR[0]) & (ranges <= CORRUGATION_FAR[1])
    assert 1.2 < np.mean(ref_25[0, far] - ref[0, far]) < 1.6


@pytest.mark.slow
@pytest.mark.requires_binary
@pytest.mark.parametrize('density, measured, bounds', [
    (1.5, 0.69, (-0.1, 1.5)),
    (2.5, 4.70, (3.4, 6.0)),
])
def test_the_pe_offset_beyond_the_corrugation_grows_with_the_bottom_density(
        density, measured, bounds):
    """mpirams (dr 5 m, dz 0.25 m, Pade 6) against COUPLE07's two-way run of
    the same corrugation with only the bottom density changed: the mean TL
    offset over 10-18.09 km is +0.69 dB at density 1.5 and +4.70 dB at 2.5
    (-0.03 dB at 1.0, against a COUPLE07 rerun at that density that is not
    shipped). The PE keeps p across the step faces, whose error grows with
    the density contrast (Collins & Siegmann Sect. 2.6). ramgeo (dr 0.5 m)
    measured +0.94 / +5.82 dB, rams on its automatic grid over a 50 m/s
    shear bottom +0.81 / +5.81 dB: its approximate energy-flux correction
    does not remove it.

    NULL CONTROL, asserted below: the flat guide sits at -0.99 dB at
    density 1.5 (-1.94 at 2.5), 0.89 dB below the density-1.5 bound, which
    sits 0.8 dB either side of the measurement.
    """
    model = RAM(backend='mpirams', dr=5.0, dz=0.25, n_pade=6, timeout=900)
    offset = _corrugation_far_offset(model, density=density)
    assert bounds[0] < offset < bounds[1], (
        f"mean TL offset beyond the corrugation {offset:+.2f} dB; measured {measured:+.2f}")
    flat_offset = _corrugation_far_offset(model, corrugated=False, density=density)
    assert not bounds[0] < flat_offset < bounds[1], (
        f"the flat-guide run sits at {flat_offset:+.2f} dB, inside the bound")

# ── Bucker's fluid waveguide, 100 Hz (COA Sect. 4.10.2) ─────────────────────
#
# 240 m of water with a 2 m/s sound-speed dip at 120 m over a 1505 m/s,
# density 2.1 half-space; source 30 m, receiver 90 m. COA's point is that the
# small speed contrast traps few modes while the density contrast puts much of
# the field in the continuous spectrum, so a trapped-mode sum is wrong even at
# long range. The reference is a wavenumber integral (Scooter from k = 0,
# cross-checked against OASES), ``bucker_waveguide_100hz_scooter.txt``.
# Agreement is the median and p90 of |dTL| over 2-20 km.

BUCKER_TABLE = 'bucker_waveguide_100hz_scooter.txt'
BUCKER_WINDOW = (2000.0, 20000.0)


def _bucker_env(bottom_density=2.1):
    """Bucker's waveguide; ``bottom_density=1.0`` removes the density
    contrast the case is built on and is the null control."""
    return Environment(
        water_density=RHO_WATER,
        bathymetry=240.0,
        ssp=SoundSpeedProfile.from_pairs([(0.0, 1500.0), (120.0, 1498.0), (240.0, 1500.0)]),
        bottom=BoundaryProperties(acoustic_type='half-space', sound_speed=1505.0,
                                  density=bottom_density, attenuation=0.0))


def _bucker_agreement(model, bottom_density=2.1):
    """``(median, p90)`` of |dTL| between ``model`` and the reference over
    ``BUCKER_WINDOW``."""
    ranges, ref = _load_tl_table(BUCKER_TABLE)
    tl = model.compute_field(_bucker_env(bottom_density),
                          Source(depths=30.0, frequencies=100.0),
                          Receiver(depths=[90.0], ranges=ranges)).dB
    diff = np.abs(np.asarray(tl, dtype=float).ravel() - ref[0])
    in_window = (ranges >= BUCKER_WINDOW[0]) & (ranges <= BUCKER_WINDOW[1])
    return float(np.median(diff[in_window])), float(np.percentile(diff[in_window], 90))


def test_the_bucker_reference_table_spans_the_published_ranges():
    """The shipped table covers 0.1-20 km in 50 m steps at the 90 m
    receiver."""
    ranges, ref = _load_tl_table(BUCKER_TABLE)
    assert ref.shape == (1, 399)
    assert ranges[0] == pytest.approx(100.0) and ranges[-1] == pytest.approx(20000.0)
    assert np.all(np.isfinite(ref))


@pytest.mark.requires_binary
@pytest.mark.requires_oases
def test_two_wavenumber_integration_codes_agree_on_the_bucker_waveguide():
    """OASES OAST at its defaults lands on the Scooter reference: 0.0145 dB
    median and 0.048 dB p90 |dTL| over 2-20 km. Two independent
    wavenumber-integration codes agreeing to hundredths of a dB is what
    makes the table a reference rather than one engine's answer. The bounds
    are 0.05 and 0.25 dB; the density-1.0 null control sits at 2.58 dB.
    """
    median, p90 = _bucker_agreement(OAST(timeout=900))
    assert median < 0.05, f"median |dTL| {median:.4f} dB; measured 0.0145"
    assert p90 < 0.25, f"p90 |dTL| {p90:.3f} dB; measured 0.048"

    null_median, _ = _bucker_agreement(OAST(timeout=900), bottom_density=1.0)
    assert null_median > 1.0, f"density-1.0 run only {null_median:.2f} dB median away"


@pytest.mark.requires_binary
@pytest.mark.parametrize('engine, measured, bounds', [
    ('kraken_leaky', (0.0198, 0.07), (0.05, 0.3)),
    ('kraken', (0.1235, 1.755), (0.3, 2.3)),
    ('ram_deep_grid', (0.0776, 0.594), (0.3, 1.2)),
])
def test_engines_reproduce_the_bucker_waveguide(engine, measured, bounds):
    """Kraken and RAM reproduce the wavenumber-integration field of
    Bucker's waveguide over 2-20 km.

    Measured median / p90 |dTL|: Kraken with ``leaky_modes=True``
    0.0198 / 0.07 dB; Kraken at its defaults 0.1235 / 1.755 dB, because its
    automatic c_high (1.05 x the 1505 m/s half-space) keeps some modes past
    the bottom speed but not all the continuous spectrum the density
    contrast feeds; RAM with ``zmax=2000`` 0.0776 / 0.594 dB. RAM's
    automatic grid is pinned by
    ``test_ram_default_domain_reaches_the_bucker_continuous_spectrum``. The
    bounds sit 2.4-3.9x above the medians and 0.2-0.6 dB above the p90s.
    Scooter is not scored here: at its defaults (an unbounded c_high) it is
    the run that generated the table.

    NULL CONTROL, asserted below: the same engine over a density-1.0
    half-space, the water's own density, sits 2.44-2.60 dB median from the
    reference.
    """
    model = {
        'kraken_leaky': lambda: Kraken(leaky_modes=True, timeout=900),
        'kraken': lambda: Kraken(timeout=900),
        'ram_deep_grid': lambda: RAM(zmax=2000.0, timeout=900),
    }[engine]()

    median, p90 = _bucker_agreement(model)
    assert median < bounds[0], f"median |dTL| {median:.3f} dB; measured {measured[0]}"
    assert p90 < bounds[1], f"p90 |dTL| {p90:.3f} dB; measured {measured[1]}"

    null_median, _ = _bucker_agreement(model, bottom_density=1.0)
    assert null_median > 3.0 * bounds[0], (
        f"the density-1.0 run is only {null_median:.2f} dB median from the "
        f"reference: this benchmark cannot see the density contrast")


@pytest.mark.requires_binary
def test_a_trapped_mode_sum_misses_the_bucker_continuous_spectrum():
    """COA Fig. 4.18b's dashed curve: the sum over the modes with real
    wavenumbers alone (Kraken with c_high at the 1505 m/s half-space speed)
    misses the continuous spectrum and is wrong out to 20 km.

    Measured: 1.574 dB median and 8.667 dB p90 |dTL| over 2-20 km, against
    0.0198 / 0.07 dB once ``leaky_modes=True`` lets Kraken solve past the
    bottom speed. Pinned on both sides: below 0.8 dB median the
    trapped-mode run would no longer show the published effect, which
    would mean the reference or the window had lost it.
    """
    median, p90 = _bucker_agreement(Kraken(c_high=1505.0, timeout=900))
    assert 0.8 < median < 2.5, f"median |dTL| {median:.3f} dB; measured 1.574"
    assert p90 > 4.0, f"p90 |dTL| {p90:.2f} dB; measured 8.667"


@pytest.mark.requires_binary
def test_ram_default_domain_reaches_the_bucker_continuous_spectrum():
    """RAM at its defaults matches the reference as well as on a pinned deep
    grid: the automatic pad holds the depth the continuous spectrum's bottom
    field reaches over the track (``_domain.leaky_field_depth``; the first
    leaky mode leaves at 3.6° and loses 1.21 dB/km), here zmax 1688.9 m.
    Measured over 2-20 km: 0.1188 dB median (0.069 / 0.252 over 2-10 / 10-20
    km), against 2.449 dB with a pad of two seabed wavelengths (zmax 588 m)
    and 0.0776 dB at zmax 2000 m.
    """
    median, _ = _bucker_agreement(RAM(timeout=900))
    assert median < 0.3, f"median |dTL| {median:.3f} dB at the automatic grid"


# ── NORDA PE Workshop I, Test Case 4c, 25 Hz ────────────────────────────────
#
# 150 km of 3410 m deep water, an upslope to 200 m over 50 km and a 200 m
# shelf to 250 km, over a 454 m sediment with a strong speed gradient that
# follows the seafloor; source 600 m, receivers 150 m and 700 m. The reference
# is COUPLE07's Test Case 4 (300 modes, single-scatter matching),
# ``pe_workshop_4c_25hz_couple07.txt``.
#
# COUPLE makes 1/c^2 piecewise linear between the listed points, so the
# environment is written that way: the water profile is resampled every 10 m
# and the sediment gradient becomes 46 homogeneous layers, each at the
# 1/c^2-linear speed of its mid-depth.
#
# Over the flat 5-150 km the comparison is pointwise (median and p90 of
# |dTL|). Over the slope and the shelf it is the RMS and mean of the
# difference of 5 km intensity averages at the 150 m receiver: 50 km of
# changing depth at 25 Hz moves the fine fringes of any two correct solutions
# apart, while the averaged level is what the range dependence sets.

PE4C_TABLE = 'pe_workshop_4c_25hz_couple07.txt'
PE4C_WATER = ((0.0, 1539.3), (30.0, 1539.8), (200.0, 1534.2), (600.0, 1502.4),
              (700.0, 1495.4), (800.0, 1491.8), (1000.0, 1488.0), (1100.0, 1487.5),
              (1200.0, 1487.9), (3410.0, 1525.0))
PE4C_BATHYMETRY = (
    (0.0, 3410.0), (150000.0, 3410.0), (153271.0, 3200.0), (156386.0, 3000.0),
    (159502.0, 2800.0), (162617.0, 2600.0), (165732.0, 2400.0), (168847.0, 2200.0),
    (171963.0, 2000.0), (175078.0, 1800.0), (178193.0, 1600.0), (181308.0, 1400.0),
    (184424.0, 1200.0), (185981.0, 1100.0), (187539.0, 1000.0), (189097.0, 900.0),
    (190654.0, 800.0), (192212.0, 700.0), (193769.0, 600.0), (195327.0, 500.0),
    (196885.0, 400.0), (198442.0, 300.0), (200000.0, 200.0), (250000.0, 200.0))
PE4C_FLAT = (5000.0, 150000.0)
PE4C_SLOPE = (152500.0, 197500.0)
PE4C_SHELF = (202500.0, 247500.0)


def _n2_linear_speed(c_top, c_bottom, fraction):
    """Sound speed ``fraction`` of the way between ``c_top`` and
    ``c_bottom`` with 1/c^2 linear, COUPLE's interpolation."""
    return (c_top ** -2 + (c_bottom ** -2 - c_top ** -2) * fraction) ** -0.5


def _pe4c_water_profile(step=10.0):
    """The water profile resampled every ``step`` m, 1/c^2-linear between
    the published points."""
    z_pub, c_pub = np.array(PE4C_WATER).T
    z = np.unique(np.concatenate([z_pub, np.arange(0.0, z_pub[-1] + step / 2, step)]))
    i = np.clip(np.searchsorted(z_pub, z, side='right') - 1, 0, len(z_pub) - 2)
    c = _n2_linear_speed(c_pub[i], c_pub[i + 1], (z - z_pub[i]) / (z_pub[i + 1] - z_pub[i]))
    return SoundSpeedProfile.from_pairs(list(zip(z, c)))


def _pe4c_seabed(seafloor_depth, n_layers=46):
    """The 454 m sediment under a seafloor at ``seafloor_depth``: top speed
    0.975 x the water speed there, base speed 1.305 x the top, as
    ``n_layers`` homogeneous layers over a half-space at the base speed."""
    z_pub, c_pub = np.array(PE4C_WATER).T
    c_top = 0.975 * float(np.interp(seafloor_depth, z_pub, c_pub))
    c_base = 1.305 * c_top
    layers = [SedimentLayer(thickness=454.0 / n_layers,
                            sound_speed=_n2_linear_speed(c_top, c_base, (i + 0.5) / n_layers),
                            density=1.5, attenuation=0.0258)
              for i in range(n_layers)]
    return SeabedColumn(layers=layers, halfspace=BoundaryProperties(
        acoustic_type='half-space', sound_speed=c_base, density=1.5, attenuation=0.0258))


def _pe4c_env(slope=True):
    """Test Case 4c; ``slope=False`` keeps 3410 m to 250 km and is the null
    control. The seabed is given at every published bathymetry point from
    150 km on, where its top speed follows the seafloor."""
    if slope:
        ranges, depths = np.array(PE4C_BATHYMETRY).T
    else:
        ranges, depths = np.array([0.0, 250000.0]), np.array([3410.0, 3410.0])
    column_ranges = ranges[(ranges == 0.0) | (ranges >= 150000.0)]
    columns = [_pe4c_seabed(float(np.interp(r, ranges, depths))) for r in column_ranges]
    bottom = (Bottom(columns=columns, ranges=column_ranges) if len(columns) > 1
              else Bottom(columns=columns))
    return Environment(water_density=RHO_WATER,
                       bathymetry=Bathymetry(ranges=ranges, depths=depths),
                       ssp=_pe4c_water_profile(), bottom=bottom)


def _intensity_average_dB(tl, n_samples):
    """``tl`` (dB) averaged in intensity over a running ``n_samples`` window."""
    intensity = 10.0 ** (-np.asarray(tl) / 10.0)
    return -10.0 * np.log10(np.convolve(intensity, np.ones(n_samples) / n_samples, mode='same'))


def _pe4c_agreement(model, slope=True):
    """``{'flat': [(median, p90) per receiver], 'slope': (rms, mean),
    'shelf': (rms, mean)}`` of ``model`` against the Test Case 4c reference;
    the slope and shelf figures are at the 150 m receiver over 5 km
    intensity averages."""
    ranges, ref = _load_tl_table(PE4C_TABLE)
    tl = model.compute_field(_pe4c_env(slope), Source(depths=600.0, frequencies=25.0),
                          Receiver(depths=[150.0, 700.0], ranges=ranges)).dB
    tl = np.asarray(tl, dtype=float).reshape(ref.shape)
    flat = (ranges >= PE4C_FLAT[0]) & (ranges < PE4C_FLAT[1])
    out = {'flat': [(float(np.median(np.abs(tl[i, flat] - ref[i, flat]))),
                     float(np.percentile(np.abs(tl[i, flat] - ref[i, flat]), 90)))
                    for i in range(2)]}
    window = int(round(5000.0 / (ranges[1] - ranges[0])))
    averaged = _intensity_average_dB(tl[0], window) - _intensity_average_dB(ref[0], window)
    for name, (r_min, r_max) in (('slope', PE4C_SLOPE), ('shelf', PE4C_SHELF)):
        part = averaged[(ranges >= r_min) & (ranges <= r_max)]
        out[name] = (float(np.sqrt(np.mean(part ** 2))), float(np.mean(part)))
    return out


def test_the_pe_workshop_4c_reference_table_spans_the_published_ranges():
    """The shipped COUPLE07 table covers 10 m to 249.91 km every 100 m at
    the 150 m and 700 m receivers."""
    ranges, ref = _load_tl_table(PE4C_TABLE)
    assert ref.shape == (2, 2500)
    assert ranges[0] == pytest.approx(10.0) and ranges[-1] == pytest.approx(249910.0)
    assert np.all(np.isfinite(ref))


@pytest.mark.slow
@pytest.mark.requires_binary
@pytest.mark.parametrize('engine, bounds', [
    # (flat median, flat p90, slope rms, shelf rms)
    ('ram', (0.7, 2.5, 0.9, 1.0)),
    ('kraken_adiabatic', (0.4, 1.0, 1.8, 2.0)),
])
def test_engines_follow_pe_workshop_4c_up_the_slope_and_onto_the_shelf(engine, bounds):
    """RAM and Kraken's adiabatic modes follow COUPLE07 through Test Case 4c.

    Measured over the flat 5-150 km, median / p90 |dTL| at 150 and 700 m:
    RAM 0.279 / 1.172 and 0.413 / 1.832 dB, Kraken 0.101 / 0.425 and
    0.153 / 0.584 dB. Over the slope and the shelf, RMS of the 5 km
    intensity averages at 150 m: RAM 0.457 and 0.547 dB, Kraken 1.253 and
    1.409 dB. RAM's p90 over the flat part is the widest figure; its
    automatic grid here is ramgeo's (dr 37.95 m, dz raised to 1.051 m to fit
    the binary's arrays, 19.7 s), and mpirams pinned to dr 10 m, dz 0.5 m,
    Pade 8 (227 s) is 0.165 / 0.615 dB at 150 m. Kraken's adiabatic sum is
    the looser one on the slope, where a 3 km descent over 50 km couples
    modes the adiabatic approximation keeps apart. The bounds sit
    0.2-0.7 dB above the measurements.

    NULL CONTROL, asserted below: the same engine with the slope removed
    sits at 4.422-4.456 dB RMS over the slope and 3.154-3.172 dB over the
    shelf.
    """
    model = {'ram': lambda: RAM(timeout=1800),
             'kraken_adiabatic': lambda: Kraken(timeout=1800)}[engine]()

    got = _pe4c_agreement(model)
    for i, depth in enumerate((150, 700)):
        assert got['flat'][i][0] < bounds[0], f"{depth} m flat median {got['flat'][i][0]:.2f} dB"
        assert got['flat'][i][1] < bounds[1], f"{depth} m flat p90 {got['flat'][i][1]:.2f} dB"
    assert got['slope'][0] < bounds[2], f"slope RMS {got['slope'][0]:.2f} dB"
    assert got['shelf'][0] < bounds[3], f"shelf RMS {got['shelf'][0]:.2f} dB"

    null = _pe4c_agreement(model, slope=False)
    assert null['slope'][0] > bounds[2] + 2.0 and null['shelf'][0] > bounds[3] + 0.8, (
        f"the slope-free run sits at {null['slope'][0]:.2f} / {null['shelf'][0]:.2f} dB "
        f"RMS: this benchmark cannot tell the slope from its absence")


@pytest.mark.slow
@pytest.mark.requires_binary
@pytest.mark.xfail(strict=True, reason=(
    "Kraken(mode_coupling='coupled') at its automatic decomposition puts the "
    "4c shelf 21.8 dB too loud (5 km averages); uniform n_segments=60-400 "
    "put the shelf 2.0-4.4 dB quieter than the reference on mean. The automatic decomposition here is the thinned one "
    "(326 asked, 200 kept, unevenly spaced). field.exe's coupled projection is not "
    "unitary on modes at and above the half-space speed (docs/models/kraken.md 6.6). "
    "Remove this marker when fixed."))
def test_kraken_coupled_modes_keep_the_pe_workshop_4c_shelf_level():
    """Kraken's coupled modes at their defaults should keep the shelf level
    COUPLE07 gives. They do not: over 202.5-247.5 km the 5 km intensity
    averages at 150 m sit 22.069 dB RMS from the reference, mean -21.761 dB,
    i.e. the field gains energy. The 700 m receiver, in the sediment there,
    is 59.0 dB too loud. The same engine with ``n_segments`` 60, 200 or 400
    (uniform) puts the shelf TL 2.954 / 4.394 / 1.994 dB above the
    reference's 95.5 dB on mean (quieter), and adiabatic modes 1.22 dB above.
    The automatic decomposition warns that it was thinned from 326 to 200
    profiles and "may not be converged"; it does not warn of a 20 dB gain.
    """
    got = _pe4c_agreement(Kraken(mode_coupling='coupled', timeout=1800))
    assert got['shelf'][0] < 6.0, f"shelf RMS {got['shelf'][0]:.2f} dB"


# ── ASA benchmark Problem I, the ideal wedge, 25 Hz ─────────────────────────
#
# Jensen & Ferla 1990, Problem I: the wedge of Problem II (200 m at the
# source, 2.86 deg up to the apex 4 km away, source 100 m, receivers 30 m and
# 150 m) with perfectly reflecting boundaries. Buckingham & Tolstoy, JASA 87,
# 1511-1513 (1990) solve it analytically for a pressure-release bottom; the
# rigid-bottom wedge (Luo, Yang, Qin & Zhang, Chin. Phys. Lett. 29, 104303,
# 2012) has the same series with the angular eigenvalues moved to
# ``(m + 1/2) pi / theta_w``, Dirichlet at the surface and Neumann at the
# bottom. The series is the field of a LINE source parallel to the apex, so
# Bellhop runs ``source_type='line'`` (its plane geometry) and is scored
# against the series itself: ``_ideal_wedge_line_tl``. No 2-D to 3-D
# conversion enters the comparison, which is why the far-field point-source
# form of ``test_benchmarks_analytic.ideal_wedge_tl`` is not used here. Etter, Underwater Acoustic
# Modeling and Simulation, Sect. 12.3, Fig. 12.1 reproduces the problem
# statements with both a line-source ("parallel to the apex, i.e., 2D
# geometry") and a point-source instruction; its parsed text does not show
# which binds to which case, so the choice rests on what the series is.
#
# Only Bellhop is measured. RAM has no spelling for a vacuum or rigid
# seabed, Kraken refuses a range-dependent deck over one (the profiles must
# share a depth below the seafloor), and a one-way solver cannot carry the
# radial standing wave the perfectly reflecting wedge sets up (see
# ``test_bellhop_ideal_wedge_matches_analytic``).

IDEAL_WEDGE_RANGES = np.arange(50.0, 3801.0, 10.0)


def _ideal_wedge_line_tl(r_from_source, z_r, bottom, z_s=100.0, R_s_apex=4000.0,
                         f=25.0, c=1500.0, slope=0.05):
    """Analytic TL of the ideal wedge for a line source parallel to the apex,
    at ``r_from_source`` (m, horizontal, toward the apex) and depth ``z_r``.

    The exact 2-D series in apex-centred polar coordinates (r, theta),
    theta measured down from the surface::

        p(r, theta) = (i pi / theta_w) sum_m sin(nu_m theta) sin(nu_m theta_s)
                      J_nu(k r_<) H1_nu(k r_>)

    with ``nu_m = m pi / theta_w`` (m = 1, 2, ...) under a ``'vacuum'``
    bottom and ``nu_m = (m + 1/2) pi / theta_w`` (m = 0, 1, ...) under a
    ``'rigid'`` one, so that ``sin(nu theta)`` vanishes at the surface and,
    at the bottom, vanishes or has zero slope. The coefficient is the one
    of the free-space line source ``(i/4) H0(kR)`` (the radial jump with
    ``W[J, H1] = 2i/(pi r)`` and the angular normalisation ``2/theta_w``), so
    TL is ``-20 log10 |sqrt(8 pi k) p|``, unit amplitude at 1 m — the
    convention ``test_bellhop_line_source_level_is_unit_amplitude_at_one_metre``
    pins on Bellhop's line source. Modes with ``nu > 1.2 k r_>`` are deeply
    evanescent and dropped."""
    from scipy.special import jv, hankel1
    k = 2 * np.pi * f / c
    theta_w = np.arctan(slope)
    theta_s = np.arctan(z_s / R_s_apex)
    r_s = np.hypot(R_s_apex, z_s)
    m0, shift = {'vacuum': (1, 0.0), 'rigid': (0, 0.5)}[bottom]
    out = []
    for rs in np.atleast_1d(r_from_source).astype(float):
        R_apex = R_s_apex - rs
        theta = np.arctan(z_r / R_apex)
        r = np.hypot(R_apex, z_r)
        r_lo, r_hi = min(r, r_s), max(r, r_s)
        p = 0j
        m = m0
        while (m + shift) * np.pi / theta_w <= 1.2 * k * r_hi:
            nu = (m + shift) * np.pi / theta_w
            p += (np.sin(nu * theta) * np.sin(nu * theta_s)
                  * jv(nu, k * r_lo) * hankel1(nu, k * r_hi))
            m += 1
        p *= 1j * np.pi / theta_w
        out.append(-20.0 * np.log10(np.abs(np.sqrt(8 * np.pi * k) * p)))
    return np.array(out)


def _ideal_wedge_env(bottom, slope=0.05):
    """The ideal wedge over a ``'vacuum'`` or ``'rigid'`` bottom;
    ``slope=0`` is the flat 200 m guide and the null control."""
    rr = np.linspace(0.0, 3800.0, 40)
    return Environment(
        water_density=RHO_WATER,
        bathymetry=Bathymetry(ranges=rr, depths=200.0 - slope * rr),
        ssp=SoundSpeedProfile.from_pairs([(0.0, 1500.0), (200.0, 1500.0)]),
        bottom=BoundaryProperties(acoustic_type=bottom))


def _ideal_wedge_agreement(bottom, slope=0.05, n_beams=4000):
    """``[(median, p90)]`` of |Bellhop - analytic| at 30 and 150 m, Bellhop
    and the series both for a line source, each over the ranges where the
    receiver is at least 5 m above the seafloor."""
    r = IDEAL_WEDGE_RANGES
    tl = Bellhop(n_beams=n_beams, timeout=900).compute_field(
        _ideal_wedge_env(bottom, slope),
        Source(depths=100.0, frequencies=25.0, source_type='line'),
        Receiver(depths=[30.0, 150.0], ranges=r)).dB
    tl = np.asarray(tl, dtype=float).reshape(2, -1)
    out = []
    for i, z_r in enumerate((30.0, 150.0)):
        in_water = (200.0 - 0.05 * r) > z_r + 5.0
        diff = np.abs(tl[i, in_water] - _ideal_wedge_line_tl(r[in_water], z_r, bottom))
        out.append((float(np.median(diff)), float(np.percentile(diff, 90))))
    return out


@pytest.mark.requires_binary
@pytest.mark.parametrize('bottom, measured, bounds', [
    ('vacuum', ((0.47, 1.683), (0.384, 1.62)), ((0.9, 2.5), (0.8, 2.4))),
    ('rigid', ((0.504, 1.648), (1.119, 3.143)), ((0.9, 2.5), (1.6, 4.0))),
])
def test_bellhop_follows_the_ideal_wedge_at_both_published_receivers(bottom, measured, bounds):
    """Bellhop's coherent beam sum for a line source follows the analytic
    ideal wedge over the published 0-3.8 km at both receivers, for the
    pressure-release bottom of Problem I and for a rigid bottom.

    Measured median / p90 |dTL| with 4000 beams: pressure-release 0.47 /
    1.683 dB at 30 m and 0.384 / 1.62 dB at 150 m; rigid 0.504 / 1.648 and
    1.119 / 3.143 dB. The mean of Bellhop minus the series is +0.01 dB at
    30 m and -0.10 / -0.19 dB at 150 m (pressure-release / rigid): no level
    offset. 16000 beams move the medians by at most 0.3 dB (0.334 / 1.228,
    0.112 / 1.05; 0.421 / 1.478, 1.059 / 2.953); the default fan is looser
    (1.036 / 3.686 at 30 m, pressure-release), hence the pinned count. The
    bounds sit 0.4-0.5 dB above the medians and 0.8-0.9 dB above the p90s.

    The rigid series is written here, so it is checked here too: at 30 m it
    sits 3.85 dB median from the pressure-release series, and Bellhop over a
    rigid bottom lands 0.504 dB from it against 4.0 dB from the
    pressure-release one, an independent method picking out the right series.

    NULL CONTROL, asserted below: Bellhop over the flat 200 m guide sits
    4.43-6.95 dB median from the wedge solution.
    """
    got = _ideal_wedge_agreement(bottom)
    for (median, p90), (b_med, b_p90), m, depth in zip(got, bounds, measured, (30, 150)):
        assert median < b_med, f"{depth} m median {median:.2f} dB; measured {m[0]}"
        assert p90 < b_p90, f"{depth} m p90 {p90:.2f} dB; measured {m[1]}"

    null = _ideal_wedge_agreement(bottom, slope=0.0)
    assert min(n[0] for n in null) > 3.0, f"flat-guide medians {null}"


@pytest.mark.parametrize('build', [
    _wedge_env, _pe3b_env, _corrugated_env, _bucker_env, _pe4c_env,
    lambda: _ideal_wedge_env('vacuum'),
], ids=['asa_wedge', 'pe_workshop_3b', 'corrugation', 'bucker',
        'pe_workshop_4c', 'ideal_wedge'])
def test_every_benchmark_environment_holds_the_references_water(build):
    """Every reference on this page was computed with water of density
    1.0 g/cm^3 (module docstring), so every builder sets it. The engine
    bounds alone do not hold the ASA wedge to it: at uacpy's default 1.027
    its RAM and Bellhop agreements (0.225 dB, 2.265 dB) still pass their
    0.3 dB and 2.6 dB bounds."""
    assert build().water_density == RHO_WATER == 1.0
