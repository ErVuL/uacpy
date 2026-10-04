"""The letter codes of a Bellhop deck and what each influence routine
implements: the ``RunType(1:1)`` letter of each run mode, the run types and
receiver grids each ``beam_type`` can run, and whether a ``grid_type`` pairs
the receiver depths with the ranges."""

from uacpy.core.run_settings import RunMode

#: uacpy run mode -> Bellhop ``RunType(1:1)``. The letters are the manual's
#: (``doc/bellhop.htm`` section 8, RUN TYPE): 'R' ray file, 'E' eigenray file,
#: 'A' amplitude-delay ascii, 'a' amplitude-delay binary, 'C' coherent TL,
#: 'I' incoherent TL, 'S' semicoherent TL.
#:
#: This is position 1 and shares its alphabet with position 2 (``beam_type``),
#: where the same characters name influence routines instead: 'C' is coherent
#: TL here but Cerveny-Cartesian there, 'S' is semicoherent here but Bucker's
#: simple Gaussian there, and 'R' is a ray trace here but Cerveny ray-centred
#: there. Never resolve a letter without knowing which position it sits in.
_RUN_MODE_TO_BELLHOP_TYPE = {
    RunMode.COHERENT_TL: 'C',
    RunMode.INCOHERENT_TL: 'I',
    RunMode.SEMICOHERENT_TL: 'S',
    RunMode.RAYS: 'R',
    RunMode.EIGENRAYS: 'E',
    RunMode.ARRIVALS: 'A',
}
# The broadband routes run an ARRIVALS deck to get the eigenray set, so both
# inherit the arrivals row of _BEAM_TYPE_RUN_TYPES.
_RUN_MODE_TO_INFLUENCE_LETTER = {
    **_RUN_MODE_TO_BELLHOP_TYPE,
    RunMode.BROADBAND: 'A',
    RunMode.TIME_SERIES: 'A',
}

# Which RunType(1:1) letters each influence routine actually implements,
# enumerated from Bellhop/influence.f90. 'G', 'B' and 'g' funnel through
# ApplyContribution (:629-652), which holds the only CALL AddArr (:640) and so
# is the sole implementer of the 'A'/'a' (arrivals) branch. 'S' (InfluenceSGB,
# :656-725) has its own CASE('E') (:701-703) writing the ray, so it serves
# eigenrays too, but it has no 'A'/'a', 'I' or 'S' branch: those fall into its
# CASE DEFAULT (:704), which writes complex pressure into U. For an 'A'/'E' run
# bellhop.f90:216 allocated U as U(1,1), so on 'A'/'a' that write runs off the
# end of the heap block; on 'I'/'S' ScalePressure's SQRT( REAL( U ) ) (:779)
# takes the root of a signed coherent sum. The two Cerveny beam types
# (InfluenceCervenyCart :157-289, InfluenceCervenyRayCen :19-153) branch on
# 'C'/'I'/'S' only and write U unconditionally, so both 'E' and 'A'/'a' hit
# that same U(1,1) overrun there. A RAYS run is safe for every beam type
# because bellhop.f90:288 writes the trajectory and never calls an influence
# routine.
_BEAM_TYPE_RUN_TYPES = {
    'G': frozenset({'C', 'I', 'S', 'E', 'A', 'R'}),
    'B': frozenset({'C', 'I', 'S', 'E', 'A', 'R'}),
    'g': frozenset({'C', 'I', 'S', 'E', 'A', 'R'}),
    'S': frozenset({'C', 'E', 'R'}),
    'C': frozenset({'C', 'I', 'S', 'R'}),
    'R': frozenset({'C', 'I', 'S', 'R'}),
}
_INFLUENCE_ROUTINE = {
    'G': 'InfluenceGeoHatCart', 'B': 'InfluenceGeoGaussianCart',
    'g': 'InfluenceGeoHatRayCen', 'S': 'InfluenceSGB',
    'C': 'InfluenceCervenyCart', 'R': 'InfluenceCervenyRayCen',
}
# Beam types whose influence routine re-reads the paired receiver depth when
# RunType(5:5)=='I' (influence.f90:461-465 and :581-585). The other four index
# the depth by the depth-loop counter, which bellhop.f90:202-204 has pinned to
# NRz_per_range == 1, so every paired receiver is evaluated at Pos%Rz(1).
_IRREGULAR_GRID_BEAM_TYPES = frozenset({'G', 'B'})
# Beam types that form the receiver index as
# INT( ( r - Pos%Rr(1) ) / Pos%Delta_r ) + 1 — influence.f90:92 ('R'), :223-224
# ('C'), :339 and :351 ('g'), each flagged "assumes uniform spacing in Pos%r".
# Pos%Delta_r is only the *last* gap (SourceReceiverPositions.f90:160), so an
# unevenly spaced range axis sends every step to the wrong column. 'G', 'B' walk
# ir with a bracket test and 'S' compares rB > Pos%Rr(ir) directly, so all three
# take an arbitrary range vector — doc/bellhop.htm's list is wrong twice, both
# including 'R' and omitting 'S'.
_UNIFORM_RANGE_BEAM_TYPES = frozenset({'g', 'C', 'R'})

# RunType(1:1) letters a beam type's influence routine implements but the
# wrapper refuses, because the level they return is not a transmission loss.
# The Cerveny routines square each beam's contribution for 'I'/'S'
# (influence.f90:140 ray-centred, :282 Cartesian), ScalePressure takes the
# root of the sum (:779) and scales it by const = -Dalpha*SQRT(freq)/c
# (:772, :774), which is linear in Dalpha. The sum grows as the beam count N,
# its root as sqrt(N), and Dalpha as 1/N, so the pressure falls as
# N**-0.5: 4.77 dB per tripling of n_beams.
_BEAM_TYPE_REFUSED_RUN_TYPES = {
    'C': frozenset({'I', 'S'}),
    'R': frozenset({'I', 'S'}),
}


def grid_is_paired(grid_type) -> bool:
    """``grid_type='I'`` writes RunType(5:5)='I', where BELLHOP walks the
    depth and range arrays together (one receiver per index) rather than
    over their Cartesian product — the pairing the ``'I'`` deck also
    enforces by requiring equal lengths
    (:func:`~uacpy.models.bellhop._checks.reject_unequal_paired_grid`).
    BELLHOP sorts each array on read (``SourceReceiverPositions.f90:224``),
    so the pairs are always a monotone diagonal.
    """
    return str(grid_type).upper() == 'I'
