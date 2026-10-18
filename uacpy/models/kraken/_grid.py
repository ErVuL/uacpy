"""The grids of a Kraken run: the mode-depth mesh (points per metre), the
merged receiver/tabulation depths, the maximum range, the range segments of a
field run and the size limits of field.exe's tabulation and ``.mod`` file."""

import math
import numpy as np
from typing import Optional, Tuple, Union
from uacpy.models._budget import memory_budget
from uacpy.core.run_settings import Notice
from uacpy.core.run_settings import RunMode
from uacpy.core.source import Source
from uacpy.core.constants import DEFAULT_SOUND_SPEED
from uacpy.core.exceptions import ConfigurationError, NumericsWarning
from uacpy.io.oalib_writer import (
    plan_multi_profile_media, at_mesh_floor, at_env_media,
    reject_coarse_at_mesh, deck_depth, SOURCE_TYPE_CODE as _SOURCE_TYPE_CODE,
)
from uacpy.models.kraken._settings import KrakenLaunch
from uacpy.models.kraken._segments import segment_environment_by_range
from uacpy.models.kraken._window import phase_speed_window


#: m — Kraken's deck RMax when no receiver range is positive (``compute_modes``
#: takes no receiver). uacpy guidance, measured: 0, ``doc/kraken.htm`` (8)'s
#: documented alternative ("If you set it to zero, then no mesh doublings will
#: be performed ... even if you set it to zero the results will usually be
#: reasonable"; ``tests/Munk/MunkK.env`` writes 0), ends the refinement on the
#: first mesh (``kraken.f90:80``). Against a converged run at 100 Hz to 10 km
#: it gave 0.27 dB median, 3.7 dB max |dTL| and a 1.09 rad mode phase error at
#: 10 km on a 100 m Pekeris guide (1700 m/s), 0.016 dB median, 0.17 dB max and
#: 0.24 rad over a 1550 m/s mud seabed, where 100 km gave none.
KRAKEN_RMAX_NO_RANGE_M = 100_000.0


#: ``KrakenField/field.f90:24`` declares ``MaxNfreq = 1000``, allocates
#: ``freqVec( MaxNfreq )`` (:164) and runs ``FreqLoop`` to that bound (:168);
#: ``KrakenField/ReadModes.f90:8,35`` repeats the constant and :187 reads
#: ``freqVec( 1 : Nfreq )`` with the ``Nfreq`` taken from the mode-file header.
#: The solver has no such cap (``misc/SourceReceiverPositions.f90:58``
#: allocates ``freqVec`` dynamically), so a longer grid is solved and written
#: happily and only overruns when field.exe reads the header back.
_FIELD_MAX_NFREQ = 1000

#: ``KrakenField/ReadModes.f90:8`` declares ``MaxN = 100001`` and reads the
#: mode file's tabulation depths into the static ``Z( MaxN )`` (:41, :188)
#: with no bound test; ``EvaluateCMMod.f90:6,136,242`` repeats the limit.
#: kraken.exe allocates its own table dynamically, so a longer grid is solved
#: and written, and field.exe then overruns: measured with 120005 depths, a
#: range-independent run completed with no message (the overrun lands on
#: live module variables), an adiabatic one died with SIGSEGV and a coupled
#: one with "Non-existing record number".
_FIELD_MAX_TABULATION_DEPTHS = 100001


# Sampling density for the mode TABULATION grid (the receiver-depth vector
# modes are written on), not for the internal medium mesh. Both manuals ask
# for this figure, and the vendored tree ships them as HTML:
#   ``doc/kraken.htm``: "Fine sampling (about 10 points/wavelength) is needed
#   to calculate the coupling integrals accurately."
#   ``doc/field.htm``: "a large number of receiver depths (NRD) when you do
#   the KRAKEN run. This number should be set to give about 10
#   points/wavelength."
#
# The INTERNAL mesh is a separate quantity with a different number, so the two
# must not be conflated: ``doc/EnvironmentalFile.htm`` says NMESH "should be
# about 10 per vertical wavelength in acoustic media. In elastic media ... 20
# per wavelength is a reasonable starting point", and
# ``misc/ReadEnvironmentMod.f90:103`` applies the conservative figure to every
# medium — ``deltaz = c / freq0 / 20``, "default sampling: 20 points per
# wavelength".
#
# The floor keeps a low-frequency run's grid at 1.5 points per metre or
# finer, whatever its wavelength (uacpy guidance, unmeasured; no manual
# sets one).
MODE_POINTS_PER_WAVELENGTH = 10.0
MODE_POINTS_PER_METER_FLOOR = 1.5


#: ``MergeVectors`` (``kraken.f90:573``) treats two tabulation depths as one
#: when they differ by less than ``100 * EPSILON`` of a single-precision REAL,
#: i.e. ~1.19e-5 m absolute — far looser than the ``%.6f`` the ``.env`` spells
#: depths at. Any grid this module builds has to separate its points by at
#: least this much, or KRAKEN silently merges a pair and the surviving one is
#: whichever came first.
_MERGE_VECTORS_TOL_M = 100.0 * float(np.finfo(np.float32).eps)


_RMAX_PINNED = 'Kraken(rmax=…)'


def merge_depths(grid, extra, max_depth):
    """``grid`` with every ``extra`` depth inside the domain merged in.

    ``extra`` wins every collision: a uniform grid point within
    :data:`_MERGE_VECTORS_TOL_M` of one is dropped rather than the other way
    round, because the whole point of merging them in is that *these* depths
    end up tabulated. Letting ``MergeVectors`` arbitrate instead leaves the
    survivor up to ordering, which put one receiver back on the interpolated
    path and left its value dependent on the source depths — 6.2e-6 relative,
    on exactly the one row whose interval a source depth split.
    """
    extra = np.atleast_1d(np.asarray(extra, dtype=float))
    extra = np.unique(extra[(extra >= 0.0) & (extra <= max_depth)])
    grid = np.asarray(grid, dtype=float)
    if not extra.size:
        return grid
    # The grid's own endpoints are load-bearing and are never displaced:
    # ``EvaluateCMMod.f90:312`` stops a coupled run outright unless
    # ``z(1) == depthT`` and ``z(NR) == depthB`` exactly (the invariant
    # ``io/oalib_writer`` documents), so a receiver a micron off the surface
    # or the bottom must snap onto the endpoint rather than replace it.
    ends = grid[[0, -1]] if grid.size else np.empty(0)
    for end in ends:
        extra = extra[np.abs(extra - end) > _MERGE_VECTORS_TOL_M]
    # Then drop the interior grid points a kept extra would collide with.
    if extra.size:
        idx = np.clip(np.searchsorted(extra, grid), 0, extra.size - 1)
        right = np.abs(grid - extra[idx])
        left = np.abs(grid - extra[np.clip(idx - 1, 0, extra.size - 1)])
        keep = np.minimum(right, left) > _MERGE_VECTORS_TOL_M
        if grid.size:
            keep[0] = keep[-1] = True
        grid = grid[keep]
    merged = np.concatenate([grid, extra])
    merged.sort()
    # Two *extras* can also sit inside MergeVectors' tolerance — receiver
    # depths are only required to differ by DECK_DEPTH_RESOLUTION_M, which is
    # ten times finer. Thin them here so the deck asks for a grid KRAKEN will
    # return unchanged; the endpoints are kept whatever else goes.
    keep = np.ones(merged.size, dtype=bool)
    last = -np.inf
    for i, value in enumerate(merged):
        if value - last > _MERGE_VECTORS_TOL_M:
            keep[i] = True
            last = value
        else:
            keep[i] = False
    keep[0] = True
    if merged.size > 1 and not keep[-1]:
        keep[-1], keep[-2] = True, False
    return merged[keep]


#: Estimated ``.mod`` size above which a run whose work directory is on
#: disk is told the file will be large.
_MOD_FILE_WARNING_BYTES = 2 * 1024 ** 3


def mode_points_per_meter(env, frequencies, *, pinned_mode_points_per_meter
                           ) -> Tuple[float, Optional[Notice]]:
    """``(points per metre, notice)`` of the mode tabulation grid: the
    pinned density, else the one derived from the run.

    ``kraken.htm`` block (9) and ``field.htm`` §(2) both require *"about
    10 points/wavelength"* on the receiver grid the modes are tabulated
    on, since that grid carries the mode shapes and the coupling
    integrals. A density fixed in pts/**metre** meets it only up to one
    frequency, so ``None`` derives it from the run instead, using the
    highest frequency and the slowest medium.

    Nothing downstream catches a grid that is too coarse:
    ``KrakenField/ReadModes.f90:78`` sets ``Tolerance = 1500/freq`` — a
    whole wavelength — so its "Modes not tabulated near requested pt."
    warning stays silent and the ``.prt`` is clean.

    An elastic layer's **shear** speed is included in the minimum, and
    NOT for the reason it looks like. No mode sample is ever taken
    inside an elastic medium: ``kraken.f90:266`` sizes the tabulation
    vector as ``SUM( N( FirstAcoustic : LastAcoustic ) )`` and the loop
    that lays out the depths, ``kraken.f90:560-565``, runs over that
    same acoustic span, so the grid stops at the last acoustic medium.
    The shear wavelength is not being resolved anywhere.

    What the term actually buys is a denser grid in the WATER COLUMN.
    ``c_s`` is typically the slowest speed in the whole problem — ~400
    m/s against 1500 m/s of water — so including it multiplies the
    density everywhere the grid does exist, and that is what field.exe
    interpolates its mode shapes and coupling integrals from. Measured on
    a 20 m elastic layer (``c_s`` = 400 m/s) over an elastic half-space
    at 200 Hz: keeping the term gives 5.0 pts/m against the 1.5 pts/m
    floor that dropping it would leave, and the two TL fields differ by
    a mean 0.033 dB and a max 0.050 dB. Small, but not zero — so the
    term stays, on this rationale rather than on the shear-sampling one.

    ``multi_profile_n_mesh`` and ``bounce._plan.resolve_n_mesh`` also read
    the shear speed, but for the different quantity that genuinely is
    sampled inside the elastic medium: AT's INTERNAL finite-difference
    mesh, which it sizes at ``c_s/(20·f)``
    (``misc/ReadEnvironmentMod.f90:101-103``).

    "Slowest medium" means slowest *anywhere in the deck*: one grid is
    tabulated for the whole multi-profile ``.env``, so the water term reads
    ``env.ssp.sound_speed`` — the full ``(n_depth, n_range)`` block — rather than
    :meth:`~uacpy.core.ssp.SoundSpeedProfile.to_pairs`, which returns the
    range-0 column by contract. The seabed term below already sweeps every
    range column of ``env.bottom``, so reading one SSP column would leave
    the two halves of the same reduction disagreeing about which ranges
    count.

    The block minimum is what the manuals' rule asks for, and its effect
    on the field is small. On a four-profile deck whose water
    runs 1500 m/s at r=0 to 1000 m/s at 10 km, at 2 kHz, the range-0
    reading gives 13.33 pts/m (6.7 points per wavelength in the slowest
    water, 2934 grid depths at dz = 0.0750 m) against 20 pts/m from the
    block minimum (4402 depths, dz = 0.0500 m) — and the two TL fields
    differ by a mean 0.004 dB / max 0.029 dB, with no receiver past
    0.1 dB. A grid sized on the range-0 column is genuinely under-sampled
    against the manuals' ~10 points/wavelength and one sized on the block
    minimum is not, but the error that buys is far from the several-dB
    regime a fixed 1.5 pts/m density produces.

    A pinned density under the manuals' figure is kept, with the notice
    the run's settings record and announce.
    """
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    freq_max = float(np.max(f)) if f.size else 0.0
    speeds = [float(env.ssp.sound_speed.min())]
    if env.bottom is not None:
        speeds.extend(s for s in env.bottom.all_sound_speeds() if s > 0)
        speeds.extend(
            s for s in (
                float(layer.shear_speed)
                for column in env.bottom.columns
                for layer in column.layers
            ) if s > 0
        )
    c_min = min(speeds) if speeds else DEFAULT_SOUND_SPEED
    needed = (MODE_POINTS_PER_WAVELENGTH * freq_max / c_min) if freq_max else 0.0
    if pinned_mode_points_per_meter is not None:
        ppm = float(pinned_mode_points_per_meter)
        notice = None
        if needed and ppm < needed:
            notice = Notice(
                f"mode_points_per_meter {ppm:g} is "
                f"{ppm * c_min / freq_max:.2g} points per wavelength at "
                f"{freq_max:g} Hz",
                f"Kraken(mode_points_per_meter={ppm:g}) gives "
                f"{ppm * c_min / freq_max:.2g} points per wavelength at "
                f"{freq_max:g} Hz, under the ~{MODE_POINTS_PER_WAVELENGTH:g} "
                f"the KRAKEN and FIELD manuals require for the mode "
                f"tabulation grid. Mode shapes and coupling integrals are "
                f"interpolated from this grid, so the error is silent. "
                f"Pass {needed:.3g} or leave mode_points_per_meter=None to "
                f"derive it.", NumericsWarning)
        return ppm, notice
    return max(MODE_POINTS_PER_METER_FLOOR, needed), None


def dense_mode_depths(env, source, *, pinned_mode_depths,
                      pinned_mode_points_per_meter):
    """``(depths, notice)``: the depth grid a MODES run with no receiver
    tabulates its modes on, and the notice of a pinned density under the
    manuals' figure (:func:`mode_points_per_meter`), or ``None``.

    The ``.mod`` eigenfunctions are sampled only where the ``.env`` asks for
    receivers, so this grid decides how much of each mode shape survives —
    a sparse one leaves too few samples per mode to plot or to reconstruct a
    field from. ``pinned_mode_depths`` verbatim when set, else
    ``max(100, total_depth * mode_points_per_meter)`` points linearly spaced
    from 0 to the total media depth (the 100 is uacpy guidance, unmeasured).

    ``env`` is the environment the caller passed. The modes are solved on
    its r = 0 profile, so the grid spans THAT water column plus THAT
    column's sediment stack — not the deepest point of a range-dependent
    bathymetry or the thickest bed along the track, either of which would
    ask KRAKEN for receivers below the profile it solves (it clamps them and
    warns about a grid this wrapper built).
    """
    if pinned_mode_depths is not None:
        return np.asarray(pinned_mode_depths, dtype=float), None
    bathy = env.bathymetry
    water = float(np.interp(0.0, np.asarray(bathy.ranges, dtype=float),
                            np.asarray(bathy.depths, dtype=float)))
    total_depth = water + (env.bottom.at(range=0.0).total_thickness()
                           if env.bottom.is_layered else 0.0)
    ppm, notice = mode_points_per_meter(
        env, source.frequencies,
        pinned_mode_points_per_meter=pinned_mode_points_per_meter)
    n_pts = max(100, int(round(float(total_depth) * ppm)))
    return np.linspace(0.0, float(total_depth), n_pts), notice


def compute_rmax(receiver) -> float:
    """The mode solver's RMax (m) from the receiver ranges: the farthest
    one — ``doc/kraken.htm`` (8), 678: "the largest range for which a field
    calculation will be desired" — or :data:`KRAKEN_RMAX_NO_RANGE_M` (100 km)
    when there is no receiver or no positive range.

    RMax scales the Richardson mesh-convergence test
    (``kraken.f90:80``: ``Error·1000·RMax < 1``): a larger value is a
    tighter tolerance (finer mesh, longer solve).
    """
    farthest = _farthest_range_m(receiver)
    return KRAKEN_RMAX_NO_RANGE_M if farthest is None else farthest


def _farthest_range_m(receiver) -> Optional[float]:
    """The farthest positive receiver range (m), ``None`` when there is no
    receiver or no positive range."""
    ranges = getattr(receiver, 'ranges', None) if receiver is not None else None
    if ranges is None:
        return None
    ranges = np.atleast_1d(np.asarray(ranges, dtype=float))
    farthest = float(np.max(ranges)) if ranges.size else 0.0
    return farthest if farthest > 0.0 else None


def rmax_zero_notice(rmax, *, model_name) -> Optional[Notice]:
    """``(note, warning)`` for an ``rmax`` of 0 the caller set, else
    ``None``: the mode solver's refinement loop ends on its first mesh
    (``kraken.f90:80`` / ``krakenc.f90:82``, ``Error*1000*RMax < 1``)."""
    if rmax is None or float(rmax) != 0.0:
        return None
    return Notice(
        "RMax 0: no mesh doubling",
        f"{model_name}(rmax=0): no mesh doublings will be performed "
        f"(doc/kraken.htm (8); kraken.f90:80 ends the refinement on the "
        f"first mesh).", NumericsWarning)


def resolve_rmax(receiver, *, pinned_rmax) -> Tuple[float, str]:
    """``(rmax, origin)``: the deck's ``RMax`` (:func:`compute_rmax`,
    or the caller's) and where it came from."""
    if pinned_rmax is not None:
        return float(pinned_rmax), _RMAX_PINNED
    rmax = compute_rmax(receiver)
    if _farthest_range_m(receiver) is None:
        return rmax, "100 km: the receiver has no positive range"
    return rmax, "the farthest receiver range (doc/kraken.htm (8))"


def resolve_modes_launch(env, source, receiver, *, tabulation_depths,
                         backend, collapse, leaky_modes, log, pinned_c_high,
                         pinned_c_low, pinned_n_mesh, pinned_rmax):
    """The one launch of a MODES run, or of a single-frequency mode
    count: the modes deck tabulating the modes at ``tabulation_depths``,
    with ``RMax`` from ``receiver.ranges`` (100 km with no receiver).
    Returns ``(launch, c_low_origin, c_high_origin, rmax_origin)``."""
    # A pinned n_mesh is checked at the deck's freq0, which is where AT
    # applies its own floor: misc/ReadEnvironmentMod.f90:103-112 sizes
    # Nneeded from freq0 alone, during the environment read, and
    # kraken.f90:75 then scales N with freq/freq0 for every swept
    # frequency — so a mesh that clears the floor at freq0 stays
    # proportionally as fine across the whole sweep. Testing max(freq)
    # instead rejected meshes the binary would have run.
    freq0 = float(np.atleast_1d(
        np.asarray(source.frequencies, dtype=float))[0])
    n_mesh = single_profile_n_mesh(env, pinned_n_mesh, freq0)
    rmax, rmax_origin = resolve_rmax(receiver,
                                     pinned_rmax=pinned_rmax)
    c_low, c_highs, c_low_origin, c_high_origin = \
        phase_speed_window(env, [env], backend=backend,
                           coupled=False, collapse=collapse,
                           leaky_modes=leaky_modes, log=log,
                           pinned_c_high=pinned_c_high,
                           pinned_c_low=pinned_c_low)
    launch = KrakenLaunch(
        deck_frequency=freq0,
        marched_frequencies=None,
        tabulation_depths=np.asarray(tabulation_depths, dtype=float),
        profile_ranges=None,
        c_low=c_low,
        c_high=c_highs,
        rmax=rmax,
        n_mesh=n_mesh,
        field_option=None,
    )
    return launch, c_low_origin, c_high_origin, rmax_origin


def single_profile_n_mesh(env, pinned_n_mesh, freq0
                          ) -> Union[int, Tuple[int, ...]]:
    """``NG`` of a single-profile deck from the caller's ``n_mesh``: 0 when
    it is unset (the binary sizes each medium), else the value as given — one
    count for every medium, or one per medium, water first — after the
    checks the reader stops on: one count per medium of the deck
    (:func:`at_env_media`), and no count under half of what AT sizes that
    medium at the deck's frequency line ``freq0``
    (``misc/ReadEnvironmentMod.f90:110-112``, *Mesh is too coarse*)."""
    if pinned_n_mesh is None:
        return 0
    if not isinstance(pinned_n_mesh, (list, tuple, np.ndarray)):
        reject_coarse_at_mesh('Kraken', int(pinned_n_mesh), env, freq0)
        return int(pinned_n_mesh)
    counts = tuple(int(n) for n in pinned_n_mesh)
    media = at_env_media(env)
    if len(counts) != len(media):
        raise ConfigurationError(
            f"Kraken(n_mesh={counts}) gives {len(counts)} count(s) for the "
            f"deck's {len(media)} media (the water column first, then each "
            f"sediment layer).",
            remediation="Pass one count per medium, one count for every "
                        "medium, or leave n_mesh unset.")
    for number, (count, medium) in enumerate(zip(counts, media), start=1):
        floor = at_mesh_floor([medium], freq0)
        if 0 < count < floor:
            raise ConfigurationError(
                f"Kraken(n_mesh={counts}) gives medium {number} {count} "
                f"points, below the {floor} misc/ReadEnvironmentMod.f90:110-112 "
                f"requires of it at {freq0:.4g} Hz; the run would stop with "
                f"'Mesh is too coarse'.",
                remediation=f"Pass at least {floor} for medium {number}, 0 "
                            f"to let KRAKEN size it, or leave n_mesh unset.")
    return counts


def band_mesh_floor_notice(counts, *, deck_frequency, lowest_frequency,
                           model_name) -> Notice:
    """``(note, warning)`` for an ``n_mesh`` of 0 the caller set on a band
    deck where :func:`band_n_mesh` would give ``counts``: some medium is
    solved on fewer than AT's 10 points at the band's lowest frequency."""
    freq0, lowest = float(deck_frequency), float(lowest_frequency)
    return Notice(
        f"n_mesh=0 leaves a medium under 10 points at {lowest:g} Hz",
        f"{model_name}(n_mesh=0): AT sizes each medium at the deck's "
        f"frequency line, {freq0:g} Hz, and solves the band's {lowest:g} Hz "
        f"on that count x {lowest:g}/{freq0:g} (kraken.f90:75, "
        f"krakenc.f90:77), under the 10 points per medium "
        f"misc/ReadEnvironmentMod.f90:105 requires. Leave n_mesh unset for "
        f"the counts that keep 10 points there ({counts}, water first).",
        NumericsWarning)


def multi_profile_n_mesh(segments, freq, *, pinned_n_mesh) -> int:
    """``NG`` mesh count written on every medium line of a multi-profile
    ``.env``.

    0 (the default ``n_mesh``) asks KRAKEN to size each medium of each
    profile itself, at 20 points per wavelength with a 10-point floor
    (``misc/ReadEnvironmentMod.f90:99-110``). Per-profile meshes are
    legal: the ``.mod`` record length has no mesh term
    (``kraken.f90:587`` / ``krakenc.f90:630`` — it is set by the
    frequency count, the source/receiver tabulation and the padded
    media count, all identical across profiles).

    A pinned ``n_mesh`` is honoured on every medium of every profile,
    after the ``Nneeded / 2`` floor check the same reader enforces.
    That floor is measured per medium off the media the deck actually
    carries (:func:`multi_profile_media`); crucially AT takes the
    wavelength from a medium's **shear** speed wherever one is set,
    and an elastic sediment's shear wavelength can be an order of
    magnitude shorter than its compressional one.
    """
    if isinstance(pinned_n_mesh, (list, tuple, np.ndarray)):
        raise ConfigurationError(
            f"Kraken(n_mesh={tuple(int(n) for n in pinned_n_mesh)}) gives one "
            f"count per medium; a range-dependent run's deck "
            f"(write_multi_profile_env) writes one count on every medium line "
            f"of every profile, and its profiles carry different media.",
            remediation="Pass one count for every medium, or leave n_mesh "
                        "unset.")
    if pinned_n_mesh is None or pinned_n_mesh <= 0:
        return 0
    floor = at_mesh_floor(multi_profile_media(segments), freq)
    if pinned_n_mesh < floor:
        raise ConfigurationError(
            f"Kraken(n_mesh={pinned_n_mesh}) is below the {floor} mesh "
            f"points misc/ReadEnvironmentMod.f90:110-112 requires for the "
            f"coarsest medium of this range-dependent environment at "
            f"{freq:.4g} Hz; the run would stop with 'Mesh is too coarse'.",
            remediation=f"Pass n_mesh >= {floor}, or leave n_mesh=0 to "
                        f"let KRAKEN size each medium of each profile "
                        f"itself.",
        )
    return int(pinned_n_mesh)


#: AT's least mesh count per medium, ``Nneeded = MAX( Nneeded, 10 )``
#: (``misc/ReadEnvironmentMod.f90:105``).
_AT_MESH_MIN_POINTS = 10


def band_n_mesh(env, *, deck_frequency, lowest_frequency
                ) -> Union[int, Tuple[int, ...]]:
    """``NG`` of a band deck whose ``n_mesh`` is unset: one count per
    medium, water first, or ``0`` when every medium is left to the binary.

    AT sizes each medium once, at the deck's frequency line ``freq0``:
    ``Nneeded = MAX( INT( thickness / ( c / freq0 / 20 ) ), 10 )``
    (``misc/ReadEnvironmentMod.f90:101-108``), and solves every frequency of
    the band on ``INT( NG * NV * freq / freq0 )`` points
    (``Kraken/kraken.f90:75``, ``Kraken/krakenc.f90:77``), so a frequency
    below ``freq0`` gets fewer points than the 10 the reader requires. A
    medium whose own ``Nneeded`` leaves fewer than 10 at the band's lowest
    frequency gets the least count that leaves 10 there, computed as the
    binary computes ``N``; every other medium gets 0 and the binary's own
    count. ``(thickness, c)`` per medium is :func:`at_env_media`.
    """
    freq0 = float(deck_frequency)
    lowest = float(lowest_frequency)
    count = math.ceil(_AT_MESH_MIN_POINTS * freq0 / lowest)
    while int(count * lowest / freq0) < _AT_MESH_MIN_POINTS:
        count += 1
    counts = tuple(
        count if count > max(int(20.0 * thickness * freq0 / c),
                             _AT_MESH_MIN_POINTS) else 0
        for thickness, c in at_env_media(env))
    return counts if any(counts) else 0


def multi_profile_media(segments):
    """``(thickness, speed)`` per medium across every profile of the deck.

    ``plan_multi_profile_media`` is the deck's geometry of record — it
    returns each profile's media already quantised to the written ``.6f``
    resolution and with the last one stretched onto the common bottom — so
    the mesh bound is read off it rather than re-derived. The water column
    of each profile is medium 1.
    """
    media = []
    for _range_m, seg in segments:
        seafloor = deck_depth(seg.depth)
        water = seg.ssp.extend_to(seafloor).to_pairs()
        media.append((seafloor, float(water[-1, 1])))

    _n_media, _bottom_depth, plans = plan_multi_profile_media(segments)
    for plan in plans:
        for top, bot, cp, cs, *_rest in plan:
            shear = float(cs or 0.0)
            media.append((float(bot) - float(top),
                          shear if shear > 0.0 else float(cp)))
    return media


def check_field_tabulation_size(mode_depths, source) -> None:
    """Refuse a mode table longer than field.exe's static ``MaxN``.

    kraken.exe tabulates the modes at the union of these depths and the
    source depths (``kraken.f90:573``), and that union is the ``NTot``
    field.exe reads into ``Z( MaxN )``; see
    :data:`_FIELD_MAX_TABULATION_DEPTHS`. The grid is refused rather than
    coarsened, because the automatic grid is sized by the wavelength
    rule the mode interpolation relies on.
    """
    n_tab = int(np.union1d(
        np.asarray(mode_depths, dtype=float),
        np.atleast_1d(np.asarray(source.depths, dtype=float))).size)
    if n_tab <= _FIELD_MAX_TABULATION_DEPTHS:
        return
    raise ConfigurationError(
        f"Kraken would tabulate the modes at {n_tab} depths, but "
        f"field.exe reads at most {_FIELD_MAX_TABULATION_DEPTHS} "
        f"(KrakenField/ReadModes.f90:8 MaxN) into a fixed array and "
        f"overruns it past that.",
        remediation=(
            "Lower mode_points_per_meter (the automatic grid is 10 points "
            "per wavelength at the top frequency and the slowest speed, "
            "shear included), lower the frequency, or cut the modelled "
            "depth (water plus sediment stack)."),
    )


def mode_file_size_notice(env, mode_depths, frequencies, *,
                          memory_backed: bool) -> Optional[Notice]:
    """The notice of the mode file this launch writes, or ``None``.

    In a work directory held in memory (``memory_backed``, from
    :func:`~uacpy.models._budget.work_dir_is_memory_backed`) the file is
    memory, so :func:`~uacpy.models._budget.memory_budget` weighs it and
    refuses one over what the host reports free; on disk an estimate past
    :data:`_MOD_FILE_WARNING_BYTES` is announced.

    kraken.exe writes one record per mode per frequency, each at least
    ``2 x NTot`` 4-byte words for the ``COMPLEX*8`` shape at every
    tabulation depth (``Kraken/kraken.f90:585-603``), so the file grows as
    (frequencies) x (modes) x (depths), and modes and depths both grow as
    depth x frequency. Measured on a 100 m Pekeris guide with 32 bins:
    1.7 MB at 500 Hz, 6.2 MB at 1 kHz, 23.6 MB at 2 kHz. The mode count
    is estimated as ``2*D*f/c_min`` (the propagating wavenumber span
    ``omega/c_min`` over the mode spacing ``pi/D``).
    """
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    depths = np.asarray(mode_depths, dtype=float)
    if not f.size or not depths.size:
        return None
    c_min = float(env.ssp.sound_speed.min())
    span = float(depths.max())
    n_modes = np.maximum(2.0 * span * f / c_min, 1.0)
    estimate = float(np.sum(n_modes)) * depths.size * 8.0
    detail = (
        f"The mode file for this run is estimated at "
        f"{estimate / 1024 ** 3:.1f} GiB ({f.size} frequencies, "
        f"{depths.size} tabulation depths, up to "
        f"{int(n_modes.max())} modes)")
    if memory_backed:
        return memory_budget(
            int(estimate), model_name='Kraken', what='mode file',
            detail=f"{detail}, written to a work directory held in memory.",
            remediation=(
                "Put the work directory on disk (work_dir=, or TMPDIR), or "
                "reduce the band, the frequency or mode_points_per_meter."))
    if estimate <= _MOD_FILE_WARNING_BYTES:
        return None
    return Notice(
        f"mode file estimated at {estimate / 1024 ** 3:.1f} GiB",
        f"Kraken: {detail}, written to the run's work directory on disk. "
        f"Reduce the band, the frequency or mode_points_per_meter to "
        f"shrink it.", NumericsWarning,
    )


def reject_band_over_field_limit(frequencies, *, model_name) -> None:
    """Refuse a native band longer than field.exe's ``MaxNfreq``
    (:data:`_FIELD_MAX_NFREQ`)."""
    # A source beam pattern on a multi-frequency run runs: field.f90
    # allocated its beam-pattern work arrays inside FreqLoop while the
    # matching DEALLOCATE sat after the loop, and uacpy's patch to that
    # block reallocates per frequency (MSrc may change with it) — see
    # third_party/MODIFICATIONS.md — so the pattern reaches every
    # frequency.
    if frequencies.size <= _FIELD_MAX_NFREQ:
        return
    raise ConfigurationError(
        f"{model_name}: {frequencies.size} frequencies exceed "
        f"field.exe's MaxNfreq = {_FIELD_MAX_NFREQ} "
        f"(KrakenField/field.f90:24). kraken.exe writes the longer "
        f"grid into the .mod header, then field.exe overruns its fixed "
        f"freqVec buffer reading it back.",
        remediation=(
            f"Pass at most {_FIELD_MAX_NFREQ} frequencies in "
            f"frequencies=, or for RunMode.TIME_SERIES shorten "
            f"output_duration / lower sample_rate so the auto-derived "
            f"grid is coarser. Longer bands can be run as successive "
            f"<= {_FIELD_MAX_NFREQ}-frequency passes and concatenated."
        ),
    )


def build_field_option(is_range_dependent: bool,
                        source: Source, run_mode: RunMode, *,
                        mode_coupling) -> str:
    """Build the 4-character option string for field.exe.

    Columns follow AT ``field.f90`` / ``ReadModes.f90``:

    * pos 1: source geometry from ``source.source_type`` — 'R' point
      source (cylindrical), 'X' line source (Cartesian), 'S' scaled
      point source.
    * pos 2: coupling — 'C' coupled modes, 'A' adiabatic.
      For NProf > 1 it is ``mode_coupling``; a range-independent run
      writes 'A', ``doc/field.htm``'s default ("'A' Adiabatic mode theory
      (default)."), which field.f90 does not read there: with NProf == 1
      it calls ``Evaluate`` (``field.f90:214-215``).
    * pos 3: source beam pattern — '*' when ``source.beam_pattern`` is
      set, else ' ' (omnidirectional). Field.exe rejects any other character
      (``field.f90:83-90``), so the elastic Comp selector (H/V/T/N)
      is not exposed here; it is only reachable if a user invokes
      ReadModes directly.
    * pos 4: 'C' coherent TL, 'I' incoherent — from ``run_mode``
      (``RunMode.INCOHERENT_TL`` is the only 'I' case; BROADBAND /
      TIME_SERIES are coherent by construction).
    """
    # Source geometry letter lands in field.exe Opt(1:1), field.f90:70-79.
    pos1 = _SOURCE_TYPE_CODE[source.source_type]
    if is_range_dependent:
        pos2 = 'C' if mode_coupling.lower() == 'coupled' else 'A'
    else:
        # Range-independent: field.htm's default; field.f90:214-215 calls
        # Evaluate when NProf == 1 and reads neither 'A' nor 'C'.
        pos2 = 'A'
    # pos3: '*' => field.exe reads <base>.sbp, else omnidirectional.
    pos3 = '*' if source.beam_pattern is not None else ' '
    pos4 = 'I' if run_mode == RunMode.INCOHERENT_TL else 'C'
    return f"{pos1}{pos2}{pos3}{pos4}"


def segment_env_for_field(env, freq=None, *, log, mode_coupling, n_segments):
    """Segment a range-dependent env into per-range profiles for the
    multi-profile kraken field run.

    Returns ``(segments, n_profiles, profile_ranges, max_total_depth)``.
    ``max_total_depth`` is the shared bottom ``write_multi_profile_env``
    will declare, taken from the same planner the writer uses so the two
    cannot drift apart: the mode-tabulation grid is built to span
    ``[0, max_total_depth]``, and ``EvaluateCMMod.f90:312-316`` stops a coupled
    run outright unless that grid ends *exactly* on the declared bottom.
    """
    segments = segment_environment_by_range(
        env, n_segments=n_segments, freq=freq)
    n_profiles = len(segments)
    _, max_total_depth, _ = plan_multi_profile_media(segments)

    profile_ranges = np.array([s[0] for s in segments])
    log(f"Range-dependent: {n_profiles} profiles, "
              f"mode_coupling={mode_coupling}")
    return segments, n_profiles, profile_ranges, max_total_depth
