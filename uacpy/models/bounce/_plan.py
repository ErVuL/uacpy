"""What one BOUNCE deck is resolved to before it is written: the
phase-velocity window ``[c_low, c_high]``, the tabulated-angle count ``NkTab``,
the range ``RMax`` the angular sampling is sized for and the mesh of each
sediment medium, from the knobs and the environment; and the refusals of a
deck the binary cannot run."""

from typing import Optional, Tuple

import numpy as np

from uacpy.core.environment import Environment
from uacpy.core.exceptions import ConfigurationError, UnsupportedFeatureError
from uacpy.core.deck_limits import NO_RECEIVER_RANGE_FALLBACK
from uacpy.core.units import m_to_km
from uacpy.io.oalib_writer import writable_layers
from uacpy.models._defaults import DEFAULT_C_MIN
from uacpy.models._window import DEFAULT_C_MAX_UNBOUNDED
from uacpy.models._knobs import is_real_number

# bounce.f90 zeroes kMin (drops the 1/cHigh term in NkTab) once cHigh > 1e6.
_KMIN_CUTOFF_CHIGH = 1.0e6

# Mesh density of each medium of the sediment stack, floored so a thin layer
# still gets a usable mesh; bounce.f90:78 allocates the difference-equation grid
# (B1..B4/rho/cP/cS) to the deck's count, so no cap applies.
#: BOUNCE's own manual asks for this, and it is NOT the generic AT auto-mesh
#: density. ``doc/bounce.htm``: "BOUNCE is very fast, there's no reason to
#: skimp. The finer the grid, the more accurate the result. I'll suggest
#: perhaps 100 points/wavelength as a good balance between run time and
#: accuracy. I have seen cases where 10 points/wavelength gave very poor
#: accuracy in R( theta )."
#:
#: 10/wavelength is exactly the binary's own acceptance floor — AT sizes
#: ``Nneeded`` at 20/wavelength (``ReadEnvironmentMod.f90:103``) and rejects a
#: deck below ``Nneeded/2`` — so the density BOUNCE will just barely accept is
#: the one its manual calls very poor. Measured on a 20 m sediment layer over
#: a half-space at 200 Hz, against a converged 400/wavelength reference:
#: max |dR| is 0.0049 at 20/wavelength, 0.0031 at 50, 0.00074 at 100 and
#: 0.00015 at 200. The cost is what the manual says it is — 0.02 s against
#: 0.03 s for the same run at 2 kHz.
_MESH_POINTS_PER_WAVELENGTH = 100
#: At least this many points per medium, uacpy guidance; BOUNCE's own
#: automatic mesh takes at least 10 (``misc/ReadEnvironmentMod.f90:105``).
_MIN_MESH_POINTS = 100


def check_knobs(*, c_low, c_high, rmax, n_angles) -> None:
    """Refuse a constructor knob no run could use: a value that is not a
    number, a non-positive phase velocity or range, a pinned ``c_low`` at
    or above ``c_high``, fewer than two angles, and ``n_angles`` pinned
    beside ``rmax`` (each sets the other).

    Run on the knobs at construction and again by every run (the attributes
    can be reassigned in between). A ``c_low`` left ``None`` is held to
    ``c_high`` once a run resolves it (:func:`check_phase_speed_window`).
    """
    for name, value, unit in (('c_low', c_low, 'm/s'),
                              ('c_high', c_high, 'm/s'),
                              ('rmax', rmax, 'm')):
        if value is None:
            continue
        if not is_real_number(value):
            raise ConfigurationError(
                f"Bounce({name}={value!r}): {name} must be a number "
                f"({unit}), or None "
                + ("for the default." if name == 'c_high' else
                   "to derive it at run time."))
    check_phase_speed_window(c_low, c_high=resolve_c_high(c_high)[0])
    if rmax is not None and float(rmax) <= 0:
        raise ConfigurationError(
            f"Bounce requires rmax > 0 (got {rmax}). RMax sets the "
            f"angular sampling density — bounce.f90:49 makes the number of "
            f"tabulated angles proportional to it, and "
            f"misc/ReadEnvironmentMod.f90:140 stops outright on a negative "
            f"value."
        )
    if n_angles is not None:
        if not (is_real_number(n_angles)
                and float(n_angles).is_integer()):
            raise ConfigurationError(
                f"Bounce(n_angles={n_angles!r}): n_angles is a count "
                f"of tabulated angles, a whole number >= 2.")
        if n_angles < 2:
            raise ConfigurationError(
                f"n_angles must be >= 2 (got {n_angles}). "
                f"bounce.f90:172 spaces the wavenumber grid as "
                f"Deltak = (kMax - kMin) / (NkTab - 1), so a single "
                f"tabulated angle divides by zero and the binary spins "
                f"until the model timeout."
            )
        if rmax is not None:
            raise ConfigurationError(
                f"Bounce(n_angles={n_angles}, rmax={rmax:g}) pins the "
                f"table's angular sampling twice: bounce.f90:49 derives the "
                f"count from RMax, NkTab = INT(1000 * RMax_km * (kMax - "
                f"kMin) / 2 pi), so n_angles sets RMax and the two "
                f"contradict each other.",
                remediation=("Pass n_angles= for the count, or rmax= for "
                             "the range the sampling is sized for, not "
                             "both."))


def resolve_c_high(c_high) -> Tuple[float, str]:
    """``(c_high, origin)``: the pinned upper phase velocity (m/s), or
    ``DEFAULT_C_MAX_UNBOUNDED`` (1e9 m/s) for ``None``, which trips
    ``bounce.f90:47``'s ``IF ( cHigh > 1.0E6 ) kMin = 0.0`` and so tabulates
    the full 0–90° grazing span."""
    if c_high is not None:
        return c_high, 'Bounce(c_high=…)'
    return (DEFAULT_C_MAX_UNBOUNDED,
            'DEFAULT_C_MAX_UNBOUNDED: the full 0–90° grazing span')


def check_phase_speed_window(c_low: Optional[float], *, c_high) -> None:
    """The phase-velocity window ``[c_low, c_high]`` BOUNCE can tabulate:
    ``c_high > 0`` always, and ``0 < c_low < c_high`` once ``c_low`` is
    known — the pinned value at construction, the resolved one in a run.
    No environment can make a non-positive phase velocity admissible.
    """
    if c_high <= 0:
        raise ConfigurationError(
            f"Bounce requires c_high > 0 strictly (got {c_high}). "
            "c_high is the largest phase velocity on the tabulated grid."
        )
    if c_low is not None:
        if c_low <= 0:
            raise ConfigurationError(
                f"Bounce requires c_low > 0 strictly (got {c_low}). "
                "c_low is the smallest phase velocity on the tabulated "
                "grid; 0 would give an infinite wavenumber."
            )
        if c_high <= c_low:
            raise ConfigurationError(
                f"c_high ({c_high}) must be strictly greater than "
                f"c_low ({c_low})."
            )


def refuse_unsized_table(receiver, *, rmax, n_angles) -> None:
    """Refuse ``receiver=None`` with nothing else to size the table:
    ``rmax`` and ``n_angles`` both unset."""
    if receiver is None and rmax is None and n_angles is None:
        raise ConfigurationError(
            "Bounce.run(receiver=None): a Receiver is required to "
            "auto-derive rmax (the range the reflection table is "
            "propagated to). Pass a Receiver, or pin the table with "
            "Bounce(rmax=...) or Bounce(n_angles=...)."
        )


def refuse_precalc_seabed(model_name: str, env: Environment) -> None:
    """Refuse a ``'precalc'`` (``.irc``) seabed, which the binary cannot
    re-read and rewrite in one launch."""
    seabed_type = env.bottom.halfspace_at(range=0.0).acoustic_type
    if seabed_type == 'precalc':
        raise UnsupportedFeatureError(
            model_name,
            "a 'precalc' (.irc) seabed — misc/RefCoef.f90:103-104 leaves "
            "xTab/fTab/gTab/iTab allocated for the table it just read, so "
            "bounce.f90:52 cannot allocate them for the table it is about "
            "to write and stops with 'Too many points in reflection "
            "coefficient'",
            alternatives=[
                "acoustic_type='file' with the equivalent .brc table, "
                "which BOUNCE reads through its own RBot array",
                "feed the .irc straight to Kraken or Scooter instead of "
                "re-running BOUNCE on it",
            ],
        )


def resolve_c_low(env: Environment, *, c_low: Optional[float]) -> float:
    """Effective ``c_low`` for this environment (m/s).

    AT's ``doc/bounce.htm``: "The angles used for calculating the
    reflection coefficient are calculated based on the phase-velocity
    interval [ CMin, CMax ]. For a full 90 degree calculation set CMin to
    the lowest speed in the problem (say 1400.0) CMax to 1.0E9." The rule
    is *the lowest speed in the problem*; the 1400 is that sentence's
    illustrative example. ``min(SSP)`` follows the rule for every speed
    that can tabulate a row: a phase speed below the seafloor water speed
    only adds ``theta = 0`` rows (below), so a slower sediment or shear
    speed, which the rule also names, would add nothing to the table.
    Capping it at ``DEFAULT_C_MIN`` keeps the tabulation grid at ``c_low = 1400``
    for every column whose water never drops below 1400 m/s —
    ``NkTab = rmax * f / c_low`` moves only once ``min(SSP)`` undercuts
    it.

    ``env.ssp.sound_speed``, not :meth:`~uacpy.core.ssp.SoundSpeedProfile.to_pairs`:
    that method returns the **range-0 column** of a range-dependent profile
    by contract (its own docstring says so), so it would read one column and
    miss a slower one further out. ``data`` is the full
    ``(n_depth, n_range)`` block, and every ``collapse['ssp']`` method is a
    per-depth reduction of those same columns, so ``min(data)`` is at or
    below the projected profile's minimum whichever method runs. The two
    agree for a 1-D profile and for the ``'r0'`` collapse default.

    Below the seafloor water speed this buys **head rows, not coverage**:
    ``bounce.f90:198-210`` computes ``theta`` only ``WHERE( k0 > kx )``
    with ``k0 = omega/HSTop%cP``, so every sample slower than the seafloor
    reference takes the ``ELSEWHERE`` branch and comes out as
    ``theta = 0, R = 1, phase = 180`` — byte-identical duplicate rows,
    which ``dedupe_reflection_file`` collapses (and ``stage_reflection_file``
    runs over every staged table). Taking the water's minimum rather than
    the literal lowest speed of every medium therefore changes no row of
    the table, and the water rows below the seafloor speed cost only rows
    that are already removed downstream.
    """
    if c_low is not None:
        return float(c_low)
    return min(DEFAULT_C_MIN, float(env.ssp.sound_speed.min()))


def reject_c_low_above_the_water(env: Environment, c_low: float) -> None:
    """``c_low`` above the water sound speed drops the grazing wedge.

    ``bounce.f90:46`` sets ``kMax = omega/cLow`` and ``:195`` takes
    ``k0 = omega/c0`` with ``c0 = HSTop%cP`` — which
    ``write_bounce_input_file`` fills with the water sound speed at the
    seafloor. ``:198-210`` computes ``theta = ATAN2( kz, kx )`` only
    ``WHERE( k0 > kx )``, so when ``cLow > c0`` the table simply starts at
    ``ATAN2( sqrt(k0**2 - kMax**2), kMax ) > 0`` instead of 0 deg. Every
    consumer then substitutes ``R = 0, phi = 0`` below that first angle
    (``misc/RefCoef.f90:137-141``, whose warning goes to the ``.prt`` only),
    and ``Bellhop/bellhop.f90:688-693`` applies it as
    ``ray2D%Amp = Amp * RInt%R``, annihilating the ray on its first bounce.
    ``c_low`` 1.3 % above the water speed already costs a mean 5.1 dB /
    max 25 dB against the same environment run as a direct half-space.
    """
    c_ref = float(np.atleast_1d(env.ssp.sound_speed_at(env.depth))[0])
    if c_low <= c_ref:
        return
    theta_min = np.degrees(np.arctan2(
        np.sqrt(max(1.0 / c_ref ** 2 - 1.0 / c_low ** 2, 0.0)),
        1.0 / c_low))
    raise ConfigurationError(
        f"Bounce(c_low={c_low}) exceeds the water sound speed at the "
        f"seafloor ({c_ref:.1f} m/s), which BOUNCE uses as its reference "
        f"speed: the table would start at {theta_min:.2f} deg grazing "
        f"instead of 0, and every consumer silently reads R = 0 below "
        f"that.",
        remediation=f"Set c_low <= {c_ref:.1f} m/s, or leave it None so "
                    f"uacpy derives min({DEFAULT_C_MIN:.0f}, min(env.ssp)) "
                    f"— AT bounce.htm's 'lowest speed in the problem', "
                    f"which covers the full grazing span for any column; "
                    f"to concentrate the samples on a narrower band, "
                    f"lower c_high instead.",
    )


def resolve_rmax(receiver, frequency: float, c_low: float, *, n_angles,
                 rmax, c_high) -> Tuple[float, str]:
    """``(rmax, origin)``: the range (m) the table's angular sampling
    is sized for, and where it came from.

    A pinned ``n_angles`` is inverted through bounce.f90:49
    ``NkTab = INT(1000*RMax_km*(kMax-kMin)/(2π))`` to the RMax half an
    angle above it, so the binary tabulates exactly ``n_angles``
    (:func:`tabulated_angle_count` on the deck's six-decimal km); a count
    that RMax cannot give is refused, naming the count it gives. Else the
    pinned ``rmax``, then ``receiver.range_max``, and 10 km when no
    receiver range is positive. ``receiver=None`` reaches here only with
    one of the two knobs set (:func:`refuse_unsized_table`).
    """
    if n_angles is not None:
        omega = 2.0 * np.pi * float(frequency)
        inv_c_diff = 1.0 / c_low
        if c_high <= _KMIN_CUTOFF_CHIGH:
            inv_c_diff -= 1.0 / float(c_high)
        if omega * inv_c_diff <= 0:
            raise ConfigurationError(
                f"Cannot derive rmax from n_angles={n_angles}: "
                f"omega·(1/cLow - 1/cHigh) is non-positive "
                f"(omega={omega:.3g}, 1/cLow-1/cHigh={inv_c_diff:.3g})."
            )
        # NkTab = INT(1000 * RMax_km * (kMax - kMin) / 2π) and
        # 1000 * RMax_km IS RMax in metres, so the km conversion cancels
        # out of the inversion and this lands directly in metres. Half an
        # angle above n_angles keeps INT on n_angles through the deck's
        # millimetre rounding of RMax.
        rmax = ((int(n_angles) + 0.5) * 2.0 * np.pi
                / (omega * inv_c_diff))
        got = tabulated_angle_count(rmax, frequency, c_low, c_high=c_high)
        if got != int(n_angles):
            raise ConfigurationError(
                f"Bounce(n_angles={n_angles}): bounce.f90:49 derives the "
                f"count from RMax, which the deck carries in km at six "
                f"decimals, and at {frequency:.6g} Hz with c_low={c_low:g} "
                f"and c_high={c_high:g} m/s the nearest RMax gives "
                f"{got} angles, not {n_angles}.",
                remediation=(f"Pass n_angles={got}, or rmax= for the "
                             f"range the sampling is sized for."))
        return rmax, f'n_angles={n_angles}'
    if rmax is not None:
        return float(rmax), 'Bounce(rmax=…)'
    recv_rmax = float(receiver.range_max)
    if recv_rmax > 0:
        return recv_rmax, 'receiver.range_max'
    return (NO_RECEIVER_RANGE_FALLBACK,
            f'the {NO_RECEIVER_RANGE_FALLBACK / 1000.0:g} km fallback '
            f'(no receiver range)')


def tabulated_angle_count(rmax: float, frequency: float, c_low: float, *,
                          c_high) -> int:
    """Angles BOUNCE will tabulate for this deck.

    Reproduces ``bounce.f90:45-49`` on the RMax string the writer is about
    to emit: ``kMin = omega / cHigh`` (zeroed at :47 once ``cHigh > 1e6``),
    ``kMax = omega / cLow`` and
    ``NkTab = INT( 1000 * RMax_km * ( kMax - kMin ) / 2 pi )``.

    The ``.6f`` round-trip is the deck's own precision: ``RMax`` is written
    in km at six decimals (``oalib_writer.write_phase_speed_and_rmax``), so
    the count has to come off the rounded value the binary will read back,
    not off the exact metres.
    """
    rmax_km = float(f"{m_to_km(rmax):.6f}")
    omega = 2.0 * np.pi * float(frequency)
    k_min = 0.0 if c_high > _KMIN_CUTOFF_CHIGH else omega / c_high
    k_max = omega / c_low
    return int(1000.0 * rmax_km * (k_max - k_min) / (2.0 * np.pi))


def refuse_too_few_angles(n_angles: int, frequency: float, c_low: float, *,
                          rmax: float, rmax_origin: str, c_high) -> None:
    """Refuse a deck that tabulates fewer than two angles: none writes an
    empty table, and one divides by zero in the binary."""
    if n_angles >= 2:
        return
    consequence = (
        "BOUNCE would write an empty reflection-coefficient table"
        if n_angles == 0 else
        "bounce.f90:172 spaces the grid as "
        "Deltak = (kMax - kMin) / (NkTab - 1), so a single angle "
        "divides by zero and the binary spins until the model timeout"
    )
    raise ConfigurationError(
        f"This deck asks for {n_angles} tabulated angle(s): "
        f"bounce.f90:49 derives NkTab = INT(1000 * RMax_km * "
        f"(kMax - kMin) / 2 pi) from RMax = {rmax:g} m at "
        f"{frequency:.4g} Hz with c_low={c_low:g} and "
        f"c_high={c_high:g} m/s. {consequence}.",
        remediation=(
            f"rmax came from {rmax_origin} — raise it (or raise "
            f"n_angles), or widen the phase-speed window so more "
            f"wavenumbers fall inside it."
        ),
    )


def resolve_n_mesh(env: Environment, frequency: float,
                   n_mesh=None) -> list:
    """Mesh-point count for each medium of the sediment stack: a pinned
    ``n_mesh`` as given (one count for every medium, or one per medium),
    else the rule below.

    ``write_bounce_input_file`` omits the water column, so the media are
    exactly the writable sediment layers (a bare half-space seabed carries
    none and returns an empty list).

    ``misc/ReadEnvironmentMod.f90:101-112`` sizes every medium
    independently: ``c = alphaR``, then ``IF ( betaR > 0.0 ) c = betaR``,
    ``deltaz = c / freq / 20``, ``Nneeded = INT( thickness / deltaz )``,
    and it aborts with *Mesh is too coarse* when the deck asks for fewer
    than ``Nneeded / 2``. The meshing speed is therefore the medium's
    **shear** speed wherever it has one — an ordinary sand (cs ~ 200 m/s
    against cp ~ 1700 m/s) needs an order of magnitude more points than its
    compressional wavelength suggests.

    ``doc/bounce.htm`` states that automatic rule ("If you set the number
    of finite-difference grid points (NPTS) to zero, then BOUNCE
    automatically calculates a value to have 20 points per wavelength. ...
    you may need to increase that to as much as 100 points per
    wavelength.") and elsewhere suggests 100 points/wavelength, calling 10
    "very poor", where the Fortran merely sets the acceptance floor at that
    10. This wrapper asks for the manual's 100 — see
    :data:`_MESH_POINTS_PER_WAVELENGTH`.

    A pinned count is refused where BOUNCE's reader would stop on it: a
    list whose length is not the number of media, or a positive count
    under ``Nneeded / 2`` of its medium (``Mesh is too coarse``,
    ``ReadEnvironmentMod.f90:105-112``, with ``Nneeded = MAX(INT(
    thickness / (c / freq / 20)), 10)``); ``0`` is the reader's automatic
    20 points per wavelength (``doc/bounce.htm``).
    """
    layers = writable_layers(env.bottom.at(range=0.0))
    if n_mesh is not None:
        counts = ([int(n) for n in n_mesh]
                  if isinstance(n_mesh, (list, tuple, np.ndarray))
                  else [int(n_mesh)] * len(layers))
        if len(counts) != len(layers):
            raise ConfigurationError(
                f"Bounce(n_mesh={list(counts)}) gives {len(counts)} "
                f"count(s) for the {len(layers)} medium/media of this "
                f"sediment stack (BOUNCE meshes the sediment layers only).",
                remediation="Give one count per sediment layer, or one "
                            "count for every layer.")
        for i, (layer, count) in enumerate(zip(layers, counts)):
            shear = float(layer.shear_speed)
            speed = shear if shear > 0.0 else float(layer.sound_speed)
            needed = max(int(20.0 * float(layer.thickness) * frequency
                             / speed), 10)
            if 0 < count < needed // 2:
                raise ConfigurationError(
                    f"Bounce(n_mesh=...) gives medium {i + 1} {count} mesh "
                    f"points, under the {needed // 2} "
                    f"misc/ReadEnvironmentMod.f90:110-112 requires at "
                    f"{frequency:g} Hz; the run would stop with 'Mesh is too "
                    f"coarse'.",
                    remediation=f"Pass at least {needed // 2} for medium "
                                f"{i + 1}, 0 for the reader's automatic "
                                f"mesh, or n_mesh=None.")
        return counts
    counts = []
    for layer in layers:
        thickness = float(layer.thickness)
        shear = float(layer.shear_speed)
        speed = shear if shear > 0.0 else float(layer.sound_speed)
        needed = _MESH_POINTS_PER_WAVELENGTH * thickness * frequency / speed
        # bounce.f90:78 allocates the mesh arrays to the deck's count, so
        # the manual's density is asked for at any size.
        counts.append(max(_MIN_MESH_POINTS, int(np.ceil(needed))))
    return counts
