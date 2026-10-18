"""The phase-speed window ``[c_low, c_high]`` an Acoustics-Toolbox deck
searches or integrates over: the default rule the engines derive an
unpinned bound from, the window a wavenumber integral resolves with the
origin of each bound, the check that a pinned pair is ordered, and what a
derived window leaves out — the steep paths its ``c_high`` cuts, the seabed
interface wave below its ``c_low`` — with the notices that state it.

The rules an engine derives on its own (Kraken's ``c_low`` of 0 and its
reflection-table ``c_high``, BOUNCE's ``min(SSP)``) stay in that engine; what
is here is what more than one engine reads.
"""

from typing import NamedTuple, Optional, Tuple

import numpy as np

from uacpy.core.boundary import BoundaryType
from uacpy.core.engine_defaults import default_c_low
from uacpy.core.environment import Environment
from uacpy.core.exceptions import ConfigurationError, NumericsWarning
from uacpy.core.run_settings import Notice
from uacpy.models._knobs import positive_finite

# The unpinned ``c_low`` is the wavenumber-integration default every engine
# shares (:func:`uacpy.core.engine_defaults.default_c_low`, with its sources):
# Scooter / SPARC pin ``k_max = omega/c_low``, so a positive floor is required
# (c_low=0 blows up the integral).
#
# ``C_HIGH_FACTOR`` pads the upper bound of the mode search over a half-space:
# :func:`resolve_phase_speed_bounds` takes ``c_high = 1.05 · max(c_max, bottom
# cp)`` so the fastest speed in the problem sits strictly inside the interval
# rather than on its edge. The 1.05 is uacpy guidance, measured on Kraken
# (models/kraken/_window.py); doc/kraken.htm leaves CHIGH to "a matter of
# experience". The wavenumber integrals (Scooter, SPARC) do not use it: their
# unpinned c_high is unbounded, k_min = 0, as Jensen et al. (*Computational
# Ocean Acoustics*, ch. 4 App. 1) recommend (:func:`resolve_window`).
# Kraken only searches [c_low, c_high] for roots, so
# there the pad means a default run returns a few modes with a phase speed
# above the bottom speed: on kraken.exe roots of a rigid seabed
# (``Kraken/BCImpedanceMod.f90:83-88``), which a run that chose kraken.exe or
# a c_high above the seabed speed says (``models/kraken/_checks.py``,
# ``kraken_exe_above_seabed_notice``; a default run says nothing), on krakenc
# leaky modes. ``models/kraken/_modes.py`` logs the count, and a field run whose
# every mode is one of them warns (``models/kraken/_model.py``, ``_warn_leaky``).
# (KRAKEN's modal-solver c_low default is the literal 0.0 written at its use
# site in ``models/kraken/_window.py`` — 0 makes KRAKEN compute the bound
# itself, per
# kraken.htm, Phase Speed Limits.)
C_HIGH_FACTOR = 1.05

# "No upper phase-speed limit" for the AT family. A vacuum / rigid boundary
# traps every mode, so the mode search must not be capped on a half-space
# speed that does not exist. Acoustics-Toolbox/doc/bounce.htm prescribes the
# value: "For a full 90 degree calculation set CMin to the lowest speed in the
# problem (say 1400.0) CMax to 1.0E9." bounce.f90:47 takes kMin = 0 above 1e6.
# Kraken's ``leaky_modes`` uses 10 x the fastest speed instead
# (``models/kraken/_window.py``, ``_LEAKY_C_HIGH_FACTOR``).
DEFAULT_C_MAX_UNBOUNDED = 1.0e9


def resolve_phase_speed_bounds(
    env: Environment,
    c_low: Optional[float] = None,
    c_high: Optional[float] = None,
) -> Tuple[float, float]:
    """Resolve effective ``(c_low, c_high)`` for an AT-family run.

    Precedence:
      1. Explicit caller values win.
      2. Otherwise: ``c_low`` from
         :func:`~uacpy.core.engine_defaults.default_c_low` and
         ``c_high = max(c_max, env.bottom.halfspace_at(range=0).sound_speed) · C_HIGH_FACTOR``.

    A **non-geoacoustic** bottom (vacuum, rigid, or a reflection table —
    'file'/'precalc') carries no physical sound speed — modes above the
    half-space speed are leaky only when there *is* a half-space to leak into,
    and a parameter-free ``BoundaryProperties`` still carries the placeholder
    ``sound_speed`` its constructor defaults to. Capping on that placeholder
    silently truncates the mode spectrum (a 100 m rigid-bottom guide at 50 Hz
    keeps 3 of its 7 modes, a 10.6 dB error; a 'file' bottom's 1600 m/s
    placeholder capped cHigh at 1680 m/s), so those boundaries resolve to
    :data:`DEFAULT_C_MAX_UNBOUNDED` instead — the AT idiom for "no upper
    limit".

    The derived ``c_low`` is the documented rule: the slowest speed of
    every medium, shear speeds and half-spaces included, × 0.85 over an
    elastic half-space and × 0.90 otherwise (``scooter.f90:193-259``;
    Jensen et al., *Computational Ocean Acoustics*, ch. 4 App. 1 step 1,
    "the lowest wave speed in the problem"). Over an elastic seabed it
    reaches below the shear speed, so the seabed interface wave is in the
    window, at the cost of wavenumbers: about 14× on elastic sand (``Nk``
    grows as ``1/c_low``, 1341/93.5 m/s; docs/models/scooter.md, "Layered
    and elastic seabeds", measures the run 14× slower).
    """
    if c_low is not None and c_high is not None:
        return float(c_low), float(c_high)
    ssp_pairs = env.ssp.to_pairs()
    halfspace = env.bottom.halfspace_at(range=0.0)
    if not BoundaryType.from_string(halfspace.acoustic_type).is_geoacoustic:
        c_high_auto = DEFAULT_C_MAX_UNBOUNDED
    else:
        c_max = max(float(ssp_pairs[:, 1].max()),
                    float(halfspace.sound_speed))
        c_high_auto = c_max * C_HIGH_FACTOR
    return (
        float(c_low) if c_low is not None else default_c_low(env)[0],
        float(c_high) if c_high is not None else c_high_auto,
    )


class PhaseSpeedWindow(NamedTuple):
    """A resolved phase-speed window (m/s) and where each bound came
    from, as the engine settings record it."""
    c_low: float
    c_high: float
    c_low_origin: str
    c_high_origin: str


def check_pinned_window(model_name: str, *, c_low, c_high) -> None:
    """Refuse a pinned bound that is not a positive finite speed (the
    integral's ``k_max = omega/c_low`` and ``k_min = omega/c_high`` need
    one), and two pinned bounds with ``c_low >= c_high``. A single pinned
    bound is held to the other once a run derives it
    (:func:`refuse_inverted_window`)."""
    positive_finite(f"{model_name}: c_low (m/s)", c_low, optional=True)
    positive_finite(f"{model_name}: c_high (m/s)", c_high, optional=True)
    if (c_low is not None and c_high is not None
            and c_low >= c_high):
        raise ConfigurationError(
            f"{model_name} spectral phase-velocity band requires "
            f"c_low < c_high; got c_low={c_low} m/s, "
            f"c_high={c_high} m/s."
        )


def refuse_inverted_window(model_name: str, cl: float, ch: float, *,
                           c_low, c_high) -> None:
    """Refuse a resolved window ``[cl, ch]`` with ``cl >= ch``, naming the
    bound the caller pinned (``c_low`` / ``c_high``, ``None`` when derived),
    since that is the one to move."""
    if cl < ch:
        return
    if c_low is not None and c_high is None:
        pinned = (f"the pinned c_low={c_low} m/s is at or above "
                  f"the env-derived c_high = {ch:.1f} m/s")
    elif c_high is not None and c_low is None:
        pinned = (f"the pinned c_high={c_high} m/s is at or "
                  f"below the env-derived c_low = {cl:.1f} m/s")
    else:
        pinned = (f"the env-derived band collapsed to c_low = "
                  f"{cl:.1f} m/s, c_high = {ch:.1f} m/s")
    raise ConfigurationError(
        f"{model_name} spectral phase-velocity band requires "
        f"c_low < c_high: {pinned}.",
        remediation="Widen the pinned bound, or leave both unset to "
                    "derive the band from the SSP and bottom.",
    )


def resolve_window(env, *, c_low, c_high, c_high_unbounded: float,
                   model_name: str) -> PhaseSpeedWindow:
    """The window of a wavenumber integral (Scooter, SPARC): the pinned
    bounds; an unpinned ``c_low`` from :func:`resolve_phase_speed_bounds`
    on the projected ``env`` (the slowest speed of every medium × 0.85 over
    an elastic half-space, × 0.90 otherwise,
    :func:`~uacpy.core.engine_defaults.default_c_low`); an unpinned ``c_high``
    of ``c_high_unbounded`` over every seabed — the engine's own "no upper
    limit", which puts ``k_min = omega/c_high`` at 0 to every grazing angle
    short of 90° — as Jensen et al. (*Computational Ocean Acoustics*, ch. 4
    App. 1 step 1) recommend: "it is recommended to initially include these
    components by choosing k min = 0"; with the origin of each.

    The constructor can only compare two pinned bounds
    (:func:`check_pinned_window`). A single pinned bound is only comparable
    once the other has been derived from this env, and an inverted pair
    reaches ReadEnvironmentMod.f90:135, which stops the binary after the
    deck has been written and the process spawned. It is refused here
    instead, naming the bound the user pinned, since that is the one to
    move.
    """
    cl, ch = resolve_phase_speed_bounds(
        env, c_low, c_high_unbounded if c_high is None else c_high)
    refuse_inverted_window(model_name, cl, ch, c_low=c_low, c_high=c_high)
    if c_high is not None:
        high_origin = f"{model_name}(c_high=…)"
    else:
        high_origin = (f"unbounded ({c_high_unbounded:g} m/s): k_min = 0, "
                       f"every phase speed (Jensen et al., Computational "
                       f"Ocean Acoustics, ch. 4 App. 1)")
    return PhaseSpeedWindow(
        c_low=cl, c_high=ch,
        c_low_origin=(f"{model_name}(c_low=…)" if c_low is not None
                      else default_c_low(env)[1]),
        c_high_origin=high_origin)


def steep_path_cut(c_water: float, c_high: float, source_depths,
                   receiver_depths, receiver_ranges
                   ) -> Optional[Tuple[float, float]]:
    """``(cut, steepest)`` in degrees when a receiver's direct or
    surface-reflected path is steeper than ``c_high`` keeps, else ``None``.

    A path at grazing angle ``θ`` in water of speed ``c`` has horizontal
    phase speed ``c / cos θ``; a spectrum cut at ``c_high`` keeps it only
    while that is below ``c_high``, i.e. ``θ <= arccos(c / c_high)`` (the
    fastest water speed makes this the tightest cut). A steeper path is
    simply absent from the field, with no sign of it in the result. One
    rule for every engine, so the geometry cannot drift between them.

    ``cut`` is ``arccos(c_water / c_high)``; ``steepest`` is the steepest
    direct or surface-reflected path from any source depth to any receiver
    at the closest positive range. ``None`` also when ``c_high`` is not
    above ``c_water`` or no receiver range is positive.
    """
    if not (c_high > c_water):
        return None
    cut = float(np.degrees(np.arccos(c_water / c_high)))
    zs = np.atleast_1d(np.asarray(source_depths, dtype=float))
    zr = np.atleast_1d(np.asarray(receiver_depths, dtype=float))
    rr = np.atleast_1d(np.asarray(receiver_ranges, dtype=float))
    rr = rr[rr > 0.0]
    if not rr.size:
        return None
    steepest = 0.0
    for s in zs:
        direct = np.abs(zr[:, None] - s)
        surface = zr[:, None] + s
        depth_span = np.maximum(direct, surface)
        steepest = max(steepest, float(np.degrees(np.arctan(
            depth_span.max() / rr.min()))))
    if not steepest > cut:
        return None
    return cut, steepest


def steep_path_clear_range(source_depths, receiver_depths,
                           cut: float) -> int:
    """The range (m, rounded up to a whole metre) from which no direct or
    surface-reflected path between ``source_depths`` and
    ``receiver_depths`` is steeper than ``cut`` degrees: the deepest
    surface-reflected span over ``tan(cut)``."""
    zs = np.atleast_1d(np.asarray(source_depths, dtype=float))
    zr = np.atleast_1d(np.asarray(receiver_depths, dtype=float))
    span = float(np.max(zr[:, None] + zs[None, :]))
    return int(np.ceil(span / np.tan(np.radians(cut))))


def steep_path_notice(env, source, receiver, c_high: float, *,
                      model_name: str, kept: str, summed_in: str,
                      evidence: str, remediation: str,
                      pinned: bool = False) -> Optional[Notice]:
    """``(note, warning)`` when a receiver's direct or surface-reflected
    path is steeper than ``c_high`` keeps (:func:`steep_path_cut`, at the
    fastest water speed), else ``None``.

    The engine names what its window keeps (``kept``: Scooter's paths,
    Kraken's modes carrying them), what the energy is missing from
    (``summed_in``), where the claim is shown (``evidence``: a measurement,
    or the docs that hold one) and its remediation, and whether the user
    pinned ``c_high`` (``pinned``: Scooter's notice concerns a pinned bound
    only, Kraken's its derived one); the message adds the range from
    which no direct or surface-reflected path is steeper than the cut
    (RAM's notice states the same for its band)."""
    c_water = float(np.nanmax(np.asarray(env.ssp.sound_speed, dtype=float)))
    reach = steep_path_cut(c_water, c_high, source.depths,
                           receiver.depths, receiver.ranges)
    if reach is None:
        return None
    cut, steepest = reach
    clear = steep_path_clear_range(source.depths, receiver.depths, cut)
    return Notice(
        f"c_high {c_high:.1f} m/s drops paths steeper than {cut:.1f}° "
        f"that the receiver geometry needs (up to {steepest:.1f}°)",
        f"{model_name}: the {'pinned' if pinned else 'auto-derived'} "
        f"c_high = {c_high:.1f} m/s keeps "
        f"{kept} up to {cut:.1f}° grazing, but the closest receiver sees a "
        f"direct or surface-reflected path at {steepest:.1f}°; that energy "
        f"is not in the {summed_in}, so the near-range field is wrong "
        f"({evidence}); from {clear} m on no direct or surface-reflected "
        f"path is steeper than {cut:.1f}°. {remediation}", NumericsWarning,
    )


#: Phase speed of the seabed interface (Scholte) wave as a fraction of the
#: slowest shear speed of the medium under the water: "generally about 0.85
#: times the minimum shear-wave speed" (JKPS §5.10.5; the wave itself is
#: introduced in §4.10.1).
SCHOLTE_SPEED_FRACTION = 0.85


def interface_wave_notice(env, source, receiver, c_low: float, frequency: float,
                          *, model_name: str) -> Optional[Notice]:
    """A notice when the derived ``c_low`` leaves out the seabed interface
    (Scholte) wave that the geometry can see, else ``None``.

    The wave travels at about :data:`SCHOLTE_SPEED_FRACTION` × the slowest
    shear speed of the medium touching the water. A derived ``c_low``
    (:func:`~uacpy.core.engine_defaults.default_c_low`) is below it over an
    elastic half-space (0.85 × a speed no faster than that shear speed);
    over elastic layers on a fluid half-space it is 0.90 × the slowest
    speed, which can sit above the wave. The wave is confined to
    the seafloor: it counts only when the source or a receiver lies within
    one of its wavelengths, ``0.85·c_s / f`` at the lowest frequency
    ``frequency``, of the seafloor (``env.depth``) — the one-wavelength
    reach is uacpy guidance, set by the measurement it rests on. Far from
    the bed leaving the wave out costs nothing measurable
    (docs/models/scooter.md, "Layered and elastic seabeds": 0.004 dB median
    for a source 75 m above an elastic sand seabed); 1 m above it, up to
    2.4 dB.
    """
    shear = []
    for column in env.bottom.columns:
        top = column.layers[0] if column.layers else column.halfspace
        if (column.layers or BoundaryType.from_string(
                top.acoustic_type).is_geoacoustic) and top.shear_speed > 0:
            shear.append(float(top.shear_speed))
    if not shear:
        return None
    speed = SCHOLTE_SPEED_FRACTION * min(shear)
    if not speed < c_low:
        return None
    wavelength = speed / float(frequency)
    seafloor = float(env.depth)
    heights = np.abs(seafloor - np.concatenate([
        np.atleast_1d(np.asarray(source.depths, dtype=float)),
        np.atleast_1d(np.asarray(receiver.depths, dtype=float))]))
    nearest = float(heights.min())
    if nearest > wavelength:
        return None
    return Notice(
        f"c_low {c_low:.1f} m/s leaves out the seabed interface wave "
        f"(~{speed:.0f} m/s) within {nearest:.3g} m of the seafloor",
        f"{model_name}: the auto-derived c_low = {c_low:.1f} m/s lies above "
        f"the seabed interface (Scholte) wave, about {speed:.0f} m/s "
        f"({SCHOLTE_SPEED_FRACTION:g} × the {min(shear):g} m/s shear speed, "
        f"JKPS §5.10.5), so the integral leaves it out; the geometry reaches "
        f"within {nearest:.3g} m of the seafloor, inside the wave's "
        f"{wavelength:.3g} m wavelength at {float(frequency):g} Hz, where it "
        f"carries energy (up to 2.4 dB measured 1 m above an elastic sand "
        f"seabed). Pass c_low below {speed:.0f} m/s to include it, at the "
        f"cost of more wavenumbers.", NumericsWarning)
