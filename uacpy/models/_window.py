"""The phase-speed window ``[c_low, c_high]`` an Acoustics-Toolbox deck
searches or integrates over (M-15): the default rule the engines derive an
unpinned bound from, the window a wavenumber integral resolves with the
origin of each bound, the check that a pinned pair is ordered, and the
near-field reach of a window — the steep paths it cuts — with the notice that
states it.

The rules an engine derives on its own (Kraken's ``c_low`` of 0 and its
reflection-table ``c_high``, BOUNCE's ``min(SSP)``) stay in that engine; what
is here is what more than one engine reads.
"""

from typing import NamedTuple, Optional, Tuple

import numpy as np

from uacpy.core.boundary import BoundaryType
from uacpy.core.environment import Environment
from uacpy.core.exceptions import ConfigurationError, NumericsWarning
from uacpy.core.run_settings import Notice

# ``C_LOW_FACTOR`` is the wavenumber-integration default: Scooter / SPARC
# pin ``k_max = omega/c_low`` so a positive floor is required (c_low=0
# blows up the integral). 0.95·c_min is the canonical conservative
# choice.
#
# ``C_HIGH_FACTOR`` pads the upper bound symmetrically: the engines take
# ``c_high = 1.05 · max(c_max, bottom cp)`` so the fastest speed in the problem
# sits strictly inside the interval rather than on its edge.
# The pad is REQUIRED by the integration models and merely TOLERATED by the
# modal one, so it cannot be tuned for either alone: for Scooter
# (``scooter.f90:67,123``) and SPARC (``SPARC._resolve_engine_settings``'s Nk)
# c_high is the lower limit of a wavenumber INTEGRAL, and padding past the
# bottom speed keeps the branch point k = omega/c_bottom strictly inside the
# window — which
# is what lets Scooter recover the lateral wave that makes it the right model
# below the modal cutoff. Kraken only searches [c_low, c_high] for roots, so
# there the same pad means a default run returns a few modes with a phase speed
# above the bottom speed (``models/kraken/_modes.py`` logs the count and
# refuses when
# every mode is one of them).
# (KRAKEN's modal-solver c_low default is the literal 0.0 written at its use
# site in ``models/kraken/_window.py`` — 0 makes KRAKEN compute the bound
# itself, per
# kraken.htm, Phase Speed Limits.)
C_LOW_FACTOR = 0.95
C_HIGH_FACTOR = 1.05

# "No upper phase-speed limit" for the AT family. A vacuum / rigid boundary
# traps every mode, so the mode search must not be capped on a half-space
# speed that does not exist. Acoustics-Toolbox/doc/bounce.htm prescribes the
# value: "For a full 90 degree calculation set CMin to the lowest speed in the
# problem (say 1400.0) CMax to 1.0E9." Kraken's ``leaky_modes`` uses the same
# number.
DEFAULT_C_MAX_UNBOUNDED = 1.0e9


def resolve_phase_speed_bounds(
    env: Environment,
    c_low: Optional[float] = None,
    c_high: Optional[float] = None,
) -> Tuple[float, float]:
    """Resolve effective ``(c_low, c_high)`` for an AT-family run.

    Precedence:
      1. Explicit caller values win.
      2. Otherwise: ``c_low = c_min · C_LOW_FACTOR`` and
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
    limit", the same value ``leaky_modes`` uses.
    """
    if c_low is not None and c_high is not None:
        return float(c_low), float(c_high)
    ssp_pairs = env.ssp.to_pairs()
    c_min = float(ssp_pairs[:, 1].min())
    halfspace = env.bottom.halfspace_at(range=0.0)
    if not BoundaryType.from_string(halfspace.acoustic_type).is_geoacoustic:
        c_high_auto = DEFAULT_C_MAX_UNBOUNDED
    else:
        c_max = max(float(ssp_pairs[:, 1].max()),
                    float(halfspace.sound_speed))
        c_high_auto = c_max * C_HIGH_FACTOR
    return (
        float(c_low) if c_low is not None else c_min * C_LOW_FACTOR,
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
    """Refuse two pinned bounds with ``c_low >= c_high`` (one pinned bound
    is held to the other once a run derives it, :func:`resolve_window`)."""
    if (c_low is not None and c_high is not None
            and c_low >= c_high):
        raise ConfigurationError(
            f"{model_name} spectral phase-velocity band requires "
            f"c_low < c_high; got c_low={c_low} m/s, "
            f"c_high={c_high} m/s."
        )


def resolve_window(env, *, c_low, c_high,
                   model_name: str) -> PhaseSpeedWindow:
    """The window of a wavenumber integral (Scooter, SPARC): the pinned
    bounds, and the unpinned ones from :func:`resolve_phase_speed_bounds`
    on the projected ``env`` — ``0.95 × min(SSP)`` and ``1.05 × max(SSP,
    seabed sound speed)``, unbounded (1e9 m/s) over a vacuum, rigid,
    ``'file'`` or ``'precalc'`` seabed — with the origin of each.

    The constructor can only compare two pinned bounds
    (:func:`check_pinned_window`). A single pinned bound is only comparable
    once the other has been derived from this env, and an inverted pair
    reaches ReadEnvironmentMod.f90:135, which stops the binary after the
    deck has been written and the process spawned. It is refused here
    instead, naming the bound the user pinned, since that is the one to
    move.
    """
    cl, ch = resolve_phase_speed_bounds(env, c_low, c_high)
    if cl >= ch:
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
    if c_high is not None:
        high_origin = f"{model_name}(c_high=…)"
    elif ch == DEFAULT_C_MAX_UNBOUNDED:
        high_origin = ("DEFAULT_C_MAX_UNBOUNDED: the seabed has no "
                       "half-space speed (vacuum, rigid, 'file' or "
                       "'precalc')")
    else:
        high_origin = f"{C_HIGH_FACTOR:g} × max(env.ssp, seabed sound speed)"
    return PhaseSpeedWindow(
        c_low=cl, c_high=ch,
        c_low_origin=(f"{model_name}(c_low=…)" if c_low is not None
                      else f"{C_LOW_FACTOR:g} × min(env.ssp)"),
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


def steep_path_notice(env, source, receiver, c_high: float, *,
                      model_name: str, kept: str, summed_in: str,
                      evidence: str, remediation: str) -> Optional[Notice]:
    """``(note, warning)`` when a receiver's direct or surface-reflected
    path is steeper than the auto-derived ``c_high`` keeps
    (:func:`steep_path_cut`, at the fastest water speed), else ``None``.

    The engine names what its window keeps (``kept``: Scooter's paths,
    Kraken's modes carrying them), what the energy is missing from
    (``summed_in``), the measurement behind the claim (``evidence``) and
    its remediation."""
    c_water = float(np.nanmax(np.asarray(env.ssp.sound_speed, dtype=float)))
    reach = steep_path_cut(c_water, c_high, source.depths,
                           receiver.depths, receiver.ranges)
    if reach is None:
        return None
    cut, steepest = reach
    return Notice(
        f"c_high {c_high:.1f} m/s drops paths steeper than {cut:.1f}° "
        f"that the receiver geometry needs (up to {steepest:.1f}°)",
        f"{model_name}: the auto-derived c_high = {c_high:.1f} m/s keeps "
        f"{kept} up to {cut:.1f}° grazing, but the closest receiver sees a "
        f"direct or surface-reflected path at {steepest:.1f}°; that energy "
        f"is not in the {summed_in}, so the near-range field is wrong "
        f"({evidence}). {remediation}", NumericsWarning,
    )
