"""The stability of the rotated Pade step (RAMS): the propagation angle
per frequency, the rotation, the step's growth and level drift, and the
stability range a deck carries."""

import numpy as np
from typing import Optional
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.models._notices import give_notice
from uacpy.models.pe_grid import (
    RAMS_STABILITY_SAFETY, rams_growth_margin, rams_stable_dr,
    rams_stable_theta, rotated_cn_growth, rotated_pade_coefficients,
)
from uacpy.core.environment import (
    Environment,
)
from uacpy.core.constants import NEPER_TO_DB
from uacpy.models.ram._domain import (
    RAMS_CN_DRIFT_BUDGET_DB,
    resolve_c0,
    water_speed_bounds,
)
from uacpy.core.exceptions import FallbackWarning, NumericsWarning


# The rotation angle a rams march takes when ``rams_rotation_angle`` is left ``None``
# and the stability rule does not ask for less (see ``resolve_rams_rotation_angle``).
RAMS_DEFAULT_THETA_DEG = 45.0


def theta_for_freq(freq: float, *, knobs) -> float:
    """Resolve ``rams_rotation_angle`` for a single frequency.

    ``rams_rotation_angle`` may be:
      - a float: same theta for every frequency (default 45.0).
      - a callable taking a float frequency in Hz and returning a
        float theta in degrees. Use this when the elastic PE needs
        different stability angles across the band.
    """
    t = knobs.rams_rotation_angle
    if t is None:
        return float(RAMS_DEFAULT_THETA_DEG)
    if callable(t):
        return float(t(float(freq)))
    return float(t)


def resolve_rams_rotation_angle(env: Environment, freq: float, *, knobs,
                       speed_bounds) -> float:
    """The rotation angle a rams march at ``freq`` on ``env`` uses.

    A pinned ``rams_rotation_angle`` (float or callable) is returned as is. Left
    ``None``, the default 45° is kept unless the stability rule finds its
    own growth above the seabed's leak, in which case the widest stable
    angle on the ladder replaces it (Milinazzo, Zala & Brooke 1997: the
    wider the angle, the better the evanescent spectrum is handled, so
    the widest that holds is the one to take); stage 3 says so
    (:func:`resolve_collins_thetas`). When no angle on the ladder is
    stable the default is returned and the grid chooser refuses the run.
    """
    lowering = rams_rotation_angle_lowering(env, freq, knobs=knobs,
                                   speed_bounds=speed_bounds)
    if lowering is None:
        return theta_for_freq(freq, knobs=knobs)
    return lowering['theta']


def rams_rotation_angle_lowering(env: Environment, freq: float, *, knobs, speed_bounds):
    """``{'requested', 'theta', 'margin'}`` when the stability rule
    lowers the default rotation angle at ``freq`` — the default, the
    widest stable angle on the ladder, and the default's growth margin —
    or ``None`` when the angle is pinned, the default holds, or no angle
    on the ladder does."""
    requested = theta_for_freq(freq, knobs=knobs)
    if knobs.rams_rotation_angle is not None:
        return None
    stab = rams_stability(env, freq, theta=requested, knobs=knobs,
                          speed_bounds=speed_bounds)
    if stab is None or stab['margin']['floor_excess'] <= 0.0:
        return None
    if stab['theta'] is None:
        return None
    return {'requested': requested, 'theta': float(stab['theta']),
            'margin': stab['margin']}


def rams_rot0(theta: float, *, knobs) -> complex:
    """``rot0`` of the rams0.5 rotated-Padé scalar ``g0``.

    ``rpade`` (``third_party/ramsurf/rams0.5.f:859-892``), transcribed
    once in :func:`~uacpy.models.pe_grid.rotated_pade_coefficients`,
    which the stability rule reads too; ``epade`` uses ``rot0 = 1`` when
    ``rams_rotation=False``.
    """
    if not knobs.rams_rotation:
        return 1.0 + 0.0j
    return rotated_pade_coefficients(int(knobs.n_pade), float(theta))[2]


def rams_stability_params(env: Environment, freq: float,
                           theta: Optional[float] = None, *, knobs,
                           speed_bounds):
    """Inputs of the rams0.5 stability rule (``pe_grid.
    rams_growth_margin``), one dict per seabed column, or ``None`` when
    the rule does not apply: ``rams_rotation=False`` marches real Padé
    coefficients (a unitary Crank-Nicolson step), and a column with no
    surficial half-space carries no seabed to leak into.

    The steep components the step amplifies leave the water on every
    bottom bounce, so the leak is set by the column's *surficial*
    material (top layer or half-space) and the water depth; the deepest
    seafloor is used for every column because it leaks least per metre.
    The water speed is the column's fastest: with ``ξ`` measured against
    ``k0 = ω/c0``, a component's grazing angle is
    ``arccos(√(1+ξ)·c_w/c0)``, so the fastest water maps a given ``ξ`` to
    the shallowest grazing angle and hence the least leak — the
    conservative side.
    """
    if not knobs.rams_rotation:
        return None
    theta_deg = (float(theta) if theta is not None
                 else theta_for_freq(freq, knobs=knobs))
    c0 = resolve_c0(env, knobs=knobs, speed_bounds=speed_bounds)
    _, water_c = water_speed_bounds(env)
    depth = float(env.bathymetry.depth)
    params = []
    for column in env.bottom.columns:
        top = column.at(depth=0.0)
        if top.acoustic_type != 'half-space' or top.sound_speed is None:
            continue
        params.append(dict(
            frequency=float(freq), c0=float(c0),
            water_sound_speed=float(water_c),
            water_density=float(env.water_density),
            seabed_speed=float(top.sound_speed),
            seabed_density=float(top.density if top.density is not None
                                 else 1.0),
            seabed_attenuation_dB_lambda=float(
                top.attenuation if top.attenuation is not None else 0.0),
            depth=depth, n_pade=int(knobs.n_pade),
            theta_deg=theta_deg,
        ))
    return params or None


def rams_stability(env: Environment, freq: float,
                    dr: Optional[float] = None, dr_max: float = 1e3,
                    theta: Optional[float] = None, *, knobs, speed_bounds):
    """The rams0.5 march's stability on ``env`` at ``freq``: the worst
    column's :func:`rams_growth_margin` at ``dr`` (``None`` for the
    rotation floor alone), the largest stable ``dr`` (``None`` when the
    rotation itself is unstable), and the largest stable rotation angle
    (``None`` when no angle on the ladder is). ``None`` when the rule
    does not apply (:func:`rams_stability_params`)."""
    params = rams_stability_params(env, freq, theta=theta, knobs=knobs,
                                   speed_bounds=speed_bounds)
    if params is None:
        return None
    margins = [rams_growth_margin(dr, **p) for p in params]
    key = 'excess' if dr is not None else 'floor_excess'
    worst = max(margins, key=lambda m: m[key])
    drs = [rams_stable_dr(dr_max, **p) for p in params]
    dr_stable = None if any(d is None for d in drs) else min(drs)
    thetas = [rams_stable_theta(**p) for p in params]
    theta_stable = (None if any(th is None for th in thetas)
                    else min(thetas))
    return {'margin': worst, 'dr': dr_stable, 'theta': theta_stable,
            'theta_requested': params[0]['theta_deg']}


def rams_rotation_remedy(stab: dict, *, knobs) -> str:
    """The sentence naming what fixes a rotation-limited rams march."""
    m = stab['margin']
    head = (f"the rotated square root itself (rams_rotation_angle="
            f"{stab['theta_requested']:.0f}°, n_pade={int(knobs.n_pade)}) "
            f"amplifies the steepest propagating components at "
            f"{m['floor']:.2e} Np/m, above the {m['floor_leak']:.2e} Np/m "
            f"this seabed leaks them at (over the "
            f"{RAMS_STABILITY_SAFETY:g}× safety margin), so no range step "
            f"can keep the march bounded.")
    if stab['theta'] is not None:
        return (head + f" Pass rams_rotation_angle={stab['theta']:.0f} (the largest "
                f"angle that is stable here), or leave rams_rotation_angle=None "
                f"to have it chosen, or use a larger n_pade.")
    return (head + " No rotation angle down to 5° is stable here; use "
            "a larger n_pade, or OAST / Scooter for this seabed.")


def warn_rams_rotation_angle_lowered(lowered, n_frequencies: int, *,
                             notices=None) -> None:
    """One warning naming every frequency whose default ``rams_rotation_angle``
    the stability rule lowered (:func:`rams_rotation_angle_lowering`)."""
    f0, first = lowered[0]
    m = first['margin']
    if len(lowered) == 1:
        give_notice(notices,
            f"RAM:rams: at f={f0:.1f} Hz the default rams_rotation_angle="
            f"{first['requested']:.0f}° would amplify the steepest "
            f"propagating components at {m['floor']:.2e} Np/m against the "
            f"{m['floor_leak']:.2e} Np/m this seabed leaks them at, "
            f"whatever the range step; using rams_rotation_angle="
            f"{first['theta']:.0f}°, the widest stable angle. Pin "
            f"rams_rotation_angle to choose yourself.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
        return
    freqs = [f for f, _ in lowered]
    angles = [low['theta'] for _, low in lowered]
    give_notice(notices,
        f"RAM:rams: at {len(lowered)} of {n_frequencies} frequencies "
        f"({min(freqs):.1f}-{max(freqs):.1f} Hz) the default rams_rotation_angle="
        f"{first['requested']:.0f}° would amplify the steepest "
        f"propagating components faster than this seabed leaks them "
        f"(at {f0:.1f} Hz: {m['floor']:.2e} against "
        f"{m['floor_leak']:.2e} Np/m), whatever the range step; those "
        f"bins use the widest stable angle, "
        f"{min(angles):.0f}-{max(angles):.0f}°. Pin rams_rotation_angle to choose "
        f"yourself.",
        FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
    )


def warn_if_rams_step_unstable(env, fc, dr, *, notices=None, knobs,
                               speed_bounds):
    """A pinned ``dr`` is the caller's, so it is marched as given — but
    a step the stability rule predicts will diverge is said so *before*
    the minutes it takes to march into NaN, with the step (or the
    rotation angle) that would hold."""
    stab = rams_stability(env, fc, dr=dr,
                          theta=resolve_rams_rotation_angle(env, fc, knobs=knobs,
                                                   speed_bounds=speed_bounds),
                                                   knobs=knobs,
                                                   speed_bounds=speed_bounds)
    if stab is None or stab['margin']['excess'] <= 0.0:
        return
    m = stab['margin']
    if stab['dr'] is None:
        remedy = rams_rotation_remedy(stab, knobs=knobs)
    else:
        remedy = (f"Its rotated Crank-Nicolson step amplifies the "
                  f"steepest propagating components (grazing "
                  f"{m['grazing_deg']:.0f}°) at "
                  f"{m['growth']:.2e} Np/m against the {m['leak']:.2e} "
                  f"Np/m this seabed leaks them at; the automatic grid "
                  f"would use dr <= {stab['dr']:.4g} m here.")
    give_notice(notices,
        f"RAM:rams: the pinned dr={dr:.4g} m at f={fc:.1f} Hz is "
        f"predicted to diverge. {remedy}",
        NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP,
    )


def rams_level_drift_dB(dr, *, freq, c0, c_min, c_max_all,
                         max_range, theta, knobs):
    """Level drift (dB) the rams0.5 rotated Crank-Nicolson step
    accumulates over ``max_range`` at ``dr``: the largest
    ``|ln|G(ξ)||/dr`` (:func:`~uacpy.models.pe_grid.rotated_cn_growth`)
    over the propagating band — from the fastest medium's critical angle
    to horizontal in the slowest water — times the range, in dB."""
    k0 = 2.0 * np.pi * float(freq) / float(c0)
    xi = np.linspace((float(c0) / float(c_max_all)) ** 2 - 1.0,
                     (float(c0) / float(c_min)) ** 2 - 1.0, 401)
    rate = np.abs(rotated_cn_growth(xi, k0, float(dr), int(knobs.n_pade),
                                    float(theta)))
    return float(np.max(rate)) * NEPER_TO_DB * float(max_range)


def rams_dr_for_level_drift(dr, *, freq, c0, c_min, c_max_all,
                             max_range, theta, knobs):
    """``dr``, or the largest step below it whose accumulated level drift
    (:func:`rams_level_drift_dB`) stays within
    :data:`RAMS_CN_DRIFT_BUDGET_DB`. The drift falls as ``dr²``, so the
    first guess scales ``dr`` by the square root of the ratio, and 10 %
    steps down absorb what that misses."""
    kw = dict(freq=freq, c0=c0, c_min=c_min, c_max_all=c_max_all,
              max_range=max_range, theta=theta)
    drift = rams_level_drift_dB(dr, **kw, knobs=knobs)
    if drift <= RAMS_CN_DRIFT_BUDGET_DB:
        return float(dr)
    dr_new = float(dr) * np.sqrt(RAMS_CN_DRIFT_BUDGET_DB / drift)
    for _ in range(50):
        if rams_level_drift_dB(dr_new, **kw,
                               knobs=knobs) <= RAMS_CN_DRIFT_BUDGET_DB:
            break
        dr_new *= 0.9
    return float(dr_new)


def collins_stability_range(kind: str, *, knobs) -> Optional[float]:
    """The stability range a Collins deck carries: the pinned value or
    0 (which the fluid codes expand to ``2 × rmax``); ``None`` on
    rams0.5, whose row 5 carries the rotation instead."""
    if kind == 'rams':
        return None
    return float(knobs.stability_range_m or 0.0)


def warn_stability_range_inert_on_a_multi_range_grid(receiver, *,
                                                      notices=None,
                                                      knobs) -> None:
    """Say so when a pinned ``stability_range_m`` will not act at the range it
    names, because the output grid decides instead.

    ``mpiramS/src/ram.f90:68`` computes ``rsc = |rg(nr) - 0| - rs`` once,
    and the march tests it at ``:251`` as
    ``if (abs(rend - rnow) < rsc)`` — where ``rend`` is reassigned to the
    *current output range* at ``:166``, not the last one. The comparison
    is therefore between the distance left to the next receiver range and
    a constant, rather than between the absolute range marched and ``rs``
    (which is what ``ramgeo/ramgeo1.5.f:368`` does, and what the
    parameter name promises). ``:253`` then zeroes ``rsc``, so it fires at
    most once.

    With one output range the two coincide — ``rend`` stays ``rmax`` and
    the test reduces to ``rnow > rs``. With more than one, the firing step
    is set by the receiver range spacing and the value of ``rs`` stops
    mattering over most of its range. Measured on a 100 m guide at 100 Hz
    with receivers at 40 ranges from 500 m to 20 km: ``stability_range_m``
    9 km apart (10 km and 19 km) gives bit-identical output, while on a
    single 20 km output range the same pair moves TL by 0.0012 dB.

    This is upstream mpiramS behaviour; the warning is uacpy declining to
    let a public parameter look like it did something.
    """
    if knobs.stability_range_m is None:
        return
    ranges = np.atleast_1d(np.asarray(receiver.ranges, dtype=float))
    if ranges.size < 2:
        return
    give_notice(notices,
        f"RAM:mpirams: stability_range_m={knobs.stability_range_m:g} m will "
        f"not switch the stability terms off at that range. ram.f90:251 tests "
        f"the distance left to the *current* output range, not the absolute "
        f"range marched, so with {ranges.size} receiver ranges the switch-off "
        f"point is set by the range spacing and the value is largely inert "
        f"(two values 9 km apart measured bit-identical on a 40-range grid). "
        f"Use a single receiver range if you need stability_range_m to act at "
        f"the range it names, or leave it unset to keep n_stability terms on "
        f"throughout.",
        FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
    )


def collins_carrier_rate(env: Environment, kind: str,
                          freq: float, theta: float, *, knobs,
                          speed_bounds) -> float:
    """Phase rate (rad/m) carried by a Collins backend's stored envelope.

    Two terms, both fast enough to alias across a ``dr·ndr`` output step:

    * Every backend writes ψ with only ``exp(i k0 r)`` factored out, so
      what remains still rotates at ``k_r - k0``. ``c0`` is the Lytaev
      expansion point rather than a medium speed, so that difference is
      not small — 0.09 rad/m against a 21 m Lytaev ``dr`` on the 250 Hz
      Pekeris reference, i.e. 1.9 rad per output step. ``k_r`` is taken
      at the mean water sound speed, which sits inside the propagating
      modal band.
    * ``rams0.5`` additionally multiplies u by ``g0 = exp(i k0 dr rot0)``
      on every range step (``rams0.5.f:848-851``), i.e. the whole
      carrier is baked in — ~0.89 rad/m at 250 Hz. The rotation makes
      ``rot0`` complex (``rams0.5.f:865-888``); only ``Re(rot0)`` is a
      phase rate — ``Im(rot0)`` is the rotation's amplitude decay per
      step, which does not alias and must not be divided out.

    :func:`interp_envelope_to_receiver_grid` divides this out before
    interpolating and restores it afterwards.
    """
    c0 = resolve_c0(env, knobs=knobs, speed_bounds=speed_bounds)
    k0 = 2.0 * np.pi * float(freq) / c0
    speeds = np.asarray(env.ssp.sound_speed, dtype=float)
    speeds = speeds[np.isfinite(speeds)]
    c_water = float(np.mean(speeds)) if speeds.size else c0
    rate = 2.0 * np.pi * float(freq) / c_water - k0
    if kind == 'rams':
        rate += k0 * rams_rot0(theta, knobs=knobs).real
    return float(rate)


def resolve_collins_thetas(env, kind: str, frequencies, *,
                            notices=None, knobs, speed_bounds) -> list:
    """The rotation angle each Collins launch's deck carries: rams0.5's
    (:func:`resolve_rams_rotation_angle`), with one warning for every bin whose
    default angle the stability rule lowered; the fluid codes'
    :func:`theta_for_freq`, which their row 5 does not read."""
    if kind != 'rams':
        return [theta_for_freq(float(f), knobs=knobs) for f in frequencies]
    thetas, lowered = [], []
    for f in frequencies:
        lowering = rams_rotation_angle_lowering(env, float(f), knobs=knobs,
                                       speed_bounds=speed_bounds)
        if lowering is None:
            thetas.append(theta_for_freq(float(f), knobs=knobs))
        else:
            thetas.append(lowering['theta'])
            lowered.append((float(f), lowering))
    if lowered:
        warn_rams_rotation_angle_lowered(lowered, len(thetas),
                                notices=notices)
    return thetas
