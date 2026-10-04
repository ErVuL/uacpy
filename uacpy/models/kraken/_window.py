"""The phase-speed window a Kraken deck integrates: ``c_low`` / ``c_high``
resolved from the knobs and the profile, and the notices of what the window
cuts (steep paths, leaky modes)."""

import numpy as np
from typing import Optional, Tuple
from uacpy.core.run_settings import Notice
from uacpy.core.boundary import BoundaryType
from uacpy.core.exceptions import ConfigurationError
from uacpy.models._window import (
    C_HIGH_FACTOR, resolve_phase_speed_bounds,
    steep_path_notice as _steep_path_notice,
)
from uacpy.models.kraken._checks import (
    _REFLECTION_TABLE_TYPES,
    has_elastic_surface,
)


#: Default c_high over a reflection-table seabed ('file' / 'precalc'), as a
#: multiple of the fastest water speed: 10x is ~84 degrees grazing. The
#: writer's default for a table is unbounded (1e9, the table carries no sound
#: speed to cap on), and KRAKENC's search over that window can keep no mode at
#: all: a BOUNCE table of an elastic half-space (cp 1600, cs 400) sized for a
#: 5 km receiver, 100 Hz, died in krakenc.f90's no-mode branch. Measured on
#: tables sized for 1, 2, 5 and 10 km: every c_high from 5000 to 30000 m/s
#: gave the same TL (0.40-0.48 dB median, 1.8-2.1 dB 95th percentile from
#: KRAKENC on the half-space itself), so the window is past every mode that
#: reaches the receivers.
_TABLE_BOTTOM_C_HIGH_FACTOR = 10.0

#: ``leaky_modes=True``'s c_high, as a multiple of the fastest speed in the
#: profile (water or seabed). Not unbounded: with c_high = 1e9 KRAKENC's
#: secant search failed outright, or returned a wrong field, on fluid seabeds
#: that solve at any finite window. 200 m of water over a 1700 m/s half-space
#: at 25 Hz, with a 1700 m/s sediment layer 0.1 / 0.5 / 20 m thick: 1e9 kept no
#: modes at 0.1 and 20 m and put the field 21 / 42 dB too quiet at 0.5 m;
#: c_high 5100 and 1e5 gave the half-space-alone answer at every thickness. Every
#: multi-profile deck carries such a layer (the pad to the common deepest
#: depth), so leaky modes failed on every range-dependent run; a two-profile
#: 200 -> 150 m wedge solves at 5100 and 15000. The near field it is for is
#: unchanged: 100 m over 1650 m/s, 200 Hz, against Scooter, 1e9, 1e5 and
#: 15000 all sit -0.029 / -0.017 dB at 200 / 400 m (5000: -0.071), and the
#: 27 leaky modes of docs/models/kraken.md section 6.5 are the same 27.
_LEAKY_C_HIGH_FACTOR = 10.0


# Where a setting came from, as KrakenSettings records it.
_C_LOW_PINNED = 'Kraken(c_low=…)'
_C_HIGH_PINNED = 'Kraken(c_high=…)'


# The window rule of a fluid or elastic half-space: the window the mode-count
# check was measured under (see _grid.mode_count_check_mesh).
_HALF_SPACE_RULE = f"{C_HIGH_FACTOR:g} × max(env.ssp, half-space sound speed)"


def c_low_for(env, *, collapse, pinned_c_low) -> float:
    """Resolved KRAKEN cLow for ``env``: an explicit ``c_low``; else 0.0
    (KRAKEN's own choice) for a fluid environment; else, once any medium
    carries shear, the minimum compressional speed the deck carries.

    With cLow = 0 and ``ElasticFlag`` set, ``krakenc.f90:189,228-230``
    fold the shear speeds into ``cMin`` and floor the search at
    ``0.99 * 0.85 * cMin`` — near 0.84x the slowest SHEAR speed — so the
    solver returns interfacial (Scholte/Stoneley) modes instead of the
    waterborne field. An elastic SURFACE (an ice canopy) counts exactly
    like an elastic bottom (``:220-222`` vs ``:210-212``). The floor
    sweeps every compressional speed of the seabed
    (``Bottom.all_sound_speeds``: a mud layer slower than the water ducts
    modes of its own, and ``:230`` only ever raises the written floor) and
    the minimum of ``env.ssp.sound_speed`` over EVERY profile column, since the
    value is stamped into every block of a multi-profile deck and a floor
    above a profile's slowest water deletes its modes. Too low is
    harmless: no eigenvalue exists below the minimum water sound speed.
    The measurements behind each rule are pinned in ``test_kraken.py``.
    """
    if pinned_c_low is not None:
        return float(pinned_c_low)
    if not env.bottom.is_elastic and not has_elastic_surface(
        env, collapse=collapse):
        # 0.0 hands the search floor to KRAKEN, which computes cLow
        # automatically (kraken.htm, Phase Speed Limits) — the right
        # choice for a fluid environment.
        return 0.0
    speeds = [float(env.ssp.sound_speed.min())]
    speeds.extend(env.bottom.all_sound_speeds())
    return min(speeds)


def validate_phase_speed_limits(*, leaky_modes, pinned_c_high, pinned_c_low):
    """Check ``0 <= c_low < c_high`` for whichever bound is set.

    ``c_high`` is checked on its own as well as against ``c_low``: with
    ``c_low=None`` the writer derives the lower bound from the SSP and
    the bottom, so a non-positive ``c_high`` would otherwise sail past
    every check here and die inside the Fortran with an empty spectrum.
    ``leaky_modes=True`` derives ``c_high`` per profile, so there is no
    pinned value to check.
    """
    cl = pinned_c_low
    ch = pinned_c_high
    if cl is not None and cl < 0:
        raise ConfigurationError(
            f"c_low must be >= 0, got {cl}."
        )
    if ch is not None and ch <= 0:
        raise ConfigurationError(
            f"c_high must be > 0, got {ch}: it is the top of the phase-"
            f"speed window kraken searches for modes, and no mode has a "
            f"non-positive phase speed."
        )
    if cl is not None and ch is not None and ch <= cl:
        raise ConfigurationError(
            f"c_high ({ch}) must be strictly greater than c_low ({cl})"
        )


def phase_speed_window(env, profiles, *, backend: str,
                        coupled: bool, collapse, leaky_modes, log,
                        pinned_c_high, pinned_c_low):
    """``(c_low, c_highs, c_low_origin, c_high_origin)``: the deck's
    phase-speed window — ``cLow`` for every profile and one ``cHigh`` per
    profile of ``profiles`` (the environment itself for a single-profile
    deck, each segment for a multi-profile one). The one decider of the
    window: every deck, the reported bounds and the checks read it.

    ``c_low``: the pinned value; else 0.0 (KRAKEN computes cLow) for a
    fluid problem; else the minimum compressional speed
    (:func:`c_low_for`).

    ``c_high``, per profile, by what bounds the modes:

    - pinned: the user's value; ``leaky_modes=True``:
      :data:`_LEAKY_C_HIGH_FACTOR` × the fastest speed in the profile,
      which asks KRAKENC for the leaky modes.
    - a fluid or elastic half-space: ``C_HIGH_FACTOR`` (1.05) × the
      fastest of the profile's range-0 water column and half-space
      (:func:`~uacpy.models._window.resolve_phase_speed_bounds`).
      Kraken, Scooter, RAM and OAST agree to 0.01-0.16 dB median on fluid,
      layered-fluid and soft elastic seabeds with it. The window keeps a
      few non-trapped modes (above the half-space speed), on the coupled
      multi-profile deck too, where capping it at the slowest fluid
      half-space speed measured MIXED against Scooter on three identical
      Pekeris columns: worse at 4 segments — 100 m, 200 Hz: mean 0.19 ->
      0.36 dB, max 1.5 -> 5.5 dB (cb 1650 m/s), max 6.8 -> 14.9 dB (cb
      1700 m/s); 90 m, cb 1680 m/s, 150 Hz: max 5.2 -> 5.9 dB — and
      better at 8 segments on the 90 m case, max 13.1 -> 5.9 dB (mean
      0.32 -> 0.24 dB). Neither choice dominates, so the window is not
      capped. The coupled deck projects the field onto each segment's
      modes with a half-space tail integral (``EvaluateCMMod.f90:189-195``)
      at every segment boundary, and the uncapped error grew from 4 to 8
      segments.
    - a reflection table ('file' / 'precalc'):
      :data:`_TABLE_BOTTOM_C_HIGH_FACTOR` × the fastest water speed (the
      measurement is at the constant).
    - a rigid or vacuum floor: unbounded on kraken.exe (0.05 dB median
      against :func:`uacpy.analytic.ideal_waveguide` at 50 and 200 Hz);
      KRAKENC, reached here only through an elastic medium (stage 2
      refuses it forced by a knob), gets the table window: under an
      elastic ice canopy every window from 3000 m/s to 1e5 m/s agrees
      with Scooter to 0.02-0.08 dB median at 50-400 Hz, while the
      unbounded one ran past a 600 s timeout at 200 Hz.
    """
    c_low = c_low_for(env, collapse=collapse, pinned_c_low=pinned_c_low)
    if pinned_c_low is not None:
        c_low_origin = _C_LOW_PINNED
    elif c_low == 0.0:
        c_low_origin = "0.0: KRAKEN computes cLow (a fluid problem)"
    else:
        c_low_origin = ("the minimum compressional speed (an elastic "
                        "medium)")
    c_highs, origins = [], []
    for profile in profiles:
        c_high, origin = profile_c_high(profile, c_low,
                                        backend=backend,
                                        leaky_modes=leaky_modes,
                                        pinned_c_high=pinned_c_high)
        c_highs.append(c_high)
        origins.append(origin)
    for c_high in c_highs:
        if not c_low < c_high:
            raise ConfigurationError(
                f"Kraken phase-speed window requires c_low < c_high: "
                f"c_low = {c_low:.1f} m/s ({c_low_origin}) is at or "
                f"above c_high = {c_high:.1f} m/s.",
                remediation="Widen the pinned bound, or leave both "
                            "unset to derive the window from the SSP and "
                            "the seabed.")
    if pinned_c_high is None and not leaky_modes and len(c_highs) == 1:
        log(
            f"c_high auto-derived from env.ssp + bottom = "
            f"{c_highs[0]:.1f} m/s (c_low = {c_low:.1f} m/s)"
        )
    return (c_low, tuple(c_highs), c_low_origin,
            '; '.join(dict.fromkeys(origins)))


def profile_c_high(profile, c_low, *, backend: str, leaky_modes, pinned_c_high
                    ) -> Tuple[float, str]:
    """``(c_high, origin)`` of one profile of the deck; see
    :func:`phase_speed_window`."""
    if pinned_c_high is not None:
        return float(pinned_c_high), _C_HIGH_PINNED
    kind = str(profile.bottom.halfspace_at(range=0.0).acoustic_type)
    fastest_water = float(np.max(np.asarray(profile.ssp.sound_speed)))
    if leaky_modes:
        return (_LEAKY_C_HIGH_FACTOR * max([fastest_water]
                                           + profile.bottom.all_sound_speeds()),
                "leaky_modes=True: _LEAKY_C_HIGH_FACTOR × the fastest speed "
                "in the profile")
    if kind in _REFLECTION_TABLE_TYPES:
        return (_TABLE_BOTTOM_C_HIGH_FACTOR * fastest_water,
                f"_TABLE_BOTTOM_C_HIGH_FACTOR × the fastest water speed "
                f"(a {kind!r} reflection table)")
    geoacoustic = BoundaryType.from_string(kind).is_geoacoustic
    if not geoacoustic and backend == 'krakenc':
        return (_TABLE_BOTTOM_C_HIGH_FACTOR * fastest_water,
                f"_TABLE_BOTTOM_C_HIGH_FACTOR × the fastest water speed "
                f"(krakenc over a {kind} floor)")
    c_high = resolve_phase_speed_bounds(profile, c_low, None)[1]
    if not geoacoustic:
        return c_high, (f"DEFAULT_C_MAX_UNBOUNDED: a {kind} floor has no "
                        f"half-space speed")
    return c_high, _HALF_SPACE_RULE


def steep_path_notice(env, source, receiver,
                       c_high: float) -> Optional[Notice]:
    """``(note, warning)`` when a receiver's direct or surface-reflected
    path is steeper than the unpinned ``c_high`` of the source's profile
    keeps (:func:`~uacpy.models._window.steep_path_notice`), else
    ``None``.

    A mode of phase speed ``c_p`` carries energy up to grazing angle
    ``arccos(c/c_p)``, so the modal sum loses every steeper path, as
    Scooter's wavenumber integral does at the same cut. Measured on a
    100 m Pekeris guide (cb 1650, 200 Hz, source 50 m, receiver 75 m)
    against Scooter(c_high=1e9, taper=0.05): +11.9 dB at 200 m and
    +1.0 dB at 400 m with the default window; ``leaky_modes=True``
    (krakenc, 10 x window) is within 0.04 dB at every range. The
    limit is the trapped-mode sum's (docs/models/kraken.md, "Near
    field"); this says when the geometry reaches it."""
    return _steep_path_notice(
        env, source, receiver, c_high, model_name='Kraken',
        kept='modes carrying paths', summed_in='modal sum',
        evidence=('measured +11.9 dB at 200 m on a 100 m Pekeris guide '
                  'against Scooter'),
        remediation=('Pass leaky_modes=True to solve the modes above the window '
                'on krakenc (within 0.04 dB there), or use Scooter for the '
                'near field.'))
