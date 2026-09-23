"""Scale conversions between the units the public uacpy API speaks (metres,
Hz, radians, knots) and the ones other layers need.

Pure conversions over numpy, plus the two meteorological scales a caller
arrives with (Beaufort force, sea state), so every layer —
``io`` at the file-format boundary, ``visualization`` at the axis-label
boundary — can reach them without importing a sibling package for arithmetic.
The io writers and readers import them from here: every writer that emits
a km-on-disk axis goes through ``m_to_km``, every reader that returns a
metres-API axis through ``km_to_m``, and phase columns through
``deg_to_rad`` / ``rad_to_deg``, so the "did I convert?" question stays
grep-able.
"""

from __future__ import annotations

import numpy as np

from uacpy.core.exceptions import ConfigurationError


def km_to_m(x):
    """Multiply a km axis by 1000 to get metres."""
    return np.asarray(x, dtype=float) * 1000.0


def m_to_km(x):
    """Divide a metres axis by 1000 to get km."""
    return np.asarray(x, dtype=float) / 1000.0


def deg_to_rad(x):
    """Convert degrees to radians."""
    return np.asarray(x, dtype=float) * (np.pi / 180.0)


def rad_to_deg(x):
    """Convert radians to degrees.

    The write direction of :func:`deg_to_rad`: uacpy carries phase in
    radians and the ``.brc``/``.trc`` reflection tables hold it in degrees,
    so :func:`~uacpy.io.refl_io.write_reflection_coefficient` converts here
    and :func:`~uacpy.io.refl_io.read_reflection_coefficient` converts back.
    """
    return np.asarray(x, dtype=float) * (180.0 / np.pi)


#: Knots per metre per second. The public noise and scattering surfaces take
#: wind in **knots** (``wind_speed_kn``) while every fetcher returns m/s, so
#: this factor sat as prose in two docstrings — "multiply the m/s returned
#: here" — and in no callable. Reading a m/s value as knots
#: understates the Wenz total by a measured 5.7 dB at 1 kHz for 10 m/s.
#: One nautical mile per hour, from the SI definition: 1852 m in an hour.
#: Written as the definition rather than as a rounded decimal, so the one
#: value in the package is the exact one. The literal it replaces differed
#: by 3.9e-9 relative — nothing numerically, and two numbers to keep in
#: step.
KNOTS_PER_M_PER_S = 3600.0 / 1852.0


def ms_to_knots(x):
    """Convert a wind speed in m/s to knots."""
    return np.asarray(x, dtype=float) * KNOTS_PER_M_PER_S


def knots_to_ms(x):
    """Convert a wind speed in knots to m/s."""
    return np.asarray(x, dtype=float) / KNOTS_PER_M_PER_S


#: Beaufort force -> (sea state, wind speed range in knots), Urick (1984).
#: It indexes the same way Knudsen, Alford & Emling (1948) Fig. 4 indexes its
#: noise curves — except at sea state 6, where they differ. Fig. 4 pairs sea
#: state 6 with wind force **7**; this table gives sea state 6 to both force 7
#: and force 8, and :func:`sea_state_to_wind_speed` resolves it to 8 on the
#: WMO convention below. So a reader reproducing Knudsen's top curve should
#: ask for force 7 (30.5 kn), not sea state 6 (37 kn) — worth 1.7 dB in the
#: ``'knudsen'`` wind model.
#:
#: Sea state 6 appears twice (forces 7 and 8), which is why the inverse
#: direction below is a one-way map: a sea state names a *band* of winds and
#: cannot recover the force that produced it. Dahl et al. (2007) resolve the
#: ambiguity the other way for the WMO scale ("WMO sea state code 6
#: corresponds to Beaufort scale of 8"), which is the convention taken here.
BEAUFORT_SCALE = {
    0: (0.0, (0.0, 1.0)),
    1: (0.5, (1.0, 3.0)),
    2: (1.0, (4.0, 6.0)),
    3: (2.0, (7.0, 10.0)),
    4: (3.0, (11.0, 16.0)),
    5: (4.0, (17.0, 21.0)),
    6: (5.0, (22.0, 27.0)),
    7: (6.0, (28.0, 33.0)),
    8: (6.0, (34.0, 40.0)),
}


def beaufort_to_wind_speed(force, *, units="kn"):
    """Representative wind speed for a Beaufort force.

    The midpoint of the force's range in :data:`BEAUFORT_SCALE`. A force
    names a band of speeds, so the midpoint is a *choice*, not a
    conversion — it is the value the noise curves indexed by force are
    plotted against. Picking an end instead moves the default (Merklinger)
    wind term at 1 kHz by -1.68 dB (7 kn) or +1.40 dB (10 kn) at force 3,
    3.08 dB end to end. Pass a wind speed directly whenever one is known;
    this exists for data that only reports a force.

    Parameters
    ----------
    force : int
        Beaufort force, 0-8.
    units : {'kn', 'm/s'}
        Unit of the returned speed. Default knots, the unit
        ``wind_speed_kn`` takes.
    """
    if force not in BEAUFORT_SCALE:
        raise ConfigurationError(
            f"beaufort_to_wind_speed: force must be one of "
            f"{sorted(BEAUFORT_SCALE)}; got {force!r}. The Urick (1984) "
            f"table this reads stops at force 8 (40 knots).")
    low, high = BEAUFORT_SCALE[force][1]
    kn = 0.5 * (low + high)
    return _in_wind_units(kn, units, "beaufort_to_wind_speed")


def sea_state_to_wind_speed(sea_state, *, units="kn"):
    """Representative wind speed for a sea state, via :data:`BEAUFORT_SCALE`.

    Sea state correlates with the noise far less well than wind speed does
    — Dahl et al. (2007): "the noise from the sea surface correlates much
    better with wind speed than with sea state" — so this is a bridge for
    data that reports only a sea state, not a preferred input.

    Sea state 6 maps to force 8 rather than 7, following Dahl's statement
    of the WMO correspondence — 37 kn rather than 30.5 kn, worth +1.67 dB
    in the default wind term at 1 kHz (+3.01 dB on ``'coates'``).
    """
    matches = [f for f, (ss, _) in BEAUFORT_SCALE.items() if ss == sea_state]
    if not matches:
        raise ConfigurationError(
            f"sea_state_to_wind_speed: sea_state must be one of "
            f"{sorted({ss for ss, _ in BEAUFORT_SCALE.values()})}; got "
            f"{sea_state!r}.")
    return beaufort_to_wind_speed(max(matches), units=units)


def wind_speed_to_beaufort(speed, *, units="kn"):
    """Beaufort force whose range contains ``speed``.

    The inverse of :func:`beaufort_to_wind_speed` at band resolution: a
    speed above force 8's range returns 8, which is the table's ceiling
    rather than the scale's.
    """
    if units not in ("kn", "m/s"):
        raise ConfigurationError(
            f"wind_speed_to_beaufort: units must be 'kn' or 'm/s'; got "
            f"{units!r}.")
    # One speed, one force. Refused by name rather than left to numpy's
    # "only 0-dimensional arrays can be converted" from the float() below,
    # which names neither this function nor the argument.
    if np.ndim(speed) != 0:
        raise ConfigurationError(
            f"wind_speed_to_beaufort: takes one speed, not an array of "
            f"{np.shape(speed)}. A force is a band, so mapping many speeds "
            f"at once hides which ones share a force; call it per speed.")
    kn = float(speed) if units == "kn" else float(ms_to_knots(speed))
    if not kn >= 0.0:
        raise ConfigurationError(
            f"wind_speed_to_beaufort: wind speed must be >= 0; got {speed!r}")
    edges = [BEAUFORT_SCALE[f][1][1] for f in sorted(BEAUFORT_SCALE)]
    return int(np.searchsorted(edges, float(kn), side="left"
                               ).clip(0, max(BEAUFORT_SCALE)))


def _in_wind_units(kn, units, who):
    """Return a knots value in the requested wind unit."""
    if units == "kn":
        return float(kn)
    if units == "m/s":
        return float(knots_to_ms(kn))
    raise ConfigurationError(
        f"{who}: units must be 'kn' or 'm/s'; got {units!r}.")


__all__ = ["km_to_m", "m_to_km", "deg_to_rad", "rad_to_deg",
           "KNOTS_PER_M_PER_S", "ms_to_knots", "knots_to_ms",
           "BEAUFORT_SCALE", "beaufort_to_wind_speed",
           "sea_state_to_wind_speed", "wind_speed_to_beaufort"]
