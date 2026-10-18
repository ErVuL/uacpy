"""Whether a weighting given as ``weighting=`` fits the record it weights.

Two checks every ``weighting=`` entry point runs (the level functions of
:mod:`uacpy.core.acoustics.levels`, :func:`~uacpy.acoustics.weighted_pressure`
and :meth:`uacpy.Field.sound_exposure_level`): an in-air hearing group
weighting a level at the underwater reference, and a weighting table that
leaves most of the record's energy outside its span. Both warn; neither
changes the level.
"""

import warnings

import numpy as np

from uacpy.core.constants import REFERENCE_PRESSURE_WATER
from uacpy.core.exceptions import ValidityWarning
from uacpy.core._warn_frames import USER_FRAME_SKIP


#: The hearing groups of Southall et al. (2019) whose weighting and
#: thresholds are for sound in air, each with its in-water counterpart: their
#: Tables 6-7 give "dB re (20 μPa)2 s in air (groups PCA and OCA only)"
#: against "dB re 1 μPa2 s under water" (45_2_southall.md, Table 6 caption).
_IN_AIR_GROUPS = {"PCA": "PCW", "OCA": "OCW"}

#: Share of the energy a weighting TABLE may leave outside its span, where it
#: carries no weight, before :func:`warn_table_leaves_energy_out` reports it.
_TABLE_DROPPED_SHARE = 0.5


def warn_in_air_group_in_water(weighting, ref, who):
    """Warn when an in-air hearing group (``'PCA'``, ``'OCA'``) weights a
    level referenced to the underwater pressure, 1 µPa: their thresholds
    are re (20 µPa)²·s, so the comparison is 26 dB off, and a record under
    water takes the group's in-water weighting."""
    if not isinstance(weighting, str):
        return
    group = weighting.upper()
    if group not in _IN_AIR_GROUPS or ref != REFERENCE_PRESSURE_WATER:
        return
    warnings.warn(
        f"{who}: {weighting!r} is an in-air hearing group (Southall et al. "
        f"2019), whose thresholds (EXPOSURE_THRESHOLDS[{group!r}]) are dB re "
        f"(20 µPa)²·s, and this level is re (1 µPa)²·s, the underwater "
        f"reference: the two differ by 26 dB. Under water, weight with "
        f"{_IN_AIR_GROUPS[group]!r}; for a record made in air, pass "
        f"ref=20e-6.",
        ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP)


def warn_table_leaves_energy_out(weighting, frequency, energy, who):
    """Warn when a ``(frequencies_Hz, W_dB)`` weighting table leaves more
    than half of the energy outside its span, where it carries no weight:
    ``energy`` is the energy per bin at ``frequency`` (Hz). A table typed in
    kHz is the usual cause, and the warning says so when the same table in
    Hz would keep more. A hearing group or a function is not a table and is
    never reported."""
    if not (isinstance(weighting, (tuple, list)) and len(weighting) == 2):
        return
    edges = np.asarray(weighting[0], dtype=float)
    f = np.asarray(frequency, dtype=float)
    e = np.asarray(energy, dtype=float)
    positive = f > 0.0
    total = float(np.sum(e[positive]))
    if not total > 0.0:
        return

    def outside(scale):
        lo, hi = scale * edges[0], scale * edges[-1]
        return float(np.sum(e[positive & ((f < lo) | (f > hi))])) / total

    share = outside(1.0)
    if share <= _TABLE_DROPPED_SHARE:
        return
    inside = 1.0 - share
    kept = ("none of the energy lies inside it" if inside <= 0.0 else
            f"only {100 * inside:.3g} % of the energy lies inside it, "
            f"{-10 * np.log10(inside):.1f} dB below the unweighted level")
    hint = (" If its frequencies are in kHz, multiply them by 1000: "
            f"{100 * (1.0 - outside(1000.0)):.3g} % of the energy would lie "
            f"inside it." if outside(1000.0) < share else "")
    warnings.warn(
        f"{who}: the weighting table spans {edges[0]:g}-{edges[-1]:g} Hz and "
        f"carries no weight outside it, and {kept}.{hint}",
        ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP)
