"""Passive and active sonar equations (Urick 1983, Ch. 2; Etter Ch. 11).

All terms are in decibels. Sign and grouping follow Urick's table of sonar
parameters (reproduced in Etter, Table 11.1):

* Echo level (active):            ``EL = SL - 2*TL + TS``
* Noise background:               ``NL - DI``
* Passive signal excess:          ``SE = SL - TL - (NL - DI) - DT``
* Active, noise-limited:          ``SE = SL - 2*TL + TS - (NL - DI) - DT``
* Active, reverberation-limited:  ``SE = SL - 2*TL + TS - RL - DT``
* Figure of merit (passive):      ``FOM = SL - (NL - DI) - DT``

Detection occurs when the signal excess ``SE`` reaches zero. ``DT`` is the
detection threshold (recognition differential) — see :mod:`uacpy.sonar.detection`.
"""

from __future__ import annotations

import numpy as np


def echo_level(source_level, tl, target_strength):
    """Active echo level at the receiver: ``SL - 2*TL + TS`` (dB)."""
    return np.asarray(source_level, float) - 2.0 * np.asarray(tl, float) \
        + np.asarray(target_strength, float)


def noise_background(noise_level, directivity_index=0.0):
    """Noise masking background: ``NL - DI`` (dB)."""
    return np.asarray(noise_level, float) - np.asarray(directivity_index, float)


def passive_signal_excess(
    source_level, tl, noise_level, directivity_index=0.0, detection_threshold=0.0
):
    """Passive signal excess ``SE = SL - TL - (NL - DI) - DT`` (dB).

    ``source_level`` is the *target* radiated level. ``SE >= 0`` means the
    target is detectable.
    """
    return (
        np.asarray(source_level, float)
        - np.asarray(tl, float)
        - noise_background(noise_level, directivity_index)
        - np.asarray(detection_threshold, float)
    )


def active_signal_excess(
    source_level,
    tl,
    target_strength,
    *,
    noise_level=None,
    directivity_index=0.0,
    reverberation_level=None,
    detection_threshold=0.0,
):
    """Active signal excess (dB), noise- or reverberation-limited.

    Provide ``noise_level`` (noise-limited), ``reverberation_level``
    (reverb-limited), or both — in which case the louder background
    (incoherent sum) is used per range.

        noise-limited:  ``SE = SL - 2*TL + TS - (NL - DI) - DT``
        reverb-limited: ``SE = SL - 2*TL + TS - RL - DT``
    """
    if noise_level is None and reverberation_level is None:
        raise ValueError(
            "active_signal_excess: provide noise_level and/or reverberation_level"
        )
    el = echo_level(source_level, tl, target_strength)
    backgrounds = []
    if noise_level is not None:
        backgrounds.append(noise_background(noise_level, directivity_index))
    if reverberation_level is not None:
        backgrounds.append(np.asarray(reverberation_level, float))
    bcast = np.broadcast_arrays(*backgrounds)
    background = 10.0 * np.log10(
        np.sum([10.0 ** (b / 10.0) for b in bcast], axis=0)
    )
    return el - background - np.asarray(detection_threshold, float)


def figure_of_merit(
    source_level, noise_level, directivity_index=0.0, detection_threshold=0.0
):
    """Figure of merit ``FOM = SL - (NL - DI) - DT`` (dB).

    Equals the maximum allowable one-way TL (passive), or two-way TL when
    ``TS = 0`` (active).
    """
    return (
        np.asarray(source_level, float)
        - noise_background(noise_level, directivity_index)
        - np.asarray(detection_threshold, float)
    )


def detection_range(ranges_m, signal_excess_db):
    """Largest range (m) at which the signal excess is still non-negative.

    Finds the outermost zero-crossing of ``signal_excess_db`` versus range by
    linear interpolation. Returns ``np.inf`` if SE >= 0 everywhere, or
    ``np.nan`` if SE < 0 everywhere.

    Parameters
    ----------
    ranges_m : array
        Monotonically increasing ranges (m).
    signal_excess_db : array
        Signal excess (dB) at each range.
    """
    r = np.asarray(ranges_m, dtype=float)
    se = np.asarray(signal_excess_db, dtype=float)
    if r.shape != se.shape:
        raise ValueError("detection_range: ranges and signal_excess shape mismatch")
    positive = se >= 0.0
    if positive.all():
        return np.inf
    if not positive.any():
        return np.nan
    # Largest range with SE >= 0 is the outermost positive sample — this
    # captures a far-edge recovery (e.g. a convergence zone giving +,-,+).
    last_pos = int(np.where(positive)[0][-1])
    if last_pos == r.size - 1:
        # SE stays/recovers positive at the far edge; detectable out to the
        # last sampled range, with no crossing-down beyond it to interpolate.
        return float(r[last_pos])
    se0, se1 = se[last_pos], se[last_pos + 1]
    frac = se0 / (se0 - se1)
    return float(r[last_pos] + frac * (r[last_pos + 1] - r[last_pos]))
