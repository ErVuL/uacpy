"""
Range-dependent environment segmentation for Kraken.

A model-support helper (not a :class:`PropagationModel`): segments a
range-dependent environment into range slices, each with a range-independent
Environment, for use by AT's multi-profile .env format.
"""

import numpy as np
from typing import List, Tuple, Optional

from uacpy.core.environment import Environment, SoundSpeedProfile
from uacpy.core.exceptions import ConfigurationError
from uacpy.io.oalib_writer import DECK_RANGE_QUANTUM_M

# Ceiling (m) on the gap between consecutive automatic segment edges:
# adiabatic/coupled mode summation interpolates each mode between profiles,
# so a slowly-varying stretch still gets a profile at least every 2 km.
# The caller-facing knob is ``n_segments`` (an explicit uniform decomposition
# overrides the automatic change-point edges entirely).
_MAX_SEGMENT_LENGTH_M = 2000.0


def segment_environment_by_range(
    env: Environment,
    n_segments: Optional[int] = None,
) -> List[Tuple[float, Environment]]:
    """
    Segment a range-dependent environment into range slices

    Parameters
    ----------
    env : Environment
        Range-dependent environment to segment
    n_segments : int, optional
        Number of evenly spaced segments (``>= 2``). If None, automatically
        determined from the environment's change points, with intermediate
        edges wherever the gap exceeds :data:`_MAX_SEGMENT_LENGTH_M`.

    Returns
    -------
    segments : list of (range, Environment)
        List of ``(range in metres, environment)`` tuples for each segment.

    Raises
    ------
    ConfigurationError
        If ``n_segments`` is given and smaller than 2.

    Notes
    -----
    Segments are created at bathymetry / SSP / RD-bottom change points and
    at regular intervals if those vary slowly. The first segment always sits
    at r=0 so no receiver range falls shoreward of every profile.
    """
    if not env.is_range_dependent:
        return [(0.0, env)]

    bathy_ranges_m = env.bathymetry.ranges
    max_range_m = float(bathy_ranges_m[-1])

    if env.ssp.is_range_dependent:
        max_range_m = max(max_range_m, float(env.ssp.ranges[-1]))
    if env.bottom.is_range_dependent:
        max_range_m = max(max_range_m, float(env.bottom.ranges[-1]))

    if max_range_m <= 0:
        return [(0.0, env)]

    if n_segments is not None:
        if n_segments < 2:
            raise ConfigurationError(
                f"n_segments must be >= 2 to describe a range-dependent "
                f"environment (got {n_segments}); 1 collapses the run to a "
                f"single profile at r=0. Pass n_segments=None for automatic "
                f"segmentation at the environment's change points."
            )
        segment_ranges_m = np.linspace(0, max_range_m, n_segments)
    else:
        # Automatic segmentation: union the change-point ranges from
        # bathymetry, 2-D SSP, and RD-bottom axes; insert intermediate
        # points where the gap between consecutive change points exceeds
        # ``_MAX_SEGMENT_LENGTH_M``.
        candidates = list(bathy_ranges_m.tolist())
        if env.ssp.is_range_dependent:
            candidates.extend(env.ssp.ranges.tolist())
        if env.bottom.is_range_dependent:
            candidates.extend(env.bottom.ranges.tolist())
        # Merge at the resolution the deck can express, not at exact float
        # equality: a bathymetry axis and an SSP axis that name the same physical
        # range through different arithmetic differ in the last bits, and both
        # would survive a set(). They then print as one token, and field.exe
        # divides by the zero gap (EvaluateADMod.f90:75) with no diagnostic.
        candidates.sort()
        key_ranges_m = []
        for r in candidates:
            if not key_ranges_m or r - key_ranges_m[-1] > DECK_RANGE_QUANTUM_M:
                key_ranges_m.append(r)
        # A profile axis that starts beyond r=0 would leave everything
        # shoreward of its first sample without a segment; anchor at 0 and let
        # the constant extrapolation every carrier already does fill it.
        if key_ranges_m[0] > 0:
            key_ranges_m.insert(0, 0.0)

        segment_ranges_m = [key_ranges_m[0]]
        for i in range(1, len(key_ranges_m)):
            prev = key_ranges_m[i - 1]
            curr = key_ranges_m[i]
            seg_length = curr - prev
            if seg_length > _MAX_SEGMENT_LENGTH_M:
                n_subseg = int(np.ceil(seg_length / _MAX_SEGMENT_LENGTH_M))
                subseg_ranges = np.linspace(
                    prev, curr, n_subseg + 1,
                )[1:-1]
                segment_ranges_m.extend(subseg_ranges)
            segment_ranges_m.append(curr)
        segment_ranges_m = sorted(set(segment_ranges_m))

    segments = []
    for r in segment_ranges_m:
        depth_at_range = float(np.asarray(env.bathymetry.eval(range=r)).flat[0])

        bottom_segment = env.bottom.at(range=r)
        ssp_at_range = env.ssp.eval(range=r).to_pairs()

        # AT ends a medium only at the SSP sample matching the mesh-line depth
        # (Acoustics-Toolbox/misc/sspMod.f90:352-362). Quantise the segment
        # depth to the nearest 0.1 m — a deliberate stability quantum, not a
        # writer requirement (``deck_depth`` round-trips 6 decimals): it keeps
        # segment decks identical under mm-scale bathymetry noise. Truncate
        # the profile above it and append an interpolated sample exactly
        # there; that appended sample is the one ``extend_to`` lands on.
        depth_rounded = float(f"{depth_at_range:.1f}")
        ssp_for_segment = ssp_at_range[ssp_at_range[:, 0] < depth_rounded].copy()
        c_at_depth = float(np.interp(depth_rounded, ssp_at_range[:, 0], ssp_at_range[:, 1]))
        ssp_for_segment = np.vstack([ssp_for_segment, [depth_rounded, c_at_depth]])
        # Kraken needs ≥2 samples per medium; on shoaling segments where the
        # seafloor is shallower than every SSP sample, prepend the surface.
        if len(ssp_for_segment) < 2:
            c_at_surface = float(np.interp(0.0, ssp_at_range[:, 0], ssp_at_range[:, 1]))
            ssp_for_segment = np.vstack([[0.0, c_at_surface], ssp_for_segment])

        seg_ssp = SoundSpeedProfile.from_pairs(
            ssp_for_segment, shape=env.ssp.shape,
        )
        env_segment = Environment(
            name=f"{env.name} @ {r / 1000.0:.1f}km",
            ssp=seg_ssp,
            bathymetry=depth_at_range,
            bottom=bottom_segment,
            surface=env.surface,
            absorption=env.absorption,
        )

        segments.append((r, env_segment))

    return segments
