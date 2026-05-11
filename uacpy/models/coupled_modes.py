"""
Range-dependent environment segmentation for KrakenField.

Segments a range-dependent environment into range slices, each with a
range-independent Environment, for use by AT's multi-profile .env format.
"""

import numpy as np
from typing import List, Tuple, Optional

from uacpy.core.environment import (
    Environment, RangeDependentBottom, RangeDependentLayeredBottom,
)


def segment_environment_by_range(
    env: Environment,
    n_segments: Optional[int] = None,
    max_segment_length: float = 2000.0,
) -> List[Tuple[float, Environment]]:
    """
    Segment a range-dependent environment into range slices

    Parameters
    ----------
    env : Environment
        Range-dependent environment to segment
    n_segments : int, optional
        Number of segments. If None, automatically determined.
    max_segment_length : float
        Maximum segment length in metres (default 2000 m).

    Returns
    -------
    segments : list of (range, Environment)
        List of ``(range in metres, environment)`` tuples for each segment.

    Notes
    -----
    Segments are created at bathymetry / SSP / RD-bottom change points and
    at regular intervals if those vary slowly.
    """
    if not env.is_range_dependent:
        return [(0.0, env)]

    bathy_ranges_m = env.bathymetry[:, 0]
    max_range_m = float(bathy_ranges_m[-1])

    if env.ssp.is_range_dependent:
        max_range_m = max(max_range_m, float(env.ssp.ranges[-1]))
    if isinstance(env.bottom, (RangeDependentBottom, RangeDependentLayeredBottom)):
        max_range_m = max(max_range_m, float(env.bottom.ranges[-1]))

    if max_range_m <= 0:
        return [(0.0, env)]

    if n_segments is not None:
        segment_ranges_m = np.linspace(0, max_range_m, n_segments)
    else:
        # Automatic segmentation: union the change-point ranges from
        # bathymetry, 2-D SSP, and RD-bottom axes; insert intermediate
        # points where the gap between consecutive change points exceeds
        # ``max_segment_length``.
        key_ranges_m = set(bathy_ranges_m.tolist())
        if env.ssp.is_range_dependent:
            key_ranges_m.update(env.ssp.ranges.tolist())
        if isinstance(env.bottom, (RangeDependentBottom, RangeDependentLayeredBottom)):
            key_ranges_m.update(env.bottom.ranges.tolist())
        key_ranges_m = sorted(key_ranges_m)

        segment_ranges_m = [key_ranges_m[0]]
        for i in range(1, len(key_ranges_m)):
            prev = key_ranges_m[i - 1]
            curr = key_ranges_m[i]
            seg_length = curr - prev
            if seg_length > max_segment_length:
                n_subseg = int(np.ceil(seg_length / max_segment_length))
                subseg_ranges = np.linspace(
                    prev, curr, n_subseg + 1,
                )[1:-1]
                segment_ranges_m.extend(subseg_ranges)
            segment_ranges_m.append(curr)
        segment_ranges_m = sorted(set(segment_ranges_m))

    segments = []
    for r in segment_ranges_m:
        depth_at_range = float(np.asarray(env.bathymetry_at_range(r)).flat[0])

        bottom_segment = env.bottom_at_range(r)
        ssp_at_range = env.ssp.eval(range=r).to_pairs()

        # Kraken .env writer uses .1f for the bottom depth on the mesh
        # line, so the deepest SSP point must match that rounded value.
        depth_rounded = float(f"{depth_at_range:.1f}")
        ssp_for_segment = ssp_at_range[ssp_at_range[:, 0] < depth_rounded].copy()
        c_at_depth = float(np.interp(depth_rounded, ssp_at_range[:, 0], ssp_at_range[:, 1]))
        ssp_for_segment = np.vstack([ssp_for_segment, [depth_rounded, c_at_depth]])
        # Kraken needs ≥2 samples per medium; on shoaling segments where the
        # seafloor is shallower than every SSP sample, prepend the surface.
        if len(ssp_for_segment) < 2:
            c_at_surface = float(np.interp(0.0, ssp_at_range[:, 0], ssp_at_range[:, 1]))
            ssp_for_segment = np.vstack([[0.0, c_at_surface], ssp_for_segment])

        from uacpy.core.environment import SoundSpeedProfile
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
