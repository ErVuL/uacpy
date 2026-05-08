"""
Range-dependent environment segmentation for KrakenField.

Segments a range-dependent environment into range slices, each with a
range-independent Environment, for use by AT's multi-profile .env format.
"""

import numpy as np
from typing import List, Tuple, Optional

from uacpy.core.environment import Environment


def segment_environment_by_range(
    env: Environment,
    n_segments: Optional[int] = None,
    max_segment_length_m: float = 2000.0,
) -> List[Tuple[float, Environment]]:
    """
    Segment a range-dependent environment into range slices

    Parameters
    ----------
    env : Environment
        Range-dependent environment to segment
    n_segments : int, optional
        Number of segments. If None, automatically determined.
    max_segment_length_m : float
        Maximum segment length in metres (default 2000 m).

    Returns
    -------
    segments : list of (range_m, Environment)
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
    if env.bottom_rd is not None:
        max_range_m = max(max_range_m, float(env.bottom_rd.ranges[-1]))
    if env.bottom_rd_layered is not None:
        max_range_m = max(max_range_m, float(env.bottom_rd_layered.ranges[-1]))

    if max_range_m <= 0:
        return [(0.0, env)]

    if n_segments is not None:
        segment_ranges_m = np.linspace(0, max_range_m, n_segments)
    else:
        # Automatic segmentation: union the change-point ranges from
        # bathymetry, 2-D SSP, and RD-bottom axes; insert intermediate
        # points where the gap between consecutive change points exceeds
        # ``max_segment_length_m``.
        key_ranges_m = set(bathy_ranges_m.tolist())
        if env.ssp.is_range_dependent:
            key_ranges_m.update(env.ssp.ranges.tolist())
        if env.bottom_rd is not None:
            key_ranges_m.update(env.bottom_rd.ranges.tolist())
        if env.bottom_rd_layered is not None:
            key_ranges_m.update(env.bottom_rd_layered.ranges.tolist())
        key_ranges_m = sorted(key_ranges_m)

        segment_ranges_m = [key_ranges_m[0]]
        for i in range(1, len(key_ranges_m)):
            segment_ranges_m.append(key_ranges_m[i])
            seg_length = key_ranges_m[i] - segment_ranges_m[-2]
            if seg_length > max_segment_length_m:
                n_subseg = int(np.ceil(seg_length / max_segment_length_m))
                subseg_ranges = np.linspace(
                    segment_ranges_m[-2],
                    key_ranges_m[i],
                    n_subseg + 1,
                )[1:-1]
                segment_ranges_m.extend(subseg_ranges)
        segment_ranges_m = sorted(set(segment_ranges_m))

    segments = []
    for range_m in segment_ranges_m:
        depth_at_range = float(np.asarray(env.bathymetry_at_range(range_m)).flat[0])

        bottom_segment = env.bottom_at_range(range_m)
        ssp_at_range = env.ssp_at_range(range_m)

        # Kraken .env writer uses .1f for the bottom depth on the mesh
        # line, so the deepest SSP point must match that rounded value.
        depth_rounded = float(f"{depth_at_range:.1f}")
        ssp_for_segment = ssp_at_range[ssp_at_range[:, 0] < depth_rounded].copy()
        c_at_depth = float(np.interp(depth_rounded, ssp_at_range[:, 0], ssp_at_range[:, 1]))
        ssp_for_segment = np.vstack([ssp_for_segment, [depth_rounded, c_at_depth]])

        from uacpy.core.environment import SoundSpeedProfile
        seg_ssp = SoundSpeedProfile.from_pairs(
            ssp_for_segment, interp=env.ssp.interp,
        )
        env_segment = Environment(
            name=f"{env.name} @ {range_m / 1000.0:.1f}km",
            depth=depth_at_range,
            ssp=seg_ssp,
            bathymetry=None,
            bottom=bottom_segment,
            surface=env.surface,
            volume_attenuation=env.volume_attenuation,
        )

        segments.append((range_m, env_segment))

    return segments
