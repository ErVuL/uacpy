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
    max_segment_length_km: float = 2.0
) -> List[Tuple[float, Environment]]:
    """
    Segment a range-dependent environment into range slices

    Parameters
    ----------
    env : Environment
        Range-dependent environment to segment
    n_segments : int, optional
        Number of segments. If None, automatically determined.
    max_segment_length_km : float
        Maximum segment length in km (default 2.0 km)

    Returns
    -------
    segments : list of (range_km, Environment)
        List of (range, environment) tuples for each segment

    Notes
    -----
    Segments are created at bathymetry change points and at regular
    intervals if bathymetry varies slowly.
    """
    if not env.is_range_dependent:
        # Range-independent: single segment at range 0
        return [(0.0, env)]

    # Get the maximum range extent from bathymetry and 2D SSP ranges
    bathy_ranges_km = env.bathymetry[:, 0] / 1000.0
    max_range_km = bathy_ranges_km[-1]

    # Also consider 2D SSP range extent (env may be RD via SSP only)
    if hasattr(env, 'ssp_2d_ranges') and env.ssp_2d_ranges is not None:
        max_range_km = max(max_range_km, np.max(env.ssp_2d_ranges))

    if max_range_km <= 0:
        return [(0.0, env)]

    if n_segments is not None:
        # User-specified number of segments
        # Distribute evenly across range
        segment_ranges_km = np.linspace(0, max_range_km, n_segments)
    else:
        # Automatic segmentation based on bathymetry points,
        # 2D SSP range points, and maximum segment length
        key_ranges_km = set(bathy_ranges_km)
        if hasattr(env, 'ssp_2d_ranges') and env.ssp_2d_ranges is not None:
            key_ranges_km.update(env.ssp_2d_ranges)
        key_ranges_km = sorted(key_ranges_km)

        segment_ranges_km = [key_ranges_km[0]]

        for i in range(1, len(key_ranges_km)):
            segment_ranges_km.append(key_ranges_km[i])

            # Add intermediate points if segment too long
            seg_length = key_ranges_km[i] - segment_ranges_km[-2]
            if seg_length > max_segment_length_km:
                n_subseg = int(np.ceil(seg_length / max_segment_length_km))
                subseg_ranges = np.linspace(
                    segment_ranges_km[-2],
                    key_ranges_km[i],
                    n_subseg + 1
                )[1:-1]
                segment_ranges_km.extend(subseg_ranges)

        segment_ranges_km = sorted(set(segment_ranges_km))

    # Create environment for each segment
    segments = []
    for range_km in segment_ranges_km:
        range_m = range_km * 1000.0

        # Interpolate bathymetry at this range
        depth_at_range = float(np.asarray(env.get_bathymetry_depth(range_m)).flat[0])

        # Create range-independent environment at this range
        # This represents the "local" environment at this range
        from uacpy.core.environment import BoundaryProperties

        # Get bottom properties at this range (handles range-dependent bottom)
        bottom_segment = env.get_bottom_at_range(range_m)

        # Get SSP at this range (handles range-dependent SSP)
        ssp_at_range = env.get_ssp_at_range(range_m)

        # Prepare SSP for this segment depth
        # Truncate or extend SSP to match segment depth
        # The Kraken .env writer uses .1f for the bottom depth on the mesh line,
        # so the deepest SSP point must match that rounded value exactly.
        depth_rounded = float(f"{depth_at_range:.1f}")
        ssp_for_segment = ssp_at_range[ssp_at_range[:, 0] < depth_rounded].copy()

        # Add final point at exactly the rounded bottom depth
        c_at_depth = float(np.interp(depth_rounded, ssp_at_range[:, 0], ssp_at_range[:, 1]))
        ssp_for_segment = np.vstack([ssp_for_segment, [depth_rounded, c_at_depth]])

        env_segment = Environment(
            name=f"{env.name} @ {range_km:.1f}km",
            depth=depth_at_range,
            ssp_type=env.ssp_type,
            ssp_data=ssp_for_segment,
            sound_speed=env.sound_speed,
            bathymetry=None,  # Flat bottom at depth_at_range
            bottom=bottom_segment,
            surface=env.surface,
            attenuation=env.attenuation
        )

        segments.append((range_km, env_segment))

    return segments
