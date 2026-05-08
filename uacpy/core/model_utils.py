"""User-facing helpers for building model inputs.

Exposes utilities for assembling receiver grids and validating source /
receiver depths against an environment. Models do their own validation
inside ``PropagationModel.validate_inputs``; these helpers exist for
user code that builds inputs programmatically.
"""

import numpy as np

from uacpy.core.exceptions import InvalidDepthError


class ReceiverGridBuilder:
    """Build common receiver-grid layouts for propagation runs."""

    @staticmethod
    def linear_grid(
        depth_min: float, depth_max: float, n_depths: int,
        range_min: float, range_max: float, n_ranges: int,
    ):
        """Linearly-spaced rectangular receiver grid.

        Returns
        -------
        depths, ranges : ndarray
            1-D arrays of depths (m) and ranges (m).
        """
        depths = np.linspace(depth_min, depth_max, n_depths)
        ranges = np.linspace(range_min, range_max, n_ranges)
        return depths, ranges

    @staticmethod
    def log_range_grid(
        depths: np.ndarray, range_min: float, range_max: float, n_ranges: int,
    ):
        """Log-spaced ranges with user-supplied depth array."""
        ranges = np.geomspace(max(range_min, 1.0), range_max, n_ranges)
        return np.asarray(depths, dtype=float), ranges

    @staticmethod
    def vertical_array(
        depth_min: float, depth_max: float, n_depths: int, range_m: float,
    ):
        """Single-range vertical receiver array."""
        return np.linspace(depth_min, depth_max, n_depths), np.array([range_m])

    @staticmethod
    def horizontal_array(
        depth_m: float, range_min: float, range_max: float, n_ranges: int,
    ):
        """Single-depth horizontal receiver array."""
        return np.array([depth_m]), np.linspace(range_min, range_max, n_ranges)


def validate_source_depth(source_depth: float, env_depth: float, margin: float = 1.0):
    """Raise ``InvalidDepthError`` if ``source_depth`` is outside the water column.

    ``margin`` (m) lets the user accept sources within the bottom layer
    or above the surface by that much; default 1 m.
    """
    if source_depth < -margin or source_depth > env_depth + margin:
        raise InvalidDepthError(
            f"Source depth {source_depth:.1f} m outside water column "
            f"[0, {env_depth:.1f}] m (margin {margin} m)."
        )


def validate_receiver_depths(
    receiver_depths: np.ndarray, env_depth: float, margin: float = 1.0,
):
    """Raise ``InvalidDepthError`` if any receiver depth is outside the water column."""
    arr = np.atleast_1d(np.asarray(receiver_depths, dtype=float))
    bad = arr[(arr < -margin) | (arr > env_depth + margin)]
    if len(bad):
        raise InvalidDepthError(
            f"Receiver depth(s) {list(bad)} outside water column "
            f"[0, {env_depth:.1f}] m (margin {margin} m)."
        )
