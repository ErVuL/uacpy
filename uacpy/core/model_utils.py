"""
Shared utilities for propagation models.

Eliminates code duplication across model implementations.
"""

import numpy as np
from typing import Optional

from uacpy.core.exceptions import InvalidDepthError
from uacpy.models.base import RunMode


class ParameterMapper:
    """
    Map user-friendly parameters to model-specific formats.

    Provides consistent parameter naming across models.
    """

    VOLUME_ATTEN_MAP = {
        'thorp': 'T',
        't': 'T',
        'francois': 'F',
        'francois-garrison': 'F',
        'f': 'F',
        'biological': 'B',
        'bio': 'B',
        'b': 'B',
        'none': None,
        '': None,
    }

    @classmethod
    def map_volume_attenuation(cls, value: Optional[str]) -> Optional[str]:
        """
        Standardize volume attenuation parameter.

        Parameters
        ----------
        value : str or None
            User-friendly name ('thorp', 'francois', 'biological', ...).

        Returns
        -------
        mapped : str or None
            Acoustics Toolbox format ('T', 'F', 'B', or None).
        """
        if value is None:
            return None

        normalized = value.lower().strip()
        return cls.VOLUME_ATTEN_MAP.get(normalized, value)

    @classmethod
    def map_run_mode_to_bellhop(cls, run_mode: RunMode) -> str:
        """
        Map a ``RunMode`` enum to the Bellhop ``run_type`` letter.

        Returns
        -------
        run_type : str
            Bellhop ``run_type`` ('C', 'I', 'S', 'R', 'E', 'A').
        """
        mapping = {
            RunMode.COHERENT_TL: 'C',
            RunMode.INCOHERENT_TL: 'I',
            RunMode.SEMICOHERENT_TL: 'S',
            RunMode.RAYS: 'R',
            RunMode.EIGENRAYS: 'E',
            RunMode.ARRIVALS: 'A',
            # Time-series synthesis is built from arrival data.
            RunMode.TIME_SERIES: 'A',
        }
        return mapping[run_mode]


class ReceiverGridBuilder:
    """
    Build standard receiver grids for common scenarios.

    Eliminates the need for users to manually construct receiver grids.
    """

    @staticmethod
    def build_tl_grid(
        env_depth: float,
        max_range: float,
        n_depths: int = 50,
        n_ranges: int = 100,
        depth_margin: float = 5.0
    ):
        """
        Build a standard TL computation grid.

        Parameters
        ----------
        env_depth : float
            Environment depth in meters.
        max_range : float
            Maximum range in meters.
        n_depths : int
            Number of depth points.
        n_ranges : int
            Number of range points.
        depth_margin : float
            Margin from surface/bottom in meters.

        Returns
        -------
        depths : ndarray
            Depth grid.
        ranges : ndarray
            Range grid (starts at max(1% of max_range, 10m) to avoid r=0).
        """
        depths = np.linspace(
            depth_margin,
            env_depth - depth_margin,
            n_depths
        )

        ranges = np.linspace(
            max(max_range * 0.01, 10.0),
            max_range,
            n_ranges
        )

        return depths, ranges

    @staticmethod
    def build_ray_grid(env_depth: float, max_range: float):
        """
        Build a 200×200 grid suitable for ray visualization.

        Parameters
        ----------
        env_depth : float
            Maximum environment depth in meters.
        max_range : float
            Maximum range in meters.

        Returns
        -------
        depths : ndarray
            Depth grid (200 points).
        ranges : ndarray
            Range grid (200 points).
        """
        depths = np.linspace(0, env_depth, 200)
        ranges = np.linspace(0, max_range, 200)
        return depths, ranges


def validate_source_depth(source_depth: float, env_depth: float, margin: float = 1.0):
    """
    Validate source depth against the environment.

    Parameters
    ----------
    source_depth : float
        Source depth in meters.
    env_depth : float
        Environment depth in meters.
    margin : float
        Safety margin in meters.

    Raises
    ------
    InvalidDepthError
        If the source depth is negative or within ``margin`` of the bottom.
    """
    if source_depth < 0:
        raise InvalidDepthError(source_depth, env_depth, "Source")

    if source_depth > env_depth - margin:
        raise InvalidDepthError(source_depth, env_depth, "Source")


def validate_receiver_depths(receiver_depths: np.ndarray, env_depth: float, margin: float = 1.0):
    """
    Validate receiver depths against the environment.

    Parameters
    ----------
    receiver_depths : ndarray
        Receiver depths in meters.
    env_depth : float
        Environment depth in meters.
    margin : float
        Safety margin in meters.

    Raises
    ------
    InvalidDepthError
        If any receiver depth is negative or within ``margin`` of the bottom.
    """
    if np.any(receiver_depths < 0):
        bad_depth = receiver_depths[receiver_depths < 0][0]
        raise InvalidDepthError(bad_depth, env_depth, "Receiver")

    if np.any(receiver_depths > env_depth - margin):
        bad_depth = receiver_depths[receiver_depths > env_depth - margin][0]
        raise InvalidDepthError(bad_depth, env_depth, "Receiver")
