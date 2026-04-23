"""
Source class for defining acoustic sources in underwater environments
"""

import numpy as np
from typing import Union, List, Optional


class Source:
    """
    Acoustic source definition

    Represents one or more acoustic sources with specified depths, frequencies,
    and (for ray models) launch angles.

    Parameters
    ----------
    depth : float or array-like
        Source depth(s) in meters. Positive down from surface.
    frequency : float or array-like
        Source frequency or frequencies in Hz
    angles : array-like, optional
        Launch angles in degrees. Used for ray-tracing models.
        Default is [-80, 80] with 361 angles.
    source_type : str, optional
        Type of source: 'point' (default) or 'line'. Used by Bellhop/Scooter
        to select point (cylindrical) vs. line (Cartesian) geometry.

    Attributes
    ----------
    depth : ndarray
        Source depth(s)
    frequency : ndarray
        Source frequency/frequencies
    angles : ndarray
        Launch angles for ray tracing
    source_type : str
        Source type

    Examples
    --------
    Single source at 50m depth, 100 Hz:

    >>> source = Source(depth=50, frequency=100)

    Vertical source array:

    >>> source = Source(depth=[10, 20, 30], frequency=200)

    Custom launch angles:

    >>> source = Source(depth=50, frequency=100,
    ...                 angles=np.linspace(-45, 45, 91))
    """

    def __init__(
        self,
        depth: Union[float, List[float], np.ndarray],
        frequency: Union[float, List[float], np.ndarray],
        angles: Optional[Union[List[float], np.ndarray]] = None,
        source_type: str = 'point',
    ):
        self.depth = np.atleast_1d(np.array(depth, dtype=np.float64))
        self.frequency = np.atleast_1d(np.array(frequency, dtype=np.float64))

        if np.any(self.depth < 0):
            raise ValueError("Source depths must be positive (down from surface)")

        if angles is None:
            self.angles = np.linspace(-80, 80, 361)
        else:
            self.angles = np.array(angles, dtype=np.float64)

        # Only 'point' and 'line' are consumed by writers.
        valid_types = ['point', 'line']
        if source_type not in valid_types:
            raise ValueError(f"source_type must be one of {valid_types}")
        self.source_type = source_type

    @property
    def n_sources(self) -> int:
        """Number of sources."""
        return len(self.depth)

    @property
    def n_frequencies(self) -> int:
        """Number of frequencies."""
        return len(self.frequency)

    @property
    def n_angles(self) -> int:
        """Number of launch angles."""
        return len(self.angles)

    def __repr__(self) -> str:
        if self.n_sources == 1:
            depth_str = f"{self.depth[0]:.1f}m"
        else:
            depth_str = f"{self.n_sources} sources"

        if self.n_frequencies == 1:
            freq_str = f"{self.frequency[0]:.1f}Hz"
        else:
            freq_str = f"{self.n_frequencies} frequencies"

        return (f"Source({depth_str}, {freq_str}, "
                f"{self.n_angles} angles, type='{self.source_type}')")

    def copy(self):
        """Return a deep copy of the source."""
        return Source(
            depth=self.depth.copy(),
            frequency=self.frequency.copy(),
            angles=self.angles.copy(),
            source_type=self.source_type,
        )
