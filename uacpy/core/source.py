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
    depths : float or array-like
        Source depth(s) in meters. Positive down from surface.
    frequencies : float or array-like
        Source frequency or frequencies in Hz
    angles : array-like, optional
        Launch angles in degrees. Used for ray-tracing models.
        Default is [-80, 80] with 361 angles.
    source_type : str, optional
        Type of source: 'point' (default) or 'line'. Used by Bellhop/Scooter
        to select point (cylindrical) vs. line (Cartesian) geometry.

    Attributes
    ----------
    depths : ndarray
        Source depth(s)
    frequencies : ndarray
        Source frequency/frequencies
    angles : ndarray
        Launch angles for ray tracing
    source_type : str
        Source type

    Examples
    --------
    Single source at 50m depth, 100 Hz:

    >>> source = Source(depths=50, frequencies=100)

    Vertical source array:

    >>> source = Source(depths=[10, 20, 30], frequencies=200)

    Custom launch angles:

    >>> source = Source(depths=50, frequencies=100,
    ...                 angles=np.linspace(-45, 45, 91))
    """

    def __init__(
        self,
        depths: Union[float, List[float], np.ndarray],
        frequencies: Union[float, List[float], np.ndarray],
        angles: Optional[Union[List[float], np.ndarray]] = None,
        source_type: str = 'point',
    ):
        self.depths = np.atleast_1d(np.array(depths, dtype=np.float64))
        self.frequencies = np.atleast_1d(np.array(frequencies, dtype=np.float64))

        if np.any(self.depths < 0):
            raise ValueError(
                f"source depths must be non-negative (down from surface), got {self.depths.tolist()}"
            )

        if np.any(self.frequencies <= 0):
            raise ValueError(
                f"source frequencies must be strictly positive (Hz), got {self.frequencies.tolist()}"
            )

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
        return len(self.depths)

    @property
    def n_frequencies(self) -> int:
        """Number of frequencies."""
        return len(self.frequencies)

    @property
    def n_angles(self) -> int:
        """Number of launch angles."""
        return len(self.angles)

    def __repr__(self) -> str:
        if self.n_sources == 1:
            depth_str = f"{self.depths[0]:.1f}m"
        else:
            depth_str = f"{self.n_sources} sources"

        if self.n_frequencies == 1:
            freq_str = f"{self.frequencies[0]:.1f}Hz"
        else:
            freq_str = f"{self.n_frequencies} frequencies"

        return (f"Source({depth_str}, {freq_str}, "
                f"{self.n_angles} angles, type='{self.source_type}')")

    def copy(self):
        """Return a deep copy of the source."""
        return Source(
            depths=self.depths.copy(),
            frequencies=self.frequencies.copy(),
            angles=self.angles.copy(),
            source_type=self.source_type,
        )
