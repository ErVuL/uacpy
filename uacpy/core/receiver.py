"""
Receiver class for defining hydrophones and receiver arrays
"""

import numpy as np
from typing import Union, List, Optional


class Receiver:
    """
    Acoustic receiver definition

    Represents one or more receivers (hydrophones) at specified depths and
    ranges.  For grid-type receivers, the model evaluates the field on the
    full depth x range cartesian grid; for line-type receivers, depths and
    ranges are paired point-by-point.

    Parameters
    ----------
    depths : float or array-like
        Receiver depth(s) in meters. Positive down from surface.
    ranges : float or array-like, optional
        Receiver range(s) in meters. Default is single point at 0m.
    receiver_type : str, optional
        Type of receiver array: 'grid' or 'line'. Default is 'grid'.

    Attributes
    ----------
    depths : ndarray
        Receiver depths
    ranges : ndarray
        Receiver ranges
    receiver_type : str
        Receiver type
    n_depths : int
        Number of depth points
    n_ranges : int
        Number of range points

    Examples
    --------
    Single receiver at 50m depth, 1km range:

    >>> rx = Receiver(depths=50, ranges=1000)

    Vertical line array:

    >>> rx = Receiver(depths=np.linspace(10, 90, 9), ranges=5000)

    Grid of receivers:

    >>> rx = Receiver(
    ...     depths=np.linspace(0, 100, 51),
    ...     ranges=np.linspace(0, 10000, 201)
    ... )
    """

    def __init__(
        self,
        depths: Optional[Union[float, List[float], np.ndarray]] = None,
        ranges: Optional[Union[float, List[float], np.ndarray]] = None,
        receiver_type: str = 'grid',
    ):
        self.receiver_type = receiver_type

        # Use depths and ranges
        if depths is None:
            depths = 0.0
        if ranges is None:
            ranges = 0.0

        self.depths = np.atleast_1d(np.array(depths, dtype=np.float64))
        self.ranges = np.atleast_1d(np.array(ranges, dtype=np.float64))

        # Validate depths
        if np.any(self.depths < 0):
            raise ValueError("Receiver depths must be positive (down from surface)")

        # For line type, ensure depths and ranges can be paired.
        if receiver_type == 'line':
            if len(self.depths) != len(self.ranges):
                if len(self.ranges) == 1:
                    self.ranges = np.full_like(self.depths, self.ranges[0])
                elif len(self.depths) == 1:
                    self.depths = np.full_like(self.ranges, self.depths[0])
                else:
                    raise ValueError(
                        "For receiver_type='line', depths and ranges must have "
                        "same length or one must be scalar"
                    )

    @property
    def n_depths(self) -> int:
        """Number of unique depth levels"""
        return len(self.depths)

    @property
    def n_ranges(self) -> int:
        """Number of unique ranges"""
        return len(self.ranges)

    @property
    def depth_min(self) -> float:
        """Minimum receiver depth"""
        return float(np.min(self.depths))

    @property
    def depth_max(self) -> float:
        """Maximum receiver depth"""
        return float(np.max(self.depths))

    @property
    def range_min(self) -> float:
        """Minimum receiver range"""
        return float(np.min(self.ranges))

    @property
    def range_max(self) -> float:
        """Maximum receiver range"""
        return float(np.max(self.ranges))

    def __repr__(self) -> str:
        if self.receiver_type == 'grid':
            return (f"Receiver(grid: {self.n_depths} depths × {self.n_ranges} ranges)")
        return (f"Receiver(line: {self.n_depths} paired points, "
                f"type='{self.receiver_type}')")

    def copy(self):
        """Create a deep copy of the receiver"""
        return Receiver(
            depths=self.depths.copy(),
            ranges=self.ranges.copy(),
            receiver_type=self.receiver_type,
        )
