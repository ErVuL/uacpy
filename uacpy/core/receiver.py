"""
Receiver class for defining hydrophones and receiver arrays
"""

import numpy as np
from typing import Union, List, Tuple, Optional


class Receiver:
    """
    Acoustic receiver definition

    Represents one or more receivers (hydrophones) at specified positions.

    Parameters
    ----------
    depths : float or array-like
        Receiver depth(s) in meters. Positive down from surface.
    ranges : float or array-like, optional
        Receiver range(s) in meters. Default is single point at 0m.
    positions : array-like, optional
        For 3D: receiver positions as [(x, y, z), ...] or [(x, y), ...]
        If provided, overrides depths and ranges.
    receiver_type : str, optional
        Type of receiver array: 'point', 'line', 'grid'. Default is 'grid'.

    Attributes
    ----------
    depths : ndarray
        Receiver depths
    ranges : ndarray
        Receiver ranges
    positions : ndarray
        3D receiver positions
    receiver_type : str
        Receiver type
    n_receivers : int
        Total number of receivers
    n_depths : int
        Number of unique depths
    n_ranges : int
        Number of unique ranges

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

    3D positions:

    >>> positions = [(100, 200, 50), (300, 400, 75)]
    >>> rx = Receiver(positions=positions)
    """

    def __init__(
        self,
        depths: Optional[Union[float, List[float], np.ndarray]] = None,
        ranges: Optional[Union[float, List[float], np.ndarray]] = None,
        positions: Optional[Union[List[Tuple], np.ndarray]] = None,
        receiver_type: str = 'grid',
    ):
        self.receiver_type = receiver_type

        # Handle 3D positions
        if positions is not None:
            positions = np.atleast_2d(positions)
            if positions.shape[1] == 2:
                # (x, y) -> add z=0
                self.positions = np.column_stack([positions, np.zeros(len(positions))])
            elif positions.shape[1] == 3:
                self.positions = positions
            else:
                raise ValueError("positions must have shape (N, 2) or (N, 3)")

            # Extract depths and ranges from positions
            self.depths = np.unique(self.positions[:, 2])
            if positions.shape[1] >= 2:
                self.ranges = np.unique(np.sqrt(
                    self.positions[:, 0]**2 + self.positions[:, 1]**2
                ))
            else:
                self.ranges = np.array([0.0])

        else:
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

            # For grid type, create mesh of all combinations
            if receiver_type == 'grid':
                r_grid, z_grid = np.meshgrid(self.ranges, self.depths)
                self.positions = np.column_stack([
                    r_grid.ravel(),
                    np.zeros(r_grid.size),
                    z_grid.ravel()
                ])
            else:
                # For line type, pair depths and ranges
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

                self.positions = np.column_stack([
                    self.ranges,
                    np.zeros_like(self.ranges),
                    self.depths
                ])

    @property
    def n_receivers(self) -> int:
        """Total number of receivers"""
        return len(self.positions)

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
        return np.min(self.depths)

    @property
    def depth_max(self) -> float:
        """Maximum receiver depth"""
        return np.max(self.depths)

    @property
    def range_min(self) -> float:
        """Minimum receiver range"""
        return np.min(self.ranges)

    @property
    def range_max(self) -> float:
        """Maximum receiver range"""
        return np.max(self.ranges)

    def get_positions(self) -> np.ndarray:
        """
        Get all receiver positions

        Returns
        -------
        positions : ndarray
            Array of shape (n_receivers, 3) with columns [x, y, z]
        """
        return self.positions.copy()

    def get_cylindrical_positions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get receiver positions in cylindrical coordinates

        Returns
        -------
        r : ndarray
            Radial distances from origin
        theta : ndarray
            Azimuthal angles in radians
        z : ndarray
            Depths
        """
        x, y, z = self.positions[:, 0], self.positions[:, 1], self.positions[:, 2]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return r, theta, z

    def subset(self, depth_range: Optional[Tuple[float, float]] = None,
               range_range: Optional[Tuple[float, float]] = None):
        """
        Create a subset of receivers within specified ranges

        Parameters
        ----------
        depth_range : tuple, optional
            (min_depth, max_depth) to include
        range_range : tuple, optional
            (min_range, max_range) to include

        Returns
        -------
        receiver : Receiver
            New receiver instance with subset of positions
        """
        mask = np.ones(self.n_receivers, dtype=bool)

        if depth_range is not None:
            z = self.positions[:, 2]
            mask &= (z >= depth_range[0]) & (z <= depth_range[1])

        if range_range is not None:
            r = np.sqrt(self.positions[:, 0]**2 + self.positions[:, 1]**2)
            mask &= (r >= range_range[0]) & (r <= range_range[1])

        subset_positions = self.positions[mask]

        return Receiver(positions=subset_positions, receiver_type=self.receiver_type)

    def __repr__(self) -> str:
        if self.receiver_type == 'grid':
            return (f"Receiver(grid: {self.n_depths} depths × {self.n_ranges} ranges "
                    f"= {self.n_receivers} receivers)")
        else:
            return f"Receiver({self.n_receivers} receivers, type='{self.receiver_type}')"

    def copy(self):
        """Create a deep copy of the receiver"""
        return Receiver(
            positions=self.positions.copy(),
            receiver_type=self.receiver_type
        )
