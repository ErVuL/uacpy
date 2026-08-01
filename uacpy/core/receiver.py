"""
Receiver class for defining hydrophones and receiver arrays
"""

import copy as _copy
import warnings

import numpy as np
from typing import Union, List, Optional
from dataclasses import dataclass

from uacpy.core.exceptions import ConfigurationError
from uacpy.core._carrier_validate import (
    _require_non_negative, _require_strictly_increasing,
)


# eq=False: a dataclass __eq__ over ndarray fields raises; compare by identity.
@dataclass(eq=False)
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
        Receiver *sampling layout*: 'grid' (default — field on the full
        depth×range cross-product) or 'line' (depths and ranges paired
        point-by-point, requiring ``len(depths) == len(ranges)``, e.g. a
        glider track or tilted array). Note: 'line' here is a coordinate-
        pairing rule and is unrelated to :class:`~uacpy.Source`'s
        ``source_type='line'`` (a physical line-source geometry). Unlike
        ``'grid'`` (whose axes must be strictly increasing), a ``'line'``
        layout is a raw sampling list: ordering and uniqueness are the
        caller's responsibility. Feeding a non-monotonic line into a reader
        that interpolates on range/depth is undefined.

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

    depths: Union[float, List[float], np.ndarray]
    ranges: Optional[Union[float, List[float], np.ndarray]] = None
    receiver_type: str = 'grid'

    def __post_init__(self):
        valid_types = ('grid', 'line')
        if self.receiver_type not in valid_types:
            raise ConfigurationError(
                f"receiver_type must be one of {list(valid_types)}, "
                f"got {self.receiver_type!r}"
            )
        if self.ranges is None:
            warnings.warn(
                "Receiver: ranges not given, defaulting to a single point at "
                "0 m (the source location), which is singular for TL/pressure "
                "runs; pass explicit ranges= to avoid this.",
                UserWarning,
                stacklevel=2,
            )
            self.ranges = 0.0

        self.depths = np.atleast_1d(np.array(self.depths, dtype=np.float64))
        self.ranges = np.atleast_1d(np.array(self.ranges, dtype=np.float64))

        if self.depths.size < 1:
            raise ConfigurationError(
                "receiver depths must contain at least one value, got empty array"
            )
        if self.ranges.size < 1:
            raise ConfigurationError(
                "receiver ranges must contain at least one value, got empty array"
            )

        _require_non_negative(
            self.depths, "receiver depths", hint="metres, positive down from surface")
        _require_non_negative(
            self.ranges, "receiver ranges", hint="metres, outward from source")

        if self.receiver_type == 'line':
            if len(self.depths) != len(self.ranges):
                if len(self.ranges) == 1:
                    self.ranges = np.full_like(self.depths, self.ranges[0])
                elif len(self.depths) == 1:
                    self.depths = np.full_like(self.ranges, self.depths[0])
                else:
                    raise ConfigurationError(
                        "For receiver_type='line', depths and ranges must "
                        "have the same length or one must be scalar"
                    )
        else:
            _require_strictly_increasing(self.depths, "Receiver.depths")
            _require_strictly_increasing(self.ranges, "Receiver.ranges")

    @property
    def n_depths(self) -> int:
        """Number of depth entries."""
        return len(self.depths)

    @property
    def n_ranges(self) -> int:
        """Number of range entries."""
        return len(self.ranges)

    @property
    def depth_min(self) -> float:
        """Minimum receiver depth."""
        return float(np.min(self.depths))

    @property
    def depth_max(self) -> float:
        """Maximum receiver depth."""
        return float(np.max(self.depths))

    @property
    def range_min(self) -> float:
        """Minimum receiver range."""
        return float(np.min(self.ranges))

    @property
    def range_max(self) -> float:
        """Maximum receiver range."""
        return float(np.max(self.ranges))

    def __repr__(self) -> str:
        if self.receiver_type == 'grid':
            return (f"Receiver(grid: {self.n_depths} depths × {self.n_ranges} ranges)")
        return (f"Receiver(line: {self.n_depths} paired points, "
                f"type='{self.receiver_type}')")

    def copy(self):
        """Deep copy (symmetric with the other carriers)."""
        return _copy.deepcopy(self)
