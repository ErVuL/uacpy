"""
Receiver class for defining hydrophones and receiver arrays
"""

import copy as _copy
import warnings

import numpy as np
from typing import Union, List, Optional
from dataclasses import dataclass

from uacpy.core.exceptions import ConfigurationError
from uacpy.core.constants import DECK_DEPTH_RESOLUTION_M, DECK_RANGE_RESOLUTION_M
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
        Receiver *sampling layout*. ``'grid'`` (default) evaluates the field
        on the full depth×range cross-product and is the only implemented
        layout. ``'line'`` — depths and ranges paired point-by-point, e.g. a
        glider track or tilted array — names the axis but **raises**: no
        model's result assembly collapses the grid to paired samples, so
        accepting it would silently return the cross-product instead. Use
        ``'grid'`` and index the diagonal:
        ``tl[np.arange(len(depths)), np.arange(len(ranges))]``. Note this
        ``'line'`` is a coordinate-pairing rule, unrelated to
        :class:`~uacpy.Source`'s ``source_type='line'`` (a physical
        line-source geometry).

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
        if self.receiver_type == 'line':
            raise ConfigurationError(
                "receiver_type='line' is not implemented — every model "
                "returns the full depth x range grid, so the paired "
                "(depths[i], ranges[i]) sampling would be silently ignored. "
                "Use receiver_type='grid' and index the diagonal yourself: "
                "tl[np.arange(len(depths)), np.arange(len(ranges))]."
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

        _require_strictly_increasing(self.depths, "Receiver.depths",
                                     min_step=DECK_DEPTH_RESOLUTION_M)
        _require_strictly_increasing(self.ranges, "Receiver.ranges",
                                     min_step=DECK_RANGE_RESOLUTION_M)

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
        return f"Receiver(grid: {self.n_depths} depths × {self.n_ranges} ranges)"

    def copy(self):
        """Deep copy (symmetric with the other carriers)."""
        return _copy.deepcopy(self)
