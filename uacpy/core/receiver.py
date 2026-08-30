"""
Receiver class for defining hydrophones and receiver arrays
"""

import copy as _copy
import warnings

import numpy as np
from typing import TYPE_CHECKING, Any, Union, List, Optional
from dataclasses import dataclass

from uacpy.core.exceptions import ConfigurationError
from uacpy.core.constants import DECK_DEPTH_RESOLUTION_M, DECK_RANGE_RESOLUTION_M
from uacpy.core._carrier_validate import (
    _reject_complex, _require_non_negative, _require_strictly_increasing,
)


#: Constructor sentinel for ``ranges``: ``None`` means "not given", which
#: ``__post_init__`` turns into a single 0 m point after warning. Typed
#: ``Any`` so the field itself can declare the ndarray every attribute read
#: sees, without the sentinel widening that declaration back to Optional.
_RANGES_NOT_GIVEN: Any = None


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

    depths: np.ndarray
    ranges: np.ndarray = _RANGES_NOT_GIVEN
    receiver_type: str = 'grid'

    if TYPE_CHECKING:
        # The two roles of a dataclass field annotation, separated: the
        # attributes hold what ``__post_init__`` normalizes them to (float64
        # ndarrays, as the Attributes section above says), while the
        # constructor keeps taking the wide input union the Parameters
        # section documents. Declaring both through the field annotation
        # alone gives the union to every attribute read, so ``r.ranges.max()``
        # and ``for d in r.depths`` are reported as errors in downstream code
        # that runs correctly. Never executed, so the decorator compiles the
        # runtime ``__init__`` from the fields exactly as before.
        def __init__(
            self,
            depths: Union[float, List[float], np.ndarray],
            ranges: Optional[Union[float, List[float], np.ndarray]] = None,
            receiver_type: str = 'grid',
        ) -> None: ...

    def __post_init__(self):
        if self.receiver_type != 'grid':
            raise ConfigurationError(
                f"receiver_type must be 'grid'; got {self.receiver_type!r}. "
                "receiver_type='line' is not implemented — every model "
                "returns the full depth x range grid, so the paired "
                "(depths[i], ranges[i]) sampling would be silently ignored. "
                "Use receiver_type='grid' and index the diagonal yourself: "
                "tl[np.arange(len(depths)), np.arange(len(ranges))]."
            )
        if self.ranges is None:
            # stacklevel=3, not 2: a dataclass reaches ``__post_init__``
            # through the ``__init__`` the decorator compiles from a string,
            # so level 2 is that generated frame and attributes the warning
            # to ``<string>:6``. Every call site then shares one dedup key
            # and only the first ``Receiver(depths=…)`` in a program warns.
            # ``skip_file_prefixes`` cannot fix it — the generated frame's
            # ``<string>`` filename matches no package prefix, so the walk
            # stops there (measured).
            # The count survives the second-entry-point test that retired
            # most hand-counts: the models build a ``Receiver`` internally,
            # but every one of those passes an explicit ``ranges=``, so this
            # branch is reachable only from a user's own constructor call.
            warnings.warn(
                "Receiver: ranges not given, defaulting to a single point at "
                "0 m (the source location), which is singular for TL/pressure "
                "runs; pass explicit ranges= to avoid this.",
                UserWarning,
                stacklevel=3,
            )
            # An array, not the 0.0 scalar: the field declares ndarray, and
            # the atleast_1d cast below turns either into the same array([0.]).
            self.ranges = np.zeros(1)

        # Ahead of the float64 casts below, which discard an imaginary part —
        # see _reject_complex for the two ways they do it.
        _reject_complex(self.depths, "receiver depths")
        _reject_complex(self.ranges, "receiver ranges")
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

# The dataclass compiles ``__init__`` from the *field* annotations, so
# ``inspect.signature`` / ``help()`` would advertise a default the annotation
# refuses (``ranges: np.ndarray = None``). Restate the input types on the
# generated ``__init__`` so the runtime signature says what the block above
# and the Parameters section say. Annotations only: no default, no field and
# no behaviour changes, and the class annotations — which are what an
# attribute read is checked against — are untouched.
Receiver.__init__.__annotations__.update(
    depths=Union[float, List[float], np.ndarray],
    ranges=Optional[Union[float, List[float], np.ndarray]],
)
