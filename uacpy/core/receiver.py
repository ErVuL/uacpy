"""
Receiver class for defining hydrophones and receiver arrays
"""

import warnings

import numpy as np
from typing import TYPE_CHECKING, Any, Union, List, Optional

from uacpy.core._repr import axis, build
from uacpy.core.exceptions import ConfigurationError, FallbackWarning
from uacpy.core.deck_limits import (
    DECK_DEPTH_RESOLUTION_M, DECK_RANGE_RESOLUTION_M,
)
from uacpy.core._validate import (
    reject_complex, require_non_negative, require_strictly_increasing,
)
from uacpy.core._carrier import (
    DeepCopyMixin, RevalidateOnAssignMixin, carrier,
)
from uacpy.core._export import CarrierExport
from uacpy.core._warn_frames import USER_FRAME_SKIP

__all__ = [
    'Receiver',
]


#: Constructor sentinel for ``ranges``: ``None`` means "not given", which
#: ``__post_init__`` turns into a single 0 m point after warning. Typed
#: ``Any`` so the field itself can declare the ndarray every attribute read
#: sees, without the sentinel widening that declaration back to Optional.
_RANGES_NOT_GIVEN: Any = None


# eq=False: a dataclass __eq__ over ndarray fields raises; compare by identity.
# The constructor keeps the input types the Parameters section documents;
# without them ``inspect.signature`` / ``help()`` would advertise a default the
# field annotation refuses (``ranges: np.ndarray = None``).
@carrier(eq=False, init_annotations=dict(
    depths=Union[float, List[float], np.ndarray],
    ranges=Optional[Union[float, List[float], np.ndarray]],
))
class Receiver(RevalidateOnAssignMixin, DeepCopyMixin, CarrierExport):
    """
    Acoustic receiver definition

    Represents one or more receivers (hydrophones) at specified depths and
    ranges. The model evaluates the field on the full depth x range
    cartesian grid.

    Parameters
    ----------
    depths : float or array-like
        Receiver depth(s) in meters. Positive down from surface. More than
        one must be strictly increasing, with a minimum step of
        ``DECK_DEPTH_RESOLUTION_M`` (the resolution the decks write).
    ranges : float or array-like, optional
        Receiver range(s) in meters. Default is single point at 0m. More
        than one must be strictly increasing, with a minimum step of
        ``DECK_RANGE_RESOLUTION_M``.

    Attributes
    ----------
    depths : ndarray
        Receiver depths
    ranges : ndarray
        Receiver ranges
    n_depths : int
        Number of depth points
    n_ranges : int
        Number of range points

    Notes
    -----
    Paired samples — depths and ranges paired point-by-point, e.g. a
    glider track or tilted array — come from a grid run: every model
    returns the full depth×range cross-product. Run a grid over the
    track's distinct depths and ranges and index the pairs, which works
    for any track (a glider's yo-yo included)::

        zu, iz = np.unique(track_depths, return_inverse=True)
        ru, ir = np.unique(track_ranges, return_inverse=True)
        tl = model.run(env, src, Receiver(depths=zu, ranges=ru)).dB
        paired = tl[iz, ir]      # one value per track point

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

    if TYPE_CHECKING:
        # The two roles of a dataclass field annotation, separated: the
        # attributes hold what ``__post_init__`` normalizes them to (float64
        # ndarrays, as the Attributes section above says), while the
        # constructor keeps taking the wide input union the Parameters
        # section documents. Declaring both through the field annotation
        # alone gives the union to every attribute read, so ``r.ranges.max()``
        # and ``for d in r.depths`` are reported as errors in downstream code
        # that runs correctly. Never executed; the runtime ``__init__`` is
        # ``@carrier``'s, with the same parameters.
        def __init__(
            self,
            depths: Union[float, List[float], np.ndarray],
            ranges: Optional[Union[float, List[float], np.ndarray]] = None,
        ) -> None: ...

    def __post_init__(self):
        if self.ranges is None:
            # The walk, not a count: ``@carrier``'s ``__init__`` and the
            # rebuild an assignment runs are package frames the walk steps
            # over, so the warning names the user's line from a
            # ``Receiver(depths=…)`` and from ``receiver.ranges = None``
            # alike, and each call site keeps its own dedup key.
            warnings.warn(
                "Receiver: ranges not given, defaulting to a single point at "
                "0 m (the source location), which is singular for TL/pressure "
                "runs; pass explicit ranges= to avoid this.",
                FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
            # An array, not the 0.0 scalar: the field declares ndarray, and
            # the atleast_1d cast below turns either into the same array([0.]).
            object.__setattr__(self, 'ranges', np.zeros(1))

        # Ahead of the float64 casts below, which discard an imaginary part —
        # see _reject_complex for the two ways they do it.
        reject_complex(self.depths, "receiver depths")
        reject_complex(self.ranges, "receiver ranges")
        # Stored with object.__setattr__: a plain store to a set field is an
        # assignment, which rebuilds the Receiver through this method
        # (_RevalidateOnAssignMixin).
        object.__setattr__(self, 'depths', np.atleast_1d(
            np.array(self.depths, dtype=np.float64)))
        object.__setattr__(self, 'ranges', np.atleast_1d(
            np.array(self.ranges, dtype=np.float64)))
        for name in ('depths', 'ranges'):
            shape = getattr(self, name).shape
            if len(shape) != 1:
                raise ConfigurationError(
                    f"Receiver.{name} must be a scalar or a 1-D axis; got "
                    f"shape {shape}. The Receiver is the cross-product of "
                    f"its depth and range axes (the grid is built for "
                    f"you), so pass the two 1-D axes, not a meshgrid.")

        if self.depths.size < 1:
            raise ConfigurationError(
                "receiver depths must contain at least one value, got empty array."
            )
        if self.ranges.size < 1:
            raise ConfigurationError(
                "receiver ranges must contain at least one value, got empty array."
            )

        require_non_negative(
            self.depths, "receiver depths", hint="metres, positive down from surface")
        require_non_negative(
            self.ranges, "receiver ranges", hint="metres, outward from source")

        require_strictly_increasing(self.depths, "Receiver.depths",
                                    min_step=DECK_DEPTH_RESOLUTION_M)
        require_strictly_increasing(self.ranges, "Receiver.ranges",
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

    def grid(self):
        """``(Z, R)``: the depth and range (m) of every receiver of the
        grid, each of shape ``(n_depths, n_ranges)`` — the depth-first
        layout of a ``(depth, range)`` Field, so ``Z[i, j]`` and
        ``R[i, j]`` are where ``field.data[i, j]`` was computed."""
        return np.meshgrid(self.depths, self.ranges, indexing='ij')

    def __repr__(self) -> str:
        return build('Receiver', [axis(self.depths, 'depths', 'm'),
                                  axis(self.ranges, 'ranges', 'm')])
