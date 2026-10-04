"""Bathymetry shape carrier: seafloor depth as a function of range.

A 1-D profile (``depth`` vs ``range``), the seafloor analogue of
:class:`uacpy.core.ssp.SoundSpeedProfile`. Re-exported from
:mod:`uacpy.core.environment` for stable import paths.
"""

import numpy as np
from uacpy.core._carrier import carrier
from uacpy.core.collapse import DEPTH_COLLAPSE_METHODS, _method_list

from uacpy.core.exceptions import ConfigurationError
from uacpy.core._grid import _RangeProfile
from uacpy.core._validate import require_positive, scalar_or_none
from uacpy.core._provenance import coerce_data_sources

__all__ = [
    'Bathymetry', 'mask_below_seafloor',
]


# eq=False: a dataclass __eq__ over ndarray fields raises; compare by identity.
@carrier(eq=False)
class Bathymetry(_RangeProfile):
    """Seafloor depth (m, positive down) as a function of range (m).

    A 1-D grid library carrier mirroring :class:`SoundSpeedProfile`: select a
    depth with :meth:`at` (nearest), :meth:`isel` (positional) or :meth:`eval`
    (interpolated). Because bathymetry has a single axis (range), those
    selectors collapse it and return the **depth value(s)** directly (a scalar
    for a scalar range, an array for an array of ranges).

    Attributes
    ----------
    ranges : ndarray, shape (N,)
        Range axis in metres, monotonically increasing (``[0.0]`` for a flat
        bottom).
    depths : ndarray, shape (N,)
        Seafloor depth in metres at each range (positive down, > 0).
    """

    ranges: np.ndarray
    depths: np.ndarray
    data_sources: tuple = ()

    _VALUE_FIELD = 'depths'
    _XARRAY_FIELDS = {'depth': 'depths', 'range': 'ranges'}
    _VALUE_LABEL = 'depth'
    _AXIS_DOWN = True

    def __post_init__(self):
        self.data_sources = coerce_data_sources(self.data_sources, "Bathymetry")
        self._init_range_profile()

    def _validate_values(self) -> None:
        require_positive(self.depths, "Bathymetry depths", hint="metres, down")

    def collapse_range(self, method: str = 'max') -> float:
        """The one depth (m) that stands for this seafloor: the reduction a
        model that cannot take a sloping bottom runs on.

        ``method`` is one of :data:`~uacpy.core.collapse.DEPTH_COLLAPSE_METHODS`:

        - ``'max'``: the deepest node (default, the project-wide
          ``collapse={'bathymetry': 'max'}``);
        - ``'median'``: the median over the nodes;
        - ``'mean'``: the mean over the nodes. Every node weighs the same
          whatever its spacing, so a profile sampled densely on a shelf leans
          toward the shelf; for a range average use
          ``np.trapezoid(depths, ranges) / span``;
        - ``'min'``: the shallowest node;
        - ``'initial'``: the depth at range 0.

        >>> Bathymetry(ranges=[0, 5000, 10000],
        ...            depths=[100, 200, 300]).collapse_range('median')
        200.0

        Parameters
        ----------
        method : str, optional
            One of :data:`~uacpy.core.collapse.DEPTH_COLLAPSE_METHODS` (see above).
            Default ``'max'``.
        """
        depths = self.depths
        if method == 'median':
            return float(np.median(depths))
        if method == 'mean':
            return float(np.mean(depths))
        if method == 'min':
            return float(np.min(depths))
        if method == 'max':
            return float(np.max(depths))
        if method == 'initial':
            return float(depths[0])
        raise ConfigurationError(
            f"Bathymetry.collapse_range: unknown method={method!r}; "
            f"valid: {_method_list(DEPTH_COLLAPSE_METHODS)}.")

    # ── constructors ────────────────────────────────────────────────────────
    @classmethod
    def coerce(cls, value) -> 'Bathymetry':
        """Coerce ``Bathymetry`` / scalar depth / ``(N, 2)`` ``(range, depth)``
        pairs into a :class:`Bathymetry`.

        ``None`` is rejected (bathymetry is required; there is no default
        seafloor).

        Parameters
        ----------
        value : Bathymetry, float or array_like
            The ``bathymetry=`` value (see above).
        """
        if isinstance(value, Bathymetry):
            return value
        # A scalar (a 0-d array included) is a flat seafloor; a bool is
        # refused as one — the shared guard says why.
        depth = scalar_or_none(value, lambda v: (
            f"Bathymetry: {v!r} is a bool, not a depth — as a scalar it "
            f"would mean a {float(v):g} m deep seafloor."))
        if depth is not None:
            return cls(ranges=np.array([0.0]), depths=np.array([depth]))
        # ``_scalar_or_none`` returned for every numeric scalar, so a 0-d
        # value here (None, a string, bytes, a 0-d string array) is not a
        # depth: ``arr`` stays None for it, as for a non-numeric sequence.
        try:
            arr = np.asarray(value, dtype=float) if np.ndim(value) else None
        except (TypeError, ValueError):
            arr = None
        if arr is None:
            raise ConfigurationError(
                f"Bathymetry: must be a positive scalar depth or shape (N, 2) "
                f"as [(range, depth), ...]; got non-numeric {value!r}.")
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ConfigurationError(
                f"Bathymetry: must be a positive scalar or shape (N, 2) as "
                f"[(range, depth), ...]; got shape {arr.shape} "
                f"(example: [(0, 100), (5000, 200)]).")
        return cls(ranges=arr[:, 0], depths=arr[:, 1])

    # ── derived ─────────────────────────────────────────────────────────────
    @property
    def depth(self) -> float:
        """Maximum seafloor depth (m) — the deepest point of the profile."""
        return float(np.max(self.depths))


def mask_below_seafloor(data, depths, ranges, bathymetry, *,
                        paired: bool = False) -> np.ndarray:
    """A copy of ``data`` with every sample below the local seafloor set to
    NaN.

    Parameters
    ----------
    data : array-like
        Depth on axis 0 and range on axis 1; any trailing axes (frequency,
        time) take the mask of their cell. With ``paired=True``, axis 0 is
        the receiver pair and runs along ``depths`` and ``ranges`` together.
    depths, ranges : array-like
        Receiver depths and ranges (m) of ``data``'s first axes.
    bathymetry : Bathymetry, float or array-like
        The seafloor: a :class:`Bathymetry`, a flat depth, or ``(N, 2)``
        ``(range, depth)`` pairs (checked by :meth:`Bathymetry.coerce`).
        Read with :meth:`Bathymetry.eval` — linear, constant past the ends.
    paired : bool, keyword-only
        ``True`` when sample ``i`` is the receiver ``(depths[i], ranges[i])``
        (a Bellhop irregular grid) rather than a ``depths x ranges`` grid.

    Returns
    -------
    ndarray
        The masked copy: a sample deeper than the seafloor at its range is
        NaN, one on the seafloor is kept. An inexact payload keeps its dtype
        (a float32 one stays float32); an integer one, which cannot hold NaN,
        becomes float64.

    Notes
    -----
    The cut is uacpy's output convention, not an engine limit: a PE marches
    through the sediment to its ``zmax`` and computes a field there that
    agrees with the models which return it. Measured on a 100 m guide over a
    50 m sediment layer at 100 Hz, receiver at 2 km, with the mask
    neutralised: |p| at 120 m is 5.4e-4 on mpiramS and 6.8e-4 on ramgeo,
    against Kraken 6.3e-4 and Scooter 6.2e-4. What the sub-bottom column
    does *not* carry uniformly is physical meaning: RAM leaves only a few
    wavelengths of real seabed before its artificial absorbing layer, and
    where that boundary falls depends on the frequency and the grid.
    Cutting at the seafloor is the depth that is the same on every engine
    and every deck.
    """
    arr = np.asarray(data)
    dtype = arr.dtype if np.issubdtype(arr.dtype, np.inexact) else np.float64
    out = arr.astype(dtype, copy=True)
    depths = np.atleast_1d(np.asarray(depths, dtype=float))
    seafloor = np.atleast_1d(Bathymetry.coerce(bathymetry).eval(
        range=np.asarray(ranges, dtype=float)))
    if paired:
        if depths.shape != seafloor.shape or out.shape[0] != depths.size:
            raise ConfigurationError(
                f"mask_below_seafloor(paired=True): {depths.size} depths, "
                f"{seafloor.size} ranges and {out.shape[0]} samples on axis "
                f"0; a paired grid has one of each per receiver.")
        below = depths > seafloor
    else:
        if out.shape[:2] != (depths.size, seafloor.size):
            raise ConfigurationError(
                f"mask_below_seafloor: data's first axes {out.shape[:2]} are "
                f"not (len(depths), len(ranges)) = "
                f"({depths.size}, {seafloor.size}).")
        below = depths[:, None] > seafloor[None, :]
    out[below, ...] = np.nan
    return out
