"""Altimetry shape carrier: sea-surface height as a function of range.

The top-surface analogue of :class:`uacpy.core.bathymetry.Bathymetry` — a 1-D
profile (``height`` vs ``range``). Heights are positive **up** (z = 0 at mean
sea level), so a crest is positive and a trough negative. Re-exported from
:mod:`uacpy.core.environment` for stable import paths.
"""

import numpy as np
from dataclasses import dataclass

from uacpy.core.exceptions import ConfigurationError
from uacpy.core._grid import _RangeProfile
from uacpy.core._carrier_validate import _require_finite


# eq=False: a dataclass __eq__ over ndarray fields raises; compare by identity.
@dataclass(eq=False, repr=False)
class Altimetry(_RangeProfile):
    """Sea-surface height (m, positive up) as a function of range (m).

    A 1-D grid library carrier mirroring :class:`Bathymetry`: select a height
    with :meth:`at` (nearest), :meth:`isel` (positional) or :meth:`eval`
    (interpolated). The single range axis is collapsed, so those return the
    **height value(s)** directly (a scalar for a scalar range, an array for an
    array of ranges).

    Attributes
    ----------
    ranges : ndarray, shape (N,)
        Range axis in metres, monotonically increasing.
    heights : ndarray, shape (N,)
        Surface height in metres at each range (positive up; any sign).
    """

    ranges: np.ndarray
    heights: np.ndarray

    _VALUE_FIELD = 'heights'
    _VALUE_LABEL = 'height'

    def __post_init__(self):
        self._init_range_profile()

    def _validate_values(self) -> None:
        _require_finite(self.heights, "Altimetry heights",
                        hint="metres, positive up")

    # ── constructors ────────────────────────────────────────────────────────
    @classmethod
    def coerce(cls, value):
        """Coerce ``None`` / ``Altimetry`` / ``(N, 2)`` ``(range, height)``
        pairs into an :class:`Altimetry` (``None`` passes through — a flat
        z = 0 surface)."""
        if value is None or isinstance(value, Altimetry):
            return value
        try:
            arr = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            raise ConfigurationError(
                f"Altimetry: must be shape (N, 2) as [(range, height_m), ...]; "
                f"got non-numeric {value!r}.")
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ConfigurationError(
                f"Altimetry: must have shape (N, 2) as [(range, height_m), "
                f"...]; got shape {arr.shape}.")
        return cls(ranges=arr[:, 0], heights=arr[:, 1])
