"""Bathymetry shape carrier: seafloor depth as a function of range.

A 1-D profile (``depth`` vs ``range``), the seafloor analogue of
:class:`uacpy.core.ssp.SoundSpeedProfile`. Re-exported from
:mod:`uacpy.core.environment` for stable import paths.
"""

import numpy as np
from dataclasses import dataclass

from uacpy.core.exceptions import ConfigurationError
from uacpy.core._grid import _RangeProfile
from uacpy.core._carrier_validate import _require_positive, _coerce_data_sources


# eq=False: a dataclass __eq__ over ndarray fields raises; compare by identity.
@dataclass(eq=False, repr=False)
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
    _VALUE_LABEL = 'depth'
    _AXIS_DOWN = True

    def __post_init__(self):
        self.data_sources = _coerce_data_sources(self.data_sources, "Bathymetry")
        self._init_range_profile()

    def _validate_values(self) -> None:
        _require_positive(self.depths, "Bathymetry depths", hint="metres, down")

    # ── constructors ────────────────────────────────────────────────────────
    @classmethod
    def coerce(cls, value) -> 'Bathymetry':
        """Coerce ``Bathymetry`` / scalar depth / ``(N, 2)`` ``(range, depth)``
        pairs into a :class:`Bathymetry`.

        ``None`` is rejected (bathymetry is required; there is no default
        seafloor).
        """
        if isinstance(value, Bathymetry):
            return value
        if isinstance(value, (bool, np.bool_)):
            raise ConfigurationError(
                f"Bathymetry: {value!r} is a bool, not a depth — as a scalar "
                f"it would mean a {float(value):g} m deep seafloor."
            )
        try:
            if np.ndim(value) == 0:
                return cls(ranges=np.array([0.0]),
                           depths=np.array([float(value)]))
            arr = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
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
