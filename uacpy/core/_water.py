"""The record a carrier keeps a water property's ``(depth, value)`` pairs
in.

A package-internal module: users give a water profile as pairs
(:func:`uacpy.core._validate.water_property`); :class:`WaterProfile` is what
:func:`uacpy.core._validate.water_pairs` builds from them and what
:class:`~uacpy.core.absorption.FrancoisGarrison` stores, saves and hands
back.
"""

from dataclasses import dataclass, field

import numpy as np

from uacpy.core._export import ExportRecord
from uacpy.core.exceptions import ConfigurationError

__all__ = ['WaterProfile']


@dataclass(frozen=True, eq=False)
class WaterProfile(ExportRecord):
    """The record a carrier stores a water property's ``(depth, value)``
    pairs as: ``values`` at ``depths`` (m, positive down), read at the
    depths evaluated by linear interpolation, the end values held beyond the
    first and last depth.

    Users give a profile as pairs, an ``(M, 2)`` array
    (:func:`~uacpy.core._validate.water_property`);
    :func:`~uacpy.core._validate.water_pairs` checks them and builds this
    record. :class:`~uacpy.core.absorption.FrancoisGarrison` hands it back
    (``law.temperature``), and every water-property input takes it again,
    so a law's water can be passed to a formula as it is. It saves and
    loads with the law that holds it.

    ``profile.values`` is the array of property values, not a method: on
    this record the field takes the name of the ``values()`` method every
    other exported object carries, so ``profile.values()`` raises
    ``TypeError`` ('numpy.ndarray' object is not callable). Read
    ``profile.values`` and ``profile.depths`` directly.

    Parameters
    ----------
    depths : array_like
        Depths (m) of the measurements, finite and strictly increasing.
    values : array_like
        The property at each depth (°C, PSU, pH, ...).

    Examples
    --------
    >>> T = WaterProfile(depths=[0.0, 60.0, 200.0], values=[22.0, 12.0, 10.0])
    >>> T([0.0, 30.0, 100.0, 500.0])
    array([22.        , 17.        , 11.42857143, 10.        ])
    """

    depths: np.ndarray
    # An explicit field() keeps the inherited protocol method values() from
    # becoming this field's default, so the field is required. The field
    # shadows the export protocol's values() on purpose: the record is the
    # carriers' storage of a profile, read as ``profile.values``.
    values: np.ndarray = field()

    _ARRAY_FIELDS = ('depths', 'values')
    _TABLE_FIELDS = ('depths', 'values')

    def __post_init__(self):
        d = np.atleast_1d(np.asarray(self.depths, dtype=float))
        v = np.atleast_1d(np.asarray(self.values, dtype=float))
        if d.ndim != 1 or d.shape != v.shape:
            raise ConfigurationError(
                f"WaterProfile: depths and values are two 1-D arrays of "
                f"equal length; got shapes {d.shape} and {v.shape}.")
        if not (np.all(np.isfinite(d)) and np.all(np.isfinite(v))):
            raise ConfigurationError(
                "WaterProfile: depths and values must be finite.")
        if np.any(np.diff(d) <= 0):
            raise ConfigurationError(
                f"WaterProfile: depths must strictly increase (m, positive "
                f"down); got {d.tolist()}.")
        object.__setattr__(self, 'depths', d)
        object.__setattr__(self, 'values', v)
        super().__post_init__()

    def __call__(self, depth):
        """The property at ``depth`` (m), interpolated linearly, the end
        values held."""
        return np.interp(np.asarray(depth, dtype=float), self.depths,
                         self.values)

    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        return (np.array_equal(self.depths, other.depths)
                and np.array_equal(self.values, other.values))

    __hash__ = None
