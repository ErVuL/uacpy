"""The sound speeds a :class:`~uacpy.core.results.Field` carries for the
time-series synthesis: where its window opens, and how much travel time a
range span spans."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from uacpy.core._records import FrozenRecord
from uacpy.core.exceptions import ConfigurationError

__all__ = ['SoundSpeeds']


@dataclass(frozen=True, eq=False)
class SoundSpeeds(FrozenRecord):
    """Named sound speeds (m/s) of the medium a Field was computed in, each
    ``None`` where its producer states none.

    :meth:`Field.to_time_trace` and :meth:`Field.synthesize_time_series`
    anchor the record on the fastest physical speed (``water_max``, else
    ``waveguide_max``; ``surface`` competes with it), so the earliest
    arrival falls inside the window, and time the receiver range span with
    ``surface``, else ``waveguide_min``.

    Attributes
    ----------
    surface : float, optional
        The water speed at the sea surface of the first profile (Bellhop).
    water_min : float, optional
        The slowest water-column speed. mpiramS reads it from its own
        profile and anchors its time window on it (``peramx.f90:295``,
        ``cmin=minval(cw)``; the psif.dat header record at ``:469``); it is
        not the waveguide's minimum, which spans the sediment too.
    water_max : float, optional
        The fastest water-column speed (Bellhop): Bellhop traces rays in
        the water and has no head wave through the seabed, so the
        waveguide's fastest speed would open the window early for nothing.
    waveguide_min, waveguide_max : float, optional
        The slowest and fastest compressional speeds of the waveguide the
        producing run resolved, water column and every bottom layer
        (:attr:`RunSettings.waveguide <uacpy.core.run_settings.RunSettings>`).
        A Field filled from its run settings states them here, so they
        survive a file that cannot hold the settings record.
    """

    surface: Optional[float] = None
    water_min: Optional[float] = None
    water_max: Optional[float] = None
    waveguide_min: Optional[float] = None
    waveguide_max: Optional[float] = None

    _REPR_UNITS = {name: 'm/s' for name in ('surface', 'water_min', 'water_max',
                                            'waveguide_min', 'waveguide_max')}

    def __post_init__(self):
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            speed = float(value)
            if not np.isfinite(speed) or speed <= 0.0:
                raise ConfigurationError(
                    f"SoundSpeeds.{f.name}={value!r}: a sound speed is a "
                    f"finite positive number of m/s.")
            object.__setattr__(self, f.name, speed)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SoundSpeeds':
        return cls(**d)

    def with_waveguide(self, waveguide) -> 'SoundSpeeds':
        """This record with ``waveguide_min`` / ``waveguide_max`` taken from
        ``waveguide`` (a record with ``c_min`` / ``c_max``) where it states
        none."""
        if waveguide is None:
            return self
        return dataclasses.replace(
            self,
            waveguide_min=(self.waveguide_min if self.waveguide_min is not None
                           else waveguide.c_min),
            waveguide_max=(self.waveguide_max if self.waveguide_max is not None
                           else waveguide.c_max))
