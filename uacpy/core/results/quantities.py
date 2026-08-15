"""The physical quantities a :class:`~uacpy.core.results.Field` can carry.

A Field is described on three independent axes — ``kind`` (what it is),
``unit`` (what it is measured in) and the data's ``dtype`` (how it is stored).
This module is the registry behind the first two: it names every quantity the
package produces, the units each may be expressed in, and how each pairing is
labelled.

The registry exists because those facts were previously re-derived by each
consumer, and every consumer got a different one wrong: the axis label was a
hardcoded string and the colour limits a test on the render mode. Registering
a quantity here is what makes a new Field kind a *data* edit rather than a
hunt through the plotters.

Deliberately **not** modelled: unit conversion, dimensional analysis, or a
per-unit "which way is louder" flag. Transmission loss is the only inverted
quantity in underwater acoustics — it is a *loss*, so less of it is louder —
and one documented special case (see :meth:`Field.max`) is cheaper than a
field that would read ``+1`` in every row but one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from uacpy.core.exceptions import ConfigurationError


@dataclass(frozen=True)
class Quantity:
    """One physical quantity: the units it may carry, and how it renders.

    ``units`` maps a unit symbol to the axis / colorbar label for that
    pairing. A quantity with a single entry has no representational choice —
    reverberation is always a level in dB — while ``pressure`` carries two,
    because ``-20·log10|p|`` is the same field written differently.

    Labels are domain vocabulary, so they live here. The **colormap** does
    not: that is a rendering choice and belongs to
    ``uacpy.visualization.style``, which core must not depend on.
    """

    name: str
    units: Mapping[str, str]

    @property
    def default_unit(self) -> str:
        return next(iter(self.units))


#: Every quantity a Field may carry. The first unit listed is the default.
QUANTITIES: Mapping[str, Quantity] = {
    q.name: q for q in (
        # Pressure is the only quantity with two units: linear Pa (complex
        # narrowband, H(f), or a real time trace) and dB, which *is*
        # transmission loss. They are one quantity, which is why a RAM TL
        # field and a Kraken complex field share a colour scale.
        Quantity('pressure', {'Pa': 'Pressure (Pa)', 'dB': 'TL (dB)'}),
        Quantity('reverberation', {'dB': 'Reverberation level (dB)'}),
        Quantity('signal_excess', {'dB': 'Signal excess (dB)'}),
        Quantity('probability_of_detection', {'1': 'Probability of detection'}),
    )
}


def quantity(kind: str) -> Quantity:
    """Look up ``kind``, or raise naming this module.

    Unknown kinds raise rather than falling back to a default: a silent
    default is how a mislabelled quantity reaches a plot looking correct.
    """
    try:
        return QUANTITIES[kind]
    except KeyError:
        raise ConfigurationError(
            f"unknown Field kind {kind!r}; register it in "
            f"uacpy/core/results/quantities.py. Known kinds: "
            f"{', '.join(sorted(QUANTITIES))}"
        ) from None


def label(kind: str, unit: str) -> str:
    """Axis / colorbar label for a ``(kind, unit)`` pairing."""
    q = quantity(kind)
    try:
        return q.units[unit]
    except KeyError:
        raise ConfigurationError(
            f"{kind!r} is not measured in {unit!r}; "
            f"valid units: {', '.join(q.units)}"
        ) from None
