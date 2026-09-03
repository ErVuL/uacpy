"""The physical quantities a :class:`~uacpy.core.results.Field` can carry.

A Field is described on three independent axes — ``kind`` (what it is),
``unit`` (what it is measured in) and the data's ``dtype`` (how it is stored).
This module is the registry behind the first two: it names every quantity the
package produces, the units each may be expressed in, and how each pairing is
labelled.

The registry exists so those facts have one home. Re-derived per consumer they
diverge, and each consumer gets a different one wrong: the axis label becomes a
hardcoded string in one plotter, the colour limits a test on the render mode in
the next, and nothing reconciles them. Registering
a quantity here is what makes a new Field kind a *data* edit rather than a
hunt through the plotters.

Deliberately **not** modelled: unit conversion, dimensional analysis, or a
per-unit "which way is louder" flag. Two of the four quantities are inverted —
transmission loss (``pressure`` in dB) and OASS reverberation, both *losses*,
so less of either is louder — and the two documented cases in
:meth:`Field.max` are cheaper than a field that would read ``+1`` in every row
but two.

The reverberation direction is not a convention uacpy chose. OASES writes
``-10·log10 E[|p_scat|²]`` in ``REVINT`` (``oassun26.f:853-858``): ``CVMAGS``
squares an accumulator that is already an intensity, ``VALG10`` takes log10,
and ``VSMUL(-5E0)`` scales it. The leading minus is what makes it a loss, and
uacpy's reader applies no sign change, so a larger stored number is a *weaker*
scattered field. Reverberation level is recovered as ``RL = SL - this``.

``REVRAN``'s block at ``oassun26.f:633-638`` is byte-identical arithmetic on a
cross-range covariance and is **not** the source of these numbers — it feeds
the contour branch, not the ``.plt`` uacpy reads. Cite ``REVINT``.
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
    reverberation is always a loss in dB — while ``pressure`` carries two,
    because ``-20·log10|p|`` is the same field written differently.

    Labels are domain vocabulary, so they live here. The **colormap** does
    not: that is a rendering choice and belongs to
    ``uacpy.visualization.style``, which core must not depend on.
    """

    name: str
    units: Mapping[str, str]


#: Every quantity a Field may carry. The first unit listed is the default.
QUANTITIES: Mapping[str, Quantity] = {
    q.name: q for q in (
        # Pressure is the only quantity with two units: linear Pa (complex
        # narrowband, H(f), or a real time trace) and dB, which *is*
        # transmission loss. They are one quantity, which is why a RAM TL
        # field and a Kraken complex field share a colour scale.
        Quantity('pressure', {'Pa': 'Pressure (Pa)', 'dB': 'TL (dB)'}),
        # A loss, not a level: OASES writes -10·log10 E[|p_scat|²], so the
        # axis rises as the scattered field weakens. Labelled for what the
        # numbers are, since the plot is where the direction is read off.
        Quantity('reverberation',
                 {'dB': 'Reverberation loss (dB re unit source)'}),
        Quantity('signal_excess', {'dB': 'Signal excess (dB)'}),
        # A signed RESIDUAL between two dB fields, not a level and not a loss:
        # zero means the two agree, and the sign says which way they differ.
        # It exists so a difference stops inheriting 'pressure', which would
        # caption it "TL (dB)" and invert a 1-D cut through it — both wrong
        # for a quantity whose zero is the meaningful value.
        Quantity('difference', {'dB': 'Difference (dB)'}),
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
