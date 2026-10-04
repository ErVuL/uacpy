"""What a reader or writer checks before it trusts its input.

Whether a grid is uniform is :func:`uacpy.core._validate.equally_spaced`;
``reject_unknown_kwargs`` refuses a keyword the writer does not take
rather than dropping it silently; ``_collapsed_pair_index`` locates the
collapsed entry in a boundary or reflection table.
"""

import numpy as np

from uacpy.core.exceptions import ConfigurationError


def reject_unknown_kwargs(writer: str, kwargs: dict, known) -> None:
    """Raise on a writer knob no block of this deck reads.

    Every deck writer that accepts ``**kwargs`` calls this first, so a
    misspelled option fails loudly instead of being silently dropped and
    leaving the deck subtly different from what the caller asked for.
    """
    unknown = sorted(set(kwargs) - set(known))
    if unknown:
        raise ConfigurationError(
            f"{writer}: parameter(s) {unknown} are not read by this deck.",
            remediation=f"Drop them, or check they belong to this writer's "
                        f"program; it reads {sorted(known)}.",
        )


def _collapsed_pair_index(written, *, raw=None, min_step=None):
    """Index ``i`` of the adjacent pair ``(i, i+1)`` a deck column loses,
    or ``None`` when every pair survives.

    ``written`` is the axis as the file will hold it — formatted tokens
    (accepted directly) or values rounded to the column's resolution. One
    rule per way a writer loses an axis, selected by the keyword:

    * default — the written column must be strictly increasing; when it is
      not (NaN pairs included, since NaN fails every comparison), the index
      of the smallest step comes back.
    * ``raw=`` — only a pair whose ``raw`` step is positive and whose
      written step is zero counts: a value the caller repeated on purpose
      is not a collision. The first such pair comes back.
    * ``min_step=`` — steps of ``written`` below ``min_step`` fail even
      where the tokens stay distinct; the index of the smallest step comes
      back. Callers pass the unrounded axis here: the rule bounds spacing,
      not token identity.

    Callers raise their own :class:`ConfigurationError` naming the pair,
    the column's resolution and the engine consequence — the six deck
    writers that lose axes this way share the detection, not the message.
    """
    written = np.asarray(written, dtype=float)
    if written.size <= 1:
        return None
    steps = np.diff(written)
    if min_step is not None:
        if steps.min() < min_step:
            return int(np.argmin(steps))
        return None
    if raw is not None:
        collision = ((np.diff(np.asarray(raw, dtype=float)) > 0.0)
                     & (steps == 0.0))
        if collision.any():
            return int(np.argmax(collision))
        return None
    if not np.all(steps > 0):
        return int(np.argmin(steps))
    return None
