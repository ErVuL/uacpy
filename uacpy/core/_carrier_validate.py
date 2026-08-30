"""Shared input validators for the core environment carriers
(:mod:`uacpy.core.bottom`, :mod:`uacpy.core.ssp`, :mod:`uacpy.core.environment`).
"""

import numpy as np

from uacpy.core.exceptions import ConfigurationError


def _validate_acoustic_type(value, label: str) -> None:
    """Reject unrecognized ``acoustic_type`` strings up front, so a typo
    like ``'halfspace'`` (vs. ``'half-space'``) fails at construction
    instead of producing a wrong Acoustics-Toolbox bottom-type code
    deep inside a writer.
    """
    from uacpy.core.constants import BoundaryType
    try:
        BoundaryType.from_string(value)
    except (ConfigurationError, ValueError, KeyError, AttributeError) as exc:
        valid = sorted({bt.value for bt in BoundaryType})
        raise ConfigurationError(
            f"{label}: acoustic_type={value!r} is not recognized. "
            f"Valid values (plus the aliases handled by "
            f"BoundaryType.from_string): {valid}"
        ) from exc


def _hint(hint: str) -> str:
    """Render an optional unit/context note appended after the rule clause.

    Kept separate from ``label`` so ``label`` ends immediately before
    ``"must be"`` — callers (and tests) can match ``"<noun> must be"`` as a
    contiguous phrase while the unit context still survives in the message.
    """
    return f" ({hint})" if hint else ""


def _reject_complex(values, label: str) -> None:
    """Raise ``ConfigurationError`` if any element is complex.

    Every core carrier calls this immediately **before** its
    ``np.array(..., dtype=float)`` cast, because that cast destroys a complex
    input in two different ways depending on the container it arrived in: an
    ndarray keeps only the real part and announces it with a
    ``numpy.exceptions.ComplexWarning``, which is a ``UserWarning`` subclass
    and so is hidden by the suite-wide ``ignore::UserWarning`` filter, while a
    scalar or a list dies in a bare ``TypeError`` out of ``float()`` that names
    no carrier and no field. Rejecting first gives all four carriers and both
    containers one typed verdict.

    An imaginary part reaching a carrier is a computation artefact, not a
    coordinate: depths, ranges, frequencies and sound speeds are real
    quantities, and every consumer downstream reads them as float64.
    """
    arr = np.asarray(values)
    if not np.iscomplexobj(arr):
        return
    flat = arr.ravel()
    if flat.size == 0:
        detail = f"an empty array of dtype {arr.dtype}"
    else:
        # The first element carrying an imaginary part, matching the flat
        # index the finiteness and sign guards above report. A complex dtype
        # whose imaginary parts are all zero is refused just the same — the
        # cast warns on the dtype, not on the values — and there the leading
        # element is what the message shows.
        offending = np.flatnonzero(flat.imag != 0)
        bad = int(offending[0]) if offending.size else 0
        detail = f"{flat[bad]} at flat index {bad} of {flat.size} value(s)"
    raise ConfigurationError(
        f"{label} must be real numbers; got complex value(s): {detail}.",
        remediation=f"Pass real {label}, taking .real explicitly if the "
                    f"imaginary part is a known computation artefact.",
    )


def _require_finite(values, label: str, *, hint: str = "") -> None:
    """Raise ``ConfigurationError`` if any element is NaN or inf.

    Accepts a scalar or any array-like. Shared by the carriers so the
    "must be finite" guard reads identically everywhere instead of being
    re-inlined per attribute.
    """
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        flat = arr.ravel()
        bad = int(np.flatnonzero(~np.isfinite(flat))[0])
        raise ConfigurationError(
            f"{label} must be finite (no NaN/inf){_hint(hint)}; "
            f"got {flat[bad]} at flat index {bad} of {flat.size} value(s)")


def _require_positive(values, label: str, *, hint: str = "") -> None:
    """Raise ``ConfigurationError`` unless every element is finite and ``> 0``.

    Finiteness is checked first (NaN/inf pass every plain ``<= 0`` test) and
    reported separately, so the sign message stays the contiguous phrase
    ``"<label> must be positive"`` that callers relied on.

    Why three of these exist
    -----------------------
    The canonical note for the whole package, so the split is not re-derived
    each audit. Three "reject a non-positive scalar" guards are shipped on
    purpose: this one,
    :func:`uacpy.acoustic_signal._signal_validate.require_positive_finite_scalar`
    and ``uacpy.sonar.target_strength._require_positive``. They were compared
    on *behaviour*, not shape, and reach the same accept/reject verdict on
    every scalar — ``0``, negatives, ``nan``, ``inf``, ``-inf``, numpy
    scalars, 0-d arrays, ``Decimal``, ``Fraction``, ``bool``, numeric strings,
    and non-numeric objects. What differs is what each names, and each is
    right for its layer:

    * this one takes an **array** and reports the offending flat index and the
      element count, because a carrier validates a whole depth or range axis
      and "which sample" is the question a user has;
    * the ``acoustic_signal`` one takes a single argument and reports the
      calling estimator, the parameter and its unit, because there the
      question is which argument of which function;
    * the ``sonar`` one is that same contract behind a fixed
      ``"target strength: "`` prefix, kept local so ``sonar`` — which today
      imports only from ``core`` — does not reach into a private module of a
      sibling package for it.

    One measured difference is *not* wording and is left standing on purpose:
    for a non-numeric argument the two scalar guards raise
    ``ConfigurationError`` naming the parameter, while this one lets the
    ``TypeError`` / ``ValueError`` out of ``np.asarray(..., dtype=float)``
    through. A failure there is a dtype or ragged-shape problem rather than a
    positivity verdict, and a carrier test pins that ``ValueError`` for a
    non-numeric ``SedimentLayer`` field, so typing it is a breaking change and
    a decision of its own.

    Consolidating them would cost the first two of those messages, so they
    stay. What the split costs instead is drift, and it has been paid twice: a
    fourth copy in ``acoustic_signal/waveforms.py`` was folded into the second
    and found to be silently accepting ``inf``, and the ``sonar`` one was
    later measured doing the same, against its own message. The agreement
    above is therefore pinned by a test — see
    ``TestThePositiveScalarGuardsAgreeAcrossLayers`` in ``test_sonar.py`` —
    rather than left to the next reader to re-measure.
    """
    arr = np.asarray(values, dtype=float)
    _require_finite(arr, label, hint=hint)
    if np.any(arr <= 0):
        flat = arr.ravel()
        bad = int(np.flatnonzero(flat <= 0)[0])
        raise ConfigurationError(
            f"{label} must be positive, > 0{_hint(hint)}; "
            f"got {flat[bad]} at flat index {bad} of {flat.size} value(s)")


def _require_non_negative(values, label: str, *, hint: str = "") -> None:
    """Raise ``ConfigurationError`` unless every element is finite and ``>= 0``.

    Finiteness is reported separately so the sign message stays the contiguous
    phrase ``"<label> must be non-negative"``.
    """
    arr = np.asarray(values, dtype=float)
    _require_finite(arr, label, hint=hint)
    if np.any(arr < 0):
        flat = arr.ravel()
        bad = int(np.flatnonzero(flat < 0)[0])
        raise ConfigurationError(
            f"{label} must be non-negative, >= 0{_hint(hint)}; "
            f"got {flat[bad]} at flat index {bad} of {flat.size} value(s)")


def _require_attenuation_in_range(value, label: str) -> None:
    """Raise ``ConfigurationError`` for an attenuation past the AT solvers' own
    ceiling — see :data:`uacpy.core.constants.MAX_ATTENUATION_DB_PER_WAVELENGTH`
    for the derivation. Above it Kraken, Scooter, OAST and Bellhop's Fortran
    build all abort in ``CRCI``, while ``bellhopcxx``/``bellhopcuda`` return a
    field with *less* loss than a low attenuation gives.
    """
    from uacpy.core.constants import MAX_ATTENUATION_DB_PER_WAVELENGTH
    if value is None:
        return
    alpha = float(value)
    if alpha > MAX_ATTENUATION_DB_PER_WAVELENGTH:
        raise ConfigurationError(
            f"{label} = {alpha:g} dB/wavelength exceeds "
            f"{MAX_ATTENUATION_DB_PER_WAVELENGTH:.4f}, above which the imaginary "
            f"part of the complex sound speed exceeds the real part and every AT "
            f"solver aborts in misc/AttenMod.f90's CRCI (:116). The bound is "
            f"independent of frequency and sound speed because uacpy writes "
            f"AttenUnit 'W' (dB/wavelength).",
            remediation="Real seabeds are well under 2 dB/wavelength. To model a "
                        "strongly absorbing bottom use a reflection-coefficient "
                        "table (acoustic_type='file') instead.",
        )


def _require_strictly_increasing(values: np.ndarray, label: str, *,
                                 min_step: float = 0.0,
                                 unit: str = 'm') -> None:
    """Raise ``ConfigurationError`` if ``values`` is not strictly
    monotonically increasing. Used to guard every range / depth axis that
    feeds into ``np.interp``, which silently produces garbage on unsorted
    ``xp``.

    ``min_step`` additionally requires neighbours to be separated by more than
    the resolution the solver decks print the axis at, since two samples closer
    than that collapse to a single token in the file. Pass
    ``DECK_RANGE_RESOLUTION_M`` for a range axis or ``DECK_DEPTH_RESOLUTION_M``
    for a depth axis (see those constants for the readers this protects), with
    ``unit`` naming what the step is measured in — the axes guarded this way
    are not all spatial (``SBP_ANGLE_RESOLUTION_DEG`` guards a degree axis).

    A zero-length axis is refused here as well. Monotonicity is vacuous on it,
    but every consumer reads the axis positionally — a deck writer takes
    element 0, ``np.interp`` refuses an empty ``xp``, and ``min``/``max`` over
    it is a zero-size reduction — so an empty axis that passes construction
    surfaces much later as a bare ``IndexError`` or ``ValueError`` from inside
    a writer, naming no input the caller supplied. A one-sample axis is a
    legitimate carrier (a single receiver depth, one profile range) and is
    accepted. This is the only ``_require_*`` guard here that rejects an empty
    array: the value predicates above are also applied to ``Field.coords``,
    where an axis sliced to nothing is a supported state (``Field.max`` reports
    it in those terms).
    """
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ConfigurationError(
            f"{label} must contain at least one value; got an empty axis. "
            f"The axis is read positionally downstream — a deck writer takes "
            f"element 0 and the interpolators refuse an empty sample vector — "
            f"so an empty one fails inside a writer instead of here.",
            remediation=(
                f"Give {label} at least one sample, or pass None where the "
                f"carrier makes the axis optional (a range axis of None is "
                f"the range-independent form)."
            ),
        )
    if arr.size == 1:
        return
    diffs = np.diff(arr)
    if not np.all(diffs > 0):
        bad = int(np.argmin(diffs))
        raise ConfigurationError(
            f"{label} must be strictly increasing; "
            f"got {arr[bad]} >= {arr[bad + 1]} at index {bad + 1} "
            f"(axis length {arr.size})"
        )
    if min_step > 0.0 and not np.all(diffs > min_step):
        bad = int(np.argmin(diffs))
        raise ConfigurationError(
            f"{label} must increase by more than {min_step:g} {unit}; "
            f"got {arr[bad]} and {arr[bad + 1]} at index {bad + 1} "
            f"({float(diffs[bad]):g} {unit} apart). "
            "The solver decks print this axis at that resolution, so the "
            "two samples collapse to one value in the file."
        )


def _coerce_data_sources(value, label: str) -> tuple:
    """Validate and freeze a carrier's ``data_sources`` into a tuple of
    provenance records, enforcing the harmonised invariant that every element
    is a :class:`~uacpy.data.sources.DataProvenance` (carries a ``.source``).

    Duck-typed so ``core`` keeps no import dependency on ``data``: a record is
    accepted iff it exposes ``.source`` with an ``.id``. A bare ``DataSource``
    (no ``.source``) or any other object is rejected with a typed error rather
    than leaking downstream to crash ``env.data_sources`` aggregation or
    ``citations()`` on ``r.source.id``. ``None`` means no provenance and
    coerces to ``()``.
    """
    if value is None:
        return ()
    records = tuple(value)
    for r in records:
        if not (hasattr(r, 'source') and hasattr(getattr(r, 'source'), 'id')):
            raise ConfigurationError(
                f"{label}: data_sources elements must be DataProvenance "
                f"records (each carrying a .source); got {type(r).__name__}. "
                f"Wrap a catalogue DataSource via "
                f"uacpy.data.DataProvenance(source=...)."
            )
    return records


def _dedupe_provenance(carriers) -> tuple:
    """Union of ``carrier.data_sources`` over ``carriers``, de-duplicated by
    source id and kept in first-seen order.

    The single home for the aggregation ``Bottom`` (over its columns),
    ``Surface`` (over its nodes) and ``Environment`` (over its five carriers)
    each expose. A carrier that is ``None``, or carries no ``data_sources``,
    contributes nothing.
    """
    seen, out = set(), []
    for carrier in carriers:
        for record in getattr(carrier, 'data_sources', ()) or ():
            if record.source.id not in seen:
                seen.add(record.source.id)
                out.append(record)
    return tuple(out)


def _sanitize_title(name: str) -> str:
    """Strip newlines/control chars and remove single quotes from a Fortran
    title field. Acoustics-Toolbox `.env` titles are single-quote-delimited and
    column-sensitive; an unsanitized name with a newline silently corrupts the
    file and the binary parses garbage downstream.

    Single quotes are **removed** (not doubled): the Fortran ``''`` escape that
    LDIFile (Kraken/Scooter) accepts is rejected by the C++ bellhopcxx title
    parser (the default Bellhop build), so a name like ``"O'Brien"`` would fail
    every Bellhop run. The title is a cosmetic label, so dropping the apostrophe
    is safe for every reader.
    """
    if name is None:
        return 'unnamed'
    s = str(name)
    s = ''.join(ch if (ord(ch) >= 32 and ch != '\x7f') else ' ' for ch in s)
    s = s.replace("'", "")
    return s.strip() or 'unnamed'
