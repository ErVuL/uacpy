"""Input guards shared by every layer: the carriers, the results, io and the
signal estimators; the uniform-axis test (:func:`equally_spaced`) the io
deck writers and the Bellhop range check share; and the one uniform-step
decider (:func:`steps_are_uniform`) the transforms ask before an FFT.

The two axis tests answer different questions, so their tolerances differ
on purpose. :func:`equally_spaced` asks whether a deck may write the axis
as "N points from x0 to x1" and have the engine regenerate every value: an
absolute question, since a deck prints in absolute units (1e-6 m,
``DECK_DEPTH_RESOLUTION``), so its 1e-9 tolerance sits under the print
resolution at any axis length. :func:`steps_are_uniform` asks whether a
transform may treat the steps as one: a relative question, answered against
the step itself.

Each refuses a bad value with a typed
:class:`~uacpy.core.exceptions.ConfigurationError` that names the argument:
the array guards (finite, positive, non-negative, strictly increasing,
real) name the offending flat index; the scalar and signal guards name the
calling function and the unit. One check warns instead of refusing: a sound
speed typed in km/s (:func:`warn_speed_typed_in_km_per_s`).
"""

import numbers
import warnings
from typing import Optional

import numpy as np

from uacpy.core.exceptions import ConfigurationError, ValidityWarning
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.deck_limits import MAX_ATTENUATION

#: Relative tolerance of "are these axis steps equal" (absolute tolerance 0):
#: every step of a uniform axis lies within this fraction of the reference
#: step. It bounds the jitter of the axes the package itself writes, measured
#: with float64: an ``np.arange(n) / fs`` time axis reaches 1.16e-9 at
#: 2**23 samples and 96 kHz (the ulp of the largest time over the step grows
#: with n), and a frequency vector printed ``%.12g`` into an AT deck and read
#: back reaches 2.0e-8 at 3000 bins of 1/3 Hz. An axis that needs resampling
#: before an FFT sits orders of magnitude above it.
UNIFORM_STEP_RTOL = 1e-6

#: g/cm³ above which a density reads as kg/m³: no water, sediment or rock
#: reaches it (Hamilton's tables top out below 3), and the smallest kg/m³
#: value a user could plausibly type (fresh water, 1000) is fifty times over
#: it. The one threshold the seabed carriers, the environment's water
#: column, reflection_coeff and bubble_resonance share.
DENSITY_UNITS_SUSPECT = 20.0


def _hint(hint: str) -> str:
    """Render an optional unit/context note appended after the rule clause.

    Kept separate from ``label`` so ``label`` ends immediately before
    ``"must be"`` — callers (and tests) can match ``"<noun> must be"`` as a
    contiguous phrase while the unit context still survives in the message.
    """
    return f" ({hint})" if hint else ""


def reject_complex(values, label: str) -> None:
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


def require_finite(values, label: str, *, hint: str = "") -> None:
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


def real_scalar_or_none(value) -> Optional[float]:
    """``float(value)`` for one real number — a ``numbers.Real`` that is
    not a bool (a Python or NumPy integer or float, a ``Fraction``), or a
    0-d array of integer or floating dtype — and ``None`` for anything
    else: a bool (bare or a 0-d array), a complex number, a string or
    bytes, a sequence, ``None``. Read as a number, a bool would mean 0 or 1,
    a complex value its real part and a string whatever it spells. A
    non-finite number is returned as it is: what a finite one means is the
    caller's to say.

    The package's one rule for a numeric input: the scalar guards
    (:func:`require_real_scalar`, :func:`require_positive_finite_scalar`),
    the array guard every carrier reads its numbers through
    (:func:`require_real_array`, element by element), the carriers' scalar
    reading (:func:`scalar_or_none`), the engines' knobs and the
    ``at``/``eval`` label guard all read through it."""
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, numbers.Real):
        try:
            return float(value)
        except OverflowError:
            return None
    try:
        number = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if number.ndim != 0 or number.dtype.kind not in 'iuf':
        return None
    return float(number)


def _holds_a_bool(values) -> bool:
    """Whether ``values`` — a scalar, a nested sequence or an array — holds
    a bool anywhere. NumPy reads ``[True, 2.0]`` as ``[1.0, 2.0]``, so the
    dtype of the array it makes does not show it."""
    if isinstance(values, (bool, np.bool_)):
        return True
    if isinstance(values, np.ndarray):
        if values.dtype == np.bool_:
            return True
        return (values.dtype == object
                and any(_holds_a_bool(v) for v in values.flat))
    if isinstance(values, (list, tuple)):
        return any(_holds_a_bool(v) for v in values)
    return False


def real_array_or_none(values) -> Optional[np.ndarray]:
    """``values`` as a float64 array when every element is one real number
    by :func:`real_scalar_or_none` — a bool looked for element by element
    (:func:`_holds_a_bool`) — and ``None`` otherwise: a bool anywhere, a
    complex value, text, ``None`` or another object, a ragged sequence."""
    if _holds_a_bool(values):
        return None
    try:
        array = np.asarray(values)
    except (TypeError, ValueError):
        return None
    if array.dtype.kind in 'iuf':
        return array.astype(np.float64)
    if array.dtype.kind != 'O':
        return None
    numbers_read = [real_scalar_or_none(v) for v in array.flat]
    if any(number is None for number in numbers_read):
        return None
    return np.array(numbers_read, dtype=np.float64).reshape(array.shape)


def _not_one_real_number(value) -> str:
    """What a scalar input must be, said against what ``value`` is, for a
    value :func:`real_scalar_or_none` does not read."""
    try:
        kind = np.asarray(value).dtype.kind
    except (TypeError, ValueError):
        kind = 'O'
    if kind == 'b':
        return "a number, not a bool"
    if kind == 'c':
        return "a real number, not complex"
    if kind in 'US':
        return "a number, not text"
    return "a scalar number"


def _not_real_numbers(values) -> str:
    """What an array input holds, for one :func:`real_array_or_none` does
    not read."""
    if _holds_a_bool(values):
        return "a bool"
    try:
        kind = np.asarray(values).dtype.kind
    except (TypeError, ValueError):
        return "a ragged sequence"
    if kind in 'US':
        return "text"
    return "a value that is not a number"


def require_real_scalar(value, label: str, *, why: str = "",
                        hint: str = "") -> float:
    """``value`` as a ``float`` when it is one real number
    (:func:`real_scalar_or_none`), finite or not; otherwise a
    ``ConfigurationError`` "``label`` must be a number, not a bool" (or not
    complex, not text, or a scalar number) naming the value, with ``hint``
    (a unit) as :func:`require_positive` gives it. ``why``, when given,
    follows the refusal with what to pass."""
    number = real_scalar_or_none(value)
    if number is None:
        raise ConfigurationError(
            f"{label} must be {_not_one_real_number(value)}{_hint(hint)} "
            f"(got {value!r}).{why}")
    return number


def require_real_array(values, label: str, *, hint: str = "") -> np.ndarray:
    """``values`` as a float64 array (0-d for one number) when every element
    is one real number by the package's one rule (:func:`real_array_or_none`);
    the guard each carrier reads its numbers through, so the cast is this
    guard's own. A scalar is refused as :func:`require_real_scalar` refuses
    it; a complex array by :func:`reject_complex`'s text, naming the first
    element with an imaginary part; an array holding a bool (anywhere: NumPy
    reads ``[True, 2.0]`` as numbers), text or anything that is not a number
    as "``label`` must be real numbers…". NaN passes: finiteness is the
    caller's to judge."""
    try:
        ndim = np.ndim(values)
    except (TypeError, ValueError):
        ndim = None
    if ndim == 0:
        return np.asarray(require_real_scalar(values, label, hint=hint))
    if ndim is not None:
        reject_complex(values, label)
    array = None if ndim is None else real_array_or_none(values)
    if array is None:
        raise ConfigurationError(
            f"{label} must be real numbers{_hint(hint)}; got "
            f"{_not_real_numbers(values)} in {values!r}.")
    return array


def scalar_or_none(value, describe_bool) -> Optional[float]:
    """``float(value)`` for one real number (:func:`real_scalar_or_none`),
    ``None`` for anything else (a sequence, an array, a carrier, a string,
    a complex value), which the caller reads its own way or refuses. A bool
    — Python or NumPy, bare or as a 0-d array — is refused with the
    ``ConfigurationError`` text ``describe_bool(value)`` composes: read as
    a scalar it would silently mean 0 or 1."""
    try:
        zero_d_bool = (np.ndim(value) == 0
                       and np.asarray(value).dtype == np.bool_)
    except (TypeError, ValueError):
        zero_d_bool = False
    if zero_d_bool:
        raise ConfigurationError(describe_bool(value))
    return real_scalar_or_none(value)


def require_positive(values, label: str, *, hint: str = "") -> None:
    """Raise ``ConfigurationError`` unless every element is finite and ``> 0``.

    Finiteness is checked first (NaN/inf pass every plain ``<= 0`` test) and
    reported separately, so the sign message stays the contiguous phrase
    ``"<label> must be positive"`` callers rely on. Takes an ARRAY and names
    the offending flat index, because a carrier validates a whole axis. The
    scalar counterpart that names a function argument and its unit is
    :func:`uacpy.core._validate.require_positive_finite_scalar`.

    Why more than one positive guard: each names what its caller's user is
    asking. A carrier's user needs the axis and the index that failed; an
    estimator's user needs the argument of the function they called and its
    unit; the target-strength formulas route through the scalar guard under
    their own ``"target strength"`` prefix. The messages may differ; the
    verdict on a scalar may not — the same frequency reaches all of them by
    different doors — and ``TestThePositiveScalarGuardsAgreeAcrossLayers``
    (test_sonar.py) pins that they agree on every scalar. The values are
    read by the package's one rule (:func:`require_real_array`): a bool, a
    complex value, text, ``None`` or any other object that is not one real
    number is refused, alone or anywhere in an array.
    """
    arr = require_real_array(values, label, hint=hint)
    require_finite(arr, label, hint=hint)
    if np.any(arr <= 0):
        flat = arr.ravel()
        bad = int(np.flatnonzero(flat <= 0)[0])
        raise ConfigurationError(
            f"{label} must be positive, > 0{_hint(hint)}; "
            f"got {flat[bad]} at flat index {bad} of {flat.size} value(s)")


#: m/s below which a sound speed reads as km/s: the slowest medium a model
#: is run over (bubbly water) is tens of m/s, and a km/s value typed for
#: water (1.4-1.6) or a seabed (1.5-6) is under it.
SPEED_UNITS_SUSPECT = 10.0


def warn_speed_typed_in_km_per_s(values, label: str) -> None:
    """``ValidityWarning`` when any sound speed in ``values`` is under
    :data:`SPEED_UNITS_SUSPECT`, naming the slowest and the m/s it
    means. A plausibility bound, so it warns and never raises."""
    arr = np.asarray(values, dtype=float)
    if arr.size and np.nanmin(arr) < SPEED_UNITS_SUSPECT:
        slowest = float(np.nanmin(arr))
        warnings.warn(
            f"{label}: {slowest:g} m/s looks like km/s; uacpy takes m/s "
            f"({slowest * 1000.0:g}). Every deck writes the value as given, "
            f"so a run over it returns numbers for a medium a thousand times "
            f"slower than meant.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP)


def require_non_negative(values, label: str, *, hint: str = "") -> None:
    """Raise ``ConfigurationError`` unless every element is finite and ``>= 0``.

    Finiteness is reported separately so the sign message stays the contiguous
    phrase ``"<label> must be non-negative"``. Values that are not real
    numbers are refused first, as by :func:`require_positive`.
    """
    arr = require_real_array(values, label, hint=hint)
    require_finite(arr, label, hint=hint)
    if np.any(arr < 0):
        flat = arr.ravel()
        bad = int(np.flatnonzero(flat < 0)[0])
        raise ConfigurationError(
            f"{label} must be non-negative, >= 0{_hint(hint)}; "
            f"got {flat[bad]} at flat index {bad} of {flat.size} value(s)")


#: dB/wavelength above which a seabed attenuation reads as another unit: ten
#: times the most lossy compressional entry of JKPS Table 1.3 (silt, 1.0) and
#: four times its most lossy shear entry (sand, 2.5). A value in dB/km or dB/m
#: typed as dB/wavelength lands here or past the solvers' ceiling.
SEABED_ATTENUATION_SUSPECT = 10.0


def warn_attenuation_beyond_seabeds(value, label: str) -> None:
    """``ValidityWarning`` when a seabed attenuation (dB/wavelength) is over
    :data:`SEABED_ATTENUATION_SUSPECT`, naming the unit
    and the converter. A plausibility bound, so it warns and never raises;
    past the Acoustics-Toolbox ceiling :func:`warn_attenuation_past_at_ceiling`
    speaks instead (one warning per value)."""
    if value is None:
        return
    alpha = float(value)
    if (SEABED_ATTENUATION_SUSPECT < alpha
            <= MAX_ATTENUATION):
        warnings.warn(
            f"{label} = {alpha:g} dB/wavelength is far beyond any seabed in "
            f"JKPS Table 1.3 (0.1-1.0 dB/wavelength compressional, 0.2-2.5 "
            f"shear). uacpy takes dB/wavelength, and a value in dB/km or dB/m "
            f"typed here reads that large; "
            f"uacpy.acoustics.convert_attenuation_units converts it.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP)


def warn_attenuation_past_at_ceiling(value, label: str) -> None:
    """``ValidityWarning`` on a carrier for an attenuation past the
    Acoustics-Toolbox ceiling
    (:data:`uacpy.core.deck_limits.MAX_ATTENUATION`): the
    carrier keeps it, because OASES and RAM take any attenuation
    (``oaseun31.f:1522-1523`` and ``ramgeo1.5.f:199`` turn it into an
    imaginary wavenumber), and the AT deck writers refuse it
    (:func:`require_attenuation_in_range`)."""
    if value is None:
        return
    alpha = float(value)
    if alpha > MAX_ATTENUATION:
        warnings.warn(
            f"{label} = {alpha:g} dB/wavelength exceeds "
            f"{MAX_ATTENUATION:.4f}, the ceiling of the "
            f"Acoustics-Toolbox solvers (misc/AttenMod.f90's CRCI, :116): "
            f"Kraken, Scooter, SPARC, Bounce and Bellhop refuse it at run, "
            f"OASES and RAM take it. uacpy takes dB/wavelength, and a value "
            f"in dB/km or dB/m typed here reads that large; "
            f"uacpy.acoustics.convert_attenuation_units converts it.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP)


def require_attenuation_in_range(value, label: str) -> None:
    """Raise ``ConfigurationError`` for an attenuation past the AT solvers' own
    ceiling — see :data:`uacpy.core.deck_limits.MAX_ATTENUATION`
    for the derivation. The Acoustics-Toolbox deck writers call it: above
    the ceiling the Fortran solvers (Kraken, Scooter, SPARC, Bounce,
    Bellhop's Fortran build) abort in ``CRCI``
    (``misc/AttenMod.f90:116-120``), while ``bellhopcxx``/``bellhopcuda``
    return a field with *less* loss than a low attenuation gives. OASES and
    RAM have no such bound (``oaseun31.f:1522-1523`` and ``ramgeo1.5.f:199``
    turn any attenuation into an imaginary wavenumber), and the carriers
    only warn (:func:`warn_attenuation_past_at_ceiling`).
    """
    if value is None:
        return
    alpha = float(value)
    if alpha > MAX_ATTENUATION:
        # The two attenuations this guard serves sit at different scales, so
        # each gets its own sentence. JKPS Table 1.3 runs α_p from 0.1
        # (basalt) to 1.0 dB/wavelength (silt) but α_s from 0.2 to 2.5, and
        # uacpy ships that 2.5 as the ``sand`` preset (core/materials.py), so
        # a shear attenuation near 2 is a real seabed.
        #
        # Both branches raise with the remediation written out at the
        # ``raise`` rather than sharing one built above: a
        # ``remediation=<name>`` hides its text from
        # ``test_error_actionability.py``'s content check, which counts such
        # sites precisely so the blind spot cannot grow.
        message = (
            f"{label} = {alpha:g} dB/wavelength exceeds "
            f"{MAX_ATTENUATION:.4f}, above which the imaginary "
            f"part of the complex sound speed exceeds the real part and every AT "
            f"solver aborts in misc/AttenMod.f90's CRCI (:116). The bound is "
            f"independent of frequency and sound speed because uacpy writes "
            f"AttenUnit 'W' (dB/wavelength)."
        )
        if 'shear_attenuation' in label:
            raise ConfigurationError(
                message,
                remediation="Shear attenuation runs higher than "
                            "compressional: JKPS Table 1.3 gives 0.2-2.5 "
                            "dB/wavelength across seafloor types (sand is "
                            "2.5, which uacpy ships as the 'sand' preset), so "
                            "a value of a few dB/wavelength is ordinary and "
                            "only a value orders of magnitude above the table "
                            "is not. To model a strongly absorbing bottom use "
                            "a reflection-coefficient table "
                            "(acoustic_type='file') instead.",
            )
        raise ConfigurationError(
            message,
            remediation="Real seabeds are well under 2 dB/wavelength "
                        "(JKPS Table 1.3 tops out at 1.0, for silt). To model "
                        "a strongly absorbing bottom use a "
                        "reflection-coefficient table (acoustic_type='file') "
                        "instead.",
        )


def require_strictly_increasing(values: np.ndarray, label: str, *,
                                min_step: float = 0.0,
                                unit: str = 'm') -> None:
    """Raise ``ConfigurationError`` if ``values`` is not strictly
    monotonically increasing. Used to guard every range / depth axis that
    feeds into ``np.interp``, which silently produces garbage on unsorted
    ``xp``.

    ``min_step`` additionally requires neighbours to be separated by more than
    the resolution the solver decks print the axis at, since two samples closer
    than that collapse to a single token in the file. Pass
    ``DECK_RANGE_RESOLUTION`` for a range axis or ``DECK_DEPTH_RESOLUTION``
    for a depth axis (see those constants for the readers this protects), with
    ``unit`` naming what the step is measured in — the axes guarded this way
    are not all spatial (``SBP_ANGLE_RESOLUTION_DEG`` guards a degree axis).

    A zero-length axis is refused here as well: every consumer reads the axis
    positionally (deck writers, ``np.interp``, ``min``/``max``), so an empty
    one would surface later as a bare IndexError naming no input. This is the
    only ``require_*`` guard that rejects an empty array — the value
    predicates above also serve ``Field.coords``, where an axis sliced to
    nothing is a supported state.

    The signal layer's frequency/time axes and the carriers' axes answer
    this one question the same way. Call it *after* a local monotonicity
    check that carries a domain-specific remediation hint: the local raise
    then answers the case it knows about and this backstops the empty axis.
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


def require_positive_finite_scalar(value, who: str, name: str,
                                   unit: str = "", *, why: str = ""):
    """Validate a scalar parameter as finite and > 0; return it as ``float``.

    The estimators divide by these scalars (a sample rate, a sensor spacing,
    a reference pressure): zero raises a raw ``ZeroDivisionError`` deep in
    scipy or silently collapses an axis, a negative value flips it, and
    NaN/Inf propagate to every output sample. The value is read by the
    package's one rule for a scalar (:func:`require_real_scalar`): a bool —
    bare or as a 0-d array — would read as 1, a complex value as its real
    part, text as what it spells, and each is refused. ``unit`` is appended
    to the message with its leading space (e.g. ``" Hz"``); ``why``, when
    given, follows every refusal with what this caller's value is used for
    or what to pass.

    The scalar positive guard; it names the caller and the unit because an
    estimator's user is asking which argument went wrong.
    :func:`uacpy.core._validate.require_positive`, the array guard, carries
    the note on why both exist and what they must agree on.
    """
    v = require_real_scalar(value, f"{who}: {name}", why=why)
    if not np.isfinite(v) or v <= 0:
        raise ConfigurationError(
            f"{who}: {name} must be > 0{unit} and finite "
            f"(got {value!r}).{why}")
    return v


def require_generator(rng, who: str):
    """``rng`` when it is a :class:`numpy.random.Generator`, a new
    ``np.random.default_rng()`` when it is ``None``; anything else (a seed
    integer, a legacy ``RandomState``) is refused, since the callers draw
    with Generator methods."""
    if rng is None:
        return np.random.default_rng()
    if not isinstance(rng, np.random.Generator):
        raise ConfigurationError(
            f"{who}: rng must be a numpy.random.Generator or None; got "
            f"{type(rng).__name__}.",
            remediation="Pass rng=np.random.default_rng(seed) for a "
                        "reproducible draw.")
    return rng


def refuse_non_finite_samples(rows, who, *, start=0, first_channel=0,
                              n_channels=1, label='sample'):
    """Refuse a block ``rows`` ``(channels, time)`` of a record that holds a
    NaN or inf, naming the earliest one: its index along the time axis
    (``start`` plus its place in the block) and, in a record of more than
    one channel, its channel (the record's other axes flattened, counted from
    ``first_channel``). ``label`` names the time index (``'flat index'``
    for a record read flattened). The one refusal of the level and exposure
    functions: a level of a record missing a sample is undefined, and a NaN
    or inf would carry into it silently."""
    bad = ~np.isfinite(np.asarray(rows))
    if not np.any(bad):
        return
    t = int(np.argmax(np.any(bad, axis=0)))
    where = f"{label} {start + t}"
    if n_channels > 1:
        where += (f" of channel {first_channel + int(np.argmax(bad[:, t]))} "
                  f"(the record's other axes flattened)")
    raise ConfigurationError(
        f"{who}: the record holds NaN or inf, the first at {where}; a level "
        f"of a record with a missing sample is undefined.",
        remediation="Remove or fill the non-finite samples, or cut the "
                    "record before that one.")


def require_finite_signal(data, who: str, name: str = "data"):
    """Validate a non-empty, finite signal and return it as an ndarray.

    ``name`` is the caller's parameter name, used in the refusals.

    Rejects an empty signal and any NaN/Inf with a typed
    :class:`~uacpy.core.exceptions.ConfigurationError`. Does **not** reject
    complex input — Welch/STFT handle complex baseband legitimately — but a
    complex array is returned as-is for the caller's own dtype handling.
    """
    arr = np.asarray(data)
    if arr.dtype == object:
        # A Field (or any object numpy cannot make an array of) reaches
        # np.isfinite as an object array and raises
        # "ufunc 'isfinite' not supported for the input types", which names
        # nothing the caller passed. The estimators here are array-in /
        # array-out by design (DOCUMENTATION.md calls acoustic_signal and
        # comms "pure computation"); `.data` is the documented bridge.
        what = type(data).__name__
        bridge = (f"Pass {what}.data (and its .sample_rate)."
                  if hasattr(data, "data") and hasattr(data, "coords")
                  else "Pass a numeric array.")
        raise ConfigurationError(
            f"{who}: {name} must be a numeric array; got {what}, which numpy "
            f"holds as dtype=object. {bridge}")
    if arr.size == 0:
        raise ConfigurationError(
            f"{who}: {name} is empty; provide a non-empty signal.")
    if not np.all(np.isfinite(arr)):
        raise ConfigurationError(
            f"{who}: {name} contains NaN or Inf, which would silently "
            "contaminate the estimate; clean the signal first. Got "
            f"{int(np.count_nonzero(~np.isfinite(arr)))} non-finite value(s) "
            f"of {arr.size}, first at flat index "
            f"{int(np.argmax(~np.isfinite(arr)))}.")
    return arr


def require_real_signal(data, who: str, *, ndim: int = 1,
                        name: str = "data", why: str = "",
                        shape_hint: str = "",
                        remediation: Optional[str] = None) -> np.ndarray:
    """Validate a real, finite signal of ``ndim`` dimensions; return it as
    ``float``.

    A complex input is refused first, because the cast to ``float`` would
    drop its imaginary part without a word; ``why`` completes that refusal
    with what the caller's transform is defined on, and ``remediation`` names
    the alternative. Then the dimension (``shape_hint`` spells the axes, e.g.
    ``" (nt, nx)"``), then :func:`require_finite_signal` (empty, NaN, Inf).
    """
    arr = np.asarray(data)
    if np.iscomplexobj(arr):
        raise ConfigurationError(
            f"{who}: {name} must be real (got complex input){why}",
            remediation=remediation)
    x = arr.astype(float)
    if x.ndim != ndim:
        raise ConfigurationError(
            f"{who}: {name} must be {ndim}-D{shape_hint}; "
            f"got shape {x.shape}.")
    require_finite_signal(x, who, name)
    return x


#: The most channels a 2-D record's layout guards take an axis to hold:
#: ``read_wav`` returns ``(n_samples, n_channels)``, so a trailing axis this
#: short (and shorter than the leading one) is channels to the estimators'
#: time-axis guard, and ``write_wav`` reads more channels than this, and
#: than frames, as a channel-first array. One threshold for the one layout
#: question both ask (uacpy guidance).
_MAX_CHANNELS_GUESSED = 64


def normalize_axis(data, axis, who: str) -> int:
    """``axis`` as a non-negative index into the dimensions of ``data`` (an
    array). Refuses an axis that is not an integer and one the array does
    not have; a negative axis counts from the end."""
    try:
        requested = int(axis)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"{who}: axis must be an integer; got {axis!r}.") from exc
    if not -data.ndim <= requested < data.ndim:
        raise ConfigurationError(
            f"{who}: axis={requested} is not an axis of an array with "
            f"shape {data.shape}.")
    return requested % data.ndim


def _nyquist_check(frequency, sample_rate, who, what, consequence,
                   relation, admissible):
    """Raise unless every entry of ``frequency`` satisfies ``admissible``.

    ``relation`` is the words for the failure ("at or above" / "above"); the
    message names the offending values and never dumps a whole grid.
    """
    f = np.asarray(frequency, dtype=float)
    nyquist = float(sample_rate) / 2.0
    bad = ~admissible(f, nyquist)
    if not np.any(bad):
        return
    tail = (f"the Nyquist frequency sample_rate/2 = {nyquist:g} Hz, so "
            f"{consequence}.")
    if f.ndim == 0:
        raise ConfigurationError(
            f"{who}: {what} ({float(f):g} Hz) is {relation} {tail}")
    offenders = np.unique(f[bad])
    raise ConfigurationError(
        f"{who}: {what} {offenders} Hz are {relation} {tail}")


def require_below_nyquist(frequency, sample_rate, who, what, consequence):
    """Refuse a *generator* frequency at or above ``sample_rate/2``.

    Strict: a sinusoid sampled exactly at Nyquist is degenerate (two samples
    per cycle carry no phase), so every waveform, sequence and modulation
    entry point rejects ``f == fs/2``. The analyser side of the same question
    is :func:`require_at_most_nyquist`, which admits it because an ``rfft``
    grid contains the Nyquist bin — a new entry point has to pick one.

    ``frequency`` may be a scalar or an array; the message names the offending
    values. ``consequence`` completes the sentence "…, so <consequence>." and
    says what the alias does to *this* caller's output.
    """
    _nyquist_check(frequency, sample_rate, who, what, consequence,
                   "at or above", lambda f, nyq: f < nyq)


def require_at_most_nyquist(frequency, sample_rate, who, what, consequence):
    """Refuse an *analyser* frequency strictly above ``sample_rate/2``.

    Admits ``f == fs/2``: the Nyquist bin is a real bin of an ``rfft`` grid and
    the default analysis grids cap themselves there, so refusing it would
    reject a caller's own default. The generator side is
    :func:`require_below_nyquist`.
    """
    _nyquist_check(frequency, sample_rate, who, what, consequence,
                   "above", lambda f, nyq: f <= nyq)


def sanitize_title(name: str) -> str:
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


def equally_spaced(x: np.ndarray, tol: float = 1e-9) -> bool:
    """
    Test whether vector x is composed of equally-spaced values.

    Parameters
    ----------
    x : ndarray
        Vector to test
    tol : float, optional
        Tolerance for equality test. Default is 1e-9.

    Returns
    -------
    is_equal : bool
        True if x is equally spaced within tolerance

    Notes
    -----
    Compares the input vector against a linearly spaced vector with
    the same start, end, and number of points. Returns True if the
    maximum absolute difference is less than the tolerance — absolute
    because the question is a deck's (may it write "N points from x0 to
    x1"?), and a deck prints in absolute units; :func:`steps_are_uniform`
    is the relative test a transform asks (see the module docstring).

    This is useful for determining if a vector can be represented
    compactly (e.g., "N points from x0 to x1") rather than storing
    all values explicitly.

    Translated from OALIB equally_spaced.m

    Examples
    --------
    >>> # Equally spaced
    >>> x = np.linspace(0, 10, 11)
    >>> equally_spaced(x)
    True

    >>> # Not equally spaced
    >>> x = np.array([0, 1, 3, 7, 10])
    >>> equally_spaced(x)
    False

    >>> # Jitter on one interior sample beyond tolerance
    >>> x = np.linspace(0, 10, 11); x[5] += 1e-6
    >>> equally_spaced(x)
    False
    """
    x = np.asarray(x).ravel()
    n = len(x)

    if n <= 1:
        return True

    # Generate equally spaced vector
    x_linspace = np.linspace(x[0], x[-1], n)

    # Compute maximum deviation
    delta = np.abs(x - x_linspace)

    # bool(), not the bare comparison: ``np.max(...) < tol`` is a numpy scalar,
    # and np.bool_ is NOT a Python bool — ``isinstance(r, bool)`` and
    # ``r is True`` are both False, and json.dumps raises TypeError on it. The
    # ``n <= 1`` branch above returns a real bool too, so the return type is
    # the same for every input length.
    return bool(np.max(delta) < tol)


def steps_are_uniform(steps, step: float) -> bool:
    """True when every value in ``steps`` equals ``step`` to within
    :data:`UNIFORM_STEP_RTOL` (relative, absolute tolerance 0): the one
    decider of whether an axis is uniform enough to transform as one."""
    return bool(np.allclose(np.asarray(steps, dtype=float), float(step),
                            rtol=UNIFORM_STEP_RTOL, atol=0.0))


def canonical_choice(value, choices, who: str, name: str) -> str:
    """The member of ``choices`` that ``value`` names, matched without
    regard to case, so ``'TEOS10'`` and ``'Hamilton'`` reach the code as
    ``'teos10'`` and ``'hamilton'``. For named choices whose case means
    nothing; a unit string or an option like ``'H1'`` stays exact.
    Raises ``ConfigurationError`` naming the choices when none matches."""
    by_fold = {str(c).casefold(): c for c in choices}
    try:
        return by_fold[str(value).casefold()]
    except KeyError:
        raise ConfigurationError(
            f"{who}: unknown {name} {value!r}.",
            remediation=f"Use one of {tuple(choices)} (in any case).") from None


def refuse_two_columns(value, *, who: str, name: str, columns: str,
                       pair: str, carrier: str, remediation: str) -> None:
    """Refuse ``value`` when it is two columns, where a table carrier takes
    ``(N, 2)`` pairs: shape ``(2, N)`` with ``N != 2`` (a pair has two
    entries, so such an input cannot be pairs), or, at any length, a list /
    tuple of two 1-D numpy arrays or a tuple of two lists — the spellings
    of ``(ranges, depths)``; at two samples each they would otherwise read
    as two pairs, a different table with no message. The one rule the
    ``ssp=``, ``bathymetry=`` and ``altimetry=`` coercers share; pass the
    caller's raw ``value``.

    Pairs are a list of tuples ``[(r0, d0), (r1, d1)]``, a tuple of tuples,
    a nested list ``[[r0, d0], [r1, d1]]`` or a ``2 x 2`` array, each read
    as two pairs, the documented form. A value numpy cannot make a float
    array of is left to the caller's own check.
    """
    try:
        shape = np.shape(np.asarray(value, dtype=float))
    except (TypeError, ValueError):
        return
    two_columns = (
        isinstance(value, (list, tuple)) and len(value) == 2
        and (all(isinstance(v, np.ndarray) and v.ndim == 1 for v in value)
             or (isinstance(value, tuple)
                 and all(isinstance(v, list) for v in value))))
    if two_columns or (len(shape) == 2 and shape[0] == 2 and shape[1] != 2):
        raise ConfigurationError(
            f"{who}: {name}=({columns}) is two columns; {name}= takes "
            f"({pair}) PAIRS or {carrier}.",
            remediation=remediation + (
                f" Two 1-D arrays, or a tuple of two lists, are read as "
                f"columns at any length; write two ({pair}) pairs as "
                f"[({pair}), ({pair})]." if two_columns else ""))


def water_pairs(value, *, name: str, who: str):
    """``(depth, value)`` pairs — a water property's profile on depths of
    its own, an ``(M, 2)`` table — as the
    :class:`~uacpy.core._water.WaterProfile` record the carriers
    store and the formulas interpolate. The depths (m, positive down) are
    finite, non-negative and strictly increasing, the values finite; any
    other table is refused naming ``who`` and the property ``name``."""
    # Deferred: core._water imports core._export, which imports this module.
    from uacpy.core._water import WaterProfile
    table = require_real_array(value, f"{who}: {name} pairs")
    if table.ndim != 2 or table.shape[1] != 2 or table.shape[0] == 0:
        raise ConfigurationError(
            f"{who}: {name} pairs have shape {table.shape}; (depth, value) "
            f"pairs have shape (M, 2), M >= 1.",
            remediation="Pass [(depth, value), ...], or "
                        "np.column_stack([depths, values]).")
    require_finite(table, f"{who}: {name} pairs")
    depths = table[:, 0]
    if np.any(depths < 0):
        raise ConfigurationError(
            f"{who}: the depths of the {name} pairs must be non-negative (m, "
            f"positive down); got {depths.tolist()}.",
            remediation="Give each pair as (depth in metres below the "
                        "surface, value).")
    require_strictly_increasing(depths, f"{who}: the depths of the {name} "
                                f"pairs")
    return WaterProfile(depths, table[:, 1])


def water_shape(value, *, name: str, who: str) -> tuple:
    """``np.shape(value)`` of a water property, refused as malformed pairs
    when ``value`` is a ragged sequence — pairs typed by hand with an entry
    missing or extra, ``[(0, 20), (100,)]`` — which numpy cannot shape, and
    refused when it is not real numbers by the package's one rule
    (:func:`real_array_or_none`: a bool, a complex value or text). A
    :class:`~uacpy.core._water.WaterProfile` is its own record."""
    # Deferred: core._water imports core._export, which imports this module.
    from uacpy.core._water import WaterProfile
    try:
        shape = np.shape(value)
    except ValueError as exc:
        raise ConfigurationError(
            f"{who}: {name} is a ragged sequence (an entry with a missing or "
            f"extra element), so it is neither one value per point nor "
            f"(depth, value) pairs.",
            remediation="Give every pair two entries, (depth, value), or "
                        "build them with np.column_stack([depths, values]).",
        ) from exc
    if (not isinstance(value, WaterProfile)
            and real_array_or_none(value) is None):
        raise ConfigurationError(
            f"{who}: {name} must be real numbers; got "
            f"{_not_real_numbers(value)} in {value!r}.",
            remediation=f"Pass {name} as a number, one per point, or "
                        f"(depth, value) pairs.")
    return shape


def water_property(value, at, *, name: str, who: str,
                   at_name: str = 'depth'):
    """A water property (temperature, salinity, pH) at the coordinate ``at``
    it is evaluated at: the one rule every water-property input of uacpy
    follows. ``value`` is exactly one of

    - a single value — constant water — returned unchanged;
    - an array of exactly ``at``'s shape — one value per point evaluated —
      returned unchanged;
    - otherwise ``(depth, value)`` pairs, an ``(M, 2)`` array — a profile
      on depths of its own (:func:`water_pairs`) — interpolated linearly
      onto ``at`` (end values held), so it needs a depth ``at``. A
      :class:`~uacpy.core._water.WaterProfile`, the record a
      carrier stores pairs as and hands back, is read the same way.

    Any other array is refused: a water array is never broadcast against
    ``at``. Values come first, so ``M`` pairs evaluated on a depth grid of
    exactly shape ``(M, 2)`` are read as values. ``at=None`` is no
    coordinate given, which the formulas evaluate at the surface, one point
    of shape ``()``. ``at_name`` names the coordinate: ``'depth'`` (m), or
    ``'pressure_dbar'`` for an equation evaluated at a pressure, where a
    profile (on depths) is refused.
    """
    # Deferred: core._water imports core._export, which imports this module.
    from uacpy.core._water import WaterProfile
    shape = water_shape(value, name=name, who=who)
    at_shape = np.shape(at)
    if isinstance(value, WaterProfile):
        profile, desc = value, ''
    elif len(shape) == 0:
        return value
    elif at is not None and shape == at_shape:
        return value
    elif len(shape) == 2 and shape[1] == 2:
        profile, desc = value, f" ((depth, value) pairs of shape {shape})"
    else:
        where = (f"no {at_name} is given, so it is evaluated at the "
                 f"surface, one point of shape ()" if at is None else
                 f"the {at_name} it is evaluated at has shape {at_shape}")
        if at is None:
            grid = (f"{at_name}=np.zeros_like({name}) (the surface, shape "
                    f"{shape}) to evaluate the array point by point")
        elif at_shape == ():
            grid = (f"{at_name}=np.full_like({name}, {at_name}) (shape "
                    f"{shape}) to evaluate the array point by point")
        else:
            grid = f"an array of shape {at_shape} for one value per point"
        raise ConfigurationError(
            f"{who}: {name} has shape {shape}, and {where}. A water property "
            f"is a single value (constant water), an array of exactly the "
            f"{at_name}'s shape (one value per point) or (depth, value) "
            f"pairs of shape (M, 2) (a profile on depths of its own, "
            f"interpolated onto the depth evaluated); an array is never "
            f"broadcast against the {at_name}.",
            remediation=f"Pass {grid}, np.column_stack([depths, values]) "
                        f"pairs for a profile, or a single value for "
                        f"constant water.")
    if at is None or at_name != 'depth':
        raise ConfigurationError(
            f"{who}: {name} is a profile on depths of its own{desc}, which "
            f"is interpolated onto the depth evaluated; pass depth=.",
            remediation=(f"Pass depth= (m); an array of {name} values of "
                         f"the coordinate's shape is read point by point "
                         f"instead."))
    if not isinstance(profile, WaterProfile):
        profile = water_pairs(profile, name=name, who=who)
    return profile(at)
