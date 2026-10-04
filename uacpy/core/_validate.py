"""Input guards shared by every layer: the carriers, the results, io and the
signal estimators; the uniform-axis test (:func:`equally_spaced`) the io
deck writers and the Bellhop range check share; and the one uniform-step
decider (:func:`steps_are_uniform`) the transforms ask before an FFT.

Each refuses a bad value with a typed
:class:`~uacpy.core.exceptions.ConfigurationError` that names the argument:
the array guards (finite, positive, non-negative, strictly increasing,
real) name the offending flat index; the scalar and signal guards name the
calling function and the unit. One check warns instead of refusing: a sound
speed typed in km/s (:func:`warn_speed_typed_in_km_per_s`).
"""

import warnings
from typing import Optional

import numpy as np

from uacpy.core.exceptions import ConfigurationError, ValidityWarning
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.deck_limits import MAX_ATTENUATION_DB_PER_WAVELENGTH

#: Relative tolerance of "are these axis steps equal" (absolute tolerance 0):
#: every step of a uniform axis lies within this fraction of the reference
#: step. It bounds the jitter of the axes the package itself writes, measured
#: with float64: an ``np.arange(n) / fs`` time axis reaches 1.16e-9 at
#: 2**23 samples and 96 kHz (the ulp of the largest time over the step grows
#: with n), and a frequency vector printed ``%.12g`` into an AT deck and read
#: back reaches 2.0e-8 at 3000 bins of 1/3 Hz. An axis that needs resampling
#: before an FFT sits orders of magnitude above it.
UNIFORM_STEP_RTOL = 1e-6


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


def scalar_or_none(value, describe_bool) -> Optional[float]:
    """``float(value)`` for a Python/NumPy number or a numeric 0-d array,
    ``None`` for anything else (a sequence, an array, a carrier). A bool —
    Python or NumPy, bare or as a 0-d array — is refused with the
    ``ConfigurationError`` text ``describe_bool(value)`` composes: read as
    a scalar it would silently mean 0 or 1. A 0-d array is a scalar in every
    respect except ``isinstance``, so both branches see through it."""
    zero_d = isinstance(value, np.ndarray) and value.ndim == 0
    if (isinstance(value, (bool, np.bool_))
            or (zero_d and value.dtype == np.bool_)):
        raise ConfigurationError(describe_bool(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if zero_d and np.issubdtype(value.dtype, np.number):
        return float(value)
    return None


def require_positive(values, label: str, *, hint: str = "") -> None:
    """Raise ``ConfigurationError`` unless every element is finite and ``> 0``.

    Finiteness is checked first (NaN/inf pass every plain ``<= 0`` test) and
    reported separately, so the sign message stays the contiguous phrase
    ``"<label> must be positive"`` callers rely on. Takes an ARRAY and names
    the offending flat index, because a carrier validates a whole axis. The
    scalar counterpart that names a function argument and its unit is
    :func:`uacpy.core._validate.require_positive_finite_scalar`;
    the two reach the same verdict on every scalar, which
    ``TestThePositiveScalarGuardsAgreeAcrossLayers`` (test_sonar.py) pins. A
    non-numeric input raises numpy's own error here — a dtype problem, not a
    sign verdict — and a carrier test pins that too.
    """
    arr = np.asarray(values, dtype=float)
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
SPEED_UNITS_SUSPECT_M_S = 10.0


def warn_speed_typed_in_km_per_s(values, label: str) -> None:
    """``ValidityWarning`` when any sound speed in ``values`` is under
    :data:`SPEED_UNITS_SUSPECT_M_S`, naming the slowest and the m/s it
    means. A plausibility bound, so it warns and never raises."""
    arr = np.asarray(values, dtype=float)
    if arr.size and np.nanmin(arr) < SPEED_UNITS_SUSPECT_M_S:
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
    phrase ``"<label> must be non-negative"``.
    """
    arr = np.asarray(values, dtype=float)
    require_finite(arr, label, hint=hint)
    if np.any(arr < 0):
        flat = arr.ravel()
        bad = int(np.flatnonzero(flat < 0)[0])
        raise ConfigurationError(
            f"{label} must be non-negative, >= 0{_hint(hint)}; "
            f"got {flat[bad]} at flat index {bad} of {flat.size} value(s)")


def require_attenuation_in_range(value, label: str) -> None:
    """Raise ``ConfigurationError`` for an attenuation past the AT solvers' own
    ceiling — see :data:`uacpy.core.deck_limits.MAX_ATTENUATION_DB_PER_WAVELENGTH`
    for the derivation. Above it Kraken, Scooter, OAST and Bellhop's Fortran
    build all abort in ``CRCI``, while ``bellhopcxx``/``bellhopcuda`` return a
    field with *less* loss than a low attenuation gives.
    """
    if value is None:
        return
    alpha = float(value)
    if alpha > MAX_ATTENUATION_DB_PER_WAVELENGTH:
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
            f"{MAX_ATTENUATION_DB_PER_WAVELENGTH:.4f}, above which the imaginary "
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
    ``DECK_RANGE_RESOLUTION_M`` for a range axis or ``DECK_DEPTH_RESOLUTION_M``
    for a depth axis (see those constants for the readers this protects), with
    ``unit`` naming what the step is measured in — the axes guarded this way
    are not all spatial (``SBP_ANGLE_RESOLUTION_DEG`` guards a degree axis).

    A zero-length axis is refused here as well: every consumer reads the axis
    positionally (deck writers, ``np.interp``, ``min``/``max``), so an empty
    one would surface later as a bare IndexError naming no input. This is the
    only ``_require_*`` guard that rejects an empty array — the value
    predicates above also serve ``Field.coords``, where an axis sliced to
    nothing is a supported state.
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
    NaN/Inf propagate to every output sample. ``unit`` is appended to the
    message with its leading space (e.g. ``" Hz"``); ``why``, when given,
    follows the refusal with what this caller's value is used for.

    One of the package's three deliberate positive-scalar guards; it names the
    caller and the unit because an estimator's user is asking which argument
    went wrong. :func:`uacpy.core._validate.require_positive` carries
    the note on why three exist and what they must agree on.
    """
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"{who}: {name} must be a scalar number "
            f"(got {value!r}).") from exc
    if not np.isfinite(v) or v <= 0:
        raise ConfigurationError(
            f"{who}: {name} must be > 0{unit} and finite "
            f"(got {value!r}).{why}")
    return v


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


def require_increasing_axis(values, label: str):
    """Refuse an empty or non-monotonic frequency/time axis, typed.

    Delegates to ``core._validate.require_strictly_increasing``, the
    canonical form of this predicate, so the signal layer's axes and the
    carrier objects' axes answer the same question the same way. Its docstring
    carries the reasoning for both halves — in particular why an empty axis is
    refused (every consumer reads the axis positionally, so it otherwise fails
    much later inside numpy naming no input the caller supplied) and why a
    one-sample axis is accepted.

    Call it *after* a local monotonicity check that carries a
    domain-specific remediation hint: the local raise then answers the case it
    knows about and this backstops the empty axis.
    """
    require_strictly_increasing(values, label)


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
    maximum absolute difference is less than the tolerance.

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


def water_property(value, at, *, name: str, who: str,
                   at_name: str = 'depth'):
    """A water property (temperature, salinity, pH, ...) at the coordinate
    ``at`` it is evaluated at: the one rule every water-property input of
    uacpy follows.

    - a single value, or an array, is returned unchanged and broadcasts with
      the other arguments as it always did;
    - ``(N, 2)`` pairs ``[(coordinate, value), ...]`` — the convention of
      :meth:`~uacpy.core.ssp.SoundSpeedProfile.from_pairs` — are
      interpolated linearly onto ``at``, their end values held beyond their
      first and last coordinate. The coordinates must be finite and strictly
      increasing. An ``(N, 2)`` array the same shape as ``at`` is a grid of
      values, not pairs.

    Pairs need ``at``: a ``ConfigurationError`` names ``at_name=`` when it is
    ``None``. ``at_name`` names the coordinate; every uacpy entry point
    evaluates water properties on depth (m), the pressure equations
    converting it afterwards.
    """
    if np.ndim(value) != 2 or np.shape(value)[1] != 2 or (
            at is not None and np.shape(at) == np.shape(value)):
        return value
    if at is None:
        raise ConfigurationError(
            f"{who}: {name} is given as ({at_name}, value) pairs, which are "
            f"interpolated onto the {at_name} evaluated; pass {at_name}=.")
    pairs = np.asarray(value, dtype=float)
    require_finite(pairs, f"{who}: {name} pairs")
    require_strictly_increasing(pairs[:, 0], f"{who}: the {at_name}s of "
                                f"{name}", unit='m' if at_name == 'depth'
                                else 'dbar')
    return np.interp(np.asarray(at, dtype=float), pairs[:, 0], pairs[:, 1])
