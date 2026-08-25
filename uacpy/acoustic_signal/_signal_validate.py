"""Shared input guard for the spectral/time-frequency estimators.

Rejects empty and NaN/Inf input with a typed ConfigurationError, so every
spectral, time-frequency and constant-Q estimator fails the same way instead of
propagating a single bad sample into an all-NaN result.
"""

import numpy as np

from uacpy.core.exceptions import ConfigurationError
from uacpy.core._carrier_validate import _require_strictly_increasing


def require_positive_finite_scalar(value, caller: str, name: str,
                                   unit: str = ""):
    """Validate a scalar parameter as finite and > 0; return it as ``float``.

    The estimators divide by these scalars (a sample rate, a sensor spacing,
    a reference pressure): zero raises a raw ``ZeroDivisionError`` deep in
    scipy or silently collapses an axis, a negative value flips it, and
    NaN/Inf propagate to every output sample. ``unit`` is appended to the
    message with its leading space (e.g. ``" Hz"``).

    One of the package's three deliberate positive-scalar guards; it names the
    caller and the unit because an estimator's user is asking which argument
    went wrong. :func:`uacpy.core._carrier_validate._require_positive` carries
    the note on why three exist and what they must agree on.
    """
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"{caller}: {name} must be a scalar number "
            f"(got {value!r}).") from exc
    if not np.isfinite(v) or v <= 0:
        raise ConfigurationError(
            f"{caller}: {name} must be > 0{unit} and finite (got {value!r}).")
    return v


def require_finite_signal(data, caller: str):
    """Validate a non-empty, finite signal and return it as an ndarray.

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
            f"{caller}: data must be a numeric array; got {what}, which numpy "
            f"holds as dtype=object. {bridge}")
    if arr.size == 0:
        raise ConfigurationError(
            f"{caller}: data is empty; provide a non-empty signal.")
    if not np.all(np.isfinite(arr)):
        raise ConfigurationError(
            f"{caller}: data contains NaN or Inf, which would silently "
            "contaminate the estimate; clean the signal first. Got "
            f"{int(np.count_nonzero(~np.isfinite(arr)))} non-finite value(s) "
            f"of {arr.size}, first at flat index "
            f"{int(np.argmax(~np.isfinite(arr)))}.")
    return arr


def _nyquist_check(frequency, sample_rate, caller, what, consequence,
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
            f"{caller}: {what} ({float(f):g} Hz) is {relation} {tail}")
    offenders = np.unique(f[bad])
    raise ConfigurationError(
        f"{caller}: {what} {offenders} Hz are {relation} {tail}")


def require_below_nyquist(frequency, sample_rate, caller, what, consequence):
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
    _nyquist_check(frequency, sample_rate, caller, what, consequence,
                   "at or above", lambda f, nyq: f < nyq)


def require_at_most_nyquist(frequency, sample_rate, caller, what, consequence):
    """Refuse an *analyser* frequency strictly above ``sample_rate/2``.

    Admits ``f == fs/2``: the Nyquist bin is a real bin of an ``rfft`` grid and
    the default analysis grids cap themselves there, so refusing it would
    reject a caller's own default. The generator side is
    :func:`require_below_nyquist`.
    """
    _nyquist_check(frequency, sample_rate, caller, what, consequence,
                   "above", lambda f, nyq: f <= nyq)


def require_increasing_axis(values, label: str):
    """Refuse an empty or non-monotonic frequency/time axis, typed.

    Delegates to ``core._carrier_validate._require_strictly_increasing``, the
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
    _require_strictly_increasing(values, label)
