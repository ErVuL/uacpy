"""Shared input guard for the spectral/time-frequency estimators.

Rejects empty and NaN/Inf input with a typed ConfigurationError, so every
spectral, time-frequency and constant-Q estimator fails the same way instead of
propagating a single bad sample into an all-NaN result.
"""

import numpy as np

from uacpy.core.exceptions import ConfigurationError


def require_finite_signal(data, caller: str):
    """Validate and return a real float ndarray of a non-empty, finite signal.

    Rejects an empty signal and any NaN/Inf with a typed
    :class:`~uacpy.core.exceptions.ConfigurationError`. Does **not** reject
    complex input — Welch/STFT handle complex baseband legitimately — but a
    complex array is returned as-is for the caller's own dtype handling.
    """
    arr = np.asarray(data)
    if arr.size == 0:
        raise ConfigurationError(
            f"{caller}: data is empty; provide a non-empty signal.")
    if not np.all(np.isfinite(arr)):
        raise ConfigurationError(
            f"{caller}: data contains NaN or Inf, which would silently "
            "contaminate the estimate; clean the signal first.")
    return arr
