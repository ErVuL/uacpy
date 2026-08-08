"""Modal / dispersion-based signal processing for shallow-water waveguides.

Tools for analysing the dispersive modal arrivals that uacpy's normal-mode
models produce: modal group velocity from the dispersion relation, the
waveguide invariant, and time-warping that linearises ideal-waveguide
dispersion so modes become tones (single-receiver mode separation and
source-range estimation).

References
----------
Jensen, Kuperman, Porter & Schmidt. *Computational Ocean Acoustics*, Ch. 5.
Bonnel, J. et al. (2013). Range estimation using time-warping. JASA 134(2).
"""

from __future__ import annotations

import numpy as np

from uacpy.core.constants import DEFAULT_SOUND_SPEED
from uacpy.core.exceptions import ConfigurationError


def modal_group_velocity(frequencies, k_horizontal):
    """Group velocity ``v_g = d(omega)/d(k_r)`` per mode from the dispersion.

    Parameters
    ----------
    frequencies : 1-D array
        Frequencies (Hz), strictly increasing.
    k_horizontal : array
        Horizontal wavenumber (rad/m). Shape ``(n_freq,)`` for one mode or
        ``(n_freq, n_modes)`` for several.

    Returns
    -------
    ndarray
        Group velocity (m/s), same shape as ``k_horizontal``.
    """
    f = np.asarray(frequencies, dtype=float)
    kr = np.asarray(k_horizontal, dtype=float)
    if f.ndim != 1 or np.any(np.diff(f) <= 0):
        raise ConfigurationError("modal_group_velocity: frequencies must be 1-D increasing")
    omega = 2.0 * np.pi * f
    if kr.shape[0] != f.size:
        raise ConfigurationError("modal_group_velocity: k_horizontal axis 0 must match freqs")
    domega = np.gradient(omega)
    if kr.ndim == 1:
        return domega / np.gradient(kr)
    return domega[:, None] / np.gradient(kr, axis=0)


def warp_signal(signal, sample_rate: float, range_m: float,
                c: float = DEFAULT_SOUND_SPEED):
    """Warp an impulsive shallow-water arrival to linearise ideal-waveguide dispersion.

    Maps original (reduced) time ``t`` to warped time ``t_w = sqrt(t^2 - t_r^2)``
    with ``t_r = range/c``, so each ideal-waveguide mode collapses to a single
    warped frequency (Bonnel et al. 2013). ``signal`` is assumed to start at the
    direct-wave arrival ``t_r``.

    Returns ``(warped, t_warp)`` — the resampled signal and its warped time axis.
    Invert with :func:`unwarp_signal`.
    """
    x = np.asarray(signal, dtype=float)
    fs = float(sample_rate)
    n = x.size
    t_r = float(range_m) / float(c)
    t = t_r + np.arange(n) / fs
    t_w = np.sqrt(np.maximum(t ** 2 - t_r ** 2, 0.0))
    n_w = n
    tw_axis = np.linspace(t_w[0], t_w[-1], n_w)
    t_orig = np.sqrt(tw_axis ** 2 + t_r ** 2)
    warped = np.interp(t_orig, t, x)
    # Unitary Jacobian weighting so energy and the inverse are well-behaved.
    warped = warped * np.sqrt(t_orig / np.maximum(tw_axis, 1.0 / fs))
    return warped, tw_axis


def unwarp_signal(warped, t_warp, sample_rate: float, range_m: float,
                  c: float = DEFAULT_SOUND_SPEED):
    """Inverse of :func:`warp_signal`; returns ``(t, signal)`` on the original grid."""
    w = np.asarray(warped, dtype=float)
    tw = np.asarray(t_warp, dtype=float)
    fs = float(sample_rate)
    t_r = float(range_m) / float(c)
    n = w.size
    t = t_r + np.arange(n) / fs
    t_w_of_t = np.sqrt(np.maximum(t ** 2 - t_r ** 2, 0.0))
    w_unweighted = w / np.sqrt(np.maximum(np.sqrt(tw ** 2 + t_r ** 2), 1.0 / fs)
                              / np.maximum(tw, 1.0 / fs))
    signal = np.interp(t_w_of_t, tw, w_unweighted)
    return t, signal
