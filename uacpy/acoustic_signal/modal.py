"""Modal / dispersion-based signal processing for shallow-water waveguides.

Tools for analysing the dispersive modal arrivals that uacpy's normal-mode
models produce: modal group velocity from the dispersion relation, and
time-warping that linearises ideal-waveguide dispersion so modes become tones
(single-receiver mode separation and source-range estimation).

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
        ``(n_freq, n_modes)`` for several. May be complex (KRAKENC's lossy
        modes): the group velocity is then ``d(omega)/d(Re k_r)`` — for
        weakly attenuated modes the propagation speed follows the real part
        of the wavenumber, while ``Im(k_r)`` is the modal attenuation and
        controls amplitude decay, not travel time (Jensen et al., *COA*,
        Sects. 2.4.5 and 5.9.2).

    Returns
    -------
    ndarray
        Group velocity (m/s), same shape as ``k_horizontal``.
    """
    f = np.asarray(frequencies, dtype=float)
    # Complex wavenumbers carry the modal attenuation in the imaginary part;
    # the dispersion (and hence the group velocity) lives in the real part.
    kr = np.real(np.asarray(k_horizontal)).astype(float)
    if f.ndim != 1 or np.any(np.diff(f) <= 0):
        raise ConfigurationError("modal_group_velocity: frequencies must be 1-D increasing")
    omega = 2.0 * np.pi * f
    if kr.shape[0] != f.size:
        raise ConfigurationError("modal_group_velocity: k_horizontal axis 0 must match freqs")
    # Both gradients are taken in index space (unit spacing). Their ratio is
    # (domega/di)/(dkr/di) = domega/dkr, so no frequency spacing is needed and
    # a non-uniform `frequencies` grid is handled without extra care.
    domega = np.gradient(omega)
    if kr.ndim == 1:
        return domega / np.gradient(kr)
    return domega[:, None] / np.gradient(kr, axis=0)


def warp_signal(signal, sample_rate: float, range_m: float,
                c: float = DEFAULT_SOUND_SPEED, *, oversample: int = 1):
    """Warp an impulsive shallow-water arrival to linearise ideal-waveguide dispersion.

    Maps original (reduced) time ``t`` to warped time ``t_w = sqrt(t^2 - t_r^2)``
    with ``t_r = range/c``, so each ideal-waveguide mode collapses to a single
    warped frequency (Bonnel et al. 2013). ``signal`` is assumed to start at the
    direct-wave arrival ``t_r``.

    Returns ``(warped, t_warp)`` — the resampled signal and its warped time axis.
    Invert with :func:`unwarp_signal`.

    Parameters
    ----------
    oversample : int, optional
        Length of the warped axis as a multiple of the input length. The map is
        expansive (``dt_w/dt = t/t_w > 1``), so at ``oversample=1`` the warped
        grid is coarser than the original in warped time and the round trip
        ``warp -> unwarp`` is **lossy** — of order 50 % relative error, and the
        error halves with each doubling of this factor. The default keeps the
        warped axis the same length as the input; raise it when the round trip
        matters. No published
        prescription for the factor exists in the corpus here, so none is
        imposed.
    """
    x = np.asarray(signal, dtype=float)
    fs = float(sample_rate)
    n = x.size
    t_r = float(range_m) / float(c)
    t = t_r + np.arange(n) / fs
    t_w = np.sqrt(np.maximum(t ** 2 - t_r ** 2, 0.0))
    n_w = int(n * max(1, int(oversample)))
    tw_axis = np.linspace(t_w[0], t_w[-1], n_w)
    t_orig = np.sqrt(tw_axis ** 2 + t_r ** 2)
    warped = np.interp(t_orig, t, x)
    # Unitary Jacobian weighting. With t = sqrt(t_w^2 + t_r^2),
    # dt/dt_w = t_w / t, so the energy-preserving weight is sqrt(t_w / t) —
    # verified numerically against np.gradient(t, t_w), and by the resulting
    # E_warp/E_in being range-INDEPENDENT (the reciprocal inflates it by ~30x
    # at 20 km and grows with range). t_w = 0 at the direct arrival t = t_r, so
    # the numerator is floored at one sample.
    warped = warped * np.sqrt(np.maximum(tw_axis, 1.0 / fs) / t_orig)
    return warped, tw_axis


def unwarp_signal(warped, t_warp, sample_rate: float, range_m: float,
                  c: float = DEFAULT_SOUND_SPEED):
    """Inverse of :func:`warp_signal`; returns ``(t, signal)`` on the original grid."""
    w = np.asarray(warped, dtype=float)
    tw = np.asarray(t_warp, dtype=float)
    fs = float(sample_rate)
    t_r = float(range_m) / float(c)
    # The output grid follows the warped axis' own extent, not ``w.size``:
    # ``warp_signal(oversample=k)`` returns k times as many samples over the
    # same warped span, and reading the length off the array would unwarp onto
    # a record k times too long, the tail of it extrapolated.
    t_end = float(np.sqrt(tw[-1] ** 2 + t_r ** 2))
    n = max(2, int(round((t_end - t_r) * fs)) + 1)
    t = t_r + np.arange(n) / fs
    t_w_of_t = np.sqrt(np.maximum(t ** 2 - t_r ** 2, 0.0))
    # Divide out the forward weight sqrt(t_w / t) applied by ``warp_signal``.
    w_unweighted = w / np.sqrt(np.maximum(tw, 1.0 / fs)
                               / np.sqrt(tw ** 2 + t_r ** 2))
    signal = np.interp(t_w_of_t, tw, w_unweighted)
    return t, signal
