"""Modal / dispersion-based signal processing for shallow-water waveguides.

Tools for analysing the dispersive modal arrivals that uacpy's normal-mode
models produce: modal group velocity from the dispersion relation, and
time-warping that linearises ideal-waveguide dispersion so modes become tones
(single-receiver mode separation and source-range estimation).

References
----------
Jensen, Kuperman, Porter & Schmidt. *Computational Ocean Acoustics*, Ch. 5.
Le Touzé, G., Nicolas, B., Mars, J. & Lacoume, J. (2009). Matched
representations and filters for guided waves. *IEEE Trans. Sign. Process.*
**57**(5), 1783-1795, doi:10.1109/tsp.2009.2013907 — where the warping
transform implemented here was introduced.
Bonnel, J., Thode, A., Wright, D. & Chapman, R. (2020). Nonlinear time-warping
made simple: A step-by-step tutorial on underwater acoustic modal separation
with a single hydrophone. *JASA* **147**(3), 1897-1926,
doi:10.1121/10.0000937 — the form and the numerical recipe implemented here:
Eqs. (7), (10), (11) and (15) on p. 1907 for the operator, Eqs. (13)-(14) on
the same page for the warped grid, and Appendix C (pp. 1922-1924) for their
derivation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from uacpy.core.constants import DEFAULT_SOUND_SPEED
from uacpy.core.exceptions import ConfigurationError
from uacpy.core._finite_difference import warn_if_storage_under_resolves
from uacpy.acoustic_signal._signal_validate import (
    require_increasing_axis, require_positive_finite_scalar)


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

    Notes
    -----
    **A finer frequency grid is not always a better derivative.** The centred
    difference's truncation error falls as the frequency step squared, but a
    ``k_horizontal`` read from a model file arrives quantized — KRAKEN's
    ``.mod`` record is ``COMPLEX*8`` — and the difference then carries about
    one storage step of noise however fine the grid is, contributing
    ``spacing(k_r)/|Δk_r|`` relative. On an exact-root ideal-waveguide control,
    refining a 40-point 30-70 Hz sweep to 400 points leaves the float64 answer
    9x more accurate and the float32 one 9x worse.

    A grid whose storage floor exceeds 1e-5 is warned about. Because this
    function is handed the *whole* sweep, the warning carries a measured
    remedy rather than a prescribed one: decimating the grid raises the
    truncation error as the square of the decimation and lowers the floor in
    proportion to it, so the widest spacing whose answer still agrees with its
    doubly-decimated self to within the floor is found by walking, and the
    recommendation is one doubling back from there. Over 30 firings on five
    ideal waveguides that step was never worse than the grid in hand and 3.3x
    better at the median; where the walk cannot widen, the message says so and
    gives the test for whether refining would help instead.

    Raises
    ------
    ConfigurationError
        ``frequencies`` is not 1-D strictly increasing or has fewer than two
        samples; ``k_horizontal``'s leading axis does not match it; or a mode
        column of ``k_horizontal`` does not rise strictly with frequency —
        flat, falling throughout, or doubling back, each named separately.
        One mode's ``k_r`` rises strictly, because ``v_g`` is the
        energy-transport speed and ``d(k_r)/d(omega) = 1/v_g`` is positive
        everywhere; a flat step divides by zero and a falling one returns a
        negative speed. Note that ``v_g`` itself is *not* monotonic — it dips
        through the Airy minimum (Jensen et al., *COA*, Fig. 2.28b) — while
        ``k_r`` still climbs.
    """
    f = np.asarray(frequencies, dtype=float)
    # Complex wavenumbers carry the modal attenuation in the imaginary part;
    # the dispersion (and hence the group velocity) lives in the real part.
    kr = np.real(np.asarray(k_horizontal)).astype(float)
    if f.ndim != 1 or np.any(np.diff(f) <= 0):
        detail = (f" with {int(np.count_nonzero(np.diff(f) <= 0))} "
                  "non-increasing step(s)") if f.ndim == 1 else ""
        raise ConfigurationError(
            "modal_group_velocity: frequencies must be 1-D increasing; got "
            f"shape {f.shape}{detail}")
    require_increasing_axis(f, "modal_group_velocity: frequencies")
    if f.size < 2:
        # np.gradient below needs two samples per axis; one reached it as
        # "Shape of array too small to calculate a numerical gradient",
        # naming no input the caller supplied.
        raise ConfigurationError(
            f"modal_group_velocity: frequencies needs at least 2 samples — "
            f"the group velocity is a numerical derivative d(omega)/d(kr) "
            f"over the axis; got {f.size}.")
    omega = 2.0 * np.pi * f
    if kr.shape[0] != f.size:
        raise ConfigurationError(
            "modal_group_velocity: k_horizontal axis 0 must match freqs; got "
            f"k_horizontal shape {kr.shape} and {f.size} frequencies")
    # Both gradients are taken in index space (unit spacing). Their ratio is
    # (domega/di)/(dkr/di) = domega/dkr, so no frequency spacing is needed and
    # a non-uniform `frequencies` grid is handled without extra care.
    dkr = np.gradient(kr) if kr.ndim == 1 else np.gradient(kr, axis=0)
    # A zero step in kr is the one value the division cannot survive: it
    # returns inf, announced by nothing but numpy's own "divide by zero".
    # Tested exactly rather than against a tolerance, because exact zero is
    # what produces the inf and a *small* step is the physical near-cutoff
    # case (kr flattens in range, not in frequency) that must keep working.
    flat = dkr == 0.0
    if np.any(flat):
        by_frequency = np.any(flat.reshape(flat.shape[0], -1), axis=1)
        first = int(np.flatnonzero(by_frequency)[0])
        raise ConfigurationError(
            f"modal_group_velocity: k_horizontal is flat in frequency at "
            f"{int(np.count_nonzero(by_frequency))} of {f.size} frequency "
            f"sample(s), first at index {first} ({f[first]:g} Hz). The group "
            f"velocity is d(omega)/d(kr), so a flat step divides by zero and "
            f"comes back inf. A propagating mode's horizontal wavenumber "
            f"rises strictly with frequency — check this is one mode's "
            f"dispersion curve sampled across frequency, not a constant "
            f"or several modes "
            f"stacked on the frequency axis.")
    # The rest of the contract: k_r must RISE with frequency. v_g is the speed
    # energy travels at, so it is positive and no larger than the fastest
    # medium speed, and d(kr)/d(omega) = 1/v_g is therefore positive at every
    # frequency. v_g itself is *not* monotonic — it dips through the Airy
    # minimum (Jensen et al., COA Sect. 2.4.4.4 and Fig. 2.28b) — but k_r
    # never falls. A falling step hands back a negative speed, which nothing
    # downstream reads as an error: it is a travel-time denominator.
    #
    # The two ways a column can fall want different remedies, so they are
    # reported separately: a column that falls *throughout* is a curve stored
    # the wrong way round, while one that both rises and falls is a mixed or
    # mode-shifted set. Each mode column is judged on its own.
    columns = dkr.reshape(dkr.shape[0], -1)
    rising = np.any(columns > 0, axis=0)
    falling = np.any(columns < 0, axis=0)
    turning = rising & falling
    if np.any(turning):
        col = columns[:, int(np.flatnonzero(turning)[0])]
        turn = int(np.flatnonzero(np.sign(col[1:]) != np.sign(col[:-1]))[0]) + 1
        raise ConfigurationError(
            f"modal_group_velocity: k_horizontal doubles back in frequency in "
            f"{int(np.count_nonzero(turning))} of {columns.shape[1]} mode "
            f"column(s), first turning at index {turn} ({f[turn]:g} Hz). The "
            f"group velocity is d(omega)/d(kr), the speed energy travels at, "
            f"so a step that reverses sign hands back a negative speed there "
            f"while the rest of the curve looks ordinary. A propagating mode's "
            f"horizontal wavenumber rises strictly with frequency — its phase "
            f"speed is bounded by the medium sound speeds — so a curve that "
            f"turns is not one mode's dispersion: check for several modes "
            f"stacked on the frequency axis, or a mode index that shifted "
            f"where the model's mode list changed length across a cutoff.")
    # Reached only once ``turning`` is empty, so no column mixes the two signs
    # and no step is zero — a column that falls at all therefore falls
    # throughout. That is why this branch has to stay BELOW the one above: run
    # first, it would claim a doubling-back curve falls all the way and send
    # the caller to reverse an axis that is not the problem.
    if np.any(falling):
        raise ConfigurationError(
            f"modal_group_velocity: k_horizontal falls with frequency "
            f"throughout {int(np.count_nonzero(falling))} of "
            f"{columns.shape[1]} mode column(s) — over "
            f"{f[0]:g}-{f[-1]:g} Hz. A propagating mode's horizontal "
            f"wavenumber rises strictly with frequency, because the group "
            f"velocity d(omega)/d(kr) is the speed energy travels at and is "
            f"positive; a falling curve makes every group velocity in the "
            f"column negative. The usual cause is a mode set stored against a "
            f"descending frequency axis: reverse k_horizontal (and "
            f"frequencies with it) so both run low to high.")
    domega = np.gradient(omega)
    v_g = domega / dkr if kr.ndim == 1 else domega[:, None] / dkr
    # The floor is measured on the step between neighbouring frequencies, not
    # on np.gradient's centred half-step, so the same expression covers both
    # this function and Modes.compute_group_velocity, which differences one
    # step directly. dkr is aligned with the FIRST of each pair.
    step = np.diff(kr, axis=0)
    warn_if_storage_under_resolves(
        kr[:-1], step, v_g[:-1], "modal_group_velocity", grid=(omega, kr))
    return v_g


#: Elements of the dense sinc matrix :func:`_resample_uniform` builds at a
#: time. Whittaker-Shannon is O(N·K), so the whole matrix for a long record
#: does not fit in memory; 2e6 doubles is 16 MB per block, and the block loop
#: costs nothing next to the matrix product itself.
_SINC_BLOCK_ELEMENTS = 2_000_000

_INTERPOLATIONS = ('linear', 'sinc')


def _resample_uniform(values, t0: float, dt: float, t_query, *, method: str,
                      who: str):
    """``values``, sampled uniformly from ``t0`` at step ``dt``, read at
    ``t_query``.

    ``'linear'`` is :func:`numpy.interp`. ``'sinc'`` is the
    Whittaker-Shannon interpolation of Bonnel et al. (2020) Eq. (C12),
    ``y(t) = sum_n y[n] sinc(t f_s - n)``, which the paper advises over linear
    because linear "sometimes creates high frequency artifacts in the warped
    signal". It is exact for a signal band-limited to the grid it is read
    from — and faithful to the aliased content of one that is not, which is
    why it is not unconditionally better (see :func:`warp_signal`).
    """
    if method not in _INTERPOLATIONS:
        raise ConfigurationError(
            f"{who}: interpolation must be one of "
            f"{', '.join(repr(m) for m in _INTERPOLATIONS)}; got {method!r}.")
    values = np.asarray(values, dtype=float)
    t_query = np.asarray(t_query, dtype=float)
    if method == 'linear':
        t_src = t0 + np.arange(values.size) * dt
        return np.interp(t_query, t_src, values)
    # ``np.sinc`` is the normalised sinc(x) = sin(pi x)/(pi x), which is the
    # kernel Eq. (C12) is written in: the argument is the query's position in
    # units of the source sample index.
    index = (t_query - t0) / dt
    n_src = np.arange(values.size)
    out = np.empty(index.size, dtype=float)
    block = max(1, _SINC_BLOCK_ELEMENTS // max(values.size, 1))
    for start in range(0, index.size, block):
        stop = min(start + block, index.size)
        out[start:stop] = np.sinc(index[start:stop, None] - n_src) @ values
    return out


def _prescribed_warped_length(n: int, fs: float, t_r: float) -> int:
    """``K`` of Bonnel et al. (2020) Eq. (14), for a record of ``n`` samples.

    Eq. (13) sets the warped sampling rate ``f_s^h = 2/dt_N`` with
    ``dt_N = (1/f_s)·t_max/h^-1(t_max)``, i.e.
    ``f_s^h = 2·f_s·h^-1(t_max)/t_max``; Eq. (14) then takes
    ``K = ceil([h^-1(t_max) - h^-1(t_min)]·f_s^h)``, and ``h^-1(t_min) = 0``
    here because the record starts at ``t_r`` (App. C 1, C 2.b).

    The factor 2 is the paper's margin over its own Nyquist bound Eq. (C8),
    ``f_s^h > [h^-1(t_max)/t_max]·f_s``.
    """
    t_max = t_r + (n - 1) / fs
    span = float(np.sqrt(max(t_max ** 2 - t_r ** 2, 0.0)))
    if span <= 0.0:
        return 2
    fs_h = 2.0 * fs * span / t_max
    return max(2, int(np.ceil(span * fs_h)))


def warp_signal(signal, sample_rate: float, range_m: float,
                c: float = DEFAULT_SOUND_SPEED, *,
                oversample: Optional[float] = None,
                interpolation: str = 'linear'):
    """Warp an impulsive shallow-water arrival to linearise ideal-waveguide dispersion.

    Maps original (reduced) time ``t`` to warped time ``t_w = sqrt(t^2 - t_r^2)``
    with ``t_r = range/c``, so each ideal-waveguide mode collapses to a single
    warped frequency. ``signal`` is assumed to start at the direct-wave arrival
    ``t_r``.

    The transform was introduced by Le Touzé et al. (2009); the form and the
    numerical recipe implemented here are Bonnel, Thode, Wright & Chapman
    (2020), *JASA* **147**(3) 1897-1926, p. 1907 — the resampling
    ``h(t) = sqrt(t^2 + t_r^2)`` of Eq. (10), whose inverse
    ``h^-1(t) = sqrt(t^2 - t_r^2)`` (Eq. 11) is the warped axis returned here,
    applied with the ``sqrt(|h'(t)|)`` weight of Eq. (7) that "ensures energy
    conservation" — written discretely as ``sqrt(t_w/h(t_w))`` in Eq. (15),
    which is the form below.

    The operator is derived there for the ideal isovelocity waveguide. On a
    real profile the warped modes are, in that paper's words, "not the
    theoretically predicted pure tones ... but instead are tilted and slightly
    curved" — still separable, but no longer single warped frequencies.

    Parameters
    ----------
    range_m, c : float
        **Trial** parameters, not measurements. They enter only through
        ``t_r = range_m/c``, and warping is "only weakly sensitive to the
        choice of ``t_r``": the paper states that "it is not required to know
        the range nor the water sound speed to apply warping", and warps every
        signal in the tutorial — experimental ones included — at r = 10 km,
        c = 1500 m/s while the true ranges are 5-15 km. Warp followed by
        inverse warp cancels ``t_r`` exactly, so a downstream localisation is
        independent of the value used. What the result *is* sensitive to is
        the **time origin** of ``signal``: see Notes.
    oversample : float, optional
        Length of the warped axis as a multiple of the input length;
        fractional factors are honoured (the length is rounded), and a factor
        below 1 is refused rather than clamped.

        ``None`` (the default) takes the prescription of Eqs. (13)-(14)
        instead: a warped rate ``f_s^h = 2/dt_N`` with
        ``dt_N = (1/f_s)·t_max/h^-1(t_max)``, and
        ``K = ceil(h^-1(t_max)·f_s^h)`` samples. In this function's terms that
        is a factor ``2·(1 + t_r/t_max)``, always in (2, 4), and the factor 2
        in it is the paper's own margin over its Nyquist bound Eq. (C8),
        ``f_s^h > [h^-1(t_max)/t_max]·f_s``.

        That bound is why a fixed factor of 1 is not merely a coarse choice:
        the map is expansive (``dt_w/dt = t/t_w > 1``), so ``oversample=1``
        puts the warped grid **below** Eq. (C8) by exactly ``(1 + t_r/t_max)``
        for every range and rate, and the round trip ``warp -> unwarp`` loses
        the top of the band. Measured on this implementation, relative
        round-trip error on white noise (broadband to Nyquist, so it loses the
        most) over sample rates 2-10 kHz and ranges 0.1-20 km: **46.5-60.5 %
        at ``oversample=1``, 15.6-20.6 % at the prescription**, 5.8-7.9 % at a
        fixed 8. On the band-limited transient the warp exists for — the 40 + 120 Hz
        Hann-windowed pair at 10 kHz over 500 m used by
        ``tests/test_modal.py`` — 0.067 %, 0.019 % and 0.0079 %. At a fixed
        factor the error roughly halves with each doubling.
    interpolation : {'linear', 'sinc'}, optional
        How ``signal`` is read at the off-grid times the warp asks for.
        ``'linear'`` (the default) is :func:`numpy.interp`. ``'sinc'`` is the
        Whittaker-Shannon interpolation of Eq. (C12),
        ``y(t) = sum_n y[n] sinc(t f_s - n)``, which the paper advises because
        linear interpolation "sometimes creates high frequency artifacts in
        the warped signal".

        Two things make it an opt-in rather than the default. It is **only**
        an improvement on a grid that satisfies Eq. (C8): at ``oversample=1``
        it makes the white-noise round trip *worse* — 65 % against linear's
        58 % at 10 kHz over 1 km — because an exact reconstruction faithfully
        reproduces the aliased content that linear interpolation was
        accidentally smoothing away. And it is an O(N·K) dense kernel,
        quadratic in record length: of order 1500-1700x
        :func:`numpy.interp`'s cost, measured at 380-430 ms against 0.24-0.26
        ms for ``n=2048``, ``K=7227`` over two runs. Paired with the default
        grid it makes the round trip essentially exact — 0.0036-0.0067 % on
        the same white-noise grid, and numerically exact on the band-limited
        transient — and it must be passed to :func:`unwarp_signal` as well to
        get that.

    Returns
    -------
    warped : ndarray
        The resampled signal on the warped time grid, Jacobian-weighted so the
        warp is energy-preserving.
    t_warp : ndarray
        The warped time axis (s) ``warped`` lives on, spanning
        ``[0, sqrt(t_end**2 - t_r**2)]`` — ``n * oversample`` samples, or the
        ``K`` of Eq. (14) when ``oversample`` is ``None``.

    Notes
    -----
    **The time origin is the sensitive parameter.** The paper's guidance, and
    the most useful thing it has to say to a caller here: pick an origin as
    close to ``r/c_w`` as possible by an iterative process — read the arrival
    time of the highest frequencies off the original spectrogram (they
    disperse least, so they arrive nearest ``t_r``), warp, inspect the warped
    spectrogram, iterate. The failure is asymmetric. Too early and "the modes
    are definitely not horizontal tones, but span a wider bandwidth across the
    warped spectrum", overlapping and interfering. Too late and "the modes
    become virtually horizontal" and separate *better* — "however, this
    improved separation comes at a price: mode 1 has vanished". In a real
    waveguide energy also arrives *before* ``r/c_w``, through the seabed, and
    ``c_w`` is not uniquely defined unless the water column is isovelocity.


    **Signal first, axis second — deliberately, and against the package's
    usual axis-first convention** (stated at
    :func:`uacpy.acoustic_signal.synthesize_noise_from_psd`, and followed by
    every other function on the ``acoustic_signal`` public surface). This pair
    is ordered to feed :func:`unwarp_signal`, whose first two parameters are
    ``(warped, t_warp)`` in exactly this order, so the round trip is
    ``unwarp_signal(*warp_signal(x, fs, r, oversample=8), fs, r)``. Swapping
    this tuple to match the convention would silently invert that call:
    ``t_warp`` and ``warped``
    have the same length, so the argument-shape checks downstream cannot tell
    them apart and the error surfaces only as a wrong answer.
    """
    x = np.asarray(signal, dtype=float)
    fs = require_positive_finite_scalar(sample_rate, "warp_signal",
                                        "sample_rate", " Hz")
    n = x.size
    range_m = require_positive_finite_scalar(range_m, "warp_signal",
                                             "range_m", " m")
    c = require_positive_finite_scalar(c, "warp_signal", "c", " m/s")
    t_r = range_m / c
    # The record is taken to start at the direct arrival: t_min = [t_r]+, the
    # convention of App. C 1, under which the warped domain is Eq. (C1)'s
    # [0, sqrt(t_max^2 - t_r^2)] — the axis returned below.
    t = t_r + np.arange(n) / fs
    t_w = np.sqrt(np.maximum(t ** 2 - t_r ** 2, 0.0))
    if oversample is None:
        n_w = _prescribed_warped_length(n, fs, t_r)
    else:
        # Scale first and round after, so a fractional factor lengthens the
        # axis instead of truncating to the integer below it (int(1.5) == 1
        # makes the accuracy knob a no-op for every non-integer value). A
        # factor below 1 would shorten the warped axis, which is the opposite
        # of what the argument is for, so it is refused rather than clamped.
        try:
            os_factor = float(oversample)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"warp_signal: oversample must be a number >= 1; got "
                f"{oversample!r}.") from exc
        if not (os_factor >= 1.0):
            raise ConfigurationError(
                f"warp_signal: oversample must be >= 1 — it is the warped "
                f"axis' length as a multiple of the input's, and the warp is "
                f"expansive; got {oversample!r}.")
        n_w = max(2, int(round(n * os_factor)))
    tw_axis = np.linspace(t_w[0], t_w[-1], n_w)
    t_orig = np.sqrt(tw_axis ** 2 + t_r ** 2)
    warped = _resample_uniform(x, t[0], 1.0 / fs, t_orig,
                               method=interpolation, who='warp_signal')
    # Unitary Jacobian weighting — ``sqrt(|h'|)``, Bonnel et al. (2020)
    # Eq. (7). App. C 2.d works it through to the form used here: "Because
    # h'(t) = t/h(t), this factor is given as sqrt(|h'[k/f_s^h]|) =
    # sqrt(t_k/h(t_k)) for the kth sample of the warped signal", which is
    # Eq. (15)'s weight. So with h(t_w) = t = sqrt(t_w^2 + t_r^2) the
    # energy-preserving weight is sqrt(t_w / t) —
    # verified numerically against np.gradient(t, t_w), and by the resulting
    # E_warp/E_in being range-INDEPENDENT (the reciprocal inflates it by ~30x
    # at 20 km and grows with range). t_w = 0 at the direct arrival t = t_r, so
    # the numerator is floored at one sample.
    warped = warped * np.sqrt(np.maximum(tw_axis, 1.0 / fs) / t_orig)
    return warped, tw_axis


def unwarp_signal(warped, t_warp, sample_rate: float, range_m: float,
                  c: float = DEFAULT_SOUND_SPEED, *,
                  interpolation: str = 'linear'):
    """Inverse of :func:`warp_signal`; returns ``(t, signal)`` on the original grid.

    ``(warped, t_warp)`` are taken in the order :func:`warp_signal` returns
    them, so ``unwarp_signal(*warp_signal(x, fs, r), fs, r)`` round-trips. The
    **return** here is axis-first ``(t, signal)``, the package convention,
    because nothing consumes it positionally.

    Bonnel et al. (2020) Eq. (16), ``y_u[n] = sqrt(t_n/h^-1(t_n)) y_w[h^-1(t_n)]``,
    with the output rate and sample count "already known: they are the same as
    for the original signal" — which is what the grid below reconstructs from
    the warped axis' own extent.

    ``interpolation`` is :func:`warp_signal`'s, and must match the forward
    call for the round trip to be exact: ``'sinc'`` (Eq. C12) buys its
    accuracy only when both directions use it.
    """
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
    dt_w = (tw[-1] - tw[0]) / max(tw.size - 1, 1)
    signal = _resample_uniform(w_unweighted, tw[0], dt_w, t_w_of_t,
                               method=interpolation, who='unwarp_signal')
    return t, signal
