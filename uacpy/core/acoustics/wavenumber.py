"""Wavenumber-domain acoustics on plain arrays.

What a spectral (wavenumber-integration) result gives you once you hold
``G(k)`` — from a Scooter/SPARC ``.grn`` file, from another code, from an
analytic kernel you wrote down. The methods of
:class:`~uacpy.core.results.GreensFunction` call these same functions.
"""

import warnings
from typing import Optional

import numpy as np

from uacpy.core.exceptions import (
    ConfigurationError, NumericsWarning, ValidityWarning,
)
from uacpy.core._fourier import dtft_at
from uacpy.core._warn_frames import USER_FRAME_SKIP

__all__ = [
    'wavenumber_taper', 'hankel_transform', 'alias_period',
    'ranges_fit_alias_period', 'wavenumbers_from_phase_speeds',
    'snapshot_frequency_component',
]


def wavenumbers_from_phase_speeds(phase_speeds: np.ndarray,
                                  frequency: float) -> np.ndarray:
    """The horizontal wavenumbers ``k = 2*pi*f / c`` (rad/m) of a phase-speed grid.

    A spectral solver samples its kernel on a phase-speed grid and stores
    that grid, not ``k``: SCOOTER recomputes ``k`` per frequency from the
    same phase speeds (``scooter.f90:127``), so ``frequency`` selects the
    axis.

    Parameters
    ----------
    phase_speeds : array_like, shape ``(nk,)``
        Phase speeds (m/s).
    frequency : float
        Hz.

    Returns
    -------
    ndarray, shape ``(nk,)``
        Wavenumbers in rad/m, in the grid's order. A zero phase speed gives
        an infinite wavenumber (numpy's divide-by-zero).
    """
    c = np.asarray(phase_speeds, dtype=float)
    return 2.0 * np.pi * float(frequency) / c


def snapshot_frequency_component(
    snapshots: np.ndarray,
    times: np.ndarray,
    frequency: float,
    *,
    source_waveform: Optional[np.ndarray] = None,
) -> np.ndarray:
    """The component at ``frequency`` of a field sampled at uniform ``times``.

    ``snapshots`` holds one wavenumber-domain slab per output time on its
    first axis, as SPARC's snapshot mode writes ``Green(Itout, irz, ik)``
    (``sparc.f90:283-289``). The transform along that axis is evaluated
    **at** ``frequency`` (between DFT bins when it falls there, see
    :func:`uacpy.core._fourier.dtft_at`), not at the nearest bin: taking the
    same bin on both
    sides of a deconvolution does not cancel the leakage on a multipath
    channel (+0.06 dB / +6 deg measured at half a bin), and evaluating both
    at the frequency does.

    With ``source_waveform`` the result is the transfer function: the
    snapshot is the source pulse convolved with the medium response, so
    ``DFT(G) / DFT(s) = h(omega0)`` (Jensen, *Computational Ocean
    Acoustics*, Eq. 8.1), the unit-source Green's function a frequency-domain
    solver computes. Both transforms are the RECTANGULAR full DFT: a taper
    would break the convolution theorem and null the transient pulse, which
    lives in the first few samples, where a Hann window is ~0.

    Without it the result is the steady-tone amplitude ``2*X_k / sum(win)``
    under a Hann window: the source spectrum times the transfer function,
    ``S(omega0)*g``, whose absolute level depends on the pulse. Its phase is
    referenced to ``t = 0`` of ``times`` (``X_k = sum_n w_n G_n
    exp(-2i pi f t_n)``), so a record starting at ``t_0 != 0`` reads the same
    phase as one starting at 0. The factor 2 is the real-signal convention —
    a real tone ``A cos(2 pi f t + phi)`` returns ``A exp(i phi)`` — and is
    applied whatever the dtype: SPARC's ``Green`` is complex-typed
    (``sparc.f90:42``) but is the response to a real source pulse, and no
    source states another factor for it.

    Parameters
    ----------
    snapshots : ndarray, shape ``(nt, ...)``
        The time-sampled field, time on axis 0.
    times : array_like, shape ``(nt,)``
        Uniform output times (s); their step sets the Nyquist frequency and,
        without ``source_waveform``, their origin is the phase reference.
    frequency : float
        Hz.
    source_waveform : array_like, shape ``(nt,)``, optional
        The source pulse sampled at ``times``.

    Returns
    -------
    ndarray, shape ``snapshots.shape[1:]``

    Raises
    ------
    ConfigurationError
        Fewer than two output times, ``frequency`` above the Nyquist
        frequency, a ``source_waveform`` of another length, or one whose
        spectrum is zero at ``frequency``.
    """
    who = "snapshot_frequency_component"
    times = np.asarray(times)
    nt = len(times)
    if nt < 2:
        raise ConfigurationError(
            f"{who}: times holds {nt} output time(s); a frequency component "
            f"needs at least two to transform.",
            remediation="Pass snapshots sampled at two or more times (for a "
                        "SPARC run: a larger n_time_samples).")
    dt = float(times[1] - times[0])
    nyquist = 0.5 / dt
    if frequency > nyquist:
        raise ConfigurationError(
            f"{who}: frequency {frequency:.3f} Hz exceeds the Nyquist "
            f"{nyquist:.3f} Hz of times' {dt:.6g} s step.",
            remediation="Sample the snapshots more finely in time (for a "
                        "SPARC run: raise n_time_samples or shorten "
                        "time_max).")
    # The transform at ``frequency`` itself, off-bin, phase referenced to
    # t = 0 of ``times``.
    if source_waveform is None:
        win = np.hanning(nt)
        return (2.0 * dtft_at(snapshots, times, frequency, window=win)
                / np.sum(win))
    s_t = np.asarray(source_waveform, dtype=float)
    if s_t.shape != (nt,):
        raise ConfigurationError(
            f"{who}: source_waveform has shape {s_t.shape}; it must be the "
            f"source pulse sampled at the {nt} output times.")
    S_at_f0 = dtft_at(s_t, times, frequency)
    if S_at_f0 == 0:
        raise ConfigurationError(
            f"{who}: the source spectrum is zero at {frequency} Hz; cannot "
            "deconvolve (check the source waveform / frequency).")
    return dtft_at(snapshots, times, frequency) / S_at_f0


def _require_ascending_k(k, who: str) -> None:
    """Refuse a wavenumber grid that is not strictly increasing. The
    transform's step ``dk = (k[-1] - k[0]) / (n - 1)`` and the taper's
    grid span read the grid as ascending (fieldsco.m:107 takes the same
    step); a descending grid would flip the field's sign and turn the
    alias guards off, silently."""
    k = np.asarray(k, dtype=float)
    if k.size > 1 and not np.all(np.diff(k) > 0):
        raise ConfigurationError(
            f"{who}: the wavenumber grid k must be strictly increasing "
            f"(rad/m); got k[0] = {k[0]:g}, k[-1] = {k[-1]:g}.",
            remediation="A grid built from ascending phase speeds is "
                        "descending in k: reverse k and the spectrum's "
                        "wavenumber axis together (k[::-1], G[..., ::-1]).")


def wavenumber_taper(k: np.ndarray, frequency: float,
                     c_low: Optional[float], c_high: Optional[float]) -> np.ndarray:
    """Build a window that tapers ``G(k)`` outside ``[ω/c_high, ω/c_low]``.

    Mirrors ``fieldsco.m:taper`` — symmetric Hanning roll-offs at the
    spectrum edges, ones in the middle. Returns ``ones`` when both bounds
    are inactive. Raises :class:`ConfigurationError` when the requested
    phase-speed band has no overlap with the file's wavenumber grid —
    the taper would zero the entire spectrum.

    Parameters
    ----------
    k : ndarray
        Wavenumber grid (rad/m), strictly increasing.
    frequency : float
        Frequency (Hz).
    c_low, c_high : float or None
        Phase-speed band (m/s), ``fieldsco.m``'s ``cmin`` / ``cmax``;
        ``None`` leaves that edge untapered.
    """
    Nk = len(k)
    win = np.ones(Nk, dtype=float)
    if Nk == 0:
        return win
    _require_ascending_k(k, "phase-speed taper")

    omega = 2.0 * np.pi * frequency
    k_left = omega / c_high if (c_high is not None and c_high > 0) else None
    k_right = omega / c_low if (c_low is not None and c_low > 0) else None

    # The pass band is [ω/c_high, ω/c_low]; the grid spans phase speeds
    # [ω/k[-1], ω/k[0]]. A band that misses the grid entirely would taper
    # every sample to zero (and the roll-off construction below would build
    # a window longer than the grid).
    c_grid_lo = omega / float(k[-1])
    c_grid_hi = omega / float(k[0])
    if k_left is not None and k_right is not None and k_left > k_right:
        raise ConfigurationError(
            f"phase-speed taper: c_low ({c_low:g} m/s) exceeds c_high "
            f"({c_high:g} m/s); the pass band [ω/c_high, ω/c_low] is empty."
        )
    if (k_left is not None and k_left > k[-1]) or \
            (k_right is not None and k_right < k[0]):
        raise ConfigurationError(
            f"phase-speed taper: the requested band "
            f"(c_low={c_low!r}, c_high={c_high!r} m/s) has no overlap with the "
            f"file's phase-speed grid [{c_grid_lo:.1f}, {c_grid_hi:.1f}] m/s "
            f"at {frequency:g} Hz — the taper would zero the entire spectrum. "
            f"Widen or drop c_low/c_high — via Scooter's taper= if that is "
            f"how they were set, or directly if this transform was called "
            f"by hand. SPARC has no taper setting; a SPARC .grn reaches this "
            f"path only through GreensFunction.snapshot_to_field."
        )
    if Nk < 4:
        return win

    # np.hanning includes the window's zero endpoints where MATLAB's
    # hanning() drops them, so each roll-off differs from fieldsco.m's by a
    # one-sample shift of the taper — sub-0.1% of the pass band.
    if k_left is not None and k_left > k[0]:
        n = 2 * round((k_left - k[0]) / (k[-1] - k[0]) * Nk) + 1
        han = np.hanning(n)
        n_half = (n - 1) // 2
        win[:n_half] *= han[:n_half]

    if k_right is not None and k_right < k[-1]:
        n = 2 * round((k[-1] - k_right) / (k[-1] - k[0]) * Nk) + 1
        han = np.hanning(n)
        n_half = (n - 1) // 2
        win[-n_half:] *= han[-n_half:]

    return win


def _zero_range_mask(ranges: np.ndarray) -> np.ndarray:
    """Ranges at which the point-source ``1/√r`` spreading factor is singular.

    Mirrors the ``abs( Rr ) < realmin`` test of ``fieldsco.m:69``.
    """
    return np.abs(np.asarray(ranges, dtype=float)) < np.finfo(float).tiny


def _warn_zero_ranges(ranges: np.ndarray, source_type: str,
                      context: str = '') -> None:
    """Warn once per transform that ``r = 0`` cells come back as no-data.

    ``skip_file_prefixes`` pins the warning on the first frame outside the
    uacpy package — the caller's own code — whatever chain of model/reader
    frames sits in between. ``context`` names the producing ``.grn`` (its
    title line) in the message.
    """
    if source_type != 'point':
        return
    n_zero = int(np.count_nonzero(_zero_range_mask(ranges)))
    if n_zero:
        origin = f" (grn title {context!r})" if context else ""
        warnings.warn(
            f"Green's-function transform: {n_zero} receiver range(s) at "
            f"r = 0{origin}: the "
            "point-source Hankel transform carries a 1/sqrt(r) "
            "cylindrical-spreading factor that is singular there, so those "
            "cells are returned as NaN (no data). Move the receiver off the "
            "source axis (e.g. r = 1 m) to get a field value.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP)


#: Relative to the largest ``|k|``: how far a wavenumber grid may sit off
#: the line ``k[0] + i*dk`` and still be transformed as uniform. SPARC builds
#: its grid in single precision (``Scooter/sparc.f90:123-124``), so its
#: values carry ~1e-7·|k| of rounding; a grid uniform in phase speed instead
#: departs from a line by orders of magnitude more.
_K_GRID_RTOL = 1e-6


def hankel_transform(
    G_src: np.ndarray,
    k: np.ndarray,
    ranges: np.ndarray,
    *,
    attenuation: float,
    source_type: str = 'point',
    spectrum: str = 'positive',
) -> np.ndarray:
    """Wavenumber → range transform of ``G(k)`` at one frequency.

    Takes the depth-dependent spectrum ``G_src`` (one row per receiver depth)
    on a uniform wavenumber grid and returns the complex pressure at
    ``ranges``. The range answer is periodic with the alias period
    ``2*pi/dk`` (:func:`alias_period`): a range at or beyond it warns, since
    folded energy cannot be told from real energy afterwards (see
    :func:`ranges_fit_alias_period`), and a range beyond ``10/dk`` is
    refused, as ``fieldsco.m:109-111`` refuses it.

    A direct (trapezoidal-rule) DFT over the uniform ``k`` grid — the matrix
    product ``-G_scaled @ X`` below — not an FFT, matching ``fieldsco.m:5``
    ("This version uses the trapezoidal rule directly to do a DFT, rather
    than an FFT").

    Implements three of ``fieldsco.m``'s four source branches (its ``'H'``
    exact-Bessel branch, ``fieldsco.m:139-144``, is not exposed):

    ============  ==================================================
    source_type   Geometry
    ------------  --------------------------------------------------
    ``'point'``   cylindrical / point source (3-D), ``√(2πr)`` denom
    ``'line'``    Cartesian / line source (2-D), ``√(2π)`` denom
    ``'scaled'``  point source, cylindrical spreading removed, ``√(2π)`` denom
    ============  ==================================================

    ==============  ================================================
    spectrum        Half / full integration
    --------------  ------------------------------------------------
    ``'positive'``  positive branch only (default; recommended)
    ``'negative'``  negative branch only
    ``'both'``      both branches summed (full real-axis integral)
    ==============  ================================================

    Parameters
    ----------
    G_src : (nrd, nk) complex
    k     : (nk,) wavenumber grid (rad/m), strictly increasing
    ranges : (nr,) output ranges (m)
    attenuation : stabilising attenuation (added to k along the +i axis)
    source_type, spectrum : see table above
    """
    who = "hankel_transform"
    if source_type not in ('point', 'line', 'scaled'):
        raise ConfigurationError(
            f"{who}: source_type must be 'point', 'line', or 'scaled', "
            f"got {source_type!r}.")
    if spectrum not in ('positive', 'negative', 'both'):
        raise ConfigurationError(
            f"{who}: spectrum must be 'positive', 'negative', or 'both', "
            f"got {spectrum!r}.")

    k = np.asarray(k, dtype=float)
    # The mean step, as fieldsco.m:107 takes it ((k(end) - k(1))/(Nk - 1));
    # the trapezoidal weight dk is one number, so a grid off the line
    # k(1) + i*dk is refused rather than weighted by its first step.
    dk = float((k[-1] - k[0]) / (len(k) - 1)) if len(k) > 1 else 1.0
    off_line = (np.max(np.abs(k - (k[0] + dk * np.arange(len(k)))))
                if len(k) > 2 else 0.0)
    if off_line > _K_GRID_RTOL * float(np.max(np.abs(k))):
        raise ConfigurationError(
            f"{who}: k is not uniformly spaced: it departs from the line "
            f"k[0] + i*dk (dk = {dk:.6g} 1/m, the mean step) by up to "
            f"{off_line:.3g} 1/m, and the transform weights every sample by "
            f"one step dk.",
            remediation="Resample the spectrum onto a uniform wavenumber "
                        "grid (np.interp onto np.linspace(k[0], k[-1], n)) "
                        "before transforming it.")
    # A uniform grid running the other way passes the line test above with
    # a negative dk; the step and the guards below read it as ascending.
    _require_ascending_k(k, who)
    # fieldsco.m:109-111's own guard: the uniform-dk DFT is periodic in range
    # with period 2*pi/dk, and beyond ~10/dk the wrapped tail is amplified
    # exponentially by the exp(+attenuation*r) stabilisation compensation (+70 dB
    # measured one alias period out) — refuse rather than return garbage.
    r_max = float(np.max(np.abs(ranges))) if np.size(ranges) else 0.0
    # 10/dk, not 2*pi/dk: fieldsco's threshold is about 1.6 alias
    # periods, looser than ranges_fit_alias_period's, because the
    # exponential stabilisation compensation is what actually makes
    # the wrapped tail unusable rather than the wrap itself.
    if r_max * dk > 10.0:
        raise ConfigurationError(
            f"{who}: max range {r_max:g} m x wavenumber step "
            f"dk = {dk:g} 1/m = {r_max * dk:.3g} > 10 — the range axis "
            f"extends {r_max * dk / (2.0 * np.pi):.3g} alias periods "
            f"(2*pi/dk = {alias_period(dk):g} m) out, past fieldsco's 10/dk "
            f"limit (fieldsco.m:109-111 stops here too).",
            remediation="Increase the spectral RMax (rmax_factor on "
                        "Scooter/SPARC) so dk shrinks, or shorten the "
                        "receiver ranges.")
    # Between 2*pi/dk and 10/dk the transform runs, as fieldsco's does, but
    # every range past the alias period carries folded energy.
    if r_max * dk >= 2.0 * np.pi:
        warnings.warn(
            f"{who}: max range {r_max:g} m is at or past the alias period "
            f"2*pi/dk = {alias_period(dk):g} m; the field there includes "
            f"energy folded in from beyond it, which cannot be separated "
            f"afterwards (ranges_fit_alias_period(dk, rmax) is False). "
            f"Refine dk (a longer spectral RMax) or shorten the ranges.",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)
    ck = k + 1j * attenuation
    abs_r = np.abs(ranges)
    # Carry the kernel at the .grn's own precision. ``G`` is written complex64
    # (``scooter.f90`` declares ``Green`` COMPLEX), so promoting it to
    # complex128 adds no information to the data and doubles the largest array
    # in the transform; measured on stress cases spanning the deepest
    # cancellation, the two paths agree to within 0.03 dB. The PHASE is a
    # different matter: ``k*r`` reaches ~1e5 rad here, so ``phase`` and the
    # exponential are evaluated in double and only the RESULT is cast down.
    # That result is not unit-modulus -- it carries ``exp(attenuation*r)``, ~2.2 at
    # the far receiver here -- but it is bounded, so the cast costs relative
    # precision only.
    dt = G_src.dtype if G_src.dtype == np.complex64 else np.complex128

    if source_type == 'line':
        # Line source: no √k weighting, no phase shift, 1/√(2π).
        factor1 = np.ones_like(ck)
        factor2 = dk / np.sqrt(2.0 * np.pi) * np.ones_like(abs_r)
        phase = np.outer(ck, abs_r)
    else:
        # Point source: phase factor exp(±i(kr - π/4)) and √k weighting.
        # 'point' adds 1/√(2πr) cylindrical spreading; 'scaled' omits it.
        factor1 = np.sqrt(ck)
        if source_type == 'point':
            # 1/√r diverges at r=0. ``fieldsco.m:69`` moves a zero range to
            # 1 m; uacpy reports no-data instead, so a cell is never labelled
            # with a range the field was not evaluated at. Kraken's modal sum
            # skips the same division (``EvaluateMod.f90:71-73``), leaving a
            # bare mode sum there — masking keeps the two models' grids
            # comparable cell by cell.
            with np.errstate(divide='ignore'):
                factor2 = dk / np.sqrt(2.0 * np.pi * abs_r)
            factor2 = np.where(_zero_range_mask(abs_r), np.nan, factor2)
        else:
            factor2 = dk / np.sqrt(2.0 * np.pi) * np.ones_like(abs_r)
        phase = np.outer(ck, abs_r)
        phase -= np.pi / 4.0

    G_scaled = G_src * factor1.astype(dt, copy=False)[np.newaxis, :]

    # Build ONE kernel, in place. ``phase`` is (nk, nr) complex128 -- 1.7 GiB on
    # a 40 kHz near-field deck -- so each temporary the chain would allocate
    # costs as much again; ``out=phase`` and the in-place operators keep the
    # double-precision array plus its cast-down copy, and nothing else.
    # ``phase`` is complex because the contour offset ``attenuation`` is its
    # imaginary part, and every branch stays complex: cos of a complex
    # argument carries the exp(attenuation*r) stabilisation, exactly as
    # exp(-i.phase) + exp(+i.phase) does.
    if spectrum == 'both':
        np.cos(phase, out=phase)
        phase *= 2.0
    else:
        phase *= -1j if spectrum == 'positive' else 1j
        np.exp(phase, out=phase)
    X = phase.astype(dt, copy=False)
    del phase

    # Negate the PRODUCT, not the kernel: unary minus binds tighter than ``@``,
    # so ``-G_scaled @ X`` would copy the whole (nrd, nk) array to flip a sign
    # where ``-(G_scaled @ X)`` flips the much smaller (nrd, nr) result.
    Y = -(G_scaled @ X)

    return Y * factor2[np.newaxis, :]


def alias_period(delta_k: float) -> float:
    """Range at which a wavenumber-integration result wraps, ``2*pi/dk``.

    The transform is a sum over a uniform ``k`` grid, so its range answer is
    **periodic** with this period. Energy from beyond it does not vanish; it
    lands back inside, added to whatever is already there.

    Parameters
    ----------
    delta_k : float
        Wavenumber step of the stored spectrum (rad/m).

    Returns
    -------
    float
        Metres. ``inf`` for ``dk = 0`` (a single wavenumber does not wrap).
    """
    dk = float(delta_k)
    if not np.isfinite(dk):
        raise ConfigurationError(
            f"alias_period: delta_k must be finite (rad/m); got "
            f"{delta_k!r}.")
    if dk < 0.0:
        raise ConfigurationError(
            f"alias_period: delta_k must be non-negative (rad/m); got "
            f"{dk:g}.")
    return float('inf') if dk == 0.0 else 2.0 * np.pi / dk


def ranges_fit_alias_period(delta_k: float, rmax: float) -> bool:
    """Whether the farthest range wanted sits inside ``2*pi/dk``.

    **Necessary, not sufficient.** A grid that passes this can still be too
    coarse — refine ``dk`` until the field stops moving. A grid that fails
    it is definitely wrong, and wrong in the direction that is hardest to
    notice: folded energy is indistinguishable from real energy once it has
    landed, and it makes the field too **loud**, which reads as a physical
    result rather than an error.

    This is the question a spectral run cannot answer for you afterwards,
    which is why it is worth asking before: nothing downstream of ``G(k)``
    can separate a folded arrival from a real one.

    Parameters
    ----------
    delta_k : float
        Wavenumber step of the stored spectrum (rad/m).
    rmax : float
        Farthest receiver range wanted (m).

    Returns
    -------
    bool
        ``True`` when ``rmax < 2*pi/dk``.
    """
    r = float(rmax)
    if not np.isfinite(r) or r < 0.0:
        raise ConfigurationError(
            f"ranges_fit_alias_period: rmax must be a non-negative, finite "
            f"range in metres; got {rmax!r}.")
    return r < alias_period(delta_k)
