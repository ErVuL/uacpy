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


def _dft_row(G: np.ndarray, f_idx: float,
             win: Optional[np.ndarray] = None) -> np.ndarray:
    """Row ``f_idx`` of ``np.fft.fft(G, axis=0)``, without the other rows.

    ``f_idx`` may be fractional: the contraction is then the transform
    evaluated at ``f_idx / nt`` cycles per sample, between the DFT bins —
    what :func:`snapshot_frequency_component` uses to evaluate at the
    frequency itself rather than at the nearest bin.

    ``np.fft.fft(cube, axis=0)[f_idx]`` transforms all ``nt`` frequencies and
    keeps one — measured 5.1x the cube in peak RSS, which is ~17 GB on a
    512 x 200 x 4096 snapshot, to produce an (nrd, nk) slab. That row is a
    contraction of ``G`` against ``exp(-2i pi f_idx n / nt)``, so it costs the
    slab plus one pass over the cube instead.

    ``win`` (a real per-sample taper) folds into the kernel. Mind what numpy
    does with it today: ``G * win[:, None, None]`` promotes a complex64 cube
    to complex128, so that FFT runs in DOUBLE precision and the windowed
    branch is accurate to ~1e-16 rather than the ~1e-7 of a single-precision
    transform. Contracting a complex128 kernel against the whole cube would
    promote it the same way and hand most of the memory back (3.2x the cube
    against 1.6x), so the windowed path casts and accumulates in complex128
    over chunks of the last axis: same accuracy, bounded scratch.
    """
    nt = G.shape[0]
    phase = np.exp(-2j * np.pi * float(f_idx) * np.arange(nt) / nt)
    if win is None:
        # Mirror numpy's own FFT precision rule so the row keeps the dtype the
        # full transform gave it: single in, single out; anything else double.
        single = np.result_type(G.dtype, np.complex64) == np.complex64
        return np.tensordot(
            phase.astype(np.complex64 if single else np.complex128), G, axes=1)
    kernel = phase * np.asarray(win, dtype=np.float64)
    out = np.empty(G.shape[1:], dtype=np.complex128)
    # ~4e6 complex128 elements (64 MB) of cast cube per block — the scratch
    # budget the synthesis chunking in acoustic_signal/_synthesis.py
    # (synthesize_cells) works to.
    per_column = max(int(np.prod(G.shape[:-1])), 1)
    step = max(1, 4_000_000 // per_column)
    for a in range(0, G.shape[-1], step):
        out[..., a:a + step] = np.tensordot(
            kernel, G[..., a:a + step].astype(np.complex128), axes=1)
    return out


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
    :func:`_dft_row`), not at the nearest bin: taking the same bin on both
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
    ``S(omega0)*g``, whose absolute level depends on the pulse.

    Parameters
    ----------
    snapshots : ndarray, shape ``(nt, ...)``
        The time-sampled field, time on axis 0.
    times : array_like, shape ``(nt,)``
        Uniform output times (s); their step sets the Nyquist frequency.
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
            "SPARC snapshot has nt<2 — cannot extract a frequency component "
            "via time-FFT. Use a larger n_time_samples."
        )
    dt = float(times[1] - times[0])
    nyquist = 0.5 / dt
    if frequency > nyquist:
        raise ConfigurationError(
            f"Source frequency {frequency:.3f} Hz exceeds the snapshot's "
            f"Nyquist {nyquist:.3f} Hz; reduce dt by raising n_time_samples or "
            "shortening time_max."
        )
    # The bin position of ``frequency`` itself: fftfreq[k] = k / (nt·dt),
    # so a fractional k evaluates the transform at the frequency, off-bin.
    bin_at_f0 = frequency * nt * dt
    if source_waveform is None:
        win = np.hanning(nt)
        return 2.0 * _dft_row(snapshots, bin_at_f0, win) / np.sum(win)
    s_t = np.asarray(source_waveform, dtype=float)
    if s_t.shape != (nt,):
        raise ConfigurationError(
            f"{who}: source_waveform has shape {s_t.shape}; it must be the "
            f"source pulse sampled at the {nt} output times.")
    S_at_f0 = _dft_row(s_t, bin_at_f0)
    if S_at_f0 == 0:
        raise ConfigurationError(
            f"{who}: the source spectrum is zero at {frequency} Hz; cannot "
            "deconvolve (check the source waveform / frequency).")
    return _dft_row(snapshots, bin_at_f0) / S_at_f0


def wavenumber_taper(k: np.ndarray, frequency: float,
                     cmin: Optional[float], cmax: Optional[float]) -> np.ndarray:
    """Build a window that tapers ``G(k)`` outside ``[ω/cmax, ω/cmin]``.

    Mirrors ``fieldsco.m:taper`` — symmetric Hanning roll-offs at the
    spectrum edges, ones in the middle. Returns ``ones`` when both bounds
    are inactive. Raises :class:`ConfigurationError` when the requested
    phase-speed band has no overlap with the file's wavenumber grid —
    the taper would zero the entire spectrum.

    Parameters
    ----------
    k : ndarray
        Wavenumber grid (rad/m).
    frequency : float
        Frequency (Hz).
    cmin, cmax : float or None
        Phase-speed band (m/s); ``None`` leaves that edge untapered.
    """
    Nk = len(k)
    win = np.ones(Nk, dtype=float)
    if Nk == 0:
        return win

    omega = 2.0 * np.pi * frequency
    k_left = omega / cmax if (cmax is not None and cmax > 0) else None
    k_right = omega / cmin if (cmin is not None and cmin > 0) else None

    # The pass band is [ω/cmax, ω/cmin]; the grid spans phase speeds
    # [ω/k[-1], ω/k[0]]. A band that misses the grid entirely would taper
    # every sample to zero (and the roll-off construction below would build
    # a window longer than the grid).
    c_grid_lo = omega / float(k[-1])
    c_grid_hi = omega / float(k[0])
    if k_left is not None and k_right is not None and k_left > k_right:
        raise ConfigurationError(
            f"phase-speed taper: cmin ({cmin:g} m/s) exceeds cmax "
            f"({cmax:g} m/s); the pass band [ω/cmax, ω/cmin] is empty."
        )
    if (k_left is not None and k_left > k[-1]) or \
            (k_right is not None and k_right < k[0]):
        raise ConfigurationError(
            f"phase-speed taper: the requested band "
            f"(cmin={cmin!r}, cmax={cmax!r} m/s) has no overlap with the "
            f"file's phase-speed grid [{c_grid_lo:.1f}, {c_grid_hi:.1f}] m/s "
            f"at {frequency:g} Hz — the taper would zero the entire spectrum. "
            f"Widen or drop cmin/cmax — via Scooter's taper= if that is "
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
    k     : (nk,) wavenumber grid
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

    dk = float(k[1] - k[0]) if len(k) > 1 else 1.0
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
            f"afterwards (ranges_fit_alias_period(dk, rmax_m) is False). "
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


def ranges_fit_alias_period(delta_k: float, rmax_m: float) -> bool:
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
    rmax_m : float
        Farthest receiver range wanted (m).

    Returns
    -------
    bool
        ``True`` when ``rmax_m < 2*pi/dk``.
    """
    r = float(rmax_m)
    if not np.isfinite(r) or r < 0.0:
        raise ConfigurationError(
            f"ranges_fit_alias_period: rmax_m must be a non-negative, finite "
            f"range in metres; got {rmax_m!r}.")
    return r < alias_period(delta_k)
