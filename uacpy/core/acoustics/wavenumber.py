"""Wavenumber-domain acoustics on plain arrays.

What a spectral (wavenumber-integration) result gives you once you hold
``G(k)`` — from a Scooter/SPARC ``.grn`` file, from another code, from an
analytic kernel you wrote down. These were private inside the ``.grn``
reader while four user-facing docstrings cited them by that private path,
which named an import no user could write.
"""

import warnings
from typing import Optional

import numpy as np

from uacpy.core.exceptions import ConfigurationError
from uacpy.core._warn_frames import USER_FRAME_SKIP


def _hanning_taper(k: np.ndarray, freq: float,
                   cmin: Optional[float], cmax: Optional[float]) -> np.ndarray:
    """Build a window that tapers ``G(k)`` outside ``[ω/cmax, ω/cmin]``.

    Mirrors ``fieldsco.m:taper`` — symmetric Hanning roll-offs at the
    spectrum edges, ones in the middle. Returns ``ones`` when both bounds
    are inactive. Raises :class:`ConfigurationError` when the requested
    phase-speed band has no overlap with the file's wavenumber grid —
    the taper would zero the entire spectrum.
    """
    Nk = len(k)
    win = np.ones(Nk, dtype=float)
    if Nk == 0:
        return win

    omega = 2.0 * np.pi * freq
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
            f"at {freq:g} Hz — the taper would zero the entire spectrum. "
            f"Widen or drop cmin/cmax — via Scooter's taper= if that is "
            f"how they were set, or directly if this transform was called "
            f"by hand. SPARC has no taper setting; a SPARC .grn reaches this "
            f"path only through grn_to_field/grn_to_transfer_function."
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
    if source_type != 'R':
        return
    n_zero = int(np.count_nonzero(_zero_range_mask(ranges)))
    if n_zero:
        origin = f" (grn title {context!r})" if context else ""
        warnings.warn(
            f"grn_reader: {n_zero} receiver range(s) at r = 0{origin}: the "
            "point-source Hankel transform carries a 1/sqrt(r) "
            "cylindrical-spreading factor that is singular there, so those "
            "cells are returned as NaN (no data). Move the receiver off the "
            "source axis (e.g. r = 1 m) to get a field value.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP)


def _hankel_transform(
    G_src: np.ndarray,
    k: np.ndarray,
    ranges: np.ndarray,
    *,
    atten: float,
    source_type: str = 'R',
    spectrum: str = 'P',
    who: str = "hankel_transform",
) -> np.ndarray:
    """Wavenumber → range transform for one (source_depth, frequency) slab.

    A direct (trapezoidal-rule) DFT over the uniform ``k`` grid — the matrix
    product ``-G_scaled @ X`` below — not an FFT, matching ``fieldsco.m:5``
    ("This version uses the trapezoidal rule directly to do a DFT, rather
    than an FFT").

    Implements three of ``fieldsco.m``'s four source branches (its ``'H'``
    exact-Bessel branch, ``fieldsco.m:139-144``, is not exposed):

    ============  ==================================================
    source_type   Geometry
    ------------  --------------------------------------------------
    ``'R'``       cylindrical / point source (3-D), ``√(2πr)`` denom
    ``'X'``       Cartesian / line source (2-D), ``√(2π)`` denom
    ``'S'``       point source, cylindrical spreading removed, ``√(2π)`` denom
    ============  ==================================================

    ============  ==================================================
    spectrum      Half / full integration
    ------------  --------------------------------------------------
    ``'P'``       positive branch only (default; recommended)
    ``'N'``       negative branch only
    ``'B'``       both branches summed (full real-axis integral)
    ============  ==================================================

    Parameters
    ----------
    G_src : (nrd, nk) complex
    k     : (nk,) wavenumber grid
    ranges : (nr,) output ranges (m)
    atten : stabilising attenuation (added to k along the +i axis)
    source_type, spectrum : see table above
    """
    if source_type not in ('R', 'X', 'S'):
        raise ConfigurationError(
            f"{who}: source_type must be 'R', 'X', or 'S', got "
            f"{source_type!r}")
    if spectrum not in ('P', 'N', 'B'):
        raise ConfigurationError(
            f"{who}: spectrum must be 'P', 'N', or 'B', got "
            f"{spectrum!r}")

    dk = float(k[1] - k[0]) if len(k) > 1 else 1.0
    # fieldsco.m:109-111's own guard: the uniform-dk DFT is periodic in range
    # with period 2*pi/dk, and beyond ~10/dk the wrapped tail is amplified
    # exponentially by the exp(+atten*r) stabilisation compensation (+70 dB
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
            f"extends past the transform's alias period 2*pi/dk = "
            f"{alias_period(dk):g} m (fieldsco.m:109-111 stops here too).",
            remediation="Increase the spectral RMax (rmax_multiplier on "
                        "Scooter/SPARC) so dk shrinks, or shorten the "
                        "receiver ranges.")
    ck = k + 1j * atten
    abs_r = np.abs(ranges)
    # Carry the kernel at the .grn's own precision. ``G`` is written complex64
    # (``scooter.f90`` declares ``Green`` COMPLEX), so promoting it to
    # complex128 adds no information to the data and doubles the largest array
    # in the transform; measured on stress cases spanning the deepest
    # cancellation, the two paths agree to within 0.03 dB. The PHASE is a
    # different matter: ``k*r`` reaches ~1e5 rad here, so ``phase`` and the
    # exponential are evaluated in double and only the RESULT is cast down.
    # That result is not unit-modulus -- it carries ``exp(atten*r)``, ~2.2 at
    # the far receiver here -- but it is bounded, so the cast costs relative
    # precision only.
    dt = G_src.dtype if G_src.dtype == np.complex64 else np.complex128

    if source_type == 'X':
        # Line source: no √k weighting, no phase shift, 1/√(2π).
        factor1 = np.ones_like(ck)
        factor2 = dk / np.sqrt(2.0 * np.pi) * np.ones_like(abs_r)
        phase = np.outer(ck, abs_r)
    else:
        # Point source: phase factor exp(±i(kr - π/4)) and √k weighting.
        # 'R' adds 1/√(2πr) cylindrical spreading; 'S' omits it.
        factor1 = np.sqrt(ck)
        if source_type == 'R':
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
    # ``phase`` is complex because the contour offset ``atten`` is its
    # imaginary part, and every branch stays complex: cos of a complex
    # argument carries the exp(atten*r) stabilisation, exactly as
    # exp(-i.phase) + exp(+i.phase) does.
    if spectrum == 'B':
        np.cos(phase, out=phase)
        phase *= 2.0
    else:
        phase *= -1j if spectrum == 'P' else 1j
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


def ranges_fit_alias_period(delta_k: float, r_max: float) -> bool:
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
    r_max : float
        Farthest receiver range wanted (m).

    Returns
    -------
    bool
        ``True`` when ``r_max < 2*pi/dk``.
    """
    r = float(r_max)
    if not np.isfinite(r) or r < 0.0:
        raise ConfigurationError(
            f"ranges_fit_alias_period: r_max must be a non-negative, finite "
            f"range in metres; got {r_max!r}.")
    return r < alias_period(delta_k)


#: Public names. ``hankel_transform`` is the wavenumber -> range transform
#: and ``wavenumber_taper`` the ``c_min``/``c_max`` phase-speed window that
#: decides which physics a spectral run keeps; the private spellings remain
#: so the ``.grn`` reader's own call sites and tests are untouched.
hankel_transform = _hankel_transform
wavenumber_taper = _hanning_taper
