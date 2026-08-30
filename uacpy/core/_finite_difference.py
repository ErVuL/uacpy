"""Resolution a stored wavenumber leaves for the finite difference over it.

``v_g = dω/dk_r`` is formed by differencing horizontal wavenumbers that a mode
solver wrote to a file. KRAKEN's ``.mod`` record is ``COMPLEX*8``
(``ReadModes.f90:243``), so ``k_r`` arrives with 24 mantissa bits and two
neighbouring frequencies differ by a whole number of float32 steps. The
difference of two such values carries about one step of quantization noise
whatever the arithmetic that follows is done in, so the *relative* error the
storage puts on ``dω/dk_r`` is

    floor ≈ spacing(k_r) / |Δk_r over one frequency step|

and shrinking the frequency step raises it. Measured on an analytic Pekeris
guide (D = 100 m, c 1500/1800 m/s, ρ 1.0/1.8, f₀ = 100 Hz) with exact
longdouble roots and a complex-step reference, against
``Modes.compute_group_velocity``:

    Δf (Hz)      steps      floor      measured error of v_g
       1.0      140844    7.1e-06                   7.29e-06
       0.5       70422    1.4e-05                   1.23e-05
       0.1       14084    7.1e-05                   4.39e-05
      0.01        1408    7.1e-04                   6.23e-04
     0.001         141    7.1e-03                   4.35e-03
    0.0001          14    7.1e-02                   6.58e-02

The floor predicts the error within a factor of 1.7 over five decades there,
and within 2.2 over four decades on an independent ideal guide (D = 70 m,
c = 1520 m/s, mode 3, 40-90 Hz).

**The floor is not the whole error, and it does not say which way to go.** The
truncation error of the difference falls as Δf² while the floor rises as 1/Δf,
so the total has an interior optimum at

    Δf* = (spacing(k_r)·v_g / (2π·|d²k_r/dω²|))**(1/3)

whose ``|d²k_r/dω²|`` no *pair* of frequencies can see. A remedy phrased on the
floor alone — "use the step that puts the floor at 1e-5" — is therefore a
recommendation for one waveguide, and on five ideal guides it made the answer
**worse in 14 of 30 firings, by up to 16x** (measured). What a whole sweep
*can* do is walk to the answer: decimating the grid by ``K`` raises the
truncation error as ``K²`` and lowers the floor as ``1/K``, so widening while
the answer still moves by less than the floor, and stopping one doubling before
it moves by more, lands below the optimum without ever estimating
``|d²k_r/dω²|``. That is :func:`coarser_step_multiple`, and over the same 30
firings it was **never worse than the step in hand** and 3.3x better at the
median.

Upcasting ``k_r`` recovers nothing here — the bits were never written — so this
module measures the difference rather than changing its dtype. It reads the
precision the values actually arrived in, not the dtype of the container they
arrived in, so a float32-sourced array that a caller has already promoted to
float64 is still measured against the float32 spacing.

This module sits in ``core`` because :meth:`uacpy.core.results.Modes.
compute_group_velocity` and :func:`uacpy.acoustic_signal.modal_group_velocity`
both difference the same wavenumbers, from different layers.
"""

import warnings

import numpy as np

from uacpy.core._warn_frames import USER_FRAME_SKIP

#: Relative accuracy above which the *storage* of ``k_r`` is worth reporting as
#: a floor under ``dω/dk_r``. Placed at the optimum measured above: at 1e-5 the
#: Pekeris control sits on its best achievable answer, at 1e-4 it is 6x worse
#: than that, and at 1e-3 it is 85x worse. The comparison is a ratio of a
#: spacing to a difference of the same quantity, so it carries no unit and does
#: not move with the wavenumber scale.
GROUP_VELOCITY_REL_FLOOR = 1e-5

#: Mantissa steps a value must sit off to count as float32 *storage* rather
#: than as a float64 array that happens to be float32-exact. The low bits of a
#: wavenumber a solver wrote are as good as random, so the chance that every
#: one of ``_DYADIC_MIN_SIZE`` values is a multiple of this many spacings is
#: ``16**-n``; an array built in float64 out of exact eighths is a multiple of
#: every one of them. Without the test, ``1.0 + arange(64)/128`` — full float64
#: precision — was reported as sitting on a 1.19e-07 grid, nine decades from
#: the truth.
_DYADIC_LOW_BITS = 16
_DYADIC_MIN_SIZE = 4


def storage_spacing(kr):
    """Spacing of the floating-point grid ``kr`` actually arrived on.

    Every value surviving a float32 round trip means the array carries at most
    24 mantissa bits however it is stored now, which is what a ``.mod`` file
    hands back — unless the values are also multiples of a much coarser power
    of two, which no float32 record is and a synthetic float64 ramp is.
    Anything else is measured against the float64 spacing, where the floor is
    eleven decades lower and this diagnostic never fires.
    """
    kr = np.asarray(kr, dtype=float)
    if kr.size:
        as32 = kr.astype(np.float32)
        if np.array_equal(as32.astype(float), kr):
            spacing32 = np.spacing(as32).astype(float)
            dyadic = (kr.size >= _DYADIC_MIN_SIZE and not np.any(
                np.remainder(kr, _DYADIC_LOW_BITS * spacing32) > 0))
            if not dyadic:
                return spacing32
    return np.spacing(kr)


def group_velocity_relative_floor(kr_ref, dk):
    """Relative error the storage of ``kr_ref`` puts on ``dω/dk``.

    ``kr_ref`` is the wavenumber the difference ``dk`` was taken at (any
    broadcast-compatible shape). A zero step is returned as ``inf``: it
    resolves nothing at all, and the caller that divides by it reports that
    separately.
    """
    spacing = storage_spacing(kr_ref)
    dk = np.abs(np.asarray(dk, dtype=float))
    nonzero = dk > 0
    return np.where(nonzero, spacing / np.where(nonzero, dk, 1.0), np.inf)


def _centred(omega, kr, k):
    """``dω/dk_r`` from points ``2k`` apart: the grid decimated by ``k``."""
    dom = omega[2 * k:] - omega[:-2 * k]
    dkr = kr[2 * k:] - kr[:-2 * k]
    if kr.ndim > 1:
        dom = dom.reshape((-1,) + (1,) * (kr.ndim - 1))
    with np.errstate(divide='ignore', invalid='ignore'):
        return dom / dkr


def coarser_step_multiple(omega, kr):
    """How much coarser this frequency grid should be, as a whole multiple.

    Decimating by ``K`` multiplies the truncation error by ``K²`` and divides
    the storage floor by ``K``, so there is always a decimation at which the
    truncation clears the quantization noise. The walk doubles ``K`` while the
    group velocity still agrees with its doubly-decimated self to within the
    floor at that decimation, and the answer is one doubling *back* from where
    it stops — a step that is still on the storage-limited side, so it cannot
    overshoot the optimum. ``1`` means the walk stopped at the first or second
    decimation, so the present spacing is already at or below the widest one
    the grid justifies; it does *not* say that a finer grid would help, which
    turns on a truncation error the grid in hand cannot resolve.

    Returns 1 for a grid too short to walk (fewer than five samples).
    """
    omega = np.asarray(omega, dtype=float)
    kr = np.asarray(kr, dtype=float)
    n = kr.shape[0]
    k = 1
    while 4 * k + 1 <= n:
        v_k = _centred(omega, kr, k)
        v_2k = _centred(omega, kr, 2 * k)
        m = min(v_2k.shape[0], v_k.shape[0] - k)
        if m <= 0:
            break
        near, far = v_k[k:k + m], v_2k[:m]
        with np.errstate(divide='ignore', invalid='ignore'):
            moved = np.abs(far - near) / np.abs(near)
        floor = group_velocity_relative_floor(
            kr[:m], kr[2 * k:2 * k + m] - kr[:m])
        moved = moved[np.isfinite(moved)]
        floor = floor[np.isfinite(floor)]
        if not moved.size or not floor.size:
            break
        if np.median(moved) > np.median(floor):
            break                       # the truncation error has emerged
        k *= 2
    return max(k // 2, 1)


def warn_if_storage_under_resolves(kr_ref, dk, v_g, caller, *, grid=None):
    """Warn when the stored ``k_r`` puts a floor under ``v_g`` worth reporting.

    Names the worst floor and the number of storage steps the difference spans.
    ``grid``, when given, is the ``(omega, kr)`` the difference was taken over
    and lets the message carry a *measured* remedy from
    :func:`coarser_step_multiple`; without it — two frequencies cannot separate
    truncation from storage — the message says what to check instead of
    prescribing a step, because a step prescribed from the floor alone is wrong
    more often than it is right (measured: worse in 14 of 30 firings).
    """
    floor = group_velocity_relative_floor(kr_ref, dk)
    under = np.isfinite(np.asarray(v_g, dtype=float)) & (
        floor > GROUP_VELOCITY_REL_FLOOR)
    if not np.any(under):
        return
    worst = float(np.max(np.where(under, floor, -np.inf)))
    spacing = float(np.max(np.broadcast_to(storage_spacing(kr_ref),
                                           np.shape(floor))[under]))
    speed = float(np.median(np.abs(np.asarray(v_g, dtype=float)[under])))

    if grid is None:
        remedy = (
            "Two frequencies cannot tell the two apart, so measure it: "
            "recompute at twice this frequency separation. If the largest "
            "change in v_g exceeds the floor above, the truncation error "
            "still leads and the separation is not what limits the answer; if "
            "it stays under, the floor is all that is left and a wider "
            "separation is better.")
    else:
        omega, kr = grid
        multiple = coarser_step_multiple(omega, kr)
        if multiple <= 1:
            remedy = (
                "Widening this grid is not indicated: it is already at or "
                "below the widest spacing still on the storage-limited side. "
                "Whether a *finer* grid would help turns on the truncation "
                "error, which needs modes this grid does not contain — "
                "recompute at half the spacing and keep the finer answer only "
                "if the largest change in v_g exceeds the floor above (right "
                "in 9 of 9 measured firings).")
        else:
            df = float(np.median(np.abs(np.diff(np.asarray(omega, float)))))
            df = multiple * df / (2.0 * np.pi)
            remedy = (
                f"On this grid the answer stops moving by more than that "
                f"floor once the frequency spacing is {multiple}x coarser "
                f"(about {df:.3g} Hz), which is the widest step still on the "
                f"storage-limited side of the optimum.")

    warnings.warn(
        f"{caller}: the wavenumber difference spans as few as "
        f"{1.0 / worst:.0f} steps of the coarsest grid these k_r could have "
        f"been stored on ({spacing:.3g} 1/m), so the storage puts a floor of "
        f"{worst:.1e} relative ({worst * speed:.3g} m/s) under "
        f"d(omega)/d(k_r) for {int(np.count_nonzero(under))} of {under.size} "
        f"value(s). The "
        f"truncation error of the difference is separate and may be larger or "
        f"smaller; the floor alone falls in proportion to the frequency step. "
        + remedy,
        UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
    )
