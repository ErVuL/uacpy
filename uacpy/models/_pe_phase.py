"""Per-backend phase-convention conversion for parabolic-equation outputs.

Every uacpy PE backend writes a slightly different quantity to disk
(carrier baked in vs bare envelope vs an extra ``exp(-iπ/4) / (4π)``
factor). Downstream broadband synthesis (``Field.synthesize_time_series``
and ``Field.to_time_trace``) expects a single canonical form: the
**engineering travelling-wave** pressure

    p̄(r, z, f)  ∝  ψ̄(r, z, f) · exp(-i k₀ r) · exp(-iπ/4) / √r

where ψ is the slow PE envelope, k₀ = ω/c₀ is the reference wavenumber,
``exp(-iπ/4)`` is the Hankel-asymptotic cylindrical-spreading phase of
``H₀⁽¹⁾(k₀r) ≈ √(2/(π k₀ r))·exp(+i(k₀r - π/4))`` under the conjugated
carrier, and the bar denotes complex conjugation (the conjugate flips the
mpiramS / Collins carrier from the ``exp(+iωt)`` to the ``exp(-iωt)``
sign uacpy uses everywhere else). mpiramS bakes the π/4 into what it
writes; the Collins codes factor out only ``exp(+i k₀ r)``
(``ramgeo1.5.f:436``, ``ramsurf1.5.f:445``, ``rams0.5.f:270``), so their
branches apply it here.

Three convention strings cover the three vendored binaries:

============  ====================================================  =======================================
convention    What the backend writes                                Fortran source
============  ====================================================  =======================================
``'mpiramS'`` ``psif = ψ · exp(+i(k₀ r + π/4)) / (4π)``              ``third_party/mpiramS/`` patched output
``'rams'``    ``ψ · exp(+i k₀ r rot₀)`` (carrier baked in via g₀)    ``rams0.5.f:848-851`` (g₀ at ``:889``)
``'ramsurf'`` ``ψ``                  (bare envelope, no carrier)     ``ramsurf1.5.f``: no ``g0`` anywhere
============  ====================================================  =======================================

Adding a fourth backend amounts to one new branch here plus declaring
the convention name in the backend's reader output — no other
``ram.py`` change required.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from uacpy.core.exceptions import ConfigurationError


# Canonical convention names — keep these in sync with the table in the
# module docstring. The values double as the strings the reader output
# carries.
MPIRAMS = 'mpiramS'
RAMS = 'rams'
RAMSURF = 'ramsurf'

_VALID_CONVENTIONS = frozenset({MPIRAMS, RAMS, RAMSURF})


def _broadcast_shape(
    psi_shape: tuple,
    range_axis: int,
    freq_axis: Optional[int],
    ranges_m: np.ndarray,
    k0: Optional[np.ndarray],
) -> tuple:
    """Return broadcast-compatible shapes for ``ranges_m`` and ``k0``
    against ``psi_shape``, both with singleton dims elsewhere."""
    ndim = len(psi_shape)
    rng_shape = [1] * ndim
    rng_shape[range_axis] = ranges_m.size
    out = (tuple(rng_shape),)
    if k0 is not None and freq_axis is not None:
        k0_shape = [1] * ndim
        k0_shape[freq_axis] = k0.size
        out = out + (tuple(k0_shape),)
    return out


def psi_to_travelling_wave(
    psi: np.ndarray,
    *,
    convention: str,
    ranges_m: np.ndarray,
    range_axis: int,
    k0: Optional[Union[float, np.ndarray]] = None,
    freq_axis: Optional[int] = None,
    apply_radial: bool = True,
) -> np.ndarray:
    """Convert a PE backend's raw output to engineering travelling-wave
    pressure ``p̄ ∝ ψ̄ · exp(-i k₀ r) / √r``.

    Parameters
    ----------
    psi : ndarray
        The complex envelope (or carrier-baked variant) read off disk.
        Any rank; ``range_axis`` and (optionally) ``freq_axis`` index
        into its shape.
    convention : str
        One of ``'mpiramS'``, ``'rams'``, ``'ramsurf'``. See the module
        docstring for what each backend writes.
    ranges_m : ndarray
        1-D array of receiver ranges in metres. Length matches
        ``psi.shape[range_axis]``. Values must be strictly positive
        when ``apply_radial=True`` or ``convention='ramsurf'``; the
        caller is responsible for clipping non-positive ranges.
    range_axis : int
        Axis of ``psi`` corresponding to ``ranges_m``.
    k0 : float or 1-D ndarray, optional
        Reference wavenumber ``ω/c₀``. Scalar for narrowband, 1-D
        ``(n_f,)`` for broadband. Required for ``'ramsurf'`` (the
        carrier the binary did not write) and ignored otherwise.
    freq_axis : int, optional
        Axis of ``psi`` corresponding to ``k0`` (broadband only).
    apply_radial : bool, optional
        When True (default), multiply by ``1/√r``. When False, leave
        radial scaling to the caller — useful when the caller will
        further interpolate / reshape before applying it.

    Returns
    -------
    ndarray, same shape as ``psi``, complex.
    """
    if convention not in _VALID_CONVENTIONS:
        raise ConfigurationError(
            f"unknown PE convention: {convention!r}; valid: "
            f"{sorted(_VALID_CONVENTIONS)}"
        )

    psi_bar = np.conj(psi)
    ranges_m = np.asarray(ranges_m, dtype=np.float64)

    rng_shape_only = _broadcast_shape(
        psi.shape, range_axis, None, ranges_m, None,
    )[0]

    # The Hankel-asymptotic cylindrical-spreading phase under the conjugated
    # carrier (see the module docstring). mpiramS bakes it into psif; the
    # Collins codes do not write it, so the RAMS / RAMSURF branches apply it.
    hankel = np.exp(-1j * np.pi / 4.0)

    if convention == MPIRAMS:
        # peramx already applies the full 3-D / Hankel-asymptotic output scaling
        # scl = exp(+i(ω/c₀·r + π/4))/(4π) (peramx.f90:420); the Collins PE
        # self-starter is itself a far-field Hankel approximation (Collins 1993;
        # JKPS), so the π/4 is the cylindrical-spreading phase, already baked in.
        # The wrapper must therefore ONLY conjugate (peramx marches the conjugate
        # time convention) and restore the 4π — applying any extra exp(±iπ/4)
        # here would double-count that phase. Verified: conj(psif)·4π matches
        # the exact wavenumber-integration field (Scooter) to ~2° over 1–3 km.
        # |TL| is unaffected.
        out = psi_bar * (4.0 * np.pi)
    elif convention == RAMS:
        # rams0.5 marches its own carrier in, one g₀ = exp(+i k₀ Δr rot₀) per
        # range step (``rams0.5.f:849-850``), so conj recovers the carrier —
        # but the code writes no Hankel π/4 (the dump at ``rams0.5.f:270``
        # divides by ``sqrt(r+eps)`` only), so it is applied here. ``rot₀`` is the
        # rotated-Padé scalar (``rams0.5.f:865-888``; exactly 1 when the
        # rotation is off, ``:909``) and is left in — it belongs to the
        # operator, not to the phase convention.
        out = psi_bar * hankel
    else:  # RAMSURF — needs explicit carrier
        if k0 is None or freq_axis is None:
            # Narrowband ramsurf: a scalar k0 with no freq_axis is OK.
            if k0 is None:
                raise ConfigurationError(
                    "convention='ramsurf' requires k0= for the carrier."
                )
            carrier = np.exp(-1j * float(k0) * ranges_m).reshape(rng_shape_only)
        else:
            k0_arr = np.atleast_1d(np.asarray(k0, dtype=np.float64))
            rng_shape, k0_shape = _broadcast_shape(
                psi.shape, range_axis, freq_axis, ranges_m, k0_arr,
            )
            carrier = np.exp(
                -1j * k0_arr.reshape(k0_shape) * ranges_m.reshape(rng_shape)
            )
        # ramsurf1.5 / ramgeo write the bare envelope: the carrier at :445 /
        # :436 carries no π/4, so the Hankel phase is applied here as well.
        out = psi_bar * carrier * hankel

    if apply_radial:
        radial = (1.0 / np.sqrt(ranges_m)).reshape(rng_shape_only)
        out = out * radial

    return out
