"""Padé-error-based grid optimizer for the acoustic parabolic equation.

Reference
---------
Lytaev, M.S. (2023). *Mesh Optimization for the Acoustic Parabolic
Equation.* Journal of Marine Science and Engineering, 11(3), 496.
https://doi.org/10.3390/jmse11030496

The PE marches on a uniform ``(Δx, Δz)`` grid using a rational Padé
approximation of the propagator ``exp(ikΔx(√(1+ξ) − 1))``. This module:

* Picks an optimal reference sound speed ``c₀`` (Eq. 15) so the spectrum
  ``ξ ∈ [ξ_min, ξ_max]`` straddles the Padé sweet spot at ``ξ = 0``.
* Computes the Padé approximation error on that interval and a Numerov
  vertical-FD error from ``Δz``.
* Searches for the coarsest ``(Δx, Δz)`` whose total error stays under a
  user accuracy budget ``ε`` over ``n_steps = ⌈x_max/Δx⌉`` range steps.

``c₀`` is a user input (it has physical meaning — the reference water
sound speed, conventionally 1500 m/s); the optimizer picks
``(Δx, Δz)`` against that value. For the performance-optimal ``c₀``
from Eq. 15, call :func:`optimal_c0` explicitly and pass the result
back in via the ``c0`` argument.

The Padé coefficients are derived numerically from the Taylor series of
``f(ξ) = exp(ikΔx(√(1+ξ)−1))`` so the same code handles any order
``[p/p]``. We use the diagonal ``[p/p]`` form because it is the standard
choice for one-way propagators (see Collins 1993).

The receiver grid stays user-controlled — the optimizer reshapes only
the internal march grid.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Padé approximant of exp(ik·Δx·(√(1+ξ) − 1))
# ─────────────────────────────────────────────────────────────────────────────


def _propagator_taylor(dx: float, k0: float, n_terms: int) -> np.ndarray:
    """Maclaurin coefficients of ``f(ξ) = exp(ikΔx(√(1+ξ) − 1))`` to order
    ``n_terms-1``.

    Built by composition: ``√(1+ξ) − 1 = Σ_{j≥1} (-1)^(j+1) (2j-2)! /
    (j!(j-1)! · 4^(j-1) · (2j-1)) · ξ^j``  (binomial series of the
    square root, minus the constant term), then ``exp(ik·Δx·g(ξ))``
    expanded via the ordinary series for ``exp``.
    """
    # Build g(ξ) = √(1+ξ) - 1 series via the binomial expansion.
    g = np.zeros(n_terms, dtype=complex)
    # Coefficient of ξ^j in √(1+ξ) is binomial(1/2, j) for j>=0; subtract
    # the j=0 term (which equals 1) to drop the "-1".
    coeff = 1.0
    for j in range(1, n_terms):
        # binomial(1/2, j) = binomial(1/2, j-1) · (1/2 - (j-1)) / j
        coeff = coeff * (0.5 - (j - 1)) / j
        g[j] = coeff
    # Convolve to compose: f = exp(α·g) where α = i·k₀·Δx.
    alpha = 1j * k0 * dx
    # Series for exp(α·g): use Cauchy product. f = sum_{m>=0} (α^m / m!) · g^m.
    f = np.zeros(n_terms, dtype=complex)
    f[0] = 1.0
    g_pow = np.zeros(n_terms, dtype=complex)
    g_pow[0] = 1.0
    fact = 1.0
    for m in range(1, n_terms):
        # g_pow ← g_pow * g (truncated to n_terms)
        new_pow = np.zeros(n_terms, dtype=complex)
        for i in range(n_terms):
            if g_pow[i] == 0:
                continue
            for j in range(1, n_terms - i):
                new_pow[i + j] += g_pow[i] * g[j]
        g_pow = new_pow
        fact *= m
        f += (alpha ** m / fact) * g_pow
        if not np.any(g_pow != 0):
            break
    return f


def _pade_pp(taylor: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
    """Diagonal ``[p/p]`` Padé approximant from Taylor coefficients.

    Solves the standard Padé linear system: given ``f(ξ) ≈ Σ c_k ξ^k``,
    find ``P, Q`` with ``deg P = deg Q = p`` and ``Q(0) = 1`` such that
    ``f · Q − P = O(ξ^{2p+1})``.

    Returns ``(P, Q)`` as length-``p+1`` arrays of polynomial coefficients
    in ascending degree order (``P[0] + P[1]ξ + … + P[p]ξ^p``).
    """
    if len(taylor) < 2 * p + 1:
        raise ValueError(
            f"Need ≥ {2 * p + 1} Taylor coefficients for a [{p}/{p}] Padé; "
            f"got {len(taylor)}."
        )
    c = taylor[: 2 * p + 1]

    # The denominator coefficients q_1..q_p solve a Hankel-type system.
    # With Q(0) = 1, the matching equations of order p+1..2p give:
    #     sum_{j=1..p} c_{p+1-j+m} · q_j = -c_{p+1+m},   m = 0..p-1
    A = np.zeros((p, p), dtype=complex)
    b = np.zeros(p, dtype=complex)
    for m in range(p):
        for j in range(1, p + 1):
            A[m, j - 1] = c[p + 1 - j + m]
        b[m] = -c[p + 1 + m]
    q_rest = np.linalg.solve(A, b)
    Q = np.zeros(p + 1, dtype=complex)
    Q[0] = 1.0
    Q[1:] = q_rest

    # Numerator: P[m] = sum_{j=0..min(m,p)} c_{m-j} · q_j  for m=0..p.
    P = np.zeros(p + 1, dtype=complex)
    for m in range(p + 1):
        s = 0.0 + 0.0j
        for j in range(min(m, p) + 1):
            s += c[m - j] * Q[j]
        P[m] = s
    return P, Q


def _eval_poly(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Horner evaluation of ``Σ coeffs[k] · x^k`` (ascending order)."""
    out = np.zeros_like(x, dtype=complex)
    for c in coeffs[::-1]:
        out = out * x + c
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Error functionals
# ─────────────────────────────────────────────────────────────────────────────


def pade_error(
    dx: float, k0: float, p: int,
    xi_min: float, xi_max: float,
    n_samples: int = 401,
) -> float:
    """Max ``|f(ξ) − P(ξ)/Q(ξ)|`` on ``[ξ_min, ξ_max]`` for a single range
    step ``Δx``.

    Lytaev (2023), R(Δx, ξ) formula in Section 4.1 —
    https://doi.org/10.3390/jmse11030496. ``f`` is the exact propagator,
    ``P/Q`` is the diagonal ``[p/p]`` Padé built around ``ξ = 0``.
    """
    # Need 2p+1 Taylor coefficients — add a few extra for stability.
    taylor = _propagator_taylor(dx, k0, n_terms=2 * p + 5)
    P, Q = _pade_pp(taylor, p)
    xi = np.linspace(xi_min, xi_max, n_samples)
    f = np.exp(1j * k0 * dx * (np.sqrt(1.0 + xi) - 1.0))
    pq = _eval_poly(P, xi) / _eval_poly(Q, xi)
    return float(np.max(np.abs(f - pq)))


def numerov_error(
    dz: float, k0: float, theta_max: float,
    alpha: float = 0.0, n_samples: int = 401,
) -> float:
    """Max FD error of the depth operator on ``k_z ∈ [-k₀ sin θ_max, 0]``.

    Lytaev (2023), Eq. (13) — *Mesh Optimization for the Acoustic
    Parabolic Equation*, https://doi.org/10.3390/jmse11030496.
    For ``alpha = 0`` this is the standard 3-point second-order operator;
    ``alpha = 1/12`` is the 4th-order Numerov correction.
    """
    if dz <= 0:
        return float("inf")
    kz_min = -k0 * np.sin(theta_max)
    kz_max = 0.0
    kz = np.linspace(kz_min, kz_max, n_samples)
    s = np.sin(0.5 * kz * dz)
    # Continuous: Δz²·k_z².  Discrete (with optional Numerov α):
    #   D_{Δz} e^{ikz·} → (1/Δz²)·(-4 sin² + α·16 sin⁴) · e^{ikz·}
    discrete_neg_kz2 = (-4.0 * s ** 2 - alpha * 16.0 * s ** 4) / (dz ** 2)
    # Compare with exact -k_z².
    return float(np.max(np.abs(discrete_neg_kz2 - (-(kz ** 2)))))


def combined_error(
    dx: float, dz: float, k0: float, p: int,
    xi_min: float, xi_max: float,
    theta_max: float, alpha: float = 0.0,
    n_xi: int = 161, n_offsets: int = 5,
) -> float:
    """Per-step error τ(Δx, Δz) — worst case of ``|f(ξ₁) - P(ξ₂)/Q(ξ₂)|``
    over ``ξ₁ ∈ [ξ_min, ξ_max]``, ``|ξ₂-ξ₁| ≤ Δξ`` where
    ``Δξ = h(Δz)/k₀²`` is the discretisation-induced wander of ``ξ``.

    Lytaev (2023), τ formula above Eq. (14) —
    https://doi.org/10.3390/jmse11030496. The discretisation spread
    forces the Padé operator (which is built around ξ = 0) to remain
    accurate not just on ``[ξ_min, ξ_max]`` but at all *nearby* points
    too — the propagator is phase-sensitive in ξ, so a small ξ shift
    can rotate the complex value substantially.
    """
    h = numerov_error(dz, k0, theta_max, alpha=alpha)
    delta_xi = h / (k0 ** 2)

    # Build the Padé approximation once (it depends only on dx, k0, p).
    taylor = _propagator_taylor(dx, k0, n_terms=2 * p + 5)
    P, Q = _pade_pp(taylor, p)

    xi1_grid = np.linspace(xi_min, xi_max, n_xi)
    if delta_xi > 0:
        offsets = np.linspace(-delta_xi, delta_xi, n_offsets)
    else:
        offsets = np.array([0.0])

    # exact propagator at ξ₁
    f_xi1 = np.exp(1j * k0 * dx * (np.sqrt(1.0 + xi1_grid) - 1.0))
    # for each offset, evaluate Padé at ξ₂ = ξ₁ + offset
    err_max = 0.0
    for off in offsets:
        xi2 = xi1_grid + off
        pq = _eval_poly(P, xi2) / _eval_poly(Q, xi2)
        diff = np.abs(f_xi1 - pq)
        m = float(np.max(diff))
        if m > err_max:
            err_max = m
    return err_max


# ─────────────────────────────────────────────────────────────────────────────
# Optimal reference sound speed (Eq. 15)
# ─────────────────────────────────────────────────────────────────────────────


def optimal_c0(c_min: float, c_max: float, theta_max: float) -> float:
    """Picks ``c₀`` so the propagation spectrum centres on the Padé sweet
    spot ``ξ = 0``.

    Lytaev (2023), Eq. (15) — *Mesh Optimization for the Acoustic
    Parabolic Equation*, https://doi.org/10.3390/jmse11030496.
    """
    return float(c_min * c_max * np.sqrt(
        (2.0 + np.sin(theta_max) ** 2) / (c_min ** 2 + c_max ** 2)
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Main optimizer
# ─────────────────────────────────────────────────────────────────────────────


def optimize_grid(
    *,
    freq: float,
    c_min: float,
    c_max: float,
    x_max: float,
    c0: float,
    theta_max: float = np.deg2rad(30.0),
    eps: float = 1e-3,
    p: int = 6,
    alpha: float = 0.0,
    dx_candidates: Optional[Sequence[float]] = None,
    dz_candidates: Optional[Sequence[float]] = None,
    dz_floor: float = 0.0,
    dz_ceiling: float = 5.0,
    dx_ceiling: Optional[float] = None,
) -> dict:
    """Find the coarsest ``(Δx, Δz)`` whose accumulated error stays under
    ``ε`` over ``⌈x_max/Δx⌉`` march steps for the given ``c₀``.

    Parameters
    ----------
    freq, c_min, c_max : float
        Operating frequency (Hz) and the slowest / fastest sound speeds
        anywhere in the propagation medium (m/s).
    x_max : float
        Maximum range (m) the PE will march to.
    c0 : float
        Reference sound speed (m/s). User input — defaults at the call
        site to 1500 m/s (water). To get the performance-optimal
        ``c₀`` from Lytaev Eq. (15), call :func:`optimal_c0` and pass
        the result in here.
    theta_max : float
        Maximum propagation angle (radians). Default 30°.
    eps : float
        Total accuracy budget (max ``|τ · n_steps|``). Default 1e-3.
    p : int
        Padé order ``[p/p]``. Default 6 (matches our RAM default).
    alpha : float
        Vertical-FD scheme parameter. ``0.0`` = standard second-order
        tridiagonal (rams0.5 / ramsurf1.5). ``1/12`` = 4th-order Numerov
        (Lytaev's enhancement, not currently used by the Collins binaries).
    dx_candidates, dz_candidates : sequences of float, optional
        Explicit grids to search. Default uses geometric ladders bounded
        by ``dz_ceiling`` (depth) and ``min(x_max, dx_ceiling)`` (range).
    dz_floor : float
        Lower bound on Δz (e.g. the rams shear-stability floor — see
        :func:`rams_dz_floor`). The optimizer never returns Δz below this.
    dz_ceiling, dx_ceiling : float
        Upper bounds.

    Returns
    -------
    dict
        Keys: ``c0`` (echoed back), ``dr``, ``dz``, ``xi_min`` /
        ``xi_max`` (Padé spectrum interval at the given ``c0``),
        ``predicted_error`` (``τ · n_steps`` at the chosen grid),
        ``alpha``, ``p``.

    Raises
    ------
    RuntimeError
        If no candidate ``(Δx, Δz)`` meets the accuracy budget. Caller
        can either widen ``ε``, raise ``p``, switch ``c0`` to a more
        favourable value (try :func:`optimal_c0`), or shrink
        ``theta_max`` / ``x_max``.
    """
    c0_use = float(c0)
    k0 = 2.0 * np.pi * freq / c0_use
    xi_min = -np.sin(theta_max) ** 2 + (c0_use / c_max) ** 2 - 1.0
    xi_max = (c0_use / c_min) ** 2 - 1.0

    # Default candidate ladders. We scan from coarse → fine and stop at
    # the first pair that satisfies the budget.
    if dx_candidates is None:
        dx_top = float(min(x_max * 0.5, dx_ceiling) if dx_ceiling else x_max * 0.5)
        # Geometric ladder from ~λ/8 up to dx_top, ratio 2.
        dx_floor = max(0.5, c0_use / freq / 8.0)
        ladder = []
        v = dx_floor
        while v <= dx_top:
            ladder.append(v)
            v *= 1.5
        ladder.append(dx_top)
        dx_candidates = sorted(set(ladder), reverse=True)
    if dz_candidates is None:
        # 0.01 m → dz_ceiling, ratio 1.5.
        ladder = []
        v = max(0.01, dz_floor) if dz_floor > 0 else 0.01
        while v <= dz_ceiling:
            ladder.append(v)
            v *= 1.5
        ladder.append(dz_ceiling)
        dz_candidates = sorted(set(ladder), reverse=True)

    best = None
    best_product = -1.0
    for dx in dx_candidates:
        if dx <= 0 or dx > x_max:
            continue
        n_steps = int(np.ceil(x_max / dx))
        for dz in dz_candidates:
            if dz < dz_floor or dz <= 0:
                continue
            tau = combined_error(
                dx, dz, k0, p, xi_min, xi_max, theta_max, alpha=alpha,
            )
            total = tau * n_steps
            if total < eps:
                product = dx * dz
                if product > best_product:
                    best_product = product
                    best = dict(
                        dr=float(dx), dz=float(dz),
                        predicted_error=float(total),
                    )
    if best is None:
        raise RuntimeError(
            f"No (Δx, Δz) candidate satisfies ε={eps:.2e} for "
            f"f={freq:.1f} Hz, c₀={c0_use:.0f} m/s, θ_max={np.rad2deg(theta_max):.1f}°, "
            f"x_max={x_max:.0f} m. Try a larger ε, higher Padé order p, "
            f"smaller θ_max, or a finer dz/dx ladder."
        )
    return dict(
        c0=c0_use,
        xi_min=float(xi_min),
        xi_max=float(xi_max),
        alpha=float(alpha),
        p=int(p),
        **best,
    )


def rams_dz_floor(c_shear_min: float, freq: float, factor: float = 0.55) -> float:
    """Lower bound on ``Δz`` for the rams0.5 elastic march so the rotated
    Padé operator does not alias high-wavenumber shear modes (the
    shear-stability constraint validated empirically on the example_06
    Pekeris-with-elastic case).

    Returns ``0`` for fluid envs (``c_shear_min == 0``).
    """
    if c_shear_min <= 0:
        return 0.0
    return float(factor * c_shear_min / max(freq, 1.0))
