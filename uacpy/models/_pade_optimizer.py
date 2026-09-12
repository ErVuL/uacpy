"""Padé-error-based grid optimizer for the acoustic parabolic equation.

Reference
---------
Lytaev, M.S. (2023). *Mesh Optimization for the Acoustic Parabolic
Equation.* Journal of Marine Science and Engineering, 11(3), 496.
https://doi.org/10.3390/jmse11030496

The PE marches on a uniform ``(Δx, Δz)`` grid using a rational Padé
approximation of the propagator ``exp(ikΔx(√(1+ξ) − 1))``. This module:

* Splits the spectrum (:func:`_bands`) into an **accuracy band** — the
  water column's propagating components out to the wider of the caller's
  aperture and the seabed's critical angle, everything that carries energy
  to the far field — and a **stability band**, the rest of the medium's
  hull (Lytaev §4.1), held non-amplifying only. Scored on the whole hull a
  granite seabed put the branch point ``ξ = −1`` inside the interval and
  every grid was refused; scored on the aperture alone a rock seabed
  (critical angle 51°) read better than sand while measuring 3 dB rms
  worse at 1–5 km, its trapped modes between 30° and 51° unseen.
* Picks a reference sound speed ``c₀`` (:func:`optimal_c0`, Eq. 15 on the
  accuracy band) so that band straddles the Padé sweet spot ``ξ = 0``.
* Computes the Padé approximation error on the accuracy band and a
  Numerov vertical-FD error from ``Δz`` at that band's steepest component.
* Searches for the coarsest ``(Δx, Δz)`` whose total error stays under a
  user accuracy budget ``ε`` over ``n_steps = ⌈x_max/Δx⌉`` range steps.

``c₀`` is a user input; the optimizer picks ``(Δx, Δz)`` against that
value. It is the *algorithmic* expansion point of the parabolic equation
— the speed factored out as ``exp(ik₀x)`` — not a physical medium speed.
For the error-optimal ``c₀`` from Eq. 15, call :func:`optimal_c0`
explicitly and pass the result back in via the ``c0`` argument.

The Padé coefficients are derived numerically from the Taylor series of
``f(ξ) = exp(ikΔx(√(1+ξ)−1))`` so the same code handles any order
``[p/p]``. We use the diagonal ``[p/p]`` form because it is the standard
choice for one-way propagators (see Collins 1993). The Collins binaries
march that approximant with one accuracy row traded for the stability
constraint ``P(−3) = 0`` (``ns = 1``, ``epade``), whose per-step error on
the propagating band is 4–12× the pure ``[p/p]`` one scored here; at the
grids this module selects the Padé term is orders of magnitude under the
``Δz`` term, so the selection and the reported error are unchanged by it.

The score ``τ·n_steps`` bounds the accumulated error of the STEEPEST
scored component; the field error measured against Kraken on a 9 × 19
receiver grid is 3–10× below it (sand, silt, rock at 100–200 Hz), and the
near field — r ≲ 15·Δx, components steeper than the band — is outside it.

The receiver grid stays user-controlled — the optimizer reshapes only
the internal march grid.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from uacpy.core.exceptions import ConfigurationError

# ─────────────────────────────────────────────────────────────────────────────
# Padé approximant of exp(ik·Δx·(√(1+ξ) − 1))
# ─────────────────────────────────────────────────────────────────────────────


def _propagator_taylor(dx: float, k0: float, n_terms: int) -> np.ndarray:
    """Maclaurin coefficients of ``f(ξ) = exp(ikΔx(√(1+ξ) − 1))`` to order
    ``n_terms-1``.

    Built by composition: ``√(1+ξ) − 1 = Σ_{j≥1} C(1/2, j) · ξ^j`` (binomial
    series of the square root, minus the constant term), then
    ``exp(ik·Δx·g(ξ))`` expanded via the ordinary series for ``exp``.
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
        raise ConfigurationError(
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


def _propagator_pade(dx: float, k0: float,
                     p: int) -> Tuple[np.ndarray, np.ndarray]:
    """``[p/p]`` Padé coefficients ``(P, Q)`` of one propagator step.

    A function of ``(dx, k0, p)`` alone. ``Δz`` reaches
    :func:`combined_error` through the spectral spread ``Δξ`` rather than
    through the approximant, so a whole ``Δz`` ladder scored at one ``Δx``
    shares a single build.
    """
    return _pade_pp(_propagator_taylor(dx, k0, n_terms=2 * p + 5), p)


def _eval_poly(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Horner evaluation of ``Σ coeffs[k] · x^k`` (ascending order)."""
    out = np.zeros_like(x, dtype=complex)
    for c in coeffs[::-1]:
        out = out * x + c
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Error functionals
# ─────────────────────────────────────────────────────────────────────────────


def numerov_error(
    dz: float, k0: float, theta_max: float,
    alpha: float = 0.0, n_samples: int = 401,
    kz_max: Optional[float] = None,
) -> float:
    """Max FD error of the depth operator on ``k_z ∈ [-k₀ sin θ_max, 0]``,
    or on ``[-kz_max, 0]`` when ``kz_max`` (rad/m, the accuracy band's
    steepest component, :func:`_bands`) is given.

    ``theta_max`` is in **radians** here, unlike :func:`optimal_c0` /
    :func:`optimize_grid` / :func:`grid_error`, whose ``theta_max`` is the
    user-facing one in degrees — those convert before calling in.

    Lytaev (2023), Eq. (13) — *Mesh Optimization for the Acoustic
    Parabolic Equation*, https://doi.org/10.3390/jmse11030496.
    For ``alpha = 0`` this is the standard 3-point second-order operator;
    ``alpha = 1/12`` is the 4th-order Numerov correction.
    """
    if dz <= 0:
        return float("inf")
    kz_min = -(k0 * np.sin(theta_max) if kz_max is None else float(kz_max))
    kz = np.linspace(kz_min, 0.0, n_samples)
    s = np.sin(0.5 * kz * dz)
    # Continuous: Δz²·k_z².  Discrete (with optional Numerov α):
    #   D_{Δz} e^{ikz·} → (1/Δz²)·(-4 sin² + α·16 sin⁴) · e^{ikz·}
    discrete_neg_kz2 = (-4.0 * s ** 2 - alpha * 16.0 * s ** 4) / (dz ** 2)
    # Compare with exact -k_z².
    return float(np.max(np.abs(discrete_neg_kz2 - (-(kz ** 2)))))


# How much the Padé operator may exceed unit modulus on the evanescent part of
# the spectrum before the candidate grid is rejected. A proper Padé
# approximation of the square-root propagator is non-amplifying there; the
# slack absorbs rounding only.
_EVANESCENT_GROWTH_TOL = 1e-9


def _step_error(
    dx: float, dz: float, k0: float, p: int,
    xi_min: float, xi_max: float,
    theta_max: float, alpha: float = 0.0,
    n_xi: int = 161, n_offsets: int = 5,
    pade: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    kz_max: Optional[float] = None,
    xi_stab_min: Optional[float] = None,
    xi_stab_max: Optional[float] = None,
) -> Tuple[float, float]:
    """``(τ, growth)`` of one step. τ(Δx, Δz) is the worst case of
    ``|f(ξ₁) - P(ξ₂)/Q(ξ₂)|`` over ``ξ₁ ∈ [ξ_min, ξ_max]``, ``|ξ₂-ξ₁| ≤ Δξ``
    where ``Δξ = h(Δz)/k₀²`` is the discretisation-induced wander of ``ξ``;
    growth is ``max |P/Q| − 1`` over the stability band outside the
    accuracy band (``0`` when there is none).

    ``theta_max`` is in **radians** (passed straight to
    :func:`numerov_error`, together with ``kz_max`` when given).
    ``xi_stab_min`` / ``xi_stab_max`` widen the sampled interval to the
    stability band (:func:`_bands`), on which the approximant is not held
    to ``f``: its evanescent part (``ξ < −1``) only has to be
    non-amplifying, the test below. Left ``None`` the accuracy band is the
    whole interval, and its own evanescent part, if any, is the part tested.

    Lytaev (2023), τ formula above Eq. (14) —
    https://doi.org/10.3390/jmse11030496. The discretisation spread
    forces the Padé operator (which is built around ξ = 0) to remain
    accurate not just on ``[ξ_min, ξ_max]`` but at all *nearby* points
    too — the propagator is phase-sensitive in ξ, so a small ξ shift
    can rotate the complex value substantially.

    ``pade`` accepts an already-built ``(P, Q)`` from
    :func:`_propagator_pade`, which is how :func:`optimize_grid` keeps the
    build out of its Δz ladder. It must have been built at this call's
    ``(dx, k0, p)``; ``None`` builds it here.
    """
    h = numerov_error(dz, k0, theta_max, alpha=alpha, kz_max=kz_max)
    delta_xi = h / (k0 ** 2)

    # Depends only on (dx, k0, p) — see _propagator_pade.
    P, Q = _propagator_pade(dx, k0, p) if pade is None else pade

    lo = xi_min if xi_stab_min is None else min(float(xi_stab_min), xi_min)
    hi = xi_max if xi_stab_max is None else max(float(xi_stab_max), xi_max)
    xi1_grid = np.linspace(lo, hi, n_xi)
    # The accuracy band itself, sampled at its ends whatever the spacing.
    in_band = (xi1_grid >= xi_min) & (xi1_grid <= xi_max)
    if xi_stab_min is not None or xi_stab_max is not None:
        xi1_grid = np.concatenate([xi1_grid, [xi_min, xi_max]])
        in_band = np.concatenate([in_band, [True, True]])
    if delta_xi > 0:
        offsets = np.linspace(-delta_xi, delta_xi, n_offsets)
    else:
        offsets = np.array([0.0])

    # Exact propagator at ξ₁, on the COMPLEX branch. ξ drops below -1 over the
    # evanescent part of the angular spectrum, which the auto grid reaches
    # whenever sin(theta_max) > sqrt(2)·c_min/c_max — with the shipped 30°
    # default, any seabed faster than ~2.83× the water speed (basalt, granite,
    # limestone under a slope). A REAL sqrt returns NaN there, `abs(NaN - pq)`
    # is NaN, and `NaN > err_max` is False, so err_max kept its 0.0 initialiser
    # and EVERY candidate grid scored a perfect zero: optimize_grid then took
    # the coarsest rung of both ladders (dx = x_max/2, dz = DZ_MAX) and the
    # march came back 360 dB rms from Scooter while the run logged "predicted
    # error 0.00e+00". The complex branch evaluates the evanescent propagator
    # exp(-k0·dx·|Im√|) that belongs there.
    f_xi1 = np.exp(1j * k0 * dx
                   * (np.sqrt((1.0 + xi1_grid).astype(complex)) - 1.0))
    # Accuracy is required only where the propagator OSCILLATES (ξ ≥ -1).
    # Below that it decays, those components never reach the receiver, and
    # demanding the Padé reproduce them rejects grids that solve the problem
    # correctly — measured on a basalt seabed (c_max/c_min = 3.5),
    # dx = 10 m / dz = 0.05 m matches Scooter to 1.77 dB rms while a
    # whole-interval test scores it unusable. What the scheme must not do in
    # the evanescent band is AMPLIFY, so that part is checked for stability
    # instead of accuracy.
    propagating = xi1_grid >= -1.0
    scored = propagating & in_band
    err_max = 0.0
    growth = 0.0
    for off in offsets:
        xi2 = xi1_grid + off
        pq = _eval_poly(P, xi2) / _eval_poly(Q, xi2)
        if scored.any():
            m = float(np.max(np.abs(f_xi1 - pq)[scored]))
            if not np.isfinite(m):
                # A non-finite score must never read as "no error" — that is
                # exactly what let the coarsest grid look perfect. Report the
                # candidate as unusable so the search rejects it; if every
                # candidate is unusable, optimize_grid raises rather than
                # returning a grid nothing vouched for.
                return float('inf'), float('inf')
            if m > err_max:
                err_max = m
        if not propagating.all():
            # Measured 1.000000 for every candidate on the basalt case, so
            # this is a safety net rather than a discriminator: a Padé that
            # grew the evanescent spectrum would blow the march up.
            g = float(np.max(np.abs(pq[~propagating])))
            if not np.isfinite(g) or g > 1.0 + _EVANESCENT_GROWTH_TOL:
                return float('inf'), float('inf')
        if not scored.all():
            # The propagating part outside the aperture is damped or garbled
            # by design (Collins' ns constraint); reported, not thresholded.
            g = float(np.max(np.abs(pq[~scored])))
            growth = float('inf') if not np.isfinite(g) else max(growth, g - 1.0)
    return err_max, growth


def combined_error(*args, **kwargs) -> float:
    """Per-step error τ(Δx, Δz) alone — :func:`_step_error`'s first value,
    same arguments."""
    return _step_error(*args, **kwargs)[0]


# ─────────────────────────────────────────────────────────────────────────────
# Spectral bands and the reference sound speed (Eq. 15)
# ─────────────────────────────────────────────────────────────────────────────


def _bands(
    c0: float, c_min: float, c_max: float, theta_max: float,
    c_min_all: Optional[float] = None, c_max_all: Optional[float] = None,
) -> dict:
    """The two ``ξ`` bands scored at ``c₀`` (``theta_max`` in **degrees**;
    ``c_min`` / ``c_max`` the water column, ``c_min_all`` / ``c_max_all``
    the whole domain, default the water bounds).

    Accuracy band ``[xi_min, xi_max]``: ``(c₀/c_min)² − 1`` down to the
    lower of Lytaev §4.1's aperture end ``−sin²θ + (c₀/c_max)² − 1`` and
    the trapped-mode end ``(c₀/c_max_all)² − 1`` — the component grazing
    the fastest medium at its critical angle (``trapped_end_binds``).
    Stability band ``[xi_stab_min, xi_stab_max]``: the whole medium's hull
    with the aperture term. ``kz_max_over_k0``: the steepest scored
    component's vertical wavenumber over ``k₀`` — ``sin θ`` while the
    aperture binds, the critical-angle component's in the slowest water
    otherwise.
    """
    c0, c_min, c_max = float(c0), float(c_min), float(c_max)
    c_min_all = c_min if c_min_all is None else min(float(c_min_all), c_min)
    c_max_all = c_max if c_max_all is None else max(float(c_max_all), c_max)
    sin2 = np.sin(np.deg2rad(float(theta_max))) ** 2
    xi_aperture = -sin2 + (c0 / c_max) ** 2 - 1.0
    xi_trapped = (c0 / c_max_all) ** 2 - 1.0
    trapped = bool(xi_trapped < xi_aperture)
    kz2 = (c0 / c_min) ** 2 - (c0 / c_max_all) ** 2 if trapped else sin2
    xi_min, xi_max = min(xi_aperture, xi_trapped), (c0 / c_min) ** 2 - 1.0
    return dict(
        xi_min=float(xi_min), xi_max=float(xi_max),
        xi_stab_min=float(min(-sin2 + (c0 / c_max_all) ** 2 - 1.0, xi_min)),
        xi_stab_max=float(max((c0 / c_min_all) ** 2 - 1.0, xi_max)),
        kz_max_over_k0=float(np.sqrt(max(kz2, 0.0))),
        trapped_end_binds=trapped,
    )


def optimal_c0(c_min: float, c_max: float, theta_max: float,
               c_max_all: Optional[float] = None) -> float:
    """Picks ``c₀`` so the accuracy band centres on the Padé sweet spot
    ``ξ = 0``: Lytaev (2023) Eq. (15) — *Mesh Optimization for the Acoustic
    Parabolic Equation*, https://doi.org/10.3390/jmse11030496 — on
    whichever end bounds the band (:func:`_bands`).

    ``theta_max`` is in **degrees**. ``c_min`` / ``c_max`` bound the water
    column; a seabed ``c_max_all`` faster than ``c_max`` adds the trapped
    end's own symmetric solution (``ξ_min = −ξ_max`` there: Eq. 15 of
    ``(c_min, c_max_all)`` at zero aperture), and the candidate leaving the
    band's larger ``|ξ|`` smaller wins — 1591 m/s on 1500 m/s water over
    sand (1600), 2047 m/s over granite (5500).
    """
    def eq15(c_hi, theta):
        s2 = np.sin(np.deg2rad(float(theta))) ** 2
        return float(c_min * c_hi * np.sqrt((2.0 + s2) / (c_min ** 2 + c_hi ** 2)))

    candidates = [eq15(c_max, theta_max)]
    if c_max_all is not None and float(c_max_all) > float(c_max):
        candidates.append(eq15(float(c_max_all), 0.0))

    def half_width(c0):
        b = _bands(c0, c_min, c_max, theta_max, c_max_all=c_max_all)
        return max(abs(b['xi_min']), abs(b['xi_max']))

    return min(candidates, key=half_width)


# ─────────────────────────────────────────────────────────────────────────────
# Main optimizer
# ─────────────────────────────────────────────────────────────────────────────

# Bounds of the Δz search ladder (m) and the geometric ratio both ladders use.
DZ_MIN = 0.01
DZ_MAX = 5.0
LADDER_RATIO = 1.5


def _ladder(low: float, high: float) -> list:
    """Geometric ladder from ``low`` to ``high``, coarsest first."""
    rungs = []
    v = float(low)
    while v <= high:
        rungs.append(v)
        v *= LADDER_RATIO
    rungs.append(float(high))
    return sorted(set(rungs), reverse=True)


def optimize_grid(
    *,
    freq: float,
    c_min: float,
    c_max: float,
    x_max: float,
    c0: float,
    theta_max: float = 30.0,
    eps: float = 1e-3,
    p: int = 6,
    alpha: float = 0.0,
    tau_cache: Optional[dict] = None,
    c_min_all: Optional[float] = None,
    c_max_all: Optional[float] = None,
    grid: Optional[Tuple[float, float]] = None,
) -> dict:
    """Find the coarsest ``(Δx, Δz)`` whose accumulated error stays under
    ``ε`` over ``⌈x_max/Δx⌉`` march steps for the given ``c₀`` — or, with
    ``grid=(dr, dz)``, score that pair instead of searching.

    The search sees Lytaev's error model only. RAM applies its own
    stability floors and array caps to the returned ``Δz`` afterwards and
    re-reports the accuracy of the grid it actually marches through the
    ``grid`` form, so a search's ``predicted_error`` describes the
    unadjusted pair.

    Parameters
    ----------
    freq, c_min, c_max : float
        Operating frequency (Hz) and the slowest / fastest sound speeds
        of the water column (m/s) — the medium of the accuracy band.
    c_min_all, c_max_all : float, optional
        Slowest / fastest sound speeds anywhere in the domain, seabed
        included, bounding the stability band (:func:`_bands`).
        Default: the water bounds, i.e. one band as in Lytaev §4.1.
    x_max : float
        Maximum range (m) the PE will march to.
    c0 : float
        Reference sound speed (m/s). The RAM wrapper resolves its
        default via :func:`optimal_c0` before calling in here; pass any
        pinned user value through unchanged.
    theta_max : float
        Maximum propagation angle in **degrees**. Default 30°.
    eps : float
        Total accuracy budget (max ``|τ · n_steps|``). Default 1e-3.
        Unused with ``grid``.
    p : int
        Padé order ``[p/p]``. Default 6 (matches our RAM default).
    alpha : float
        Vertical-FD scheme parameter. ``0.0`` = standard second-order
        tridiagonal (rams0.5 / ramsurf1.5). ``1/12`` = 4th-order Numerov
        (Lytaev's enhancement, not currently used by the Collins binaries).
    tau_cache : dict, optional
        Memo for the per-step error τ(Δx, Δz), shareable across calls. τ is
        a property of the grid and the medium; ``eps`` is only the threshold
        it is compared against, so RAM's ε-relaxation ladder
        (``RAM._optimize_grid_relaxing``) otherwise rescores an identical
        candidate set on every retry. Each key carries every input τ depends
        on, so a cache handed to a call with different physics misses rather
        than answering from the wrong medium. This is **not** a search knob:
        the ``(Δx, Δz)`` selected and its ``predicted_error`` are the same
        values with the cache as without.
    grid : (float, float), optional
        A ``(dr, dz)`` to score in place of the ladders; the result then
        describes that pair whatever its error (``inf`` when unusable).

    Returns
    -------
    dict
        Keys: ``c0`` (echoed back), ``dr``, ``dz``, ``xi_min`` /
        ``xi_max`` (the accuracy band at the given ``c0``),
        ``xi_stab_min`` / ``xi_stab_max`` (the stability band),
        ``kz_max`` (rad/m, the steepest scored component),
        ``trapped_end_binds`` (it is the seabed's critical-angle mode),
        ``predicted_error`` (``τ · n_steps`` at the grid), ``growth``
        (``max |P/Q| − 1`` per step over the stability band outside the
        accuracy band; 0 when there is none), ``alpha``, ``p``.

    Raises
    ------
    RuntimeError
        If no candidate ``(Δx, Δz)`` meets the accuracy budget. The
        message states the search, not a remedy: RAM catches it and
        raises the user-facing ``ConfigurationError``.
    """
    c0_use = float(c0)
    k0 = 2.0 * np.pi * freq / c0_use
    theta_max_rad = np.deg2rad(float(theta_max))
    bands = _bands(c0_use, c_min, c_max, theta_max,
                   c_min_all=c_min_all, c_max_all=c_max_all)
    xi_min, xi_max = bands['xi_min'], bands['xi_max']
    xi_stab_min, xi_stab_max = bands['xi_stab_min'], bands['xi_stab_max']
    kz_max = k0 * bands.pop('kz_max_over_k0')

    # Candidate ladders, scanned in full; the pair maximising ``dx·dz``
    # among those inside the budget wins.
    dx_top = x_max * 0.5
    if grid is not None:
        dx_candidates, dz_candidates = [float(grid[0])], [float(grid[1])]
    else:
        dx_candidates = _ladder(max(0.5, c0_use / freq / 8.0), dx_top)
        # The ladder deliberately runs below any caller's own Δz floor.
        # Floors are applied to the RESULT (RAM's ``c_min/(16 f)`` cost
        # floor, ``_compute_grid_lytaev``), so a rung under the floor still
        # selects a marchable grid once floored — and flooring the ladder
        # instead was measured to make the marched field WORSE, because the
        # extra ε relaxation it forces licenses a coarser Δx (800 Hz over
        # 2 km moved further from a converged ramgeo march). What the caller
        # must not do is report this search's ε as the marched grid's
        # accuracy; RAM rescores that grid and reports both.
        dz_candidates = _ladder(DZ_MIN, DZ_MAX)

    cache = {} if tau_cache is None else tau_cache

    best = None
    best_product = -1.0
    for dx in dx_candidates:
        n_steps = int(np.ceil(x_max / dx))
        # One Padé build per Δx instead of one per (Δx, Δz): the approximant
        # is a function of (dx, k0, p) alone (:func:`_propagator_pade`), while
        # the ladder below moves only Δz. Built lazily so a retry whose whole
        # Δz ladder is already memoised builds nothing.
        pade = None
        for dz in dz_candidates:
            key = (dx, dz, k0, p, xi_min, xi_max, xi_stab_min, xi_stab_max,
                   kz_max, alpha)
            scored = cache.get(key)
            if scored is None:
                if pade is None:
                    pade = _propagator_pade(dx, k0, p)
                scored = cache[key] = _step_error(
                    dx, dz, k0, p, xi_min, xi_max, theta_max_rad, alpha=alpha,
                    pade=pade, kz_max=kz_max,
                    xi_stab_min=xi_stab_min, xi_stab_max=xi_stab_max,
                )
            tau, growth = scored
            total = tau * n_steps
            if grid is not None or total < eps:
                product = dx * dz
                if product > best_product:
                    best_product = product
                    best = dict(
                        dr=float(dx), dz=float(dz),
                        predicted_error=float(total), growth=float(growth),
                    )
    if best is None:
        # Internal control-flow signal: RAM catches this RuntimeError to fall
        # back to a user-pinned / coarser grid. NOT a user-facing config error —
        # do not retype to ConfigurationError (RAM's except clause keys on it).
        raise RuntimeError(
            f"No (Δx, Δz) candidate satisfies ε={eps:.2e} for "
            f"f={freq:.1f} Hz, c₀={c0_use:.0f} m/s, θ_max={float(theta_max):.1f}°, "
            f"x_max={x_max:.0f} m on the accuracy band "
            f"ξ ∈ [{xi_min:.3f}, {xi_max:.3f}] with the Δz ladder "
            f"[{DZ_MIN:g}, {DZ_MAX:g}] m and the Δx ladder "
            f"[{dx_candidates[-1]:.3g}, {dx_top:.3g}] m."
        )
    return dict(
        c0=c0_use, kz_max=float(kz_max), alpha=float(alpha), p=int(p),
        **bands, **best,
    )


def grid_error(*, dr: float, dz: float, **kwargs) -> float:
    """Accumulated Padé error ``τ · n_steps`` at an arbitrary ``(dr, dz)``:
    :func:`optimize_grid`'s ``predicted_error`` for ``grid=(dr, dz)``, same
    remaining keyword arguments."""
    return optimize_grid(grid=(dr, dz), **kwargs)['predicted_error']


def rams_dz_shear_cap(c_shear_min: float, freq: float,
                      per_wavelength: float = 14.0) -> float:
    """Upper bound on ``Δz`` for the rams0.5 elastic march: the shear
    wavelength must be resolved, so ``Δz <= λ_s / per_wavelength``.

    Collins (1991), JASA 89(3) 1050-1057 — the higher-order-Padé elastic PE
    rams0.5 implements — states the grid of all four of its worked examples,
    and ``λ_s/Δz`` is 85 (ex. A), 24 (B), **14** (C) and 64 (D). 14 is the
    coarsest grid the author himself uses, so it is the coarsest value with a
    citation behind it; anything coarser rests on measurement alone.

    Measured against OASES (wavenumber integration, exact for these
    range-independent elastic half-spaces) on example D's own environment and
    on a 50 Hz shelf case: ``λ_s/14`` gives 0.81 dB and 3.58 dB, while
    ``0.55 λ_s`` — a *lower* bound this function replaced — gives 134 dB and
    141 dB, the march having diverged.

    Returns ``0`` for fluid envs (``c_shear_min == 0``).
    """
    if c_shear_min <= 0:
        return 0.0
    return float(c_shear_min / (per_wavelength * max(freq, 1.0)))
