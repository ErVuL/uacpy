"""Tier-1 range-dependent benchmark — a two-way coupled-mode (mode-matching)
reference for a stepped isovelocity waveguide, and the propagation models
measured against it.

The rest of the analytic benchmark suite anchors *range-independent* physics
against closed forms (``test_benchmarks_analytic.py``). The one closed-form
range-dependent case in ocean acoustics — the ideal wedge — is a two-way
problem a one-way PE structurally cannot solve, so it anchors ray models only.
This module closes that gap with a reference a one-way solver legitimately can
be measured against: a single stepped depth discontinuity, solved by two-way
mode matching, in the weak-coupling regime where the reflected modal power the
PE discards is negligible.

Geometry
--------
Isovelocity water, pressure-release surface AND pressure-release bottom
(Dirichlet-Dirichlet) in both segments. Depth ``D1`` for ``r < r_step``,
``D2`` beyond it; source at ``r = 0`` in segment 1. The Dirichlet-Dirichlet
modes are elementary (``gamma_m = m*pi/D``, ``Z_m = sqrt(2/D) sin(gamma_m z)``),
so every mode-matching integral is analytic and the reference is exact up to
modal truncation — no root finding, no bottom half-space tail. Setting
``D2 == D1`` reduces it to ``dirichlet_modal_tl`` in
``test_benchmarks_analytic.py``, which Kraken lands on at 0.014 dB median over
a true vacuum bottom.

Formulation
-----------
Jensen, Kuperman, Porter & Schmidt, *Computational Ocean Acoustics* (2nd ed.),
Sect. 5.11.1 "Coupled Modes", following Evans, JASA 74, 188-195 (1983). The
field in segment ``j`` is a sum of outgoing and incoming modal waves
(JKPS 5.239) whose range functions are normalised at the segment boundaries so
only decaying exponentials appear inside a segment (JKPS 5.240-5.242); the
segments are glued by continuity of pressure and of radial particle velocity
(JKPS 5.244-5.253), each imposed as a moment condition — the depth error is
required to have vanishing components on the retained mode set rather than to
vanish pointwise.

JKPS project both conditions onto the *next* segment's modes because Evans'
geometry keeps the total depth fixed (the step is inside a penetrable bottom),
so both mode sets span the same interval. A Dirichlet step does not: the two
mode sets span ``[0, D1]`` and ``[0, D2]``. The convention adopted here is the
classical one for a waveguide step, which is what makes the two conditions
well-posed on their own domains:

* **Pressure** is matched over the *wider* segment's depth interval
  ``[0, max(D1, D2)]`` and projected onto the *wider* segment's modes. Over the
  vertical step face the narrow segment has no fluid, and the face is part of
  the pressure-release boundary of the fluid domain, so the narrow-side
  pressure enters the integral extended by zero. That single integral
  therefore imposes both continuity on the overlap and ``p = 0`` on the face.
* **Radial velocity** is matched over the *narrower* interval
  ``[0, min(D1, D2)]`` — the only depths where fluid exists on both sides — and
  projected onto the *narrower* segment's modes.

That is ``max(M1, M2)`` plus ``min(M1, M2)`` equations for the ``M1 + M2``
unknown scattered amplitudes.

Energy conservation and reciprocity do **not** certify that choice: they follow
from pairing pressure with ``C`` and velocity with ``C^T``, and the rival
convention — a *rigid* step face — satisfies both just as exactly. What
certifies it is the face itself
(``test_the_matched_field_leaves_the_step_face_pressure_release``, where the
rigid convention's face pressure plateaus instead of converging) and an
independent finite-difference solve of the Helmholtz equation on this stepped
domain, in which the soft-face junction converges to the FD answer while the
rigid-face one plateaus at 1.0e-1 relative
(``round32/build32-coupledmode.md`` Sect. 3.5).

The mode sets are truncated at a common vertical wavenumber
``gamma <= mode_reach * k`` rather than at a common mode *count*: a step
solved with mode counts whose ratio is not ``D1/D2`` converges to the wrong
answer (the "relative convergence" of the mode-matching literature), and a
common ``gamma`` cutoff enforces ``M1/M2 = D1/D2`` automatically.
``mode_reach = 1`` retains the propagating modes only; the evanescent modes
retained above that are what make the matching two-way rather than adiabatic.

Convention
----------
Same far-field prefactor and 1 m TL reference as ``pekeris_modal_tl`` and
``dirichlet_modal_tl`` in ``test_benchmarks_analytic.py``, so a model checked
against this reference is checked on the same convention as the existing ones.
"""
import numpy as np
import pytest

pytestmark = [pytest.mark.benchmark]

from uacpy import (Environment, SoundSpeedProfile, BoundaryProperties, Bathymetry,
                   Source, Receiver, RAM, Bellhop)
from uacpy.tests.test_benchmarks_analytic import dirichlet_modal_tl

# ── Dirichlet-Dirichlet modes and the mode-matching integrals ───────────────

def dd_modes(depth, f, c, mode_reach=1.0):
    """Vertical and horizontal wavenumbers of the Dirichlet-Dirichlet modes of
    an isovelocity layer of thickness ``depth``, retaining every mode with
    ``gamma_m <= mode_reach * k``.

    ``gamma_m = m*pi/depth`` exactly (both boundaries pressure-release), and
    ``k_m = sqrt(k^2 - gamma_m^2)`` taken on the branch with ``Im k_m >= 0``,
    so a propagating mode is real-positive and an evanescent mode is
    positive-imaginary (decaying away from its source). ``mode_reach = 1``
    gives the propagating set; larger values add the evanescent modes the
    mode matching needs.

    Raises if a retained mode sits within ``1e-6`` (relative to ``k``) of exact
    cutoff: ``k_m = 0`` is a genuine singularity of both the source amplitude
    ``1/sqrt(k_m)`` and the ``K2^-1`` of the up-step matching, and unlike a
    range-independent modal sum the mode cannot simply be dropped — it carries
    a non-zero share of the interface conditions. That guard is on the
    *propagating* cutoff and is separate from the truncation tie below.

    ``mode_reach * k * depth / pi`` can land exactly on an integer, in which
    case ``floor`` keeps a top mode whose ``gamma`` equals ``mode_reach * k``
    exactly and a hair of arithmetic the other way would have dropped it. It
    does so here: at ``mode_reach = 24``, ``c = 1500`` and ``f = 25`` the value
    is ``160.0`` for a 200 m layer, ``200.0`` for 250 m and ``132.0`` for
    165 m, all exact in IEEE double. The retained mode is deeply evanescent
    (``gamma = 24 k``), so this decides a truncation boundary and not a
    physical mode: forcing the count down by one at the benchmark fixture moves
    the reference by ``max|dTL| = 2.825e-04 dB`` (median 2.6e-05 dB), three
    decades under the tightest engine bound in this module. It is a
    reproducibility wrinkle, not a correctness risk — but a mode one ULP from a
    cutoff has bitten this codebase before (see
    ``test_a_waveguide_exactly_at_a_cutoff_does_not_generate_the_cutoff_mode``
    in ``test_benchmarks_analytic.py``), so it is written down.
    """
    k = 2 * np.pi * f / c
    n = int(np.floor(mode_reach * k * depth / np.pi))
    if n < 1:
        raise ValueError(f"no modes retained for depth={depth}, mode_reach={mode_reach}")
    gamma = np.arange(1, n + 1) * np.pi / depth
    km = np.sqrt((k**2 - gamma**2).astype(complex))
    if np.any(np.abs(km) < 1e-6 * k):
        bad = np.argmin(np.abs(km)) + 1
        raise ValueError(
            f"mode {bad} of the {depth} m layer sits at cutoff (k_m/k="
            f"{abs(km[bad - 1]) / k:.3e}); move the geometry off the cutoff")
    return gamma, km


def overlap_matrix(D1, D2, n1, n2):
    """``C[l, m] = int_0^min(D1,D2) Z_l^2(z) Z_m^1(z) dz`` — the mode-matching
    integral of JKPS 5.247/5.252, evaluated in closed form.

    ``int_0^L sin(a z) sin(b z) dz = sin((a-b)L)/(2(a-b)) - sin((a+b)L)/(2(a+b))``,
    with the ``a == b`` limit ``L/2 - sin(2aL)/(4a)``. With ``D1 == D2`` and
    ``n1 == n2`` this is the identity, which is what collapses the step to the
    range-independent modal sum.
    """
    L = min(D1, D2)
    g1 = np.arange(1, n1 + 1) * np.pi / D1
    g2 = np.arange(1, n2 + 1) * np.pi / D2
    a, b = g2[:, None], g1[None, :]
    dif, sm = a - b, a + b
    near = np.abs(dif) < 1e-12 * sm
    safe = np.where(near, 1.0, dif)
    first = np.where(near, L / 2.0, np.sin(safe * L) / (2.0 * safe))
    return (2.0 / np.sqrt(D1 * D2)) * (first - np.sin(sm * L) / (2.0 * sm))


def step_scatter(D1, D2, km1, km2, inc1, inc2, axis_phase=None):
    """Scattered modal amplitudes at a single Dirichlet step.

    Parameters are the two depths, the two horizontal-wavenumber sets, and the
    amplitudes arriving at the interface from segment 1 (``inc1``, travelling
    outward) and from segment 2 (``inc2``, travelling inward), all referenced
    to the interface itself. Returns ``(B, T)``: the amplitude reflected back
    into segment 1 and the amplitude transmitted into segment 2, both
    referenced to the interface.

    ``axis_phase`` folds in the boundary condition at ``r = 0`` (JKPS 5.257):
    in a cylindrical geometry the wave reflected back toward the axis
    refocuses and re-emerges outward, so the amplitude actually incident on
    the step is ``inc1 + axis_phase * B`` rather than ``inc1``. Pass ``None``
    for a plane geometry or for the bare junction operator.

    The two matching conditions are assembled as one linear system in
    ``(B, T)``; which of them is projected onto which mode set follows the
    module docstring — pressure over the wider interval onto the wider
    segment's modes, radial velocity over the narrower interval onto the
    narrower segment's modes.
    """
    n1, n2 = len(km1), len(km2)
    C = overlap_matrix(D1, D2, n1, n2)
    K1, K2 = np.diag(km1), np.diag(km2)
    E = np.diag(np.zeros(n1, dtype=complex) if axis_phase is None
                else np.asarray(axis_phase, dtype=complex))
    I1, I2 = np.eye(n1), np.eye(n2)
    A = np.zeros((n1 + n2, n1 + n2), dtype=complex)
    rhs = np.zeros(n1 + n2, dtype=complex)
    if D2 >= D1:
        # pressure over [0, D2] onto segment-2 modes; velocity over [0, D1]
        # onto segment-1 modes.
        A[:n2, :n1] = -C @ (I1 + E)
        A[:n2, n1:] = I2
        rhs[:n2] = C @ inc1 - inc2
        A[n2:, :n1] = K1 @ (I1 - E)
        A[n2:, n1:] = C.T @ K2
        rhs[n2:] = K1 @ inc1 + C.T @ K2 @ inc2
    else:
        # pressure over [0, D1] onto segment-1 modes; velocity over [0, D2]
        # onto segment-2 modes.
        A[:n1, :n1] = I1 + E
        A[:n1, n1:] = -C.T
        rhs[:n1] = -inc1 + C.T @ inc2
        A[n1:, :n1] = C @ K1 @ (I1 - E)
        A[n1:, n1:] = K2
        rhs[n1:] = C @ K1 @ inc1 + K2 @ inc2
    x = np.linalg.solve(A, rhs)
    return x[:n1], x[n1:]


# ── the reference field ─────────────────────────────────────────────────────

def coupled_mode_step_tl(z_s, z_r, ranges, f, D1, D2, r_step, c,
                         mode_reach=6.0, axis_return=True):
    """Transmission loss of a stepped Dirichlet-Dirichlet waveguide by two-way
    mode matching — the range-dependent counterpart of ``dirichlet_modal_tl``.

    Source at ``r = 0``, depth ``z_s``, in a layer of thickness ``D1``; the
    bottom steps to ``D2`` at ``r = r_step``. ``ranges`` may straddle the step.
    ``TL = -20 log10 |p|`` on the same far-field prefactor and 1 m reference as
    the range-independent references, so ``D2 == D1`` with ``mode_reach = 1``
    reproduces ``dirichlet_modal_tl`` exactly.

    ``axis_return`` includes the refocusing of the step-reflected wave at
    ``r = 0``; setting it False drops that term and measures its size.
    """
    ranges = np.atleast_1d(ranges).astype(float)
    g1, km1 = dd_modes(D1, f, c, mode_reach)
    g2, km2 = dd_modes(D2, f, c, mode_reach)
    Z1s = np.sqrt(2.0 / D1) * np.sin(g1 * z_s)

    # Incident modal amplitude at the step: the range-independent source field
    # sqrt(2*pi/r) * sum_m Z_m(z_s) Z_m(z) exp(i k_m r)/sqrt(k_m), read at r_step.
    inc1 = Z1s * np.exp(1j * km1 * r_step) / np.sqrt(km1)
    axis = -1j * np.exp(2j * km1 * r_step) if axis_return else None
    B, T = step_scatter(D1, D2, km1, km2, inc1, np.zeros(len(km2), dtype=complex),
                        axis_phase=axis)

    Z1r = np.sqrt(2.0 / D1) * np.sin(g1 * z_r) if z_r <= D1 else np.zeros(len(g1))
    Z2r = np.sqrt(2.0 / D2) * np.sin(g2 * z_r) if z_r <= D2 else np.zeros(len(g2))
    out = []
    for r in ranges:
        if r <= r_step:
            amp = (Z1s * np.exp(1j * km1 * r) / np.sqrt(km1)
                   + B * np.exp(1j * km1 * (r_step - r)))
            if axis_return:
                amp = amp - 1j * B * np.exp(1j * km1 * (r_step + r))
            p = np.sqrt(2 * np.pi / r) * np.sum(amp * Z1r)
        else:
            p = np.sqrt(2 * np.pi / r) * np.sum(
                T * np.exp(1j * km2 * (r - r_step)) * Z2r)
        out.append(-20.0 * np.log10(np.abs(p)))
    return np.array(out)


def step_power_split(f, D1, D2, c, z_s, r_step, mode_reach=6.0):
    """Reflected and transmitted fractions of the incident modal power flux at
    the step, summed over the propagating modes.

    A mode of amplitude ``a_m`` carries radial power flux proportional to
    ``Re(k_m) |a_m|^2`` (the cross terms vanish by depth orthogonality and the
    evanescent modes carry no net real flux), and the cylindrical spreading
    factored out of the field is common to both segments, so the fractions are
    ratios of ``sum_m k_m |a_m|^2``. Returns ``(R, T, residual)`` with
    ``residual = |1 - R - T|``, the energy-conservation error.
    """
    g1, km1 = dd_modes(D1, f, c, mode_reach)
    _, km2 = dd_modes(D2, f, c, mode_reach)
    inc1 = (np.sqrt(2.0 / D1) * np.sin(g1 * z_s)) * np.exp(1j * km1 * r_step) / np.sqrt(km1)
    B, T = step_scatter(D1, D2, km1, km2, inc1, np.zeros(len(km2), dtype=complex))
    p1, p2 = km1.imag == 0, km2.imag == 0
    flux_i = np.sum(km1[p1].real * np.abs(inc1[p1])**2)
    flux_r = np.sum(km1[p1].real * np.abs(B[p1])**2)
    flux_t = np.sum(km2[p2].real * np.abs(T[p2])**2)
    R, Tf = flux_r / flux_i, flux_t / flux_i
    return R, Tf, abs(1.0 - R - Tf)


def step_scattering_matrix(f, D1, D2, c, mode_reach=6.0):
    """Power-normalised scattering matrix of the step, restricted to the
    propagating modes.

    Column ``j`` is the outgoing propagating amplitudes produced by a unit
    incoming power flux in propagating mode ``j`` (evanescent incoming
    amplitudes zero), with every amplitude scaled by ``sqrt(k_m)`` so it
    carries unit power. A lossless reciprocal junction gives a matrix that is
    unitary (energy conservation) and symmetric (reciprocity).
    """
    _, km1 = dd_modes(D1, f, c, mode_reach)
    _, km2 = dd_modes(D2, f, c, mode_reach)
    p1, p2 = km1.imag == 0, km2.imag == 0
    n1, n2 = len(km1), len(km2)
    cols = []
    for side, idx in ((1, np.flatnonzero(p1)), (2, np.flatnonzero(p2))):
        for m in idx:
            i1 = np.zeros(n1, dtype=complex)
            i2 = np.zeros(n2, dtype=complex)
            if side == 1:
                i1[m] = 1.0 / np.sqrt(km1[m].real)
            else:
                i2[m] = 1.0 / np.sqrt(km2[m].real)
            B, T = step_scatter(D1, D2, km1, km2, i1, i2)
            cols.append(np.concatenate([np.sqrt(km1[p1].real) * B[p1],
                                        np.sqrt(km2[p2].real) * T[p2]]))
    return np.array(cols).T


def exact_hankel_step_tl(z_s, z_r, ranges, f, D1, D2, r_step, c):
    """The same cylindrical problem with EXACT Hankel/Bessel range functions
    instead of their far-field asymptotes — the cross-check on the axis term.

    ``coupled_mode_step_tl`` follows JKPS 5.242 and works in the asymptotic
    range functions, which forces the condition at ``r = 0`` (JKPS 5.257) to be
    supplied by hand as the focal phase ``-i``. Here regularity at the axis is
    imposed exactly instead: the homogeneous part of the segment-1 field is
    ``J_0(k_m r)``, which is regular at the origin by construction and carries
    no phase put in by hand. Segment 1 is
    ``(i/4) Z_m(z_s) H_0^(1)(k_m r) + c_m J_0(k_m r)`` and segment 2 is
    ``t_m H_0^(1)(k_m r)``, matched by the same two conditions on the same
    intervals.

    Propagating modes only, so every Bessel argument is real: ``J_0`` of a
    large imaginary argument is ``I_0`` and overflows, and an evanescent
    reflected wave never reaches the axis anyway.
    """
    from scipy.special import hankel1, jv
    g1, k1 = dd_modes(D1, f, c, 1.0)
    g2, k2 = dd_modes(D2, f, c, 1.0)
    k1, k2 = k1.real, k2.real
    n1, n2 = len(k1), len(k2)
    Cm = overlap_matrix(D1, D2, n1, n2)
    src = 0.25j * np.sqrt(2.0 / D1) * np.sin(g1 * z_s)
    P1_src, P1_c = src * hankel1(0, k1 * r_step), jv(0, k1 * r_step)
    V1_src = src * k1 * (-hankel1(1, k1 * r_step))
    V1_c = k1 * (-jv(1, k1 * r_step))
    P2_t, V2_t = hankel1(0, k2 * r_step), k2 * (-hankel1(1, k2 * r_step))
    A = np.zeros((n1 + n2, n1 + n2), dtype=complex)
    rhs = np.zeros(n1 + n2, dtype=complex)
    if D2 >= D1:
        A[:n2, :n1] = -Cm * P1_c[None, :]; A[:n2, n1:] = np.diag(P2_t)
        rhs[:n2] = Cm @ P1_src
        A[n2:, :n1] = np.diag(V1_c);       A[n2:, n1:] = Cm.T * V2_t[None, :]
        rhs[n2:] = V1_src
    else:
        A[:n1, :n1] = np.diag(P1_c);       A[:n1, n1:] = -Cm.T * P2_t[None, :]
        rhs[:n1] = -P1_src
        A[n1:, :n1] = Cm * V1_c[None, :];  A[n1:, n1:] = -np.diag(V2_t)
        rhs[n1:] = -Cm @ V1_src
    x = np.linalg.solve(A, rhs)
    cm, tm = x[:n1], x[n1:]
    Z1r = np.sqrt(2.0 / D1) * np.sin(g1 * z_r)
    Z2r = np.sqrt(2.0 / D2) * np.sin(g2 * z_r)
    out = []
    for r in np.atleast_1d(ranges).astype(float):
        if r <= r_step:
            p = np.sum((src * hankel1(0, k1 * r) + cm * jv(0, k1 * r)) * Z1r)
        else:
            p = np.sum(tm * hankel1(0, k2 * r) * Z2r)
        out.append(-20.0 * np.log10(np.abs(p * 4 * np.pi)))
    return np.array(out)


def plane_step_field(z_s, x_s, z_r, x_r, f, D1, D2, x_step, c, mode_reach=6.0):
    """Field of a line source in the stepped guide, plane (2-D) geometry.

    The cylindrical reference has its source pinned to ``r = 0``, so source
    and receiver cannot be exchanged across the step there. In plane geometry
    the guide runs to both infinities and a line source at ``x_s`` radiates
    both ways, so the exchange is well posed and Helmholtz reciprocity applies:
    the value returned must be unchanged under
    ``(z_s, x_s) <-> (z_r, x_r)``.

    The free modal Green's function of a Dirichlet-Dirichlet guide is
    ``i * sum_m Z_m(z) Z_m(z') exp(i k_m |x - x'|) / (2 k_m)``, which solves
    ``(grad^2 + k^2) G = -delta``; the step is applied by the same junction
    operator as the cylindrical case, with no axis term. Writing the modal
    amplitude as ``1/(2 i k_m)`` instead of ``i/(2 k_m)`` flips the sign of
    everything this returns, which neither the reciprocity test (an overall
    constant divides out of ``|a - b| / max(|a|, |b|)``) nor a transmission
    loss (``-20 log10|p|``) can see — but it is not the physical Green's
    function, so the ``i/(2 k_m)`` form is the one used.
    """
    if (x_s - x_step) * (x_r - x_step) > 0:
        raise ValueError("source and receiver must sit on opposite sides of the step")
    g1, km1 = dd_modes(D1, f, c, mode_reach)
    g2, km2 = dd_modes(D2, f, c, mode_reach)
    Z1 = lambda z: np.sqrt(2.0 / D1) * np.sin(g1 * z)
    Z2 = lambda z: np.sqrt(2.0 / D2) * np.sin(g2 * z)
    z1, z2 = np.zeros(len(km1), dtype=complex), np.zeros(len(km2), dtype=complex)
    if x_s < x_step:
        inc1 = Z1(z_s) * np.exp(1j * km1 * (x_step - x_s)) * (1j / (2 * km1))
        B, T = step_scatter(D1, D2, km1, km2, inc1, z2)
        return np.sum(T * np.exp(1j * km2 * (x_r - x_step)) * Z2(z_r))
    inc2 = Z2(z_s) * np.exp(1j * km2 * (x_s - x_step)) * (1j / (2 * km2))
    B, T = step_scatter(D1, D2, km1, km2, z1, inc2)
    return np.sum(B * np.exp(1j * km1 * (x_step - x_r)) * Z1(z_r))


# ── benchmark fixture ───────────────────────────────────────────────────────

C_W, FREQ = 1500.0, 25.0
D1, R_STEP, Z_S = 200.0, 1500.0, 100.0
Z_R = [30.0, 77.0]
RANGES = np.arange(1600.0, 4001.0, 40.0)

# Weak coupling: a 2% down-step reflects 2.27e-4 of the incident modal power,
# so a one-way solver is short of the exact answer by 0.001 dB of level and has
# a whole field to match. It is also the step whose transmitted energy stays
# inside a Pade-6 aperture: the source at mid-depth excites modes 1/3/5 of the
# 200 m guide (8/26/47 deg), and D2 = 204 m puts 34/32/33% of the transmitted
# flux on modes 1/3/5 of the 204 m guide, at the same 8/26/47 deg. Measured at
# D2 = 250 m instead, 5.9% of the flux lands on modes at 57 and 74 deg, outside
# that aperture, and RAM's median error rises from 0.069 to 0.167 dB. Both
# distributions are measurements; no mechanism is claimed for them here.
D2_WEAK = 204.0
# Strong coupling: 12.5% of the incident power comes back. Used to document
# what the one-way engines do when the reflected field is not negligible.
D2_STRONG = 178.0
# The reference is converged at this truncation (see the convergence test) and
# a 323 x 323 solve costs milliseconds.
MODE_REACH = 24.0

# RAM carries no spelling for a vacuum (``RAM.validate_inputs`` refuses every
# non-geoacoustic bottom), so the pressure-release floor reaches it as a
# near-massless half-space at the water's own sound speed — the same surrogate
# ``test_ram_pressure_release_waveguide_matches_dirichlet_modal_sum`` uses, and
# validated there against a true ``'vacuum'`` bottom through Kraken.
RHO_SOFT = 1e-4
_SOFT = BoundaryProperties(acoustic_type='half-space', sound_speed=C_W,
                           density=RHO_SOFT, attenuation=0.0)
_VACUUM = BoundaryProperties(acoustic_type='vacuum')


def _stepped_env(D2, bottom, ramp=0.5):
    """The stepped waveguide as a uacpy ``Environment``. ``D2 == D1`` removes
    the step and is the null control.

    The writers interpolate the bathymetry linearly between control points, so
    the discontinuity is written as two points ``2*ramp`` apart. RAM's answer
    moves by 0.086 dB of median as ``ramp`` opens from 0.5 m to 50 m, so the
    1 m ramp is not what the engine comparison measures.
    """
    if D2 == D1:
        bath = Bathymetry(ranges=np.array([0.0, 6000.0]), depths=np.array([D1, D1]))
    else:
        bath = Bathymetry(ranges=np.array([0.0, R_STEP - ramp, R_STEP + ramp, 6000.0]),
                          depths=np.array([D1, D1, D2, D2]))
    return Environment(
        bathymetry=bath,
        ssp=SoundSpeedProfile.from_pairs([(0.0, C_W), (max(D1, D2), C_W)]),
        bottom=bottom)


def _reference(D2, ranges=RANGES, mode_reach=MODE_REACH):
    """The coupled-mode TL over the ``Z_R`` x ``ranges`` receiver table."""
    return np.array([coupled_mode_step_tl(Z_S, zr, ranges, FREQ, D1, D2, R_STEP,
                                          C_W, mode_reach=mode_reach) for zr in Z_R])


def _abs_dtl(tl, ref):
    """(median, p90, max) of ``|model_TL - reference_TL|`` over the table."""
    d = np.abs(np.asarray(tl, dtype=float).reshape(ref.shape) - ref).ravel()
    return np.median(d), np.percentile(d, 90), np.max(d)


def _src_rcv(ranges=RANGES):
    return (Source(depths=Z_S, frequencies=FREQ),
            Receiver(depths=Z_R, ranges=ranges))


# ── the reference validates itself ──────────────────────────────────────────

def test_a_stepless_coupled_mode_solution_is_the_flat_dirichlet_modal_sum():
    """With ``D2 == D1`` the overlap matrix is the identity, the reflected
    amplitudes vanish and the coupled-mode field collapses to the closed-form
    range-independent modal sum of ``test_benchmarks_analytic``. Retaining the
    propagating modes only (``mode_reach = 1``) makes the two mode sets
    identical, and the fields then agree to 5.7e-13 dB over 121 ranges at three
    receiver depths — floating-point noise on a 40-76 dB field.

    Carrying the evanescent modes as well (``mode_reach = 6``) moves the field
    by 3.5e-5 to 7.7e-4 dB: that is the near-field tail of the source, which
    the propagating-only sum omits and the coupled-mode reference keeps. It is
    a real term, not a discrepancy, and it is four decades below the tolerance
    of any engine test here.
    """
    ranges = np.arange(300.0, 3301.0, 25.0)
    for z_r in (30.0, 45.0, 77.0):
        flat = dirichlet_modal_tl(Z_S, z_r, ranges, FREQ, D1, C_W)
        prop = coupled_mode_step_tl(Z_S, z_r, ranges, FREQ, D1, D1, R_STEP, C_W,
                                    mode_reach=1.0)
        assert np.max(np.abs(prop - flat)) < 1e-10, (
            f"z_r={z_r}: max|dTL|={np.max(np.abs(prop - flat)):.3e} dB")
        evan = coupled_mode_step_tl(Z_S, z_r, ranges, FREQ, D1, D1, R_STEP, C_W,
                                    mode_reach=6.0)
        assert np.max(np.abs(evan - flat)) < 5e-3, (
            f"z_r={z_r}: evanescent tail moved TL by "
            f"{np.max(np.abs(evan - flat)):.3e} dB")


def test_the_mode_matching_overlap_integrals_match_numerical_quadrature():
    """The closed form of ``overlap_matrix`` is the integral it claims to be.

    ``int_0^L sin(a z) sin(b z) dz`` is evaluated analytically as
    ``sin((a-b)L)/(2(a-b)) - sin((a+b)L)/(2(a+b))``, and the ``a == b`` limit
    is taken by hand. Neither energy conservation nor reciprocity can see an
    error in it: both survive *any* matrix ``C`` as long as pressure is matched
    with ``C`` and velocity with ``C^T``, because they follow from that
    pairing and not from the entries. The degenerate case cannot see it either
    — with ``D1 == D2`` the sum-frequency term is ``sin((l+m)*pi) = 0``. So the
    entries are checked directly, against Simpson quadrature of the actual mode
    products on 200001 depth points. Measured agreement 1e-12 absolute on
    entries of order 1.
    """
    from scipy.integrate import simpson
    for D1_, D2_ in ((200.0, 204.0), (200.0, 250.0), (200.0, 165.0), (200.0, 200.0)):
        n1, n2 = 9, 11
        C = overlap_matrix(D1_, D2_, n1, n2)
        z = np.linspace(0.0, min(D1_, D2_), 200001)
        Z1 = np.sqrt(2.0 / D1_) * np.sin(np.outer(np.arange(1, n1 + 1) * np.pi / D1_, z))
        Z2 = np.sqrt(2.0 / D2_) * np.sin(np.outer(np.arange(1, n2 + 1) * np.pi / D2_, z))
        quad = np.array([[simpson(Z2[l] * Z1[m], x=z) for m in range(n1)]
                         for l in range(n2)])
        err = np.max(np.abs(C - quad))
        assert err < 1e-9, f"D1={D1_} D2={D2_}: max|C - quadrature|={err:.3e}"


def test_the_matched_field_leaves_the_step_face_pressure_release():
    """The vertical step face carries ``p = 0``, and the matched field imposes it.

    A pressure-release bottom makes the *whole* boundary of the fluid domain
    pressure-release, the vertical face between ``D1`` and ``D2`` included. The
    rival convention — pressure matched on the overlap alone, velocity matched
    over the wide interval with the narrow side taken as zero — is a **rigid**
    face, and it conserves energy and satisfies reciprocity just as exactly, so
    neither of those tests can tell the two apart. What tells them apart is the
    face itself.

    Measured: the RMS of ``|p|`` over the face, relative to its RMS over the
    overlap, falls 1.64e-2 -> 1.03e-2 -> 5.97e-3 -> 2.38e-3 as ``mode_reach``
    goes 6 -> 12 -> 24 -> 48 at ``D2 = 204 m`` (and 7.1e-3 -> 4.8e-4 at
    ``D2 = 250 m``) — converging to the pressure release the geometry asks for.
    Under the rigid-face convention the same ratio plateaus at 9.7e-2 and
    2.2e-1 and never converges.

    An independent finite-difference solve of the Helmholtz equation on this
    stepped domain confirms the choice from outside the mode-matching
    framework: the soft-face junction converges to the FD answer (2.80e-2 ->
    7.34e-3 relative as the grid halves from 5 m to 1 m) while the rigid-face
    one plateaus at 1.0e-1. See ``round32/build32-coupledmode.md`` Sect. 3.5.

    That certification is obtained at ``D2 = 250 m`` and ``D2 = 165 m``, not at
    the ``D2 = 204 m`` benchmark step, where the FD cannot arbitrate the face
    at all: neither convention converges there and the two sit within a factor
    2 of each other (``round32/verify32-cm.md`` Sect. 4.3). At a 2% step the
    two junctions are very nearly the same operator — the reference fields they
    produce differ by 0.0218 dB median, against 0.356 dB at ``D2 = 250 m`` and
    0.639 dB at ``D2 = 178 m`` (``verify32-cm.md`` Sect. 6). So the face is
    certified where it is measurable and carried to the benchmark by the
    junction being the same construction, and at the benchmark it is not a
    material question. The ``D2 = 250 m`` leg asserted below is the one that
    carries the discrimination.
    """
    for D2, bound in ((D2_WEAK, 2e-2), (250.0, 5e-3)):
        ratios = []
        for reach in (6.0, 48.0):
            g1, km1 = dd_modes(D1, FREQ, C_W, reach)
            g2, km2 = dd_modes(D2, FREQ, C_W, reach)
            inc = (np.sqrt(2.0 / D1) * np.sin(g1 * Z_S)) * np.exp(1j * km1 * R_STEP) / np.sqrt(km1)
            _, T = step_scatter(D1, D2, km1, km2, inc,
                                np.zeros(len(km2), dtype=complex))
            face = np.linspace(D1, D2, 401)[1:]
            overlap = np.linspace(0.0, min(D1, D2), 801)[1:-1]
            p_face = (np.sqrt(2.0 / D2) * np.sin(np.outer(face, g2))) @ T
            p_over = (np.sqrt(2.0 / D2) * np.sin(np.outer(overlap, g2))) @ T
            ratios.append(np.sqrt(np.mean(np.abs(p_face)**2))
                          / np.sqrt(np.mean(np.abs(p_over)**2)))
        assert ratios[0] < 5e-2, f"D2={D2}: face/overlap at reach 6 = {ratios[0]:.3e}"
        assert ratios[1] < bound, f"D2={D2}: face/overlap at reach 48 = {ratios[1]:.3e}"
        assert ratios[1] < 0.5 * ratios[0], (
            f"D2={D2}: face pressure is not converging to zero "
            f"({ratios[0]:.3e} -> {ratios[1]:.3e}); the face is not being "
            f"held pressure-release")


def test_step_scattering_conserves_modal_power_flux():
    """In a lossless waveguide the reflected plus transmitted modal power flux
    equals the incident, summed over the propagating modes.

    Measured over the ladder below at the production truncation:
    ``|1 - R - T| <= 6.7e-16``, i.e. the residual is at the rounding floor of
    the linear solve. The stronger statement is on the junction's own
    scattering operator: with every propagating amplitude scaled to carry unit
    power, ``S`` is unitary to 4.4e-15, which conserves energy for *every*
    incident field rather than for the one this fixture launches.

    Energy conservation alone does not certify the step-face condition — the
    rival convention that treats the face as rigid conserves energy just as
    exactly. ``test_the_matched_field_leaves_the_step_face_pressure_release``
    and the finite-difference solve it cites are what separate the two.
    """
    for D2 in (201.0, 204.0, 222.0, 250.0, 275.0, 198.0, 192.0, 178.0, 165.0):
        R, T, residual = step_power_split(FREQ, D1, D2, C_W, Z_S, R_STEP,
                                          mode_reach=MODE_REACH)
        assert residual < 1e-12, f"D2={D2}: |1-R-T|={residual:.3e}"
        assert 0.0 < R < 1.0 and 0.0 < T < 1.0, (D2, R, T)
        S = step_scattering_matrix(FREQ, D1, D2, C_W, mode_reach=MODE_REACH)
        unitary = np.max(np.abs(S.conj().T @ S - np.eye(S.shape[0])))
        assert unitary < 1e-11, f"D2={D2}: max|S^H S - I|={unitary:.3e}"


def test_step_scattering_is_reciprocal():
    """Source and receiver exchange leaves the field unchanged.

    Two independent statements, both measured at the production truncation.
    (a) The power-normalised scattering matrix of the junction is symmetric to
    4.3e-15 — the mode-space form of reciprocity, and the one that covers every
    incident field. (b) In plane geometry, where a line source can sit on
    either side of the step and the exchange is well posed (the cylindrical
    reference pins its source to ``r = 0``), swapping source and receiver
    *across* the step reproduces the complex field to 3.5e-15 relative.

    A mismatched pair of projection domains — pressure and velocity matched
    over the same interval, or projected onto the same mode set — breaks (b)
    while leaving the field finite and plausible, which is what makes this
    worth asserting.
    """
    for D2 in (204.0, 222.0, 250.0, 196.0, 165.0):
        S = step_scattering_matrix(FREQ, D1, D2, C_W, mode_reach=MODE_REACH)
        assert np.max(np.abs(S - S.T)) < 1e-11, (
            f"D2={D2}: max|S - S^T|={np.max(np.abs(S - S.T)):.3e}")
        for (zs, xs, zr, xr) in [(100.0, 400.0, 30.0, 2600.0),
                                 (37.0, 900.0, 141.0, 4100.0),
                                 (163.0, 120.0, 88.0, 3300.0)]:
            zr = min(zr, D2 - 1.0)
            fwd = plane_step_field(zs, xs, zr, xr, FREQ, D1, D2, R_STEP, C_W,
                                   mode_reach=MODE_REACH)
            rev = plane_step_field(zr, xr, zs, xs, FREQ, D1, D2, R_STEP, C_W,
                                   mode_reach=MODE_REACH)
            rel = abs(fwd - rev) / max(abs(fwd), abs(rev))
            assert rel < 1e-11, f"D2={D2} ({zs},{xs})<->({zr},{xr}): rel={rel:.3e}"


def test_the_reference_field_has_stopped_moving_at_the_retained_mode_count():
    """Truncation convergence: the retained mode set reaches far enough into
    the evanescent spectrum that the field no longer depends on where it stops.

    ``mode_reach`` is a cutoff on vertical wavenumber (``gamma <= reach * k``),
    not a mode count, so the two segments keep the ``M1/M2 = D1/D2`` ratio a
    step needs. Measured at the benchmark case against ``mode_reach = 48``
    (M1 = 320, M2 = 326): reach 6 sits 1.0e-2 dB away, reach 12 7.1e-3, reach
    16 2.6e-3 and reach 24 — the production setting — 1.3e-3 dB, on a field
    running 47.9 to 69.8 dB. Convergence is oscillating rather than monotone,
    so the bound below is on the settled level and not on a per-step decrease.

    The residual is three decades below the tightest engine bound in this
    module, so no engine result here is truncation-limited.
    """
    settled = _reference(D2_WEAK, mode_reach=48.0)
    moved = {reach: np.max(np.abs(_reference(D2_WEAK, mode_reach=reach) - settled))
             for reach in (6.0, 12.0, 16.0, MODE_REACH)}
    assert moved[6.0] < 5e-2, moved
    assert moved[MODE_REACH] < 1e-2, moved
    assert moved[MODE_REACH] < moved[6.0], moved


def test_the_reference_refocuses_the_step_reflection_at_the_range_axis():
    """The wave the step sends back toward ``r = 0`` refocuses there and
    re-emerges outward, and the reference carries it with the right phase.

    JKPS 5.257 states the condition at the axis, and it is not ``b = 0``: the
    inward and outward parts combine into the regular ``J_0``, so an incoming
    amplitude is accompanied by an outgoing one at ``exp(-i pi/4)/exp(i pi/4)
    = -i``, the focal phase. Working in asymptotic range functions forces that
    factor to be supplied by hand, so it is checked against a solution of the
    same problem in exact Hankel and Bessel functions, where regularity at the
    origin is imposed by construction (``exact_hankel_step_tl``).

    Measured over ranges 1600-4000 m at ``z_r = 30 m``, propagating modes only
    on both sides:

    ==========  ==========================  ==========================
    step        asymptotic + axis vs exact  axis dropped vs exact
    ==========  ==========================  ==========================
    D2 = 178 m  0.0088 dB median            1.0127 dB median
    D2 = 165 m  0.0071 dB median            2.1135 dB median
    ==========  ==========================  ==========================

    and in front of the step (400-1400 m) 0.0104 against 1.9765 dB at
    ``D2 = 178 m``. The weakly-coupled cases cannot separate the two — at
    ``D2 = 250 m`` the term is worth 0.0043 dB, below the 0.003-0.06 dB the
    far-field asymptote itself costs — so the strong steps are what pins it.
    """
    for D2, floor in ((D2_STRONG, 0.5), (165.0, 1.0)):
        for ranges in (RANGES, np.arange(400.0, 1401.0, 20.0)):
            exact = exact_hankel_step_tl(Z_S, 30.0, ranges, FREQ, D1, D2, R_STEP, C_W)
            withax = coupled_mode_step_tl(Z_S, 30.0, ranges, FREQ, D1, D2, R_STEP,
                                          C_W, mode_reach=1.0, axis_return=True)
            without = coupled_mode_step_tl(Z_S, 30.0, ranges, FREQ, D1, D2, R_STEP,
                                           C_W, mode_reach=1.0, axis_return=False)
            good = np.median(np.abs(withax - exact))
            bad = np.median(np.abs(without - exact))
            assert good < 0.05, (
                f"D2={D2}, r={ranges[0]:.0f}-{ranges[-1]:.0f} m: the asymptotic "
                f"solution with the axis term sits {good:.4f} dB from the exact "
                f"Hankel solution")
            assert bad > floor, (
                f"D2={D2}: dropping the axis term only moves the field "
                f"{bad:.4f} dB from the exact solution, so this fixture does "
                f"not pin the axis condition")


def test_a_two_percent_step_is_weakly_coupled_and_a_ten_percent_step_is_not():
    """The regimes the engine benchmarks below rely on, pinned as numbers.

    ``D2 = 204 m`` (a 2% down-step) returns 2.27e-4 of the incident modal power
    — 0.001 dB of level — so a one-way solver discards nothing it needs.
    ``D2 = 178 m`` (an 11% up-step) returns 0.1246, i.e. 0.58 dB, which no
    one-way solver can produce. Up-steps couple far more strongly than
    down-steps of the same size because modes reach cutoff and reflect: at
    +/-2% the fractions are 2.2652e-4 (down) against 4.5140e-4 (up), and by
    +/-11% they are 6.18e-5 against 0.1246.

    ``R`` here is the fraction reflected out of *this fixture's* incident
    field, so it carries the incident modal phases and therefore depends on
    where the step is put: measured, ``R(204)`` runs 7.81e-5, 9.92e-5,
    2.27e-4, 3.01e-4, 2.85e-4, 3.47e-4 as ``r_step`` moves 500, 1000, 1500,
    2000, 2500, 3000 m. Every one of those is under the 1e-3 the bound asks
    for, but the number is a property of the fixture and not of the step
    alone. The phase-free statement is the mean reflectance over singly
    incident modes, ``sum_l |S_lm|^2`` averaged over ``m``: 5.68e-4 at
    ``D2 = 204``, 1.90e-3 at 208, 1.85e-4 at 250 and 1.37e-4 at 222 — so the
    junction's reflectivity is non-monotone in step size independently of any
    incident field, with the 25% down-step reflecting less than the 2% one.
    """
    R_weak, _, _ = step_power_split(FREQ, D1, D2_WEAK, C_W, Z_S, R_STEP,
                                    mode_reach=MODE_REACH)
    R_strong, _, _ = step_power_split(FREQ, D1, D2_STRONG, C_W, Z_S, R_STEP,
                                      mode_reach=MODE_REACH)
    assert R_weak < 1e-3, f"R(D2={D2_WEAK})={R_weak:.3e}"
    assert R_strong > 0.05, f"R(D2={D2_STRONG})={R_strong:.3e}"
    up = step_power_split(FREQ, D1, 2 * D1 - D2_WEAK, C_W, Z_S, R_STEP,
                          mode_reach=MODE_REACH)[0]
    assert up > R_weak, (up, R_weak)


def test_the_step_moves_the_reference_field_far_more_than_any_engine_bound():
    """The benchmark can see the step at all.

    Beyond a 2% step the coupled-mode field differs from the same waveguide
    with the step removed by 2.69 dB median / 10.16 dB p90 / 15.59 dB max — a
    4 m depth change reorganises the whole interference pattern because it
    shifts every modal wavenumber (mode 5 by 1.7e-3 rad/m, which is 4.2 rad
    over the 2.5 km beyond the step). The engine tolerances below sit at 0.35
    to 1.5 dB of median, so the step-removed control cannot pass them; each
    engine test asserts that explicitly.
    """
    med, p90, mx = _abs_dtl(_reference(D1), _reference(D2_WEAK))
    assert med > 1.5, f"median={med:.2f} dB"
    assert p90 > 5.0, f"p90={p90:.2f} dB"


# ── the engines measured against it ─────────────────────────────────────────

@pytest.mark.requires_binary
def test_ram_stepped_waveguide_matches_coupled_mode_reference():
    """RAM (mpiramS, the default PE backend) reproduces the two-way
    coupled-mode field beyond a weakly-coupled step.

    Measured: median 0.069 dB, p90 0.197 dB, max 0.550 dB over 2 depths x 61
    ranges of a 47.9-69.8 dB field. The bounds below are 5x and 4.6x that. The
    result is insensitive to the discretisation — median 0.069-0.080 and p90
    0.197-0.237 across dr 4-10 m, dz 0.25-0.5 m and Pade 6-8 — and byte-for-byte
    repeatable. ``max`` is deliberately unbounded: the deepest cells sit near
    interference nulls where a sub-metre shift dominates, and Pade 8 moves that
    one cell to 1.899 dB while its median stays at 0.076 dB.

    The grid is pinned rather than left to the Lytaev optimizer, which relaxes
    ``theta_max`` to 20 deg on this problem and lands at 2.74 dB median — the
    47 deg mode carrying a third of the transmitted energy is outside a 20 deg
    aperture.

    NULL CONTROL, asserted below: the same engine run with the step removed
    sits at 2.702 dB median / 10.135 dB p90 against this same reference, 7.7x
    and 11x outside the bounds. The tolerance therefore measures the step's
    physics and not merely that a PE produces a plausible waveguide field.
    """
    src, rcv = _src_rcv()
    ram = RAM(backend='mpiramS', dr=5.0, dz=0.25, np_pade=6, timeout=900)
    ref = _reference(D2_WEAK)

    med, p90, mx = _abs_dtl(ram.compute_tl(_stepped_env(D2_WEAK, _SOFT), src, rcv).db, ref)
    assert med < 0.35, f"median |dTL|={med:.3f} dB"
    assert p90 < 0.9, f"p90 |dTL|={p90:.3f} dB"

    n_med, n_p90, _ = _abs_dtl(ram.compute_tl(_stepped_env(D1, _SOFT), src, rcv).db, ref)
    assert n_med > 1.5 and n_p90 > 3.0, (
        f"step-removed run is only {n_med:.3f} dB median / {n_p90:.3f} dB p90 "
        f"from the stepped reference: this benchmark cannot tell the step's "
        f"physics from its absence")


@pytest.mark.requires_binary
def test_ramgeo_stepped_waveguide_matches_coupled_mode_reference():
    """RAM's ``ramgeo`` backend — Collins' range-dependent layered-fluid PE, a
    separate Fortran code from mpiramS with its own bathymetry handling
    (sediment layers parallel the seafloor) — lands on the same reference.

    Measured: median 0.078 dB, p90 0.255 dB, max 0.670 dB; repeatable. The
    bounds are 5x and 3.9x that. Two independent PE implementations agreeing
    with the coupled-mode field to under a tenth of a dB is what rules out a
    shared uacpy-side transcription error in the stepped-bathymetry writer.

    This leg does NOT run on the ``dr = 5 m`` it is handed. uacpy warns and
    reduces it: "dr=5 m exceeds the closest profile-section spacing (1 m), so
    the binary could consume only part of the 4-section environment ... dr has
    been reduced to 1 m" (one section per range step, ``ramgeo1.5.f:194-195``).
    The step in ``_stepped_env`` is written as two control points 1 m apart, so
    ramgeo's effective ``dr`` is pinned to that ramp width rather than to the
    constructor argument. The measurement is unaffected — 0.078/0.255 either
    way — but a change to ``ramp`` moves this backend's grid with it.

    NULL CONTROL, asserted below: step removed, 2.696 dB median / 10.141 dB
    p90 against the stepped reference.
    """
    src, rcv = _src_rcv()
    ram = RAM(backend='ramgeo', dr=5.0, dz=0.25, np_pade=6, timeout=900)
    ref = _reference(D2_WEAK)

    med, p90, mx = _abs_dtl(ram.compute_tl(_stepped_env(D2_WEAK, _SOFT), src, rcv).db, ref)
    assert med < 0.4, f"median |dTL|={med:.3f} dB"
    assert p90 < 1.0, f"p90 |dTL|={p90:.3f} dB"

    n_med, n_p90, _ = _abs_dtl(ram.compute_tl(_stepped_env(D1, _SOFT), src, rcv).db, ref)
    assert n_med > 1.5 and n_p90 > 3.0, (
        f"step-removed run is only {n_med:.3f} dB median / {n_p90:.3f} dB p90 "
        f"from the stepped reference: this benchmark cannot tell the step's "
        f"physics from its absence")


@pytest.mark.requires_binary
def test_bellhop_stepped_waveguide_matches_coupled_mode_reference():
    """Bellhop's coherent beam sum over the same step, against the same
    reference — and over a true ``'vacuum'`` bottom, which a ray model can
    spell and RAM cannot, so this leg carries no near-massless surrogate.

    Measured: median 0.832 dB, p90 2.910 dB, max 6.150 dB. That is a ray/beam
    model at 25 Hz in a 200 m guide (6 modes), so ~1 dB is the genuine method
    error — the same order as the 1.5 dB the ideal-wedge benchmark allows it in
    ``test_benchmarks_analytic``. Stable in beam count: median 0.829-0.892 and
    p90 2.63-2.95 over 2000-16000 beams, so the bounds below sit 1.35x and
    1.22x above the worst measurement.

    NULL CONTROL, asserted below: step removed, 2.615 dB median / 10.254 dB
    p90 — 2.18x and 2.85x outside the bounds. Bellhop's is the thinnest margin
    in this module on both sides at once, which is a statement about a ray
    model at 25 Hz and not about the reference: the bound cannot be loosened
    far before the null run would pass it, and it cannot be tightened far
    before the beam-count spread reaches it.
    """
    src, rcv = _src_rcv()
    bh = Bellhop(n_beams=8000, timeout=900)
    ref = _reference(D2_WEAK)

    med, p90, mx = _abs_dtl(bh.compute_tl(_stepped_env(D2_WEAK, _VACUUM), src, rcv).db, ref)
    assert med < 1.2, f"median |dTL|={med:.3f} dB"
    assert p90 < 3.6, f"p90 |dTL|={p90:.3f} dB"

    n_med, n_p90, _ = _abs_dtl(bh.compute_tl(_stepped_env(D1, _VACUUM), src, rcv).db, ref)
    assert n_med > 1.2 and n_p90 > 3.6, (
        f"step-removed run is only {n_med:.3f} dB median / {n_p90:.3f} dB p90 "
        f"from the stepped reference: this benchmark cannot tell the step's "
        f"physics from its absence")


@pytest.mark.requires_binary
def test_a_one_way_pe_loses_the_stepped_field_once_the_step_reflects():
    """The limit of the weak-coupling benchmark, as an executable statement.

    A one-way PE discards the reflected field by construction. Where that field
    is 2.27e-4 of the incident power RAM matches the two-way reference to
    0.069 dB of median; where it is 0.125 (an 11% up-step) the same engine on
    the same grid sits at 1.131 dB median / 2.986 dB p90, a 16x degradation,
    and at 0.177 (a 17.5% up-step, not run here) at 2.161 dB median. The
    reflected power fraction, not the depth change, is what predicts it: the
    25% *down*-step at D2 = 250 m reflects only 8.7e-5 and RAM holds 0.167 dB
    there.

    Measured on the other side of the step as well: at ranges 400-1400 m, in
    front of the discontinuity, RAM sits 0.098 dB from the two-way reference
    and 0.005 dB from the step-removed one — its segment-1 field is the flat
    field, carrying none of the returning wave. That 0.098 dB is what the
    reflected plus axis-refocused field is worth at this step.
    """
    src, rcv = _src_rcv()
    ram = RAM(backend='mpiramS', dr=5.0, dz=0.25, np_pade=6, timeout=900)
    weak, _, _ = _abs_dtl(
        ram.compute_tl(_stepped_env(D2_WEAK, _SOFT), src, rcv).db, _reference(D2_WEAK))
    strong, _, _ = _abs_dtl(
        ram.compute_tl(_stepped_env(D2_STRONG, _SOFT), src, rcv).db, _reference(D2_STRONG))
    assert weak < 0.35, f"weak-coupling median |dTL|={weak:.3f} dB"
    assert strong > 4 * weak, (
        f"strong-coupling median |dTL|={strong:.3f} dB is not clearly worse "
        f"than the weak-coupling {weak:.3f} dB")
