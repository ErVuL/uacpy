"""Tier-1 numerical benchmarks — validate model *output* against closed-form
analytic references, not merely that a run completes.

Unlike the rest of the suite (which checks shapes, finiteness, and contract
behaviour), these tests assert that the numbers uacpy produces match an
independent analytic solution to within a tight tolerance. They are the
foundation of the model-validation effort: a transcription error in a reader
or a vendored routine that still yields finite, plausible output would be
caught here.

References
----------
* Porter, *The KRAKEN Normal Mode Program* (SACLANTCEN SM-245, 2001): Pekeris
  normal-mode theory, the characteristic equation, and the modal-sum
  transmission loss (eq. 2.19). (Freely distributed with the Acoustics
  Toolbox; ``docs/KrakenNormalModeProgram_2001.pdf``.)
* Brekhovskikh & Lysanov, *Fundamentals of Ocean Acoustics*: Rayleigh
  plane-wave reflection coefficient.
* Buckingham & Tolstoy, JASA 87, 1990: the analytical 'ideal wedge'
  (ASA benchmark problem 1).

Convention
----------
The AT ``.env`` writer emits no explicit water-column density, so the AT
binaries (Kraken, Bounce, OASES) use their default ``rho_water = 1.0 g/cm^3``.
The analytic references below use the same value so the comparison is
apples-to-apples; the bottom density/speed are taken from the env exactly.
"""
import numpy as np
import pytest

pytestmark = [pytest.mark.benchmark, pytest.mark.requires_binary]

from uacpy import (Environment, SoundSpeedProfile, BoundaryProperties, Bathymetry,
                   Source, Receiver, Kraken, Scooter, RAM, Bounce, Bellhop, RunMode)

RHO_WATER = 1.0  # g/cm^3 — AT default when the .env omits a water density

# ── analytic references ─────────────────────────────────────────────────────

def pekeris_trapped_wavenumbers(f, depth, c_w, c_b, rho_w, rho_b):
    """Horizontal wavenumbers ``k_m`` of the trapped modes of a Pekeris
    waveguide — isovelocity water with a pressure-release surface over a fluid
    half-space (``c_b > c_w``).

    Mode shape ``sin(kz_w z)`` in the water, evanescent in the bottom. The
    interface conditions (continuity of pressure and of ``(1/rho) dphi/dz``)
    give the characteristic equation
        g(k) = (kz_w/rho_w)·cos(kz_w·D) + (kz_b/rho_b)·sin(kz_w·D) = 0,
    with kz_w = sqrt(k0_w^2 − k^2), kz_b = sqrt(k^2 − k0_b^2). Trapped modes
    have k0_b < k < k0_w. Returns k descending (mode 1 first), as Kraken does.
    """
    from scipy.optimize import brentq
    w = 2 * np.pi * f
    k0w, k0b = w / c_w, w / c_b

    def g(k):
        kzw = np.sqrt(max(k0w**2 - k**2, 0.0))
        kzb = np.sqrt(max(k**2 - k0b**2, 0.0))
        return (kzw / rho_w) * np.cos(kzw * depth) + (kzb / rho_b) * np.sin(kzw * depth)

    ks = np.linspace(k0b * (1 + 1e-9), k0w * (1 - 1e-9), 40000)
    vals = np.array([g(k) for k in ks])
    roots = []
    for i in range(len(ks) - 1):
        if vals[i] == 0.0:
            roots.append(ks[i])
        elif vals[i] * vals[i + 1] < 0:
            roots.append(brentq(g, ks[i], ks[i + 1], xtol=1e-13, rtol=1e-14))
    return np.array(sorted(roots, reverse=True))


def rayleigh_reflection(theta_grazing_deg, c_w, c_b, rho_w, rho_b):
    """Complex pressure reflection coefficient at a fluid–fluid interface for a
    plane wave incident from the water at grazing angle ``theta``.

        R = (rho_b·p_zw − rho_w·p_zb) / (rho_b·p_zw + rho_w·p_zb),
    p_zw = sin(theta)/c_w, p_zb = sqrt(1/c_b^2 − (cos(theta)/c_w)^2) (complex
    above the critical angle, where |R| = 1).
    """
    th = np.radians(np.asarray(theta_grazing_deg, dtype=float))
    p_zw = np.sin(th) / c_w
    p_zb = np.sqrt((1.0 / c_b**2) - (np.cos(th) / c_w) ** 2 + 0j)
    return (rho_b * p_zw - rho_w * p_zb) / (rho_b * p_zw + rho_w * p_zb)


def critical_grazing_deg(c_w, c_b):
    """Critical grazing angle (deg) for c_b > c_w: cos(theta_c) = c_w / c_b."""
    return np.degrees(np.arccos(c_w / c_b))


def lloyd_mirror_tl(ranges, z_s, z_r, f, c):
    """Transmission loss of the Lloyd-mirror field: a point source and its
    pressure-release surface image in a homogeneous medium,
        p(r) = e^{i k R1}/R1 − e^{i k R2}/R2,
    R1 = sqrt(r^2 + (z_r−z_s)^2), R2 = sqrt(r^2 + (z_r+z_s)^2), k = 2πf/c.
    TL = −20·log10|p| (1 m reference, matching the models' convention).
    """
    r = np.asarray(ranges, dtype=float)
    k = 2 * np.pi * f / c
    R1 = np.sqrt(r**2 + (z_r - z_s) ** 2)
    R2 = np.sqrt(r**2 + (z_r + z_s) ** 2)
    p = np.exp(1j * k * R1) / R1 - np.exp(1j * k * R2) / R2
    return -20.0 * np.log10(np.abs(p))


def pekeris_modal_tl(z_s, z_r, ranges, f, depth, c_w, c_b, rho_w, rho_b):
    """Analytic transmission loss of a Pekeris waveguide by the normal-mode sum
    (Porter, *KRAKEN Normal Mode Program* manual, eq. 2.19):
        p(r, z) ~ sqrt(2π/r) · Σ_m Z_m(z_s) Z_m(z) e^{i k_m r}/sqrt(k_m),
    TL = −20·log10|p|. Modes Z_m are sin(γ_m z) in the water with the bottom
    evanescent tail, depth-normalised so ∫ Z_m²/ρ dz = 1; rho_w = 1 makes the
    1/ρ(z_s) prefactor unity. k_m from the (validated) characteristic equation.
    """
    w = 2 * np.pi * f
    k1, k2 = w / c_w, w / c_b
    km = pekeris_trapped_wavenumbers(f, depth, c_w, c_b, rho_w, rho_b)
    gamma = np.sqrt(k1**2 - km**2)          # vertical wavenumber in the water
    beta = np.sqrt(km**2 - k2**2)           # bottom decay rate
    norm = ((depth / 2 - np.sin(2 * gamma * depth) / (4 * gamma))
            + np.sin(gamma * depth) ** 2 / (rho_b * 2 * beta))

    def Z(z):
        return np.sin(gamma * z) / np.sqrt(norm)

    out = []
    for r in np.atleast_1d(ranges).astype(float):
        p = np.sqrt(2 * np.pi / r) * np.sum(
            Z(z_s) * Z(z_r) * np.exp(1j * km * r) / np.sqrt(km))
        out.append(-20.0 * np.log10(np.abs(p)))
    return np.array(out)


def ideal_wedge_tl(r_from_source, z_r, z_s, R_s_apex, f, c, slope):
    """Analytic transmission loss of the ASA 'ideal wedge' (benchmark problem 1,
    Buckingham & Tolstoy, JASA 87, 1990): an isovelocity wedge with a
    pressure-release surface AND pressure-release bottom.

    Exact 2-D Dirichlet-wedge Green's function, separable in apex-centred polar
    coordinates (r, θ) with θ measured down from the surface:
        p_2D(r,θ) = (iπ/θ_w)·Σ_m sin(ν_m θ) sin(ν_m θ_s)·J_{ν_m}(k r_<)·H^{(1)}_{ν_m}(k r_>),
    ν_m = mπ/θ_w, θ_w = atan(slope). Converted to the 3-D point-source TL
    convention the propagation models use via the free-space factor
    |G_3D/G_2D| = √(k/2πr_horiz), referenced to 1/(4π) at 1 m.
    """
    from scipy.special import jv, hankel1
    k = 2 * np.pi * f / c
    theta_w = np.arctan(slope)
    theta_s = np.arctan(z_s / R_s_apex)
    r_s = np.hypot(R_s_apex, z_s)
    out = []
    for rs in np.atleast_1d(r_from_source).astype(float):
        R_apex = R_s_apex - rs                       # receiver horizontal range from apex
        theta = np.arctan(z_r / R_apex)
        r = np.hypot(R_apex, z_r)
        r_lo, r_hi = min(r, r_s), max(r, r_s)
        p = 0j
        m = 1
        while m * np.pi / theta_w <= 1.2 * k * r_hi:   # drop deeply evanescent modes
            nu = m * np.pi / theta_w
            p += (np.sin(nu * theta) * np.sin(nu * theta_s)
                  * jv(nu, k * r_lo) * hankel1(nu, k * r_hi))
            m += 1
        p2d = (1j * np.pi / theta_w) * p
        p3d = p2d * np.sqrt(k / (2 * np.pi * rs))      # 2-D line -> 3-D point source
        out.append(-20.0 * np.log10(np.abs(p3d * 4 * np.pi)))
    return np.array(out)


# ── shared Pekeris environment ──────────────────────────────────────────────

C_W, C_B, RHO_B, DEPTH, FREQ = 1500.0, 1800.0, 1.8, 100.0, 50.0

def _pekeris_env():
    return Environment(
        bathymetry=DEPTH,
        ssp=SoundSpeedProfile.from_pairs([(0.0, C_W), (DEPTH, C_W)]),
        bottom=BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=C_B, density=RHO_B, attenuation=0.0),
    )


# Modal-sum TL grid for the lossless `_pekeris_env` (so the trapped-mode
# analytic sum `pekeris_modal_tl` is the exact reference): one source, a
# depth x range receiver table. Reused to validate every range-independent
# field engine (Kraken / Scooter / OAST / RAM) against the same closed form.
_MODAL_ZS = 25.0
_MODAL_DEPTHS = [30.0, 50.0, 75.0]
_MODAL_RANGES = np.array([2000., 4000., 6000., 8000., 10000.])


def _modal_src_rcv():
    return (Source(depths=_MODAL_ZS, frequencies=FREQ),
            Receiver(depths=_MODAL_DEPTHS, ranges=_MODAL_RANGES))


def _modal_abs_dtl(tl):
    """Return (|model_TL − analytic_modal_sum_TL| flattened, analytic flattened)
    over the `_MODAL` depth x range grid."""
    tl = np.asarray(tl).reshape(len(_MODAL_DEPTHS), len(_MODAL_RANGES))
    ana = np.array([pekeris_modal_tl(_MODAL_ZS, zr, _MODAL_RANGES, FREQ, DEPTH,
                                     C_W, C_B, RHO_WATER, RHO_B)
                    for zr in _MODAL_DEPTHS])
    return np.abs(tl - ana).ravel(), ana.ravel()


# ── benchmarks ──────────────────────────────────────────────────────────────

def test_kraken_modes_match_pekeris_analytic():
    """Kraken eigenvalues k_m match the analytic Pekeris characteristic equation."""
    modes = Kraken(timeout=120).compute_modes(_pekeris_env(),
                                              Source(depths=20.0, frequencies=FREQ))
    k_num = np.sort(np.real(np.asarray(modes.k)))[::-1]          # descending
    k_ana = pekeris_trapped_wavenumbers(FREQ, DEPTH, C_W, C_B, RHO_WATER, RHO_B)

    assert k_ana.size >= 3, "analytic solver should find the trapped modes"
    # Mode count may differ by at most one near the cutoff (mesh sensitivity).
    assert abs(k_num.size - k_ana.size) <= 1, (k_num.size, k_ana.size)

    n = min(k_num.size, k_ana.size)
    # Kraken's Richardson-extrapolated eigenvalues match analytic to ~2.5e-8
    # (rel). atol=1e-5 is ~2000x that floor — tight enough to catch a real
    # scaling/units/reader bug, loose enough to survive mesh/platform drift.
    np.testing.assert_allclose(k_num[:n], k_ana[:n], atol=1e-5, rtol=0,
                               err_msg=f"k_num={k_num}\nk_ana={k_ana}")


def test_kraken_tl_matches_pekeris_modal_sum():
    """Absolute transmission loss. uacpy's Kraken field (modes -> field.exe ->
    .shd reader) must reproduce the analytic Pekeris normal-mode-sum TL
    (Porter Kraken manual eq. 2.19) across a depth x range table — this validates the field
    *assembly and reader*, not just the eigenvalues. Measured agreement is
    ~0.02 dB median / 0.07 dB max; atol=0.3 dB is a tight, robust bound."""
    src, rcv = _modal_src_rcv()
    d, _ = _modal_abs_dtl(Kraken(timeout=120).compute_tl(_pekeris_env(), src, rcv).tl)
    assert np.median(d) < 0.1, f"median |dTL|={np.median(d):.3f} dB"
    assert np.max(d) < 0.4, f"max |dTL|={np.max(d):.3f} dB"


def test_scooter_tl_matches_pekeris_modal_sum():
    """Scooter (finite-element wavenumber integration) absolute TL matches the
    analytic Pekeris normal-mode sum — two *independent exact* range-independent
    methods. Measured median ~0.03 dB / p90 ~0.13 dB; bounds are tight."""
    src, rcv = _modal_src_rcv()
    d, _ = _modal_abs_dtl(Scooter(timeout=120).compute_tl(_pekeris_env(), src, rcv).tl)
    assert np.median(d) < 0.2, f"median |dTL|={np.median(d):.3f} dB"
    assert np.percentile(d, 90) < 0.6, f"p90 |dTL|={np.percentile(d, 90):.3f} dB"


@pytest.mark.requires_oases
def test_oast_tl_matches_pekeris_modal_sum():
    """OAST (OASES wavenumber integration) absolute TL matches the analytic
    Pekeris modal sum over the strong-field cells (deep interference nulls,
    where TL is hyper-sensitive, are excluded). Measured median ~0.02 dB."""
    from uacpy.models.oases import OAST
    src, rcv = _modal_src_rcv()
    d, ana = _modal_abs_dtl(OAST(timeout=120).run(_pekeris_env(), src, rcv).tl)
    good = ana < 70.0
    assert good.sum() >= 10, "need enough strong-field cells"
    assert np.median(d[good]) < 0.3, f"median |dTL|={np.median(d[good]):.3f} dB"
    assert np.percentile(d[good], 90) < 1.0, f"p90 |dTL|={np.percentile(d[good], 90):.3f} dB"


def test_ram_tl_matches_pekeris_modal_sum():
    """RAM (parabolic equation) absolute TL agrees with the analytic Pekeris
    modal sum within PE accuracy. The PE under-resolves the steep modes of a
    shallow 100 m / 50 Hz waveguide, so ~2 dB median is the genuine method
    error — this validates RAM is physically correct in the ballpark (catches
    a grossly-wrong field), not a tight match. Measured median ~2.3 dB."""
    src, rcv = _modal_src_rcv()
    d, _ = _modal_abs_dtl(RAM(timeout=180).compute_tl(_pekeris_env(), src, rcv).tl)
    assert np.median(d) < 3.5, f"median |dTL|={np.median(d):.2f} dB"


def test_bellhop_ideal_wedge_matches_analytic():
    """ASA 'ideal wedge' (benchmark problem 1): uacpy reproduces the exact
    Buckingham-Tolstoy analytic solution. The ideal wedge has pressure-release
    surface AND bottom (a vacuum half-space), which uacpy models with
    ``BoundaryProperties(acoustic_type='vacuum')``; Bellhop runs the coherent
    field on the 2.86 deg up-slope. Bellhop is a ray/beam model at 25 Hz, so
    ~1 dB absolute agreement with the exact analytic field is expected —
    the interference structure (shape) is reproduced to ~0.4 dB."""
    f, c, slope, R_s, z_s, z_r = 25.0, 1500.0, 0.05, 4000.0, 100.0, 30.0
    r_src = np.array([500., 1000., 1500., 2000., 2500., 3000.])
    rr = np.linspace(0.0, 3800.0, 40)
    env = Environment(
        bathymetry=Bathymetry(ranges=rr, depths=200.0 - slope * rr),
        ssp=SoundSpeedProfile.from_pairs([(0.0, c), (200.0, c)]),
        bottom=BoundaryProperties(acoustic_type='vacuum'))
    tl_bh = np.asarray(Bellhop(n_beams=4000, timeout=180).compute_tl(
        env, Source(depths=z_s, frequencies=f),
        Receiver(depths=[z_r], ranges=r_src)).tl).ravel()
    tl_ana = ideal_wedge_tl(r_src, z_r, z_s, R_s, f, c, slope)

    diff = tl_bh - tl_ana
    assert np.median(np.abs(diff)) < 1.5, f"median |dTL|={np.median(np.abs(diff)):.2f} dB"
    assert np.max(np.abs(diff)) < 2.5, f"max |dTL|={np.max(np.abs(diff)):.2f} dB"
    # interference structure: residual after removing the constant ray-model bias
    assert np.std(diff) < 0.7, f"shape mismatch, std={np.std(diff):.2f} dB"


def test_bounce_reflection_matches_rayleigh():
    """Bounce |R(theta)| matches the analytic Rayleigh coefficient, including
    the total-internal-reflection plateau and the critical angle."""
    rc = Bounce(c_low=1400.0, c_high=20000.0, timeout=120).compute_reflection(
        _pekeris_env(), Source(depths=20.0, frequencies=FREQ),
        Receiver(depths=[50.0], ranges=[1000.0]))
    theta = np.asarray(rc.theta, dtype=float)
    R_num = np.abs(np.asarray(rc.R, dtype=float))
    R_ana = np.abs(rayleigh_reflection(theta, C_W, C_B, RHO_WATER, RHO_B))
    theta_c = critical_grazing_deg(C_W, C_B)                      # ≈ 33.56°

    # Below critical: total internal reflection, |R| ≈ 1.
    below = theta < theta_c - 3.0
    assert np.all(R_num[below] > 0.98), R_num[below]

    # Away from the critical knee, |R| matches Rayleigh tightly. Bounce's
    # coarse (~31-pt) angular grid puts the floor near 4.5e-3; atol=1e-2 sits
    # just above it.
    far = np.abs(theta - theta_c) > 2.0
    np.testing.assert_allclose(R_num[far], R_ana[far], atol=1e-2, rtol=0,
                               err_msg=f"theta={theta}\nnum={R_num}\nana={R_ana}")

    # Critical angle: first drop below 0.99 lands within ~3° of theory.
    drop = theta[R_num < 0.99]
    assert drop.size and abs(drop[0] - theta_c) < 3.0, (drop[:3], theta_c)


@pytest.mark.requires_oases
def test_oasr_reflection_matches_rayleigh():
    """OASR |R(theta)| matches the analytic Rayleigh coefficient (independent
    wavenumber-integration code path for the same physics)."""
    from uacpy.models.oases import OASR
    rc = OASR(timeout=120).run(_pekeris_env(),
                               Source(depths=20.0, frequencies=FREQ),
                               Receiver(depths=[50.0], ranges=[1000.0]),
                               run_mode=RunMode.REFLECTION)
    theta = np.asarray(rc.theta, dtype=float)
    R_num = np.abs(np.asarray(rc.R, dtype=float))
    R_ana = np.abs(rayleigh_reflection(theta, C_W, C_B, RHO_WATER, RHO_B))
    theta_c = critical_grazing_deg(C_W, C_B)

    # OASR's fine direct grid matches analytic to ~1e-6; atol=1e-3 is a tight,
    # meaningful bound (1000x the floor).
    far = (np.abs(theta - theta_c) > 2.0) & (theta > 1.0) & (theta < 89.0)
    np.testing.assert_allclose(R_num[far], R_ana[far], atol=1e-3, rtol=0,
                               err_msg=f"theta={theta}\nnum={R_num}\nana={R_ana}")


def test_bellhop_lloyd_mirror():
    """Bellhop coherent TL reproduces the analytic Lloyd-mirror interference
    pattern. A deep, impedance-matched bottom (c=c_w, rho=rho_w → R=0) removes
    the bottom path, leaving only the direct ray and its pressure-release
    surface image — the geometry a ray model is exact for."""
    c, z_s, z_r, f = C_W, 18.0, 18.0, 200.0
    env = Environment(
        bathymetry=3000.0,
        ssp=SoundSpeedProfile.from_pairs([(0.0, c), (3000.0, c)]),
        bottom=BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=c, density=RHO_WATER, attenuation=0.0))
    ranges = np.linspace(200.0, 2000.0, 120)
    tl_num = np.asarray(Bellhop(timeout=120).compute_tl(
        env, Source(depths=z_s, frequencies=f),
        Receiver(depths=[z_r], ranges=ranges)).tl).ravel()
    tl_ana = lloyd_mirror_tl(ranges, z_s, z_r, f, c)

    # Compare away from deep interference nulls (TL there is hyper-sensitive to
    # sub-wavelength geometry and not a useful discriminator).
    good = tl_ana < 55.0
    assert good.sum() >= 15, "need enough non-null samples"
    d = np.abs(tl_num - tl_ana)[good]
    assert np.median(d) < 2.0, f"median |dTL|={np.median(d):.2f} dB"
    assert np.percentile(d, 90) < 2.5, f"p90 |dTL|={np.percentile(d, 90):.2f} dB"
