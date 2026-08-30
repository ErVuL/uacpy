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


def dirichlet_modal_tl(z_s, z_r, ranges, f, depth, c):
    """Analytic transmission loss of an isovelocity waveguide that is
    pressure-release at the surface AND at the bottom — the range-independent
    limit of the ideal wedge, and the control for the wedge benchmark below.

    Both boundaries being Dirichlet makes every mode elementary: the water
    holds the whole mode (no evanescent bottom tail to normalise against), so
    gamma_m = m*pi/D exactly, k_m = sqrt(k^2 - gamma_m^2), and the depth-
    normalised shape is Z_m = sqrt(2/D)*sin(gamma_m z). Summed with the same
    far-field prefactor and 1 m TL reference as ``pekeris_modal_tl``, so a
    model checked against one is checked on the same convention as the other.
    """
    k = 2 * np.pi * f / c
    gamma = np.arange(1, int(np.floor(k * depth / np.pi)) + 1) * np.pi / depth
    km = np.sqrt(np.maximum(k**2 - gamma**2, 0.0))
    # A mode AT cutoff carries k_m = 0 and the far-field term below divides by
    # sqrt(k_m), so it has to go. Which side of cutoff the top mode lands on is
    # decided by the float value of k*depth/pi, and a waveguide sitting exactly
    # at a cutoff can put it one ULP inside: at 25 Hz in 150 m, k*depth/pi
    # evaluates to 5.000000000000001, so mode 5 is generated with 1/sqrt(k_m)
    # 6907x the largest other mode and swamps the sum. Testing `gamma < k` does
    # not catch that — gamma IS below k, by one ULP. Guard on the wavenumber
    # actually used instead: k_m > 1e-6*k drops only modes within 5e-13
    # (relative) of exact cutoff — three decades above what double precision
    # resolves k*depth/pi to, and three below any mode that carries energy.
    keep = km > 1e-6 * k
    gamma, km = gamma[keep], km[keep]
    Zs = np.sqrt(2.0 / depth) * np.sin(gamma * z_s)
    Zr = np.sqrt(2.0 / depth) * np.sin(gamma * z_r)
    out = []
    for r in np.atleast_1d(ranges).astype(float):
        p = np.sqrt(2 * np.pi / r) * np.sum(
            Zs * Zr * np.exp(1j * km * r) / np.sqrt(km))
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

def test_modal_propagation_loss_matches_pekeris_modal_sum():
    """``Modes.modal_propagation_loss`` reproduces the analytic Pekeris
    normal-mode sum in *absolute* dB — the prefactor, the 4*pi TL reference and
    the g/cm^3 density convention, none of which a relative test can see. A
    missing 4*pi/1000 in the prefactor is a flat +81.98 dB offset."""
    src, _ = _modal_src_rcv()
    modes = Kraken(timeout=120).compute_modes(_pekeris_env(), src)
    f = modes.modal_propagation_loss(
        source_depth=_MODAL_ZS,
        receiver_depths=np.array(_MODAL_DEPTHS),
        ranges_m=_MODAL_RANGES,
    )
    d, _ = _modal_abs_dtl(-20.0 * np.log10(np.abs(np.asarray(f.data))))
    assert np.median(d) < 0.1, f"median |dTL|={np.median(d):.3f} dB"
    assert np.max(d) < 0.4, f"max |dTL|={np.max(d):.3f} dB"


def test_kraken_modes_match_pekeris_analytic():
    """Kraken eigenvalues k_m match the analytic Pekeris characteristic equation."""
    modes = Kraken(timeout=120).compute_modes(_pekeris_env(),
                                              Source(depths=20.0, frequencies=FREQ))
    k_num = np.sort(np.real(np.asarray(modes.k)))[::-1]          # descending
    k_ana = pekeris_trapped_wavenumbers(FREQ, DEPTH, C_W, C_B, RHO_WATER, RHO_B)

    # 4 modes are trapped here: (2*f*D/c_w)*sqrt(1-(c_w/c_b)^2) = 3.7, rounded
    # up by the half-mode of the pressure-release/fluid pair. The assertion
    # allows one short so a mesh that drops the mode nearest cutoff still
    # exercises the eigenvalue comparison below.
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
    d, _ = _modal_abs_dtl(Kraken(timeout=120).compute_tl(_pekeris_env(), src, rcv).db)
    assert np.median(d) < 0.1, f"median |dTL|={np.median(d):.3f} dB"
    assert np.max(d) < 0.4, f"max |dTL|={np.max(d):.3f} dB"


def test_scooter_tl_matches_pekeris_modal_sum():
    """Scooter (finite-element wavenumber integration) absolute TL matches the
    analytic Pekeris normal-mode sum — two *independent exact* range-independent
    methods. Measured median ~0.03 dB / p90 ~0.13 dB; bounds are tight."""
    src, rcv = _modal_src_rcv()
    d, _ = _modal_abs_dtl(Scooter(timeout=120).compute_tl(_pekeris_env(), src, rcv).db)
    assert np.median(d) < 0.2, f"median |dTL|={np.median(d):.3f} dB"
    assert np.percentile(d, 90) < 0.6, f"p90 |dTL|={np.percentile(d, 90):.3f} dB"


@pytest.mark.requires_oases
def test_oast_tl_matches_pekeris_modal_sum():
    """OAST (OASES wavenumber integration) absolute TL matches the analytic
    Pekeris modal sum over the strong-field cells (deep interference nulls,
    where TL is hyper-sensitive, are excluded). Measured median ~0.02 dB."""
    from uacpy.models.oases import OAST
    src, rcv = _modal_src_rcv()
    d, ana = _modal_abs_dtl(OAST(timeout=120).run(_pekeris_env(), src, rcv).db)
    # The analytic field over this 3x5 grid runs 49.0-66.6 dB with a single
    # outlier at 75.2 dB, so 70 sits in the gap and excises exactly that one
    # null cell. Keeping it would let a sub-metre range difference dominate the
    # median of fifteen samples.
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
    d, _ = _modal_abs_dtl(RAM(timeout=180).compute_tl(_pekeris_env(), src, rcv).db)
    assert np.median(d) < 3.5, f"median |dTL|={np.median(d):.2f} dB"


# Pressure-release (Dirichlet-Dirichlet) waveguide — the flat control for the
# ideal-wedge benchmark below: same boundaries, same frequency, same source and
# receiver depths, slope removed. It is what isolates a wedge disagreement to
# the *slope* rather than to the boundary condition or the engine's absolute
# level. 200 m at 25 Hz holds six trapped modes; the source at mid-depth
# excites the odd three.
_PR_DEPTH, _PR_F, _PR_ZS, _PR_ZR = 200.0, 25.0, 100.0, 30.0
_PR_RANGES = np.arange(300.0, 3301.0, 25.0)

# RAM's decks carry no spelling for a vacuum (``RAM.validate_inputs`` refuses
# every non-geoacoustic bottom type), so the pressure-release floor reaches it
# as a near-massless half-space at the water's own sound speed: with c_b = c_w
# the Rayleigh coefficient collapses to (rho_b - rho_w)/(rho_b + rho_w) at
# *every* grazing angle, i.e. |R| = 0.99980 — 0.0017 dB per bounce from a true
# pressure release. The two tests below check that surrogate against a true
# ``'vacuum'`` bottom through the same analytic reference.
RHO_SOFT = 1e-4


def _pressure_release_env(bottom):
    return Environment(
        bathymetry=_PR_DEPTH,
        ssp=SoundSpeedProfile.from_pairs([(0.0, C_W), (_PR_DEPTH, C_W)]),
        bottom=bottom)


def _pr_src_rcv():
    return (Source(depths=_PR_ZS, frequencies=_PR_F),
            Receiver(depths=[_PR_ZR], ranges=_PR_RANGES))


def _pr_abs_dtl(tl):
    ana = dirichlet_modal_tl(_PR_ZS, _PR_ZR, _PR_RANGES, _PR_F, _PR_DEPTH, C_W)
    return np.abs(np.asarray(tl, dtype=float).ravel() - ana)


def test_a_waveguide_exactly_at_a_cutoff_does_not_generate_the_cutoff_mode():
    """A mode at cutoff carries ``k_m = 0`` and the far-field sum divides by
    ``sqrt(k_m)``, so a waveguide sitting exactly at a cutoff must give the
    same field as one a hair below it.

    25 Hz in 150 m is such a waveguide: ``k*depth/pi`` is 5 in exact
    arithmetic and ``5.000000000000001`` in floating point, so ``floor`` hands
    back mode 5 and the ``gamma < k`` test that used to guard this waves it
    through — ``gamma`` is below ``k``, by one ULP. The mode arrives with
    ``1/sqrt(k_m)`` 6907x the largest other mode's and swamps the sum. Latent
    when found: both live callers of ``dirichlet_modal_tl`` sit at
    ``k*depth/pi = 6.667``.

    The perturbation below is 1.5e-7 m of water depth, which no physical field
    can resolve, and the bound is 1e-4 dB against a measured 1.4e-6 dB. The
    receiver is at 45 m and not at this suite's usual 30 m: mode 5 of a 150 m
    guide has nodes every 30 m, so a receiver at 30 m multiplies the spurious
    mode by its own zero and hides it entirely.
    """
    z_s, z_r, ranges, f, c = 100.0, 45.0, np.array([300.0, 1000.0, 3300.0]), 25.0, C_W
    depth = 150.0
    assert 2 * np.pi * f / c * depth / np.pi == 5.000000000000001, (
        "this fixture is only a test while k*depth/pi rounds ABOVE the "
        "integer cutoff")
    at_cutoff = dirichlet_modal_tl(z_s, z_r, ranges, f, depth, c)
    below = dirichlet_modal_tl(z_s, z_r, ranges, f, depth * (1 - 1e-9), c)
    assert np.max(np.abs(at_cutoff - below)) < 1e-4, (
        f"TL moved by {np.max(np.abs(at_cutoff - below)):.4g} dB over a "
        f"1.5e-7 m depth change: the mode at cutoff is being generated")


def test_kraken_vacuum_waveguide_matches_dirichlet_modal_sum():
    """Kraken over a true ``'vacuum'`` bottom reproduces the closed-form
    Dirichlet-Dirichlet modal sum in absolute dB. This is what makes
    ``dirichlet_modal_tl`` usable as a reference for the other engines: an
    independent eigenvalue solver lands on it over 121 ranges with no free
    parameter. Measured median 0.014 dB / max 0.091 dB, so the bounds below sit
    7x and 4x above the measurement."""
    src, rcv = _pr_src_rcv()
    d = _pr_abs_dtl(Kraken(timeout=120).compute_tl(
        _pressure_release_env(BoundaryProperties(acoustic_type='vacuum')),
        src, rcv).db)
    assert np.median(d) < 0.1, f"median |dTL|={np.median(d):.3f} dB"
    assert np.max(d) < 0.4, f"max |dTL|={np.max(d):.3f} dB"


def test_ram_pressure_release_waveguide_matches_dirichlet_modal_sum():
    """RAM reproduces the same closed-form Dirichlet modal sum to 0.012 dB
    median — ~190x tighter than the ~2.3 dB the Pekeris case above allows it.
    The mid-depth source is what buys that: it excites only the odd modes, so
    the steepest one carrying energy has gamma_5/k = 0.75, a 49 deg ray well
    inside the Pade-6 aperture, where the Pekeris fixture's source excites the
    near-cutoff mode the PE under-resolves.

    The discretisation is pinned rather than left to the Lytaev optimizer: the
    default grid gives 3.55 dB median on this problem, and ``dr=5, dz=0.25,
    np_pade=6`` gives 0.012 dB median / 0.043 dB p90 / 0.185 dB max. The bounds
    below are 12x and 11x that measurement. ``max`` is deliberately not bounded
    — the field runs 40.7 to 74.4 dB and the deepest cells are hypersensitive
    to a sub-metre shift of a null (a nearby grid, dr=2.5, moves one cell to
    3.97 dB while its median stays at 0.064 dB).

    Paired with the Kraken test above this also validates the near-massless
    half-space as a stand-in for the vacuum bottom RAM cannot spell: both
    engines land on the same analytic field from opposite sides of that
    substitution.
    """
    src, rcv = _pr_src_rcv()
    d = _pr_abs_dtl(RAM(dr=5.0, dz=0.25, np_pade=6, timeout=300).compute_tl(
        _pressure_release_env(BoundaryProperties(
            acoustic_type='half-space', sound_speed=C_W,
            density=RHO_SOFT, attenuation=0.0)),
        src, rcv).db)
    assert np.median(d) < 0.15, f"median |dTL|={np.median(d):.3f} dB"
    assert np.percentile(d, 90) < 0.5, f"p90 |dTL|={np.percentile(d, 90):.3f} dB"


def test_bellhop_ideal_wedge_matches_analytic():
    """ASA 'ideal wedge' (benchmark problem 1): uacpy reproduces the exact
    Buckingham-Tolstoy analytic solution. The ideal wedge has pressure-release
    surface AND bottom (a vacuum half-space), which uacpy models with
    ``BoundaryProperties(acoustic_type='vacuum')``; Bellhop runs the coherent
    field on the 2.86 deg up-slope. Bellhop is a ray/beam model at 25 Hz, so
    ~1 dB absolute agreement with the exact analytic field is expected —
    the interference structure (shape) is reproduced to ~0.4 dB.

    Bellhop is the only engine here, and RAM is deliberately absent. A PE
    marches one way, and the ideal wedge is a two-way problem: at a receiver
    between the source and the apex the exact Green's function carries
    J_nu(k r_recv), and J_nu = (H1_nu + H2_nu)/2 is a radial *standing* wave —
    the apex-ward wave plus the wave the perfectly reflecting wedge returns
    down-slope from each mode's cutoff radius. Measured, on this fixture with
    the near-massless bottom of ``RHO_SOFT``: RAM sits 5.03 dB from the exact
    solution and 0.94 dB from the same series with the receiver-side J_nu
    replaced by H2_nu/2, i.e. it reproduces the one-way half and nothing else.
    That gap is structural, not a grid: it moves by 0.09 dB over dr 1-10 m,
    dz 0.1-1.0 m, Pade 4 and 8, and both the mpiramS and ramgeo backends, while
    RAM on the same boundaries with the slope removed matches
    ``dirichlet_modal_tl`` to 0.012 dB (the test above). Reversing the geometry
    does not help: down-slope the standing wave moves into the source-side
    factor J_nu(k r_src), and comparing the two analytic series against each
    other — source 2 km from the apex, receivers out to 4 km — leaves them
    3.68 dB apart in median, so there is no one-way half to match. Adding RAM would
    need a bound near 10 dB, which would assert nothing this fixture can catch.
    """
    f, c, slope, R_s, z_s, z_r = 25.0, 1500.0, 0.05, 4000.0, 100.0, 30.0
    r_src = np.array([500., 1000., 1500., 2000., 2500., 3000.])
    rr = np.linspace(0.0, 3800.0, 40)
    env = Environment(
        bathymetry=Bathymetry(ranges=rr, depths=200.0 - slope * rr),
        ssp=SoundSpeedProfile.from_pairs([(0.0, c), (200.0, c)]),
        bottom=BoundaryProperties(acoustic_type='vacuum'))
    tl_bh = np.asarray(Bellhop(n_beams=4000, timeout=180).compute_tl(
        env, Source(depths=z_s, frequencies=f),
        Receiver(depths=[z_r], ranges=r_src)).db).ravel()
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

    # Away from the critical knee, |R| matches Rayleigh to machine precision:
    # the analytic side is evaluated on Bounce's own theta grid, so the grid's
    # coarseness contributes no error and there is no discretisation term.
    # Measured residual 3.3e-16; atol=1e-9 is ~3e6 above that floor, tight
    # enough to catch a 1e-7 relative error in the interface parameters.
    far = np.abs(theta - theta_c) > 2.0
    np.testing.assert_allclose(R_num[far], R_ana[far], atol=1e-9, rtol=0,
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
    """Bellhop coherent TL reproduces the analytic Lloyd-mirror field across the
    whole sampled span. A deep, impedance-matched bottom (c=c_w, rho=rho_w → R=0)
    removes the bottom path, leaving only the direct ray and its pressure-release
    surface image — the geometry a ray model is exact for."""
    c, z_s, z_r, f = C_W, 18.0, 18.0, 200.0
    env = Environment(
        bathymetry=3000.0,
        ssp=SoundSpeedProfile.from_pairs([(0.0, c), (3000.0, c)]),
        bottom=BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=c, density=RHO_WATER, attenuation=0.0))
    ranges = np.linspace(200.0, 2000.0, 120)
    tl_ana = lloyd_mirror_tl(ranges, z_s, z_r, f, c)
    # No sample is excluded: with the path-length difference k(R2-R1) ≈ 2*k*z_s*z_r/r
    # the last constructive maximum falls at 2*k*z_s*z_r/pi = 173 m, before the span
    # starts, so tl_ana is strictly monotone here and holds no interference null.
    assert np.all(np.diff(tl_ana) > 0)
    assert 2 * (2 * np.pi * f / c) * z_s * z_r / np.pi < ranges[0]

    src = Source(depths=z_s, frequencies=f)
    rcv = Receiver(depths=[z_r], ranges=ranges)
    # Hat beams (beam_type='G') carry the geometric two-ray sum exactly; the fan is
    # narrowed to ±20°, which still spans the 10.2° steepest path (200 m receiver),
    # so 5001 rays resolve the near-grazing surface image at the far end.
    tl_hat = np.asarray(Bellhop(timeout=120, beam_type='G', n_beams=5001,
                                alpha=(-20.0, 20.0)).compute_tl(env, src, rcv).db).ravel()
    d_hat = np.abs(tl_hat - tl_ana)
    assert np.max(d_hat) < 0.05, f"max |dTL|={np.max(d_hat):.4f} dB"

    # The default geometric Gaussian beams spread energy across the beam width and
    # so do not reproduce this field exactly: the residual measures 0.7 dB at 200 m,
    # rises to 2.5 dB by 850 m and holds near that out to 2 km. Bounded, not pinned.
    d_gauss = np.abs(np.asarray(Bellhop(timeout=120).compute_tl(env, src, rcv).db)
                     .ravel() - tl_ana)
    assert np.max(d_gauss) < 3.0, f"max |dTL|={np.max(d_gauss):.2f} dB"
    assert np.median(d_gauss[ranges < 400.0]) < 1.0


def test_semicoherent_is_the_lloyd_shaded_incoherent_sum():
    """SEMICOHERENT_TL is the incoherent intensity sum with every launch
    amplitude pre-shaded by the Lloyd-mirror source-image factor
    ``sqrt(2)·|sin(omega·z_s·sin(alpha)/c)|`` (``bellhop.f90:276-278``;
    bellhop.md §4 "Semi-coherent is not a blend of the two" — the physical
    ``2·sin(k z_s sin(theta))`` image
    interference of COA §1, renormalised to unit mean square). On the
    impedance-matched Lloyd geometry only the direct ray and its surface
    image arrive, so both sums close analytically:

        I_inc  = 1/R1² + 1/R2²
        I_semi = w(α₁)/R1² + w(α₂)/R2²,   w = 2·sin²(k·z_s·sin α)

    with the straight-ray launch angles sin α₁ = (z_r−z_s)/R1,
    sin α₂ = (z_r+z_s)/R2 (the image path launches upward; |sin| makes the
    sign irrelevant). The range window keeps both weights in [0.14, 1.9] —
    no cell sits on a shading null where a dB comparison diverges — while
    the shading itself sweeps ~6 dB, so the assertion cannot pass on the
    unshaded sum. Hat beams carry the geometric two-path sum exactly (the
    coherent variant above measures 0.05 dB max); 0.5 dB is 10x that floor."""
    c, z_s, z_r, f = C_W, 20.0, 60.0, 200.0
    ranges = np.linspace(1000.0, 2500.0, 40)
    env = Environment(
        bathymetry=3000.0,
        ssp=SoundSpeedProfile.from_pairs([(0.0, c), (3000.0, c)]),
        bottom=BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=c, density=RHO_WATER,
                                  attenuation=0.0))
    src = Source(depths=z_s, frequencies=f)
    rcv = Receiver(depths=[z_r], ranges=ranges)
    model = Bellhop(timeout=120, beam_type='G', n_beams=5001,
                    alpha=(-20.0, 20.0))
    tl_inc = np.asarray(model.run(env, src, rcv,
                                  run_mode=RunMode.INCOHERENT_TL).db).ravel()
    tl_semi = np.asarray(model.run(env, src, rcv,
                                   run_mode=RunMode.SEMICOHERENT_TL).db).ravel()

    k = 2.0 * np.pi * f / c
    R1 = np.hypot(ranges, z_r - z_s)
    R2 = np.hypot(ranges, z_r + z_s)
    w1 = 2.0 * np.sin(k * z_s * (z_r - z_s) / R1) ** 2
    w2 = 2.0 * np.sin(k * z_s * (z_r + z_s) / R2) ** 2
    assert w1.min() > 0.1 and w2.min() > 0.1   # window stays off the nulls
    ana_inc = -10.0 * np.log10(1.0 / R1**2 + 1.0 / R2**2)
    ana_semi = -10.0 * np.log10(w1 / R1**2 + w2 / R2**2)

    d_inc = np.abs(tl_inc - ana_inc)
    assert np.max(d_inc) < 0.5, f"incoherent max |dTL|={np.max(d_inc):.3f} dB"
    d_semi = np.abs(tl_semi - ana_semi)
    assert np.max(d_semi) < 0.5, (
        f"semicoherent max |dTL|={np.max(d_semi):.3f} dB")
    # The shading in isolation, with the shared two-path spreading divided
    # out: it sweeps ~6 dB over this window, so matching it to 0.5 dB pins
    # the sqrt(2)|sin| factor itself, not just the intensity sum.
    shading = ana_semi - ana_inc
    assert np.ptp(shading) > 2.0
    assert np.max(np.abs((tl_semi - tl_inc) - shading)) < 0.5


def lloyd_mirror_pressure(ranges, z_s, z_r, f, c):
    """Complex Lloyd-mirror pressure, COA (Jensen et al.) Eq. (1.19)::

        p(r) = e^{i k R1}/R1 - e^{i k R2}/R2

    with the ``exp(-i w t)`` time factor suppressed and unit amplitude at 1 m.
    The Acoustics Toolbox reports the conjugate convention ``e^{i(w t - k r)}``
    (``KrakenField/EvaluateMod.f90:42``), so callers compare against
    ``conj`` of this.
    """
    r = np.asarray(ranges, dtype=float)
    k = 2 * np.pi * f / c
    R1 = np.sqrt(r**2 + (z_r - z_s) ** 2)
    R2 = np.sqrt(r**2 + (z_r + z_s) ** 2)
    return np.exp(1j * k * R1) / R1 - np.exp(1j * k * R2) / R2


def lloyd_mirror_pressure_2d(ranges, z_s, z_r, f, c):
    """Complex Lloyd-mirror pressure for a *line* source (2-D)::

        p(r) = (i/4) [ H0(k R1) - H0(k R2) ]

    The 2-D Green's function is the Hankel function, whose asymptotic form
    ``sqrt(2/pi k R) exp(i(kR - pi/4))`` carries a ``pi/4`` the 3-D
    ``exp(ikR)/R`` does not — the factor AT's purely real line-source scaling
    omits. Same ``exp(-i w t)`` convention as :func:`lloyd_mirror_pressure`.
    """
    from scipy.special import hankel1
    r = np.asarray(ranges, dtype=float)
    k = 2 * np.pi * f / c
    R1 = np.sqrt(r**2 + (z_r - z_s) ** 2)
    R2 = np.sqrt(r**2 + (z_r + z_s) ** 2)
    return 0.25j * (hankel1(0, k * R1) - hankel1(0, k * R2))


@pytest.mark.requires_binary
def test_bellhop_line_source_carries_the_2d_quarter_wave_phase():
    """A line source must match the 2-D Hankel solution, pi/4 included.

    Without it Bellhop sits +pi/4 from the exact 2-D field while its point
    source sits at the beam-approximation bias; the assertion is that both
    land on the *same* residual, which is what pins the pi/4 rather than
    absorbing it into a loose tolerance.
    """
    c, z_s, z_r, f = C_W, 25.0, 100.0, 100.0
    ranges = np.array([1000.0, 1500.0, 2000.0, 3000.0, 4000.0])
    env = Environment(
        bathymetry=4000.0,
        ssp=SoundSpeedProfile.from_pairs([(0.0, c), (4000.0, c)]),
        bottom=BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=c, density=RHO_WATER,
                                  attenuation=0.0))
    rcv = Receiver(depths=[z_r], ranges=ranges)
    m = Bellhop(timeout=120)
    w = lambda a: np.mod(a + 180.0, 360.0) - 180.0

    p_pt = np.asarray(m.run(env, Source(depths=z_s, frequencies=f,
                                        source_type='point'), rcv).data).ravel()
    p_ln = np.asarray(m.run(env, Source(depths=z_s, frequencies=f,
                                        source_type='line'), rcv).data).ravel()
    bias = np.mean(w(np.angle(
        p_pt / np.conj(lloyd_mirror_pressure(ranges, z_s, z_r, f, c)), deg=True)))
    line = np.mean(w(np.angle(
        p_ln / np.conj(lloyd_mirror_pressure_2d(ranges, z_s, z_r, f, c)), deg=True)))
    assert abs(line - bias) < 10.0, (
        f"line-source phase is {line - bias:.1f} deg from the point-source "
        f"beam bias; ~45 deg means the 2-D exp(-i pi/4) is missing")


@pytest.mark.requires_binary
@pytest.mark.parametrize('model_cls', [Bellhop, Scooter])
def test_complex_field_phase_matches_lloyd_mirror(model_cls):
    """Complex pressure must match the analytic Lloyd mirror in *phase*.

    ``lloyd_mirror_tl`` above validates |p| only, which is blind to an overall
    sign: AT's ``ScalePressure`` (``Bellhop/influence.f90:757-795``) sets
    ``const = -1`` for geometric beams and scales the field by ``const/sqrt(r)``,
    inverting Bellhop's ``.shd`` relative to the convention Kraken and Scooter
    report. TL plots look identical either way, so only a phase assertion
    catches it.
    """
    c, z_s, z_r, f = C_W, 25.0, 100.0, 100.0
    ranges = np.array([1000.0, 1500.0, 2000.0, 3000.0, 4000.0])
    env = Environment(
        bathymetry=4000.0,
        ssp=SoundSpeedProfile.from_pairs([(0.0, c), (4000.0, c)]),
        bottom=BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=c, density=RHO_WATER,
                                  attenuation=0.0))
    p_num = np.asarray(model_cls(timeout=120).run(
        env, Source(depths=z_s, frequencies=f),
        Receiver(depths=[z_r], ranges=ranges)).data).ravel()
    p_ana = np.conj(lloyd_mirror_pressure(ranges, z_s, z_r, f, c))

    ratio = p_num / p_ana
    ang = np.angle(ratio, deg=True)
    # Wrap into (-180, 180] before averaging so a near-pi error cannot
    # average away against its own wrap-around.
    ang = np.mod(ang + 180.0, 360.0) - 180.0
    assert np.abs(np.mean(ang)) < 20.0, (
        f"{model_cls.__name__} complex phase is {np.mean(ang):.1f} deg from the "
        f"analytic Lloyd mirror (~180 deg means an un-undone sign)")
    assert np.abs(np.mean(np.abs(ratio)) - 1.0) < 0.1, (
        f"{model_cls.__name__} amplitude ratio "
        f"{np.mean(np.abs(ratio)):.3f} != 1")


@pytest.mark.requires_binary
def test_scaled_cylindrical_removes_exactly_the_spreading_term():
    """``source_type='scaled'`` is 'point with cylindrical spreading removed'.

    ``KrakenField/field.f90:76`` and ``Matlab/Scooter/fieldsco.m:133`` define
    it that way, so
    ``p_scaled = p_point * sqrt(r)`` and therefore
    ``TL_point - TL_scaled = 10*log10(r)`` exactly. Checked through the model,
    not just the transform helper.
    """
    c = C_W
    env = Environment(
        bathymetry=200.0,
        ssp=SoundSpeedProfile.from_pairs([(0.0, c), (200.0, c)]),
        bottom=BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=1800.0, density=1.8,
                                  attenuation=0.5))
    ranges = np.array([1000.0, 2000.0, 4000.0])
    rcv = Receiver(depths=[100.0], ranges=ranges)

    def tl(source_type):
        return np.asarray(Scooter(timeout=120).run(
            env, Source(depths=50.0, frequencies=100.0,
                        source_type=source_type), rcv).db).ravel()

    np.testing.assert_allclose(tl('point') - tl('scaled'),
                               10.0 * np.log10(ranges), atol=0.01)


@pytest.mark.requires_binary
class TestBounceReflectsTheSeabedNotTheOcean:
    """BOUNCE shoots the impedance from the bottom half-space up through every
    acoustic medium and forms R at the top of medium 1
    (``Kraken/bounce.f90:178-201``), with the reference speed taken from the
    top half-space when it is 'A' and hardcoded to 1500 otherwise. So a *seabed* reflection coefficient needs the
    water column absent and an 'A' top at the water speed — which is exactly
    what doc/bounce.htm prescribes.
    """

    C_B, RHO_B, C_W_ALT = 1700.0, 1.8, 1450.0

    def _rc(self, water_depth, c_water):
        """``(theta_deg, |R|, phase_deg)``. The phase is the discriminating
        quantity: a lossless isovelocity column near BOUNCE's reference speed
        is nearly impedance-matched, so the water column it must not contain
        barely moves |R| but rotates the phase substantially."""
        from uacpy.models import Bounce
        env = Environment(
            bathymetry=float(water_depth),
            ssp=SoundSpeedProfile.from_pairs(
                [(0.0, c_water), (float(water_depth), c_water)]),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=self.C_B, density=self.RHO_B,
                                      attenuation=0.0))
        r = Bounce(timeout=300).run(
            env, Source(depths=1.0, frequencies=200.0),
            Receiver(depths=[1.0], ranges=[1000.0]))
        full = r.metadata['full_result']
        return (np.asarray(full['theta'], dtype=float),
                np.asarray(full['R'], dtype=float),
                np.asarray(full['phi'], dtype=float))

    @staticmethod
    def _wrap(deg):
        return (np.asarray(deg) + 180.0) % 360.0 - 180.0

    def test_reflection_coefficient_is_independent_of_water_depth(self):
        """A seabed RC cannot depend on how much water sits above it."""
        ta, ra, pa = self._rc(100.0, self.C_W_ALT)
        tb, rb, pb = self._rc(400.0, self.C_W_ALT)
        for th in (20.0, 45.0, 60.0, 75.0):
            ia = int(np.argmin(np.abs(ta - th)))
            ib = int(np.argmin(np.abs(tb - th)))
            dphase = float(self._wrap(pa[ia] - pb[ib]))
            assert abs(dphase) < 5.0, (
                f"phase at {th:.0f} deg moves {dphase:.1f} deg between "
                f"D=100 m and D=400 m — the water column is inside the "
                f"reflection coefficient")
            assert float(ra[ia]) == pytest.approx(float(rb[ib]), abs=0.02)

    def test_matches_rayleigh_at_a_water_speed_other_than_1500(self):
        """1500 m/s is BOUNCE's hardcoded fallback reference speed, so it is
        the one water speed that cannot expose a missing 'A' top."""
        theta, r, phi = self._rc(100.0, self.C_W_ALT)
        exact = rayleigh_reflection(theta, self.C_W_ALT, self.C_B,
                                    RHO_WATER, self.RHO_B)
        keep = (theta > 5.0) & (theta < 85.0)
        mag_err = np.abs(r[keep] - np.abs(exact[keep]))
        phase_err = np.abs(self._wrap(
            phi[keep] - np.degrees(np.angle(exact[keep]))))
        assert np.median(mag_err) < 0.03, (
            f"median ||R| error {np.median(mag_err):.3f} vs analytic Rayleigh "
            f"at c_w={self.C_W_ALT} m/s")
        assert np.median(phase_err) < 10.0, (
            f"median phase error {np.median(phase_err):.1f} deg vs analytic "
            f"Rayleigh at c_w={self.C_W_ALT} m/s")
