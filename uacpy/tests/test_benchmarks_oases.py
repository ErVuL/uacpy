"""Tier-1 numerical benchmarks for the OASES family — OASP, OASN, OASS and
OASSP output against references that do not come from OASES.

``test_benchmarks_analytic.py`` pins OAST (the narrowband wavenumber
integrator) to the Pekeris modal sum. The four programs here share OAST's
kernel but return different products — a broadband ``H(f)``, a noise
covariance, a reverberation loss and a scattered-field realisation — and
each product needs a check that its *numbers* are right, not only that
they are finite. Every
test below compares against a closed form, or an integral of one, evaluated
in this file:

* OASP: the Pekeris normal-mode sum (Porter, *KRAKEN Normal Mode Program*,
  eq. 2.19), reused from ``test_benchmarks_analytic``.
* OASN: the surface-noise cross-spectral density of a homogeneous half-space
  (Cron & Sherman, JASA 34, 1962; Jensen, Kuperman, Porter & Schmidt,
  *Computational Ocean Acoustics* §9.2.3, the dipole-sheet limit).
* OASS / OASSP: the first-order small-perturbation (Rayleigh–Rice) bistatic
  cross-section of a pressure-release rough interface, integrated over the
  interface (Thorsos & Jackson, JASA 86, 1989, for the 2-D result; the 3-D
  form is derived in :func:`dirichlet_spm_scattered_intensity`).

Every bound is a multiple of a value measured on 2026-09-19 and recorded in
the fixer ledger; none was loosened to pass. Each file-level reference is
scale-sensitive somewhere: the OASN coherence is normalised, so its absolute
level is pinned separately — see the test docstrings.
"""
import warnings

import numpy as np
import pytest

pytestmark = [pytest.mark.benchmark, pytest.mark.requires_binary,
              pytest.mark.requires_oases]

from numpy import trapezoid

from uacpy import (Environment, SoundSpeedProfile, BoundaryProperties, Source,
                   Receiver, RunMode)
from uacpy.tests.test_benchmarks_analytic import (
    pekeris_modal_tl, _pekeris_env, _modal_src_rcv, _MODAL_ZS, _MODAL_DEPTHS,
    _MODAL_RANGES, FREQ, DEPTH, C_W, C_B, RHO_WATER, RHO_B)


def _quiet(fn, *args, **kwargs):
    """Run an OASES model with its licence / grid-substitution warnings
    muted; the tests here assert on numbers, and each warning has its own
    test elsewhere."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


# ── OASP: broadband H(f) reduced to one bin, against the modal sum ──────────

def test_oasp_broadband_bin_matches_pekeris_modal_sum_at_the_bin_frequency():
    """OASP is OAST's kernel swept over an FFT frequency ladder, so the bin
    of its broadband ``H(f)`` nearest the OAST benchmark frequency must
    reproduce the analytic Pekeris modal sum in absolute dB, on the same
    3 x 5 depth x range grid ``test_oast_tl_matches_pekeris_modal_sum`` uses.

    The ladder does not land on 50 Hz: the realised bin is 49.922 Hz, and the
    modal sum is evaluated *at that frequency* — compared at 50.000 Hz the
    same field reads 0.36 dB median / 3.99 dB max off, because a 0.08 Hz
    shift over 10 km moves the interference pattern, not because OASP is
    wrong. Measured at the bin: median 0.0068 dB, p90 0.023 dB over all 15
    cells, max 0.29 dB at the one 79 dB null cell; over the strong-field
    cells (analytic TL < 70 dB, the same mask as the OAST test) p90 is
    0.017 dB. Bounds sit 15x and 18x above the measurement; a field off by a
    factor 2 is 6.02 dB in every cell.
    """
    from uacpy.models import OASP
    src, rcv = _modal_src_rcv()
    field = _quiet(OASP(n_time_samples=512, timeout=300).run,
                   _pekeris_env(), src, rcv, run_mode=RunMode.BROADBAND,
                   frequencies=np.linspace(40.0, 60.0, 21))
    f_axis = np.asarray(field.coords['frequency'], dtype=float)
    ib = int(np.argmin(np.abs(f_axis - FREQ)))
    f_bin = float(f_axis[ib])
    assert abs(f_bin - FREQ) < 0.5, f"nearest bin {f_bin:.3f} Hz is not near {FREQ}"
    h = np.asarray(field.data)[:, :, ib]                  # (n_depth, n_range)
    tl = -20.0 * np.log10(np.abs(h))
    ana = np.array([pekeris_modal_tl(_MODAL_ZS, zr, _MODAL_RANGES, f_bin, DEPTH,
                                     C_W, C_B, RHO_WATER, RHO_B)
                    for zr in _MODAL_DEPTHS])
    d = np.abs(tl - ana).ravel()
    good = ana.ravel() < 70.0
    assert good.sum() >= 10, "need enough strong-field cells"
    assert np.median(d[good]) < 0.1, f"median |dTL|={np.median(d[good]):.3f} dB"
    assert np.percentile(d[good], 90) < 0.3, (
        f"p90 |dTL|={np.percentile(d[good], 90):.3f} dB")


# ── OASN: surface-generated noise in an infinitely deep ocean ───────────────

# 100 Hz in isovelocity water over a TRANSPARENT half-space (same speed and
# density as the water, no absorption) — the plane-wave reflection
# coefficient is identically zero, so nothing returns from the bottom and
# the receivers see the sheet of surface sources as if the ocean were
# infinitely deep. The array sits 200-400 m down: 13-27 wavelengths from the
# surface, so the evanescent part of the sheet's field (which falls as
# exp(-2 q z)) is gone, and 2.5 m spacing samples the coherence at 0.17 λ.
_NOISE_F = 100.0
_NOISE_DEPTH = 2000.0
_NOISE_LEVEL_DB = 60.0
_NOISE_ARRAY = np.arange(200.0, 400.1, 2.5)


def dipole_sheet_vertical_coherence(k_dz):
    """Normalised cross-spectral density of surface-generated noise between
    two points separated vertically by ``dz`` in a homogeneous half-space
    under a pressure-release surface, ``x = k dz``.

    A sheet of uncorrelated monopoles just below a pressure-release surface
    radiates as dipoles (JKPS §9.2.3), and the intensity per unit solid angle
    at depth is ∝ cos θ, θ from the vertical (Cron & Sherman's surface-noise
    model — the source's cos²θ dipole pattern times the 1/cos θ growth of
    sheet area per solid angle). With ``u = cos θ``,
        C(x) = ∫_0^1 u e^{ixu} du / ∫_0^1 u du = 2 [(e^{ix} − 1)/x² − i e^{ix}/x],
    C(0) = 1. The sign of the imaginary part follows OASES's ``.xsm``
    convention as uacpy reads it: ``Im C_12 > 0`` for the deeper hydrophone
    second (measured), i.e. ``x = k (z_2 − z_1)``.
    """
    x = np.asarray(k_dz, dtype=float)
    out = np.ones(x.shape, dtype=complex)
    nz = x != 0.0
    xs = x[nz]
    e = np.exp(1j * xs)
    out[nz] = 2.0 * ((e - 1.0) / xs**2 - 1j * e / xs)
    return out


@pytest.fixture(scope='module')
def oasn_halfspace_covariance():
    from uacpy.models import OASN
    env = Environment(
        water_density=RHO_WATER, bathymetry=_NOISE_DEPTH,
        ssp=SoundSpeedProfile.from_pairs([(0.0, C_W), (_NOISE_DEPTH, C_W)]),
        bottom=BoundaryProperties(acoustic_type='half-space', sound_speed=C_W,
                                  density=RHO_WATER, attenuation=0.0))
    cov = _quiet(OASN(surface_noise_level=_NOISE_LEVEL_DB, timeout=300)
                 .compute_covariance, env,
                 Source(depths=5.0, frequencies=_NOISE_F),
                 Receiver(depths=_NOISE_ARRAY, ranges=[0.0]))
    c = np.asarray(cov.covariance)
    assert c.shape == (1, _NOISE_ARRAY.size, _NOISE_ARRAY.size), c.shape
    return c[0]


def test_oasn_surface_noise_level_in_an_infinitely_deep_ocean_is_the_source_level(
        oasn_halfspace_covariance):
    """``oasn.tex`` Block VI defines ``SSLEV`` as "the acoustic pressure the
    same source distribution would yield in an infinitely deep ocean", and
    the transparent-bottom fixture *is* that ocean, so every diagonal element
    of the covariance must read ``10**(SSLEV/10)``. This is the
    scale-sensitive half of the OASN check: the coherence test below is
    normalised and cannot see a factor. Measured −0.019 .. −0.021 dB over the
    81 sensors (the finite λ/30 sheet depth OASN uses, ``oasnun22.f:515``,
    accounts for −0.031 dB of that in exact quadrature); the 0.2 dB bound is
    10x the measurement, and a factor 2 in power is +3.01 dB.
    """
    diag_db = 10.0 * np.log10(np.real(np.diag(oasn_halfspace_covariance)))
    err = diag_db - _NOISE_LEVEL_DB
    assert np.max(np.abs(err)) < 0.2, (
        f"diagonal is {err.min():.3f}..{err.max():.3f} dB from SSLEV")


def test_oasn_vertical_coherence_matches_the_dipole_sheet_closed_form(
        oasn_halfspace_covariance):
    """The complex coherence ``C_ij / sqrt(C_ii C_jj)`` of the OASN covariance
    along a vertical array must follow the half-space surface-noise closed
    form of :func:`dipole_sheet_vertical_coherence` at every pair. Measured
    over the 3240 upper-triangle pairs (k·dz up to 84): |Δ| median 0.0042,
    p90 0.0065, max 0.0079; an exact finite-sheet-depth quadrature reference
    sits within 0.003 of the closed form. Bounds are 5x and 6x the
    measurement. A wrong sign convention on the imaginary part reads as
    |Δ| ≈ 2·|Im C| ≈ 1.7 at kd = π, so the sign is pinned too.
    """
    c = oasn_halfspace_covariance
    assert np.max(np.abs(c - c.conj().T)) == 0.0, "OASN covariance is Hermitian"
    d = np.real(np.diag(c))
    coh = c / np.sqrt(np.outer(d, d))
    k = 2 * np.pi * _NOISE_F / C_W
    ref = dipole_sheet_vertical_coherence(
        k * (_NOISE_ARRAY[np.newaxis, :] - _NOISE_ARRAY[:, np.newaxis]))
    iu = np.triu_indices(_NOISE_ARRAY.size, 1)
    err = np.abs(coh - ref)[iu]
    assert np.median(err) < 0.02, f"median |dC|={np.median(err):.4f}"
    assert np.max(err) < 0.05, f"max |dC|={np.max(err):.4f}"


# ── OASS / OASSP: a rough pressure-release interface, first-order ───────────

# Isovelocity water, no waveguide: the rough interface is the only reflector.
# For OASS the rough interface is the sea surface and the seabed is a
# transparent half-space (R = 0 at every angle); for OASSP — whose binary
# reads its scattering interface from the first .rhs record and so only
# accepts a rough BOTTOM — the geometry is mirrored: an infinite water
# column above (water upper half-space, no sea surface) and a rough
# pressure-release ('vacuum') seabed. Both are the same Dirichlet
# scattering problem seen from opposite sides. The mean field carries a
# 1e-4 m roughness — enough for SCTRHS to write boundary operators (it skips
# ROUGH² < 1e-10, oaseun31.f:2310), far too little to change the coherent
# field (coherent loss ∝ (2 k σ sin θ)² ≈ 1e-7) — while the scattering deck
# carries the 0.5 m rms below, so the run is the Born approximation the
# first-order reference assumes.
_SCAT_F = 200.0
_SCAT_H = 0.5           # rms roughness (m) of the scattering deck
_SCAT_L = 10.0          # Gaussian correlation length (m)
_SCAT_HEIGHT = 50.0     # source height above the rough interface (m)
_SCAT_WATER = 200.0
# OASS and OASP evaluate a UNIFORM range axis and interpolate a
# non-uniform request in dB (measured: a 100 m / 2500 m mixed grid read
# 1.2 dB off at 300 m), and OASS's reverberation synthesis is accurate over
# the first ~40 % of the receiver window only (measured −0.1 dB at 37 %,
# −0.4 dB at 60 %, −2 to −5 dB at the window end — see the fixer ledger). So
# the receivers run uniformly to 4 km and the assertions stop at 1.5 km.
_SCAT_RANGES = np.linspace(200.0, 4000.0, 39)
_SCAT_ASSERTED = _SCAT_RANGES <= 1500.0
_SCAT_MEAN_FIELD_ROUGHNESS = 1e-4


def gaussian_roughness_spectrum_2d(K, h, L):
    """Two-dimensional isotropic roughness spectrum of a Gaussian-correlated
    surface, ``<ζ(x) ζ(x+s)> = h² exp(−s²/(2L²))``, normalised so that
    ``∫ W d²K = h²``: ``W(K) = h² L²/(2π) · exp(−K² L²/2)``.

    The exponent is OASES's own: its Gaussian branch writes
    ``P(η) = sqrt(2π) L exp(−L² η²/2)`` (``oaseun31.f``, ``function p``),
    the 1-D transform of the same correlation function.
    """
    return h**2 * L**2 / (2 * np.pi) * np.exp(-0.5 * K**2 * L**2)


def dirichlet_spm_scattered_intensity(r, z_s, z_r, f, c, h, L, n=1000,
                                      span=8.0):
    """Mean scattered intensity ``E|p_s|²`` at horizontal range ``r`` and
    height ``z_r`` above a rough pressure-release plane, from a unit point
    source (``p = e^{ikR}/R``) at height ``z_s`` — the first-order
    small-perturbation result integrated over the whole plane.

    First-order SPM for a Dirichlet surface ``z = ζ(x)``: the boundary
    condition ``p(x, ζ) = 0`` expanded to first order gives the scattered
    field's spectrum ``S(K) = −2i q_i Z(K − K_i)`` for an incident plane wave
    of unit amplitude with horizontal wavenumber ``K_i`` and vertical
    wavenumber ``q_i = k sin θ_i`` (``Z`` the surface's Fourier transform).
    Stationary phase in the direction ``(K_s, q_s)`` and the ensemble
    average ``E|Z(K)|² = A W(K)/(2π)²`` over a patch of area ``A`` give the
    bistatic cross-section per unit area
        σ(θ_i, θ_s, ΔK) = 4 k⁴ sin²θ_i sin²θ_s W(|K_s − K_i|),
    (Thorsos & Jackson 1989 is the 2-D analogue, ``4 k³ ... W_1``). With
    spherical spreading on both legs,
        E|p_s|²(r, z_r) = ∬ σ / (R_i² R_s²) dA.
    The integral is taken on a tan-stretched grid concentrated under the
    source and the receiver; ``n=700/1400/2800`` agree to 0.0001 dB.
    """
    k = 2 * np.pi * f / c
    D = max(z_s, z_r, 25.0)
    u = np.linspace(-1.0, 1.0, n)

    def stretch(u, a, b, conc=1.4):
        return a + (b - a) * (0.5 + 0.5 * np.tan(conc * u) / np.tan(conc))

    xs = np.concatenate([stretch(u, -span * D, r / 2),
                         stretch(u, r / 2, r + span * D)[1:]])
    ys = stretch(u, -span * D, span * D)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    Ri = np.sqrt(X**2 + Y**2 + z_s**2)
    Rs = np.sqrt((X - r)**2 + Y**2 + z_r**2)
    sin_i, sin_s = z_s / Ri, z_r / Rs
    dK = np.hypot(k * (r - X) / Rs - k * X / Ri, -k * Y / Rs - k * Y / Ri)
    integrand = (4 * k**4 * sin_i**2 * sin_s**2
                 * gaussian_roughness_spectrum_2d(dK, h, L) / (Ri**2 * Rs**2))
    return trapezoid(trapezoid(integrand, ys, axis=1), xs)


def _rough_surface_env(surface_roughness):
    return Environment(
        water_density=RHO_WATER, bathymetry=_SCAT_WATER,
        ssp=SoundSpeedProfile.from_pairs([(0.0, C_W), (_SCAT_WATER, C_W)]),
        surface=BoundaryProperties(acoustic_type='vacuum',
                                   roughness=surface_roughness),
        bottom=BoundaryProperties(acoustic_type='half-space', sound_speed=C_W,
                                  density=RHO_WATER, attenuation=0.0))


def _oass_surface_loss(rms, receiver_heights=(50.0, 100.0)):
    """OASS reverberation loss (−10·log10 E|p_s|², dB) from the rough sea
    surface, source ``_SCAT_HEIGHT`` below it, receivers at the given
    depths; shape (n_depth, n_range)."""
    from uacpy.models import OASS
    res = _quiet(OASS(correlation_length=_SCAT_L, rms_roughness=rms,
                      interface=2, timeout=300).run,
                 _rough_surface_env(_SCAT_MEAN_FIELD_ROUGHNESS),
                 Source(depths=_SCAT_HEIGHT, frequencies=_SCAT_F),
                 Receiver(depths=list(receiver_heights), ranges=_SCAT_RANGES),
                 run_mode=RunMode.REVERBERATION)
    assert res.kind == 'reverberation'
    return np.asarray(res.data, dtype=float)


@pytest.fixture(scope='module')
def oass_surface_loss_h05():
    return _oass_surface_loss(_SCAT_H)


def test_oass_reverberation_loss_matches_the_first_order_perturbation_integral(
        oass_surface_loss_h05):
    """OASS's reverberation loss from a rough sea surface, with a smooth
    mean field (Born), must equal −10·log10 of the Dirichlet SPM patch
    integral in absolute dB. Not a Lambert comparison: Lambert's μ has no
    mapping from (rms, correlation length), whereas the perturbation
    cross-section is what OASS's Kuperman–Schmidt kernel reduces to in a
    half-space, so the level is pinned tightly rather than windowed.

    Measured over the 14 asserted ranges x 2 depths: |Δ| 0.02-0.03 dB
    through 1 km, 0.10-0.14 dB at 1.5 km, median 0.063, max 0.136. The bounds
    are 4x and 3.7x that. Beyond 40 % of the receiver window OASS's
    reverberation synthesis under-resolves — measured −0.38 dB at 2.3 km and
    −4.7 dB at 4 km on this 4 km window, restored to ≤ 0.15 dB by
    ``nw_samples=8192`` on the mean-field OAST — so the assertions stop at
    1.5 km while the receivers run to 4 km. A field off by a factor 2 is
    3.01 dB.
    """
    ref = np.array([[-10.0 * np.log10(dirichlet_spm_scattered_intensity(
                        r, _SCAT_HEIGHT, zr, _SCAT_F, C_W, _SCAT_H, _SCAT_L))
                     for r in _SCAT_RANGES[_SCAT_ASSERTED]] for zr in (50.0, 100.0)])
    d = np.abs(oass_surface_loss_h05[:, _SCAT_ASSERTED] - ref)
    assert np.median(d) < 0.25, f"median |dRL|={np.median(d):.3f} dB\n{d}"
    assert np.max(d) < 0.5, f"max |dRL|={np.max(d):.3f} dB\n{d}"


def test_oass_scattered_intensity_scales_with_the_square_of_the_rms_roughness(
        oass_surface_loss_h05):
    """First-order perturbation theory is linear in the roughness amplitude,
    so doubling the rms roughness of the scattering deck (the mean field
    unchanged) must lower the reverberation loss by exactly 20·log10(2) at
    every cell. Measured: −6.02 dB at all 78 cells out to 4 km. This
    pins the exponent of the law, not the level (it is scale-invariant); the
    level test above is the scale-sensitive one.
    """
    loss_h10 = _oass_surface_loss(2.0 * _SCAT_H, receiver_heights=(50.0, 100.0))
    step = loss_h10 - oass_surface_loss_h05
    assert np.allclose(step, -20.0 * np.log10(2.0), atol=0.05), step


# ── OASSP: one realisation of the scattered field ───────────────────────────

_OASSP_WATER = 100.0
_OASSP_N_REALISATIONS = 8
# OASSP's own uniform grid, 1.5 km deep. The 4 km grid OASS runs on makes
# oassp2 size its realisation patch at 8192 range samples (12.6 km) and die
# with SIGSEGV inside PV/VMOV (measured 2026-09-19); at 1.5 km it runs.
_OASSP_RANGES = np.linspace(200.0, 1500.0, 14)


def _rough_seabed_env():
    """The mirrored geometry: no sea surface (a water upper half-space) and
    a rough pressure-release seabed — the only interface OASSP's binary can
    scatter from."""
    return Environment(
        water_density=RHO_WATER, bathymetry=_OASSP_WATER,
        ssp=SoundSpeedProfile.from_pairs([(0.0, C_W), (_OASSP_WATER, C_W)]),
        surface=BoundaryProperties(acoustic_type='half-space', sound_speed=C_W,
                                   density=RHO_WATER, attenuation=0.0),
        bottom=BoundaryProperties(acoustic_type='vacuum',
                                  roughness=_SCAT_MEAN_FIELD_ROUGHNESS))


def _oassp_realisation(rms, realization):
    """Scattered ``H(f)`` (n_range, n_freq) at the receivers 50 m above the
    rough seabed, and the frequency axis, for one seeded realisation."""
    from uacpy.models import OASSP
    res = _quiet(OASSP(correlation_length=_SCAT_L, rms_roughness=rms,
                       n_time_samples=256, freq_min=150.0, freq_max=250.0,
                       realization=realization, timeout=300).run,
                 _rough_seabed_env(),
                 Source(depths=_OASSP_WATER - _SCAT_HEIGHT, frequencies=_SCAT_F),
                 Receiver(depths=[_OASSP_WATER - 50.0], ranges=_OASSP_RANGES),
                 run_mode=RunMode.BROADBAND)
    return (np.asarray(res.data)[0],
            np.asarray(res.coords['frequency'], dtype=float))


def test_oassp_realisation_is_linear_in_the_rms_roughness():
    """OASSP's scattered field is the first-order perturbation solution for
    one seeded surface, so doubling the rms roughness of the same
    realisation must double the complex field at every cell and bin — no
    phase change, no other dependence. Measured: |H(1.0 m)| / |H(0.5 m)| =
    2.000000 at min and max over 14 ranges x 52 bins, phase of the ratio
    ≤ 2.3e-6 deg. The bound is 1e-3 on the ratio. Scale-invariant by
    construction: it pins the exponent of the law.
    """
    h05, f05 = _oassp_realisation(_SCAT_H, realization=3)
    h10, f10 = _oassp_realisation(2.0 * _SCAT_H, realization=3)
    np.testing.assert_array_equal(f05, f10)
    ratio = h10 / h05
    assert np.all(np.abs(np.abs(ratio) - 2.0) < 1e-3), np.abs(ratio)
    assert np.max(np.abs(np.angle(ratio))) < 1e-4


def test_oassp_ensemble_intensity_follows_the_perturbation_integral_range_law():
    """The band-and-ensemble mean of |H_s|² over seeded realisations must
    decay with range at the slope of the first-order perturbation integral,
    and its level must stay where it was measured relative to that integral.

    The level is NOT pinned to the integral itself: without option ``'P'``
    OASSP synthesises the field with a Hankel transform, so the realised
    roughness is a function of range alone (an axisymmetric surface), whose
    second moment is not that of the 2-D isotropic surface the integral —
    and OASS's REVINT, which matches it to 0.03 dB — describe. Measured over
    8 realisations x 14 ranges x 52 bins: 10·log10(ΣI / ΣI_ref) = −5.3 dB,
    per-realisation −11.1 .. −0.3 dB, and a per-range ratio with no trend
    (−6.5 .. −2.9 dB). Slope of the ensemble+band mean over 200–1500 m:
    −41.9 dB/decade vs −43.0 for the integral. The slope bound is 3 dB per
    decade; the level window is the measured offset ±3 dB, which a factor
    2 in amplitude (+6.02 dB) leaves — a regression pin on the current
    behaviour, stated as such, not a physics claim. An exact OASSP level
    check needs ``plane_geometry=True`` against a 2-D line-source
    perturbation reference; that is parked in the fixer ledger.
    """
    realisations = [_oassp_realisation(_SCAT_H, k)
                    for k in range(_OASSP_N_REALISATIONS)]
    f_axis = realisations[0][1]
    for _, f_k in realisations[1:]:
        np.testing.assert_array_equal(f_k, f_axis)
    intensity = np.mean([np.abs(h)**2 for h, _ in realisations],
                        axis=0)                                 # (n_range, n_f)
    ref = np.array([[dirichlet_spm_scattered_intensity(
                        r, _SCAT_HEIGHT, 50.0, f, C_W, _SCAT_H, _SCAT_L, n=500)
                     for f in f_axis] for r in _OASSP_RANGES])
    band = intensity.mean(axis=1)
    band_ref = ref.mean(axis=1)
    lr = np.log10(_OASSP_RANGES)
    slope = np.polyfit(lr, 10.0 * np.log10(band), 1)[0]
    slope_ref = np.polyfit(lr, 10.0 * np.log10(band_ref), 1)[0]
    assert abs(slope - slope_ref) < 3.0, (
        f"range slope {slope:.2f} dB/decade vs integral {slope_ref:.2f}")
    level = 10.0 * np.log10(intensity.sum() / ref.sum())
    assert -5.3 - 3.0 < level < -5.3 + 3.0, (
        f"ensemble level {level:.2f} dB re the perturbation integral left "
        f"the measured −5.3 ± 3 dB window")
