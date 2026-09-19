"""Tier-1 numerical benchmark for SPARC — its time-domain field, reduced to a
narrowband transfer function, against the exact modal sum of a rigid-bottom
waveguide.

SPARC is the time-marched FFP of the Acoustics Toolbox: it produces
``p(z, r, t)`` for a source pulse and refuses CW transmission loss. Its only
cross-model check so far (``test_cross_model_broadband``) ties it to Kraken
on three cells. SPARC only runs vacuum or rigid bottoms, which rules out the
Pekeris sum every other engine is pinned to — but an isovelocity guide with a
pressure-release surface and a RIGID bottom has an elementary exact modal
sum, so the reference here is a closed form, not another engine. Kraken's
agreement with the same closed form is recorded in the fixer ledger
(median 0.18 dB, reading low) and is not what this test pins.

Reference: the normal-mode sum of Porter, *KRAKEN Normal Mode Program*,
eq. 2.19, with the rigid-bottom modes ``Z_m = sqrt(2/D) sin(γ_m z)``,
``γ_m = (m − ½)π/D``, ``k_m = sqrt(k² − γ_m²)`` — the same far-field
prefactor and 1 m reference as ``pekeris_modal_tl`` in
``test_benchmarks_analytic``.
"""
import warnings

import numpy as np
import pytest

pytestmark = [pytest.mark.benchmark, pytest.mark.requires_binary]

from uacpy import Environment, BoundaryProperties, Source, Receiver, RunMode

# A 100 m rigid guide at 1500 m/s: mode m cuts on at (2m − 1)·c/(4D) =
# 3.75, 11.25, 18.75, 26.25, 33.75, 41.25 Hz. The comparison bins sit between
# the last two, where five modes propagate and the slowest (mode 5, group
# speed ≈ 654 m/s at 37.5 Hz) reaches 1.2 km in 1.8 s — well inside the 4 s
# record, so the deconvolved spectrum holds every mode's energy.
_DEPTH, _C, _FC = 100.0, 1500.0, 37.5
_ZS = 20.0
_ZR = np.array([35.0, 65.0])
_RANGES = np.array([800.0, 1000.0, 1200.0])
_T_MAX = 4.0                      # bins land on n/4 Hz
_TARGET_BINS = np.array([36.0, 36.75, 37.5, 38.25, 39.0])
# SPARC's 'PN+N' pulse against cans.f90's closed form carries a global
# factor 2 in amplitude, measured and documented by
# test_cross_model_broadband.test_sparc_pn_n_pulse_deconvolves_onto_kraken_broadband.
_DOCUMENTED_GAIN_DB = 20.0 * np.log10(2.0)


def rigid_bottom_modal_tl(z_s, z_r, ranges, f, depth, c):
    """Exact transmission loss of an isovelocity waveguide, pressure-release
    at the surface and rigid at the bottom, by the normal-mode sum (Porter,
    KRAKEN manual eq. 2.19): γ_m = (m − ½)π/D, k_m = sqrt(k² − γ_m²), the
    depth-normalised mode sqrt(2/D)·sin(γ_m z), and
        p(r, z) = sqrt(2π/r) Σ_m Z_m(z_s) Z_m(z) e^{i k_m r} / sqrt(k_m),
    TL = −20·log10|p| (1 m reference). Only propagating modes are kept, and
    a mode within 1e-6 (relative) of cutoff is dropped for the same reason
    ``dirichlet_modal_tl`` drops it: 1/sqrt(k_m) diverges there.
    """
    k = 2 * np.pi * f / c
    m = np.arange(1, 400)
    gamma = (m - 0.5) * np.pi / depth
    gamma = gamma[gamma < k]
    km = np.sqrt(k**2 - gamma**2)
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


def sparc_pseudo_gaussian(t, f):
    """AT's 'P' source pulse (``tslib/cans.f90:26-29``):
    ``s(t) = 0.75 − cos(ωt) + 0.25·cos(2ωt)`` on ``[0, 1/f]``, zero
    elsewhere. Copied from ``test_cross_model_broadband._sparc_pseudo_gaussian``
    so this module does not import that one."""
    w = 2.0 * np.pi * f
    s = 0.75 - np.cos(w * t) + 0.25 * np.cos(2.0 * w * t)
    return np.where((t >= 0.0) & (t <= 1.0 / f), s, 0.0)


def test_sparc_deconvolved_spectrum_matches_the_rigid_guide_modal_sum():
    """SPARC's snapshot field, Fourier-transformed and divided by the
    analytic source spectrum, must give the exact rigid-guide modal-sum TL at
    every (depth, range, bin) cell up to the documented factor-2 pulse
    convention — and that convention must stay at 6.02 dB, which is what
    makes the test fail for a field off by a factor 2 (the gain would read
    12 dB).

    Measured over 2 depths x 3 ranges x 5 bins (30 cells): SPARC − closed
    form runs +5.18 .. +6.25 dB with median +5.81 dB; after removing that
    common gain, |Δ| median 0.15 dB, p90 0.30 dB, max 0.63 dB. Bounds are
    3.3x and 2.6x those; the gain window of ±1.0 dB around 6.02 dB is the
    one the Kraken-tied test uses. Kraken at the same bins reads the closed
    form 0.18 dB low (median), so the closed form — not Kraken — is the
    reference here.
    """
    from uacpy.models import SPARC
    env = Environment(name='sparc-rigid-guide', bathymetry=_DEPTH, ssp=_C,
                      bottom=BoundaryProperties(acoustic_type='rigid'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ts = SPARC(verbose=False, pulse_type='PN+N', output_mode='S',
                   n_t_out=2048, t_max=_T_MAX, f_min=5.0, f_max=75.0,
                   rmax_safety_margin=7.0, timeout=600.0).run(
            env, Source(depths=_ZS, frequencies=_FC),
            Receiver(depths=_ZR, ranges=_RANGES), run_mode=RunMode.TIME_SERIES)
    times = np.asarray(ts.coords['time'], dtype=float)
    dt = float(times[1] - times[0])
    traces = np.asarray(ts.data)                      # (n_depth, n_range, n_t)
    assert traces.shape[:2] == (_ZR.size, _RANGES.size), traces.shape
    assert np.all(np.isfinite(traces))
    spectra = np.fft.rfft(traces, axis=-1)
    freqs = np.fft.rfftfreq(traces.shape[-1], dt)
    bins = np.array([int(np.argmin(np.abs(freqs - f))) for f in _TARGET_BINS])
    f_bins = freqs[bins]
    assert np.all(np.abs(f_bins - _TARGET_BINS) < 0.3), f_bins
    source_spectrum = np.fft.rfft(sparc_pseudo_gaussian(times, _FC))[bins]
    assert np.all(np.abs(source_spectrum) > 1.0)      # well off any pulse null
    tl_sparc = -20.0 * np.log10(
        np.abs(spectra[:, :, bins] / source_spectrum[np.newaxis, np.newaxis, :]))

    ana = np.array([[rigid_bottom_modal_tl(_ZS, zr, _RANGES, fb, _DEPTH, _C)
                     for fb in f_bins] for zr in _ZR])   # (n_depth, n_bin, n_range)
    ana = np.transpose(ana, (0, 2, 1))                    # (n_depth, n_range, n_bin)
    diff = tl_sparc - ana
    assert np.all(np.isfinite(diff)), (tl_sparc, ana)
    gain = float(np.median(diff))
    assert gain == pytest.approx(_DOCUMENTED_GAIN_DB, abs=1.0), (
        f"SPARC common gain {gain:.2f} dB moved away from the documented "
        f"factor-2 pulse convention ({_DOCUMENTED_GAIN_DB:.2f} dB)")
    resid = np.abs(diff - gain)
    assert np.median(resid) < 0.5, (
        f"gain-removed median |dTL|={np.median(resid):.2f} dB\n{diff}")
    assert np.percentile(resid, 90) < 0.8, (
        f"gain-removed p90 |dTL|={np.percentile(resid, 90):.2f} dB\n{diff}")
