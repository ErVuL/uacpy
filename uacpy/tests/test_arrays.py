"""Tests for ``uacpy.acoustic_signal.arrays`` — steering vectors and the
Bartlett / MVDR / MUSIC beamformers on a synthetic line array.
"""

import numpy as np
import pytest

from uacpy.acoustic_signal import (
    bartlett_spectrum,
    music_spectrum,
    mvdr_spectrum,
    sample_covariance,
    steering_vectors,
    shading_taper,
)
from uacpy.core.exceptions import ConfigurationError
from uacpy.acoustic_signal.arrays import beamform

#: The scalars every sample-rate / dimension guard must refuse.
BAD_SCALARS = [0.0, -100.0, np.nan, np.inf]


FREQ = 1000.0
C = 1500.0


def _array():
    # Half-wavelength spacing at FREQ.
    spacing = C / FREQ / 2.0
    return np.arange(16) * spacing


def _snapshots(positions, true_angle_deg, n_snaps=400, snr=20.0):
    a = steering_vectors(positions, [true_angle_deg], FREQ, C)[0]
    rng = np.random.default_rng(0)
    src = rng.standard_normal(n_snaps) + 1j * rng.standard_normal(n_snaps)
    noise_amp = 10 ** (-snr / 20.0)
    noise = noise_amp * (rng.standard_normal((positions.size, n_snaps))
                         + 1j * rng.standard_normal((positions.size, n_snaps)))
    return np.outer(a, src) + noise


class TestSteering:
    def test_unit_norm_rows(self):
        e = steering_vectors(_array(), [-30, 0, 30], FREQ, C)
        assert np.allclose(np.linalg.norm(e, axis=1), 1.0)
        assert e.shape == (3, 16)


class TestCovariance:
    def test_sample_covariance_is_hermitian(self):
        x = _snapshots(_array(), 10.0)
        R = sample_covariance(x)
        assert np.allclose(R, R.conj().T)

    def test_diagonal_loading_adds_trace_fraction(self):
        # diagonal_loading is documented as a *fraction of trace(R)/N* on
        # the diagonal: R_loaded == R + 0.1·(tr(R)/N)·I. The identity is
        # pure arithmetic on the same R, so 1e-12 absolute (well above the
        # ~1e-16 float noise of the addition) is the right scale.
        x = _snapshots(_array(), 10.0)
        R = sample_covariance(x)
        R_loaded = sample_covariance(x, diagonal_loading=0.1)
        n = R.shape[0]
        expected = R + 0.1 * (np.trace(R).real / n) * np.eye(n)
        np.testing.assert_allclose(R_loaded, expected, atol=1e-12)

    def test_requires_2d(self):
        with pytest.raises(ConfigurationError):
            sample_covariance(np.zeros(8))


class TestBeamformers:
    # 241 nodes over 120 deg = 0.5 deg spacing, and the spectra are read with
    # a bare argmax, so the abs=1.0 below is a two-node allowance on the scan
    # grid — not a bearing-accuracy claim.
    angles = np.linspace(-60, 60, 241)

    def test_bartlett_recovers_doa(self):
        R = sample_covariance(_snapshots(_array(), 15.0))
        p = bartlett_spectrum(R, steering_vectors(_array(), self.angles, FREQ, C))
        assert self.angles[np.argmax(p)] == pytest.approx(15.0, abs=1.0)

    def test_mvdr_recovers_doa(self):
        R = sample_covariance(_snapshots(_array(), -20.0))
        p = mvdr_spectrum(R, steering_vectors(_array(), self.angles, FREQ, C))
        assert self.angles[np.argmax(p)] == pytest.approx(-20.0, abs=1.0)

    def test_music_recovers_doa(self):
        R = sample_covariance(_snapshots(_array(), 5.0))
        p = music_spectrum(R, steering_vectors(_array(), self.angles, FREQ, C), 1)
        assert self.angles[np.argmax(p)] == pytest.approx(5.0, abs=1.0)

    def test_music_rejects_bad_source_count(self):
        R = sample_covariance(_snapshots(_array(), 5.0))
        with pytest.raises(ConfigurationError):
            music_spectrum(R, steering_vectors(_array(), self.angles, FREQ, C), 16)


class TestTaper:
    """RMS normalisation, not unit mean. ``steering_vectors`` rows are
    unit-norm, so a taper multiplied into one must preserve that; normalising
    to unit *mean* scaled every power the taper touched by ``mean(w**2)`` —
    +2.04 dB for Hann on 16 elements, in the direction that makes an array
    look better than it is."""

    @pytest.mark.parametrize("window", ["boxcar", "hann", "hamming",
                                        ("chebwin", 30)])
    def test_rms_normalised(self, window):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = shading_taper(16, window)
        assert w.size == 16
        assert np.mean(w ** 2) == pytest.approx(1.0)
        # equivalently ||w|| = sqrt(N), which is what keeps e*w unit-norm
        assert np.linalg.norm(w) == pytest.approx(4.0)

    def test_boxcar_is_all_ones(self):
        # The discriminating case: under unit-mean this was also all-ones, so
        # it alone cannot tell the two normalisations apart — but under RMS it
        # must stay all-ones, which pins the scale rather than just the shape.
        assert np.allclose(shading_taper(16, "boxcar"), 1.0)


@pytest.mark.parametrize("true_deg", [15.0, -30.0, 45.0])
def test_beamform_resolves_true_angle_not_mirror(true_deg):
    """beamform must resolve a source at +theta (not the mirror -theta) —
    consistent with bartlett/mvdr/music (the steering vector is conjugated)."""
    from uacpy.acoustic_signal import beamform, steering_vectors
    c, f = 1500.0, 1500.0
    pos = np.arange(16) * (c / f / 2.0)
    ang = np.linspace(-60, 60, 241)
    a = steering_vectors(pos, [true_deg], f, c)[0]
    snr, angles, _ = beamform(a[:, None], pos, f, angles=ang, SL=0, NL=0)
    assert abs(angles[np.argmax(snr[:, 0])] - true_deg) < 1.0


def test_mvdr_music_no_divide_warning_and_music_peaks_at_source():
    import warnings
    pos = np.arange(6) * 0.75
    angles = np.linspace(-60, 60, 121)
    e = steering_vectors(pos, angles, 1000.0)
    src = steering_vectors(pos, [10.0], 1000.0)[0]
    R = np.eye(6, dtype=complex) + 8 * np.outer(src, src.conj())
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)   # no spurious 1/denom warning
        m = mvdr_spectrum(R, e)
        mu = music_spectrum(R, e, 1)
    assert np.all(np.isfinite(m))
    # the sharp MUSIC peak at the source direction is the intended behaviour,
    # preserved (not clamped) — it localises the source at 10 deg
    assert abs(angles[np.argmax(mu)] - 10.0) < 2.0


class TestPowerlessCovariance:
    """``diagonal_loading`` is a *fraction of* ``trace(R)/N``, so it vanishes
    with the trace: it stabilises a rank-deficient covariance that still
    carries power but cannot rescue an all-zero one. That is ordinary data —
    ``sample_covariance`` of a silent segment (dead element, digital silence)
    returns exactly it. Without the guard MVDR's inverse is singular and
    MUSIC's noise subspace is arbitrary, so both must decline rather than
    return a finite *uniform* pseudospectrum that looks like an answer."""

    @staticmethod
    def _rig(n=8):
        from uacpy.acoustic_signal.arrays import steering_vectors
        return steering_vectors(np.arange(n) * 0.75,
                                np.linspace(-90.0, 90.0, 37), 1000.0)

    def test_silence_gives_nan_and_warns_not_a_raw_linalg_error(self):
        from uacpy.acoustic_signal.arrays import (
            sample_covariance, mvdr_spectrum, music_spectrum)
        E = self._rig()
        R = sample_covariance(np.zeros((8, 200), dtype=complex))
        for fn in (lambda: mvdr_spectrum(R, E),
                   lambda: music_spectrum(R, E, 1)):
            with pytest.warns(UserWarning, match='no power'):
                out = fn()
            assert np.all(np.isnan(out))

    def test_bartlett_reports_zero_power_which_is_the_true_answer(self):
        from uacpy.acoustic_signal.arrays import (
            sample_covariance, bartlett_spectrum)
        R = sample_covariance(np.zeros((8, 200), dtype=complex))
        assert np.all(bartlett_spectrum(R, self._rig()) == 0.0)

    def test_a_powered_rank_deficient_covariance_resolves(self):
        """The case the loading exists for must keep working: fewer snapshots
        than elements, but non-zero trace."""
        from uacpy.acoustic_signal.arrays import (
            steering_vectors, sample_covariance, mvdr_spectrum)
        rng = np.random.default_rng(0)
        a = steering_vectors(np.arange(8) * 0.75, [15.0], 1000.0).T * np.sqrt(8)
        x = a @ (rng.normal(size=(1, 4)) + 1j * rng.normal(size=(1, 4)))
        angles = np.linspace(-90.0, 90.0, 721)
        P = mvdr_spectrum(sample_covariance(x),
                          steering_vectors(np.arange(8) * 0.75, angles, 1000.0))
        assert np.all(np.isfinite(P))
        assert angles[np.argmax(P)] == pytest.approx(15.0, abs=0.5)


@pytest.mark.requires_binary
def test_look_direction_agrees_with_an_independently_propagated_field():
    """A downgoing arrival is reported at *positive* declination.

    The field comes from Bellhop, so its depth-phase sign is the one the solver
    produces under the Acoustics-Toolbox ``exp(+i*omega*t)`` convention
    (``KrakenField/EvaluateMod.f90:42``) with depth positive down
    (``Bellhop/bellhop.f90:453``), rather than one this module assumes. A test
    that builds its snapshots from :func:`steering_vectors` cannot detect a
    mirrored look direction, because the same convention appears on both sides
    and the error cancels.
    """
    from uacpy import (Environment, Source, Receiver, Bellhop,
                       SoundSpeedProfile, BoundaryProperties)
    from uacpy.acoustic_signal import beamform

    c, freq, z_src, range_m = 1500.0, 200.0, 200.0, 10_000.0
    z_rcv = np.arange(1400.0, 1601.0, 3.0)
    env = Environment(
        bathymetry=5000.0,
        ssp=SoundSpeedProfile.from_pairs([(0.0, c), (5000.0, c)]),
        bottom=BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=1800.0, density=1.8,
                                  attenuation=0.5),
    )
    field = Bellhop(beam_type='G', n_beams=4001, alpha=(6.0, 9.0)).compute_tl(
        env, Source(depths=z_src, frequencies=freq),
        Receiver(depths=z_rcv, ranges=[range_m]))
    p = np.asarray(field.data).ravel()

    # The narrow fan traces the direct eigenray only, so this is one plane wave
    # and the array sees a single unambiguous arrival angle.
    amp = np.abs(p)
    assert amp.max() / amp.min() < 1.05

    expected_deg = np.degrees(np.arctan2(z_rcv.mean() - z_src, range_m))
    assert expected_deg > 0.0                      # receivers below the source

    angles = np.arange(-20.0, 20.01, 0.05)
    out = beamform(p, z_rcv, freq, angles=angles, c=c)
    peak_deg = out.angles[np.argmax(out.snr)]
    assert peak_deg == pytest.approx(expected_deg, abs=0.5)


# ── Analytic physics pins (docs/guide/arrays.md) ────────────────────────────
#
# Deterministic rank-one/theory covariances only — no snapshots, no noise
# draws — so every number below is a property of the array geometry and the
# processors, reproducible to machine precision. References per pin:
# Van Trees *Optimum Array Processing*; Balanis *Antenna Theory* (uniform
# aperture sidelobe/resolution criterion); Abraham *Underwater Acoustic
# Signal Processing* (isotropic-noise correlation Eq. 8.226, window
# sidelobe levels §4.10); Butler & Sherman *Transducers and Arrays* (§7.10
# grating lobes, §8.16 isotropic spatial correlation).

from uacpy.acoustic_signal import beamform as beamform_fn

LAM = C / FREQ                     # 1.5 m at 1000 Hz in 1500 m/s water


def _pattern_db(n, d, steer_deg=0.0, weights=None,
                angles=np.linspace(-90.0, 90.0, 36001)):
    """Deterministic beampattern in dB: Bartlett scan of the rank-one
    covariance of a single plane wave (the arrays.md figure method)."""
    pos = np.arange(n) * d
    a = steering_vectors(pos, [steer_deg], FREQ, C)[0]
    e = steering_vectors(pos, angles, FREQ, C)
    if weights is not None:
        e = e * weights
    p = bartlett_spectrum(np.outer(a, a.conj()), e).real
    return angles, 10.0 * np.log10(np.maximum(p / p.max(), 1e-300))


def _local_maxima(values):
    """Indices of strict interior local maxima."""
    v = np.asarray(values)
    return np.nonzero((v[1:-1] > v[:-2]) & (v[1:-1] > v[2:]))[0] + 1


class TestManifoldPhaseStep:
    def test_adjacent_element_phase_step_is_2pi_d_over_lambda_sin_theta(self):
        # e_n = exp(-j·k·z_n·sinθ)/√N, so the element-to-element phase
        # step is exactly -2π(d/λ)·sinθ (the -j Hermitian-form convention
        # pinned in test_beamform_resolves_true_angle_not_mirror).
        d, theta = 0.6 * LAM, 25.0
        e = steering_vectors(np.arange(8) * d, [theta], FREQ, C)[0]
        steps = np.angle(e[1:] * np.conj(e[:-1]))
        expected = -2.0 * np.pi * (d / LAM) * np.sin(np.deg2rad(theta))
        np.testing.assert_allclose(steps, expected, atol=1e-12)


class TestRankOneBeampattern:
    """Uniform 16-element λ/2 pattern anatomy (arrays.md §3 table, N=16):
    6.35° −3 dB mainlobe, first null at arcsin(λ/L), −13.15 dB first
    sidelobe. L = N·d throughout (the convention the 0.886 formula is
    written in)."""

    N, D = 16, LAM / 2.0

    def test_minus_3db_width_is_0886_lambda_over_L(self):
        ang, pdb = _pattern_db(self.N, self.D)
        width = np.ptp(ang[pdb >= -3.0])
        L = self.N * self.D
        # 0.886·λ/L rad = 6.346°; the discrete 16-element pattern measures
        # 6.348° (grid 0.005°). Standard uniform-aperture HPBW factor
        # (sinc² argument 1.392 = 0.443·π; Van Trees/Cochran).
        assert width == pytest.approx(np.degrees(0.886 * LAM / L), abs=0.05)
        assert width == pytest.approx(6.35, abs=0.05)

    def test_first_null_at_arcsin_lambda_over_L(self):
        # At u = λ/L the element phasors are the N-th roots of unity and
        # sum to exactly zero, so the pattern at arcsin(λ/L) is a
        # machine-precision null (not merely a dip).
        null_deg = np.degrees(np.arcsin(LAM / (self.N * self.D)))
        pos = np.arange(self.N) * self.D
        a = steering_vectors(pos, [0.0], FREQ, C)[0]
        e = steering_vectors(pos, [null_deg - 0.5, null_deg, null_deg + 0.5],
                             FREQ, C)
        p = bartlett_spectrum(np.outer(a, a.conj()), e).real
        assert null_deg == pytest.approx(7.18, abs=0.01)   # the doc value
        assert p[1] < 1e-12 * max(p[0], p[2])

    def test_first_sidelobe_level(self):
        ang, pdb = _pattern_db(self.N, self.D)
        null_deg = np.degrees(np.arcsin(LAM / (self.N * self.D)))
        beyond = ang > null_deg
        peaks = _local_maxima(pdb[beyond])
        first_sidelobe = pdb[beyond][peaks[0]]
        # Discrete N=16 value −13.147 dB (doc table −13.15; the N→∞ sinc
        # asymptote is −13.26 dB, Balanis' "≈ −13.5"). abs=0.05 covers the
        # scan grid.
        assert first_sidelobe == pytest.approx(-13.15, abs=0.05)


class TestTwoSourceRayleighResolution:
    """Two equal uncorrelated sources on the Rayleigh scale (arrays.md §3):
    at Δθ = arcsin(λ/L) the Bartlett dip between the peaks is only
    ~0.9 dB; the peaks merge at 0.83·arcsin(λ/L). Deterministic
    R = a₁a₁ᴴ + a₂a₂ᴴ (no noise), so both numbers are exact properties of
    the pattern (Rayleigh criterion = FNBW/2, Balanis §2.2)."""

    N, D = 16, LAM / 2.0

    def _scan(self, sep_frac):
        rayleigh = np.degrees(np.arcsin(LAM / (self.N * self.D)))
        half = 0.5 * sep_frac * rayleigh
        pos = np.arange(self.N) * self.D
        a1 = steering_vectors(pos, [-half], FREQ, C)[0]
        a2 = steering_vectors(pos, [+half], FREQ, C)[0]
        R = np.outer(a1, a1.conj()) + np.outer(a2, a2.conj())
        ang = np.linspace(-rayleigh, rayleigh, 4001)   # mainlobe region only
        p = bartlett_spectrum(R, steering_vectors(pos, ang, FREQ, C)).real
        return ang, p

    def test_dip_at_rayleigh_separation_is_0p9_db(self):
        ang, p = self._scan(1.0)
        peaks = _local_maxima(p)
        assert len(peaks) == 2                       # still two maxima
        mid = p[np.argmin(np.abs(ang))]
        dip_db = 10.0 * np.log10(p[peaks].max() / mid)
        # Measured 0.915 dB on this pattern; the doc rounds to 0.9.
        assert dip_db == pytest.approx(0.92, abs=0.05)

    def test_peaks_survive_just_above_the_merge_point(self):
        ang, p = self._scan(0.86)                    # 0.83 + margin
        assert len(_local_maxima(p)) == 2

    def test_peaks_merge_below_083_rayleigh(self):
        ang, p = self._scan(0.80)                    # 0.83 − margin
        peaks = _local_maxima(p)
        assert len(peaks) == 1                       # single central blob
        assert abs(ang[peaks[0]]) < 0.1


class TestGratingLobes:
    """Grating lobes are full-height mainlobe replicas at u₀ ± m·λ/d
    (Butler & Sherman Eq. 7.10c: sinθ = sinθ₀ ± λ/D), kept out of the
    visible region iff d < λ/(1 + |sinθ₀|) (arrays.md §2)."""

    N = 16

    def test_full_wavelength_spacing_aliases_broadside_to_endfire(self):
        pos = np.arange(self.N) * LAM              # d = λ
        a = steering_vectors(pos, [0.0], FREQ, C)[0]
        e = steering_vectors(pos, [-90.0, 0.0, 90.0], FREQ, C)
        p = bartlett_spectrum(np.outer(a, a.conj()), e).real
        # u = sin θ ± 1 puts identical-height copies at both endfires.
        assert p[0] == pytest.approx(p[1], rel=1e-9)
        assert p[2] == pytest.approx(p[1], rel=1e-9)

    def test_steered_alias_position_is_arcsin_u0_minus_lambda_over_d(self):
        d, steer = 0.75 * LAM, 45.0
        ang, pdb = _pattern_db(self.N, d, steer_deg=steer)
        alias = np.degrees(np.arcsin(np.sin(np.deg2rad(steer)) - LAM / d))
        assert alias == pytest.approx(-38.77, abs=0.01)   # the doc's −39°
        i = int(np.argmin(np.abs(ang - alias)))
        assert pdb[i] == pytest.approx(0.0, abs=0.01)     # full height

    @pytest.mark.parametrize('d_frac,has_alias', [(0.63, False),
                                                  (0.70, True)])
    def test_visible_region_bound_at_lambda_over_one_plus_sin(self, d_frac,
                                                              has_alias):
        # Steered to 30°, the bound is d < λ/1.5 = 0.667λ: 0.63λ keeps the
        # alias out of the visible region (everything off the mainlobe
        # stays ordinary ≤ −13 dB sidelobe), 0.70λ pulls a near-full
        # replica inside (measured −0.0 dB at −68.2°).
        steer = 30.0
        d = d_frac * LAM
        ang, pdb = _pattern_db(self.N, d, steer_deg=steer)
        u = np.sin(np.deg2rad(ang))
        u0 = np.sin(np.deg2rad(steer))
        # Mask the mainlobe (2 null-widths around u₀).
        off_main = np.abs(u - u0) > 2.0 * LAM / (self.N * d)
        top = pdb[off_main].max()
        if has_alias:
            assert top > -1.0
        else:
            assert top < -10.0


class TestIsotropicNoiseArrayGain:
    """White-noise gain is 10·log10(N) at any spacing; against isotropic
    noise the element correlation is sinc(2Δz/λ) (Abraham Eq. 8.226;
    Butler & Sherman Eq. 8.16 — exactly zero for every pair at λ/2), so a
    16-element array holds 12.0 dB at λ/2 but only 9.1 dB at λ/4 and
    6.2 dB at λ/8 (arrays.md §2)."""

    N = 16

    @pytest.mark.parametrize('d_frac,gain_db', [
        (0.5, 12.04),      # = 10·log10(16): isotropic noise spatially white
        (0.25, 9.12),
        (0.125, 6.23),
    ])
    def test_broadside_gain_against_isotropic_noise(self, d_frac, gain_db):
        pos = np.arange(self.N) * d_frac * LAM
        e = steering_vectors(pos, [0.0], FREQ, C)
        # Unit-element-power plane wave and unit-element-power isotropic
        # noise: AG = (eᴴS e)/(eᴴQ e) with S = p pᴴ and Q the sinc matrix.
        p = np.sqrt(self.N) * e[0].conj()            # |p_n| = 1, broadside
        S = np.outer(p, p.conj())
        dz = np.abs(pos[:, None] - pos[None, :])
        Q = np.sinc(2.0 * dz / LAM)                  # sin(k·d)/(k·d)
        ag = (bartlett_spectrum(S, e) / bartlett_spectrum(Q, e)).real[0]
        # Computed 12.041 / 9.118 / 6.232 dB; the doc rounds to one place.
        assert 10.0 * np.log10(ag) == pytest.approx(gain_db, abs=0.01)

    def test_half_wavelength_matches_white_noise_gain(self):
        # At λ/2 every off-diagonal sinc is zero, so the isotropic field
        # is spatially white and the gain equals 10·log10(N) exactly.
        pos = np.arange(self.N) * LAM / 2.0
        dz = np.abs(pos[:, None] - pos[None, :])
        Q = np.sinc(2.0 * dz / LAM)
        np.testing.assert_allclose(Q, np.eye(self.N), atol=1e-15)


class TestShadingTapers:
    """The arrays.md §4 shading table, 32 elements at λ/2 (deterministic
    patterns): Hann −31.5 dB peak sidelobe / 5.35° width / −1.90 dB gain,
    Chebyshev-50 −50.0 dB / 4.90° / −1.54 dB. Hann's highest sidelobe
    31.5 dB down and the Dolph–Chebyshev constant equiripple floor are the
    textbook window values (Abraham §4.10)."""

    N, D = 32, LAM / 2.0

    def _taper_pattern(self, window):
        w = shading_taper(self.N, window)
        return _pattern_db(self.N, self.D, weights=w,
                           angles=np.linspace(-90.0, 90.0, 72001))

    def _width_and_sidelobe(self, window):
        ang, pdb = self._taper_pattern(window)
        width = np.ptp(ang[pdb >= -3.0])
        peaks = _local_maxima(pdb)
        off_main = peaks[np.abs(ang[peaks]) > 1e-9]
        return width, pdb[off_main].max()

    def test_hann_pattern_numbers(self):
        width, sll = self._width_and_sidelobe('hann')
        # Measured 5.318° on the discrete 32-element pattern (the doc's
        # figure grid rounds to 5.35°) and −31.47 dB.
        assert width == pytest.approx(5.35, abs=0.05)
        assert sll == pytest.approx(-31.5, abs=0.1)

    def test_chebyshev_50_pattern_numbers(self):
        width, sll = self._width_and_sidelobe(('chebwin', 50))
        # Measured 4.867° (doc 4.90°); the equiripple floor sits at the
        # design level −50.0 dB by construction.
        assert width == pytest.approx(4.90, abs=0.05)
        assert sll == pytest.approx(-50.0, abs=0.1)

    def test_hann_end_weights_are_exactly_zero(self):
        # shading_taper builds symmetric (fftbins=False) windows: a
        # 32-element Hann array spends its two end elements on nothing.
        w = shading_taper(32, 'hann')
        assert w[0] == 0.0 and w[-1] == 0.0

    @pytest.mark.parametrize('window,loss_db', [('hann', -1.90),
                                                (('chebwin', 50), -1.54),
                                                ('boxcar', 0.0)])
    def test_white_noise_gain_loss_formula(self, window, loss_db):
        # ΔG = 10·log10((Σw)² / (N·Σw²)) — the arrays.md closed form.
        w = shading_taper(self.N, window)
        dg = 10.0 * np.log10(w.sum() ** 2 / (self.N * np.sum(w ** 2)))
        assert dg == pytest.approx(loss_db, abs=0.01)


class TestBeamformOutputContract:
    """beamform returns 20·log10|eᴴp| + SL − NL on the −90:1:90 default
    grid with SL defaulting to 150 dB; unit-norm steering folds the array
    gain in, so a unit-element-amplitude plane wave peaks at
    10·log10(N) + SL − NL (arrays.md §3)."""

    def test_output_is_20log10_quadratic_form_plus_sl_minus_nl(self):
        pos = _array()
        rng = np.random.default_rng(2)
        p = rng.standard_normal(16) + 1j * rng.standard_normal(16)
        angles = np.linspace(-60.0, 60.0, 25)
        res = beamform_fn(p[:, None], pos, FREQ, angles=angles,
                          SL=150.0, NL=37.0)
        e = steering_vectors(pos, angles, FREQ, C)
        expected = 20.0 * np.log10(np.abs(e.conj() @ p)) + 150.0 - 37.0
        np.testing.assert_allclose(res.snr[:, 0], expected, atol=1e-12)

    def test_default_grid_is_minus90_to_90_in_1deg_steps(self):
        p = np.ones(16, dtype=complex)
        res = beamform_fn(p[:, None], _array(), FREQ, SL=0.0)
        np.testing.assert_array_equal(res.angles, np.arange(-90, 91, 1))
        assert len(res.angles) == 181

    def test_unit_plane_wave_peaks_at_10log10_n_and_sl_defaults_to_150(self):
        pos = _array()
        k = 2.0 * np.pi * FREQ / C
        p = np.exp(-1j * k * pos * np.sin(np.deg2rad(20.0)))   # |p_n| = 1
        res = beamform_fn(p[:, None], pos, FREQ, SL=0.0, NL=0.0)
        assert res.peak_snr == pytest.approx(10.0 * np.log10(16), abs=1e-9)
        assert res.angles[np.argmax(res.snr[:, 0])] == 20
        # SL defaults to 150 dB and enters additively.
        res_default = beamform_fn(p[:, None], pos, FREQ, NL=0.0)
        assert res_default.peak_snr - res.peak_snr == pytest.approx(150.0)


class TestPowerAverageEqualsCovarianceBeamforming:
    def test_snapshot_power_average_is_bartlett_of_sample_covariance(self):
        """Averaging beam power over snapshots and beamforming the sample
        covariance are the same arithmetic in a different order
        (arrays.md §9: identical to ~3e-13 dB):
        mean_k |eᴴx_k|² == eᴴ (X Xᴴ / K) e exactly."""
        pos = _array()
        rng = np.random.default_rng(1)
        X = rng.standard_normal((16, 64)) + 1j * rng.standard_normal((16, 64))
        ang = np.linspace(-90.0, 90.0, 181)
        snr = beamform_fn(X, pos, FREQ, angles=ang, SL=0.0, NL=0.0).snr
        power_avg = 10.0 * np.log10(np.mean(10.0 ** (snr / 10.0), axis=1))
        bart = 10.0 * np.log10(
            bartlett_spectrum(sample_covariance(X),
                              steering_vectors(pos, ang, FREQ, C)).real)
        np.testing.assert_allclose(power_avg, bart, atol=1e-10)


class TestMusicModelOrder:
    """MUSIC's two directions of order error are not symmetric (arrays.md
    §6): too few sources collapses the pair into one blob; too many is
    benign. Deterministic theory covariance R = a₁a₁ᴴ + a₂a₂ᴴ + σ²I with
    the pair at ±3° (inside the 16-element Rayleigh scale)."""

    @staticmethod
    def _theory_R():
        pos = _array()
        a1 = steering_vectors(pos, [-3.0], FREQ, C)[0]
        a2 = steering_vectors(pos, [+3.0], FREQ, C)[0]
        return (np.outer(a1, a1.conj()) + np.outer(a2, a2.conj())
                + 0.1 * np.eye(16))

    # The scan grid contains ±3.0 exactly, so the correct-order peaks land
    # on the true bearings to grid precision.
    angles = np.linspace(-90.0, 90.0, 3601)

    def _mainlobe_peaks(self, n_sources):
        E = steering_vectors(_array(), self.angles, FREQ, C)
        p = music_spectrum(self._theory_R(), E, n_sources)
        peaks = _local_maxima(p)
        peaks = peaks[np.abs(self.angles[peaks]) < 7.2]   # Rayleigh region
        return self.angles[peaks], p

    def test_too_few_sources_merges_the_pair(self):
        found, p = self._mainlobe_peaks(1)
        # One genuine signal eigenvector lands in the "noise" subspace:
        # both sources collapse into a single blob near broadside, and the
        # pseudospectrum contrast falls by an order of magnitude.
        assert len(found) == 1
        assert abs(found[0]) < 1.0

    def test_correct_order_resolves_both_sources(self):
        found, p = self._mainlobe_peaks(2)
        np.testing.assert_allclose(np.sort(found), [-3.0, 3.0], atol=0.1)

    def test_too_many_sources_is_benign(self):
        found, p = self._mainlobe_peaks(4)
        np.testing.assert_allclose(np.sort(found), [-3.0, 3.0], atol=0.1)

    def test_wrong_order_contrast_collapses(self):
        _, p1 = self._mainlobe_peaks(1)
        _, p2 = self._mainlobe_peaks(2)
        contrast = lambda p: 10.0 * np.log10(p.max() / np.median(p))
        assert contrast(p2) - contrast(p1) > 15.0


class TestArrayGuards:
    @pytest.mark.parametrize("bad", BAD_SCALARS)
    def test_steering_vectors_bad_sound_speed_raises(self, bad):
        with pytest.raises(ConfigurationError,
                           match="c must be > 0 m/s and finite"):
            steering_vectors([0.0, 0.75], [0.0], 100.0, bad)

    @pytest.mark.parametrize("bad", BAD_SCALARS)
    def test_steering_vectors_bad_frequency_raises(self, bad):
        with pytest.raises(ConfigurationError,
                           match="frequency must be > 0 Hz and finite"):
            steering_vectors([0.0, 0.75], [0.0], bad)

    def test_beamform_rejects_zero_sound_speed(self):
        with pytest.raises(ConfigurationError, match="c must be > 0 m/s"):
            beamform(np.ones((2, 3)), np.array([0.0, 0.75]), 100.0, c=0.0)

    @pytest.mark.parametrize("bad", [-0.1, np.nan])
    def test_sample_covariance_negative_or_nan_loading_raises(self, bad):
        # A negative loading was skipped by the `> 0` application branch, so
        # the caller got an unregularised R that looked regularised.
        x = np.ones((2, 4), dtype=complex)
        with pytest.raises(ConfigurationError,
                           match="diagonal_loading must be >= 0"):
            sample_covariance(x, diagonal_loading=bad)

    def test_positive_loading_scales_the_trace_by_one_plus_dl(self):
        x = (np.random.default_rng(7).standard_normal((3, 16))
             + 1j * np.random.default_rng(8).standard_normal((3, 16)))
        t0 = np.trace(sample_covariance(x)).real
        t1 = np.trace(sample_covariance(x, diagonal_loading=0.5)).real
        assert t1 == pytest.approx(1.5 * t0)


class TestBeamformValidatesOwnArguments:
    @pytest.mark.parametrize("bad_c", [0.0, np.inf])
    def test_zero_or_infinite_sound_speed_error_names_beamform(self, bad_c):
        with pytest.raises(ConfigurationError,
                           match="beamform: c must be > 0 m/s and finite"):
            beamform(np.ones((4, 3), dtype=complex), np.arange(4.0),
                     100.0, c=bad_c)

    def test_negative_frequency_error_names_beamform(self):
        with pytest.raises(ConfigurationError,
                           match="beamform: frequency must be > 0 Hz and "
                                 "finite"):
            beamform(np.ones((4, 3), dtype=complex), np.arange(4.0), -5.0)

    def test_steering_vectors_keeps_its_own_sound_speed_guard(self):
        with pytest.raises(ConfigurationError,
                           match="steering_vectors: c must be > 0 m/s and "
                                 "finite"):
            steering_vectors(np.arange(4.0), [0.0], 100.0, c=0.0)

    def test_valid_arguments_beamform(self):
        snr, angles, peak = beamform(np.ones((4, 3), dtype=complex),
                                     np.arange(4.0), 100.0)
        assert snr.shape == (angles.size, 3) and np.isfinite(peak)
